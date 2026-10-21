# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `maxtext.m3.core.layers` and `maxtext.m3.infra.remat`.

Remat's one job is to be invisible: same outputs, same gradients, same
parameter tree. The tests here pin all three, plus the policy-name mapping.
"""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core.layers import Layers, Remat
from maxtext.m3.infra import remat


class _Block(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.lin = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x, scale=1.0):
    return x + scale * checkpoint_name(self.lin(x), "mlpwo")


class _Model(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.layers = Layers([_Block(rngs) for _ in range(3)])


def _loss(model, x):
  return jnp.sum(model.layers(x, scale=0.5) ** 2)


class LayersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = _Model(nnx.Rngs(0))
    self.x = jax.random.normal(jax.random.key(1), (2, 4))

  def test_call_applies_layers_in_order_with_kwargs(self):
    expected = self.x
    for block in self.model.layers:
      expected = block(expected, scale=0.5)
    np.testing.assert_array_equal(self.model.layers(self.x, scale=0.5), expected)

  def test_param_paths_index_each_layer(self):
    paths = {path[:3] for path, _ in nnx.to_flat_state(nnx.state(self.model, nnx.Param))}
    self.assertEqual(paths, {("layers", i, "lin") for i in range(3)})

  def test_survives_split_merge(self):
    merged = nnx.merge(*nnx.split(self.model))
    self.assertIsInstance(merged.layers, Layers)
    np.testing.assert_array_equal(merged.layers(self.x), self.model.layers(self.x))

  def test_gradients_reach_every_layer(self):
    grads = nnx.grad(_loss)(self.model, self.x)
    for path, grad in nnx.to_flat_state(nnx.state(grads, nnx.Param)):
      self.assertGreater(float(jnp.abs(grad[...]).sum()), 0.0, path)

  def test_tuple_carry(self):
    """Carries are arbitrary pytrees: MoE decoders thread `(hidden, aux_loss)`."""

    class _Counting(nnx.Module):

      def __call__(self, carry):
        x, n = carry
        return x * 2, n + 1

    layers = Layers([_Counting(), _Counting()])
    x, n = layers((self.x, 0))
    np.testing.assert_array_equal(x, self.x * 4)
    self.assertEqual(n, 2)


class RematTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.x = jax.random.normal(jax.random.key(1), (2, 4))

  @parameterized.named_parameters(
      ("minimal", "minimal"),
      ("save_qkv_proj", "save_qkv_proj"),
      ("save_dot_except_mlp", "save_dot_except_mlp"),
      ("minimal_offloaded", "minimal_offloaded"),
  )
  def test_named_policies_resolve_to_callables(self, name):
    resolved = remat.resolve_policy(SimpleNamespace(remat_policy=name))
    self.assertIsInstance(resolved, Remat)
    self.assertTrue(callable(resolved.policy))

  def test_full_saves_nothing(self):
    """`None` is JAX's "save nothing" policy; it must stay distinct from remat being off."""
    self.assertEqual(remat.resolve_policy(SimpleNamespace(remat_policy="full")), Remat(policy=None))

  def test_none_turns_remat_off(self):
    self.assertIsNone(remat.resolve_policy(SimpleNamespace(remat_policy="none")))

  def test_unknown_policy_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown remat_policy"):
      remat.resolve_policy(SimpleNamespace(remat_policy="save_everything_twice"))

  def test_custom_policy_reads_tensor_lists(self):
    config = SimpleNamespace(remat_policy="custom", tensors_on_device=["mlpwo"], tensors_to_offload=[])
    self.assertTrue(callable(remat.resolve_policy(config).policy))

  @parameterized.named_parameters(("full", "full"), ("minimal", "minimal"))
  def test_remat_is_invisible(self, policy):
    """Same parameter tree, same outputs, same gradients, with and without remat."""
    plain = _Model(nnx.Rngs(0))
    rematted = remat.maybe_apply_remat(_Model(nnx.Rngs(0)), SimpleNamespace(remat_policy=policy))
    self.assertIsNotNone(rematted.layers.remat)

    plain_params = nnx.to_flat_state(nnx.state(plain, nnx.Param))
    remat_params = nnx.to_flat_state(nnx.state(rematted, nnx.Param))
    self.assertEqual([p for p, _ in plain_params], [p for p, _ in remat_params])

    np.testing.assert_allclose(rematted.layers(self.x, scale=0.5), plain.layers(self.x, scale=0.5), rtol=1e-6)
    g_plain = nnx.to_flat_state(nnx.state(nnx.grad(_loss)(plain, self.x), nnx.Param))
    g_remat = nnx.to_flat_state(nnx.state(nnx.grad(_loss)(rematted, self.x), nnx.Param))
    for (_, a), (_, b) in zip(g_plain, g_remat):
      np.testing.assert_allclose(a[...], b[...], rtol=1e-5)

  def test_remat_setting_survives_split_merge_and_jit(self):
    model = remat.maybe_apply_remat(_Model(nnx.Rngs(0)), SimpleNamespace(remat_policy="minimal"))
    merged = nnx.merge(*nnx.split(model))
    self.assertEqual(merged.layers.remat, model.layers.remat)
    out = nnx.jit(lambda m, x: m.layers(x, scale=0.5))(model, self.x)
    np.testing.assert_allclose(out, model.layers(self.x, scale=0.5), rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
