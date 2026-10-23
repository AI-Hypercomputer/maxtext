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

"""Tests for m3 scan and Qwen3's legacy-compatible scanned checkpoint layout."""

from types import SimpleNamespace

from absl.testing import absltest, parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.infra import scan
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model


def tiny_config(scan_layers=False, param_scan_axis=1):
  """Only the configuration consumed by the dense Qwen3 model."""
  return SimpleNamespace(
      scan_layers=scan_layers,
      param_scan_axis=param_scan_axis,
      num_decoder_layers=3,
      emb_dim=16,
      mlp_dim=32,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=4,
      vocab_size=32,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      normalization_layer_epsilon=1e-6,
      use_qk_norm=True,
      rope_max_timescale=1000000,
      float32_logits=True,
      logits_dot_in_fp32=True,
      normalize_embedding_logits=False,
      final_logits_soft_cap=0,
      cast_logits_to_fp32=True,
  )


def flat_params(model):
  return {p: v[...] for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}


class ScanTest(parameterized.TestCase):

  def test_qwen3_vl_rejects_scan(self):
    with self.assertRaisesRegex(ValueError, "Qwen3-VL.*scan_layers=false"):
      create_qwen3_vl_model(tiny_config(scan_layers=True), rngs=nnx.Rngs(0))

  @parameterized.parameters(0, 1, 2)
  def test_qwen3_scan_forward_gradients_and_checkpoint_layout(self, axis):
    plain = create_qwen3_model(tiny_config(), rngs=nnx.Rngs(0))
    scanned = create_qwen3_model(tiny_config(True, axis), rngs=nnx.Rngs(0))
    p_plain, p_scan = flat_params(plain), flat_params(scanned)
    # Compiled initializers may differ from eager ones by float32 rounding.
    for path, value in p_plain.items():
      if path[0] == "decoder" and path[1].startswith("layers_"):
        i = int(path[1].split("_")[1])
        target = ("decoder", "layers") + path[2:]
        np.testing.assert_allclose(jnp.take(p_scan[target], i, axis=min(axis, value.ndim)), value, rtol=2e-6, atol=3e-7)
      else:
        np.testing.assert_array_equal(p_scan[path], value)

    tokens = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    positions = jnp.array([[0, 1, 0, 1], [0, 1, 2, 3]])
    segments = jnp.array([[1, 1, 2, 2], [1, 1, 1, 1]])

    def logits(m):
      return m(tokens, positions, decoder_segment_ids=segments)

    np.testing.assert_allclose(logits(scanned), logits(plain), rtol=1e-5, atol=1e-5)
    merged = nnx.merge(*nnx.split(scanned))
    np.testing.assert_allclose(nnx.jit(logits)(merged), logits(plain), rtol=1e-5, atol=1e-5)
    self.assertIn("scan[", str(jax.make_jaxpr(lambda: logits(scanned))()))

    def loss(m):
      return -jnp.mean(jax.nn.log_softmax(logits(m))[..., 0])

    loss_plain, grad_plain = nnx.value_and_grad(loss)(plain)
    loss_scan, grad_scan = nnx.value_and_grad(loss)(scanned)
    np.testing.assert_allclose(loss_scan, loss_plain, rtol=1e-5)
    g_scan = flat_params(grad_scan)
    for path, value in flat_params(grad_plain).items():
      if path[0] == "decoder" and path[1].startswith("layers_"):
        target = ("decoder", "layers") + path[2:]
        actual = jnp.take(g_scan[target], int(path[1].split("_")[1]), axis=min(axis, value.ndim))
      else:
        actual = g_scan[path]
      np.testing.assert_allclose(actual, value, rtol=2e-4, atol=2e-6, err_msg=str(path))


class CounterLayer(nnx.Module):
  """A stateful layer with scalar and rank-one parameters."""

  def __init__(self, offset):
    self.scale = nnx.Param(jnp.array(float(offset)))
    self.bias = nnx.Param(jnp.ones(2), out_sharding=jax.sharding.PartitionSpec())
    self.calls = nnx.Variable(jnp.array(0), out_sharding=jax.sharding.PartitionSpec())

  def __call__(self, carry, *, increment=1):
    self.calls[...] += increment
    return carry * self.scale[...] + self.bias[...]


class ScanInfrastructureTest(absltest.TestCase):

  def test_direct_initialization_preserves_rng_and_state(self):
    def make_layer(rngs):
      layer = CounterLayer(1)
      layer.scale = nnx.Param(jax.random.normal(rngs.params(), ()))
      return layer

    plain_rngs, direct_rngs = nnx.Rngs(0), nnx.Rngs(0)
    plain = [make_layer(plain_rngs) for _ in range(3)]
    direct = scan.init_scan(make_layer, num_layers=3, rngs=direct_rngs, param_scan_axis=1)
    np.testing.assert_allclose(direct.scale[...], jnp.stack([layer.scale[...] for layer in plain]), rtol=2e-6, atol=3e-7)
    np.testing.assert_array_equal(jax.random.key_data(direct_rngs.params()), jax.random.key_data(plain_rngs.params()))
    expected = jnp.ones(2)
    for layer in plain:
      expected = layer(expected)
    np.testing.assert_allclose(direct(jnp.ones(2)), expected, rtol=1e-5)
    np.testing.assert_array_equal(direct.calls[...], [1, 1, 1])

  def test_state_and_metadata_survive_repeated_calls(self):
    scanned = scan.init_scan(lambda _: CounterLayer(1), num_layers=2, rngs=nnx.Rngs(0), param_scan_axis=1)
    self.assertEqual(scanned.bias.get_metadata()["out_sharding"], jax.sharding.PartitionSpec(None, None))
    for _ in range(2):
      np.testing.assert_allclose(scanned(jnp.zeros(2), increment=2), [2, 2])
    np.testing.assert_array_equal(scanned.calls[...], [4, 4])
    self.assertEqual(scanned.calls.get_metadata()["out_sharding"], jax.sharding.PartitionSpec(None))


class AddOne(nnx.Module):
  """A stateless layer still needs to execute once per layer in a scan."""

  def __call__(self, carry):
    return carry + 1


class ScanBoundaryTest(absltest.TestCase):

  def test_named_sharding_is_preserved(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    self.enter_context(jax.set_mesh(mesh))
    spec = jax.sharding.PartitionSpec("data")

    def make_layer(_):
      layer = CounterLayer(1)
      layer.bias = nnx.Param(jax.device_put(jnp.ones(2), jax.sharding.NamedSharding(mesh, spec)), out_sharding=spec)
      return layer

    scanned = scan.init_scan(make_layer, num_layers=2, rngs=nnx.Rngs(0), param_scan_axis=1)
    expected = jax.sharding.PartitionSpec("data", None)
    self.assertEqual(scanned.bias[...].sharding.spec, expected)
    np.testing.assert_allclose(nnx.jit(lambda m: m(jnp.zeros(2)))(scanned), [2, 2])
    self.assertEqual(scanned.bias.get_metadata()["out_sharding"], expected)


class PrivateScale(nnx.Module):
  """Private names are valid model attributes and parameters."""

  def __init__(self):
    self._scale = nnx.Param(jnp.array(2.0))
    self._offset = 1.0

  def __call__(self, carry):
    return carry * self._scale[...] + self._offset


class ScanReviewRegressionTest(absltest.TestCase):

  def test_private_layer_attributes_are_preserved(self):
    scanned = scan.init_scan(lambda _: PrivateScale(), num_layers=2, rngs=nnx.Rngs(0), param_scan_axis=1)
    self.assertEqual(float(scanned(jnp.array(1.0))), 7.0)

  def test_stateless_layers_execute_each_iteration(self):
    scanned = scan.init_scan(lambda _: AddOne(), num_layers=3, rngs=nnx.Rngs(0), param_scan_axis=1)
    self.assertEqual(float(scanned(jnp.array(1.0))), 4.0)


if __name__ == "__main__":
  absltest.main()
