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

"""Tests for `maxtext.m3.core.moe` and `maxtext.m3.core.attention`.

The MoE runtime's plain-`jnp` executors are checked against each other and
against a per-token reference written the slow, obvious way; the Pallas
executors are covered on TPU in `kernels_test.py`. The attention function is
checked against `jax.nn.dot_product_attention`.
"""

import functools
from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core import attention, moe
from maxtext.m3.core.kernels import XLA_ATTENTION
from maxtext.m3.core.sharding import DefaultSharding


def _reference_moe(x, weights, indices, wi_0, wi_1, wo):
  """Loop over tokens and their selected experts. Float32."""
  batch, length, _ = x.shape
  out = np.zeros((batch, length, wo.shape[-1]), np.float32)
  for b in range(batch):
    for s in range(length):
      for k in range(indices.shape[-1]):
        e = int(indices[b, s, k])
        hidden = jax.nn.silu(x[b, s] @ wi_0[e]) * (x[b, s] @ wi_1[e])
        out[b, s] += float(weights[b, s, k]) * np.asarray(hidden @ wo[e])
  return out


class MoeRuntimeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    keys = jax.random.split(jax.random.key(0), 6)
    self.num_experts, self.top_k, embed, mlp = 6, 2, 8, 16
    self.x = jax.random.normal(keys[0], (2, 5, embed), jnp.float32)
    logits = jax.random.normal(keys[1], (2, 5, self.num_experts))
    self.weights, self.indices = jax.lax.top_k(jax.nn.softmax(logits), self.top_k)
    self.kernels = (
        jax.random.normal(keys[2], (self.num_experts, embed, mlp)) * 0.1,
        jax.random.normal(keys[3], (self.num_experts, embed, mlp)) * 0.1,
        jax.random.normal(keys[4], (self.num_experts, mlp, embed)) * 0.1,
    )

  @parameterized.named_parameters(("dense", moe.DENSE), ("ragged_dot", moe.RAGGED_DOT))
  def test_executor_matches_reference(self, executor):
    out = moe.routed_experts(self.x, self.weights, self.indices, *self.kernels, executor=executor)
    expected = _reference_moe(self.x, self.weights, self.indices, *self.kernels)
    self.assertEqual(out.dtype, self.x.dtype)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

  def test_executors_agree_in_bfloat16(self):
    x = self.x.astype(jnp.bfloat16)
    kernels = [k.astype(jnp.bfloat16) for k in self.kernels]
    weights = self.weights.astype(jnp.bfloat16)
    dense = moe.routed_experts(x, weights, self.indices, *kernels, executor=moe.DENSE)
    ragged = moe.routed_experts(x, weights, self.indices, *kernels, executor=moe.RAGGED_DOT)
    self.assertEqual(dense.dtype, jnp.bfloat16)
    np.testing.assert_allclose(dense.astype(jnp.float32), ragged.astype(jnp.float32), rtol=5e-2, atol=5e-2)

  @parameterized.named_parameters(("dense", moe.DENSE), ("ragged_dot", moe.RAGGED_DOT))
  def test_gradients_flow_to_selected_experts_only(self, executor):
    def loss(kernels):
      return jnp.sum(moe.routed_experts(self.x, self.weights, self.indices, *kernels, executor=executor) ** 2)

    grads = jax.grad(loss)(self.kernels)
    selected = set(np.unique(np.asarray(self.indices)))
    for g in grads:
      touched = {e for e in range(self.num_experts) if float(jnp.abs(g[e]).sum()) > 0}
      self.assertEqual(touched, selected)

  def test_unknown_executor_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown MoE executor"):
      moe.routed_experts(self.x, self.weights, self.indices, *self.kernels, executor=moe.Executor("gmm"))

  def test_combine_weights_scatters_top_k(self):
    table = moe.combine_weights(self.weights, self.indices, self.num_experts)
    self.assertEqual(table.shape, (2, 5, self.num_experts))
    np.testing.assert_allclose(jnp.take_along_axis(table, self.indices, axis=-1), self.weights)
    np.testing.assert_allclose(jnp.sum(table, axis=-1), jnp.sum(self.weights, axis=-1), rtol=1e-6)

  def test_load_balance_loss_is_minimal_when_balanced(self):
    """A uniform router sending equal traffic to every expert scores exactly `weight`."""
    num_experts = 4
    probs = jnp.full((1, 8, num_experts), 1.0 / num_experts)
    indices = jnp.reshape(jnp.arange(16) % num_experts, (1, 8, 2))
    loss = moe.load_balance_loss(indices, probs, weight=0.01)
    np.testing.assert_allclose(loss, 0.01, rtol=1e-6)
    # Collapsing onto one expert is `num_experts` times worse.
    collapsed = moe.load_balance_loss(
        jnp.zeros_like(indices), jax.nn.one_hot(jnp.zeros((1, 8), int), num_experts), weight=0.01
    )
    np.testing.assert_allclose(collapsed, 0.01 * num_experts, rtol=1e-6)

  def test_executor_from_config(self):
    sharding = DefaultSharding()

    def config(**overrides):
      fields = {"sparse_matmul": True, "megablox": True, "use_tokamax_gmm": False, "capacity_factor": -1.0}
      return SimpleNamespace(**{**fields, **overrides})

    self.assertEqual(moe.executor_from_config(config(sparse_matmul=False), sharding), moe.DENSE)
    self.assertEqual(moe.executor_from_config(config(megablox=False), sharding), moe.RAGGED_DOT)
    # The Pallas kinds carry the mesh axes of the tokens, the expert axis and the expert-MLP axis.
    megablox = moe.executor_from_config(config(), sharding)
    self.assertEqual(megablox, moe.Executor("megablox", batch=("data", "fsdp", "expert"), experts="expert", mlp="tensor"))
    self.assertEqual(moe.executor_from_config(config(use_tokamax_gmm=True), sharding).kind, "tokamax")
    with self.assertRaisesRegex(ValueError, "dropless"):
      moe.executor_from_config(config(capacity_factor=1.25), sharding)


class AttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(("mha", 4, 4), ("gqa", 4, 2))
  def test_matches_jax_reference(self, heads, kv_heads):
    keys = jax.random.split(jax.random.key(0), 3)
    b, t, d = 2, 6, 8
    q = jax.random.normal(keys[0], (b, t, heads, d)) * d**-0.5
    k = jax.random.normal(keys[1], (b, t, kv_heads, d))
    v = jax.random.normal(keys[2], (b, t, kv_heads, d))
    positions = jnp.broadcast_to(jnp.arange(t), (b, t))
    out = attention.dot_product_attention(q, k, v, positions=positions, kernel=XLA_ATTENTION)
    expected = jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

  def test_segments_do_not_attend_across_boundaries(self):
    keys = jax.random.split(jax.random.key(1), 3)
    b, t, d = 1, 6, 4
    q, k, v = (jax.random.normal(key, (b, t, 1, d)) for key in keys)
    positions = jnp.array([[0, 1, 2, 0, 1, 2]])
    segment_ids = jnp.array([[1, 1, 1, 2, 2, 2]])
    attend = functools.partial(attention.dot_product_attention, kernel=XLA_ATTENTION)
    packed = attend(q, k, v, positions=positions, segment_ids=segment_ids)
    separate = jnp.concatenate(
        [
            attend(q[:, :3], k[:, :3], v[:, :3], positions=positions[:, :3]),
            attend(q[:, 3:], k[:, 3:], v[:, 3:], positions=positions[:, 3:]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(packed, separate, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
