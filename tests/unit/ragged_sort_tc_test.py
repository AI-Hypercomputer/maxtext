# Copyright 2023–2026 Google LLC
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

"""Correctness tests for the TensorCore ragged sort / unsort (ring_ragged_sort_tc, ring_ragged_unsort_tc).

Each expert-parallel shard's TC sort + unsort is compared, in value and in gradients w.r.t. the tokens and the
routing weights, against a pure-JAX reference of the same truncated-buffer semantics:

  buf[i] = x[order[start + i] // topk]                for i < count, else 0
  y[t]   = sum_{i < count, order[start+i]//topk == t} w[order[start + i]] * f(buf)[i]

where `order` is the stable argsort of the flat expert ids, `[start, end)` is the shard's slot window and
`count = min(end - start, buffer_size)`.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

from maxtext.kernels.ragged import ragged_sort_tc


def _reference_sort_unsort(x, topk_indices, w, f, *, num_experts, topk, shard_idx, ep_size, buffer_size):
  """Pure-JAX reference of one shard's truncated ragged sort -> f -> weighted unsort."""
  n_tok = x.shape[0]
  flat = topk_indices.reshape(-1).astype(jnp.int32)
  n = flat.shape[0]
  order = jnp.argsort(flat, stable=True).astype(jnp.int32)
  group_sizes = jnp.bincount(flat, length=num_experts)
  offsets = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])
  local = num_experts // ep_size
  start = offsets[shard_idx * local]
  end = offsets[(shard_idx + 1) * local]
  cap = min(buffer_size, n)
  count = jnp.minimum(end - start, cap)
  rows = jax.lax.dynamic_slice_in_dim(jnp.pad(order, (0, cap)), start, cap)
  valid = jnp.arange(cap) < count
  tok = jnp.where(valid, rows // topk, 0)
  buf = jnp.where(valid[:, None], x[tok], jnp.zeros((), x.dtype))
  h = f(buf)
  w_rows = jnp.where(valid, w.reshape(-1)[jnp.where(valid, rows, 0)], 0.0)
  scaled = (h.astype(jnp.float32) * w_rows[:, None]).astype(h.dtype)
  y = jnp.zeros((n_tok, x.shape[1]), jnp.float32).at[tok].add(jnp.where(valid[:, None], scaled, 0).astype(jnp.float32))
  return buf, group_sizes, y.astype(x.dtype)


class RaggedSortTcTest(parameterized.TestCase):
  """TC ragged sort vs a pure-JAX reference, per expert-parallel shard."""

  def setUp(self):
    super().setUp()
    if not any(d.platform == "tpu" for d in jax.devices()):
      self.skipTest("The TC ragged sort kernels require TPU.")

  def _run(
      self,
      *,
      num_tokens=512,
      hidden=512,
      num_experts=16,
      topk=4,
      buffer_factor=1.5,
      gather_block_size=128,
      reduce_block_size=128,
      flatten_block_size=0,
      mask_padding=True,
      dtype=jnp.bfloat16,
      skewed=False,
  ):
    """Runs the TC sort -> f -> unsort per shard and checks it against the reference; returns (counts, cap)."""
    devices = jax.devices()
    ep_size = max(d for d in (8, 4, 2, 1) if len(devices) % d == 0 and num_experts % d == 0)
    mesh = Mesh(np.asarray(devices[:ep_size]), ("expert",))
    n = num_tokens * topk
    buffer_size = int(buffer_factor * n / ep_size)

    k_x, k_e, k_w, k_m, k_ct = jax.random.split(jax.random.PRNGKey(0), 5)
    x = jax.random.normal(k_x, (num_tokens, hidden), jnp.float32).astype(dtype)
    if skewed:
      # Concentrate traffic on the first experts so some shards overflow the buffer and others get few rows.
      logits = jax.random.normal(k_e, (num_tokens, num_experts)) - 0.5 * jnp.arange(num_experts)
    else:
      logits = jax.random.normal(k_e, (num_tokens, num_experts))
    _, topk_indices = jax.lax.top_k(logits, topk)
    topk_indices = topk_indices.astype(jnp.int32)
    w = jax.random.uniform(k_w, (num_tokens, topk), jnp.float32)
    m = jax.random.normal(k_m, (hidden,), jnp.float32)
    ct = jax.random.normal(k_ct, (ep_size, num_tokens, hidden), jnp.float32)

    def f(buf):
      # Row-wise nonlinearity standing in for the expert MLP.
      return jnp.tanh(buf.astype(jnp.float32) * m).astype(buf.dtype)

    def tc_shard(x, topk_indices, w):
      buf, group_sizes, _, routing = ragged_sort_tc.ring_ragged_sort_tc(
          x,
          topk_indices,
          num_experts,
          topk,
          "expert",
          ep_size,
          buffer_size,
          gather_block_size=gather_block_size,
          reduce_block_size=reduce_block_size,
          mask_padding=mask_padding,
          flatten_block_size=flatten_block_size,
      )
      y = ragged_sort_tc.ring_ragged_unsort_tc(
          f(buf),
          routing,
          topk,
          w.reshape(-1),
          gather_block_size=gather_block_size,
          mask_padding=mask_padding,
          flatten_block_size=flatten_block_size,
      )
      count = routing.count
      return buf[None], group_sizes[None], y[None], count[None]

    def ref_shard(x, topk_indices, w):
      shard_idx = jax.lax.axis_index("expert")
      buf, group_sizes, y = _reference_sort_unsort(
          x,
          topk_indices,
          w,
          f,
          num_experts=num_experts,
          topk=topk,
          shard_idx=shard_idx,
          ep_size=ep_size,
          buffer_size=buffer_size,
      )
      return buf[None], group_sizes[None], y[None]

    rep = P()
    shard = P("expert")
    tc_fn = jax.shard_map(
        tc_shard, mesh=mesh, in_specs=(rep, rep, rep), out_specs=(shard, shard, shard, shard), check_vma=False
    )
    ref_fn = jax.shard_map(
        ref_shard, mesh=mesh, in_specs=(rep, rep, rep), out_specs=(shard, shard, shard), check_vma=False
    )

    def tc_loss(x, w):
      _, _, y, _ = tc_fn(x, topk_indices, w)
      return jnp.sum(y.astype(jnp.float32) * ct)

    def ref_loss(x, w):
      _, _, y = ref_fn(x, topk_indices, w)
      return jnp.sum(y.astype(jnp.float32) * ct)

    with jax.set_mesh(mesh):
      buf_tc, gs_tc, y_tc, count_tc = jax.jit(tc_fn)(x, topk_indices, w)
      buf_ref, gs_ref, y_ref = jax.jit(ref_fn)(x, topk_indices, w)
      loss_tc, (dx_tc, dw_tc) = jax.jit(jax.value_and_grad(tc_loss, argnums=(0, 1)))(x, w)
      loss_ref, (dx_ref, dw_ref) = jax.jit(jax.value_and_grad(ref_loss, argnums=(0, 1)))(x, w)

    np.testing.assert_array_equal(np.asarray(gs_tc), np.asarray(gs_ref))
    # The gather is an exact copy; rows past count are zero with mask_padding, otherwise unspecified.
    for s in range(ep_size):
      c = int(count_tc[s])
      rows = slice(None) if mask_padding else slice(0, c)
      np.testing.assert_array_equal(
          np.asarray(buf_tc[s][rows].astype(jnp.float32)), np.asarray(buf_ref[s][rows].astype(jnp.float32))
      )
    tol = {"rtol": 2e-2, "atol": 2e-2} if dtype == jnp.bfloat16 else {"rtol": 1e-5, "atol": 1e-5}
    np.testing.assert_allclose(np.asarray(y_tc, np.float32), np.asarray(y_ref, np.float32), **tol)
    np.testing.assert_allclose(float(loss_tc), float(loss_ref), rtol=tol["rtol"])
    for g_tc, g_ref in ((dx_tc, dx_ref), (dw_tc, dw_ref)):
      g_ref = np.asarray(g_ref, np.float32)
      atol = tol["atol"] * max(1.0, float(np.max(np.abs(g_ref))))
      np.testing.assert_allclose(np.asarray(g_tc, np.float32), g_ref, rtol=tol["rtol"], atol=atol)
    return np.asarray(count_tc), buffer_size

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("bf16", jnp.bfloat16),
      ("f32", jnp.float32),
  )
  def test_sort_unsort(self, dtype):
    self._run(dtype=dtype)

  @pytest.mark.tpu_only
  def test_sort_unsort_truncated_buffer(self):
    """Skewed routing with a small buffer: some shards drop slots past the buffer."""
    counts, buffer_size = self._run(buffer_factor=0.75, skewed=True)
    self.assertTrue(np.any(counts == buffer_size), msg=f"expected an overflowing shard, got counts {counts}")

  @pytest.mark.tpu_only
  def test_sort_unsort_no_mask_padding(self):
    self._run(mask_padding=False, skewed=True)

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("flatten_128", 128),
      ("flatten_256", 256),
  )
  def test_sort_unsort_flatten(self, flatten_block_size):
    self._run(flatten_block_size=flatten_block_size, skewed=True)

  @pytest.mark.tpu_only
  @parameterized.named_parameters(
      ("small_blocks", 64, 64),
      ("large_blocks", 1024, 896),
  )
  def test_sort_unsort_block_sizes(self, gather_block_size, reduce_block_size):
    self._run(gather_block_size=gather_block_size, reduce_block_size=reduce_block_size, skewed=True)

  @pytest.mark.tpu_only
  def test_sort_unsort_dsv3_hidden(self):
    """DeepSeek-V3 hidden size (7168 = 56 x 128) with top-8 routing."""
    self._run(num_tokens=256, hidden=7168, num_experts=32, topk=8, buffer_factor=1.25, skewed=True)


if __name__ == "__main__":
  absltest.main()
