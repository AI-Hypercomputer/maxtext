# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for kernels/megablox/split_gmm.py against a pure-JAX model of gmm_v2 / tgmm_v2.

The model reproduces the one gmm_v2 property the chaining relies on: inside every
sublane tile the kernel writes, rows that belong to a foreign group are zeroed
(not left at their `partial_sum` value), so the second chained call clobbers the
tail of the last `rhs_lo` expert when the expert boundary is not tile aligned.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.megablox import split_gmm

_SUBLANES = 16  # bf16 sublane tiling emulated by the model


def _row_groups(group_sizes, num_rows):
  ends = jnp.cumsum(group_sizes)
  return jnp.searchsorted(ends, jnp.arange(num_rows), side="right")


def _model_gmm(lhs, rhs, group_sizes, group_offset, partial_sum, zero_initialize, tile, out_dtype):
  """lhs [M, K] @ rhs [G, K, N] for the rows of groups [offset, offset + G), with gmm_v2 tile masking."""
  del tile
  m = lhs.shape[0]
  g = rhs.shape[0]
  grp = _row_groups(group_sizes, m)
  local = grp - group_offset
  visited = (local >= 0) & (local < g)
  # Every sublane tile containing a visited row is written in full; foreign rows in it become 0.
  tile_id = jnp.arange(m) // _SUBLANES
  written_tile = jnp.zeros((m // _SUBLANES + 1,), bool).at[tile_id].max(visited)
  written = written_tile[tile_id]
  w = jnp.take(rhs, jnp.clip(local, 0, g - 1), axis=0)  # [M, K, N]
  acc = jnp.einsum("mk,mkn->mn", lhs.astype(jnp.float32), w.astype(jnp.float32), precision="highest")
  base = jnp.zeros_like(acc) if partial_sum is None else partial_sum.astype(jnp.float32)
  acc = jnp.where(visited[:, None], base + acc, 0.0)
  if zero_initialize or partial_sum is None:
    out = acc
  else:
    out = jnp.where(written[:, None], acc, base)
  return out.astype(out_dtype)


def _model_tgmm(lhs, rhs, group_sizes, group_offset, num_actual_groups, tile, out_dtype):
  """drhs[g] = lhs[rows of g].T @ rhs[rows of g] for g in [offset, offset + num_actual_groups)."""
  del tile
  grp = _row_groups(group_sizes, lhs.shape[0])
  onehot = (grp[:, None] == (group_offset + jnp.arange(num_actual_groups))[None, :]).astype(jnp.float32)
  out = jnp.einsum("mg,mk,mn->gkn", onehot, lhs.astype(jnp.float32), rhs.astype(jnp.float32), precision="highest")
  return out.astype(out_dtype)


split_gmm.IMPLS["model"] = (_model_gmm, _model_tgmm)


def _dense_reference(lhs, rhs_lo, rhs_hi, group_sizes, base):
  rhs = jnp.concatenate([rhs_lo, rhs_hi], axis=0)
  grp = _row_groups(group_sizes, lhs.shape[0])
  local = grp - base
  valid = (local >= 0) & (local < rhs.shape[0])
  w = jnp.take(rhs, jnp.clip(local, 0, rhs.shape[0] - 1), axis=0)
  out = jnp.einsum("mk,mkn->mn", lhs.astype(jnp.float32), w.astype(jnp.float32), precision="highest")
  return jnp.where(valid[:, None], out, 0.0)


class SplitGmmTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("aligned_boundary", (16, 32, 16, 16, 48, 32), 0),
      ("unaligned_boundary", (10, 23, 9, 30, 20, 36), 0),
      ("group_offset_and_empty_experts", (7, 0, 25, 0, 9, 14, 33, 8), 2),
      ("empty_last_lo_expert", (20, 0, 12, 24), 0),
  )
  def test_matches_dense_reference(self, group_sizes, base):
    group_sizes = jnp.asarray(group_sizes, jnp.int32)
    num_groups = int(group_sizes.shape[0])
    g_lo = (num_groups - base) // 2
    g_hi = num_groups - base - g_lo
    m, k, n = int(group_sizes.sum()), 32, 48
    rng = np.random.default_rng(0)
    lhs = jnp.asarray(rng.standard_normal((m, k)), jnp.float32)
    rhs_lo = jnp.asarray(rng.standard_normal((g_lo, k, n)), jnp.float32)
    rhs_hi = jnp.asarray(rng.standard_normal((g_hi, k, n)), jnp.float32)
    grad = jnp.asarray(rng.standard_normal((m, n)), jnp.float32)

    def split(lhs, rhs_lo, rhs_hi):
      return split_gmm.gmm_split_experts(
          lhs,
          rhs_lo,
          rhs_hi,
          group_sizes,
          tiling=None,
          preferred_element_type=jnp.float32,
          group_offset=base,
          impl="model",
      )

    def dense(lhs, rhs_lo, rhs_hi):
      return _dense_reference(lhs, rhs_lo, rhs_hi, group_sizes, base)

    out, vjp = jax.vjp(split, lhs, rhs_lo, rhs_hi)
    out_ref, vjp_ref = jax.vjp(dense, lhs, rhs_lo, rhs_hi)
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), rtol=1e-5, atol=1e-5)
    for got, want in zip(vjp(grad), vjp_ref(grad)):
      np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)

  def test_model_clobbers_unaligned_boundary_without_repair(self):
    """Sanity check of the model: a naive chain loses the rows the repair window restores."""
    group_sizes = jnp.asarray((10, 23, 9, 30), jnp.int32)
    m, k, n = int(group_sizes.sum()), 8, 8
    rng = np.random.default_rng(1)
    lhs = jnp.asarray(rng.standard_normal((m, k)), jnp.float32)
    rhs = jnp.asarray(rng.standard_normal((4, k, n)), jnp.float32)
    out1 = _model_gmm(lhs, rhs[:2], group_sizes, 0, None, True, None, jnp.float32)
    out2 = _model_gmm(lhs, rhs[2:], group_sizes, 2, out1, False, None, jnp.float32)
    ref = _dense_reference(lhs, rhs[:2], rhs[2:], group_sizes, 0)
    boundary = 33
    tile_start = boundary - boundary % _SUBLANES
    np.testing.assert_array_equal(np.asarray(out2[tile_start:boundary]), 0.0)
    np.testing.assert_allclose(np.asarray(out2[:tile_start]), np.asarray(ref[:tile_start]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out2[boundary:]), np.asarray(ref[boundary:]), rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("unscanned", ()),
      ("scanned_1d", (5,)),
      ("scanned_2d", (5, 5)),
  )
  def test_convert_checkpoint_tree_roundtrip_and_expert_halves(self, scan_dims):
    e, emb, mlp2 = 8, 16, 12
    normal_wi = jnp.arange(int(np.prod((e, *scan_dims, emb, mlp2))), dtype=jnp.float32).reshape(
        e, *scan_dims, emb, mlp2
    )
    wo = jnp.ones((e, *scan_dims, mlp2 // 2, emb), dtype=jnp.float32)
    tree = {"params": {"decoder": {"moe_block": {"wi": {"value": normal_wi}, "wo": {"value": wo}}}}}

    split_tree = split_gmm.convert_checkpoint_tree(tree, to_split=True)
    split_wi = split_tree["params"]["decoder"]["moe_block"]["wi"]["value"]
    self.assertEqual(split_wi.shape, (2, *scan_dims, e, emb // 2, mlp2))
    np.testing.assert_array_equal(split_tree["params"]["decoder"]["moe_block"]["wo"]["value"], wo)

    # Verify that each scanned layer's two halves reshape directly into W[:E/2] and W[E/2:].
    idx = (slice(None), *((0,) * len(scan_dims)), slice(None), slice(None))
    w_layer = normal_wi[idx]  # (E, emb, mlp2)
    s_layer = split_wi[(slice(None), *((0,) * len(scan_dims)), slice(None), slice(None), slice(None))]
    np.testing.assert_array_equal(s_layer[0].reshape(e // 2, emb, mlp2), w_layer[: e // 2])
    np.testing.assert_array_equal(s_layer[1].reshape(e // 2, emb, mlp2), w_layer[e // 2 :])

    roundtrip_tree = split_gmm.convert_checkpoint_tree(split_tree, to_split=False)
    np.testing.assert_array_equal(roundtrip_tree["params"]["decoder"]["moe_block"]["wi"]["value"], normal_wi)


if __name__ == "__main__":
  absltest.main()

