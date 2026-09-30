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

"""Expert-split grouped matmul for the prefused MoE `wi` (moe_split_expert_weight_layout).

The routed-expert weight is stored as two contiguous expert blocks
`rhs_lo = W[base : base+G_lo]` and `rhs_hi = W[base+G_lo : base+G_lo+G_hi]`
(each all-gathered separately, ~half the bytes of the fused weight), so the
full `[G, K, N]` weight is never materialized by a concatenate.

    out = gmm(lhs, concat([rhs_lo, rhs_hi]))            (never built)

is computed as two gmm_v2 calls that accumulate into ONE output buffer:

    out1 = gmm_v2(lhs, rhs_lo, group_offset=base)                  # rows >= b are 0
    out  = gmm_v2(lhs, rhs_hi, group_offset=base+G_lo,
                  partial_sum=out1, zero_initialize=False)         # in-place on out1

where b is the first row of expert base+G_lo in the (expert-sorted) lhs. gmm_v2
masks rows of foreign groups inside every tile it writes, so the second call
zeroes the (< one sublane) rows of expert base+G_lo-1 that share the sublane
containing row b. Those rows are restored from a 32-row window of out1 that is
read *before* the second call (an optimization_barrier orders the read before
the in-place kernel so XLA does not copy the whole buffer).

The backward pass uses the same chaining for dlhs, and two tgmm_v2 calls with
group offsets for drhs (each produces its half directly -> no split copy).
"""

import functools

import jax
import jax.numpy as jnp

# Rows restored around the expert-block boundary. Must be >= the largest lhs
# sublane tiling (bf16: 16, int8/fp8: 32).
_REPAIR_ROWS = 32


def _tpu_gmm(lhs, rhs, group_sizes, group_offset, partial_sum, zero_initialize, tile, out_dtype, transpose_rhs=False):
  from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2  # pylint: disable=import-outside-toplevel

  tile_info = (
      gmm_v2.calculate_tiling if tile is None else gmm_v2.TileSizes(tile_m=tile[0], tile_k=tile[1], tile_n=tile[2])
  )
  return gmm_v2.gmm_v2(
      lhs=lhs,
      rhs=rhs,
      group_sizes=group_sizes,
      rhs_scale=None,
      tile_info=tile_info,
      preferred_element_type=out_dtype,
      partial_sum=partial_sum,
      group_offset=jnp.asarray(group_offset, jnp.int32),
      zero_initialize=zero_initialize,
      **({"transpose_rhs": True} if transpose_rhs else {}),
  )


def _tpu_tgmm(lhs, rhs, group_sizes, group_offset, num_actual_groups, tile, out_dtype):
  from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2  # pylint: disable=import-outside-toplevel

  tile_info = (
      tgmm_v2.calculate_tgmm_tiling if tile is None else gmm_v2.TileSizes(tile_m=tile[0], tile_k=tile[1], tile_n=tile[2])
  )
  return tgmm_v2.tgmm_v2(
      lhs=lhs,
      rhs=rhs,
      group_sizes=group_sizes,
      num_actual_groups=num_actual_groups,
      rhs_scale=None,
      precision=jax.lax.Precision.DEFAULT,
      preferred_element_type=out_dtype,
      group_offset=jnp.asarray(group_offset, jnp.int32),
      tile_info=tile_info,
  )


# name -> (gmm_fn, tgmm_fn). Tests register a pure-JAX model of the kernels.
IMPLS = {"tpu": (_tpu_gmm, _tpu_tgmm)}


def _boundary_row(group_sizes, first_hi_group):
  """First row owned by group `first_hi_group` (= #rows of all earlier groups)."""
  idx = jnp.arange(group_sizes.shape[0], dtype=jnp.int32)
  return jnp.sum(jnp.where(idx < first_hi_group, group_sizes, 0)).astype(jnp.int32)


def _chained_pair(gmm_fn, lhs, rhs_lo, rhs_hi, group_sizes, base, tile, out_dtype, transpose_rhs=False):
  """lhs @ [rhs_lo; rhs_hi] (grouped) accumulated into a single buffer."""
  g_lo = rhs_lo.shape[0]
  kw = {"transpose_rhs": True} if transpose_rhs else {}
  out1 = gmm_fn(lhs, rhs_lo, group_sizes, base, None, True, tile, out_dtype, **kw)
  m = out1.shape[0]
  w = min(_REPAIR_ROWS, m)
  b = _boundary_row(group_sizes, base + g_lo)
  start = jnp.clip((b // w) * w, 0, m - w)
  saved = jax.lax.dynamic_slice_in_dim(out1, start, w, axis=0)
  # Order the window read before the in-place (aliased) second kernel.
  saved, out1 = jax.lax.optimization_barrier((saved, out1))
  out2 = gmm_fn(lhs, rhs_hi, group_sizes, base + g_lo, out1, False, tile, out_dtype, **kw)
  cur = jax.lax.dynamic_slice_in_dim(out2, start, w, axis=0)
  rows = start + jnp.arange(w, dtype=jnp.int32)[:, None]
  fixed = jnp.where(rows < b, saved, cur)
  return jax.lax.dynamic_update_slice_in_dim(out2, fixed, start, axis=0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _split_gmm(lhs, rhs_lo, rhs_hi, group_sizes, cfg):
  return _split_gmm_fwd(lhs, rhs_lo, rhs_hi, group_sizes, cfg)[0]


def _split_gmm_fwd(lhs, rhs_lo, rhs_hi, group_sizes, cfg):
  tiling, out_dtype, base, impl, _ = cfg
  gmm_fn, _ = IMPLS[impl]
  fwd_tile = None if tiling is None else tiling[0:3]
  out = _chained_pair(gmm_fn, lhs, rhs_lo, rhs_hi, group_sizes, base, fwd_tile, out_dtype)
  return out, (lhs, rhs_lo, rhs_hi, group_sizes)


def _split_gmm_bwd(cfg, res, grad):
  tiling, _, base, impl, use_dlhs_transpose_rhs = cfg
  gmm_fn, tgmm_fn = IMPLS[impl]
  lhs, rhs_lo, rhs_hi, group_sizes = res
  g_lo, g_hi = rhs_lo.shape[0], rhs_hi.shape[0]
  dlhs_tile = None if tiling is None else tiling[3:6]
  drhs_tile = None if tiling is None else tiling[6:9]
  grad = grad.astype(lhs.dtype)
  dlhs = _chained_pair(
      gmm_fn,
      grad,
      rhs_lo if use_dlhs_transpose_rhs else rhs_lo.swapaxes(1, 2),
      rhs_hi if use_dlhs_transpose_rhs else rhs_hi.swapaxes(1, 2),
      group_sizes,
      base,
      dlhs_tile,
      lhs.dtype,
      transpose_rhs=use_dlhs_transpose_rhs,
  )
  drhs_lo = tgmm_fn(lhs, grad, group_sizes, base, g_lo, drhs_tile, rhs_lo.dtype)
  drhs_hi = tgmm_fn(lhs, grad, group_sizes, base + g_lo, g_hi, drhs_tile, rhs_hi.dtype)
  return dlhs, drhs_lo, drhs_hi, None


_split_gmm.defvjp(_split_gmm_fwd, _split_gmm_bwd)


def gmm_split_experts(
    lhs: jax.Array,
    rhs_lo: jax.Array,
    rhs_hi: jax.Array,
    group_sizes: jax.Array,
    *,
    tiling: tuple[int, ...] | None,
    preferred_element_type,
    group_offset: int = 0,
    impl: str = "tpu",
    use_dlhs_transpose_rhs: bool = False,
) -> jax.Array:
  """Grouped matmul against two contiguous expert blocks without concatenating them.

  Args:
    lhs: [M, K] expert-sorted tokens.
    rhs_lo: [G_lo, K, N] weights for experts [group_offset, group_offset+G_lo).
    rhs_hi: [G_hi, K, N] weights for the following G_hi experts.
    group_sizes: int32[num_groups] rows per expert.
    tiling: 9-tuple (fwd m,k,n, dlhs m,k,n, drhs m,k,n) or None for heuristics.
    preferred_element_type: output dtype.
    group_offset: static python int, first expert handled by rhs_lo.
    impl: key into IMPLS.
    use_dlhs_transpose_rhs: pass transpose_rhs=True to gmm_v2 in backward dlhs.

  Returns:
    [M, N] = rows of each expert multiplied by that expert's weight; rows outside
    [group_offset, group_offset+G_lo+G_hi) experts are zero.
  """
  assert isinstance(group_offset, int), "gmm_split_experts needs a static group_offset"
  assert rhs_lo.shape[1:] == rhs_hi.shape[1:], (rhs_lo.shape, rhs_hi.shape)
  cfg = (
      None if tiling is None else tuple(int(t) for t in tiling),
      jnp.dtype(preferred_element_type),
      group_offset,
      impl,
      bool(use_dlhs_transpose_rhs),
  )
  return _split_gmm(lhs, rhs_lo, rhs_hi, group_sizes, cfg)


def split_wi_to_normal(w: jax.Array) -> jax.Array:
  """Converts `wi` from `[2, *scan_dims, E, emb // 2, 2 * mlp]` to `[E, *scan_dims, emb, 2 * mlp]`."""
  assert w.ndim >= 4 and w.shape[0] == 2, f"Expected split-expert wi shape (2, ..., E, emb//2, 2*mlp), got {w.shape}"
  scan_dims = w.shape[1:-3]
  num_experts, emb, mlp2 = w.shape[-3], 2 * w.shape[-2], w.shape[-1]
  w = jnp.moveaxis(w, 0, -4)  # (*scan_dims, 2, E, emb // 2, 2 * mlp)
  w = w.reshape(*scan_dims, num_experts, emb, mlp2)
  return jnp.moveaxis(w, -3, 0)  # (E, *scan_dims, emb, 2 * mlp)


def normal_wi_to_split(w: jax.Array) -> jax.Array:
  """Converts `wi` from `[E, *scan_dims, emb, 2 * mlp]` to `[2, *scan_dims, E, emb // 2, 2 * mlp]`."""
  assert w.ndim >= 3 and w.shape[-2] % 2 == 0, f"Expected normal wi shape (E, ..., emb, 2*mlp), got {w.shape}"
  num_experts = w.shape[0]
  scan_dims = w.shape[1:-2]
  emb, mlp2 = w.shape[-2], w.shape[-1]
  w = jnp.moveaxis(w, 0, -3)  # (*scan_dims, E, emb, 2 * mlp)
  w = w.reshape(*scan_dims, 2, num_experts, emb // 2, mlp2)
  return jnp.moveaxis(w, -4, 0)  # (2, *scan_dims, E, emb // 2, 2 * mlp)


def convert_checkpoint_tree(state_pytree, *, to_split: bool):
  """Converts all routed MoE `wi` leaves (`params` and `opt_state`) in a checkpoint pytree.

  Args:
    state_pytree: Loaded checkpoint pytree (e.g. `params` or full training state).
    to_split: If True, converts standard `[E, *scan_dims, emb, 2 * mlp]` `wi` leaves to
      `[2, *scan_dims, E, emb // 2, 2 * mlp]` (`moe_split_expert_weight_layout=True`).
      If False, converts split-expert `wi` leaves back to standard MaxText shape.

  Returns:
    A new pytree with all `wi` array leaves reshaped to the target layout.
  """
  fn = normal_wi_to_split if to_split else split_wi_to_normal

  def _map_leaf(path, x):
    keys = [getattr(k, "key", str(k)) for k in path]
    if "wi" in keys and isinstance(x, (jax.Array, jnp.ndarray)):
      return fn(x)
    return x

  return jax.tree.map_with_path(_map_leaf, state_pytree)

