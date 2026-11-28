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

"""Fused Pallas TPU kernels for Kimi-K3 Block Attention Residuals (AttnRes).

Implements software-pipelined async-DMA forward and backward Pallas kernels for
Block Attention Residuals (`_apply_attn_res` and fused `_apply_attn_res` +
sublayer `RMSNorm`):

1. Active-slot-only ping-pong async DMA and compute (`pltpu.make_async_copy`,
   `@pl.loop(0, num_committed)`): transfers, scores, unpacks, and scatters only
   the `0 .. num_committed - 1` active block slots (plus optional `pending` when
   `has_pending` is true) rather than all `M` padded buffer slots.
2. 64-row VREG strip loops (`strip_t = gcd(block_t, 64)`, `sub_d = 256`,
   4x-unrolled `1024`-column outer groups): amortizes cross-lane XLU reduction
   and MXU latencies over 64 token rows while keeping per-sub-chunk register
   footprints at 16 F32 VREGs with `(1, D)` weight and `(8, D)` weight-grad
   tiles.
3. Compact forward statistics (`fwd_stats` of shape `[B, T, 128]`): saves
   `(probs, r_inv, beta, r_norm)` from forward and unpacks all 3 per-slot
   vectors simultaneously in backward via a single exact 0/1 `(128, 384)` MXU
   projection per active slot, skipping depth RMSNorm, query dot-product,
   softmax, and sublayer `r_norm` recomputation.
"""

import dataclasses
import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_STAT_STRIDE = 32
_STAT_WIDTH = 128
_MIN_STRIP_T = 16
_MAX_STRIP_T = 64
_WGRAD_ROWS = 8
_DEFAULT_FWD_BLOCK_T = 128
_DEFAULT_BWD_BLOCK_T = 64
_DEFAULT_CHUNK_D = 1024
_DEFAULT_FALLBACK_VMEM_CAPACITY_BYTES = 64 << 20
_HBM_SPEC = pl.BlockSpec(memory_space=pltpu.HBM)
_SMEM_SPEC = pl.BlockSpec(memory_space=pltpu.SMEM)


@dataclasses.dataclass(frozen=True)
class AttnResKernelConfig:
  """Tiling and compiler configuration for the Pallas AttnRes TPU kernels."""

  fwd_block_t: int = _DEFAULT_FWD_BLOCK_T
  bwd_block_t: int = _DEFAULT_BWD_BLOCK_T
  chunk_d: int = _DEFAULT_CHUNK_D
  vmem_limit_bytes: int | None = None
  interpret: bool = False


def _tpu_vmem_capacity_bytes(default_bytes: int = _DEFAULT_FALLBACK_VMEM_CAPACITY_BYTES) -> int:
  """Returns active per-core TPU VMEM capacity in bytes, or `default_bytes` off TPU."""
  try:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes)
  except (ValueError, RuntimeError, NotImplementedError, AttributeError):
    return default_bytes


def _resolve_vmem_limit_bytes(config: AttnResKernelConfig) -> int:
  """Resolves the per-kernel `vmem_limit_bytes` for the active TPU generation."""
  if config.vmem_limit_bytes is not None:
    return config.vmem_limit_bytes
  cap = _tpu_vmem_capacity_bytes()
  return max(12 << 20, cap - (4 << 20))


def _vmem_working_budget(vmem_limit_bytes: int) -> int:
  """Returns usable VMEM budget after reserving compiler spill/constant headroom."""
  headroom = (8 << 20) if vmem_limit_bytes >= (32 << 20) else (2 << 20)
  return max(4 << 20, vmem_limit_bytes - headroom)


def _estimate_vmem_bytes(
    block_t: int,
    m: int,
    d: int,
    *,
    p_itemsize: int,
    b_itemsize: int,
    h_itemsize: int,
    out_itemsize: int,
    has_pending: bool,
    apply_sublayer_norm: bool,
    is_bwd: bool,
    return_stats: bool = True,
) -> int:
  """Estimates explicit VMEM footprint (double-buffered tiles + scratch) in bytes."""
  strip_t = math.gcd(block_t, _MAX_STRIP_T)
  if is_bwd:
    bytes_per_token = (
        2 * d * (out_itemsize + 2 * p_itemsize + 2 * m * b_itemsize + (2 * b_itemsize if has_pending else 0))
        + 2 * _STAT_WIDTH * 4
    )
    fixed_bytes = (
        _WGRAD_ROWS * d * (8 + (out_itemsize + 4 if apply_sublayer_norm else 0))
        + (6 * _STAT_STRIDE + 3) * strip_t * 128 * 4
        + (strip_t * d * h_itemsize if apply_sublayer_norm else 0)
        + (512 << 10)
    )
  else:
    bytes_per_token = 2 * d * (p_itemsize + m * b_itemsize + out_itemsize + (b_itemsize if has_pending else 0)) + (
        2 * _STAT_WIDTH * 4 if return_stats else 0
    )
    fixed_bytes = (
        _WGRAD_ROWS * d * (4 + (out_itemsize if apply_sublayer_norm else 0))
        + (3 * _STAT_STRIDE + 3) * strip_t * 128 * 4
        + (strip_t * d * h_itemsize if apply_sublayer_norm else 0)
    )
  return block_t * bytes_per_token + fixed_bytes


def _out_struct(shape: tuple[int, ...], dtype: Any, like: jax.Array) -> jax.ShapeDtypeStruct:
  return jax.ShapeDtypeStruct(
      shape,
      dtype,
      manual_axis_type=jax.typeof(like).manual_axis_type,
  )


def _whole_2d(rows: int, cols: int) -> pl.BlockSpec:
  return pl.BlockSpec((rows, cols), lambda *_: (0, 0))


def _rsqrt(x: jax.Array, *, interpret: bool) -> jax.Array:
  """Full-precision float32 inverse square root (1 NR step on TPU `erq`)."""
  r = jax.lax.rsqrt(x)
  if interpret:
    return r
  return r * (1.5 - 0.5 * x * (r * r))


def _tok_slice(tile: int | jax.Array, block_t: int) -> pl.Slice:
  if isinstance(tile, int):
    return pl.ds(tile * block_t, block_t)
  return pl.ds(pl.multiple_of(tile * block_t, block_t), block_t)


def _row_chunks(rows: int, chunk: int, body) -> None:
  if chunk >= rows:
    body(pl.ds(0, rows))
    return

  @pl.loop(0, rows // chunk)
  def _(i):
    body(pl.ds(pl.multiple_of(i * chunk, chunk), chunk))


def _sub_slice(ig: int | jax.Array, group_d: int, sub: int, sub_d: int) -> pl.Slice:
  if isinstance(ig, int):
    return pl.ds(ig * group_d + sub * sub_d, sub_d)
  base = pl.multiple_of(ig * group_d, group_d) + sub * sub_d
  return pl.ds(pl.multiple_of(base, sub_d), sub_d)


def _col_each_grouped(num_groups: int, group_d: int, subs_per_group: int, sub_d: int, body) -> None:
  """Iterates over `num_groups` hardware loop steps, unrolling `subs_per_group` sub-chunks."""
  if num_groups == 1:
    for sub in range(subs_per_group):
      body(_sub_slice(0, group_d, sub, sub_d))
    return

  @pl.loop(0, num_groups)
  def _(ig):
    for sub in range(subs_per_group):
      body(_sub_slice(ig, group_d, sub, sub_d))


def _col_sum_grouped(
    num_groups: int,
    group_d: int,
    subs_per_group: int,
    sub_d: int,
    slot_ref,
    term_fn,
) -> jax.Array:
  """Sums `(strip_t, 128)` across `subs_per_group` in VREGs and `num_groups` via `slot_ref`."""
  if num_groups == 1:
    acc = term_fn(_sub_slice(0, group_d, 0, sub_d))
    for sub in range(1, subs_per_group):
      acc = acc + term_fn(_sub_slice(0, group_d, sub, sub_d))
    return acc

  @pl.loop(0, num_groups)
  def _(ig):
    acc = term_fn(_sub_slice(ig, group_d, 0, sub_d))
    for sub in range(1, subs_per_group):
      acc = acc + term_fn(_sub_slice(ig, group_d, sub, sub_d))
    slot_ref[...] = jnp.where(ig == 0, acc, slot_ref[...] + acc)

  return slot_ref[...]


def _col_sum2_grouped(
    num_groups: int,
    group_d: int,
    subs_per_group: int,
    sub_d: int,
    slot0_ref,
    slot1_ref,
    term2_fn,
) -> tuple[jax.Array, jax.Array]:
  """Sums pairs of `(strip_t, 128)` across `subs_per_group` in VREGs and `num_groups` via VMEM."""
  if num_groups == 1:
    a0, a1 = term2_fn(_sub_slice(0, group_d, 0, sub_d))
    for sub in range(1, subs_per_group):
      t0, t1 = term2_fn(_sub_slice(0, group_d, sub, sub_d))
      a0 = a0 + t0
      a1 = a1 + t1
    return a0, a1

  @pl.loop(0, num_groups)
  def _(ig):
    a0, a1 = term2_fn(_sub_slice(ig, group_d, 0, sub_d))
    for sub in range(1, subs_per_group):
      t0, t1 = term2_fn(_sub_slice(ig, group_d, sub, sub_d))
      a0 = a0 + t0
      a1 = a1 + t1
    slot0_ref[...] = jnp.where(ig == 0, a0, slot0_ref[...] + a0)
    slot1_ref[...] = jnp.where(ig == 0, a1, slot1_ref[...] + a1)

  return slot0_ref[...], slot1_ref[...]


def _tile_cols(vec: jax.Array, tiles_128: int) -> jax.Array:
  return vec if tiles_128 == 1 else jnp.tile(vec, (1, tiles_128))


def _fold_cols_128(x: jax.Array) -> jax.Array:
  """Sums `(strip_t, sub_d)` across 128-lane tiles into `(strip_t, 128)`."""
  acc = x[:, 0:128]
  for c in range(128, x.shape[1], 128):
    acc = acc + x[:, c : c + 128]
  return acc


def _fold_rows_8(x: jax.Array) -> jax.Array:
  """Sums `(strip_t, sub_d)` across 8-row sublane groups into `(8, sub_d)`."""
  acc = x[0:_WGRAD_ROWS, :]
  for r in range(_WGRAD_ROWS, x.shape[0], _WGRAD_ROWS):
    acc = acc + x[r : r + _WGRAD_ROWS, :]
  return acc


def _unpack_slot_triple(
    stats_vec: jax.Array,
    slot_idx: int | jax.Array,
    r_iota3: jax.Array | None,
    c_iota3: jax.Array | None,
    *,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Unpacks `(prob, r_inv, beta)` for `slot_idx` in a single `(strip_t, 384)` MXU projection."""
  sh = stats_vec.shape
  if interpret:
    p_v = jnp.broadcast_to(jax.lax.dynamic_slice_in_dim(stats_vec, slot_idx, 1, axis=1), sh)
    r_v = jnp.broadcast_to(jax.lax.dynamic_slice_in_dim(stats_vec, _STAT_STRIDE + slot_idx, 1, axis=1), sh)
    b_v = jnp.broadcast_to(jax.lax.dynamic_slice_in_dim(stats_vec, 2 * _STAT_STRIDE + slot_idx, 1, axis=1), sh)
    return p_v, r_v, b_v
  assert r_iota3 is not None and c_iota3 is not None
  sel3 = (
      ((r_iota3 == slot_idx) & (c_iota3 < 128))
      | ((r_iota3 == (_STAT_STRIDE + slot_idx)) & (c_iota3 >= 128) & (c_iota3 < 256))
      | ((r_iota3 == (2 * _STAT_STRIDE + slot_idx)) & (c_iota3 >= 256))
  ).astype(jnp.float32)
  out3 = jax.lax.dot_general(
      stats_vec,
      sel3,
      (((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )
  return out3[:, 0:128], out3[:, 128:256], out3[:, 256:384]


def _unpack_single_col(
    stats_vec: jax.Array,
    col_idx: int,
    r_iota1: jax.Array | None,
    *,
    interpret: bool,
) -> jax.Array:
  """Broadcasts static column `col_idx` of `stats_vec` across all 128 lanes."""
  if interpret:
    return jnp.broadcast_to(stats_vec[:, col_idx : col_idx + 1], stats_vec.shape)
  assert r_iota1 is not None
  sel1 = (r_iota1 == col_idx).astype(jnp.float32)
  return jax.lax.dot_general(
      stats_vec,
      sel1,
      (((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
      precision=jax.lax.Precision.HIGHEST,
  )


def _resolve_norm_dtypes(
    h_dtype: Any,
    apply_sublayer_norm: bool,
    norm_dtype: Any,
) -> tuple[Any, Any]:
  """Resolves `(y_dtype, out_dtype)` matching MaxText `RMSNorm` (`dtype=None` -> `float32`)."""
  if not apply_sublayer_norm:
    return h_dtype, h_dtype
  resolved_dt = jnp.dtype(norm_dtype) if norm_dtype is not None else jnp.dtype(jnp.float32)
  return resolved_dt, resolved_dt


def _choose_tiling(
    total_tokens: int,
    d: int,
    target_block_t: int,
    target_chunk_d: int,
    vmem_bytes_fn,
    max_vmem_bytes: int,
) -> tuple[int, int, int, int]:
  """Selects `(block_t, strip_t, group_d, sub_d)` fitting VMEM and 16-VREG sub-chunks."""
  block_t = math.gcd(total_tokens, target_block_t)
  while block_t > _MIN_STRIP_T and vmem_bytes_fn(block_t) > max_vmem_bytes:
    if block_t % (2 * _MIN_STRIP_T) == 0:
      block_t //= 2
    else:
      block_t = _MIN_STRIP_T
  if block_t < _MIN_STRIP_T or block_t % _MIN_STRIP_T != 0 or total_tokens % block_t != 0:
    raise ValueError(
        f"Invalid token tiling for Pallas AttnRes: total_tokens={total_tokens}, "
        f"target_block_t={target_block_t}, chosen block_t={block_t}."
    )
  needed = vmem_bytes_fn(block_t)
  if needed > max_vmem_bytes:
    raise ValueError(
        f"Pallas AttnRes working set ({needed / (1 << 20):.1f} MiB at block_t={block_t}) "
        f"exceeds VMEM budget ({max_vmem_bytes / (1 << 20):.1f} MiB)."
    )
  strip_t = math.gcd(block_t, _MAX_STRIP_T)
  max_sub_d = max(128, (16 * 1024) // strip_t)
  sub_d = math.gcd(d, min(target_chunk_d, max_sub_d))
  max_group_d = max(sub_d, (64 * 1024) // strip_t)
  group_d = math.gcd(d, max(sub_d, min(target_chunk_d, max_group_d)))
  if group_d % sub_d != 0:
    group_d = sub_d
  if sub_d < 128 or sub_d % 128 != 0:
    raise ValueError(f"Embedding dimension {d} must be divisible by 128; got {sub_d=}.")
  return block_t, strip_t, group_d, sub_d


def _axis_shards(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, entry: Any) -> int:
  if entry is None:
    return 1
  axes = (entry,) if isinstance(entry, str) else tuple(entry)
  return math.prod(mesh.shape.get(ax, 1) for ax in axes if ax is not None)


def _active_entry_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, entry: Any) -> tuple[str, ...]:
  if entry is None:
    return ()
  raw = (entry,) if isinstance(entry, str) else tuple(entry)
  return tuple(ax for ax in raw if ax is not None and mesh.shape.get(ax, 1) > 1)


def _norm_partitions(spec: jax.sharding.PartitionSpec, ndim: int) -> tuple[Any, ...]:
  parts = tuple(spec.partitions)
  if len(parts) < ndim:
    parts = parts + (None,) * (ndim - len(parts))
  return parts[:ndim]


def _has_active_axes(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh, axes: Any) -> bool:
  return any(mesh.shape.get(ax, 1) > 1 for ax in axes)


def _has_explicit_mesh(x: jax.Array) -> bool:
  sharding = getattr(jax.typeof(x), "sharding", None)
  if not isinstance(sharding, jax.sharding.NamedSharding) or sharding.mesh.empty:
    return False
  return any(t == jax.sharding.AxisType.Explicit for t in sharding.mesh.axis_types)


def can_use_attn_res_pallas(
    prefix_sum: jax.Array,
    blocks: jax.Array | None = None,
    *,
    pending: jax.Array | None = None,
    sublayer_norm_scale: jax.Array | None = None,
    norm_dtype: Any = None,
    mesh: jax.sharding.Mesh | None = None,
    activation_pspec: jax.sharding.PartitionSpec | None = None,
    config: AttnResKernelConfig = AttnResKernelConfig(),
    interpret: bool | None = None,
) -> bool:
  """Returns whether inputs satisfy the Pallas TPU kernel shape, sharding, and VMEM contract."""
  use_interpret = config.interpret if interpret is None else interpret
  if not use_interpret and jax.default_backend() != "tpu":
    return False
  if prefix_sum.ndim != 3 or prefix_sum.dtype not in (jnp.bfloat16, jnp.float32):
    return False
  batch, seq_len, d = prefix_sum.shape
  if d < 128 or d % 128 != 0:
    return False
  if blocks is not None:
    if blocks.ndim != 4 or blocks.dtype not in (jnp.bfloat16, jnp.float32):
      return False
    if blocks.shape[:2] != (batch, seq_len) or blocks.shape[3] != d or blocks.shape[2] <= 0:
      return False
    total_slots = blocks.shape[2] + (1 if pending is not None else 0) + 1
    if total_slots > _STAT_STRIDE:
      return False
  if pending is not None:
    if pending.shape != prefix_sum.shape or pending.dtype not in (jnp.bfloat16, jnp.float32):
      return False

  b_shards = 1
  t_shards = 1
  sharding = getattr(jax.typeof(prefix_sum), "sharding", None)
  if _has_explicit_mesh(prefix_sum):
    assert isinstance(sharding, jax.sharding.NamedSharding)
    exp_mesh = sharding.mesh
    p_spec = sharding.spec
    if _has_active_axes(exp_mesh, p_spec.unreduced):
      return False
    p0, p1, p2 = _norm_partitions(p_spec, 3)
    if _axis_shards(exp_mesh, p2) > 1:
      return False
    if blocks is not None:
      b_sharding = getattr(jax.typeof(blocks), "sharding", None)
      if not isinstance(b_sharding, jax.sharding.NamedSharding) or b_sharding.mesh != exp_mesh:
        return False
      b_spec = b_sharding.spec
      if _has_active_axes(exp_mesh, b_spec.unreduced):
        return False
      b0, b1, b2, b3 = _norm_partitions(b_spec, 4)
      if (
          _active_entry_axes(exp_mesh, b0) != _active_entry_axes(exp_mesh, p0)
          or _active_entry_axes(exp_mesh, b1) != _active_entry_axes(exp_mesh, p1)
          or _axis_shards(exp_mesh, b2) > 1
          or _axis_shards(exp_mesh, b3) > 1
      ):
        return False
    if pending is not None:
      pd_sharding = getattr(jax.typeof(pending), "sharding", None)
      if not isinstance(pd_sharding, jax.sharding.NamedSharding) or pd_sharding.mesh != exp_mesh:
        return False
      pd_spec = pd_sharding.spec
      if _has_active_axes(exp_mesh, pd_spec.unreduced):
        return False
      pd0, pd1, pd2 = _norm_partitions(pd_spec, 3)
      if (
          _active_entry_axes(exp_mesh, pd0) != _active_entry_axes(exp_mesh, p0)
          or _active_entry_axes(exp_mesh, pd1) != _active_entry_axes(exp_mesh, p1)
          or _axis_shards(exp_mesh, pd2) > 1
      ):
        return False
    b_shards = _axis_shards(exp_mesh, p0)
    t_shards = _axis_shards(exp_mesh, p1)
  elif mesh is not None and not mesh.empty and activation_pspec is not None:
    if len(activation_pspec) >= 3 and _axis_shards(mesh, activation_pspec[2]) > 1:
      return False
    if len(activation_pspec) >= 1:
      b_shards = _axis_shards(mesh, activation_pspec[0])
    if len(activation_pspec) >= 2:
      t_shards = _axis_shards(mesh, activation_pspec[1])
  elif isinstance(sharding, jax.sharding.NamedSharding) and not sharding.mesh.empty:
    p0, p1, p2 = _norm_partitions(sharding.spec, 3)
    if _axis_shards(sharding.mesh, p2) > 1:
      return False
    b_shards = _axis_shards(sharding.mesh, p0)
    t_shards = _axis_shards(sharding.mesh, p1)

  if batch % b_shards != 0 or seq_len % t_shards != 0:
    return False
  local_tokens = (batch // b_shards) * (seq_len // t_shards)
  if local_tokens < _MIN_STRIP_T or (local_tokens % _MIN_STRIP_T != 0):
    return False

  if blocks is not None and (not use_interpret or config.vmem_limit_bytes is not None):
    apply_sublayer_norm = sublayer_norm_scale is not None
    h_dtype = jnp.result_type(prefix_sum.dtype, blocks.dtype)
    _, out_dtype = _resolve_norm_dtypes(h_dtype, apply_sublayer_norm, norm_dtype)
    budget = _vmem_working_budget(_resolve_vmem_limit_bytes(config))
    min_bwd_vmem = _estimate_vmem_bytes(
        _MIN_STRIP_T,
        blocks.shape[2],
        d,
        p_itemsize=prefix_sum.dtype.itemsize,
        b_itemsize=blocks.dtype.itemsize,
        h_itemsize=jnp.dtype(h_dtype).itemsize,
        out_itemsize=jnp.dtype(out_dtype).itemsize,
        has_pending=pending is not None,
        apply_sublayer_norm=apply_sublayer_norm,
        is_bwd=True,
    )
    if min_bwd_vmem > budget:
      return False

  return True


def _fwd_kernel(
    meta_ref,
    p_hbm,
    b_hbm,
    sw_ref,
    *rest_refs,
    m: int,
    d: int,
    block_t: int,
    strip_t: int,
    group_d: int,
    sub_d: int,
    num_tiles: int,
    res_eps: float,
    norm_eps: float,
    h_dtype: Any,
    y_dtype: Any,
    out_dtype: Any,
    apply_sublayer_norm: bool,
    has_pending_arg: bool,
    return_stats: bool,
    interpret: bool,
):
  """Async-DMA ping-pong forward kernel for `attn_res_pool` (+ `rms_norm`)."""
  idx = 0
  if has_pending_arg:
    pend_hbm = rest_refs[idx]
    idx += 1
  else:
    pend_hbm = None

  if apply_sublayer_norm:
    eff_scale_ref = rest_refs[idx]
    idx += 1
  else:
    eff_scale_ref = None

  out_hbm = rest_refs[idx]
  idx += 1
  if return_stats:
    stats_hbm = rest_refs[idx]
    idx += 1
  else:
    stats_hbm = None

  p_vmem = rest_refs[idx]
  b_vmem = rest_refs[idx + 1]
  out_vmem = rest_refs[idx + 2]
  score_vmem = rest_refs[idx + 3]
  r_vmem = rest_refs[idx + 4]
  beta_vmem = rest_refs[idx + 5]
  scalar_vmem = rest_refs[idx + 6]
  in_sems = rest_refs[idx + 7]
  out_sems = rest_refs[idx + 8]
  idx += 9

  if has_pending_arg:
    pend_vmem = rest_refs[idx]
    idx += 1
  else:
    pend_vmem = None

  if return_stats:
    stats_vmem = rest_refs[idx]
    idx += 1
  else:
    stats_vmem = None

  if apply_sublayer_norm:
    h_scratch = rest_refs[idx]
    idx += 1
  else:
    h_scratch = None

  inv_d = 1.0 / float(d)
  num_groups = d // group_d
  subs_per_group = group_d // sub_d
  tiles_128 = sub_d // 128
  n_comm = meta_ref[0]
  has_pend = meta_ref[1] if has_pending_arg else 0
  pref_idx = m + (1 if has_pending_arg else 0)
  iota = jax.lax.broadcasted_iota(jnp.int32, (strip_t, _STAT_WIDTH), 1)

  def col_each(body):
    _col_each_grouped(num_groups, group_d, subs_per_group, sub_d, body)

  def start_load(tile, buf):
    sem = in_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(p_hbm.at[ts, :], p_vmem.at[buf, :, :], sem).start()

    @pl.loop(0, n_comm)
    def _(k):
      pltpu.make_async_copy(b_hbm.at[k, ts, :], b_vmem.at[buf, k, :, :], sem).start()

    if has_pending_arg:
      assert pend_hbm is not None and pend_vmem is not None

      @pl.when(has_pend != 0)
      def _():
        pltpu.make_async_copy(pend_hbm.at[ts, :], pend_vmem.at[buf, :, :], sem).start()

  def wait_load(tile, buf):
    sem = in_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(p_hbm.at[ts, :], p_vmem.at[buf, :, :], sem).wait()

    @pl.loop(0, n_comm)
    def _(k):
      pltpu.make_async_copy(b_hbm.at[k, ts, :], b_vmem.at[buf, k, :, :], sem).wait()

    if has_pending_arg:
      assert pend_hbm is not None and pend_vmem is not None

      @pl.when(has_pend != 0)
      def _():
        pltpu.make_async_copy(pend_hbm.at[ts, :], pend_vmem.at[buf, :, :], sem).wait()

  def start_store(tile, buf):
    sem = out_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(out_vmem.at[buf, :, :], out_hbm.at[ts, :], sem).start()
    if return_stats:
      assert stats_vmem is not None and stats_hbm is not None
      pltpu.make_async_copy(stats_vmem.at[buf, :, :], stats_hbm.at[ts, :], sem).start()

  def wait_store(tile, buf):
    sem = out_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(out_vmem.at[buf, :, :], out_hbm.at[ts, :], sem).wait()
    if return_stats:
      assert stats_vmem is not None and stats_hbm is not None
      pltpu.make_async_copy(stats_vmem.at[buf, :, :], stats_hbm.at[ts, :], sem).wait()

  def score_ref_slot(read_chunk_fn):
    def _term2(cols):
      v_c = read_chunk_fn(cols).astype(jnp.float32)
      return _fold_cols_128(v_c * v_c), _fold_cols_128(v_c * sw_ref[:, cols])

    sum_sq, sum_sw = _col_sum2_grouped(
        num_groups, group_d, subs_per_group, sub_d, scalar_vmem.at[0], scalar_vmem.at[1], _term2
    )
    r_128 = jnp.broadcast_to(
        _rsqrt(jnp.sum(sum_sq, axis=-1, keepdims=True) * inv_d + res_eps, interpret=interpret),
        (strip_t, 128),
    )
    dot_128 = jnp.broadcast_to(jnp.sum(sum_sw, axis=-1, keepdims=True), (strip_t, 128))
    s_128 = r_128 * dot_128
    b_128 = (r_128 * r_128) * (s_128 * inv_d)
    return r_128, s_128, b_128

  def compute_tile(buf):
    def strip_body(rows):
      r_pref, s_pref, b_pref = score_ref_slot(lambda cols: p_vmem[buf, rows, cols])
      score_vmem[pref_idx] = s_pref
      r_vmem[pref_idx] = r_pref
      beta_vmem[pref_idx] = b_pref
      scalar_vmem[2] = s_pref

      @pl.loop(0, n_comm)
      def _(k):
        r_k, s_k, b_k = score_ref_slot(lambda cols: b_vmem[buf, k, rows, cols])
        score_vmem[k] = s_k
        r_vmem[k] = r_k
        beta_vmem[k] = b_k
        scalar_vmem[2] = jnp.maximum(scalar_vmem[2], s_k)

      if has_pending_arg:
        assert pend_vmem is not None
        pv = pend_vmem

        @pl.when(has_pend != 0)
        def _():
          r_pd, s_pd, b_pd = score_ref_slot(lambda cols: pv[buf, rows, cols])
          score_vmem[m] = s_pd
          r_vmem[m] = r_pd
          beta_vmem[m] = b_pd
          scalar_vmem[2] = jnp.maximum(scalar_vmem[2], s_pd)

      max_s = scalar_vmem[2]
      e_pref = jnp.exp(score_vmem[pref_idx] - max_s)
      score_vmem[pref_idx] = e_pref
      scalar_vmem[1] = e_pref

      @pl.loop(0, n_comm)
      def _(k):
        e_k = jnp.exp(score_vmem[k] - max_s)
        score_vmem[k] = e_k
        scalar_vmem[1] = scalar_vmem[1] + e_k

      if has_pending_arg:

        @pl.when(has_pend != 0)
        def _():
          e_pd = jnp.exp(score_vmem[m] - max_s)
          score_vmem[m] = e_pd
          scalar_vmem[1] = scalar_vmem[1] + e_pd

      inv_sum = 1.0 / scalar_vmem[1]
      score_vmem[pref_idx] = score_vmem[pref_idx] * inv_sum
      for k in range(m):
        score_vmem[k] = jnp.where(k < n_comm, score_vmem[k] * inv_sum, 0.0)
      if has_pending_arg:
        score_vmem[m] = jnp.where(has_pend != 0, score_vmem[m] * inv_sum, 0.0)

      def pool_chunk(cols):
        h_c = _tile_cols(score_vmem[pref_idx], tiles_128) * p_vmem[buf, rows, cols].astype(jnp.float32)
        for k in range(m):
          v_k = jnp.where(k < n_comm, b_vmem[buf, k, rows, cols], 0).astype(jnp.float32)
          h_c = h_c + _tile_cols(score_vmem[k], tiles_128) * v_k
        if has_pending_arg:
          assert pend_vmem is not None
          v_pd = jnp.where(has_pend != 0, pend_vmem[buf, rows, cols], 0).astype(jnp.float32)
          h_c = h_c + _tile_cols(score_vmem[m], tiles_128) * v_pd
        return h_c

      if not apply_sublayer_norm:

        def _store_pool(cols):
          out_vmem[buf, rows, cols] = pool_chunk(cols).astype(h_dtype).astype(out_dtype)

        col_each(_store_pool)
        r_norm_128 = jnp.zeros((strip_t, 128), dtype=jnp.float32)
      else:
        assert h_scratch is not None and eff_scale_ref is not None
        hs = h_scratch
        es = eff_scale_ref

        def _pool_and_sq(cols):
          h_cast = pool_chunk(cols).astype(h_dtype)
          hs[:, cols] = h_cast
          h_f32 = h_cast.astype(jnp.float32)
          return _fold_cols_128(h_f32 * h_f32)

        sum_h_sq = _col_sum_grouped(num_groups, group_d, subs_per_group, sub_d, scalar_vmem.at[0], _pool_and_sq)
        r_norm_128 = jnp.broadcast_to(
            _rsqrt(jnp.sum(sum_h_sq, axis=-1, keepdims=True) * inv_d + norm_eps, interpret=interpret),
            (strip_t, 128),
        )
        scalar_vmem[0] = r_norm_128

        def _norm_and_store(cols):
          rn_c = _tile_cols(scalar_vmem[0], tiles_128)
          y_cast = (hs[:, cols].astype(jnp.float32) * rn_c).astype(y_dtype)
          out_vmem[buf, rows, cols] = (y_cast.astype(out_dtype) * es[:, cols]).astype(out_dtype)

        col_each(_norm_and_store)

      if return_stats:
        assert stats_vmem is not None
        st0 = jnp.where(iota == pref_idx, score_vmem[pref_idx], 0.0)
        st0 = jnp.where(iota == _STAT_STRIDE + pref_idx, r_vmem[pref_idx], st0)
        st0 = jnp.where(iota == 2 * _STAT_STRIDE + pref_idx, beta_vmem[pref_idx], st0)
        st0 = jnp.where(iota == 3 * _STAT_STRIDE, r_norm_128, st0)
        scalar_vmem[1] = st0

        @pl.loop(0, n_comm)
        def _(k):
          cur = scalar_vmem[1]
          cur = jnp.where(iota == k, score_vmem[k], cur)
          cur = jnp.where(iota == _STAT_STRIDE + k, r_vmem[k], cur)
          cur = jnp.where(iota == 2 * _STAT_STRIDE + k, beta_vmem[k], cur)
          scalar_vmem[1] = cur

        if has_pending_arg:

          @pl.when(has_pend != 0)
          def _():
            cur = scalar_vmem[1]
            cur = jnp.where(iota == m, score_vmem[m], cur)
            cur = jnp.where(iota == _STAT_STRIDE + m, r_vmem[m], cur)
            cur = jnp.where(iota == 2 * _STAT_STRIDE + m, beta_vmem[m], cur)
            scalar_vmem[1] = cur

        stats_vmem[buf, rows, :] = scalar_vmem[1]

    _row_chunks(block_t, strip_t, strip_body)

  if num_tiles == 1:
    start_load(0, 0)
    wait_load(0, 0)
    compute_tile(0)
    start_store(0, 0)
    wait_store(0, 0)
  else:
    start_load(0, 0)

    @pl.loop(0, num_tiles)
    def _(tile):
      buf = jax.lax.rem(tile, 2)
      nxt_buf = 1 - buf

      @pl.when(tile + 1 < num_tiles)
      def _():
        start_load(tile + 1, nxt_buf)

      @pl.when(tile >= 2)
      def _():
        wait_store(tile - 2, buf)

      wait_load(tile, buf)
      compute_tile(buf)
      start_store(tile, buf)

    wait_store(num_tiles - 2, (num_tiles - 2) % 2)
    wait_store(num_tiles - 1, (num_tiles - 1) % 2)


def _bwd_kernel(
    meta_ref,
    dout_hbm,
    p_hbm,
    b_hbm,
    sw_ref,
    stats_hbm,
    *rest_refs,
    m: int,
    d: int,
    block_t: int,
    strip_t: int,
    group_d: int,
    sub_d: int,
    num_tiles: int,
    h_dtype: Any,
    y_dtype: Any,
    out_dtype: Any,
    apply_sublayer_norm: bool,
    has_pending_arg: bool,
    interpret: bool,
):
  """Async-DMA ping-pong backward kernel consuming saved forward `fwd_stats`."""
  idx = 0
  if has_pending_arg:
    pend_hbm = rest_refs[idx]
    idx += 1
  else:
    pend_hbm = None

  if apply_sublayer_norm:
    eff_scale_ref = rest_refs[idx]
    idx += 1
  else:
    eff_scale_ref = None

  gp_out_hbm = rest_refs[idx]
  gb_out_hbm = rest_refs[idx + 1]
  dsw_out_ref = rest_refs[idx + 2]
  idx += 3

  if apply_sublayer_norm:
    dgamma_out_ref = rest_refs[idx]
    idx += 1
  else:
    dgamma_out_ref = None

  if has_pending_arg:
    gpend_out_hbm = rest_refs[idx]
    idx += 1
  else:
    gpend_out_hbm = None

  dout_vmem = rest_refs[idx]
  p_vmem = rest_refs[idx + 1]
  b_vmem = rest_refs[idx + 2]
  gp_out_vmem = rest_refs[idx + 3]
  gb_out_vmem = rest_refs[idx + 4]
  coeff_vmem = rest_refs[idx + 5]
  aux_vmem = rest_refs[idx + 6]
  scalar_vmem = rest_refs[idx + 7]
  stats_vmem = rest_refs[idx + 8]
  in_sems = rest_refs[idx + 9]
  out_sems = rest_refs[idx + 10]
  idx += 11

  if has_pending_arg:
    pend_vmem = rest_refs[idx]
    gpend_out_vmem = rest_refs[idx + 1]
    idx += 2
  else:
    pend_vmem = None
    gpend_out_vmem = None

  if apply_sublayer_norm:
    h_scratch = rest_refs[idx]
    idx += 1
  else:
    h_scratch = None

  inv_d = 1.0 / float(d)
  num_groups = d // group_d
  subs_per_group = group_d // sub_d
  tiles_128 = sub_d // 128
  n_comm = meta_ref[0]
  has_pend = meta_ref[1] if has_pending_arg else 0
  pref_idx = m + (1 if has_pending_arg else 0)

  def col_each(body):
    _col_each_grouped(num_groups, group_d, subs_per_group, sub_d, body)

  def _zero_wgrad(cols):
    z8 = jnp.zeros((_WGRAD_ROWS, sub_d), dtype=jnp.float32)
    dsw_out_ref[:, cols] = z8
    if apply_sublayer_norm:
      assert dgamma_out_ref is not None
      dgamma_out_ref[:, cols] = z8

  col_each(_zero_wgrad)

  def _zero_buf_tile(vmem_3d, buf_idx):
    def _zr(rows):
      z_bf = jnp.zeros((strip_t, sub_d), dtype=vmem_3d.dtype)

      def _zc(cols):
        vmem_3d[buf_idx, rows, cols] = z_bf

      col_each(_zc)

    _row_chunks(block_t, strip_t, _zr)

  @pl.when(n_comm < m)
  def _():
    @pl.loop(n_comm, m)
    def _(k):
      def _zbk_rows(rows):
        z_bf = jnp.zeros((strip_t, sub_d), dtype=gb_out_vmem.dtype)

        def _zbk_cols(cols):
          gb_out_vmem[0, k, rows, cols] = z_bf
          gb_out_vmem[1, k, rows, cols] = z_bf

        col_each(_zbk_cols)

      _row_chunks(block_t, strip_t, _zbk_rows)

  if has_pending_arg:
    assert gpend_out_vmem is not None
    gp_zero_ref = gpend_out_vmem

    @pl.when(has_pend == 0)
    def _():
      _zero_buf_tile(gp_zero_ref, 0)
      _zero_buf_tile(gp_zero_ref, 1)

  def start_load(tile, buf):
    sem = in_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(dout_hbm.at[ts, :], dout_vmem.at[buf, :, :], sem).start()
    pltpu.make_async_copy(p_hbm.at[ts, :], p_vmem.at[buf, :, :], sem).start()

    @pl.loop(0, n_comm)
    def _(k):
      pltpu.make_async_copy(b_hbm.at[k, ts, :], b_vmem.at[buf, k, :, :], sem).start()

    pltpu.make_async_copy(stats_hbm.at[ts, :], stats_vmem.at[buf, :, :], sem).start()

    if has_pending_arg:
      assert pend_hbm is not None and pend_vmem is not None

      @pl.when(has_pend != 0)
      def _():
        pltpu.make_async_copy(pend_hbm.at[ts, :], pend_vmem.at[buf, :, :], sem).start()

  def wait_load(tile, buf):
    sem = in_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(dout_hbm.at[ts, :], dout_vmem.at[buf, :, :], sem).wait()
    pltpu.make_async_copy(p_hbm.at[ts, :], p_vmem.at[buf, :, :], sem).wait()

    @pl.loop(0, n_comm)
    def _(k):
      pltpu.make_async_copy(b_hbm.at[k, ts, :], b_vmem.at[buf, k, :, :], sem).wait()

    pltpu.make_async_copy(stats_hbm.at[ts, :], stats_vmem.at[buf, :, :], sem).wait()

    if has_pending_arg:
      assert pend_hbm is not None and pend_vmem is not None

      @pl.when(has_pend != 0)
      def _():
        pltpu.make_async_copy(pend_hbm.at[ts, :], pend_vmem.at[buf, :, :], sem).wait()

  def start_store(tile, buf):
    sem = out_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(gp_out_vmem.at[buf, :, :], gp_out_hbm.at[ts, :], sem).start()

    @pl.loop(0, m)
    def _(k):
      pltpu.make_async_copy(gb_out_vmem.at[buf, k, :, :], gb_out_hbm.at[k, ts, :], sem).start()

    if has_pending_arg:
      assert gpend_out_vmem is not None and gpend_out_hbm is not None
      pltpu.make_async_copy(gpend_out_vmem.at[buf, :, :], gpend_out_hbm.at[ts, :], sem).start()

  def wait_store(tile, buf):
    sem = out_sems.at[buf]
    ts = _tok_slice(tile, block_t)
    pltpu.make_async_copy(gp_out_vmem.at[buf, :, :], gp_out_hbm.at[ts, :], sem).wait()

    @pl.loop(0, m)
    def _(k):
      pltpu.make_async_copy(gb_out_vmem.at[buf, k, :, :], gb_out_hbm.at[k, ts, :], sem).wait()

    if has_pending_arg:
      assert gpend_out_vmem is not None and gpend_out_hbm is not None
      pltpu.make_async_copy(gpend_out_vmem.at[buf, :, :], gpend_out_hbm.at[ts, :], sem).wait()

  def read_dh_chunk(buf, rows, cols):
    if apply_sublayer_norm:
      assert h_scratch is not None
      return h_scratch[:, cols].astype(jnp.float32)
    return dout_vmem[buf, rows, cols].astype(h_dtype).astype(jnp.float32)

  def dot_dh_slot(buf, rows, read_chunk_fn):
    sum_dp = _col_sum_grouped(
        num_groups,
        group_d,
        subs_per_group,
        sub_d,
        scalar_vmem.at[0],
        lambda cols: _fold_cols_128(read_dh_chunk(buf, rows, cols) * read_chunk_fn(cols).astype(jnp.float32)),
    )
    return jnp.broadcast_to(jnp.sum(sum_dp, axis=-1, keepdims=True), (strip_t, 128))

  use_mxu_unpack = not interpret
  r_iota3 = jax.lax.broadcasted_iota(jnp.int32, (_STAT_WIDTH, 384), 0) if use_mxu_unpack else None
  c_iota3 = jax.lax.broadcasted_iota(jnp.int32, (_STAT_WIDTH, 384), 1) if use_mxu_unpack else None
  r_iota1 = (
      jax.lax.broadcasted_iota(jnp.int32, (_STAT_WIDTH, _STAT_WIDTH), 0)
      if (use_mxu_unpack and apply_sublayer_norm)
      else None
  )

  def compute_tile(buf):
    def strip_body(rows):
      stats_vec = stats_vmem[buf, rows, :]
      z128 = jnp.zeros((strip_t, 128), dtype=jnp.float32)
      for k in range(m):
        coeff_vmem[0, k] = z128
      if has_pending_arg:
        coeff_vmem[0, m] = z128

      p_pr, r_pr, b_pr = _unpack_slot_triple(stats_vec, pref_idx, r_iota3, c_iota3, interpret=interpret)
      coeff_vmem[0, pref_idx] = p_pr
      aux_vmem[0, pref_idx] = r_pr
      aux_vmem[1, pref_idx] = b_pr
      r_norm_saved = (
          _unpack_single_col(stats_vec, 3 * _STAT_STRIDE, r_iota1, interpret=interpret) if apply_sublayer_norm else None
      )

      @pl.loop(0, n_comm)
      def _(k):
        pk, rk, bk = _unpack_slot_triple(stats_vec, k, r_iota3, c_iota3, interpret=interpret)
        coeff_vmem[0, k] = pk
        aux_vmem[0, k] = rk
        aux_vmem[1, k] = bk

      if has_pending_arg:

        @pl.when(has_pend != 0)
        def _():
          ppd, rpd, bpd = _unpack_slot_triple(stats_vec, m, r_iota3, c_iota3, interpret=interpret)
          coeff_vmem[0, m] = ppd
          aux_vmem[0, m] = rpd
          aux_vmem[1, m] = bpd

      if apply_sublayer_norm:
        assert h_scratch is not None and eff_scale_ref is not None and dgamma_out_ref is not None
        assert r_norm_saved is not None
        hs = h_scratch
        es = eff_scale_ref
        dg = dgamma_out_ref

        def pool_chunk(cols):
          h_c = _tile_cols(coeff_vmem[0, pref_idx], tiles_128) * p_vmem[buf, rows, cols].astype(jnp.float32)
          for k in range(m):
            v_k = jnp.where(k < n_comm, b_vmem[buf, k, rows, cols], 0).astype(jnp.float32)
            h_c = h_c + _tile_cols(coeff_vmem[0, k], tiles_128) * v_k
          if has_pending_arg:
            assert pend_vmem is not None
            v_pd = jnp.where(has_pend != 0, pend_vmem[buf, rows, cols], 0).astype(jnp.float32)
            h_c = h_c + _tile_cols(coeff_vmem[0, m], tiles_128) * v_pd
          return h_c

        r_norm_128 = r_norm_saved
        scalar_vmem[0] = r_norm_128

        def _norm_pass1(cols):
          h_cast = pool_chunk(cols).astype(h_dtype)
          hs[:, cols] = h_cast
          h_f32 = h_cast.astype(jnp.float32)
          dout_c = dout_vmem[buf, rows, cols]
          rn_c = _tile_cols(scalar_vmem[0], tiles_128)
          y_cast = (h_f32 * rn_c).astype(y_dtype)
          dg[:, cols] = dg[:, cols] + _fold_rows_8(dout_c.astype(jnp.float32) * y_cast.astype(jnp.float32))
          dy_c = (dout_c.astype(out_dtype) * es[:, cols]).astype(y_dtype).astype(jnp.float32)
          return _fold_cols_128(dy_c * h_f32)

        sum_dy_h = _col_sum_grouped(num_groups, group_d, subs_per_group, sub_d, scalar_vmem.at[1], _norm_pass1)
        dot_dy_h_128 = jnp.broadcast_to(jnp.sum(sum_dy_h, axis=-1, keepdims=True) * inv_d, (strip_t, 128))
        scalar_vmem[1] = (r_norm_128 * r_norm_128) * dot_dy_h_128

        def _norm_pass2(cols):
          h_f32 = hs[:, cols].astype(jnp.float32)
          dout_c = dout_vmem[buf, rows, cols]
          rn_c = _tile_cols(scalar_vmem[0], tiles_128)
          rn_sq_c = _tile_cols(scalar_vmem[1], tiles_128)
          dy_c = (dout_c.astype(out_dtype) * es[:, cols]).astype(y_dtype).astype(jnp.float32)
          dh_cast = (rn_c * (dy_c - h_f32 * rn_sq_c)).astype(h_dtype)
          hs[:, cols] = dh_cast
          return _fold_cols_128(dh_cast.astype(jnp.float32) * p_vmem[buf, rows, cols].astype(jnp.float32))

        sum_dp_pref = _col_sum_grouped(num_groups, group_d, subs_per_group, sub_d, scalar_vmem.at[2], _norm_pass2)
        dp_pref_128 = jnp.broadcast_to(jnp.sum(sum_dp_pref, axis=-1, keepdims=True), (strip_t, 128))
      else:
        dp_pref_128 = dot_dh_slot(buf, rows, lambda cols: p_vmem[buf, rows, cols])

      aux_vmem[2, pref_idx] = dp_pref_128
      scalar_vmem[2] = coeff_vmem[0, pref_idx] * dp_pref_128

      @pl.loop(0, n_comm)
      def _(k):
        dp_k = dot_dh_slot(buf, rows, lambda cols: b_vmem[buf, k, rows, cols])
        aux_vmem[2, k] = dp_k
        scalar_vmem[2] = scalar_vmem[2] + coeff_vmem[0, k] * dp_k

      if has_pending_arg:
        assert pend_vmem is not None
        pv_dp = pend_vmem

        @pl.when(has_pend != 0)
        def _():
          dp_pd = dot_dh_slot(buf, rows, lambda cols: pv_dp[buf, rows, cols])
          aux_vmem[2, m] = dp_pd
          scalar_vmem[2] = scalar_vmem[2] + coeff_vmem[0, m] * dp_pd

      sum_p_dp = scalar_vmem[2]
      ds_pref = coeff_vmem[0, pref_idx] * (aux_vmem[2, pref_idx] - sum_p_dp)
      csw_pref = ds_pref * aux_vmem[0, pref_idx]
      cv_pref = -ds_pref * aux_vmem[1, pref_idx]
      p_pref_s = coeff_vmem[0, pref_idx]

      @pl.loop(0, n_comm)
      def _(k):
        ds_k = coeff_vmem[0, k] * (aux_vmem[2, k] - sum_p_dp)
        coeff_vmem[1, k] = ds_k * aux_vmem[0, k]
        coeff_vmem[2, k] = -ds_k * aux_vmem[1, k]

      def _scat_p(cols):
        dh_c = read_dh_chunk(buf, rows, cols)
        sw_c = sw_ref[:, cols]
        p_c = p_vmem[buf, rows, cols].astype(jnp.float32)
        csw_c = _tile_cols(csw_pref, tiles_128)
        gp_out_vmem[buf, rows, cols] = (
            _tile_cols(p_pref_s, tiles_128) * dh_c + csw_c * sw_c + _tile_cols(cv_pref, tiles_128) * p_c
        ).astype(gp_out_vmem.dtype)
        dsw_out_ref[:, cols] = dsw_out_ref[:, cols] + _fold_rows_8(csw_c * p_c)

      col_each(_scat_p)

      @pl.loop(0, n_comm)
      def _(k):
        pk_s = coeff_vmem[0, k]
        csw_k = coeff_vmem[1, k]
        cv_k = coeff_vmem[2, k]

        def _scat_bk(cols):
          dh_c = read_dh_chunk(buf, rows, cols)
          sw_c = sw_ref[:, cols]
          v_k = b_vmem[buf, k, rows, cols].astype(jnp.float32)
          csw_kc = _tile_cols(csw_k, tiles_128)
          gb_out_vmem[buf, k, rows, cols] = (
              _tile_cols(pk_s, tiles_128) * dh_c + csw_kc * sw_c + _tile_cols(cv_k, tiles_128) * v_k
          ).astype(gb_out_vmem.dtype)
          dsw_out_ref[:, cols] = dsw_out_ref[:, cols] + _fold_rows_8(csw_kc * v_k)

        col_each(_scat_bk)

      if has_pending_arg:
        assert pend_vmem is not None and gpend_out_vmem is not None
        pv_scat = pend_vmem
        gp_scat = gpend_out_vmem

        @pl.when(has_pend != 0)
        def _():
          ds_pd = coeff_vmem[0, m] * (aux_vmem[2, m] - sum_p_dp)
          p_pd_s = coeff_vmem[0, m]
          csw_pd = ds_pd * aux_vmem[0, m]
          cv_pd = -ds_pd * aux_vmem[1, m]

          def _scat_pd(cols):
            dh_c = read_dh_chunk(buf, rows, cols)
            sw_c = sw_ref[:, cols]
            v_pd = pv_scat[buf, rows, cols].astype(jnp.float32)
            csw_pdc = _tile_cols(csw_pd, tiles_128)
            gp_scat[buf, rows, cols] = (
                _tile_cols(p_pd_s, tiles_128) * dh_c + csw_pdc * sw_c + _tile_cols(cv_pd, tiles_128) * v_pd
            ).astype(gp_scat.dtype)
            dsw_out_ref[:, cols] = dsw_out_ref[:, cols] + _fold_rows_8(csw_pdc * v_pd)

          col_each(_scat_pd)

    _row_chunks(block_t, strip_t, strip_body)

  if num_tiles == 1:
    start_load(0, 0)
    wait_load(0, 0)
    compute_tile(0)
    start_store(0, 0)
    wait_store(0, 0)
  else:
    start_load(0, 0)

    @pl.loop(0, num_tiles)
    def _(tile):
      buf = jax.lax.rem(tile, 2)
      nxt_buf = 1 - buf

      @pl.when(tile + 1 < num_tiles)
      def _():
        start_load(tile + 1, nxt_buf)

      @pl.when(tile >= 2)
      def _():
        wait_store(tile - 2, buf)

      wait_load(tile, buf)
      compute_tile(buf)
      start_store(tile, buf)

    wait_store(num_tiles - 2, (num_tiles - 2) % 2)
    wait_store(num_tiles - 1, (num_tiles - 1) % 2)


def attn_res_fwd_local(
    prefix_sum: jax.Array,
    blocks: jax.Array,
    num_committed: int | jax.Array,
    res_norm_scale: jax.Array,
    res_proj: jax.Array,
    sublayer_norm_scale: jax.Array | None = None,
    *,
    pending: jax.Array | None = None,
    has_pending: bool | jax.Array | None = None,
    res_epsilon: float = 1e-6,
    norm_epsilon: float = 1e-6,
    norm_dtype: Any = None,
    return_stats: bool = False,
    config: AttnResKernelConfig = AttnResKernelConfig(),
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Per-shard local Pallas forward call for `attn_res_pool` (+ `rms_norm`)."""
  if (pending is None) != (has_pending is None):
    raise ValueError("pending and has_pending must both be provided or both be None.")
  batch, seq_len, m, d = blocks.shape
  if m <= 0:
    raise ValueError(f"attn_res_fwd_local requires blocks.shape[2] >= 1; got {blocks.shape}.")
  total_tokens = batch * seq_len
  has_pending_arg = pending is not None
  if m + (1 if has_pending_arg else 0) + 1 > _STAT_STRIDE:
    raise ValueError(f"Too many candidate slots ({m=}) for packed {_STAT_STRIDE}-slot stats vector.")

  apply_sublayer_norm = sublayer_norm_scale is not None
  h_dtype = jnp.result_type(prefix_sum.dtype, blocks.dtype)
  y_dtype, out_dtype = _resolve_norm_dtypes(h_dtype, apply_sublayer_norm, norm_dtype)

  vmem_limit = _resolve_vmem_limit_bytes(config)
  vmem_budget = _vmem_working_budget(vmem_limit)
  block_t, strip_t, group_d, sub_d = _choose_tiling(
      total_tokens,
      d,
      config.fwd_block_t,
      config.chunk_d,
      lambda bt: _estimate_vmem_bytes(
          bt,
          m,
          d,
          p_itemsize=prefix_sum.dtype.itemsize,
          b_itemsize=blocks.dtype.itemsize,
          h_itemsize=jnp.dtype(h_dtype).itemsize,
          out_itemsize=jnp.dtype(out_dtype).itemsize,
          has_pending=has_pending_arg,
          apply_sublayer_norm=apply_sublayer_norm,
          is_bwd=False,
          return_stats=return_stats,
      ),
      vmem_budget,
  )
  num_tiles = total_tokens // block_t

  p_flat = prefix_sum.reshape(total_tokens, d)
  b_flat = jnp.swapaxes(blocks.reshape(total_tokens, m, d), 0, 1)
  proj_1d = res_proj.reshape(d)
  sw_2d = (res_norm_scale.astype(jnp.float32) * proj_1d.astype(jnp.float32)).reshape(1, d)

  if has_pending_arg:
    meta = jnp.stack(
        [
            jnp.asarray(num_committed, dtype=jnp.int32).reshape(()),
            jnp.asarray(has_pending, dtype=jnp.int32).reshape(()),
        ]
    )
  else:
    meta = jnp.asarray(num_committed, dtype=jnp.int32).reshape(1)

  in_specs = [_SMEM_SPEC, _HBM_SPEC, _HBM_SPEC, _whole_2d(1, d)]
  args = [meta, p_flat, b_flat, sw_2d]
  if has_pending_arg:
    assert pending is not None
    in_specs.append(_HBM_SPEC)
    args.append(pending.astype(blocks.dtype).reshape(total_tokens, d))
  if apply_sublayer_norm:
    assert sublayer_norm_scale is not None
    eff_scale_2d = sublayer_norm_scale.astype(jnp.float32).astype(out_dtype).reshape(1, d)
    in_specs.append(_whole_2d(1, d))
    args.append(eff_scale_2d)

  out_struct = _out_struct((total_tokens, d), out_dtype, prefix_sum)
  if return_stats:
    out_shape: Any = (out_struct, _out_struct((total_tokens, _STAT_WIDTH), jnp.float32, prefix_sum))
    out_specs: Any = (_HBM_SPEC, _HBM_SPEC)
  else:
    out_shape = out_struct
    out_specs = _HBM_SPEC

  scratch_shapes: list[Any] = [
      pltpu.VMEM((2, block_t, d), prefix_sum.dtype),
      pltpu.VMEM((2, m, block_t, d), blocks.dtype),
      pltpu.VMEM((2, block_t, d), out_dtype),
      pltpu.VMEM((_STAT_STRIDE, strip_t, 128), jnp.float32),
      pltpu.VMEM((_STAT_STRIDE, strip_t, 128), jnp.float32),
      pltpu.VMEM((_STAT_STRIDE, strip_t, 128), jnp.float32),
      pltpu.VMEM((3, strip_t, 128), jnp.float32),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
  ]
  if has_pending_arg:
    scratch_shapes.append(pltpu.VMEM((2, block_t, d), blocks.dtype))
  if return_stats:
    scratch_shapes.append(pltpu.VMEM((2, block_t, _STAT_WIDTH), jnp.float32))
  if apply_sublayer_norm:
    scratch_shapes.append(pltpu.VMEM((strip_t, d), h_dtype))

  flops = total_tokens * d * (4 * (m + 1) + (3 if apply_sublayer_norm else 0))
  bytes_accessed = (
      total_tokens * d * (prefix_sum.dtype.itemsize + m * blocks.dtype.itemsize + jnp.dtype(out_dtype).itemsize)
  )

  res = pl.pallas_call(
      functools.partial(
          _fwd_kernel,
          m=m,
          d=d,
          block_t=block_t,
          strip_t=strip_t,
          group_d=group_d,
          sub_d=sub_d,
          num_tiles=num_tiles,
          res_eps=float(res_epsilon),
          norm_eps=float(norm_epsilon),
          h_dtype=h_dtype,
          y_dtype=y_dtype,
          out_dtype=out_dtype,
          apply_sublayer_norm=apply_sublayer_norm,
          has_pending_arg=has_pending_arg,
          return_stats=return_stats,
          interpret=config.interpret,
      ),
      out_shape=out_shape,
      grid=(1,),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
          vmem_limit_bytes=vmem_limit,
      ),
      cost_estimate=pl.CostEstimate(
          flops=flops,
          transcendentals=total_tokens * (m + 3),
          bytes_accessed=bytes_accessed,
      ),
      interpret=config.interpret,
      name="attn_res_rms_norm_fwd" if apply_sublayer_norm else "attn_res_pool_fwd",
  )(*args)

  if return_stats:
    out_flat, stats_flat = res
    return out_flat.reshape(batch, seq_len, d), stats_flat.reshape(batch, seq_len, _STAT_WIDTH)
  return res.reshape(batch, seq_len, d)


def attn_res_bwd_local(
    grad_out: jax.Array,
    prefix_sum: jax.Array,
    blocks: jax.Array,
    num_committed: int | jax.Array,
    res_norm_scale: jax.Array,
    res_proj: jax.Array,
    fwd_stats: jax.Array,
    sublayer_norm_scale: jax.Array | None = None,
    *,
    pending: jax.Array | None = None,
    has_pending: bool | jax.Array | None = None,
    norm_dtype: Any = None,
    config: AttnResKernelConfig = AttnResKernelConfig(),
) -> tuple[jax.Array, ...]:
  """Per-shard local Pallas backward call for `attn_res_pool` (+ `rms_norm`)."""
  if (pending is None) != (has_pending is None):
    raise ValueError("pending and has_pending must both be provided or both be None.")

  batch, seq_len, m, d = blocks.shape
  if m <= 0:
    raise ValueError(f"attn_res_bwd_local requires blocks.shape[2] >= 1; got {blocks.shape}.")
  total_tokens = batch * seq_len
  has_pending_arg = pending is not None
  if m + (1 if has_pending_arg else 0) + 1 > _STAT_STRIDE:
    raise ValueError(f"Too many candidate slots ({m=}) for packed {_STAT_STRIDE}-slot stats vector.")

  apply_sublayer_norm = sublayer_norm_scale is not None
  h_dtype = jnp.result_type(prefix_sum.dtype, blocks.dtype)
  y_dtype, out_dtype = _resolve_norm_dtypes(h_dtype, apply_sublayer_norm, norm_dtype)

  vmem_limit = _resolve_vmem_limit_bytes(config)
  vmem_budget = _vmem_working_budget(vmem_limit)
  block_t, strip_t, group_d, sub_d = _choose_tiling(
      total_tokens,
      d,
      config.bwd_block_t,
      config.chunk_d,
      lambda bt: _estimate_vmem_bytes(
          bt,
          m,
          d,
          p_itemsize=prefix_sum.dtype.itemsize,
          b_itemsize=blocks.dtype.itemsize,
          h_itemsize=jnp.dtype(h_dtype).itemsize,
          out_itemsize=grad_out.dtype.itemsize,
          has_pending=has_pending_arg,
          apply_sublayer_norm=apply_sublayer_norm,
          is_bwd=True,
      ),
      vmem_budget,
  )
  num_tiles = total_tokens // block_t

  dout_flat = grad_out.reshape(total_tokens, d)
  p_flat = prefix_sum.reshape(total_tokens, d)
  b_flat = jnp.swapaxes(blocks.reshape(total_tokens, m, d), 0, 1)
  proj_1d = res_proj.reshape(d)
  ns_f32 = res_norm_scale.astype(jnp.float32)
  pr_f32 = proj_1d.astype(jnp.float32)
  sw_2d = (ns_f32 * pr_f32).reshape(1, d)

  if has_pending_arg:
    meta = jnp.stack(
        [
            jnp.asarray(num_committed, dtype=jnp.int32).reshape(()),
            jnp.asarray(has_pending, dtype=jnp.int32).reshape(()),
        ]
    )
  else:
    meta = jnp.asarray(num_committed, dtype=jnp.int32).reshape(1)

  in_specs = [_SMEM_SPEC, _HBM_SPEC, _HBM_SPEC, _HBM_SPEC, _whole_2d(1, d), _HBM_SPEC]
  args: list[Any] = [meta, dout_flat, p_flat, b_flat, sw_2d, fwd_stats.reshape(total_tokens, _STAT_WIDTH)]

  if has_pending_arg:
    assert pending is not None
    in_specs.append(_HBM_SPEC)
    args.append(pending.astype(blocks.dtype).reshape(total_tokens, d))

  if apply_sublayer_norm:
    assert sublayer_norm_scale is not None
    eff_scale_2d = sublayer_norm_scale.astype(jnp.float32).astype(out_dtype).reshape(1, d)
    in_specs.append(_whole_2d(1, d))
    args.append(eff_scale_2d)

  out_shapes: list[Any] = [
      _out_struct((total_tokens, d), prefix_sum.dtype, prefix_sum),
      _out_struct((m, total_tokens, d), blocks.dtype, prefix_sum),
      _out_struct((_WGRAD_ROWS, d), jnp.float32, prefix_sum),
  ]
  out_specs: list[Any] = [_HBM_SPEC, _HBM_SPEC, _whole_2d(_WGRAD_ROWS, d)]
  if apply_sublayer_norm:
    out_shapes.append(_out_struct((_WGRAD_ROWS, d), jnp.float32, prefix_sum))
    out_specs.append(_whole_2d(_WGRAD_ROWS, d))
  if has_pending_arg:
    assert pending is not None
    out_shapes.append(_out_struct((total_tokens, d), pending.dtype, prefix_sum))
    out_specs.append(_HBM_SPEC)

  scratch_shapes: list[Any] = [
      pltpu.VMEM((2, block_t, d), grad_out.dtype),
      pltpu.VMEM((2, block_t, d), prefix_sum.dtype),
      pltpu.VMEM((2, m, block_t, d), blocks.dtype),
      pltpu.VMEM((2, block_t, d), prefix_sum.dtype),
      pltpu.VMEM((2, m, block_t, d), blocks.dtype),
      pltpu.VMEM((3, _STAT_STRIDE, strip_t, 128), jnp.float32),
      pltpu.VMEM((3, _STAT_STRIDE, strip_t, 128), jnp.float32),
      pltpu.VMEM((3, strip_t, 128), jnp.float32),
      pltpu.VMEM((2, block_t, _STAT_WIDTH), jnp.float32),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
  ]
  if has_pending_arg:
    assert pending is not None
    scratch_shapes.extend(
        [
            pltpu.VMEM((2, block_t, d), blocks.dtype),
            pltpu.VMEM((2, block_t, d), pending.dtype),
        ]
    )
  if apply_sublayer_norm:
    scratch_shapes.append(pltpu.VMEM((strip_t, d), h_dtype))

  flops = total_tokens * d * (6 * (m + 1) + (6 if apply_sublayer_norm else 0))
  bytes_accessed = (
      total_tokens * d * (grad_out.dtype.itemsize + 2 * prefix_sum.dtype.itemsize + 2 * m * blocks.dtype.itemsize)
  )

  outs = pl.pallas_call(
      functools.partial(
          _bwd_kernel,
          m=m,
          d=d,
          block_t=block_t,
          strip_t=strip_t,
          group_d=group_d,
          sub_d=sub_d,
          num_tiles=num_tiles,
          h_dtype=h_dtype,
          y_dtype=y_dtype,
          out_dtype=out_dtype,
          apply_sublayer_norm=apply_sublayer_norm,
          has_pending_arg=has_pending_arg,
          interpret=config.interpret,
      ),
      out_shape=tuple(out_shapes),
      grid=(1,),
      in_specs=in_specs,
      out_specs=tuple(out_specs),
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
          vmem_limit_bytes=vmem_limit,
      ),
      cost_estimate=pl.CostEstimate(
          flops=flops,
          transcendentals=0,
          bytes_accessed=bytes_accessed,
      ),
      interpret=config.interpret,
      name="attn_res_rms_norm_bwd" if apply_sublayer_norm else "attn_res_pool_bwd",
  )(*args)

  grad_p = outs[0].reshape(batch, seq_len, d)
  grad_b = jnp.swapaxes(outs[1], 0, 1).reshape(batch, seq_len, m, d)
  dsw_1d = jnp.sum(outs[2], axis=0)
  grad_res_ns = (dsw_1d * pr_f32).astype(res_norm_scale.dtype).reshape(res_norm_scale.shape)
  grad_res_pr = (dsw_1d * ns_f32).astype(res_proj.dtype).reshape(res_proj.shape)

  out_idx = 3
  result: list[jax.Array] = [grad_p, grad_b, grad_res_ns, grad_res_pr]
  if apply_sublayer_norm:
    assert sublayer_norm_scale is not None
    grad_sub_scale = jnp.sum(outs[out_idx], axis=0).astype(sublayer_norm_scale.dtype).reshape(sublayer_norm_scale.shape)
    result.append(grad_sub_scale)
    out_idx += 1
  if has_pending_arg:
    result.append(outs[out_idx].reshape(batch, seq_len, d))
  return tuple(result)


def _replicated_spec(x: jax.Array) -> jax.sharding.PartitionSpec:
  return jax.sharding.PartitionSpec(*((None,) * x.ndim))


def _ensure_replicated_explicit(
    x: jax.Array,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> jax.Array:
  """All-gathers a sharded 1D/2D AttnRes weight to replicated layout in Explicit mode."""
  sharding = getattr(jax.typeof(x), "sharding", None)
  rep_spec = _replicated_spec(x)
  if isinstance(sharding, jax.sharding.NamedSharding) and sharding.spec == rep_spec:
    return x
  return jax.sharding.reshard(x, jax.sharding.NamedSharding(mesh, rep_spec))


def _weight_shard_out_spec(
    ct_spec: jax.sharding.PartitionSpec,
    ndim: int,
) -> jax.sharding.PartitionSpec:
  """Returns the `shard_map` output spec for a full-shape local weight gradient."""
  if any(p is not None for p in ct_spec.partitions):
    if ct_spec.unreduced:
      return jax.sharding.PartitionSpec(*((None,) * ndim), unreduced=ct_spec.unreduced)
    return jax.sharding.PartitionSpec(*((None,) * ndim))
  return ct_spec


def _restore_weight_grad_explicit(
    grad: jax.Array,
    ct_spec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> jax.Array:
  """Reshards a full-shape weight gradient back to its Explicit cotangent spec."""
  if any(p is not None for p in ct_spec.partitions):
    return jax.sharding.reshard(grad, jax.sharding.NamedSharding(mesh, ct_spec))
  return grad


def _cotangent_spec(x: jax.Array) -> jax.sharding.PartitionSpec:
  sharding = jax.typeof(x).sharding
  assert isinstance(sharding, jax.sharding.NamedSharding)
  return sharding.spec.to_ct_spec()


def _spec_axes(
    spec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> tuple[str, ...]:
  """Returns unique non-trivial mesh axes referenced by `spec` partitions and `unreduced`."""
  axes: list[str] = []
  for part in spec.partitions:
    if part is None:
      continue
    raw = (part,) if isinstance(part, str) else tuple(part)
    for ax in raw:
      if ax is not None and mesh.shape.get(ax, 1) > 1 and ax not in axes:
        axes.append(ax)
  for ax in sorted(spec.unreduced):
    if mesh.shape.get(ax, 1) > 1 and ax not in axes:
      axes.append(ax)
  return tuple(axes)


def _reduce_for_spec(
    x: jax.Array,
    spec: jax.sharding.PartitionSpec,
    varying_axes: tuple[str, ...],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> jax.Array:
  """Reduces local partial `x` across `varying_axes` not retained in `spec`."""
  kept = set(spec.unreduced) | set(_spec_axes(spec, mesh))
  missing = tuple(ax for ax in varying_axes if ax not in kept)
  if missing:
    return jax.lax.psum(x.astype(jnp.float32), missing).astype(x.dtype)
  return x


def _auto_token_axes(
    mesh: jax.sharding.Mesh | None,
    activation_pspec: jax.sharding.PartitionSpec | None,
) -> tuple[Any, Any, tuple[str, ...]]:
  """Extracts `(b_axis, t_axis, active_psum_axes)` for `ShardMode.AUTO`."""
  if mesh is None or mesh.empty or activation_pspec is None:
    return None, None, ()
  b_ax = activation_pspec[0] if len(activation_pspec) >= 1 else None
  t_ax = activation_pspec[1] if len(activation_pspec) >= 2 else None
  active: list[str] = []
  for entry in (b_ax, t_ax):
    if entry is None:
      continue
    raw = (entry,) if isinstance(entry, str) else tuple(entry)
    for ax in raw:
      if ax is not None and mesh.shape.get(ax, 1) > 1 and ax not in active:
        active.append(ax)
  return b_ax, t_ax, tuple(active)


def _dispatch_fwd(
    prefix_sum: jax.Array,
    blocks: jax.Array,
    num_committed: int | jax.Array,
    res_norm_scale: jax.Array,
    res_proj: jax.Array,
    sublayer_norm_scale: jax.Array | None,
    *,
    pending: jax.Array | None = None,
    has_pending: bool | jax.Array | None = None,
    res_epsilon: float = 1e-6,
    norm_epsilon: float = 1e-6,
    norm_dtype: Any = None,
    return_stats: bool = False,
    mesh: jax.sharding.Mesh | None = None,
    activation_pspec: jax.sharding.PartitionSpec | None = None,
    config: AttnResKernelConfig = AttnResKernelConfig(),
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Dispatches `attn_res_fwd_local` across Explicit, Auto, or single-device modes."""
  n_arr = jnp.asarray(num_committed, dtype=jnp.int32)
  hp_arr = jnp.asarray(has_pending, dtype=jnp.bool_) if has_pending is not None else None
  common_kw = {
      "res_epsilon": res_epsilon,
      "norm_epsilon": norm_epsilon,
      "norm_dtype": norm_dtype,
      "return_stats": return_stats,
      "config": config,
  }

  if _has_explicit_mesh(prefix_sum):
    exp_mesh = jax.typeof(prefix_sum).sharding.mesh
    out_spec = jax.typeof(prefix_sum).sharding.spec
    varying_axes = _spec_axes(out_spec, exp_mesh)
    out_specs = (out_spec, out_spec) if return_stats else out_spec
    ns_rep = _ensure_replicated_explicit(res_norm_scale, exp_mesh)
    pr_rep = _ensure_replicated_explicit(res_proj, exp_mesh)
    sub_rep = _ensure_replicated_explicit(sublayer_norm_scale, exp_mesh) if sublayer_norm_scale is not None else None

    def _run_exp(p, b, n, ns, pr, *extra):
      idx = 0
      sub_s = extra[idx] if sub_rep is not None else None
      idx += 1 if sub_rep is not None else 0
      pend_in = extra[idx] if pending is not None else None
      hp_in = extra[idx + 1] if pending is not None else None
      res = attn_res_fwd_local(p, b, n, ns, pr, sub_s, pending=pend_in, has_pending=hp_in, **common_kw)
      if isinstance(res, tuple):
        return (
            _reduce_for_spec(res[0], out_spec, varying_axes, exp_mesh),
            _reduce_for_spec(res[1], out_spec, varying_axes, exp_mesh),
        )
      return _reduce_for_spec(res, out_spec, varying_axes, exp_mesh)

    extra_args: list[Any] = []
    if sub_rep is not None:
      extra_args.append(sub_rep)
    if pending is not None:
      extra_args.extend([pending, hp_arr])
    return jax.shard_map(_run_exp, mesh=exp_mesh, out_specs=out_specs, check_vma=False)(
        prefix_sum, blocks, n_arr, ns_rep, pr_rep, *extra_args
    )

  b_ax, t_ax, _ = _auto_token_axes(mesh, activation_pspec)
  if mesh is not None and not mesh.empty and mesh.size > 1 and activation_pspec is not None:
    P = jax.sharding.PartitionSpec
    tok_pspec = P(b_ax, t_ax, None)
    blk_pspec = P(b_ax, t_ax, None, None)
    w1_pspec = P(*((None,) * res_norm_scale.ndim))
    w2_pspec = P(*((None,) * res_proj.ndim))
    in_specs_list: list[Any] = [tok_pspec, blk_pspec, P(), w1_pspec, w2_pspec]
    call_args: list[Any] = [prefix_sum, blocks, n_arr, res_norm_scale, res_proj]
    if sublayer_norm_scale is not None:
      in_specs_list.append(P(*((None,) * sublayer_norm_scale.ndim)))
      call_args.append(sublayer_norm_scale)
    if pending is not None:
      in_specs_list.extend([tok_pspec, P()])
      call_args.extend([pending, hp_arr])

    out_specs_auto = (tok_pspec, tok_pspec) if return_stats else tok_pspec

    def _run_auto(p, b, n, ns, pr, *extra):
      idx = 0
      sub_s = extra[idx] if sublayer_norm_scale is not None else None
      idx += 1 if sublayer_norm_scale is not None else 0
      pend_in = extra[idx] if pending is not None else None
      hp_in = extra[idx + 1] if pending is not None else None
      return attn_res_fwd_local(p, b, n, ns, pr, sub_s, pending=pend_in, has_pending=hp_in, **common_kw)

    return jax.shard_map(
        _run_auto,
        mesh=mesh,
        in_specs=tuple(in_specs_list),
        out_specs=out_specs_auto,
        check_vma=False,
    )(*call_args)

  return attn_res_fwd_local(
      prefix_sum,
      blocks,
      n_arr,
      res_norm_scale,
      res_proj,
      sublayer_norm_scale,
      pending=pending,
      has_pending=hp_arr,
      **common_kw,
  )


def _dispatch_bwd(
    grad_out: jax.Array,
    prefix_sum: jax.Array,
    blocks: jax.Array,
    num_committed: int | jax.Array,
    res_norm_scale: jax.Array,
    res_proj: jax.Array,
    fwd_stats: jax.Array,
    sublayer_norm_scale: jax.Array | None,
    *,
    pending: jax.Array | None = None,
    has_pending: bool | jax.Array | None = None,
    norm_dtype: Any = None,
    mesh: jax.sharding.Mesh | None = None,
    activation_pspec: jax.sharding.PartitionSpec | None = None,
    config: AttnResKernelConfig = AttnResKernelConfig(),
) -> tuple[jax.Array, ...]:
  """Dispatches `attn_res_bwd_local` across Explicit, Auto, or single-device modes."""
  n_arr = jnp.asarray(num_committed, dtype=jnp.int32)
  hp_arr = jnp.asarray(has_pending, dtype=jnp.bool_) if has_pending is not None else None
  common_kw = {"norm_dtype": norm_dtype, "config": config}

  if _has_explicit_mesh(prefix_sum):
    prefix_sharding = jax.typeof(prefix_sum).sharding
    assert isinstance(prefix_sharding, jax.sharding.NamedSharding)
    exp_mesh = prefix_sharding.mesh
    prefix_ct_spec = _cotangent_spec(prefix_sum)
    grad_out_sharding = jax.typeof(grad_out).sharding
    if isinstance(grad_out_sharding, jax.sharding.NamedSharding):
      # Reduce any unreduced cotangent axes (e.g. from downstream contractions over replicated
      # axes) that are not preserved on `prefix_ct_spec`, and align spatial partitions with
      # `prefix_sum` so `shard_map` slices `grad_out`, `prefix_sum`, and `blocks` identically.
      target_unreduced = grad_out_sharding.spec.unreduced & prefix_ct_spec.unreduced
      target_spec = jax.sharding.PartitionSpec(
          *prefix_sharding.spec.partitions,
          unreduced=target_unreduced,
          reduced=prefix_ct_spec.reduced,
      )
      target_sharding = jax.sharding.NamedSharding(exp_mesh, target_spec)
      if grad_out_sharding != target_sharding:
        grad_out = jax.sharding.reshard(grad_out, target_sharding)

    grad_out_unreduced = (
        jax.typeof(grad_out).sharding.spec.unreduced
        if isinstance(jax.typeof(grad_out).sharding, jax.sharding.NamedSharding)
        else frozenset()
    )
    varying_axes = _spec_axes(
        jax.sharding.PartitionSpec(*prefix_sharding.spec.partitions, unreduced=grad_out_unreduced),
        exp_mesh,
    )
    ns_ct_spec = _cotangent_spec(res_norm_scale)
    pr_ct_spec = _cotangent_spec(res_proj)
    sub_ct_spec = _cotangent_spec(sublayer_norm_scale) if sublayer_norm_scale is not None else None
    ns_rep = _ensure_replicated_explicit(res_norm_scale, exp_mesh)
    pr_rep = _ensure_replicated_explicit(res_proj, exp_mesh)
    sub_rep = _ensure_replicated_explicit(sublayer_norm_scale, exp_mesh) if sublayer_norm_scale is not None else None
    out_specs_list = [
        prefix_ct_spec,
        _cotangent_spec(blocks),
        _weight_shard_out_spec(ns_ct_spec, res_norm_scale.ndim),
        _weight_shard_out_spec(pr_ct_spec, res_proj.ndim),
    ]
    if sublayer_norm_scale is not None:
      assert sub_ct_spec is not None
      out_specs_list.append(_weight_shard_out_spec(sub_ct_spec, sublayer_norm_scale.ndim))
    if pending is not None:
      out_specs_list.append(_cotangent_spec(pending))
    out_specs_tuple = tuple(out_specs_list)

    def _run_exp(dy, p, b, n, ns, pr, st_in, *extra):
      idx = 0
      sub_s = extra[idx] if sub_rep is not None else None
      idx += 1 if sub_rep is not None else 0
      pend_in = extra[idx] if pending is not None else None
      hp_in = extra[idx + 1] if pending is not None else None
      outs = attn_res_bwd_local(dy, p, b, n, ns, pr, st_in, sub_s, pending=pend_in, has_pending=hp_in, **common_kw)
      return tuple(_reduce_for_spec(o, sp, varying_axes, exp_mesh) for o, sp in zip(outs, out_specs_tuple))

    extra_args: list[Any] = []
    if sub_rep is not None:
      extra_args.append(sub_rep)
    if pending is not None:
      extra_args.extend([pending, hp_arr])
    raw_outs = jax.shard_map(_run_exp, mesh=exp_mesh, out_specs=out_specs_tuple, check_vma=False)(
        grad_out, prefix_sum, blocks, n_arr, ns_rep, pr_rep, fwd_stats, *extra_args
    )
    res_list = list(raw_outs)
    res_list[2] = _restore_weight_grad_explicit(res_list[2], ns_ct_spec, exp_mesh)
    res_list[3] = _restore_weight_grad_explicit(res_list[3], pr_ct_spec, exp_mesh)
    if sublayer_norm_scale is not None:
      assert sub_ct_spec is not None
      res_list[4] = _restore_weight_grad_explicit(res_list[4], sub_ct_spec, exp_mesh)
    return tuple(res_list)

  b_ax, t_ax, active_axes = _auto_token_axes(mesh, activation_pspec)
  if mesh is not None and not mesh.empty and mesh.size > 1 and activation_pspec is not None:
    P = jax.sharding.PartitionSpec
    tok_pspec = P(b_ax, t_ax, None)
    blk_pspec = P(b_ax, t_ax, None, None)
    w1_pspec = P(*((None,) * res_norm_scale.ndim))
    w2_pspec = P(*((None,) * res_proj.ndim))

    in_specs_list: list[Any] = [tok_pspec, tok_pspec, blk_pspec, P(), w1_pspec, w2_pspec, tok_pspec]
    call_args: list[Any] = [grad_out, prefix_sum, blocks, n_arr, res_norm_scale, res_proj, fwd_stats]
    if sublayer_norm_scale is not None:
      in_specs_list.append(P(*((None,) * sublayer_norm_scale.ndim)))
      call_args.append(sublayer_norm_scale)
    if pending is not None:
      in_specs_list.extend([tok_pspec, P()])
      call_args.extend([pending, hp_arr])

    out_specs_auto_list: list[Any] = [tok_pspec, blk_pspec, w1_pspec, w2_pspec]
    if sublayer_norm_scale is not None:
      out_specs_auto_list.append(P(*((None,) * sublayer_norm_scale.ndim)))
    if pending is not None:
      out_specs_auto_list.append(tok_pspec)

    def _run_auto(dy, p, b, n, ns, pr, st_in, *extra):
      idx = 0
      sub_s = extra[idx] if sublayer_norm_scale is not None else None
      idx += 1 if sublayer_norm_scale is not None else 0
      pend_in = extra[idx] if pending is not None else None
      hp_in = extra[idx + 1] if pending is not None else None
      outs = list(attn_res_bwd_local(dy, p, b, n, ns, pr, st_in, sub_s, pending=pend_in, has_pending=hp_in, **common_kw))
      if active_axes:
        outs[2] = jax.lax.psum(outs[2].astype(jnp.float32), active_axes).astype(res_norm_scale.dtype)
        outs[3] = jax.lax.psum(outs[3].astype(jnp.float32), active_axes).astype(res_proj.dtype)
        if sublayer_norm_scale is not None:
          outs[4] = jax.lax.psum(outs[4].astype(jnp.float32), active_axes).astype(sublayer_norm_scale.dtype)
      return tuple(outs)

    return jax.shard_map(
        _run_auto,
        mesh=mesh,
        in_specs=tuple(in_specs_list),
        out_specs=tuple(out_specs_auto_list),
        check_vma=False,
    )(*call_args)

  return attn_res_bwd_local(
      grad_out,
      prefix_sum,
      blocks,
      n_arr,
      res_norm_scale,
      res_proj,
      fwd_stats,
      sublayer_norm_scale,
      pending=pending,
      has_pending=hp_arr,
      **common_kw,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13))
def _attn_res_vjp(
    prefix_sum: jax.Array,
    blocks: jax.Array,
    num_committed: jax.Array,
    res_norm_scale: jax.Array,
    res_proj: jax.Array,
    sublayer_norm_scale: jax.Array | None,
    pending: jax.Array | None,
    has_pending: jax.Array | None,
    res_epsilon: float,
    norm_epsilon: float,
    norm_dtype: Any,
    mesh: jax.sharding.Mesh | None,
    activation_pspec: jax.sharding.PartitionSpec | None,
    config: AttnResKernelConfig,
) -> jax.Array:
  """Custom-VJP entry point for sharded or local Pallas Block Attention Residuals."""
  out = _dispatch_fwd(
      prefix_sum,
      blocks,
      num_committed,
      res_norm_scale,
      res_proj,
      sublayer_norm_scale,
      pending=pending,
      has_pending=has_pending,
      res_epsilon=res_epsilon,
      norm_epsilon=norm_epsilon,
      norm_dtype=norm_dtype,
      return_stats=False,
      mesh=mesh,
      activation_pspec=activation_pspec,
      config=config,
  )
  assert isinstance(out, jax.Array)
  return out


def _attn_res_vjp_fwd(
    prefix_sum,
    blocks,
    num_committed,
    res_norm_scale,
    res_proj,
    sublayer_norm_scale,
    pending,
    has_pending,
    res_epsilon,
    norm_epsilon,
    norm_dtype,
    mesh,
    activation_pspec,
    config,
):
  """Forward rule for `_attn_res_vjp`, returning `(out, residuals)`."""
  out, fwd_stats = _dispatch_fwd(
      prefix_sum,
      blocks,
      num_committed,
      res_norm_scale,
      res_proj,
      sublayer_norm_scale,
      pending=pending,
      has_pending=has_pending,
      res_epsilon=res_epsilon,
      norm_epsilon=norm_epsilon,
      norm_dtype=norm_dtype,
      return_stats=True,
      mesh=mesh,
      activation_pspec=activation_pspec,
      config=config,
  )
  return (
      out,
      (prefix_sum, blocks, num_committed, res_norm_scale, res_proj, sublayer_norm_scale, pending, has_pending, fwd_stats),
  )


def _attn_res_vjp_bwd(res_epsilon, norm_epsilon, norm_dtype, mesh, activation_pspec, config, res, grad_out):
  """Backward rule for `_attn_res_vjp`, returning cotangents for differentiable inputs."""
  del res_epsilon, norm_epsilon
  prefix_sum, blocks, num_committed, res_norm_scale, res_proj, sublayer_norm_scale, pending, has_pending, fwd_stats = res
  outs = _dispatch_bwd(
      grad_out,
      prefix_sum,
      blocks,
      num_committed,
      res_norm_scale,
      res_proj,
      fwd_stats,
      sublayer_norm_scale,
      pending=pending,
      has_pending=has_pending,
      norm_dtype=norm_dtype,
      mesh=mesh,
      activation_pspec=activation_pspec,
      config=config,
  )
  gp, gb, g_ns, g_pr = outs[0], outs[1], outs[2], outs[3]
  idx = 4
  if sublayer_norm_scale is not None:
    g_sub = outs[idx]
    idx += 1
  else:
    g_sub = None
  g_pend = outs[idx] if pending is not None else None
  return gp, gb, None, g_ns, g_pr, g_sub, g_pend, None


_attn_res_vjp.defvjp(_attn_res_vjp_fwd, _attn_res_vjp_bwd)


def attn_res_pallas(
    prefix_sum: jax.Array,
    blocks: jax.Array,
    num_committed: int | jax.Array,
    res_norm_scale: jax.Array,
    res_proj: jax.Array,
    *,
    pending: jax.Array | None = None,
    has_pending: bool | jax.Array | None = None,
    epsilon: float = 1e-6,
    mesh: jax.sharding.Mesh | None = None,
    activation_pspec: jax.sharding.PartitionSpec | None = None,
    config: AttnResKernelConfig = AttnResKernelConfig(),
) -> jax.Array:
  """Differentiable Pallas Block Attention Residuals pooling."""
  if (pending is None) != (has_pending is None):
    raise ValueError("pending and has_pending must both be provided or both be None.")
  return _attn_res_vjp(
      prefix_sum,
      blocks,
      jnp.asarray(num_committed, dtype=jnp.int32),
      res_norm_scale,
      res_proj,
      None,
      pending,
      jnp.asarray(has_pending, dtype=jnp.bool_) if has_pending is not None else None,
      float(epsilon),
      1e-6,
      None,
      mesh,
      activation_pspec,
      config,
  )


def attn_res_rms_norm_pallas(
    prefix_sum: jax.Array,
    blocks: jax.Array,
    num_committed: int | jax.Array,
    res_norm_scale: jax.Array,
    res_proj: jax.Array,
    sublayer_norm_scale: jax.Array,
    *,
    pending: jax.Array | None = None,
    has_pending: bool | jax.Array | None = None,
    res_epsilon: float = 1e-6,
    norm_epsilon: float = 1e-6,
    norm_dtype: Any = None,
    mesh: jax.sharding.Mesh | None = None,
    activation_pspec: jax.sharding.PartitionSpec | None = None,
    config: AttnResKernelConfig = AttnResKernelConfig(),
) -> jax.Array:
  """Differentiable fused Pallas Block Attention Residuals + sublayer RMSNorm."""
  if (pending is None) != (has_pending is None):
    raise ValueError("pending and has_pending must both be provided or both be None.")
  resolved_norm_dtype = jnp.dtype(norm_dtype) if norm_dtype is not None else jnp.dtype(jnp.float32)
  return _attn_res_vjp(
      prefix_sum,
      blocks,
      jnp.asarray(num_committed, dtype=jnp.int32),
      res_norm_scale,
      res_proj,
      sublayer_norm_scale,
      pending,
      jnp.asarray(has_pending, dtype=jnp.bool_) if has_pending is not None else None,
      float(res_epsilon),
      float(norm_epsilon),
      resolved_norm_dtype,
      mesh,
      activation_pspec,
      config,
  )
