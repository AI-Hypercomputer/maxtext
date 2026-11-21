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

"""Standalone single-TensorCore Pallas `gmm_v2` and `tgmm_v2` for Flint MoE."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_DEFAULT_VMEM_LIMIT_BYTES: int = 48 * 1024 * 1024
_TARGET_ZERO_REF_BYTES: int = 2 * 1024 * 1024


def align_to(x: int, alignment: int) -> int:
  """Rounds `x` up to the nearest multiple of `alignment`."""
  return pl.cdiv(x, alignment) * alignment


def _ensure_tpu_backend() -> Any:
  """Verifies live TPU backend or offline TPU AOT mesh, else fails fast."""
  try:
    return pltpu.get_tpu_info()
  except (ValueError, RuntimeError) as exc:
    raise ValueError(
        "pallas_ragged_dot requires TPU backend or TPU AOT mesh, got" f" default_backend={jax.default_backend()!r}."
    ) from exc


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  """SMEM tile-to-group and tile-to-row-offset tables."""

  gm_id_to_group_id: Any
  gm_id_to_m_offset: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef:
  """RHS weight and optional per-group FP32 scale refs."""

  weight: Any
  scale: Any | None = None


@dataclasses.dataclass(frozen=True)
class Dimensions:
  """Problem geometry for `gmm_v2` and `tgmm_v2`."""

  size_m: int
  size_k: int
  size_n: int
  size_group: int
  size_lhs_group: int
  size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class TileSizes:
  """Selected VMEM tile sizes and dynamic M-bucketing parameters."""

  tile_m: int
  tile_k: int
  tile_n: int
  bucket_base: int
  lhs_buffer_count: int = 2


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
  """Bundled static configuration passed into Pallas kernels."""

  dims: Dimensions
  tiles: TileSizes
  lhs_dtype: jnp.dtype
  rhs_dtype: jnp.dtype
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  zero_init: bool = True
  transpose_rhs: bool = False
  has_rhs_scale: bool = False
  rhs_scale_on_k: bool = False
  lhs_qtype: jnp.dtype | None = None


def fill_metadata(
    lhs_group_sizes_ref: Any,
    group_offset_ref: Any,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
  """Populates SMEM `gm_id_to_group_id` and `gm_id_to_m_offset` tables."""
  group_offset = group_offset_ref[0]
  max_num_group = group_offset + cfgs.dims.size_group
  metadata_ref.gm_id_to_m_offset[0] = 0

  @jax.named_scope("inner_tm_loop")
  def inner_tm_loop(
      tm_id: jax.Array,
      curr_m_offset: jax.Array,
      *,
      end_m_offset: jax.Array,
      group_id: jax.Array,
  ) -> jax.Array:
    local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
    tm_size = jnp.minimum(cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset)
    metadata_ref.gm_id_to_group_id[tm_id] = group_id
    next_m_offset = curr_m_offset + tm_size
    metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
    metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset
    return next_m_offset

  @jax.named_scope("outer_group_loop")
  def outer_group_loop(
      lhs_group_id: jax.Array,
      carry: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    num_gm, start_m_offset = carry
    group_id = lhs_group_id - group_offset
    group_size = lhs_group_sizes_ref[lhs_group_id]
    end_m_offset = start_m_offset + group_size

    local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
    aligned_group_size = group_size + local_offset
    curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

    should_process = jnp.logical_and(group_size > 0, group_id >= 0)
    curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
    next_num_gm = num_gm + curr_num_gm

    tm_loop_fn = functools.partial(
        inner_tm_loop,
        end_m_offset=end_m_offset,
        group_id=group_id,
    )
    lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)
    return next_num_gm, end_m_offset

  num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (jnp.int32(0), jnp.int32(0)))
  return num_gm


def _validate_tile_m_override(name: str, val: int | None) -> None:
  if val is None:
    return
  if isinstance(val, bool) or val < 128 or val % 128 != 0:
    raise ValueError(f"{name} must be an int >= 128 and divisible by 128, got {val!r}.")


def calculate_gmm_tiling(
    dims: Dimensions,
    *,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    vmem_limit_bytes: int,
    transpose_rhs: bool,
    zero_initialize: bool,
    tile_m_override: int | None = None,
    has_rhs_scale: bool = False,
    rhs_scale_on_k: bool = False,
) -> TileSizes:
  """Selects single-core `gmm_v2` tile sizes within `vmem_limit_bytes`."""
  _validate_tile_m_override("gmm_tile_m", tile_m_override)
  tpu_info = _ensure_tpu_backend()
  num_lanes = tpu_info.num_lanes
  mxu_size = tpu_info.mxu_column_size
  gen = tpu_info.generation if isinstance(tpu_info.generation, int) else 0

  default_base_m = min(
      256 if (gen >= 8 or transpose_rhs) else 128,
      dims.size_m,
  )
  default_base_m = max(default_base_m, dims.size_lhs_sublane)

  if tile_m_override is not None:
    tile_m = max(min(tile_m_override, dims.size_m), dims.size_lhs_sublane)
  else:
    tile_m = default_base_m

  lhs_bytes = jnp.dtype(lhs_dtype).itemsize
  rhs_bytes = jnp.dtype(rhs_dtype).itemsize
  out_bytes = jnp.dtype(out_dtype).itemsize
  acc_bytes = jnp.dtype(acc_dtype).itemsize
  min_k_align = num_lanes

  zero_reserve = min(_TARGET_ZERO_REF_BYTES, dims.size_m * num_lanes * out_bytes) if zero_initialize else 0

  def _vmem_estimate(tm: int, tn: int, tk: int, lhs_buffer_count: int = 2) -> int:
    lhs_vmem = lhs_buffer_count * tm * tk * lhs_bytes
    rhs_vmem = 3 * tk * tn * rhs_bytes
    rhs_scale_vmem = 2 * (tk if rhs_scale_on_k else tn) * 4 if has_rhs_scale else 0
    acc_vmem = tm * tn * acc_bytes
    out_vmem = 2 * tm * tn * out_bytes
    return lhs_vmem + rhs_vmem + rhs_scale_vmem + acc_vmem + out_vmem + zero_reserve

  tile_n_limit = min(mxu_size * 2, dims.size_n)
  num_k_tiles = 1
  num_n_tiles = 1
  tile_k = align_to(dims.size_k, min_k_align)
  tile_n = align_to(dims.size_n, num_lanes)

  prev_tile_n = tile_n
  while _vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes and tile_n > tile_n_limit:
    num_n_tiles += 1
    tile_n = align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles
    if tile_n < tile_n_limit or tile_n >= prev_tile_n:
      break
    prev_tile_n = tile_n

  if tile_n < tile_n_limit and num_n_tiles > 1:
    num_n_tiles -= 1
    tile_n = align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles

  prev_tile_k = tile_k
  while _vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes and tile_k > min_k_align:
    num_k_tiles += 1
    tile_k = align_to(dims.size_k, num_k_tiles * min_k_align) // num_k_tiles
    if tile_k < min_k_align or tile_k >= prev_tile_k:
      break
    prev_tile_k = tile_k

  if tile_k < min_k_align and num_k_tiles > 1:
    num_k_tiles -= 1
    tile_k = align_to(dims.size_k, num_k_tiles * min_k_align) // num_k_tiles

  while _vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes and tile_m > num_lanes:
    tile_m = max(num_lanes, tile_m - num_lanes)

  if _vmem_estimate(tile_m, tile_n, tile_k) > vmem_limit_bytes:
    raise ValueError(
        f"Could not fit gmm_v2 tiles ({tile_m}, {tile_k}, {tile_n}) for"
        f" {dims=} in vmem_limit_bytes={vmem_limit_bytes}."
    )

  if tile_m_override is None:
    max_num_buckets = 8 if gen >= 8 else 4
    bucket_base = tile_m
    for _ in range(1, max_num_buckets):
      new_tile_m = tile_m + bucket_base
      if new_tile_m > dims.size_m:
        break
      if _vmem_estimate(new_tile_m, tile_n, tile_k) > vmem_limit_bytes:
        break
      tile_m = new_tile_m
  else:
    if tile_m >= default_base_m and tile_m % default_base_m == 0:
      bucket_base = default_base_m
    elif tile_m >= num_lanes and tile_m % num_lanes == 0:
      bucket_base = num_lanes
    else:
      bucket_base = tile_m

  max_lhs_buffers = 4 if gen >= 8 else 3
  lhs_buffer_count = 2
  for count in range(max_lhs_buffers, 2, -1):
    if _vmem_estimate(tile_m, tile_n, tile_k, lhs_buffer_count=count) <= vmem_limit_bytes:
      lhs_buffer_count = count
      break

  return TileSizes(
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
      bucket_base=bucket_base,
      lhs_buffer_count=lhs_buffer_count,
  )


class _GmmIndexMaps:
  """BlockSpec index maps for `gmm_v2`."""

  def __init__(
      self,
      metadata_ref: MetadataRef,
      cfgs: GmmConfigs,
      *,
      num_gm: jax.Array,
  ) -> None:
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs
    self.num_gm = num_gm

  def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array) -> tuple[Any, int, jax.Array]:
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    return (pl.ds(row_start, row_end - row_start), 0, k_id)

  def rhs_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    if self.cfgs.transpose_rhs:
      return (group_id, n_id, k_id)
    return (group_id, k_id, n_id)

  def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array) -> tuple[jax.Array, int, jax.Array]:
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, 0, k_id if self.cfgs.rhs_scale_on_k else n_id)

  def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array) -> tuple[Any, int, jax.Array]:
    is_last_gm = gm_id == (self.num_gm - 1)
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    return (pl.ds(row_start, row_end - row_start), 0, n_id)


def _generate_gmm_block_specs(
    metadata_ref: MetadataRef,
    cfgs: GmmConfigs,
    *,
    num_gm: jax.Array,
) -> tuple[tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
  """Builds `(lhs_spec, WeightsRef(rhs_spec, rhs_scale_spec)), out_spec`."""
  imap = _GmmIndexMaps(metadata_ref, cfgs, num_gm=num_gm)
  bounded_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)
  lhs_spec = pl.BlockSpec(
      (bounded_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
      imap.lhs_index_map,
      pipeline_mode=pl.Buffered(buffer_count=cfgs.tiles.lhs_buffer_count),
  )
  rhs_block = (
      (None, cfgs.tiles.tile_n, cfgs.tiles.tile_k) if cfgs.transpose_rhs else (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n)
  )
  rhs_spec = pl.BlockSpec(
      rhs_block,
      imap.rhs_index_map,
      pipeline_mode=pl.Buffered(buffer_count=3),
  )
  rhs_scale_spec = None
  if cfgs.has_rhs_scale:
    scale_cols = cfgs.tiles.tile_k if cfgs.rhs_scale_on_k else cfgs.tiles.tile_n
    rhs_scale_spec = pl.BlockSpec(
        (None, 1, scale_cols),
        imap.rhs_scale_index_map,
    )
  out_spec = pl.BlockSpec(
      (bounded_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
      imap.out_index_map,
  )
  return (lhs_spec, WeightsRef(weight=rhs_spec, scale=rhs_scale_spec)), out_spec


def _gmm_inner_kernel(
    step: pltpu.PipelineStep,
    tiled_lhs_ref: Any,
    tiled_rhs_ref: WeightsRef,
    tiled_out_ref: Any,
    partial_out_ref: Any,
    acc_ref: Any,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> None:
  """Per-tile MXU pipeline body for `gmm_v2`."""
  gm_id = step.index[1]
  m_start = metadata_ref.gm_id_to_m_offset[gm_id]
  m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
  m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
  m_start_local = m_start - m_offset
  m_end_local = m_end - m_offset

  def _matmul(
      is_first_k_step: bool,
      is_last_k_step: bool,
      bucket_m: int,
  ) -> None:
    mxu_size = pltpu.get_tpu_info().mxu_column_size
    tiled_lhs = tiled_lhs_ref[...].reshape(-1, cfgs.tiles.tile_k)[:bucket_m]
    tiled_rhs = tiled_rhs_ref.weight[...]
    rhs_tile_n = tiled_rhs.shape[0] if cfgs.transpose_rhs else tiled_rhs.shape[1]
    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    has_k_tail = is_last_k_step and valid_k != 0
    if has_k_tail:
      mask_lhs = lax.broadcasted_iota(jnp.int32, tiled_lhs.shape, 1) < valid_k
      tiled_lhs = jnp.where(mask_lhs, tiled_lhs, 0)
      if tiled_rhs.dtype.itemsize >= 2:
        rhs_k_axis = 1 if cfgs.transpose_rhs else 0
        mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, rhs_k_axis) < valid_k
        tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

    if cfgs.lhs_qtype is not None:
      q_dt = jnp.dtype(cfgs.lhs_qtype)
      max_val = jnp.float32(float(jnp.finfo(q_dt).max))
      if cfgs.has_rhs_scale and cfgs.rhs_scale_on_k:
        assert tiled_rhs_ref.scale is not None
        lhs_f32 = tiled_lhs.astype(jnp.float32) * tiled_rhs_ref.scale[...].astype(jnp.float32)
        if has_k_tail:
          mask_lhs = lax.broadcasted_iota(jnp.int32, lhs_f32.shape, 1) < valid_k
          lhs_f32 = jnp.where(mask_lhs, lhs_f32, 0.0)
        lhs_amax = jnp.max(jnp.abs(lhs_f32), axis=1, keepdims=True)
      else:
        lhs_amax = jnp.max(jnp.abs(tiled_lhs), axis=1, keepdims=True).astype(jnp.float32)
        lhs_f32 = tiled_lhs.astype(jnp.float32)
      lhs_nz = lhs_amax > 0.0
      lhs_safe = jnp.where(lhs_nz, lhs_amax, jnp.float32(1.0))
      lhs_scale = jnp.where(lhs_nz, lhs_safe / max_val, jnp.float32(1.0))
      lhs_inv = jnp.where(lhs_nz, max_val / lhs_safe, jnp.float32(0.0))
      lhs_in = jnp.clip(lhs_f32 * lhs_inv, -max_val, max_val).astype(q_dt)
      if tiled_rhs.dtype != q_dt:
        tiled_rhs = tiled_rhs.astype(q_dt)
      matmul_prec = None
    else:
      lhs_scale = None
      if tiled_lhs.dtype != tiled_rhs.dtype:
        common_dt = jnp.promote_types(tiled_lhs.dtype, tiled_rhs.dtype)
        tiled_lhs = tiled_lhs.astype(common_dt)
        tiled_rhs = tiled_rhs.astype(common_dt)
      lhs_in = tiled_lhs
      matmul_prec = lax.Precision.HIGHEST if lhs_in.dtype == jnp.float32 else None

    acc_list = []
    for start_n in range(0, rhs_tile_n, mxu_size):
      end_n = min(rhs_tile_n, start_n + mxu_size)
      if cfgs.transpose_rhs:
        block_acc = lax.dot_general(
            lhs_in,
            tiled_rhs[start_n:end_n, :],
            (((1,), (1,)), ((), ())),
            precision=matmul_prec,
            preferred_element_type=jnp.float32,
        )
      else:
        block_acc = jnp.matmul(
            lhs_in,
            tiled_rhs[:, start_n:end_n],
            precision=matmul_prec,
            preferred_element_type=jnp.float32,
        )
      acc_list.append(block_acc)
    acc_f32 = jnp.concatenate(acc_list, axis=1)
    if lhs_scale is not None:
      acc_f32 = acc_f32 * lhs_scale
    if cfgs.has_rhs_scale and not cfgs.rhs_scale_on_k:
      assert tiled_rhs_ref.scale is not None
      acc_f32 = acc_f32 * tiled_rhs_ref.scale[...].astype(jnp.float32)
    acc = acc_f32.astype(acc_ref.dtype)

    if not is_first_k_step:
      acc = acc + acc_ref[:bucket_m]
    acc_m = acc.shape[0]

    if is_last_k_step:
      iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
      mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
      acc_masked = jnp.where(mask, acc, 0)

      tiled_out_2d_ref = tiled_out_ref.reshape(-1, cfgs.tiles.tile_n)
      tiled_out_2d_ref[:acc_m] = acc_masked.astype(tiled_out_ref.dtype)

      partial_out_zeros = jnp.zeros_like(partial_out_ref)
      tiled_out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])
      last_row = m_end_local // cfgs.dims.size_lhs_sublane
      partial_out_ref[...] = jnp.where(
          m_end_local % cfgs.dims.size_lhs_sublane == 0,
          partial_out_zeros,
          tiled_out_ref[last_row],
      )
    else:
      acc_ref[:acc_m] = acc

  def run_matmul_step(bucket_idx: int) -> None:
    bucket_m = cfgs.tiles.bucket_base * (bucket_idx + 1)

    def matmul_first_last() -> None:
      _matmul(is_first_k_step=True, is_last_k_step=True, bucket_m=bucket_m)

    def matmul_first() -> None:
      _matmul(is_first_k_step=True, is_last_k_step=False, bucket_m=bucket_m)

    def matmul_mid() -> None:
      _matmul(is_first_k_step=False, is_last_k_step=False, bucket_m=bucket_m)

    def matmul_last() -> None:
      _matmul(is_first_k_step=False, is_last_k_step=True, bucket_m=bucket_m)

    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    k_id = step.index[2]
    is_first_k_step = k_id == 0
    is_last_k_step = k_id == (num_k - 1)
    lax.cond(
        is_first_k_step,
        lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
        lambda: lax.cond(is_last_k_step, matmul_last, matmul_mid),
    )

  num_buckets = cfgs.tiles.tile_m // cfgs.tiles.bucket_base
  branches = [functools.partial(run_matmul_step, bucket_idx=i) for i in range(num_buckets)]
  bucket_idx = (m_end_local - 1) // cfgs.tiles.bucket_base
  lax.switch(bucket_idx, branches)


def _gmm_zero_out_start(
    out_ref: Any,
    zero_ref: Any,
    semaphore_ref: Any,
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
) -> jax.Array:
  """Starts async DMAs zeroing unused leading/trailing output rows."""
  num_lanes = pltpu.get_tpu_info().num_lanes
  zero_ref[...] = jnp.zeros_like(zero_ref)

  zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
  out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
  row_size = zero_dma.shape[0]

  compute_start = metadata_ref.gm_id_to_m_offset[0]
  compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

  left_zero_start = 0
  left_zero_end = compute_start // dims.size_lhs_sublane
  left_num_loops = pl.cdiv(left_zero_end - left_zero_start, row_size)

  right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
  right_zero_end = out_dma.shape[0]
  right_num_loops = pl.cdiv(right_zero_end - right_zero_start, row_size)

  def fill_zero(
      i: jax.Array,
      zero_size: jax.Array,
      *,
      start: int | jax.Array,
      end: int | jax.Array,
  ) -> jax.Array:
    dma_start = start + i * row_size
    dma_end = jnp.minimum(dma_start + row_size, end)
    dma_size = dma_end - dma_start
    for n_start in range(0, out_dma.shape[-1], num_lanes):
      n_end = n_start + num_lanes
      pltpu.make_async_copy(
          src_ref=zero_dma.at[pl.ds(0, dma_size)],
          dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :, n_start:n_end],
          sem=semaphore_ref.at[0],
      ).start(priority=1)
    return zero_size + dma_size

  zero_size = lax.fori_loop(
      0,
      left_num_loops,
      lambda i, zs: fill_zero(i, zs, start=left_zero_start, end=left_zero_end),
      jnp.int32(0),
  )
  zero_size = lax.fori_loop(
      0,
      right_num_loops,
      lambda i, zs: fill_zero(i, zs, start=right_zero_start, end=right_zero_end),
      zero_size,
  )
  return zero_size


def _gmm_zero_out_end(
    out_ref: Any,
    semaphore_ref: Any,
    zero_size: jax.Array,
    *,
    dims: Dimensions,
) -> None:
  """Drains async zero-fill DMAs started by `_gmm_zero_out_start`."""
  out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
  pltpu.make_async_copy(
      src_ref=out_dma.at[pl.ds(0, zero_size)],
      dst_ref=out_dma.at[pl.ds(0, zero_size)],
      sem=semaphore_ref.at[0],
  ).wait()


def _gmm_kernel_main(
    lhs_group_sizes_ref: Any,
    group_offset_ref: Any,
    lhs_ref: Any,
    rhs_ref: Any,
    out_ref: Any,
    partial_out_ref: Any,
    acc_ref: Any,
    metadata_ref: MetadataRef,
    zero_ref: Any,
    semaphore_ref: Any,
    *,
    cfgs: GmmConfigs,
) -> None:
  """Single-core `pallas_call` entrypoint for `gmm_v2`."""
  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.dims.size_n, cfgs.tiles.tile_n)
  num_gm = fill_metadata(
      lhs_group_sizes_ref,
      group_offset_ref,
      metadata_ref,
      cfgs=cfgs,
  )

  zero_size = jnp.int32(0)
  if cfgs.zero_init:
    zero_size = _gmm_zero_out_start(
        out_ref,
        zero_ref,
        semaphore_ref,
        metadata_ref,
        num_gm,
        dims=cfgs.dims,
    )

  in_specs, out_spec = _generate_gmm_block_specs(metadata_ref, cfgs, num_gm=num_gm)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(_gmm_inner_kernel, cfgs=cfgs),
      grid=(num_n, num_gm, num_k),
      in_specs=in_specs,
      out_specs=out_spec,
      core_axis_name=None,
      dimension_semantics=None,
      _explicit_indices=True,
  )
  lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
  out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
  pipeline_fn(
      lhs_in,
      rhs_ref,
      out_in,
      scratches=[partial_out_ref, acc_ref, metadata_ref],
  )

  if cfgs.zero_init:
    _gmm_zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def gmm_v2(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jax.typing.DTypeLike,
    transpose_rhs: bool = False,
    zero_initialize: bool = True,
    tile_m_override: int | None = None,
    rhs_scale: jax.Array | None = None,
    rhs_scale_on_k: bool = False,
    lhs_qtype: jax.typing.DTypeLike | None = None,
    vmem_limit_bytes: int = _DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """Single-TensorCore grouped `[M, K] @ [G, K, N] -> [M, N]` Pallas kernel."""
  tpu_info = _ensure_tpu_backend()
  if lhs.ndim != 2 or rhs.ndim != 3 or group_sizes.ndim != 1:
    raise ValueError(
        "Expected lhs [M, K], rhs [G, K, N] (or [G, N, K]), group_sizes [G];"
        f" got {lhs.shape}, {rhs.shape}, {group_sizes.shape}."
    )
  orig_size_m, size_k = lhs.shape
  if transpose_rhs:
    size_group, size_n, rhs_k = rhs.shape
  else:
    size_group, rhs_k, size_n = rhs.shape
  if rhs_k != size_k or size_group > group_sizes.shape[0]:
    raise ValueError(
        f"Incompatible gmm_v2 shapes: lhs={lhs.shape}, rhs={rhs.shape},"
        f" group_sizes={group_sizes.shape}, {transpose_rhs=}."
    )
  if orig_size_m <= 0 or size_k <= 0 or size_n <= 0 or size_group <= 0:
    raise ValueError(f"All gmm_v2 dimensions must be > 0, got lhs={lhs.shape}," f" rhs={rhs.shape}.")
  has_rhs_scale = rhs_scale is not None
  lhs_q_dt = jnp.dtype(lhs_qtype) if lhs_qtype is not None else None
  if has_rhs_scale:
    expected_scale_len = size_k if rhs_scale_on_k else size_n
    if rhs_scale.shape != (size_group, 1, expected_scale_len):
      raise ValueError(
          f"Expected rhs_scale shape {(size_group, 1, expected_scale_len)} for"
          f" {rhs_scale_on_k=}, got {rhs_scale.shape}."
      )

  size_lhs_sublane = min(tpu_info.get_sublane_tiling(lhs.dtype), orig_size_m)
  padded_m = align_to(orig_size_m, size_lhs_sublane)
  if padded_m != orig_size_m:
    lhs = jnp.pad(lhs, ((0, padded_m - orig_size_m), (0, 0)))

  out_dt = jnp.dtype(preferred_element_type)
  acc_dt = jnp.dtype(jnp.float32)
  dims = Dimensions(
      size_m=padded_m,
      size_k=size_k,
      size_n=size_n,
      size_group=size_group,
      size_lhs_group=group_sizes.shape[0],
      size_lhs_sublane=size_lhs_sublane,
  )
  tiles = calculate_gmm_tiling(
      dims,
      lhs_dtype=lhs.dtype,
      rhs_dtype=rhs.dtype,
      out_dtype=out_dt,
      acc_dtype=acc_dt,
      vmem_limit_bytes=vmem_limit_bytes,
      transpose_rhs=transpose_rhs,
      zero_initialize=zero_initialize,
      tile_m_override=tile_m_override,
      has_rhs_scale=has_rhs_scale,
      rhs_scale_on_k=rhs_scale_on_k,
  )
  cfgs = GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_dtype=lhs.dtype,
      rhs_dtype=rhs.dtype,
      out_dtype=out_dt,
      acc_dtype=acc_dt,
      zero_init=zero_initialize,
      transpose_rhs=transpose_rhs,
      has_rhs_scale=has_rhs_scale,
      rhs_scale_on_k=rhs_scale_on_k,
      lhs_qtype=lhs_q_dt,
  )

  num_lanes = tpu_info.num_lanes
  aligned_k = align_to(dims.size_k, tiles.tile_k)
  aligned_rhs_n = align_to(dims.size_n, tiles.tile_n)
  if aligned_k != dims.size_k:
    lhs = jnp.pad(lhs, ((0, 0), (0, aligned_k - dims.size_k)))
  if transpose_rhs:
    if aligned_rhs_n != dims.size_n or aligned_k != dims.size_k:
      rhs = jnp.pad(
          rhs,
          (
              (0, 0),
              (0, aligned_rhs_n - dims.size_n),
              (0, aligned_k - dims.size_k),
          ),
      )
  elif aligned_k != dims.size_k or aligned_rhs_n != dims.size_n:
    rhs = jnp.pad(
        rhs,
        (
            (0, 0),
            (0, aligned_k - dims.size_k),
            (0, aligned_rhs_n - dims.size_n),
        ),
    )

  rhs_scale_in = None
  if rhs_scale is not None:
    scale_tile = tiles.tile_k if rhs_scale_on_k else tiles.tile_n
    aligned_scale_len = align_to(rhs_scale.shape[2], scale_tile)
    rhs_scale_in = rhs_scale.astype(jnp.float32)
    if aligned_scale_len != rhs_scale_in.shape[2]:
      rhs_scale_in = jnp.pad(
          rhs_scale_in,
          ((0, 0), (0, 0), (0, aligned_scale_len - rhs_scale_in.shape[2])),
          constant_values=1.0,
      )

  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
  scratch_shapes: list[Any] = [
      pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), out_dt),
      pltpu.VMEM((tiles.tile_m, tiles.tile_n), acc_dt),
      MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]
  if zero_initialize:
    out_bytes = out_dt.itemsize
    tile_zero_m = min(_TARGET_ZERO_REF_BYTES // num_lanes // out_bytes, dims.size_m)
    tile_zero_m = max(
        dims.size_lhs_sublane,
        (tile_zero_m // dims.size_lhs_sublane) * dims.size_lhs_sublane,
    )
    scratch_shapes.extend(
        [
            pltpu.VMEM((tile_zero_m, num_lanes), out_dt),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    )
  else:
    scratch_shapes.extend([None, None])

  aligned_n = align_to(dims.size_n, max(num_lanes, tiles.tile_n))
  out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), out_dt)
  group_offset = jnp.zeros((1,), dtype=jnp.int32)
  scope_name = (
      f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  rhs_hbm_spec = WeightsRef(
      weight=hbm_spec,
      scale=hbm_spec if has_rhs_scale else None,
  )
  rhs_arg = WeightsRef(weight=rhs, scale=rhs_scale_in)
  out = pl.pallas_call(
      functools.partial(_gmm_kernel_main, cfgs=cfgs),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm_spec, rhs_hbm_spec],
          out_specs=hbm_spec,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=scope_name,
  )(group_sizes.astype(jnp.int32), group_offset, lhs, rhs_arg)
  return out[:orig_size_m, : dims.size_n]


def calculate_tgmm_tiling(
    dims: Dimensions,
    *,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    vmem_limit_bytes: int,
    tile_m_override: int | None = None,
) -> TileSizes:
  """Selects single-core `tgmm_v2` tile sizes within `vmem_limit_bytes`."""
  _validate_tile_m_override("tgmm_tile_m", tile_m_override)
  tpu_info = _ensure_tpu_backend()
  num_lanes = tpu_info.num_lanes
  mxu_size = tpu_info.mxu_column_size

  default_base_m = max(min(256, dims.size_m), dims.size_lhs_sublane)
  if tile_m_override is not None:
    tile_m = max(min(tile_m_override, dims.size_m), dims.size_lhs_sublane)
  else:
    tile_m = default_base_m

  acc_bytes = jnp.dtype(acc_dtype).itemsize
  out_bytes = jnp.dtype(out_dtype).itemsize
  lhs_bytes = jnp.dtype(lhs_dtype).itemsize
  rhs_bytes = jnp.dtype(rhs_dtype).itemsize
  zero_reserve = min(_TARGET_ZERO_REF_BYTES, dims.size_k * num_lanes * out_bytes)

  def _within_vmem(tm: int, tk: int, tn: int) -> bool:
    if tk * tn > 1024 * 1024:
      return False
    num_buffers = 2
    mask_scratch = tm * (tk + tn) * 4
    budget = (
        tk * tn * (2 * acc_bytes + num_buffers * out_bytes)
        + (num_buffers + 2) * (tm * tk * lhs_bytes)
        + (num_buffers + 1) * (tm * tn * rhs_bytes)
        + mask_scratch
        + zero_reserve
    )
    return budget <= vmem_limit_bytes

  tile_n_lower = min(mxu_size * 2, dims.size_n)
  num_k_tiles = 1
  num_n_tiles = 1
  tile_n = align_to(dims.size_n, num_lanes)
  tile_k = align_to(dims.size_k, num_lanes)

  prev_tile_n = tile_n
  while not _within_vmem(tile_m, tile_k, tile_n):
    num_n_tiles += 1
    tile_n = align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles
    if tile_n < tile_n_lower or tile_n >= prev_tile_n:
      break
    prev_tile_n = tile_n

  if tile_n < tile_n_lower and num_n_tiles > 1:
    num_n_tiles -= 1
    tile_n = align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles

  prev_tile_k = tile_k
  while not _within_vmem(tile_m, tile_k, tile_n):
    num_k_tiles += 1
    tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles
    if tile_k < num_lanes or tile_k >= prev_tile_k:
      break
    prev_tile_k = tile_k

  if tile_k < num_lanes and num_k_tiles > 1:
    num_k_tiles -= 1
    tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

  while not _within_vmem(tile_m, tile_k, tile_n) and tile_m > num_lanes:
    tile_m = max(num_lanes, tile_m - num_lanes)

  if not _within_vmem(tile_m, tile_k, tile_n):
    raise ValueError(
        f"Could not fit tgmm_v2 tiles ({tile_m}, {tile_k}, {tile_n}) for"
        f" {dims=} in vmem_limit_bytes={vmem_limit_bytes}."
    )

  if tile_m_override is None:
    bucket_base = tile_m
    max_auto_tile_m = min(dims.size_m, 512)
    for _ in range(1, 4):
      new_tile_m = tile_m + bucket_base
      if new_tile_m > max_auto_tile_m:
        break
      if not _within_vmem(new_tile_m, tile_k, tile_n):
        break
      tile_m = new_tile_m
  else:
    if tile_m >= default_base_m and tile_m % default_base_m == 0:
      bucket_base = default_base_m
    elif tile_m >= num_lanes and tile_m % num_lanes == 0:
      bucket_base = num_lanes
    else:
      bucket_base = tile_m

  return TileSizes(
      tile_m=tile_m,
      tile_k=tile_k,
      tile_n=tile_n,
      bucket_base=bucket_base,
  )


class _TgmmIndexMaps:
  """BlockSpec index maps for `tgmm_v2` (`grid = (num_n, num_k, num_gm)`)."""

  def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs) -> None:
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs

  def lhs_index_map(self, _: jax.Array, k_id: jax.Array, gm_id: jax.Array) -> tuple[Any, int, jax.Array]:
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    return (pl.ds(row_start, row_end - row_start), 0, k_id)

  def rhs_index_map(self, n_id: jax.Array, _: jax.Array, gm_id: jax.Array) -> tuple[Any, int, jax.Array]:
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    return (pl.ds(row_start, row_end - row_start), 0, n_id)

  def out_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, k_id, n_id)


def _generate_tgmm_block_specs(
    metadata_ref: MetadataRef,
    cfgs: GmmConfigs,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
  """Builds `(lhs_spec, rhs_spec), out_spec` for `tgmm_v2`."""
  imap = _TgmmIndexMaps(metadata_ref, cfgs)
  bounded_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)
  lhs_spec = pl.BlockSpec(
      (bounded_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
      imap.lhs_index_map,
  )
  rhs_spec = pl.BlockSpec(
      (bounded_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
      imap.rhs_index_map,
  )
  out_spec = pl.BlockSpec(
      (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
      imap.out_index_map,
  )
  return (lhs_spec, rhs_spec), out_spec


def _tgmm_inner_kernel(
    step: pltpu.PipelineStep,
    tiled_lhs_ref: Any,
    tiled_rhs_ref: Any,
    tiled_out_ref: Any,
    acc_ref: Any,
    metadata_ref: MetadataRef,
    *,
    num_gm: jax.Array,
    cfgs: GmmConfigs,
) -> None:
  """Per-tile MXU pipeline body for `tgmm_v2` (`lhs.T @ rhs`)."""
  lhs_2d = tiled_lhs_ref.reshape(-1, tiled_lhs_ref.shape[-1])
  rhs_2d = tiled_rhs_ref.reshape(-1, tiled_rhs_ref.shape[-1])
  gm_id = step.index[2]

  m_start = metadata_ref.gm_id_to_m_offset[gm_id]
  m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
  m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
  m_start_local = m_start - m_offset
  m_end_local = m_end - m_offset

  def _matmul(
      is_new_group: bool,
      is_group_changing: bool,
      bucket_m: int,
  ) -> None:
    lhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, lhs_2d.shape[-1]), 0)
    lhs_mask = jnp.logical_and(m_start_local <= lhs_iota, lhs_iota < m_end_local)
    lhs_masked = jnp.where(lhs_mask, lhs_2d[:bucket_m], 0)

    rhs_iota = lax.broadcasted_iota(jnp.int32, (bucket_m, rhs_2d.shape[-1]), 0)
    rhs_mask = jnp.logical_and(m_start_local <= rhs_iota, rhs_iota < m_end_local)
    rhs_masked = jnp.where(rhs_mask, rhs_2d[:bucket_m], 0)

    if lhs_masked.dtype != rhs_masked.dtype:
      common_dt = jnp.promote_types(lhs_masked.dtype, rhs_masked.dtype)
      lhs_masked = lhs_masked.astype(common_dt)
      rhs_masked = rhs_masked.astype(common_dt)

    dot_prec = lax.Precision.HIGHEST if lhs_masked.dtype == jnp.float32 else None
    acc = jax.lax.dot_general(
        lhs_masked,
        rhs_masked,
        (((0,), (0,)), ((), ())),
        precision=dot_prec,
        preferred_element_type=jnp.float32,
    ).astype(acc_ref.dtype)

    if not is_new_group:
      acc = acc + acc_ref[...]

    if is_group_changing:
      tiled_out_ref[...] = acc.astype(tiled_out_ref.dtype)
    else:
      acc_ref[...] = acc

  prev_gm_id = jnp.where(gm_id > 0, gm_id - 1, 0)
  new_group = jnp.logical_or(
      gm_id == 0,
      metadata_ref.gm_id_to_group_id[gm_id] != metadata_ref.gm_id_to_group_id[prev_gm_id],
  )
  is_last_gm = gm_id == (num_gm - 1)
  next_gm_id = jnp.where(is_last_gm, gm_id, gm_id + 1)
  group_is_changing = jnp.logical_or(
      is_last_gm,
      metadata_ref.gm_id_to_group_id[gm_id] != metadata_ref.gm_id_to_group_id[next_gm_id],
  )

  def run_matmul_step(bucket_m: int) -> None:
    def matmul_new_and_changing() -> None:
      _matmul(is_new_group=True, is_group_changing=True, bucket_m=bucket_m)

    def matmul_new_only() -> None:
      _matmul(is_new_group=True, is_group_changing=False, bucket_m=bucket_m)

    def matmul_mid() -> None:
      _matmul(is_new_group=False, is_group_changing=False, bucket_m=bucket_m)

    def matmul_changing_only() -> None:
      _matmul(is_new_group=False, is_group_changing=True, bucket_m=bucket_m)

    lax.cond(
        new_group,
        lambda: lax.cond(group_is_changing, matmul_new_and_changing, matmul_new_only),
        lambda: lax.cond(group_is_changing, matmul_changing_only, matmul_mid),
    )

  bucket_base = cfgs.tiles.bucket_base
  num_buckets = cfgs.tiles.tile_m // bucket_base
  bucket_idx = jnp.maximum(0, (m_end_local - 1) // bucket_base)
  lax.switch(
      bucket_idx,
      [functools.partial(run_matmul_step, bucket_m=bucket_base * (i + 1)) for i in range(num_buckets)],
  )


def _tgmm_zero_out_start(
    lhs_group_sizes_ref: Any,
    group_offset_ref: Any,
    out_ref: Any,
    zero_ref: Any,
    semaphore_ref: Any,
) -> jax.Array:
  """Starts async DMAs zeroing `out_ref[g]` for empty groups (`size == 0`)."""
  num_actual_groups, aligned_k, aligned_n = out_ref.shape
  tile_zero_k = zero_ref.shape[0]
  zero_3d = zero_ref.reshape(1, tile_zero_k, -1)
  num_lanes = pltpu.get_tpu_info().num_lanes
  zero_3d[...] = jnp.zeros_like(zero_3d)

  def fill_zero(local_group_id: jax.Array, should_copy: jax.Array) -> None:
    should_copy_int = should_copy.astype(jnp.int32)
    for i in range(pl.cdiv(aligned_k, tile_zero_k)):
      size_k_to_copy = min(tile_zero_k, aligned_k - i * tile_zero_k)
      for j in range(aligned_n // num_lanes):
        src = zero_3d.at[pl.ds(0, should_copy_int), pl.ds(0, size_k_to_copy)]
        dst = out_ref.at[
            pl.ds(local_group_id, should_copy_int),
            pl.ds(i * tile_zero_k, size_k_to_copy),
            pl.ds(j * num_lanes, num_lanes),
        ]
        pltpu.make_async_copy(
            src_ref=src,
            dst_ref=dst,
            sem=semaphore_ref.at[0],
        ).start(priority=1)

  group_offset = group_offset_ref[0]

  def body(local_group_id: jax.Array, num_groups_to_zero: jax.Array) -> jax.Array:
    global_group_id = local_group_id + group_offset
    should_copy = lhs_group_sizes_ref[global_group_id] == 0
    fill_zero(local_group_id, should_copy)
    return num_groups_to_zero + should_copy.astype(jnp.int32)

  return lax.fori_loop(0, num_actual_groups, body, jnp.int32(0))


def _tgmm_zero_out_end(
    num_groups_to_zero: jax.Array,
    out_ref: Any,
    semaphore_ref: Any,
) -> None:
  """Drains async zero-fill DMAs started by `_tgmm_zero_out_start`."""
  dst = out_ref.at[pl.ds(0, num_groups_to_zero)]
  pltpu.make_async_copy(
      src_ref=dst,
      dst_ref=dst,
      sem=semaphore_ref.at[0],
  ).wait()


def _tgmm_kernel_main(
    lhs_group_sizes_ref: Any,
    group_offset_ref: Any,
    lhs_ref: Any,
    rhs_ref: Any,
    out_ref: Any,
    acc_ref: Any,
    metadata_ref: MetadataRef,
    zero_ref: Any,
    semaphore_ref: Any,
    *,
    cfgs: GmmConfigs,
) -> None:
  """Single-core `pallas_call` entrypoint for `tgmm_v2`."""
  num_groups_to_zero = _tgmm_zero_out_start(
      lhs_group_sizes_ref,
      group_offset_ref,
      out_ref,
      zero_ref,
      semaphore_ref,
  )
  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.dims.size_n, cfgs.tiles.tile_n)
  num_gm = fill_metadata(
      lhs_group_sizes_ref,
      group_offset_ref,
      metadata_ref,
      cfgs=cfgs,
  )
  in_specs, out_spec = _generate_tgmm_block_specs(metadata_ref, cfgs)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(_tgmm_inner_kernel, num_gm=num_gm, cfgs=cfgs),
      grid=(num_n, num_k, num_gm),
      in_specs=in_specs,
      out_specs=out_spec,
      core_axis_name=None,
      dimension_semantics=None,
      _explicit_indices=True,
  )
  lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
  rhs_in = rhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, rhs_ref.shape[-1])
  pipeline_fn(
      lhs_in,
      rhs_in,
      out_ref,
      scratches=[acc_ref, metadata_ref],
  )
  _tgmm_zero_out_end(num_groups_to_zero, out_ref, semaphore_ref)


def tgmm_v2(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    num_actual_groups: int,
    preferred_element_type: jax.typing.DTypeLike,
    tile_m_override: int | None = None,
    vmem_limit_bytes: int = _DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """Single-TensorCore grouped `lhs.T @ rhs -> [G, K, N]` Pallas kernel."""
  tpu_info = _ensure_tpu_backend()
  if lhs.ndim != 2 or rhs.ndim != 2 or group_sizes.ndim != 1:
    raise ValueError(
        "Expected lhs [M, K], rhs [M, N], group_sizes [G];" f" got {lhs.shape}, {rhs.shape}, {group_sizes.shape}."
    )
  orig_size_m, size_k = lhs.shape
  rhs_m, size_n = rhs.shape
  if rhs_m != orig_size_m or num_actual_groups > group_sizes.shape[0]:
    raise ValueError(
        f"Incompatible tgmm_v2 shapes: lhs={lhs.shape}, rhs={rhs.shape},"
        f" group_sizes={group_sizes.shape}, {num_actual_groups=}."
    )
  if orig_size_m <= 0 or size_k <= 0 or size_n <= 0 or num_actual_groups <= 0:
    raise ValueError(
        f"All tgmm_v2 dimensions must be > 0, got lhs={lhs.shape}," f" rhs={rhs.shape}, {num_actual_groups=}."
    )

  num_lanes = tpu_info.num_lanes
  raw_sublane = max(
      tpu_info.get_sublane_tiling(lhs.dtype),
      tpu_info.get_sublane_tiling(rhs.dtype),
  )
  size_lhs_sublane = min(raw_sublane, orig_size_m)
  padded_m = align_to(orig_size_m, size_lhs_sublane)
  if padded_m != orig_size_m:
    lhs = jnp.pad(lhs, ((0, padded_m - orig_size_m), (0, 0)))
    rhs = jnp.pad(rhs, ((0, padded_m - orig_size_m), (0, 0)))

  out_dt = jnp.dtype(preferred_element_type)
  acc_dt = jnp.dtype(jnp.float32)
  dims = Dimensions(
      size_m=padded_m,
      size_k=size_k,
      size_n=size_n,
      size_group=num_actual_groups,
      size_lhs_group=group_sizes.shape[0],
      size_lhs_sublane=size_lhs_sublane,
  )
  tiles = calculate_tgmm_tiling(
      dims,
      lhs_dtype=lhs.dtype,
      rhs_dtype=rhs.dtype,
      out_dtype=out_dt,
      acc_dtype=acc_dt,
      vmem_limit_bytes=vmem_limit_bytes,
      tile_m_override=tile_m_override,
  )
  cfgs = GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_dtype=lhs.dtype,
      rhs_dtype=rhs.dtype,
      out_dtype=out_dt,
      acc_dtype=acc_dt,
      zero_init=True,
      transpose_rhs=False,
  )

  aligned_n = align_to(dims.size_n, num_lanes)
  aligned_k = align_to(dims.size_k, tiles.tile_k)
  if aligned_k != dims.size_k:
    lhs = jnp.pad(lhs, ((0, 0), (0, aligned_k - dims.size_k)))
  if aligned_n != dims.size_n:
    rhs = jnp.pad(rhs, ((0, 0), (0, aligned_n - dims.size_n)))
  out_init = jax.ShapeDtypeStruct((num_actual_groups, aligned_k, aligned_n), out_dt)
  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1

  out_bytes = out_dt.itemsize
  size_out_sublane = tpu_info.get_sublane_tiling(out_dt)
  tile_zero_k = min(_TARGET_ZERO_REF_BYTES // num_lanes // out_bytes, dims.size_k)
  tile_zero_k = max(
      size_out_sublane,
      (tile_zero_k // size_out_sublane) * size_out_sublane,
  )
  scratch_shapes: list[Any] = [
      pltpu.VMEM((tiles.tile_k, tiles.tile_n), acc_dt),
      MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
      pltpu.VMEM((tile_zero_k, num_lanes), out_dt),
      pltpu.SemaphoreType.DMA((1,)),
  ]

  group_offset = jnp.zeros((1,), dtype=jnp.int32)
  scope_name = (
      f"tgmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  out = pl.pallas_call(
      functools.partial(_tgmm_kernel_main, cfgs=cfgs),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm_spec, hbm_spec],
          out_specs=hbm_spec,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=scope_name,
  )(group_sizes.astype(jnp.int32), group_offset, lhs, rhs)
  return out[:, : dims.size_k, : dims.size_n]


def _validate_pallas_ragged_dot_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    gmm_tile_m: int | None,
    tgmm_tile_m: int | None,
) -> None:
  """Fails fast on non-TPU backend or unaligned `M, K, N` dimensions."""
  _ensure_tpu_backend()
  _validate_tile_m_override("gmm_tile_m", gmm_tile_m)
  _validate_tile_m_override("tgmm_tile_m", tgmm_tile_m)
  if lhs.ndim != 2 or rhs.ndim != 3 or group_sizes.ndim != 1:
    raise ValueError(
        "pallas_ragged_dot expects lhs [M, K], rhs [G, K, N], group_sizes [G];"
        f" got {lhs.shape}, {rhs.shape}, {group_sizes.shape}."
    )
  size_m, size_k = lhs.shape
  size_g, rhs_k, size_n = rhs.shape
  if rhs_k != size_k or group_sizes.shape[0] != size_g:
    raise ValueError(
        f"Shape mismatch in pallas_ragged_dot: lhs={lhs.shape}," f" rhs={rhs.shape}, group_sizes={group_sizes.shape}."
    )
  if size_m <= 0 or size_k <= 0 or size_n <= 0 or size_m % 128 != 0 or size_k % 128 != 0 or size_n % 128 != 0:
    raise ValueError(
        "pallas_ragged_dot requires positive M, K, N divisible by 128, got" f" (M={size_m}, K={size_k}, N={size_n})."
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def pallas_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    gmm_tile_m: int | None = None,
    tgmm_tile_m: int | None = None,
) -> jax.Array:
  """Differentiable grouped matmul backed by `gmm_v2` and `tgmm_v2`."""
  _validate_pallas_ragged_dot_inputs(lhs, rhs, group_sizes, gmm_tile_m, tgmm_tile_m)
  return gmm_v2(
      lhs,
      rhs,
      group_sizes.astype(jnp.int32),
      preferred_element_type=preferred_element_type,
      transpose_rhs=False,
      zero_initialize=True,
      tile_m_override=gmm_tile_m,
  )


def _pallas_ragged_dot_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    gmm_tile_m: int | None = None,
    tgmm_tile_m: int | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  _validate_pallas_ragged_dot_inputs(lhs, rhs, group_sizes, gmm_tile_m, tgmm_tile_m)
  gs_i32 = group_sizes.astype(jnp.int32)
  out = gmm_v2(
      lhs,
      rhs,
      gs_i32,
      preferred_element_type=preferred_element_type,
      transpose_rhs=False,
      zero_initialize=True,
      tile_m_override=gmm_tile_m,
  )
  return out, (lhs, rhs, gs_i32)


def _pallas_ragged_dot_bwd(
    preferred_element_type: jnp.dtype,
    gmm_tile_m: int | None,
    tgmm_tile_m: int | None,
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    do: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  del preferred_element_type
  lhs, rhs, gs_i32 = residuals
  dlhs = gmm_v2(
      do,
      rhs,
      gs_i32,
      preferred_element_type=lhs.dtype,
      transpose_rhs=True,
      zero_initialize=True,
      tile_m_override=gmm_tile_m,
  )
  drhs = tgmm_v2(
      lhs,
      do,
      gs_i32,
      num_actual_groups=rhs.shape[0],
      preferred_element_type=rhs.dtype,
      tile_m_override=tgmm_tile_m,
  )
  return dlhs, drhs, None


pallas_ragged_dot.defvjp(_pallas_ragged_dot_fwd, _pallas_ragged_dot_bwd)


def _validate_pallas_ragged_dot_transposed_rhs_inputs(
    lhs: jax.Array,
    rhs_t: jax.Array,
    group_sizes: jax.Array,
    gmm_tile_m: int | None,
    tgmm_tile_m: int | None,
) -> None:
  """Fails fast on non-TPU backend or unaligned `lhs [M,K], rhs_t [G,N,K]`."""
  _ensure_tpu_backend()
  _validate_tile_m_override("gmm_tile_m", gmm_tile_m)
  _validate_tile_m_override("tgmm_tile_m", tgmm_tile_m)
  if lhs.ndim != 2 or rhs_t.ndim != 3 or group_sizes.ndim != 1:
    raise ValueError(
        "pallas_ragged_dot_transposed_rhs expects lhs [M, K], rhs_t [G, N, K],"
        f" group_sizes [G]; got {lhs.shape}, {rhs_t.shape},"
        f" {group_sizes.shape}."
    )
  size_m, size_k = lhs.shape
  size_g, size_n, rhs_k = rhs_t.shape
  if rhs_k != size_k or group_sizes.shape[0] != size_g:
    raise ValueError(
        f"Shape mismatch in pallas_ragged_dot_transposed_rhs: lhs={lhs.shape},"
        f" rhs_t={rhs_t.shape}, group_sizes={group_sizes.shape}."
    )
  if size_m <= 0 or size_k <= 0 or size_n <= 0 or size_m % 128 != 0 or size_k % 128 != 0 or size_n % 128 != 0:
    raise ValueError(
        "pallas_ragged_dot_transposed_rhs requires positive M, K, N divisible"
        f" by 128, got (M={size_m}, K={size_k}, N={size_n})."
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def pallas_ragged_dot_transposed_rhs(
    lhs: jax.Array,
    rhs_t: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    gmm_tile_m: int | None = None,
    tgmm_tile_m: int | None = None,
) -> jax.Array:
  """Differentiable grouped `lhs [M, K] @ rhs_t [G, N, K]^T -> [M, N]`."""
  _validate_pallas_ragged_dot_transposed_rhs_inputs(lhs, rhs_t, group_sizes, gmm_tile_m, tgmm_tile_m)
  return gmm_v2(
      lhs,
      rhs_t,
      group_sizes.astype(jnp.int32),
      preferred_element_type=preferred_element_type,
      transpose_rhs=True,
      zero_initialize=True,
      tile_m_override=gmm_tile_m,
  )


def _pallas_ragged_dot_transposed_rhs_fwd(
    lhs: jax.Array,
    rhs_t: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    gmm_tile_m: int | None = None,
    tgmm_tile_m: int | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  _validate_pallas_ragged_dot_transposed_rhs_inputs(lhs, rhs_t, group_sizes, gmm_tile_m, tgmm_tile_m)
  gs_i32 = group_sizes.astype(jnp.int32)
  out = gmm_v2(
      lhs,
      rhs_t,
      gs_i32,
      preferred_element_type=preferred_element_type,
      transpose_rhs=True,
      zero_initialize=True,
      tile_m_override=gmm_tile_m,
  )
  return out, (lhs, rhs_t, gs_i32)


def _pallas_ragged_dot_transposed_rhs_bwd(
    preferred_element_type: jnp.dtype,
    gmm_tile_m: int | None,
    tgmm_tile_m: int | None,
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    do: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  del preferred_element_type
  lhs, rhs_t, gs_i32 = residuals
  dlhs = gmm_v2(
      do,
      rhs_t,
      gs_i32,
      preferred_element_type=lhs.dtype,
      transpose_rhs=False,
      zero_initialize=True,
      tile_m_override=gmm_tile_m,
  )
  drhs_t = tgmm_v2(
      do,
      lhs,
      gs_i32,
      num_actual_groups=rhs_t.shape[0],
      preferred_element_type=rhs_t.dtype,
      tile_m_override=tgmm_tile_m,
  )
  return dlhs, drhs_t, None


pallas_ragged_dot_transposed_rhs.defvjp(
    _pallas_ragged_dot_transposed_rhs_fwd,
    _pallas_ragged_dot_transposed_rhs_bwd,
)
