# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Forked from:
# https://github.com/openxla/tokamax/blob/3f332fcf85dcb87aab661d00228ed71a09b5fd56/
# tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py
"""TGMM kernel"""

import dataclasses
import functools
from typing import Any, Callable, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OperandRef:
  """Bundles a kernel operand with its optional per-N scale.

  Registered as a pytree so it can be passed as a single 'pl.pallas_call' /
  'emit_pipeline' operand (and in_spec). When 'scale' is None it contributes no
  pytree leaf, so the kernel signature stays fixed-arity regardless of whether a
  scale is present; the kernel just reads 'rhs_ref.scale' (None or a ref).
  """

  value: Any
  scale: Any | None = None


TileTgmmFn = Callable[
    [
        gmm_v2.Dimensions,
        gmm_v2.InputConfigs,
        gmm_v2.InputConfigs,
        int,
        jnp.dtype,
        jnp.dtype,
        int,
        bool,
    ],
    gmm_v2.TileSizes,
]


def get_scope_name(cfgs: gmm_v2.GmmConfigs) -> str:
  dims = cfgs.dims
  tiles = cfgs.tiles
  return (
      f"tgmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )


def get_cost_estimate(cfgs: gmm_v2.GmmConfigs) -> pl.CostEstimate:
  dims = cfgs.dims
  flops = 2 * dims.size_m * dims.size_k * dims.size_n
  lhs_bytes = dims.size_m * dims.size_k * cfgs.lhs_cfgs.dtype.itemsize
  rhs_bytes = dims.size_m * dims.size_n * cfgs.rhs_cfgs.dtype.itemsize
  out_bytes = dims.size_group * dims.size_k * dims.size_n * cfgs.out_dtype.itemsize
  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=lhs_bytes + rhs_bytes + out_bytes,
      transcendentals=0,
  )


def calculate_tgmm_tiling(
    dims: gmm_v2.Dimensions,
    lhs_cfgs: gmm_v2.InputConfigs,
    rhs_cfgs: gmm_v2.InputConfigs,
    vmem_limit_bytes: int,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
    target_zero_ref_bytes: int,
    has_partial_sum: bool = False,
) -> gmm_v2.TileSizes:
  """Calculate optimal tile sizes for TGMM kernel."""
  # In tgmm, we calculate lhs.T @ dout which doesn't require quantization.
  # Since we use it in MOE, the m can be dynamic and small. So we don't
  # want it to be too big. At the same time, because the mxu size is 256, the
  # rhs is divided into 256x256 tiles. The lhs is divided to blocks of 256-wide
  # (256 on the contracting dimension) rows. So any size less than 256 will
  # have the same perf as using 256.
  bf16_bf16_tile_m = 256
  tile_m = min(bf16_bf16_tile_m, dims.size_m)
  tile_m = max(tile_m, dims.size_lhs_sublane)

  num_k_tiles = num_n_tiles = 1
  num_lanes = pltpu.get_tpu_info().num_lanes
  tile_n = gmm_v2.align_to(dims.size_n, num_lanes)
  # To avoid stalling MXU, we add some buffer room where tile_n cannot go
  # smaller than 2x of mxu_column_size.
  tile_n_lower_bound = pltpu.get_tpu_info().mxu_column_size * 2
  tile_n_lower_bound = min(tile_n_lower_bound, dims.size_n)
  tile_k = gmm_v2.align_to(dims.size_k, num_lanes)

  def within_vmem_limit(tile_m, tile_k, tile_n):
    acc_bytes = jax.dtypes.itemsize_bits(acc_dtype) // 8
    out_bytes = jax.dtypes.itemsize_bits(out_dtype) // 8
    lhs_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    rhs_bytes = jax.dtypes.itemsize_bits(rhs_cfgs.dtype) // 8
    num_buffers = 2
    # For lhs, we use (num_buffers+1). +1 is needed because we are doing
    # lhs.T @ rhs, lhs cannot be fed directly into MXU and has to go through
    # XLU's transpose. in order to reduce redundant XLU computation, instead
    # of performing XLU's transpose every time lhs is pushed into XLU, it
    # caches the transposed value into VMEM. this increases VMEM requirement.
    ps_bytes = tile_k * tile_n * num_buffers * out_bytes if has_partial_sum else 0
    budget = (
        tile_k * tile_n * (acc_bytes + num_buffers * out_bytes)
        + ps_bytes
        + (num_buffers + 1) * (tile_m * tile_k * lhs_bytes)
        + num_buffers * (tile_m * tile_n * rhs_bytes)
        # Reserve VMEM for zero_ref. Use the upper bound target_zero_ref_bytes
        # since the actual zero_ref size depends on out_dtype/size_k and is
        # always <= this value.
        + target_zero_ref_bytes
    )
    return budget <= vmem_limit_bytes

  prev_tile_n = tile_n
  while not within_vmem_limit(tile_m, tile_k, tile_n):
    num_n_tiles += 1
    # The reason why we do "tile_n * num_n_tiles must cover size_n." is
    # tile_n must be a multiple of num_lanes and
    # tile_n * num_n_tiles must cover size_n.
    tile_n = gmm_v2.align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles
    # If size_n is small and awkwardly sized (e.g., size_n=100, num_lanes=128),
    # align_to(100, N*128) // N can get stuck at a constant value (128) as N
    # grows. If that constant value is above the floor and budget still
    # doesn't fit, the loop never terminates. That's why we need to check if
    # "tile_n >= prev_tile_n".
    if tile_n < tile_n_lower_bound or tile_n >= prev_tile_n:
      break
    prev_tile_n = tile_n

  if tile_n >= tile_n_lower_bound and within_vmem_limit(tile_m, tile_k, tile_n):
    return gmm_v2.TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

  if tile_n < tile_n_lower_bound:
    num_n_tiles -= 1
    tile_n = gmm_v2.align_to(dims.size_n, num_n_tiles * num_lanes) // num_n_tiles

  prev_tile_k = tile_k
  while not within_vmem_limit(tile_m, tile_k, tile_n):
    num_k_tiles += 1
    tile_k = gmm_v2.align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles
    if tile_k < num_lanes or tile_k >= prev_tile_k:
      break
    prev_tile_k = tile_k

  if tile_k < num_lanes:
    num_k_tiles -= 1
    tile_k = gmm_v2.align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

  if not within_vmem_limit(tile_m, tile_k, tile_n):
    raise ValueError(
        f"Could not find valid tile sizes for tgmm. dims={dims},"
        f" tiles=({tile_m},{tile_k},{tile_n}), vmem={vmem_limit_bytes}"
    )
  return gmm_v2.TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def make_tgmm_configs(
    lhs: jax.Array,  # [m, k]
    rhs: jax.Array,  # [m, n]
    rhs_scale: jax.Array,  # [1, 1, n] (per-N scale)
    partial_sum: jax.Array | None,
    group_sizes: jax.Array,
    num_actual_groups: int,
    *,
    tile_info: gmm_v2.TileSizes | TileTgmmFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype | None,
    target_zero_ref_bytes: int,
):
  """Fills the GMM config for the TGMM kernel."""
  assert out_dtype, "out_dtype cannot be None"
  assert lhs.shape[0] == rhs.shape[0], (
      f"lhs and rhs m-dim mismatch: {lhs.shape[0]}!={rhs.shape[0]} {lhs.shape}" f" vs {rhs.shape}"
  )
  size_m, size_k = lhs.shape
  _, size_n = rhs.shape
  if rhs_scale is not None:
    # rhs_scale.shape[0] is the number of quant blocks along the m (reduction)
    # dimension. tgmm_v2 only implements per-N (per-output-channel) scaling,
    # i.e. a single m-block: rhs_scale.shape == (1, 1, size_n). A leading dim
    # > 1 means the caller wants sub-channel (per-m-block) quantization.
    if rhs_scale.ndim == 3 and rhs_scale.shape[0] > 1:
      raise NotImplementedError(
          "tgmm_v2 only supports per-N rhs_scale with shape (1, 1, size_n);"
          f" got {rhs_scale.shape}, which implies {rhs_scale.shape[0]}"
          " sub-channel quant blocks along the m (reduction) dimension."
          " Sub-channel quantization is not implemented."
      )
    assert rhs_scale.shape == (1, 1, size_n), (
        f"expecting rhs_scale.shape to be (1, 1, size_n) but got" f" {rhs_scale.shape}"
    )
  # size_lhs_sublane is used in tgmm_inner_kernel to set the
  # (m/size_lhs_sublane, size_lhs_sublane, ...) reshape tile used on the m-axis
  # for both 'tiled_lhs_ref' and 'tiled_rhs_ref'.
  size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
  size_lhs_sublane = min(size_lhs_sublane, size_m)
  size_rhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(rhs.dtype)
  size_rhs_sublane = min(size_rhs_sublane, size_m)
  assert size_lhs_sublane == size_rhs_sublane, (
      f"size_lhs_sublane should be the same as size_rhs_sublane {lhs.dtype=}," f" {rhs.dtype=}"
  )
  dims = gmm_v2.Dimensions(
      size_m=size_m,
      size_k=size_k,
      size_n=size_n,
      size_group=num_actual_groups,  # weight.shape[0]
      size_lhs_group=group_sizes.shape[0],
      size_lhs_sublane=size_lhs_sublane,
  )

  rhs_quant_block_size_m = size_m
  rhs_cfgs = gmm_v2.InputConfigs(
      quant_dtype=None,
      quant_block_size=rhs_quant_block_size_m,
      dtype=rhs.dtype,
      has_scale=(rhs_scale is not None),
  )
  lhs_cfgs = gmm_v2.InputConfigs(
      quant_dtype=None,
      quant_block_size=-1,
      dtype=lhs.dtype,
  )

  fuse_act = None  # fuse_act has to be None in tgmm.
  if acc_dtype is None:
    acc_dtype = jnp.float32.dtype
  if isinstance(tile_info, gmm_v2.TileSizes):
    tiles = tile_info
  else:
    tiles = tile_info(
        dims,
        lhs_cfgs,
        rhs_cfgs,
        vmem_limit_bytes,  # pyrefly: ignore[bad-argument-type]
        out_dtype,
        acc_dtype,
        target_zero_ref_bytes,
        partial_sum is not None,
    )

  return gmm_v2.GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_cfgs=lhs_cfgs,
      rhs_cfgs=rhs_cfgs,
      has_partial_sum=(partial_sum is not None),
      out_dtype=jnp.dtype(out_dtype),
      acc_dtype=jnp.dtype(acc_dtype),
      # GMM's 'zero_init' zeros unvisited m-rows via DMA, which doesn't apply to
      # tgmm's [num_groups, k, n] output. The actual zero-initialization for
      # tgmm accumulation happens at the 'pallas_call' level.
      zero_init=False,
      fuse_act=fuse_act,
  )


def tgmm_inner_kernel(
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: OperandRef,
    # .value: [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # .scale: [1, 1, tile_n] or None
    tiled_ps_ref: jax.Array | None,
    tiled_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: gmm_v2.MetadataRef,
    *,
    cfgs: gmm_v2.GmmConfigs,
):
  """Inner kernel for TGMM computation.

  This kernel performs the matrix multiplication for a single tile of the output
  in the TGMM operation (lhs.T @ rhs). It handles masking for partial groups
  and accumulation across different group-major tiles.

  Args:
    tiled_lhs_ref: Reference to the tiled LHS data.
    tiled_rhs_ref: OperandRef bundling the tiled RHS data ('.value') and its
      optional per-N scale ('.scale', None when there is no scale).
    tiled_out_ref: Reference to the tiled output buffer [None, tile_k, tile_n].
    acc_ref: Scratch memory for accumulation [tile_k, tile_n].
    metadata_ref: Contains metadata like group offsets and group IDs.
    cfgs: GmmConfigs object containing kernel configurations.
  """
  # NB: grid=(num_n, num_k, num_gm)
  tiled_rhs_scale_ref = tiled_rhs_ref.scale

  tiled_lhs_ref = tiled_lhs_ref.reshape(-1, tiled_lhs_ref.shape[-1])
  tiled_rhs_ref = tiled_rhs_ref.value.reshape(-1, tiled_rhs_ref.value.shape[-1])
  gm_id = pl.program_id(2)

  def _matmul(is_new_group: bool, is_group_changing: bool):

    # Mask out invalid rows in the LHS/RHS tiles.
    # The DMA loads tiles aligned to sublane boundaries, but the actual group
    # data may not start/end on those boundaries.
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset
    lhs_iota = lax.broadcasted_iota(jnp.int32, tiled_lhs_ref.shape, 0)
    lhs_mask = jnp.logical_and(m_start_local <= lhs_iota, lhs_iota < m_end_local)
    lhs_masked = jnp.where(lhs_mask, tiled_lhs_ref[...], 0)
    # If there are no NaNs, masking both lhs and rhs shouldn't be necessary.
    # But without masking both, we sometimes see the result contain NaNs so we
    # decide to mask both to be safe.
    rhs_iota = lax.broadcasted_iota(jnp.int32, tiled_rhs_ref.shape, 0)
    rhs_mask = jnp.logical_and(m_start_local <= rhs_iota, rhs_iota < m_end_local)
    rhs_masked = jnp.where(rhs_mask, tiled_rhs_ref[...], 0)

    acc = jax.lax.dot_general(
        lhs_masked,
        rhs_masked,
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    if not is_new_group:
      acc += acc_ref[...]

    if is_group_changing:
      if cfgs.rhs_cfgs.has_scale:
        scale_slice = tiled_rhs_scale_ref[0]  # pyrefly: ignore[unsupported-operation]
        acc *= scale_slice
      if cfgs.has_partial_sum:
        acc += tiled_ps_ref[...].astype(acc.dtype)
      tiled_out_ref[...] = acc.astype(tiled_out_ref.dtype)
    else:
      acc_ref[...] = acc

  @jax.named_scope("matmul_new_group_and_changing")
  def matmul_new_group_and_changing():
    _matmul(is_new_group=True, is_group_changing=True)

  @jax.named_scope("matmul_new_group")
  def matmul_new_group():
    _matmul(is_new_group=True, is_group_changing=False)

  @jax.named_scope("matmul")
  def matmul():
    _matmul(is_new_group=False, is_group_changing=False)

  @jax.named_scope("matmul_group_changing")
  def matmul_group_changing():
    _matmul(is_new_group=False, is_group_changing=True)

  prev_gm_id = jnp.where(gm_id > 0, gm_id - 1, 0)
  is_first_gm = gm_id == 0
  group_id_changed = metadata_ref.gm_id_to_group_id[gm_id] != metadata_ref.gm_id_to_group_id[prev_gm_id]
  new_group = jnp.logical_or(is_first_gm, group_id_changed)

  is_last_gm = gm_id == (pl.num_programs(2) - 1)
  next_gm_id = jnp.where(is_last_gm, gm_id, gm_id + 1)
  next_group_id = metadata_ref.gm_id_to_group_id[next_gm_id]
  cur_group_id = metadata_ref.gm_id_to_group_id[gm_id]
  group_is_changing = jnp.logical_or(is_last_gm, cur_group_id != next_group_id)

  lax.cond(
      new_group,
      lambda: lax.cond(
          group_is_changing,
          # gm_id is the only one in its group =>
          # group_size + local_offset ≤ tile_m.
          matmul_new_group_and_changing,
          # matmul_new_group: first gm_id of a multi-gm group =>
          # group spans ≥ 2 gm_ids.
          matmul_new_group,
      ),
      lambda: lax.cond(
          group_is_changing,
          # matmul_group_changing: last gm_id of a multi-gm group.
          matmul_group_changing,
          # matmul: middle gm_id => group spans ≥ 3 gm_ids =>
          # group_size + local_offset > 2*tile_m.
          matmul,
      ),
  )


class TgmmIndexMaps:
  """Index maps for TGMM kernel."""

  def __init__(self, metadata_ref: gmm_v2.MetadataRef, cfgs: gmm_v2.GmmConfigs):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs

  def lhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, k_id)

  def rhs_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return (pl.ds(row_start, row_size), 0, n_id)

  def rhs_scale_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    return (0, 0, n_id)

  def out_index_map(self, n_id: jax.Array, k_id: jax.Array, gm_id: jax.Array):
    group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
    return (group_id, k_id, n_id)


def generate_tgmm_block_specs(
    metadata_ref: gmm_v2.MetadataRef, cfgs: gmm_v2.GmmConfigs
) -> Tuple[Tuple[pl.BlockSpec, OperandRef, pl.BlockSpec | None], pl.BlockSpec]:
  """Generates block specs for the given lhs, rhs, and out refs."""
  index_map = TgmmIndexMaps(metadata_ref, cfgs)
  # NB: in tgmm, LHS is reshaped from (M, K) to (-1, size_lhs_sublane, K) so
  # that DMA transfers are aligned to sublane boundaries. The first dimension
  # after this reshape has size tile_m // size_lhs_sublane — i.e., the number of
  # "sublane-rows" in a tile.
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)
  lhs_block_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
      index_map.lhs_index_map,
  )
  rhs_block_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
      index_map.rhs_index_map,
  )
  rhs_scale_block_spec = None
  if cfgs.rhs_cfgs.has_scale:
    rhs_scale_block_spec = pl.BlockSpec(
        (1, 1, cfgs.tiles.tile_n),
        index_map.rhs_scale_index_map,
    )
  rhs_spec = OperandRef(value=rhs_block_spec, scale=rhs_scale_block_spec)
  out_block_spec = pl.BlockSpec(
      (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
      index_map.out_index_map,
  )
  ps_block_spec = None
  if cfgs.has_partial_sum:
    ps_block_spec = pl.BlockSpec(
        (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )
  in_specs = (lhs_block_spec, rhs_spec, ps_block_spec)
  return in_specs, out_block_spec


def zero_out_start(
    lhs_group_sizes_ref,  # int32[size_lhs_group]
    group_offset_ref,  # int32[1]
    out_ref,  # [num_actual_groups, k, n]
    zero_ref,  # [tile_zero_k, num_lanes]
    semaphore_ref,  # [1]
):
  """If group_sizes[i]==0, kick off async DMAs to zero out drhs[i]."""
  num_actual_groups, aligned_k, aligned_n = out_ref.shape
  tile_zero_k = zero_ref.shape[0]
  zero_ref = zero_ref.reshape(1, tile_zero_k, -1)
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert aligned_n % num_lanes == 0

  zero_ref[...] = jnp.zeros_like(zero_ref)

  def fill_zero(local_group_id, should_copy):
    should_copy_int = should_copy.astype(int)
    for i in range(pl.cdiv(aligned_k, tile_zero_k)):
      size_k_to_copy = min(tile_zero_k, aligned_k - i * tile_zero_k)
      for j in range(aligned_n // num_lanes):
        src = zero_ref.at[pl.ds(0, should_copy_int), pl.ds(0, size_k_to_copy)]
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
    return 1

  num_groups_to_zero = 0
  group_offset = group_offset_ref[0]
  for local_group_id in range(num_actual_groups):
    global_group_id = local_group_id + group_offset
    should_copy = lhs_group_sizes_ref[global_group_id] == 0
    num_groups_to_zero += should_copy.astype(int)
    fill_zero(local_group_id, should_copy)

  return num_groups_to_zero


def zero_out_end(
    num_groups_to_zero,
    out_ref,  # [num_actual_groups, k, n]
    semaphore_ref,  # [1]
):
  """Drain the DMAs started by zero_out_start."""
  dst = out_ref.at[pl.ds(0, num_groups_to_zero),]
  src = dst
  pltpu.make_async_copy(
      src_ref=src,
      dst_ref=dst,
      sem=semaphore_ref.at[0],
  ).wait()


def tgmm_kernel_main(
    lhs_group_sizes_ref,  # int32[size_lhs_group]
    group_offset_ref,  # int32[1]
    lhs_ref,  # [m, k]
    rhs_ref,  # OperandRef: .value [m, n], .scale [1, 1, n] or None
    partial_sum_ref,  # [num_actual_groups, k, n] or None
    out_ref,  # [num_actual_groups, k, n]
    # Scratch memory
    acc_ref: jax.Array,  # [tile_k, tile_n]
    metadata_ref: gmm_v2.MetadataRef,
    zero_ref: jax.Array,  # [tile_zero_k, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    *,
    cfgs,
):
  """Main kernel function for TGMM computation.

  Args:
    lhs_group_sizes_ref: Reference to the group sizes of lhs.
    group_offset_ref: Reference to the group offset.
    lhs_ref: Reference to the LHS array [m, k].
    rhs_ref: OperandRef bundling the RHS array ('.value' [m, n]) and its
      optional per-N scale ('.scale' [1, 1, n], None when there is no scale).
    out_ref: Reference to the output array [num_groups, k, n].
    acc_ref: Scratch memory reference for accumulation [tile_k, tile_n].
    metadata_ref: Reference to the metadata structure.
    zero_ref: Scratch buffer for zeroing empty groups' output.
    semaphore_ref: DMA semaphore for the zeroing copies.
    cfgs: GmmConfigs object containing kernel configurations.
  """
  num_groups_to_zero = zero_out_start(
      lhs_group_sizes_ref,
      group_offset_ref,
      out_ref,
      zero_ref,
      semaphore_ref,
  )

  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.dims.size_n, cfgs.tiles.tile_n)
  num_gm = gmm_v2.fill_metadata(
      lhs_group_sizes_ref,
      group_offset_ref,
      metadata_ref,
      cfgs=cfgs,
  )

  in_specs, out_specs = generate_tgmm_block_specs(metadata_ref, cfgs)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(tgmm_inner_kernel, cfgs=cfgs),
      grid=(num_n, num_k, num_gm),
      in_specs=in_specs,
      out_specs=out_specs,
  )
  lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
  rhs_value = rhs_ref.value
  rhs_in = rhs_value.reshape(-1, cfgs.dims.size_lhs_sublane, rhs_value.shape[-1])
  rhs_operand = OperandRef(value=rhs_in, scale=rhs_ref.scale)
  ps_in = None
  if cfgs.has_partial_sum:
    ps_in = partial_sum_ref
  scratches = [acc_ref, metadata_ref]

  pipeline_fn(lhs_in, rhs_operand, ps_in, out_ref, scratches=scratches)
  zero_out_end(
      num_groups_to_zero,
      out_ref,
      semaphore_ref,
  )


def validate_tgmm_inputs(
    group_sizes: jax.Array,
    num_actual_groups: int,
    group_offset: jax.Array | None = None,
) -> None:
  """Validates inputs to 'tgmm_v2'.

  Call this eagerly before invoking the kernel. It is not jit-safe because it
  concretizes 'group_offset'.

  Args:
    group_sizes: The sizes of each group.
    num_actual_groups: The number of actual groups.
    group_offset: An optional offset for the group indices.
  """
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  if group_sizes.size < group_offset[0] + num_actual_groups:
    raise ValueError(
        f"group_sizes.size ({group_sizes.size}) must be >= group_offset"
        f" ({group_offset[0]}) + num_actual_groups ({num_actual_groups})"
    )


@jax.jit(
    static_argnames=[
        "num_actual_groups",
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
    ],
)
def tgmm_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_m, size_n]
    group_sizes: jax.Array,
    num_actual_groups: int,
    rhs_scale: jax.Array | None = None,  # [1, 1, size_n] (per-N scale)
    partial_sum: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    *,
    tile_info: gmm_v2.TileSizes | TileTgmmFn = calculate_tgmm_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
):
  """Computes a transposed grouped matrix multiplication.

  This kernel computes
  grad_rhs=lhs[sizes[i-1]:sizes[i], :].T @ rhs[sizes[i-1]:sizes[i], :], aka
  grad_rhs = lhs.T @ grad.

  Args:
    lhs: The left-hand side array with shape [size_m, size_k].
    rhs: The right-hand side array with shape [size_m, size_n].
    group_sizes: The group sizes of lhs with shape [size_lhs_group].
    num_actual_groups: The actual number of groups: weight.shape[0].
    rhs_scale: The per-N scale of the rhs.
    group_offset: An optional offset for the group indices.
    tile_info: Specifies the tiling strategy. Can be a `TileSizes` object or a
      function to calculate it.
    vmem_limit_bytes: The VMEM limit in bytes for the kernel.
    precision: Unused. Exists for compatibility reasons.
    preferred_element_type: Optional jnp.dtype for the output matrix.
    acc_dtype: Optional jnp.dtype for the accumulator.

  Returns:
    The result of the transposed grouped matrix multiplication, with shape
    [num_actual_groups, size_k, size_n].
  """
  del precision
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

  # Target VMEM size for the zero-init scratch buffer (zero_ref). The actual
  # allocation may be smaller after capping by size_k and rounding down to a
  # sublane multiple, so this also serves as an upper bound used by
  # calculate_tgmm_tiling when reserving VMEM for zero_ref.
  target_zero_ref_bytes = 2 * 1024 * 1024

  cfgs = make_tgmm_configs(
      lhs,
      rhs,
      rhs_scale,  # pyrefly: ignore[bad-argument-type]
      partial_sum,
      group_sizes,
      num_actual_groups,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      out_dtype=preferred_element_type,  # pyrefly: ignore[bad-argument-type]
      acc_dtype=acc_dtype,
      target_zero_ref_bytes=target_zero_ref_bytes,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles

  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_n = gmm_v2.align_to(dims.size_n, num_lanes)
  # Pad K up to a tile_k multiple so (a) every k-tile written by the matmul
  # stays in-bounds, and (b) the zero-init path can slice in sublane-aligned
  # chunks. tile_k is num_lanes-aligned, which is also sublane-tile-aligned.
  aligned_k = gmm_v2.align_to(dims.size_k, tiles.tile_k)
  out_init = jax.ShapeDtypeStruct((num_actual_groups, aligned_k, aligned_n), cfgs.out_dtype)
  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
  scratch_shapes = [
      # acc_ref
      pltpu.VMEM((tiles.tile_k, tiles.tile_n), cfgs.acc_dtype),
      # metadata_ref
      gmm_v2.MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]

  # Prepare zero initializing the drhs[i, :, :] where the group_size[i] is 0.
  out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
  tile_zero_k = target_zero_ref_bytes // num_lanes // out_bytes
  tile_zero_k = min(tile_zero_k, dims.size_k)
  size_out_sublane = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
  tile_zero_k = (tile_zero_k // size_out_sublane) * size_out_sublane
  assert tile_zero_k > 0
  scratch_shapes += [
      pltpu.VMEM((tile_zero_k, num_lanes), cfgs.out_dtype),
      pltpu.SemaphoreType.DMA((1,)),
  ]

  if rhs_scale is not None:
    rhs_scale = rhs_scale.astype(jnp.float32)
    pad_n = aligned_n - dims.size_n
    if pad_n > 0:
      rhs_scale = jnp.pad(rhs_scale, ((0, 0), (0, 0), (0, pad_n)))
  rhs = OperandRef(value=rhs, scale=rhs_scale)  # pyrefly: ignore[bad-assignment]
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  partial_sum_spec = None
  if partial_sum is not None:
    partial_sum_spec = hbm_spec
  in_specs = [
      hbm_spec,  # lhs
      # the tree.map build a
      # OperandRef(value=hbm_spec, scale=None if scale is None else hbm_spec.
      jax.tree.map(lambda _: hbm_spec, rhs),  # rhs
      partial_sum_spec,
  ]

  raw_out = pl.pallas_call(
      functools.partial(tgmm_kernel_main, cfgs=cfgs),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=in_specs,
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          scratch_shapes=scratch_shapes,  # pyrefly: ignore[bad-argument-type]
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=get_scope_name(cfgs),
      cost_estimate=get_cost_estimate(cfgs),
      # the metadata here is for profiling, debugging, and cost modeling.
      # It does not affect the kernel's computation.
      metadata=gmm_v2.get_metadata(cfgs),
  )(group_sizes, group_offset, lhs, rhs, partial_sum)[:, : dims.size_k, : dims.size_n]

  if partial_sum is not None:
    local_group_sizes = lax.dynamic_slice(group_sizes, (group_offset[0],), (num_actual_groups,))
    empty_mask = (local_group_sizes == 0).reshape(num_actual_groups, 1, 1)
    return jnp.where(empty_mask, partial_sum, raw_out)
  return raw_out
