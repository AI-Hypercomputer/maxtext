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
# https://github.com/openxla/tokamax/blob/3f332fcf85dcb87aab661d00228ed71a09b5fd56/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py
"""TGMM kernel."""

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
        acc += tiled_ps_ref[...].astype(acc.dtype)  # pyrefly: ignore[unsupported-operation]
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


def validate_tgmm_operand_shapes(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
) -> None:
  """Validates the ranks and shared dimension of the TGMM operands.

  Both TGMM entry points contract `lhs[size_m, size_k]` against
  `rhs[size_m, size_n]` over the shared leading `size_m` axis. Neither is
  batched: a rank-3 operand is *not* interpreted as a stack of matrices, and
  passing one used to fall through into Pallas and fail deep inside block-spec
  construction with an error naming none of the offending arguments. Reject it
  here, up front, with a message that names the rank and the shape.

  This is a pure shape/rank check on abstract values, so it is jit-safe and can
  be called from inside a traced function.

  Args:
    lhs: The left-hand side operand, expected to be rank-2 [size_m, size_k].
    rhs: The right-hand side operand, expected to be rank-2 [size_m, size_n].
    group_sizes: The per-group row counts of `lhs`, expected to be rank-1
      [size_lhs_group].

  Raises:
    ValueError: If `lhs` or `rhs` is not rank-2, if `group_sizes` is not an
      integer rank-1 array, if `lhs` and `rhs` disagree on `size_m`, or if any
      operand dimension is non-positive.
  """
  for name, operand, dim_names in (
      ("lhs", lhs, "[size_m, size_k]"),
      ("rhs", rhs, "[size_m, size_n]"),
  ):
    if operand.ndim != 2:
      raise ValueError(
          f"tgmm {name} must be a rank-2 {dim_names} array, but got rank"
          f" {operand.ndim} with shape {tuple(operand.shape)}. Batched (>2D)"
          " operands are not supported; reshape or loop over the leading axes"
          " at the call site."
      )

  if group_sizes.ndim != 1:
    raise ValueError(
        "tgmm group_sizes must be a rank-1 [size_lhs_group] array, but got"
        f" rank {group_sizes.ndim} with shape {tuple(group_sizes.shape)}."
    )

  if lhs.shape[0] != rhs.shape[0]:
    raise ValueError(
        "tgmm lhs and rhs must agree on the contracted size_m dimension, but"
        f" got lhs.shape[0]={lhs.shape[0]} and rhs.shape[0]={rhs.shape[0]}"
        f" (lhs.shape={tuple(lhs.shape)}, rhs.shape={tuple(rhs.shape)})."
    )

  if min(lhs.shape) <= 0 or min(rhs.shape) <= 0:
    raise ValueError(
        "tgmm operands must have positive dimensions, but got"
        f" lhs.shape={tuple(lhs.shape)}, rhs.shape={tuple(rhs.shape)}."
    )

  if group_sizes.shape[0] <= 0:
    raise ValueError("tgmm group_sizes must be non-empty, but got shape" f" {tuple(group_sizes.shape)}.")

  if not jnp.issubdtype(jnp.dtype(group_sizes.dtype), jnp.integer):
    raise ValueError("tgmm group_sizes must have an integer dtype, but got" f" {group_sizes.dtype}.")


def _normalize_group_offset(
    group_offset: Any,
    group_sizes: jax.Array,
    num_actual_groups: int,
) -> jax.Array:
  """Validates and normalizes `group_offset` to an `int32[1]` array."""
  if num_actual_groups <= 0:
    raise ValueError(f"tgmm num_actual_groups must be positive, but got {num_actual_groups}.")
  if group_offset is None:
    offset_arr = jnp.array([0], dtype=jnp.int32)
    offset_concrete = 0
  else:
    if hasattr(group_offset, "ndim"):
      if group_offset.ndim > 1 or (group_offset.ndim == 1 and group_offset.shape != (1,)):
        raise ValueError(
            "tgmm group_offset must be a scalar or a shape (1,) array, but got" f" shape {tuple(group_offset.shape)}."
        )
      if not jnp.issubdtype(jnp.dtype(group_offset.dtype), jnp.integer):
        raise ValueError("tgmm group_offset must have an integer dtype, but got" f" {group_offset.dtype}.")
    elif not isinstance(group_offset, int) or isinstance(group_offset, bool):
      raise ValueError(
          "tgmm group_offset must be an integer scalar or int32[1] array, but" f" got {type(group_offset).__name__}."
      )
    offset_concrete = None
    if not isinstance(group_offset, jax.core.Tracer):
      offset_concrete = int(jnp.asarray(group_offset).reshape(-1)[0])
    offset_arr = jnp.asarray(group_offset, dtype=jnp.int32).reshape((1,))

  if offset_concrete is not None:
    if offset_concrete < 0:
      raise ValueError(f"tgmm group_offset must be non-negative, but got {offset_concrete}.")
    if group_sizes.shape[0] < offset_concrete + num_actual_groups:
      raise ValueError(
          f"group_sizes.size ({group_sizes.shape[0]}) must be >= group_offset"
          f" ({offset_concrete}) + num_actual_groups ({num_actual_groups})"
      )
  return offset_arr


def validate_tgmm_inputs(
    group_sizes: jax.Array,
    num_actual_groups: int,
    group_offset: Any = None,
    *,
    lhs: jax.Array | None = None,
    rhs: jax.Array | None = None,
) -> None:
  """Validates inputs to 'tgmm_v2'.

  Call this eagerly before invoking the kernel. It is not jit-safe because it
  concretizes 'group_offset'.

  Args:
    group_sizes: The sizes of each group.
    num_actual_groups: The number of actual groups.
    group_offset: An optional offset for the group indices.
    lhs: Optional left-hand side operand. When both `lhs` and `rhs` are given,
      their ranks and shared `size_m` dimension are checked as well, via
      `validate_tgmm_operand_shapes`. Omitted by default so that existing
      callers, which pass only the group metadata, are unaffected.
    rhs: Optional right-hand side operand. See `lhs`.

  Raises:
    ValueError: If `group_sizes` is too short for the requested group range,
      or if the operand shape checks fail.
  """
  if (lhs is None) != (rhs is None):
    raise ValueError(
        "validate_tgmm_inputs takes lhs and rhs together or not at all, but"
        f" got lhs={'None' if lhs is None else 'an array'} and"
        f" rhs={'None' if rhs is None else 'an array'}."
    )
  if lhs is not None and rhs is not None:
    validate_tgmm_operand_shapes(lhs, rhs, group_sizes)

  _normalize_group_offset(group_offset, group_sizes, num_actual_groups)


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
    group_offset: Any = None,
    *,
    tile_info: gmm_v2.TileSizes | TileTgmmFn = calculate_tgmm_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: Any = None,
    acc_dtype: Any = None,
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
  validate_tgmm_operand_shapes(lhs, rhs, group_sizes)
  group_offset = _normalize_group_offset(group_offset, group_sizes, num_actual_groups)
  if preferred_element_type is None:
    preferred_element_type = lhs.dtype
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

  size_m = lhs.shape[0]
  sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
  padded_m = gmm_v2.align_to(size_m, sublane_tiling)
  if padded_m != size_m:
    pad = padded_m - size_m
    lhs = jnp.pad(lhs, ((0, pad), (0, 0)))
    rhs = jnp.pad(rhs, ((0, pad), (0, 0)))

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

  input_output_aliases = {}
  if partial_sum is not None:
    flat_args_preceding = (group_sizes, group_offset, lhs, rhs)
    leaves = jax.tree_util.tree_leaves(flat_args_preceding)
    partial_sum_idx = sum(1 for x in leaves if x is not None)
    input_output_aliases = {partial_sum_idx: 0}

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
      metadata=gmm_v2.get_metadata(cfgs),  # pyrefly: ignore[bad-argument-type]
      input_output_aliases=input_output_aliases,
  )(group_sizes, group_offset, lhs, rhs, partial_sum)[:, : dims.size_k, : dims.size_n]

  if partial_sum is not None:
    local_group_sizes = lax.dynamic_slice(group_sizes, (group_offset[0],), (num_actual_groups,))
    empty_mask = (local_group_sizes == 0).reshape(num_actual_groups, 1, 1)
    return jnp.where(empty_mask, partial_sum, raw_out)
  return raw_out


# =============================================================================
# Spatial-minor TGMM.
# =============================================================================
#
# The standard `tgmm_v2` above emits `[g, k, n]` in the default (row-major)
# layout, i.e. physically `g` majormost, `k` on sublanes and `n` on lanes.
# Downstream consumers (collectives, optimiser state, fused epilogues) sometimes
# want the *spatial* feature axis on sublanes and the group axis minormost
# instead, which otherwise forces XLA to materialise a separate full-tensor
# relayout copy.
#
# The spatial-minor Pallas kernel below produces that layout natively. It is
# reached through `tgmm_gnk_v2` (layout-dispatched, defaults to the fast
# `tgmm_v2` fallback) or `tgmm_spatial_minor_v2` (the same function, defaulted
# to this kernel). Either way the result is a logical `[g, n, k]` array; this
# kernel's is the one whose XLA layout is `{0, 1, 2}`, i.e. physically:
#
#     k (majormost)  ->  n (sublane, 8-aligned)  ->  g (lane, 128-aligned)
#
# which is byte-identical to a default-layout `[k, n, g]` array. The kernel
# therefore writes a plain `[k, n, g]` Pallas output and finishes with a
# `jnp.transpose(..., (2, 1, 0))` that XLA folds into a free bitcast once the
# consumer pins the `{0, 1, 2}` layout.
#
# Note `result[g, n, k] == dW_g[k, n]`, i.e. this is the *transposed* per-expert
# gradient. The un-transposed spatial-minor variant (`[g, k, n]` with `k` on
# sublanes) is obtained by simply swapping the operands, since
# `tgmm(lhs, rhs)[g] == tgmm(rhs, lhs)[g].T`.
#
# Two-phase pipeline:
#
#   1. MXU per-group matmul into VMEM stage buffer:
#      `stage[bk/128, Gp, bn, 128]` in bf16 (or staged in-place inside `out_ref`
#      when `num_m_tiles == 1` and `G == Gp == 128`), accumulated in f32 with a
#      direct-to-stage fast path for single-chunk groups.
#   2. On-chip XLU transpose epilogue on the last m step:
#      Contiguous VREG 3-step transpose `(1, 0, 2) -> (0, 2, 1) -> (1, 0, 2)`
#      on `(128, 8, 128)` uint32 blocks across dual XLUs, writing to the
#      double-buffered output window `[bk, bn, Gp]`.

# `major_to_minor` permutations for the logical `[g, n, k]` result, as accepted
# by `jax.experimental.layout.Layout(major_to_minor=...)`.
#
# XLA prints layouts as `minor_to_major`, i.e. the reverse of `major_to_minor`
# (xla_data.proto LayoutProto: "from minor (fastest varying index) to major"),
# so the printed string is the tuple read backwards.

# `k` majormost, `n` on sublanes, `g` minormost on lanes. Prints as `{0,1,2}`.
# This is the byte order `tgmm_spatial_minor_v2`'s Pallas kernel emits natively,
# and the *only* layout for which running that kernel is worth its cost.
SPATIAL_MINOR_MAJOR_TO_MINOR = (2, 1, 0)

# JAX's default row-major layout for a rank-3 array. Prints as `{2,1,0}`.
DEFAULT_MAJOR_TO_MINOR = (0, 1, 2)


def get_spatial_minor_scope_name(cfgs: gmm_v2.GmmConfigs, bg: int) -> str:
  dims = cfgs.dims
  tiles = cfgs.tiles
  return (
      f"tgmm_spatial_minor_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}-bg_{bg}"
  )


def get_spatial_minor_cost_estimate(cfgs: gmm_v2.GmmConfigs) -> pl.CostEstimate:
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


def calculate_tgmm_spatial_minor_tiling(
    dims: gmm_v2.Dimensions,
    lhs_cfgs: gmm_v2.InputConfigs,
    rhs_cfgs: gmm_v2.InputConfigs,
    vmem_limit_bytes: int,
    out_dtype: jnp.dtype,
) -> gmm_v2.TileSizes:
  """Calculate optimal tile sizes for spatial minor TGMM kernel.

  Estimates the scoped VMEM the kernel allocates. `G` is `dims.size_group`.
  `Gp` is the group count the kernel emits: `G` aligned to 8, or to `num_lanes` when
  `G > num_lanes`. `lanes_g` is `Gp` aligned to `num_lanes`, the lane footprint
  of the output window.
    - lhs block:  2 * tile_m * tile_k * lhs_bytes (double buffered)
    - rhs block:  2 * tile_m * tile_n * rhs_bytes (double buffered)
    - out window: 2 * tile_k * tile_n * lanes_g * out_bytes (double buffered)
    - stage:      tile_k * tile_n * Gp * out_bytes, omitted when the stage is
                  held in place in `out_ref` (num_m_tiles == 1 and
                  G == Gp == num_lanes)
    - acc:        tile_n * tile_k * 4 (always float32)

  Tile size choices:
    - tile_m: coarse TM (e.g. 2048 or 4096 covering M where possible).
    - tile_k: multiple of 128 (e.g. 1024, 512, 256, 128).
    - tile_n: multiple of 16 (e.g. 128).

  Args:
    dims: Problem dimensions.
    lhs_cfgs: Input configuration for the left-hand side operand.
    rhs_cfgs: Input configuration for the right-hand side operand.
    vmem_limit_bytes: VMEM budget to fit the tiling within.
    out_dtype: Element type of the kernel output and of the stage.

  Returns:
    The selected tile sizes (tile_m, tile_k, tile_n).
  """
  num_lanes = pltpu.get_tpu_info().num_lanes
  if dims.size_group <= num_lanes:
    lanes_g = num_lanes
    stage_g = max(8, gmm_v2.align_to(dims.size_group, 8))
  else:
    lanes_g = gmm_v2.align_to(dims.size_group, num_lanes)
    stage_g = lanes_g

  # The kernel accumulates in float32; acc_dtype only reaches tgmm_v2.
  acc_bytes = jnp.dtype(jnp.float32).itemsize
  out_bytes = jnp.dtype(out_dtype).itemsize
  lhs_bytes = jnp.dtype(lhs_cfgs.dtype).itemsize
  rhs_bytes = jnp.dtype(rhs_cfgs.dtype).itemsize

  def budget(tile_m: int, tile_k: int, tile_n: int) -> int:
    inplace_stage = (tile_m >= dims.size_m) and (lanes_g == num_lanes) and (dims.size_group == lanes_g)
    stage_buffers = 0 if inplace_stage else 1
    return (
        # lhs block: 2 * TM * tile_k * lhs_bytes (double buffered)
        2 * tile_m * tile_k * lhs_bytes
        # rhs block: 2 * TM * tile_n * rhs_bytes (double buffered)
        + 2 * tile_m * tile_n * rhs_bytes
        # out window: 2 * tile_k * tile_n * lanes_g * out_bytes (double buffer)
        + 2 * tile_k * tile_n * lanes_g * out_bytes
        # stage: omitted when held in place in out_ref
        + stage_buffers * tile_k * tile_n * stage_g * out_bytes
        # acc: tile_n * tile_k * acc_bytes (f32)
        + tile_n * tile_k * acc_bytes
    )

  aligned_k = gmm_v2.align_to(dims.size_k, 128)
  aligned_n = gmm_v2.align_to(dims.size_n, 16)
  tile_n = min(128, max(16, aligned_n))

  # Coarse TM: 4096 or 2048 covering M where possible, with fallbacks if needed.
  if dims.size_m > 2048:
    m_candidates = (4096, 2048, 1024, 512, 256, 128)
  else:
    m_candidates = (2048, 1024, 512, 256, 128)

  seen_k = set()
  for cand_k in (1024, 512, 256, 128):
    tile_k = min(cand_k, aligned_k)
    if tile_k in seen_k:
      continue
    seen_k.add(tile_k)
    for tile_m in m_candidates:
      if budget(tile_m, tile_k, tile_n) <= vmem_limit_bytes:
        return gmm_v2.TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

  raise ValueError(
      "Could not find valid tile sizes for tgmm_spatial_minor."
      f" dims={dims}, {lanes_g=}, vmem={vmem_limit_bytes},"
      f" smallest budget={budget(128, min(128, aligned_k), tile_n)}"
  )


def _sm_kernel(
    m_bounds: jax.Array,  # int32[G + 1], SMEM scalar prefetch
    g_first: jax.Array,  # int32[num_m_tiles], SMEM scalar prefetch
    g_last: jax.Array,  # int32[num_m_tiles], SMEM scalar prefetch
    lhs_ref: Any,  # [TM, bk], VMEM block
    rhs_ref: Any,  # [TM, bn], VMEM block
    out_ref: Any,  # [bk, bn, Gp], VMEM block
    acc_ref: Any,  # [bn, bk], VMEM scratch (float32)
    stage_ref: Any = None,  # [bk // 128, Gp, bn, 128] or None (in-place)
    *,
    G: int,
    Gp: int,
    bk: int,
    bn: int,
    TM: int,
    tm: int,
):
  """Kernel for spatial minor TGMM with on-chip XLU transpose epilogue."""
  k_id, n_id, mt = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  tile_lo = mt * TM
  tile_hi = tile_lo + TM
  if stage_ref is None:
    stage_ref = out_ref.reshape(bk // 128, Gp, bn, 128)

  @pl.when((k_id == 0) & (n_id == 0) & (mt == 0))
  def _zero_padding_groups():
    if Gp > G:
      stage_ref[...] = jnp.zeros(stage_ref.shape, dtype=stage_ref.dtype)

  # Phase 1: MXU per-group matmul into the stage.
  def group_body(g, carry):
    g_lo = m_bounds[g]
    g_hi = m_bounds[g + 1]

    lo = jnp.maximum(g_lo, tile_lo) - tile_lo
    hi = jnp.minimum(g_hi, tile_hi) - tile_lo
    c_start = lo // tm
    c_end = pl.cdiv(hi, tm)

    def _run_multi_chunk():
      @pl.when(g_lo == g_hi)
      def _zero_empty_group():
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=acc_ref.dtype)

      def chunk_body(c, chunk_carry):
        row = pl.multiple_of(c * tm, tm)
        x = lhs_ref[pl.ds(row, tm), :]
        y = rhs_ref[pl.ds(row, tm), :]
        r = row + lax.broadcasted_iota(jnp.int32, (tm, 1), 0)
        keep = (r >= lo) & (r < hi)
        y = jnp.where(keep, y, 0)
        dot = lax.dot_general(
            y,
            x,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )  # [bn, bk]

        @pl.when((c == c_start) & (g_lo >= tile_lo))
        def _init_acc():
          acc_ref[...] = dot

        @pl.when((c > c_start) | (g_lo < tile_lo))
        def _accum_acc():
          acc_ref[...] = acc_ref[...] + dot

        return chunk_carry

      @pl.when(hi > lo)
      def _compute_chunks():
        lax.fori_loop(c_start, c_end, chunk_body, None)

      @pl.when(g_hi <= tile_hi)  # group ends in this m tile
      def _store_stage():
        acc = acc_ref[...].astype(stage_ref.dtype)
        for kc in range(bk // 128):  # static, aligned stores
          stage_ref[kc, g, :, :] = acc[:, kc * 128 : (kc + 1) * 128]

    if tm < 128:
      is_single_chunk = (g_lo >= tile_lo) & (g_hi <= tile_hi) & (c_end - c_start == 1)

      @pl.when(is_single_chunk)
      def _single_chunk_group():
        row = pl.multiple_of(c_start * tm, tm)
        x = lhs_ref[pl.ds(row, tm), :]
        y = rhs_ref[pl.ds(row, tm), :]
        r = row + lax.broadcasted_iota(jnp.int32, (tm, 1), 0)
        keep = (r >= lo) & (r < hi)
        y = jnp.where(keep, y, 0)
        dot = lax.dot_general(
            y,
            x,
            (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(stage_ref.dtype)
        for kc in range(bk // 128):
          stage_ref[kc, g, :, :] = dot[:, kc * 128 : (kc + 1) * 128]

      @pl.when(~is_single_chunk)
      def _multi_chunk_or_empty_group():
        _run_multi_chunk()

    else:
      _run_multi_chunk()

    return carry

  lax.fori_loop(g_first[mt], g_last[mt] + 1, group_body, None)

  # Phase 2: XLU transpose, once per output tile on the last m step.
  @pl.when(mt == pl.num_programs(2) - 1)
  def _epilogue():
    num_g_chunks = pl.cdiv(Gp, 128)
    if stage_ref.dtype == jnp.bfloat16:
      half = bn // 2
      stage_u32 = stage_ref.bitcast(jnp.uint32)  # [bk/128, Gp, half, 128]
      out_u32 = out_ref.bitcast(jnp.uint32)  # [bk, half, Gp]
      for gc in range(num_g_chunks):
        g_slice = slice(None) if num_g_chunks == 1 else slice(gc * 128, (gc + 1) * 128)
        for kc in range(bk // 128):
          k_slice = slice(kc * 128, (kc + 1) * 128)
          for p in range(half // 8):
            p_slice = slice(p * 8, (p + 1) * 8)
            v = stage_u32[kc, g_slice, p_slice, :]
            v1 = jnp.transpose(v, (1, 0, 2))
            v2 = jnp.transpose(v1, (0, 2, 1))
            v3 = jnp.transpose(v2, (1, 0, 2))
            out_u32[k_slice, p_slice, g_slice] = v3
    else:
      for gc in range(num_g_chunks):
        g_slice = slice(None) if num_g_chunks == 1 else slice(gc * 128, (gc + 1) * 128)
        for kc in range(bk // 128):
          k_slice = slice(kc * 128, (kc + 1) * 128)
          for p in range(bn // 8):
            p_slice = slice(p * 8, (p + 1) * 8)
            v = stage_ref[kc, g_slice, p_slice, :]
            v1 = jnp.transpose(v, (1, 0, 2))
            v2 = jnp.transpose(v1, (0, 2, 1))
            v3 = jnp.transpose(v2, (1, 0, 2))
            out_ref[k_slice, p_slice, g_slice] = v3


def _tgmm_v2_transposed(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    num_actual_groups: int,
    group_offset: jax.Array,
    *,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype,
    acc_dtype: jnp.dtype,
) -> jax.Array:
  """Computes the `[g, n, k]` TGMM result using the fast `tgmm_v2` kernel.

  `tgmm_v2` emits `[g, k, n]`; the trailing `jnp.transpose(..., (0, 2, 1))` is
  a logical relabelling. Whether it costs anything depends on the layout pinned
  on the result: for `{1, 2, 0}` (`major_to_minor=(0, 2, 1)`) XLA folds it into
  a zero-copy bitcast, while for `{0, 1, 2}` or `{1, 0, 2}` it becomes a
  relayout copy that still runs faster than the spatial minor kernel.

  Args:
    lhs: The left-hand side array with shape [size_m, size_k].
    rhs: The right-hand side array with shape [size_m, size_n].
    group_sizes: The group sizes of lhs with shape [size_lhs_group].
    num_actual_groups: The actual number of groups.
    group_offset: Offset for the group indices, already normalised to a
      one-element array.
    vmem_limit_bytes: VMEM limit to forward, or None to let `tgmm_v2` choose its
      own default. The spatial minor arm resolves its default only after
      dispatch, so that default never reaches this function.
    out_dtype: Resolved output dtype, passed through so that both dispatch arms
      return the identical dtype.
    acc_dtype: Resolved accumulator dtype for `tgmm_v2`. Only this arm honours
      it; the spatial minor kernel always accumulates in float32.

  Returns:
    The transposed grouped matmul result with shape
    `[num_actual_groups, size_n, size_k]`.
  """
  out_gkn = tgmm_v2(
      lhs,
      rhs,
      group_sizes,
      num_actual_groups=num_actual_groups,
      group_offset=group_offset,
      vmem_limit_bytes=vmem_limit_bytes,
      preferred_element_type=out_dtype,
      acc_dtype=acc_dtype,
  )
  return jnp.transpose(out_gkn, (0, 2, 1))


@jax.jit(
    static_argnames=[
        "num_actual_groups",
        "out_major_to_minor",
        "tile_info",
        "vmem_limit_bytes",
        "preferred_element_type",
        "acc_dtype",
        "bg",
    ],
)
def tgmm_gnk_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_m, size_n]
    group_sizes: jax.Array,
    num_actual_groups: int,
    group_offset: Any = None,
    *,
    out_major_to_minor: Tuple[int, ...] = DEFAULT_MAJOR_TO_MINOR,
    tile_info: gmm_v2.TileSizes | None = None,
    vmem_limit_bytes: int | None = None,
    preferred_element_type: Any = None,
    acc_dtype: Any = None,
    bg: int | None = None,
):
  """Transposed grouped matmul, layout-dispatched over two implementations.

  Computes the same per-group products as `tgmm_v2`, i.e.
  `dW_g = lhs[group g].T @ rhs[group g]` with shape `[size_k, size_n]`, and
  returns them as a logical `[num_actual_groups, size_n, size_k]` array with
  `result[g, n, k] == dW_g[k, n]`.

  The name is deliberately layout-neutral: this function returns `[g, n, k]`
  whichever implementation it picks. If you specifically want the
  spatial-minor kernel, call `tgmm_spatial_minor_v2`, which is this function
  with `out_major_to_minor` defaulted to `SPATIAL_MINOR_MAJOR_TO_MINOR`.

  ## Choosing an implementation: `out_major_to_minor`

  `out_major_to_minor` is the `major_to_minor` permutation the caller intends
  to pin on the *result*, as accepted by
  `jax.experimental.layout.Layout(major_to_minor=...)`. Pass the same tuple you
  pass to `out_shardings`:

      fmt = Format(Layout(major_to_minor=tgmm.SPATIAL_MINOR_MAJOR_TO_MINOR),
                   sharding)
      f = jax.jit(lambda *a: tgmm.tgmm_gnk_v2(
                      *a, out_major_to_minor=tgmm.SPATIAL_MINOR_MAJOR_TO_MINOR),
                  out_shardings=fmt)

  Dispatch is a single equality test:

    * `SPATIAL_MINOR_MAJOR_TO_MINOR == (2, 1, 0)`, which XLA prints as
      `{0,1,2}` (`k` majormost, `n` on sublanes, `g` minormost on lanes),
      runs the native spatial-minor Pallas kernel below.
    * Anything else, including the default `DEFAULT_MAJOR_TO_MINOR ==
      (0, 1, 2)`, runs `tgmm_v2` followed by a logical transpose.

  Both arms return the identical logical array, with the identical dtype, to
  within bf16/f32 rounding. Only the emitted HLO differs.

  This parameter is a *kernel selection hint only*: it does not itself pin the
  layout, and cannot. It has to be supplied explicitly because a traced
  function has no way to observe the layout its caller will request.
  `out_shardings` layouts are applied by pjit at lowering time, after the
  jaxpr is complete, so nothing is visible to a tracer. (`ShapedArray.layout`
  exists but is pinned to `AutoLayout` under a normal `jax.jit`, and
  `with_layout_constraint`'s abstract eval returns its input aval unchanged,
  so neither offers a read path.) An API that appeared to detect the layout
  automatically would be lying, so this one asks.

  Pin the layout on the *outermost* `jax.jit`. PJRT reads the result layout
  mode only from the `main` function, so an `out_shardings` on an inner jit is
  silently dropped. Combined with an
  `out_major_to_minor=SPATIAL_MINOR_MAJOR_TO_MINOR` hint, this would give you
  the slow kernel and an XLA relayout copy.

  ## Performance and memory trade-offs

  On Ghostfish (TPU7x) at `m=4096, k=7168, n=2048, bf16`, the native
  spatial-minor kernel runs in 8.74 ms at `g=128` (vs 6.16 ms for `tgmm_v2`
  plus a standalone relayout copy) and in 3.27 ms at `g=16` (vs 2.92 ms),
  while cutting peak HBM in half at `g=128` (3.83 GB vs 7.59 GB) and reducing
  peak HBM at `g=16` (3.83 GB vs 4.30 GB) because no intermediate `[g, k, n]`
  buffer is materialized in HBM.

  `tgmm_gnk_v2` defaults to `DEFAULT_MAJOR_TO_MINOR` so callers that want
  standard layouts route to `tgmm_v2`, while callers that pin `{0, 1, 2}` pass
  `out_major_to_minor=SPATIAL_MINOR_MAJOR_TO_MINOR` (or call
  `tgmm_spatial_minor_v2` directly).

  ## Parameters `tgmm_v2` has and this does not

  `tgmm_v2` also takes `rhs_scale`, `partial_sum` and `precision`. They are
  deliberately *not* offered here rather than being accepted and dropped.
  `precision` is a documented no-op in `tgmm_v2` (both kernels hardcode an f32
  MXU accumulation); `rhs_scale` and `partial_sum` are honoured by `tgmm_v2`
  but have no implementation on the spatial-minor arm, whose accumulator is
  `[tile_n, tile_k]` and whose output staging is transposed. Accepting them
  here would work on the fallback arm and silently do nothing on the arm this
  function is named for, which is exactly the failure mode the explicit
  `out_major_to_minor` dispatch exists to avoid. Call `tgmm_v2` directly if you
  need them.

  Args:
    lhs: The left-hand side array with shape [size_m, size_k].
    rhs: The right-hand side array with shape [size_m, size_n].
    group_sizes: The group sizes of lhs with shape [size_lhs_group]. These
      partition `lhs`/`rhs` rows in order, and are indexed *globally*: entries
      before `group_offset` still advance the row cursor even though they are
      not computed.
    num_actual_groups: The actual number of groups: weight.shape[0]. This is the
      size of the returned group axis, and the width of the window of
      `group_sizes` that is computed.
    group_offset: Index of the first group to compute, as a one-element int32
      array (`jnp.array([q])`); `None` means zero. Groups `group_sizes[q : q +
      num_actual_groups]` are computed, reading the rows they own, and the
      result is written *locally*: `out[i]` holds the product for global group
      `q + i`. It is the caller's responsibility to ensure `group_sizes.size >=
      q + num_actual_groups`; `validate_tgmm_inputs` checks that eagerly,
      outside `jit`, because `q` is a traced value here.
    out_major_to_minor: The `major_to_minor` layout the caller intends to pin on
      the result; selects the implementation. See above.
    tile_info: Optional explicit `TileSizes`, for the spatial-minor kernel only;
      defaults to a VMEM-budgeted heuristic. Unlike `tgmm_v2`'s parameter of the
      same name this must be a concrete `TileSizes`, not a callable. Supplying
      it together with a non-spatial-minor `out_major_to_minor` is an error; see
      Raises.
    vmem_limit_bytes: The VMEM limit in bytes for the kernel. When None, each
      arm resolves it to `int(vmem_capacity_bytes * 0.9)` on its own. Honoured
      on both paths.
    preferred_element_type: Optional jnp.dtype for the output matrix. Defaults
      to `lhs.dtype` on both paths. Note this also sets the `bg` alignment
      granularity on the spatial-minor path, via `32 // itemsize`.
    acc_dtype: Optional dtype of the cross tile accumulator on the `tgmm_v2`
      fallback. Defaults to float32. It never changes the MXU accumulation type,
      which is always f32. The spatial minor kernel ignores it: its accumulator
      is always float32 and its stage is held in the output dtype.
    bg: Group block size, accepted for the spatial minor kernel only. It is
      validated to be a positive multiple of `32 // itemsize` of the output
      dtype, so that it covers whole 32 byte words. The kernel always stages all
      groups of the output window, so `bg` does not change the tiling or the
      result. Supplying it together with a non spatial minor
      `out_major_to_minor` is an error; see Raises.

  Returns:
    The transposed grouped matmul result with shape
    `[num_actual_groups, size_n, size_k]`.

  Raises:
    ValueError: If `out_major_to_minor` is not a permutation of (0, 1, 2); if
      the operand shapes fail `validate_tgmm_operand_shapes`; if `group_sizes`
      is shorter than `num_actual_groups`; if a spatial-minor-only argument
      (`tile_info`, `bg`) is supplied alongside a non-spatial-minor
      `out_major_to_minor`; or if `bg` is not a multiple of the output dtype's
      32-byte word.
  """
  return tgmm_spatial_minor_v2(
      lhs,
      rhs,
      group_sizes,
      num_actual_groups,
      group_offset,
      out_major_to_minor=out_major_to_minor,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      preferred_element_type=preferred_element_type,
      acc_dtype=acc_dtype,
      bg=bg,
  )


@jax.jit(
    static_argnames=[
        "num_actual_groups",
        "out_major_to_minor",
        "tile_info",
        "vmem_limit_bytes",
        "preferred_element_type",
        "acc_dtype",
        "bg",
    ]
)
def tgmm_spatial_minor_v2(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_m, size_n]
    group_sizes: jax.Array,
    num_actual_groups: int,
    group_offset: Any = None,
    *,
    out_major_to_minor: Tuple[int, ...] = SPATIAL_MINOR_MAJOR_TO_MINOR,
    tile_info: gmm_v2.TileSizes | None = None,
    vmem_limit_bytes: int | None = None,
    preferred_element_type: Any = None,
    acc_dtype: Any = None,
    bg: int | None = None,
):
  """Computes transposed grouped matmul emitting spatial minor layout [k, n, g].

  This function emits the native `{0,1,2}` (spatial-minor: `[k, n, g]`) byte
  order directly from the Pallas kernel. If `out_major_to_minor` is set to any
  other layout, it routes to `tgmm_v2` followed by a logical transpose.

  Args:
    lhs: The left-hand side array with shape [size_m, size_k].
    rhs: The right-hand side array with shape [size_m, size_n].
    group_sizes: The group sizes of lhs with shape [size_lhs_group].
    num_actual_groups: The actual number of groups: weight.shape[0].
    group_offset: Index of the first group to compute. See `tgmm_gnk_v2`.
    out_major_to_minor: The `major_to_minor` layout the caller intends to pin on
      the result; selects the implementation. Defaults to
      `SPATIAL_MINOR_MAJOR_TO_MINOR`.
    tile_info: Optional explicit `TileSizes` for the spatial-minor kernel.
    vmem_limit_bytes: The VMEM limit in bytes for the kernel.
    preferred_element_type: Optional jnp.dtype for the output matrix.
    acc_dtype: Optional accumulator dtype, honoured by the `tgmm_v2` fallback
      only. The spatial minor kernel always accumulates in float32 and holds its
      stage in the output dtype.
    bg: Group block size for the spatial minor kernel. It is validated to be a
      positive multiple of `32 // itemsize` of the output dtype (32 byte output
      alignment), but the kernel always stages all groups of the output window,
      so it does not change the tiling. See `tgmm_gnk_v2`.

  Returns:
    The transposed grouped matmul result with shape
    `[num_actual_groups, size_n, size_k]`.
  """
  validate_tgmm_operand_shapes(lhs, rhs, group_sizes)

  out_major_to_minor = tuple(out_major_to_minor)
  if sorted(out_major_to_minor) != [0, 1, 2]:
    raise ValueError(
        "out_major_to_minor must be a permutation of (0, 1, 2) describing the"
        " layout of the rank-3 [g, n, k] result, but got"
        f" {out_major_to_minor}."
    )

  if group_sizes.shape[0] < num_actual_groups:
    raise ValueError(
        f"group_sizes has {group_sizes.shape[0]} entries, which is fewer than"
        f" num_actual_groups={num_actual_groups}. group_sizes must cover the"
        " whole requested group window"
        " [group_offset, group_offset + num_actual_groups). Call"
        " validate_tgmm_inputs eagerly, outside jit, to check the"
        " group_offset-dependent bound too."
    )

  group_offset = _normalize_group_offset(group_offset, group_sizes, num_actual_groups)

  size_m, size_k = lhs.shape
  _, size_n = rhs.shape

  out_dtype = jnp.dtype(preferred_element_type if preferred_element_type is not None else lhs.dtype)
  acc_dtype = jnp.dtype(acc_dtype if acc_dtype is not None else jnp.float32)

  size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
  size_rhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(rhs.dtype)
  assert size_lhs_sublane == size_rhs_sublane, (
      f"size_lhs_sublane should be the same as size_rhs_sublane {lhs.dtype=}," f" {rhs.dtype=}"
  )

  if out_major_to_minor != SPATIAL_MINOR_MAJOR_TO_MINOR:
    supplied = [name for name, value in (("bg", bg), ("tile_info", tile_info)) if value is not None]
    if supplied:
      raise ValueError(
          f"{', '.join(supplied)} configure the spatial-minor kernel only, but"
          f" out_major_to_minor={out_major_to_minor} selects the tgmm_v2"
          " fallback, which cannot honour them. Either pass"
          " out_major_to_minor=SPATIAL_MINOR_MAJOR_TO_MINOR"
          f" ({SPATIAL_MINOR_MAJOR_TO_MINOR}) to run the kernel they configure,"
          f" or drop {' and '.join(supplied)}. To tune the fallback, call"
          " tgmm_v2 directly: it takes its own tile_info."
      )
    padded_m = gmm_v2.align_to(size_m, size_lhs_sublane)
    if padded_m != size_m:
      pad = padded_m - size_m
      lhs = jnp.pad(lhs, ((0, pad), (0, 0)))
      rhs = jnp.pad(rhs, ((0, pad), (0, 0)))
    return _tgmm_v2_transposed(
        lhs,
        rhs,
        group_sizes,
        num_actual_groups,
        group_offset,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
    )

  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

  # pylint: disable=invalid-name
  G = num_actual_groups
  num_lanes = pltpu.get_tpu_info().num_lanes
  Gp = max(8, gmm_v2.align_to(G, 8)) if G <= num_lanes else gmm_v2.align_to(G, num_lanes)

  if bg is not None:
    word_bytes = 32
    align_g = max(1, word_bytes // jnp.dtype(out_dtype).itemsize)
    if bg <= 0:
      raise ValueError(f"bg={bg} must be positive.")
    if bg % align_g != 0:
      raise ValueError(
          f"bg={bg} must be a multiple of {align_g} so that"
          f" output-window DMA offsets are {word_bytes}-byte aligned for"
          f" out_dtype={out_dtype}."
      )

  # Global row bounds of the G computed groups: skipped groups still own rows.
  all_ends = jnp.cumsum(group_sizes.astype(jnp.int32))
  all_starts = all_ends - group_sizes.astype(jnp.int32)
  off = group_offset[0]
  starts = lax.dynamic_slice(all_starts, (off,), (G,))
  ends = lax.dynamic_slice(all_ends, (off,), (G,))
  m_bounds = jnp.concatenate([starts, ends[-1:]])  # int32[G + 1], contiguous

  dims = gmm_v2.Dimensions(
      size_m=size_m,
      size_k=size_k,
      size_n=size_n,
      size_group=num_actual_groups,
      size_lhs_group=group_sizes.shape[0],
      size_lhs_sublane=size_lhs_sublane,
  )
  lhs_cfgs = gmm_v2.InputConfigs(quant_dtype=None, quant_block_size=-1, dtype=lhs.dtype)
  rhs_cfgs = gmm_v2.InputConfigs(quant_dtype=None, quant_block_size=-1, dtype=rhs.dtype)

  if tile_info is not None:
    if not isinstance(tile_info, gmm_v2.TileSizes):
      raise TypeError("tile_info must be a concrete gmm_v2.TileSizes instance, but got" f" {type(tile_info).__name__}.")
    if tile_info.tile_k % 128 != 0:
      raise ValueError(f"tile_info.tile_k ({tile_info.tile_k}) must be a multiple of 128.")
    if tile_info.tile_n % 16 != 0:
      raise ValueError(f"tile_info.tile_n ({tile_info.tile_n}) must be a multiple of 16.")
    if tile_info.tile_m % size_lhs_sublane != 0:
      raise ValueError(
          f"tile_info.tile_m ({tile_info.tile_m}) must be a multiple of" f" sublane tiling ({size_lhs_sublane})."
      )
    tiles = tile_info
  else:
    tiles = calculate_tgmm_spatial_minor_tiling(
        dims,
        lhs_cfgs,
        rhs_cfgs,
        vmem_limit_bytes,
        out_dtype,
    )

  TM = tiles.tile_m
  bk = tiles.tile_k
  bn = tiles.tile_n
  avg_group_m = max(
      size_lhs_sublane,
      gmm_v2.align_to(pl.cdiv(size_m, num_actual_groups), size_lhs_sublane),
  )
  tm = min(128, TM, max(32, avg_group_m))

  num_m_tiles = pl.cdiv(size_m, TM)
  num_k = gmm_v2.align_to(size_k, bk) // bk
  num_n = gmm_v2.align_to(size_n, bn) // bn

  Kp = num_k * bk
  Np = num_n * bn
  padded_m = num_m_tiles * TM

  if padded_m != size_m:
    pad_m = padded_m - size_m
    lhs = jnp.pad(lhs, ((0, pad_m), (0, 0)))
    rhs = jnp.pad(rhs, ((0, pad_m), (0, 0)))

  if Kp != size_k:
    lhs = jnp.pad(lhs, ((0, 0), (0, Kp - size_k)))

  if Np != size_n:
    rhs = jnp.pad(rhs, ((0, 0), (0, Np - size_n)))

  # Group range each m tile visits. Each group, empty ones included, is
  # finalized by exactly one tile, so every stage slot is written.
  last_touch = jnp.maximum(ends - 1, starts)  # non-decreasing
  tile_lo = jnp.arange(num_m_tiles, dtype=jnp.int32) * TM
  g_first = jnp.searchsorted(last_touch, tile_lo, side="left")
  g_last = jnp.searchsorted(starts, tile_lo + TM, side="left") - 1
  g_last = g_last.at[-1].set(G - 1)

  cfgs = gmm_v2.GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_cfgs=lhs_cfgs,
      rhs_cfgs=rhs_cfgs,
      has_partial_sum=False,
      out_dtype=out_dtype,
      acc_dtype=acc_dtype,
      zero_init=False,
      fuse_act=None,
  )

  use_inplace_out_stage = (num_m_tiles == 1) and (Gp == num_lanes) and (G == Gp)
  scratch_shapes = [pltpu.VMEM((bn, bk), jnp.float32)]  # acc
  if not use_inplace_out_stage:
    scratch_shapes.append(pltpu.VMEM((bk // 128, Gp, bn, 128), out_dtype))  # stage

  out_kng = pl.pallas_call(
      functools.partial(
          _sm_kernel,
          G=G,
          Gp=Gp,
          bk=bk,
          bn=bn,
          TM=TM,
          tm=tm,
      ),
      out_shape=jax.ShapeDtypeStruct((Kp, Np, Gp), out_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,  # m_bounds, g_first, g_last
          grid=(num_k, num_n, num_m_tiles),
          in_specs=[
              pl.BlockSpec((TM, bk), lambda k, n, mt, *_: (mt, k)),
              pl.BlockSpec((TM, bn), lambda k, n, mt, *_: (mt, n)),
          ],
          # Ignores mt, so each output tile is written back once, double
          # buffered: the writeback of tile t overlaps compute of tile t+1.
          out_specs=pl.BlockSpec((bk, bn, Gp), lambda k, n, mt, *_: (k, n, 0)),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
          vmem_limit_bytes=vmem_limit_bytes,
      ),
      name=get_spatial_minor_scope_name(cfgs, Gp),
      cost_estimate=get_spatial_minor_cost_estimate(cfgs),
      metadata=gmm_v2.get_metadata(cfgs),  # pyrefly: ignore[bad-argument-type]
  )(m_bounds, g_first, g_last, lhs, rhs)

  if out_kng.shape != (size_k, size_n, num_actual_groups):
    out_kng = out_kng[:size_k, :size_n, :num_actual_groups]
  return jnp.transpose(out_kng, (2, 1, 0))
