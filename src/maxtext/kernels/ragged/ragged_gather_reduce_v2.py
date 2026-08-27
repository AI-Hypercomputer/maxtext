# Copyright 2026 Google LLC
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

"""Ragged gather reduce kernel implementation from tpu-inference."""
# pylint: disable=line-too-long

import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


# ceil up to the nearest multiple of b.
def _align_to(a, b):
  return pl.cdiv(a, b) * b


class _CostModelConstants:
  # Limit on the number of outer loop pipeline iterations. Too many iterations
  # cause high cumulative pipeline overhead (e.g., from frequent pipeline
  # startup/teardown bubbles). We try to find partitioning that does not exceed
  # this limit on iterations.
  MAX_ITERATIONS: int = 40

  # Upper cap on the column chunk size processed per inner pipeline step.
  # While larger chunk sizes help utilize bandwidth better, excessively large
  # chunk sizes cause large pipeline bubbles. We cap it here to balance
  # efficiency and bubble sizes.
  MAX_COL_CHUNK_SIZE: int = 1024


# pylint: disable=missing-class-docstring
@dataclasses.dataclass(frozen=True)
class _Config:
  input_size: int
  hidden_size: int
  source_rows: int
  reduce_group_size: int
  in_dtype: Any
  core_axis_name: str
  subcore_axis_name: str
  tpu_info: pltpu.TpuInfo
  use_single_sparsecore: bool = False

  def __post_init__(self):
    # Only supports either bf16 or fp32 for now.
    assert self.in_dtype in (jnp.bfloat16, jnp.float32)

  @property
  def sc_info(self):
    sc_info = self.tpu_info.sparse_core
    assert sc_info is not None
    return sc_info

  @property
  def num_sc_cores(self) -> int:
    return 1 if self.use_single_sparsecore else self.sc_info.num_cores

  @property
  def num_tot_cores(self) -> int:
    return self.num_sc_cores * self.sc_info.num_subcores

  def get_num_row_subchunks(self, num_row_partitions: int) -> int:
    base_block_size = self.sc_info.num_lanes * num_row_partitions
    input_size_block = pl.cdiv(self.input_size, base_block_size)
    return max(1, min(4, input_size_block))

  def get_row_chunk_size(self, num_row_partitions: int) -> int:
    return self.sc_info.num_lanes * self.get_num_row_subchunks(
        num_row_partitions
    )

  @property
  def num_row_subchunks(self) -> int:
    return self.get_num_row_subchunks(self.num_row_partitions)

  @property
  def output_size(self) -> int:
    return self.input_size // self.reduce_group_size

  @property
  def in_dtype_bytes(self) -> int:
    return jax.dtypes.itemsize_bits(self.in_dtype) // 8

  @property
  def padded_input_size(self) -> int:
    align_val = self.num_row_partitions * self.reduce_group_size
    return _align_to(self.input_size, align_val)

  @property
  def should_fallback(self) -> bool:
    # For a small {input + output} both likely fit in TensorCore VMEM, where a
    # plain TC gather-reduce beats routing through SparseCore and HBM.
    if self.tpu_info.sparse_core is None:
      return True
    if self.num_row_partitions > self.sc_info.num_lanes:
      return True
    vmem_capacity_threshold = self.tpu_info.vmem_capacity_bytes * 0.6
    source_size = self.source_rows * self.hidden_size * self.in_dtype_bytes
    return source_size * 2 < vmem_capacity_threshold

  @property
  def row_partition_size(self) -> int:
    return self.padded_input_size // self.num_row_partitions

  @property
  def row_partition_size_padded(self) -> int:
    # Pad each partition to a whole number of windows (fixed per-window DMA size).
    return _align_to(self.row_partition_size, self.window_size)

  @property
  def max_blocks_per_partition(self) -> int:
    return pl.cdiv(self.row_partition_size, self.row_chunk_size)

  @property
  def max_window(self) -> int:
    """Largest window of row-blocks whose resident sort permutation fits SPMEM.

    Streaming a fixed window instead of the whole partition makes SPMEM use
    independent of input_size; the clamp keeps small inputs single-window.
    """
    # Per-subcore tile_spmem budget in 32-bit words, kept 10% under to leave
    # headroom for TC-tiling padding.
    words_per_subcore = self.sc_info.vmem_capacity_bytes // 4
    num_simd_lanes = self.sc_info.num_lanes
    # Input-size-independent resident scratch (32-bit words): prev-row carry
    # (col_size), out_vmem + column gather double-buffer (3*lanes*col_chunk),
    # the num_rows + next-window-first-row vectors (2*lanes), and 6 row
    # index/dma buffers + the row gather pipeline double-buffers (10*row_chunk).
    fixed = (
        self.col_size
        + 3 * num_simd_lanes * self.col_chunk_size
        + 2 * num_simd_lanes
        + 10 * self.row_chunk_size
    )
    window = (int(words_per_subcore * 0.9) - fixed) // self.row_chunk_size
    return max(1, min(window, max(1, self.max_blocks_per_partition)))

  @property
  def window_size(self) -> int:
    """Number of rows whose sort permutation is resident per window."""
    return self.max_window * self.row_chunk_size

  @property
  def row_chunk_size(self) -> int:
    """Number of rows handled per row-pipeline block."""
    return self.get_row_chunk_size(self.num_row_partitions)

  @property
  def num_col_chunks(self) -> int:
    return self.col_size // self.col_chunk_size

  @property
  def row_shift(self) -> int:
    """log2 of how many source rows pack into one uint32 gather element.

    The SparseCore indirect DMA requires 32-bit elements: bfloat16 packs two
    source rows per uint32 (shift 1), float32 is 1:1 (shift 0).
    """
    input_packing = 32 // jax.dtypes.itemsize_bits(self.in_dtype)
    return input_packing.bit_length() - 1

  @property
  def num_row_partitions(self) -> int:
    """Calculates the number of row partitions."""
    num_simd_lanes = self.sc_info.num_lanes
    num_row_partitions = self.num_tot_cores // self.num_column_partitions
    return num_row_partitions

  @property
  def num_column_partitions(self) -> int:
    """Calculates the number of column partitions."""
    # DMA constraint requires column partition to be multiple of lane size.
    # Prefer to use a large number of column partitions, as long as each
    # partition's size is not too small for DMA pipeline efficiency and each
    # partition's size can divide the hidden size.
    # Each column partition will do DMA pipelining on col_size.
    num_lanes = self.tpu_info.num_lanes
    num_simd_lanes = self.sc_info.num_lanes
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        self.num_tot_cores % (num_column_partitions * 2) == 0
        and self.hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and (
            self.hidden_size // (num_column_partitions * 2 * num_lanes)
            >= preferred_num_stages
            or self.num_tot_cores // num_column_partitions > num_simd_lanes
        )
    ):
      next_candidate = num_column_partitions * 2
      next_row_partitions = self.num_tot_cores // next_candidate

      # Calculate exactly how many pipeline invocations (outer loop).
      row_chunk_size = self.get_row_chunk_size(next_row_partitions)
      num_iterations = self.input_size // (row_chunk_size * next_row_partitions)

      # Too many row partitions for the SIMD lanes: split the columns
      # further before weighing the iteration count.
      if self.num_tot_cores // num_column_partitions > num_simd_lanes:
        num_column_partitions = next_candidate
        continue

      # Too many iterations cause high cumulative pipeline overhead. Set
      # the limit based on empirical data.
      if num_iterations > _CostModelConstants.MAX_ITERATIONS:
        break

      num_column_partitions = next_candidate
    return num_column_partitions

  @property
  def aligned_hidden_size(self) -> int:
    """Calculates the aligned hidden size."""
    num_lanes = self.tpu_info.num_lanes
    return _align_to(self.hidden_size, num_lanes * self.num_column_partitions)

  @property
  def col_size(self) -> int:
    """Calculates the column size."""
    return self.aligned_hidden_size // self.num_column_partitions

  @property
  def col_chunk_size(self) -> int:
    """Picks the column chunk size the inner pipeline gathers at a time.

    The chunk is the largest divisor of ``col_size`` whose gather double-buffer
    still fits comfortably in SparseCore VMEM.
    """
    match self.tpu_info.generation:
      case 6:
        target_bytes = int(256 * 1024 * 0.95)
      case 7:
        target_bytes = int(512 * 1024 * 0.95)
      case _:
        target_bytes = int(128 * 1024 * 0.95)

    # uint32 gather buffer, double-buffered by emit_pipeline.
    num_simd_lanes = self.sc_info.num_lanes
    num_lanes = self.tpu_info.num_lanes
    bytes_per_col = num_simd_lanes * 4 * 2
    max_safe_col = (target_bytes // bytes_per_col // num_lanes) * num_lanes

    # Larger chunk sizes cause larger pipeline bubbles, so cap it.
    max_safe_col = min(max_safe_col, _CostModelConstants.MAX_COL_CHUNK_SIZE)

    start_col = (min(self.col_size, max_safe_col) // num_lanes) * num_lanes
    for chunk in range(start_col, num_lanes - 1, -num_lanes):
      if self.col_size % chunk == 0:
        return chunk
    return num_lanes


def get_cost_estimate(
    padded_input_size: int,
    aligned_hidden_size: int,
    reduce_group_size: int,
    input_dtype_bytes: int,
    bytes_accessed_override: int = -1,
    flops_override: int = -1,
) -> pl.CostEstimate:
  """Returns a cost estimate for the ragged gather-reduce kernel.

  The kernel gathers rows, multiplies each by a scalar weight, and reduces
  (sums) every ``reduce_group_size`` rows into one output row.

  Args:
    padded_input_size: Total number of source rows (after padding).
    aligned_hidden_size: Number of columns (after alignment).
    reduce_group_size: Number of source rows reduced into each output row.
    input_dtype_bytes: Size of one input element in bytes.
    bytes_accessed_override: If > 0, use this value as bytes_accessed instead
      of auto-computing.  -1 (default) means auto-compute.
    flops_override: If > 0, use this value as the flop count instead of
      auto-computing.  -1 (default) means auto-compute.

  Returns:
    A ``pl.CostEstimate`` suitable for XLA scheduling.
  """
  if flops_override > 0:
    flops = flops_override
  else:
    flops = 2 * padded_input_size * aligned_hidden_size

  if bytes_accessed_override > 0:
    bytes_accessed = bytes_accessed_override
  else:
    bytes_in = padded_input_size * aligned_hidden_size * input_dtype_bytes  # input rows
    bytes_in += padded_input_size * 4  # src_indices (int32)
    bytes_in += padded_input_size * 4  # dst_indices (int32)
    bytes_in += padded_input_size * 4  # topk_weights (float32)
    output_rows = padded_input_size // reduce_group_size
    bytes_out = output_rows * aligned_hidden_size * 4  # output rows (float32)
    bytes_accessed = bytes_in + bytes_out

  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=bytes_accessed,
      transcendentals=0,
  )


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
  """Fallback implementation using JAX ops for non-SparseCore TPU or small inputs."""
  out = x[indices] * topk_weights[:, None].astype(jnp.float32)
  out = jnp.where(valid_rows_mask[:, None], out, 0)
  out = out.reshape(-1, reduce_group_size, out.shape[-1])
  out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
  return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class IndexRef:
  num_src_rows_per_row_partition: Any
  indices: Any
  sorted_by_validity: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DataRef:
  source: Any
  topk_weights: Any
  out: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ScratchRef:
  num_rows_per_row_partition_vmem: Any
  next_window_first_row_vmem: Any
  prev_iter_last_row_vmem: Any
  prev_dst_row_smem: Any
  sorted_by_validity_vmem: Any
  src_indices_vmem: Any
  dst_indices_vmem: Any
  tw_f32_vmem: Any
  dma_src_row_vmem: Any
  dma_dst_row_vmem: Any
  prev_dst_val_vmem: Any
  out_vmem: Any
  sem: Any

  @classmethod
  def create_scratch_types(cls, cfg: _Config) -> Any:
    num_simd_lanes = cfg.sc_info.num_lanes
    indices_vmem = pltpu.VMEM((cfg.row_chunk_size,), jnp.int32)
    return cls(
        num_rows_per_row_partition_vmem=pltpu.VMEM(
            (num_simd_lanes,), jnp.int32
        ),
        next_window_first_row_vmem=pltpu.VMEM((num_simd_lanes,), jnp.int32),
        prev_iter_last_row_vmem=pltpu.VMEM(
            (cfg.col_size // cfg.col_chunk_size, cfg.col_chunk_size),
            jnp.float32,
        ),
        prev_dst_row_smem=pltpu.SMEM((1,), jnp.int32),
        sorted_by_validity_vmem=pltpu.VMEM((cfg.window_size,), jnp.int32),
        src_indices_vmem=indices_vmem,
        dst_indices_vmem=indices_vmem,
        dma_src_row_vmem=indices_vmem,
        dma_dst_row_vmem=indices_vmem,
        prev_dst_val_vmem=indices_vmem,
        tw_f32_vmem=pltpu.VMEM((cfg.row_chunk_size,), jnp.float32),
        out_vmem=pltpu.VMEM((num_simd_lanes, cfg.col_chunk_size), jnp.float32),
        sem=pltpu.SemaphoreType.DMA((2,)),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KernelRefs:
  index: IndexRef
  data: DataRef
  scratch: ScratchRef

  @classmethod
  def create(
      cls,
      scalar_ref: IndexRef,
      in_hbm_ref: Any,
      topk_weights_hbm_ref: Any,
      out_hbm_ref: Any,
      scratch_ref: ScratchRef,
  ) -> "KernelRefs":
    return cls(
        index=scalar_ref,
        data=DataRef(
            source=in_hbm_ref,
            topk_weights=topk_weights_hbm_ref,
            out=out_hbm_ref,
        ),
        scratch=scratch_ref,
    )


def _preprocess_scalar_data(
    indices: jax.Array,
    valid_rows_mask: jax.Array,
    cfg: _Config,
) -> tuple[IndexRef, jax.Array]:
  """Sorts valid source rows to the front of each row partition.

  Args:
    indices: Indices for gather.
    valid_rows_mask: Mask indicating valid rows.
    cfg: Ragged gather reduce config.

  Returns:
    scalar: IndexRef holding indices, sorted_by_validity, num_src_rows_per_row_partition.
    mask: per output group, whether the group has any valid source row.
  """
  num_simd_lanes = cfg.sc_info.num_lanes
  valid_rows_mask_2d = valid_rows_mask.reshape(cfg.num_row_partitions, -1)

  # Stable sort of a boolean key is a stable partition: valid rows keep their
  # relative order and move ahead of the invalid ones.
  sorted_by_validity = jnp.argsort(
      ~valid_rows_mask_2d, descending=False, stable=True, axis=-1
  )
  sorted_by_validity += (
      jnp.arange(cfg.num_row_partitions)[:, None] * cfg.row_partition_size
  )

  padding = cfg.row_partition_size_padded - cfg.row_partition_size
  if padding > 0:
    sorted_by_validity = jnp.pad(sorted_by_validity, ((0, 0), (0, padding)))
  sorted_by_validity = sorted_by_validity.reshape(-1)

  num_src_rows_per_row_partition = jnp.pad(
      jnp.sum(valid_rows_mask_2d, axis=-1).astype(jnp.int32),
      (0, max(0, num_simd_lanes - cfg.num_row_partitions)),
  )
  mask = jnp.any(valid_rows_mask.reshape(-1, cfg.reduce_group_size), axis=-1)
  scalar = IndexRef(
      indices=indices,
      sorted_by_validity=sorted_by_validity.astype(jnp.int32),
      num_src_rows_per_row_partition=num_src_rows_per_row_partition,
  )
  return scalar, mask


def _pack_scalars_to_vector(scalar_list: list[jax.Array]) -> jax.Array:
  """Pack list of scalar values into a single VMEM lane."""
  num_lanes = len(scalar_list)

  idx_vec = jax.lax.broadcasted_iota(jnp.int32, (num_lanes,), 0)
  vec = jnp.zeros((num_lanes,), jnp.int32)
  for i in range(num_lanes):
    vec = jnp.where(idx_vec == i, scalar_list[i], vec)
  return vec


def _row_gather_specs(
    sorted_by_validity_vmem: jax.Ref, cfg: _Config
) -> tuple[pl.BlockSpec, ...]:
  """Indirect BlockSpec gathering rows of a 1-D input."""

  num_simd_lanes = cfg.sc_info.num_lanes

  def row_index_map(r: int | jax.Array, *, offset: int) -> jax.Array:
    start = r * cfg.row_chunk_size + offset * num_simd_lanes
    return sorted_by_validity_vmem[pl.ds(start, num_simd_lanes)]

  return tuple([
      pl.BlockSpec(
          (pl.Indirect(num_simd_lanes),),
          functools.partial(row_index_map, offset=offset),
      )
      for offset in range(cfg.num_row_subchunks)
  ])


def _col_gather_spec(
    src_indices_vmem: jax.Ref,
    col_start: jax.Array,
    cfg: _Config,
) -> pl.BlockSpec:
  """Indirect BlockSpec gathering columns of a 2-D input."""

  num_simd_lanes = cfg.sc_info.num_lanes

  def col_index_map(
      s: int | jax.Array, c: int | jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    row = jnp.bitwise_right_shift(
        src_indices_vmem[pl.ds(s * num_simd_lanes, num_simd_lanes)],
        cfg.row_shift,
    )
    col = col_start // cfg.col_chunk_size + c
    return (row, col)

  return pl.BlockSpec(
      (pl.Indirect(num_simd_lanes), cfg.col_chunk_size),
      col_index_map,
  )


def _row_kernel(
    src_indices_refs: tuple[jax.Ref, ...],
    topk_weights_refs: tuple[jax.Ref, ...],
    *,
    refs: KernelRefs,
    num_rows_current_row_partition: jax.Array,
    col_start: jax.Array,
    window_block_base: jax.Array,
    blocks_in_window: jax.Array,
    cfg: _Config,
):
  in_32b_hbm_ref = refs.data.source.bitcast(jnp.uint32)
  num_simd_lanes = cfg.sc_info.num_lanes
  window_words = cfg.window_size

  row_block_id = pl.program_id(0)
  # Absolute row-block index within the partition for the validity mask
  global_block_id = window_block_base + row_block_id

  # Destination output row of each source row in this block.
  dst_indices_list = []
  for s in range(cfg.num_row_subchunks):
    start = row_block_id * cfg.row_chunk_size + s * num_simd_lanes
    gather_dst = refs.scratch.sorted_by_validity_vmem[
        pl.ds(start, num_simd_lanes)
    ]
    dst_indices_list.append(gather_dst // cfg.reduce_group_size)

  # Stage the gathered indices/weights and the destinations in VMEM.
  for s in range(cfg.num_row_subchunks):
    sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
    refs.scratch.dst_indices_vmem[sub] = dst_indices_list[s]
    refs.scratch.src_indices_vmem[sub] = src_indices_refs[s][...]
    refs.scratch.tw_f32_vmem[sub] = topk_weights_refs[s][...]

  # For each sub-chunk, the destination of the row just before it -- the
  # seed for the segmented reduction's "same group as previous row" test.
  for s in range(cfg.num_row_subchunks):
    if s == 0:
      prev_dst = refs.scratch.prev_dst_row_smem[0]
    else:
      prev_dst = dst_indices_list[s - 1][num_simd_lanes - 1]
    refs.scratch.prev_dst_val_vmem[
        pl.ds(s * num_simd_lanes, num_simd_lanes)
    ] = jnp.broadcast_to(prev_dst, (num_simd_lanes,))

  # For each source row, find the VMEM row that will hold its group's fully
  # reduced value -- the last row of the group within this block. Scanning
  # backwards, a row inherits its successor's merge target when they share
  # a destination, otherwise it is its own target.
  rev_src_row_idx_in_vmem = []
  rev_is_row_valid = []
  for row_vmem_idx in reversed(range(cfg.row_chunk_size)):
    if row_vmem_idx == cfg.row_chunk_size - 1:
      next_src_row_idx = row_vmem_idx
    else:
      quot, rem = divmod(row_vmem_idx, num_simd_lanes)
      quot_next, rem_next = divmod(row_vmem_idx + 1, num_simd_lanes)
      same_group_as_next = jnp.logical_and(
          rev_is_row_valid[-1],
          dst_indices_list[quot][rem] == dst_indices_list[quot_next][rem_next],
      )
      next_src_row_idx = jnp.where(
          same_group_as_next, rev_src_row_idx_in_vmem[-1], row_vmem_idx
      )
    global_row_idx = global_block_id * cfg.row_chunk_size + row_vmem_idx
    rev_is_row_valid.append(global_row_idx < num_rows_current_row_partition)
    rev_src_row_idx_in_vmem.append(next_src_row_idx)
  src_row_idx_in_vmem = rev_src_row_idx_in_vmem[::-1]
  is_row_valid = rev_is_row_valid[::-1]

  # Check if the current group continues into the next block. If it does, we
  # defer the HBM write to the next block to avoid race conditions.
  is_last_block_in_window = (row_block_id + 1) == blocks_in_window
  next_block_first_row = (global_block_id + 1) * cfg.row_chunk_size
  # Fetch the first row index of the next block. It lives either in the current
  # window VMEM or the prefetched next-window VMEM.
  next_block_first_row_in_window = jnp.minimum(
      (row_block_id + 1) * cfg.row_chunk_size, window_words - num_simd_lanes
  )
  next_block_first_idx = jnp.where(
      is_last_block_in_window,
      refs.scratch.next_window_first_row_vmem[...][0],
      refs.scratch.sorted_by_validity_vmem[
          pl.ds(next_block_first_row_in_window, num_simd_lanes)
      ][0],
  )
  group_continues = jnp.logical_and(
      next_block_first_row < num_rows_current_row_partition,
      (next_block_first_idx // cfg.reduce_group_size)
      == dst_indices_list[-1][num_simd_lanes - 1],
  )

  # Per source row, the (VMEM source row, HBM destination row) of its
  # scatter. Rows whose group is not yet fully reduced in this sub-chunk,
  # and padding rows, are routed to a throwaway row.
  garbage_dst = refs.data.out.shape[0] - 1
  dma_src_rows = []
  dma_dst_rows = []
  for s in range(cfg.num_row_subchunks):
    sub_src = []
    sub_dst = []
    for i in range(num_simd_lanes):
      global_idx = s * num_simd_lanes + i
      merge_target = src_row_idx_in_vmem[global_idx]
      is_final_write = jnp.logical_and(
          is_row_valid[global_idx],
          merge_target < (s + 1) * num_simd_lanes,
      )
      # Only the last sub-chunk's group can reach the block's final row; earlier
      # sub-chunks already route such a group to garbage.
      if s == cfg.num_row_subchunks - 1:
        merges_at_block_end = merge_target == cfg.row_chunk_size - 1
        spans_next_block = jnp.logical_and(merges_at_block_end, group_continues)
        is_final_write = jnp.logical_and(
            is_final_write, jnp.logical_not(spans_next_block)
        )
      sub_src.append(
          jnp.where(is_final_write, merge_target % num_simd_lanes, 0)
      )
      sub_dst.append(
          jnp.where(is_final_write, dst_indices_list[s][i], garbage_dst)
      )
    dma_src_rows.append(sub_src)
    dma_dst_rows.append(sub_dst)

  for s in range(cfg.num_row_subchunks):
    sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
    refs.scratch.dma_src_row_vmem[sub] = _pack_scalars_to_vector(
        dma_src_rows[s]
    )
    refs.scratch.dma_dst_row_vmem[sub] = _pack_scalars_to_vector(
        dma_dst_rows[s]
    )

  col_pipeline = pltpu.emit_pipeline(
      functools.partial(
          _col_kernel,
          refs=refs,
          col_start=col_start,
          cfg=cfg,
      ),
      grid=(cfg.num_row_subchunks, cfg.num_col_chunks),
      in_specs=_col_gather_spec(refs.scratch.src_indices_vmem, col_start, cfg),
  )

  col_pipeline(in_32b_hbm_ref)
  refs.scratch.prev_dst_row_smem[0] = dst_indices_list[-1][num_simd_lanes - 1]


def _col_kernel(
    gather_ref: jax.Ref,
    *,
    refs: KernelRefs,
    col_start: jax.Array,
    cfg: _Config,
):
  s = pl.program_id(0)
  c = pl.program_id(1)
  col_hbm_start = col_start + c * cfg.col_chunk_size
  send_sem = refs.scratch.sem.at[1]

  num_simd_lanes = cfg.sc_info.num_lanes

  row_slice = pl.ds(s * num_simd_lanes, num_simd_lanes)
  tw_slice = refs.scratch.tw_f32_vmem[row_slice]
  dst_slice = refs.scratch.dst_indices_vmem[row_slice]
  src_idx_slice = refs.scratch.src_indices_vmem[row_slice]
  prev_dst_vals_vec = refs.scratch.prev_dst_val_vmem[row_slice]

  @plsc.parallel_loop(0, cfg.col_chunk_size, step=num_simd_lanes)
  def col_loop(col_compute_offset: jax.Array):
    col_slice = pl.ds(col_compute_offset, num_simd_lanes)
    # Running sum, seeded by the carry from the previous sub-chunk.
    previous_accumulated_data = refs.scratch.prev_iter_last_row_vmem[
        c, col_slice
    ]

    for row_src in range(num_simd_lanes):
      val_u32 = gather_ref[row_src, col_slice]
      if cfg.in_dtype == jnp.bfloat16:
        # The two bfloat16 rows packed in one uint32 word sit in the low
        # (even row) or high (odd row) 16 bits. Shift the wanted half
        # into the float32 sign/exponent position and clear the rest.
        is_even_row = jnp.bitwise_and(src_idx_slice[row_src], 1) == 0
        shift = jnp.where(is_even_row, 16, 0)
        lower_mask = jnp.uint32(jnp.iinfo(jnp.uint16).max)
        upper_mask = jnp.left_shift(lower_mask, 16)
        shifted = jnp.bitwise_and(jnp.left_shift(val_u32, shift), upper_mask)
        data_f32 = plsc.bitcast(shifted, jnp.float32)
      else:
        data_f32 = plsc.bitcast(val_u32, jnp.float32)
      data_f32 *= tw_slice[row_src]

      # Reduction: accumulate while the destination group is unchanged,
      # restart otherwise. Sorting guarantees rows of one group are
      # contiguous.
      dst_row_hbm = dst_slice[row_src]
      if row_src == 0:
        prev_dst = prev_dst_vals_vec[0]
      else:
        prev_dst = dst_slice[row_src - 1]
      accumulated_data = jnp.where(
          dst_row_hbm == prev_dst,
          previous_accumulated_data + data_f32,
          data_f32,
      )
      previous_accumulated_data = accumulated_data

      # The output buffer stays float32: a bfloat16 output would be
      # (16, 128)-tiled and the per-row scatter below writes a single
      # row at an arbitrary, non-tile-aligned destination, which is only
      # legal for 32-bit elements. The cast happens in the wrapper.
      refs.scratch.out_vmem[row_src, col_slice] = accumulated_data
      if row_src == num_simd_lanes - 1:
        refs.scratch.prev_iter_last_row_vmem[c, col_slice] = accumulated_data

  # Scatter every source row's reduced value to its output row. Rows
  # that share a group write the same value (idempotent); rows routed to
  # the garbage destination are harmless.
  dma_src_row_slice = refs.scratch.dma_src_row_vmem[row_slice]
  dma_dst_row_slice = refs.scratch.dma_dst_row_vmem[row_slice]
  copies = []
  for i in range(num_simd_lanes):
    copy = pltpu.make_async_copy(
        refs.scratch.out_vmem.at[
            dma_src_row_slice[i], pl.ds(0, cfg.col_chunk_size)
        ],
        refs.data.out.at[
            dma_dst_row_slice[i], pl.ds(col_hbm_start, cfg.col_chunk_size)
        ],
        send_sem,
    )
    copy.start()
    copies.append(copy)
  for copy in copies:
    copy.wait()


def call_kernel_pipeline(
    row_partition_id: jax.Array,
    refs: KernelRefs,
    col_start: jax.Array,
    cfg: _Config,
):
  num_rows_per_row_partition = refs.scratch.num_rows_per_row_partition_vmem[...]
  num_rows_current_row_partition = jnp.array(0, jnp.int32)
  for i in range(cfg.num_row_partitions):
    num_rows_current_row_partition = jnp.where(
        row_partition_id == i,
        num_rows_per_row_partition[i],
        num_rows_current_row_partition,
    )
  num_row_blocks = pl.cdiv(num_rows_current_row_partition, cfg.row_chunk_size)

  # Sentinel for the cross-block reduction carry (no previous group).
  refs.scratch.prev_dst_row_smem[0] = -1

  num_windows = pl.cdiv(num_row_blocks, cfg.max_window)
  pl.loop(0, num_windows)(
      functools.partial(
          _window_kernel,
          refs=refs,
          row_partition_id=row_partition_id,
          num_row_blocks=num_row_blocks,
          num_rows_current_row_partition=num_rows_current_row_partition,
          col_start=col_start,
          cfg=cfg,
      )
  )


def _window_kernel(
    window_id: jax.Array,
    *,
    refs: KernelRefs,
    row_partition_id: jax.Array,
    num_row_blocks: jax.Array,
    num_rows_current_row_partition: jax.Array,
    col_start: jax.Array,
    cfg: _Config,
):
  """Stages one window of the sort permutation, then runs its row blocks."""
  num_simd_lanes = cfg.sc_info.num_lanes
  sorted_by_validity = refs.index.sorted_by_validity
  recv_sem = refs.scratch.sem.at[0]
  window_words = cfg.window_size
  window_start = (
      row_partition_id * cfg.row_partition_size_padded
      + window_id * window_words
  )
  window_block_base = window_id * cfg.max_window

  # Streaming one window at a time bounds the resident scratch
  sorted_dma = pltpu.make_async_copy(
      sorted_by_validity.at[pl.ds(window_start, window_words)],
      refs.scratch.sorted_by_validity_vmem,
      recv_sem,
  )
  sorted_dma.start()
  sorted_dma.wait()

  # Prefetch the next window's first source row so the last block of this window
  # can detect a group continuing across the window boundary. Clamped, so the
  # last window fetches a harmless past-the-end row instead.
  last_start = sorted_by_validity.shape[0] - num_simd_lanes
  next_window_start = jnp.minimum(window_start + window_words, last_start)
  next_window_rows = sorted_by_validity.at[
      pl.ds(next_window_start, num_simd_lanes)
  ]
  next_window_dma = pltpu.make_async_copy(
      next_window_rows,
      refs.scratch.next_window_first_row_vmem,
      recv_sem,
  )
  next_window_dma.start()
  next_window_dma.wait()

  blocks_in_window = jnp.minimum(
      cfg.max_window, num_row_blocks - window_block_base
  )

  # The index maps read the resident sort window staged above.
  row_gather_specs = _row_gather_specs(
      refs.scratch.sorted_by_validity_vmem, cfg
  )
  row_pipeline_fn = pltpu.emit_pipeline(
      functools.partial(
          _row_kernel,
          cfg=cfg,
          refs=refs,
          num_rows_current_row_partition=num_rows_current_row_partition,
          col_start=col_start,
          window_block_base=window_block_base,
          blocks_in_window=blocks_in_window,
      ),
      grid=(blocks_in_window,),
      in_specs=(row_gather_specs, row_gather_specs),
  )
  row_pipeline_fn(
      ((refs.index.indices,) * cfg.num_row_subchunks),
      ((refs.data.topk_weights,) * cfg.num_row_subchunks),
  )


def main_kernel(
    scalar_ref: IndexRef,
    in_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    scratch_ref: ScratchRef,
    *,
    cfg: _Config,
):
  # Step 1: Resolve this core's row/column partition and its column slice.
  core_id = jax.lax.axis_index((cfg.core_axis_name, cfg.subcore_axis_name))
  row_partition_id = core_id // cfg.num_column_partitions
  col_partition_id = core_id % cfg.num_column_partitions

  col_start = col_partition_id * cfg.col_size

  refs = KernelRefs.create(
      scalar_ref=scalar_ref,
      in_hbm_ref=in_hbm_ref,
      topk_weights_hbm_ref=topk_weights_hbm_ref,
      out_hbm_ref=out_hbm_ref,
      scratch_ref=scratch_ref,
  )

  # Step 2: Stage this partition's row count (the sort permutation is streamed
  # one window at a time in call_kernel_pipeline, bounding the resident scratch).
  recv_sem = refs.scratch.sem.at[0]
  num_rows_dma = pltpu.make_async_copy(
      refs.index.num_src_rows_per_row_partition,
      refs.scratch.num_rows_per_row_partition_vmem,
      recv_sem,
  )
  num_rows_dma.start()
  num_rows_dma.wait()

  call_kernel_pipeline(
      row_partition_id=row_partition_id,
      refs=refs,
      col_start=col_start,
      cfg=cfg,
  )


def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    enforce_fallback: bool = False,
    flops_override: int = -1,
    bytes_accessed_override: int = -1,
    use_single_sparsecore: bool = False,
) -> jax.Array:
  """Gathers ``x`` by ``indices``, weights and masks, then reduces by group.

  Args:
    x: 2-D input features, ``(num_rows, hidden_size)``.
    indices: 1-D gather indices, ``(input_size,)``.
    topk_weights: 1-D per-row weights, ``(input_size,)``.
    valid_rows_mask: 1-D bool mask of valid gathered rows, ``(input_size,)``.
    reduce_group_size: number of consecutive rows summed into one output row.
    enforce_fallback: if True, force use of reference fallback implementation.
    flops_override: optional flop count override for cost estimate.
    bytes_accessed_override: optional bytes accessed override for cost estimate.
    use_single_sparsecore: if True, run on 1 SparseCore instead of all.

  Returns:
    Reduced output, ``(input_size // reduce_group_size, hidden_size)``.
  """
  # Step 1: Choose the implementation (TensorCore fallback or SparseCore).
  # Guard against eager initialization on non-TPU hardware (e.g. during CPU tests).
  # pltpu.get_tpu_info() expects TPU hardware and will crash if executed on CPU.
  if enforce_fallback or jax.devices()[0].platform != "tpu":
    return _fallback_implementation(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return _fallback_implementation(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )

  # Step 2: Create config object.
  cfg = _Config(
      input_size=indices.size,
      hidden_size=x.shape[-1],
      source_rows=x.shape[0],
      reduce_group_size=reduce_group_size,
      in_dtype=x.dtype,
      core_axis_name="core",
      subcore_axis_name="subcore",
      tpu_info=pltpu.get_tpu_info(),
      use_single_sparsecore=use_single_sparsecore,
  )

  # Step 3: Fallback to compiler version if needed.
  if cfg.should_fallback:
    return _fallback_implementation(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size
    )

  # Step 4: Pre-process inputs (weights, padding, sort by validity).
  # Simplify topk gather by using fp32 and ensure data is always word aligned.
  topk_weights_f32 = topk_weights.astype(jnp.float32)

  # Pad the input so each row partition holds a whole number of reduce
  # groups; no group is then split across two physical cores.
  valid_rows_mask = jnp.pad(
      valid_rows_mask,
      (0, cfg.padded_input_size - cfg.input_size),
      constant_values=False,
  )

  scalar, mask = _preprocess_scalar_data(indices, valid_rows_mask, cfg)

  # Step 5: Launch the SparseCore kernel.
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=cfg.num_sc_cores,
      num_subcores=cfg.sc_info.num_subcores,
      core_axis_name=cfg.core_axis_name,
      subcore_axis_name=cfg.subcore_axis_name,
  )

  # The output gets one extra row: the kernel's garbage scatter destination.
  out = pl.kernel(
      functools.partial(main_kernel, cfg=cfg),
      out_type=jax.ShapeDtypeStruct(
          (
              cfg.padded_input_size // reduce_group_size + 1,
              cfg.aligned_hidden_size,
          ),
          jnp.float32,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
          needs_layout_passes=False,
      ),
      cost_estimate=get_cost_estimate(
          padded_input_size=cfg.padded_input_size,
          aligned_hidden_size=cfg.aligned_hidden_size,
          reduce_group_size=reduce_group_size,
          input_dtype_bytes=cfg.in_dtype_bytes,
          flops_override=flops_override,
          bytes_accessed_override=bytes_accessed_override,
      ),
      scratch_types=(ScratchRef.create_scratch_types(cfg),),
      mesh=vector_mesh,
      name="sc_ragged_gather_reduce_v2",
  )(scalar, x, topk_weights_f32)

  # Step 6: Post-process the output (drop padding, zero empty groups, cast).
  out = out[: cfg.output_size, : cfg.hidden_size]
  out = jnp.where(mask[: cfg.output_size, None], out, jnp.zeros_like(out))
  return out.astype(x.dtype)
