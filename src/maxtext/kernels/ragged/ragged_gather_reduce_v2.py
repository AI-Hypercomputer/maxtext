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
# Source from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


# pylint: disable=missing-class-docstring
@dataclasses.dataclass(frozen=True)
class _Config:
  num_row_partitions: int
  num_column_partitions: int
  reduce_group_size: int
  col_size: int
  col_chunk_size: int
  num_row_subchunks: int
  num_simd_lanes: int
  topk_dtype: Any
  in_dtype: Any
  core_axis_name: str
  subcore_axis_name: str

  @property
  def row_chunk_size(self) -> int:
    """Number of rows handled per row-pipeline block."""
    return self.num_simd_lanes * self.num_row_subchunks

  @property
  def row_shift(self) -> int:
    """log2 of how many source rows pack into one uint32 gather element.

    The SparseCore indirect DMA requires 32-bit elements: bfloat16 packs two
    source rows per uint32 (shift 1), float32 is 1:1 (shift 0).
    """
    input_packing = 32 // jax.dtypes.itemsize_bits(self.in_dtype)
    return input_packing.bit_length() - 1


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
  # Flops:
  #   - one multiply per element for weighting: padded_input_size * aligned_hidden_size
  #   - one add per element for reduction:       padded_input_size * aligned_hidden_size
  if flops_override > 0:
    flops = flops_override
  else:
    flops = 2 * padded_input_size * aligned_hidden_size

  if bytes_accessed_override > 0:
    bytes_accessed = bytes_accessed_override
  else:
    # Bytes accessed:
    #   read  – input rows + src_indices (int32) + dst_indices (int32) + topk_weights (f32)
    #   write – output rows (float32)
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Inputs:
  num_src_rows_per_row_partition: Any
  x: Any
  indices: Any
  topk_weights: Any
  sorted_by_validity: Any


# pylint: disable=missing-class-docstring
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Scratch:
  num_rows_per_row_partition_vmem: Any
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

  def __len__(self) -> int:
    return len(dataclasses.fields(self))

  def __getitem__(self, index: Any):
    return getattr(self, dataclasses.fields(self)[index].name)


# pylint: disable=missing-class-docstring
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


# ceil up to the nearest multiple of b.
def _align_to(a, b):
  return pl.cdiv(a, b) * b


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


def _calculate_num_column_partitions(
    hidden_size: int, input_size: int, num_cores: int, num_lanes: int, num_simd_lanes: int
) -> int:
  """Calculates the number of row partitions."""
  # Each column partition should be multiple of 128 (number of lanes) due to
  # DMA requirements.
  # Prefer to use a large number of column partitions, as long as each
  # partition's size is not too small for DMA pipeline efficiency and each
  # partition's size can divide the hidden size.

  # Each column partition will do DMA pipelining on col_size.
  preferred_num_stages = 4
  num_column_partitions = 1
  while (
      num_cores % (num_column_partitions * 2) == 0
      and hidden_size % (num_lanes * num_column_partitions * 2) == 0
      and hidden_size // (num_column_partitions * 2 * num_lanes) >= preferred_num_stages
  ):
    next_candidate = num_column_partitions * 2
    next_row_partitions = num_cores // next_candidate

    # Calculate exactly how many pipeline invocations (outer loop)
    _, row_chunk_size = _calculate_row_tiling(input_size, num_simd_lanes, next_row_partitions)
    num_iterations = input_size // (row_chunk_size * next_row_partitions)

    # Ensure we satisfy the hardware constraint (num_row_partitions <= num_simd_lanes) first.
    if num_cores // num_column_partitions > num_simd_lanes:
      num_column_partitions = next_candidate
      continue

    # Too many iterations cause high cumulative pipeline overhead. Set the
    # limit based on empirical data.
    if num_iterations > _CostModelConstants.MAX_ITERATIONS:
      break

    num_column_partitions = next_candidate

  return num_column_partitions


def _calculate_row_tiling(
    input_size: int,
    num_simd_lanes: int,
    num_row_partitions: int,
) -> tuple[int, int]:
  """Calculates the number of row subchunks and row chunk size."""
  base_block_size = num_simd_lanes * num_row_partitions
  num_row_subchunks = max(1, min(4, pl.cdiv(input_size, base_block_size)))
  row_chunk_size = num_simd_lanes * num_row_subchunks
  return num_row_subchunks, row_chunk_size


def _calculate_col_chunk_size(col_size: int, num_simd_lanes: int) -> int:
  """Picks the column chunk size the inner pipeline gathers at a time.

  The chunk is the largest divisor of ``col_size`` whose gather double-buffer
  still fits comfortably in SparseCore VMEM.
  """
  generation = pltpu.get_tpu_info().generation
  match generation:
    case 6:
      target_bytes = int(256 * 1024 * 0.95)
    case 7:
      target_bytes = int(512 * 1024 * 0.95)
    case _:
      target_bytes = int(128 * 1024 * 0.95)

  # uint32 gather buffer, double-buffered by emit_pipeline.
  bytes_per_col = num_simd_lanes * 4 * 2
  max_safe_col = (target_bytes // bytes_per_col // 128) * 128

  # Larger chunk sizes cause larger pipeline bubbles, so cap it at 1024.
  max_safe_col = min(max_safe_col, _CostModelConstants.MAX_COL_CHUNK_SIZE)

  start_col = (min(col_size, max_safe_col) // 128) * 128
  for chunk in range(start_col, 127, -128):
    if col_size % chunk == 0:
      return chunk
  return 128


def _compute_sorted_by_validity(
    valid_rows_mask_2d: jax.Array,
    num_row_partitions: int,
    row_chunk_size: int,
) -> jax.Array:
  """Computes row index permutation sorted by validity for SparseCore tiling."""
  row_partition_size = valid_rows_mask_2d.shape[1]
  # Stable sort of a boolean key is a stable partition: valid rows keep their
  # relative order and move ahead of the invalid ones.
  sorted_by_validity = jnp.argsort(~valid_rows_mask_2d, descending=False, stable=True, axis=-1)
  sorted_by_validity += jnp.arange(num_row_partitions)[:, None] * row_partition_size

  pad_to = _align_to(row_partition_size, row_chunk_size)
  if pad_to > row_partition_size:
    sorted_by_validity = jnp.pad(
        sorted_by_validity,
        ((0, 0), (0, pad_to - row_partition_size)),
        constant_values=0,
    )
  return sorted_by_validity.reshape(-1).astype(jnp.int32)


def _preprocess(
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    num_row_partitions: int,
    num_simd_lanes: int,
    row_chunk_size: int,
    precomputed_sorted_by_validity: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Sorts valid source rows to the front of each row partition.

  Returns:
    sorted_by_validity: original row index of each slot after the stable
      sort, flattened across partitions and padded to ``row_chunk_size``.
    num_src_rows_per_row_partition: valid row count per partition, padded to
      ``num_simd_lanes`` so the kernel can load it as a single vector.
    mask: per output group, whether the group has any valid source row.
  """
  valid_rows_mask_2d = valid_rows_mask.reshape(num_row_partitions, -1)

  if precomputed_sorted_by_validity is not None:
    sorted_by_validity = precomputed_sorted_by_validity
  else:
    sorted_by_validity = _compute_sorted_by_validity(valid_rows_mask_2d, num_row_partitions, row_chunk_size)

  num_src_rows_per_row_partition = jnp.pad(
      jnp.sum(valid_rows_mask_2d, axis=-1).astype(jnp.int32),
      (0, max(0, num_simd_lanes - num_row_partitions)),
  )
  mask = jnp.any(valid_rows_mask.reshape(-1, reduce_group_size), axis=-1)
  return (
      sorted_by_validity,
      num_src_rows_per_row_partition,
      mask,
  )


def _pack_scalars_to_vector(scalar_list: list[jax.Array], num_simd_lanes: int) -> jax.Array:
  """Builds a lane vector from per-lane scalars.

  SparseCore cannot store individual scalars into VMEM lanes, so the vector is
  assembled with masked accumulation before being stored.
  """
  idx_vec = jnp.arange(num_simd_lanes)
  vec = jnp.zeros((num_simd_lanes,), jnp.int32)
  for i in range(num_simd_lanes):
    vec += (idx_vec == i).astype(jnp.int32) * scalar_list[i]
  return vec


def _row_gather_spec(
    sorted_by_validity_vmem: jax.Ref,
    sub: int,
    *,
    num_simd_lanes: int,
    row_chunk_size: int,
) -> pl.BlockSpec:
  """Indirect BlockSpec gathering sub-chunk ``sub``'s rows of a 1-D input."""
  return pl.BlockSpec(
      (pl.Indirect(num_simd_lanes),),
      lambda i, s=sub: (sorted_by_validity_vmem[pl.ds(i * row_chunk_size + s * num_simd_lanes, num_simd_lanes)],),
  )


def main_kernel(
    inputs: _Inputs,
    out_hbm_ref: jax.Ref,
    scratch: _Scratch,
    *,
    cfg: _Config,
):
  """Main kernel for ragged gather-reduce."""
  # Step 1: Resolve this core's row/column partition and its column slice.
  num_simd_lanes = cfg.num_simd_lanes
  col_chunk_size = cfg.col_chunk_size
  num_row_subchunks = cfg.num_row_subchunks
  row_chunk_size = cfg.row_chunk_size

  num_col_chunks = cfg.col_size // col_chunk_size

  core_id = jax.lax.axis_index((cfg.core_axis_name, cfg.subcore_axis_name))
  row_partition_id = core_id // cfg.num_column_partitions
  col_partition_id = core_id % cfg.num_column_partitions

  row_partition_size_padded = inputs.sorted_by_validity.shape[0] // cfg.num_row_partitions
  row_start_padded = row_partition_id * row_partition_size_padded
  col_start = col_partition_id * cfg.col_size

  # Step 2: Stage this partition's row count and sort permutation into VMEM.
  recv_sem = scratch.sem.at[0]
  num_rows_dma = pltpu.make_async_copy(
      inputs.num_src_rows_per_row_partition.at[pl.ds(0, num_simd_lanes)],
      scratch.num_rows_per_row_partition_vmem,
      recv_sem,
  )
  sorted_dma = pltpu.make_async_copy(
      inputs.sorted_by_validity.at[pl.ds(row_start_padded, row_partition_size_padded)],
      scratch.sorted_by_validity_vmem,
      recv_sem,
  )
  num_rows_dma.start()
  sorted_dma.start()
  num_rows_dma.wait()
  sorted_dma.wait()

  num_rows_per_row_partition = scratch.num_rows_per_row_partition_vmem[...]
  num_rows_current_row_partition = jnp.array(0, jnp.int32)
  for i in range(cfg.num_row_partitions):
    num_rows_current_row_partition = jnp.where(
        row_partition_id == i,
        num_rows_per_row_partition[i],
        num_rows_current_row_partition,
    )
  num_row_blocks = pl.cdiv(num_rows_current_row_partition, row_chunk_size)

  # Step 3: Run the gather / weighted segmented-reduce / scatter pipeline.

  # The SparseCore indirect DMA requires 32-bit elements, so x is gathered
  # through a uint32 reinterpretation. bfloat16 packs two source rows per
  # uint32 row (row index >> 1); float32 is 1:1 (row index unchanged).
  in_32b_hbm_ref = inputs.x.bitcast(jnp.uint32)

  # Sentinel for the cross-block reduction carry (no previous group).
  scratch.prev_dst_row_smem[0] = -1

  # One gather per sub-chunk for ``indices``, then the same for
  # ``topk_weights``.
  row_pipeline_in_specs = (
      tuple(
          _row_gather_spec(
              scratch.sorted_by_validity_vmem,
              sub,
              num_simd_lanes=num_simd_lanes,
              row_chunk_size=row_chunk_size,
          )
          for sub in range(num_row_subchunks)
      )
      * 2
  )

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_row_blocks,),
      in_specs=row_pipeline_in_specs,
      out_specs=(),
  )
  def row_pipeline(*args):
    src_indices_refs = args[:num_row_subchunks]
    topk_weights_refs = args[num_row_subchunks : 2 * num_row_subchunks]
    # pylint: disable=unbalanced-tuple-unpacking
    (
        src_indices_vmem_sc,
        dst_indices_vmem_sc,
        tw_f32_vmem_sc,
        dma_src_row_vmem_sc,
        dma_dst_row_vmem_sc,
        prev_dst_val_vmem_sc,
        out_vmem_sc,
        sem_sc,
    ) = args[-8:]

    row_block_id = pl.program_id(0)

    # Destination output row of each source row in this block.
    dst_indices_list = [
        scratch.sorted_by_validity_vmem[
            pl.ds(
                row_block_id * row_chunk_size + s * num_simd_lanes,
                num_simd_lanes,
            )
        ]
        // cfg.reduce_group_size
        for s in range(num_row_subchunks)
    ]

    # Stage the gathered indices/weights and the destinations in VMEM.
    for s in range(num_row_subchunks):
      sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
      src_indices_vmem_sc[sub] = src_indices_refs[s][...]
      dst_indices_vmem_sc[sub] = dst_indices_list[s]

      tw = topk_weights_refs[s][...]
      if cfg.topk_dtype == jnp.bfloat16:
        tw_f32 = plsc.bitcast(jnp.bitwise_left_shift(tw, 16), jnp.float32)
      else:
        tw_f32 = plsc.bitcast(tw, jnp.float32)
      tw_f32_vmem_sc[sub] = tw_f32

    # For each sub-chunk, the destination of the row just before it -- the
    # seed for the segmented reduction's "same group as previous row" test.
    for s in range(num_row_subchunks):
      if s == 0:
        prev_dst = scratch.prev_dst_row_smem[0]
      else:
        prev_dst = dst_indices_list[s - 1][num_simd_lanes - 1]
      prev_dst_val_vmem_sc[pl.ds(s * num_simd_lanes, num_simd_lanes)] = jnp.broadcast_to(prev_dst, (num_simd_lanes,))

    def get_dst_idx(global_idx):
      return dst_indices_list[global_idx // num_simd_lanes][global_idx % num_simd_lanes]

    # For each source row, find the VMEM row that will hold its group's fully
    # reduced value -- the last row of the group within this block. Scanning
    # backwards, a row inherits its successor's merge target when they share
    # a destination, otherwise it is its own target.
    src_row_idx_in_vmem = []
    row_valid_vec = []
    for row_vmem_idx in reversed(range(row_chunk_size)):
      global_row_idx = row_block_id * row_chunk_size + row_vmem_idx
      row_valid_vec.append(global_row_idx < num_rows_current_row_partition)
      if row_vmem_idx == row_chunk_size - 1:
        src_row_idx_in_vmem.append(row_vmem_idx)
      else:
        same_group_as_next = jnp.logical_and(
            row_valid_vec[-2],
            get_dst_idx(row_vmem_idx) == get_dst_idx(row_vmem_idx + 1),
        ).astype(jnp.int32)
        src_row_idx_in_vmem.append(same_group_as_next * src_row_idx_in_vmem[-1] + (1 - same_group_as_next) * row_vmem_idx)
    src_row_idx_in_vmem.reverse()
    row_valid_vec.reverse()

    # Per source row, the (VMEM source row, HBM destination row) of its
    # scatter. Rows whose group is not yet fully reduced in this sub-chunk,
    # and padding rows, are routed to a throwaway row.
    garbage_dst = out_hbm_ref.shape[0] - 1
    dma_src_rows = []
    dma_dst_rows = []
    for s in range(num_row_subchunks):
      sub_src = []
      sub_dst = []
      for i in range(num_simd_lanes):
        global_idx = s * num_simd_lanes + i
        merge_target = src_row_idx_in_vmem[global_idx]
        is_final_write = jnp.logical_and(
            row_valid_vec[global_idx],
            merge_target < (s + 1) * num_simd_lanes,
        )
        sub_src.append(jnp.where(is_final_write, merge_target % num_simd_lanes, 0))
        sub_dst.append(jnp.where(is_final_write, dst_indices_list[s][i], garbage_dst))
      dma_src_rows.append(sub_src)
      dma_dst_rows.append(sub_dst)

    for s in range(num_row_subchunks):
      sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
      dma_src_row_vmem_sc[sub] = _pack_scalars_to_vector(dma_src_rows[s], num_simd_lanes)
      dma_dst_row_vmem_sc[sub] = _pack_scalars_to_vector(dma_dst_rows[s], num_simd_lanes)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(num_row_subchunks, num_col_chunks),
        in_specs=pl.BlockSpec(
            (pl.Indirect(num_simd_lanes), col_chunk_size),
            lambda s, c: (
                jnp.bitwise_right_shift(
                    src_indices_vmem_sc[pl.ds(s * num_simd_lanes, num_simd_lanes)],
                    cfg.row_shift,
                ),
                col_start // col_chunk_size + c,
            ),
        ),
        out_specs=(),
    )
    def col_pipeline(gather_ref, sem_inner):
      s = pl.program_id(0)
      c = pl.program_id(1)
      col_hbm_start = col_start + c * col_chunk_size
      send_sem = sem_inner.at[1]

      row_slice = pl.ds(s * num_simd_lanes, num_simd_lanes)
      tw_slice = tw_f32_vmem_sc[row_slice]
      dst_slice = dst_indices_vmem_sc[row_slice]
      src_idx_slice = src_indices_vmem_sc[row_slice]
      prev_dst_vals_vec = prev_dst_val_vmem_sc[row_slice]

      def col_loop(col_compute_offset):
        col_slice = pl.ds(col_compute_offset, num_simd_lanes)
        # Running sum, seeded by the carry from the previous sub-chunk.
        previous_accumulated_data = scratch.prev_iter_last_row_vmem[c, col_slice]

        for row_src in range(num_simd_lanes):
          val_u32 = gather_ref[row_src, col_slice]
          if cfg.in_dtype == jnp.bfloat16:
            # The two bfloat16 rows packed in one uint32 word sit in the low
            # (even row) or high (odd row) 16 bits. Shift the wanted half
            # into the float32 sign/exponent position and clear the rest.
            shift = jnp.where(jnp.bitwise_and(src_idx_slice[row_src], 1) == 0, 16, 0)
            shifted = jnp.bitwise_and(jnp.left_shift(val_u32, shift), jnp.uint32(0xFFFF0000))
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
          out_vmem_sc[row_src, col_slice] = accumulated_data
          if row_src == num_simd_lanes - 1:
            scratch.prev_iter_last_row_vmem[c, col_slice] = accumulated_data

      plsc.parallel_loop(0, col_chunk_size, step=num_simd_lanes)(col_loop)

      # Scatter every source row's reduced value to its output row. Rows
      # that share a group write the same value (idempotent); rows routed to
      # the garbage destination are harmless.
      dma_src_row_slice = dma_src_row_vmem_sc[row_slice]
      dma_dst_row_slice = dma_dst_row_vmem_sc[row_slice]
      copies = []
      for i in range(num_simd_lanes):
        copy = pltpu.make_async_copy(
            out_vmem_sc.at[dma_src_row_slice[i], pl.ds(0, col_chunk_size)],
            out_hbm_ref.at[dma_dst_row_slice[i], pl.ds(col_hbm_start, col_chunk_size)],
            send_sem,
        )
        copy.start()
        copies.append(copy)
      for copy in copies:
        copy.wait()

    # pylint: disable=no-value-for-parameter
    col_pipeline(in_32b_hbm_ref, scratches=(sem_sc,))
    scratch.prev_dst_row_smem[0] = dst_indices_list[-1][num_simd_lanes - 1]

  row_pipeline(
      *([inputs.indices] * num_row_subchunks),
      *([inputs.topk_weights] * num_row_subchunks),
      scratches=(
          scratch.src_indices_vmem,
          scratch.dst_indices_vmem,
          scratch.tw_f32_vmem,
          scratch.dma_src_row_vmem,
          scratch.dma_dst_row_vmem,
          scratch.prev_dst_val_vmem,
          scratch.out_vmem,
          scratch.sem,
      ),
  )


def get_sorted_by_validity(
    hidden_size: int,
    indices_size: int,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array | None:
  """Computes sorted_by_validity row index permutation from valid_rows_mask."""
  if jax.devices()[0].platform != "tpu":
    return None
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return None

  num_simd_lanes = sc_info.num_lanes
  num_lanes = pltpu.get_tpu_info().num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores

  num_column_partitions = _calculate_num_column_partitions(
      hidden_size, indices_size, num_cores, num_lanes, num_simd_lanes
  )
  num_row_partitions = num_cores // num_column_partitions
  _, row_chunk_size = _calculate_row_tiling(indices_size, num_simd_lanes, num_row_partitions)

  padded_input_size = _align_to(indices_size, num_row_partitions * reduce_group_size)
  padded_valid_rows_mask = jnp.pad(
      valid_rows_mask,
      (0, padded_input_size - indices_size),
      constant_values=False,
  )

  valid_rows_mask_2d = padded_valid_rows_mask.reshape(num_row_partitions, -1)
  return _compute_sorted_by_validity(valid_rows_mask_2d, num_row_partitions, row_chunk_size)


def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    enforce_fallback: bool = False,
    flops_override: int = -1,
    bytes_accessed_override: int = -1,
    precomputed_sorted_by_validity: jax.Array | None = None,
) -> jax.Array:
  """Gathers ``x`` by ``indices``, weights and masks, then reduces by group.

  Args:
    x: 2-D input features, ``(num_rows, hidden_size)``.
    indices: 1-D gather indices, ``(input_size,)``.
    topk_weights: 1-D per-row weights, ``(input_size,)``.
    valid_rows_mask: 1-D bool mask of valid gathered rows, ``(input_size,)``.
    reduce_group_size: number of consecutive rows summed into one output row.
    precomputed_sorted_by_validity: optional pre-computed sorted_by_validity array from
      ``get_sorted_by_validity``.

  Returns:
    Reduced output, ``(input_size // reduce_group_size, hidden_size)``.
  """
  # Step 1: Choose the implementation (TensorCore fallback or SparseCore).
  # Guard against eager initialization on non-TPU hardware (e.g. during CPU tests).
  # pltpu.get_tpu_info() expects TPU hardware and will crash if executed on CPU.
  if enforce_fallback or jax.devices()[0].platform != "tpu":
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  # For a small {input + output} both likely fit in TensorCore VMEM, where a
  # plain TC gather-reduce beats routing through SparseCore and HBM. This
  # also keeps the kernel off configs with num_row_partitions > num_simd_lanes.
  dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
  # if (jnp.size(x) * dtype_bytes * 2
  #         < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6):
  #     return _fallback_implementation(x, indices, topk_weights,
  #                                     valid_rows_mask, reduce_group_size)

  # Step 2: Derive the kernel configuration (core grid and column tiling).
  hidden_size = x.shape[-1]
  input_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_lanes = pltpu.get_tpu_info().num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores

  num_column_partitions = _calculate_num_column_partitions(hidden_size, input_size, num_cores, num_lanes, num_simd_lanes)
  num_row_partitions = num_cores // num_column_partitions
  assert num_row_partitions <= num_simd_lanes, f"{num_row_partitions=} must be <= {num_simd_lanes=}"
  num_row_subchunks, row_chunk_size = _calculate_row_tiling(input_size, num_simd_lanes, num_row_partitions)

  aligned_hidden_size = _align_to(hidden_size, 128 * num_column_partitions)
  col_size = aligned_hidden_size // num_column_partitions
  col_chunk_size = _calculate_col_chunk_size(col_size, num_simd_lanes)

  # Step 3: Pre-process inputs (weights, padding, sort by validity).
  # The kernel gathers x through a uint32 reinterpretation; carry the weights
  # the same way so they can be bitcast back to float32 on SparseCore.
  if topk_weights.dtype == jnp.bfloat16:
    topk_weights_u32 = jax.lax.bitcast_convert_type(topk_weights, jnp.uint16).astype(jnp.uint32)
  else:
    topk_weights_u32 = jax.lax.bitcast_convert_type(topk_weights, jnp.uint32)

  # Pad the input so each row partition holds a whole number of reduce
  # groups; no group is then split across two physical cores.
  padded_input_size = _align_to(input_size, num_row_partitions * reduce_group_size)
  valid_rows_mask = jnp.pad(
      valid_rows_mask,
      (0, padded_input_size - input_size),
      constant_values=False,
  )

  sorted_by_validity, num_src_rows_per_row_partition, mask = _preprocess(
      valid_rows_mask,
      reduce_group_size,
      num_row_partitions,
      num_simd_lanes,
      row_chunk_size,
      precomputed_sorted_by_validity=precomputed_sorted_by_validity,
  )

  # Step 4: Launch the SparseCore kernel.
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  cfg = _Config(
      num_row_partitions=num_row_partitions,
      num_column_partitions=num_column_partitions,
      reduce_group_size=reduce_group_size,
      col_size=col_size,
      col_chunk_size=col_chunk_size,
      num_row_subchunks=num_row_subchunks,
      num_simd_lanes=num_simd_lanes,
      topk_dtype=topk_weights.dtype,
      in_dtype=x.dtype,
      core_axis_name=vector_mesh.core_axis_name,
      subcore_axis_name=vector_mesh.subcore_axis_name,
  )

  # The output gets one extra row: the kernel's garbage scatter destination.
  out = pl.kernel(
      functools.partial(main_kernel, cfg=cfg),
      out_type=jax.ShapeDtypeStruct(
          (padded_input_size // reduce_group_size + 1, aligned_hidden_size),
          jnp.float32,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
          needs_layout_passes=False,
      ),
      cost_estimate=get_cost_estimate(
          padded_input_size=padded_input_size,
          aligned_hidden_size=aligned_hidden_size,
          reduce_group_size=reduce_group_size,
          input_dtype_bytes=dtype_bytes,
          flops_override=flops_override,
          bytes_accessed_override=bytes_accessed_override,
      ),
      scratch_types=(
          _Scratch(
              num_rows_per_row_partition_vmem=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              prev_iter_last_row_vmem=pltpu.VMEM((col_size // col_chunk_size, col_chunk_size), jnp.float32),
              prev_dst_row_smem=pltpu.SMEM((1,), jnp.int32),
              sorted_by_validity_vmem=pltpu.VMEM((sorted_by_validity.size // num_row_partitions,), jnp.int32),
              src_indices_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              dst_indices_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              tw_f32_vmem=pltpu.VMEM((row_chunk_size,), jnp.float32),
              dma_src_row_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              dma_dst_row_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              prev_dst_val_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              out_vmem=pltpu.VMEM((num_simd_lanes, col_chunk_size), jnp.float32),
              sem=pltpu.SemaphoreType.DMA((2,)),
          ),
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather_reduce_v2",
  )(
      _Inputs(
          num_src_rows_per_row_partition=num_src_rows_per_row_partition,
          x=x,
          indices=indices,
          topk_weights=topk_weights_u32,
          sorted_by_validity=sorted_by_validity,
      ),
  )

  # Step 5: Post-process the output (drop padding, zero empty groups, cast).
  out = out[: input_size // reduce_group_size, :hidden_size]
  out = jnp.where(mask[: input_size // reduce_group_size, None], out, jnp.zeros_like(out))
  return out.astype(x.dtype)
