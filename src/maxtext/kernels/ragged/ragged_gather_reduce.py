# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/tpu_db_only.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas kernel for ragged gather reduce on SparseCore."""

import functools
import jax
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as np
import jax.numpy as jnp


def _ragged_gather_reduce_fallback(
    x: jnp.ndarray,
    indices: jnp.ndarray,
    topk_weights: jnp.ndarray,
    valid_rows_mask: jnp.ndarray,
    reduce_group_size: int,
    padded_input_size: int,
) -> jnp.ndarray:
  """Fallback JAX implementation of ragged gather reduce."""
  input_size, hidden_size = x.shape
  num_rows = padded_input_size // reduce_group_size
  out = jnp.zeros((num_rows, hidden_size), dtype=x.dtype)

  # Collect the gathered tokens.
  gathered_tokens = x[indices]
  # Scale by topk weights and mask out invalid tokens!
  scaled_tokens = gathered_tokens * topk_weights[:, jnp.newaxis] * valid_rows_mask[:, jnp.newaxis]

  # Calculate dst_indices (original token i contributes to row i // reduce_group_size)
  token_indices = jnp.arange(indices.shape[0])
  dst_indices = token_indices // reduce_group_size

  # Segment sum.
  out = out.at[dst_indices].add(scaled_tokens)
  return out


def main_kernel(
    # Inputs.
    num_rows_per_row_partition_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    src_indices_div_hbm_ref: jax.Ref,
    src_indices_mod_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    num_rows_per_row_partition_vmem_ref: jax.Ref,
    in_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    prev_iter_last_row_vmem_ref: jax.Ref,
    src_indices_div_vmem_ref: jax.Ref,
    src_indices_mod_vmem_ref: jax.Ref,
    dst_indices_vmem_ref: jax.Ref,
    topk_weights_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
    is_bf16: bool,
):
  """SparseCore kernel for ragged gather reduce."""
  # Local subcore indices.
  core_id = shard_map.current_device_index(core_axis_name)
  subcore_id = shard_map.current_device_index(subcore_axis_name)
  num_subcores = 2 # TPU v6e has 2 subcores per core.

  # Row partition is mapped to subcores.
  row_partition_id = core_id * num_subcores + subcore_id
  # Column partition is mapped to vector mesh.
  col_partition_id = core_id % num_column_partitions

  # Dimension sizes.
  shape = in_hbm_ref.shape
  col_size = shape[-1] // num_column_partitions
  
  num_simd_lanes = 8
  num_lanes = 128
  
  # Strides.
  row_partition_size = num_rows_per_row_partition_ref.shape[0] // (num_row_partitions)
  # The maximum number of source rows this subcore can process.
  num_rows_current_row_partition = row_partition_size

  # Semaphores.
  recv_sem = sem_ref[0]
  send_sem = sem_ref[1]

  def inner_kernel_corrected():
    num_rows_current_row_partition = jnp.array(0, jnp.int32)
    num_rows_per_row_partition = num_rows_per_row_partition_vmem_ref[...]
    for i in range(num_row_partitions):
      num_rows_current_row_partition = jnp.where(
          row_partition_id == i,
          num_rows_per_row_partition[i],
          num_rows_current_row_partition,
      )

    row_tile_size = num_simd_lanes
    num_row_tiles = pl.cdiv(num_rows_current_row_partition, row_tile_size)
    row_start = row_partition_id * row_partition_size
    col_start = col_partition_id * col_size

    @pl.loop(0, num_row_tiles)
    def row_loop(row_block_id):
      row_tile_start = row_start + row_block_id * num_simd_lanes
      prev_dst_row_hbm = jnp.where(row_block_id == 0, -1, dst_indices_vmem_ref[...][num_simd_lanes - 1])
      dma_list = []
      dma_list.append(
          pltpu.make_async_copy(
              src_indices_div_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              src_indices_div_vmem_ref,
              recv_sem,
          )
      )
      dma_list.append(
          pltpu.make_async_copy(
              src_indices_mod_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              src_indices_mod_vmem_ref,
              recv_sem,
          )
      )
      dma_list.append(
          pltpu.make_async_copy(
              dst_indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              dst_indices_vmem_ref,
              recv_sem,
          )
      )
      dma_list.append(
          pltpu.make_async_copy(
              topk_weights_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              topk_weights_vmem_ref,
              recv_sem,
          )
      )
      jax.tree.map(lambda x: x.start(), dma_list)
      jax.tree.map(lambda x: x.wait(), dma_list)

      src_indices_div = src_indices_div_vmem_ref[...]
      src_indices_mod = src_indices_mod_vmem_ref[...]
      dst_indices = dst_indices_vmem_ref[...]
      topk_weights = topk_weights_vmem_ref[...]

      in_32b_hbm_ref = in_hbm_ref
      out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

      loop_step = num_lanes

      for col_vmem_start in range(0, col_size, loop_step):
        col_hbm_start = col_start + col_vmem_start
        for row_vmem in range(num_simd_lanes):
          row_hbm = src_indices_div[row_vmem]
          pltpu.make_async_copy(
              in_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, 128)],
              in_vmem_ref.at[row_vmem, pl.ds(0, 128)],
              recv_sem,
          ).start()

      for col_vmem_start in range(0, col_size, loop_step):
        col_hbm_start = col_start + col_vmem_start
        packed_rows_registers = [[] for _ in range(num_simd_lanes)]

        for col_compute_offset in range(0, loop_step, num_simd_lanes):
          col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)
          previous_accumulated_data = None
          for row_src in range(num_simd_lanes):
            if is_bf16:
              is_aligned = (col_compute_offset % 16) == 0
              read_offset = col_compute_offset if is_aligned else col_compute_offset - 8
              data_16 = in_vmem_ref[row_src, pl.ds(read_offset, 16)]
              data = data_16[:8] if is_aligned else data_16[8:]
            else:
              data = in_vmem_ref[row_src, pl.ds(col_compute_offset, num_simd_lanes)]

            if is_bf16:
              row_src_pack = src_indices_mod[row_src]
              data = jnp.bitwise_left_shift(data, jnp.where(row_src_pack == 0, 16, 0))
              data = jnp.bitwise_and(data, -65536)
              data = jax.lax.bitcast_convert_type(data, jnp.float32)
            else:
              data = jax.lax.bitcast_convert_type(data, jnp.float32)

            data = data * topk_weights[row_src]
            dst_row_hbm = dst_indices[row_src]
            if row_src == 0:
              prev_row_hbm = prev_dst_row_hbm
              previous_accumulated_data = jax.lax.bitcast_convert_type(
                  prev_iter_last_row_vmem_ref[0, col_slice], jnp.float32
              )
            else:
              prev_row_hbm = dst_indices[row_src - 1]
              assert previous_accumulated_data is not None

            accumulated_data = jnp.where(
                dst_row_hbm == prev_row_hbm,
                previous_accumulated_data + data,
                data,
            )
            previous_accumulated_data = accumulated_data
            data_to_write = jax.lax.bitcast_convert_type(accumulated_data, jnp.uint32)
            if not is_bf16:
              flat_slice = pl.ds(row_src * col_size + col_vmem_start + col_compute_offset, num_simd_lanes)
              out_vmem_ref[0, flat_slice] = data_to_write

            if is_bf16:
              data_bf16 = accumulated_data.astype(jnp.bfloat16)
              data_u16 = jax.lax.bitcast_convert_type(data_bf16, jnp.uint16)
              data_u32 = data_u16.astype(jnp.uint32)
              packed_val = [
                  jnp.bitwise_or(jnp.bitwise_left_shift(data_u32[1], 16), data_u32[0]),
                  jnp.bitwise_or(jnp.bitwise_left_shift(data_u32[3], 16), data_u32[2]),
                  jnp.bitwise_or(jnp.bitwise_left_shift(data_u32[5], 16), data_u32[4]),
                  jnp.bitwise_or(jnp.bitwise_left_shift(data_u32[7], 16), data_u32[6]),
              ]
              packed_rows_registers[row_src].extend(packed_val)

            if row_src == num_simd_lanes - 1:
              prev_iter_last_row_vmem_ref[0, col_slice] = data_to_write

        src_row_idx_in_vmem = []
        row_valid_vec = []
        for row_vmem_idx in reversed(range(num_simd_lanes)):
          row_valid = row_block_id * num_simd_lanes + row_vmem_idx < num_rows_current_row_partition
          row_valid_vec.append(row_valid)
          if row_vmem_idx == num_simd_lanes - 1:
            src_row_idx_in_vmem.append(row_vmem_idx)
          else:
            next_row_valid = row_valid_vec[-2]
            src_row_idx_in_vmem.append(
                jnp.where(
                    jnp.logical_and(
                        next_row_valid,
                        (dst_indices[row_vmem_idx] == dst_indices[row_vmem_idx + 1]),
                    ),
                    src_row_idx_in_vmem[-1],
                    row_vmem_idx,
                )
            )
        src_row_idx_in_vmem.reverse()
        row_valid_vec.reverse()

        if is_bf16:
          for r in range(num_simd_lanes):
            packed_row = jnp.array(packed_rows_registers[r])
            write_col_vmem_start = pl.multiple_of(col_vmem_start // 2, 64)
            for c in range(0, 64, 8):
              out_vmem_ref[0, pl.ds(r * (col_size // 2) + write_col_vmem_start + c, 8)] = packed_row[c : c + 8]
            for c in range(64, 128, 8):
              out_vmem_ref[0, pl.ds(r * (col_size // 2) + write_col_vmem_start + c, 8)] = jnp.zeros((8,), jnp.uint32)

        last_valid_src_row_vmem = -1
        last_valid_dst_row_hbm = -1
        for i, (src_row_idx_in_vmem_val, row_valid) in enumerate(zip(src_row_idx_in_vmem, row_valid_vec, strict=True)):
          src_row_vmem = jnp.where(row_valid, src_row_idx_in_vmem_val, last_valid_src_row_vmem)
          dst_row_hbm = jnp.where(row_valid, dst_indices[i], last_valid_dst_row_hbm)
          
          if is_bf16:
            write_col_hbm_start = pl.multiple_of(col_partition_id * (col_size // 2) + (col_vmem_start // 2), 64)
            write_col_vmem_start = pl.multiple_of(col_vmem_start // 2, 64)
            flat_col_vmem_start_1 = pl.multiple_of(src_row_vmem * (col_size // 2) + write_col_vmem_start, 64)
            pltpu.make_async_copy(
                out_vmem_ref.at[0, pl.ds(flat_col_vmem_start_1, 64)],
                out_32b_hbm_ref.at[dst_row_hbm, pl.ds(write_col_hbm_start, 64)],
                send_sem,
            ).start()
          else:
            flat_col_vmem_start = pl.multiple_of(src_row_vmem * col_size + col_vmem_start, 128)
            pltpu.make_async_copy(
                out_vmem_ref.at[0, pl.ds(flat_col_vmem_start, 128)],
                out_32b_hbm_ref.at[dst_row_hbm, pl.ds(col_hbm_start, 128)],
                send_sem,
            ).start()
            
          last_valid_src_row_vmem = src_row_vmem
          last_valid_dst_row_hbm = dst_row_hbm

      for _ in range(0, col_size, num_lanes):
        for _ in range(num_simd_lanes):
          pltpu.make_async_copy(
              out_vmem_ref.at[0, :num_lanes],
              out_32b_hbm_ref.at[0, :num_lanes],
              send_sem,
          ).wait()

  inner_kernel_corrected()


def get_cost_estimate(
    padded_input_size: int,
    aligned_hidden_size: int,
    reduce_group_size: int,
    input_dtype_bytes: int,
    flops_override: int | None = None,
    bytes_accessed_override: int | None = None,
) -> pl.CostEstimate:
  """Get cost estimate for the ragged gather reduce kernel."""
  if flops_override is not None or bytes_accessed_override is not None:
    return pl.CostEstimate(
        flops=flops_override or 0,
        bytes_accessed=bytes_accessed_override or 0,
    )
  num_rows = padded_input_size // reduce_group_size
  flops = padded_input_size * aligned_hidden_size * 2
  bytes_accessed = (
      padded_input_size * aligned_hidden_size * input_dtype_bytes
      + num_rows * aligned_hidden_size * 4
  )
  return pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed)


_COMPILER_PARAMS = {
    "num_sc_buffers": 2,
    "ring_buffer_size": 1,
    "unroll_factor": 1,
}

_OUT_KW = "out"
_SCRATCH_KW = "scratch"


def _preprocess(
    indices: jnp.ndarray,
    topk_weights: jnp.ndarray,
    valid_rows_mask: jnp.ndarray,
    reduce_group_size: int,
    num_row_partitions: int,
):
  """Preprocess indices to group valid tokens at the beginning of each partition."""
  # Reshape valid_rows_mask to (num_row_partitions, partition_size)
  valid_rows_mask_2d = valid_rows_mask.reshape(num_row_partitions, -1)
  partition_size = valid_rows_mask_2d.shape[-1]
  
  # Stable sort so that valid tokens (True) come first, invalid (False) come last.
  # argsort(~mask) places True (invalid) last, so False (valid) comes first!
  sorted_by_validity = jnp.argsort(~valid_rows_mask_2d, descending=False, stable=True, axis=-1)
  
  # Convert partition-local sorted indices to global indices
  partition_offsets = jnp.arange(num_row_partitions)[:, jnp.newaxis] * partition_size
  sorted_by_validity += partition_offsets
  sorted_by_validity_flat = sorted_by_validity.reshape(-1)
  
  # Gather sorted indices and weights
  src_indices = indices[sorted_by_validity_flat]
  topk_weights_sorted = topk_weights[sorted_by_validity_flat]
  
  # Calculate dst_indices based on the original token IDs
  dst_indices = sorted_by_validity_flat // reduce_group_size
  
  # Calculate count of valid tokens per partition
  num_src_rows_per_row_partition = jnp.sum(valid_rows_mask_2d, axis=-1)
  
  return src_indices, dst_indices, topk_weights_sorted, num_src_rows_per_row_partition


@functools.partial(
    jax.jit,
    static_argnums=(4,), # reduce_group_size
    static_argnames=("flops_override", "bytes_accessed_override", "enforce_fallback"),
)
def ragged_gather_reduce(
    x: jnp.ndarray,
    indices: jnp.ndarray,
    topk_weights: jnp.ndarray,
    valid_rows_mask: jnp.ndarray,
    reduce_group_size: int,
    flops_override: int | None = None,
    bytes_accessed_override: int | None = None,
    enforce_fallback: bool = False,
) -> jnp.ndarray:
  """Wrapper for the ragged gather reduce SparseCore kernel."""
  sc_info = plsc.get_sparse_core_info()
  
  # Calculate padded_input_size locally based on the shape of indices!
  padded_input_size = indices.shape[0]

  if sc_info is None or enforce_fallback:
    return _ragged_gather_reduce_fallback(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size, padded_input_size
    )

  input_size, hidden_size = x.shape
  is_bf16 = x.dtype == jnp.bfloat16

  num_simd_lanes = 8
  num_rows_partitions = sc_info.num_cores * 2 # 2 subcores per core
  num_column_partitions = sc_info.num_cores

  # Preprocess indices using the valid_rows_mask.
  # This sorts the tokens so all valid ones come first, and calculates the exact
  # valid token count per partition, allowing the kernel to skip invalid tokens!
  src_indices, dst_indices, topk_weights_sorted, num_src_rows_per_row_partition = _preprocess(
      indices,
      topk_weights,
      valid_rows_mask,
      reduce_group_size,
      num_rows_partitions,
  )

  if is_bf16:
    # Outer Input Bitcast: convert bfloat16 to uint16 (same width, supported!),
    # and then perform bitwise packing to uint32 in JAX space!
    # This transforms the input into a pure 32-bit HBM tensor, completely bypassing
    # all boundary element-size-changing layout cast crashes!
    x_u16 = jax.lax.bitcast_convert_type(x, jnp.uint16)
    even_u32 = x_u16[:, 0::2].astype(jnp.uint32)
    odd_u32 = x_u16[:, 1::2].astype(jnp.uint32)
    x_input = jnp.bitwise_or(jnp.bitwise_left_shift(odd_u32, 16), even_u32)
    dtype_bytes = 2
  else:
    x_input = x
    dtype_bytes = 4

  dtype = x_input.dtype

  # col_size is the column size of the 32-bit reference!
  # For bf16, hidden_size is 4096, but x_input has 2048 columns!
  # So col_size = 2048 / num_column_partitions = 256.
  # For f32, hidden_size is 4096, and x_input has 4096 columns!
  # So col_size = 4096 / num_column_partitions = 512.
  aligned_hidden_size = x_input.shape[-1]
  col_size = aligned_hidden_size // num_column_partitions

  # Pre-calculate divided and modulo indices in JAX to completely
  # avoid tracer division and modulo arithmetic inside the kernel!
  input_packing = 2 if is_bf16 else 1
  src_indices_div = src_indices // input_packing
  src_indices_mod = src_indices % input_packing

  # Output shape: (R, C // 2) of float32 (carrier) for bf16, and (R, C) for f32.
  out_shape = (
      padded_input_size // reduce_group_size,
      aligned_hidden_size,
  )
  out_dtype = jnp.float32

  # Wrap the sharding mesh locally using sc_info (completely self-contained!)
  vector_mesh_wrapped = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=2,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  out = pl.kernel(  # pytype: disable=wrong-keyword-args
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh_wrapped.core_axis_name,
          subcore_axis_name=vector_mesh_wrapped.subcore_axis_name,
          num_row_partitions=num_rows_partitions,
          num_column_partitions=num_column_partitions,
          is_bf16=is_bf16,
      ),
      compiler_params=pltpu.CompilerParams(  # pytype: disable=wrong-keyword-args
          **_COMPILER_PARAMS,
      ),
      cost_estimate=get_cost_estimate(
          padded_input_size=padded_input_size,
          aligned_hidden_size=aligned_hidden_size,
          reduce_group_size=reduce_group_size,
          input_dtype_bytes=dtype_bytes,
          flops_override=flops_override,
          bytes_accessed_override=bytes_accessed_override,
      ),
      mesh=vector_mesh_wrapped,
      name="sc_ragged_gather_reduce",
      **{
          _OUT_KW: jax.ShapeDtypeStruct(
              out_shape,
              out_dtype,
          ),
          _SCRATCH_KW: dict(  # pylint: disable=use-dict-literal
              num_rows_per_row_partition_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              in_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32), # 2D uint32 buffer (squeezing/slicing is supported!)
              out_vmem_ref=pltpu.VMEM((1, num_simd_lanes * col_size), jnp.uint32), # 1D uint32 buffer (squeezing is supported!)
              prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size), jnp.uint32),
              src_indices_div_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              src_indices_mod_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.float32),
              sem_ref=pltpu.SemaphoreType.DMA((2,)),
          ),
      },
  )(
      num_src_rows_per_row_partition,
      x_input,
      src_indices_div,
      src_indices_mod,
      dst_indices,
      topk_weights_sorted,
  )

  if is_bf16:
    # Outer Output Unpacking: bitcast to uint32, extract bits, cast to uint16,
    # bitcast to bf16, and interleave via stacking and reshaping!
    out_u32 = jax.lax.bitcast_convert_type(out, jnp.uint32)
    even_u16 = jnp.bitwise_and(out_u32, 65535).astype(jnp.uint16)
    odd_u16 = jnp.bitwise_right_shift(out_u32, 16).astype(jnp.uint16)
    even_bf16 = jax.lax.bitcast_convert_type(even_u16, jnp.bfloat16)
    odd_bf16 = jax.lax.bitcast_convert_type(odd_u16, jnp.bfloat16)
    
    # Interleave even and odd columns!
    stacked = jnp.stack([even_bf16, odd_bf16], axis=-1)
    out = stacked.reshape(padded_input_size // reduce_group_size, hidden_size)
  else:
    out = out.astype(x.dtype)

  return out[: (input_size // reduce_group_size), :hidden_size]
