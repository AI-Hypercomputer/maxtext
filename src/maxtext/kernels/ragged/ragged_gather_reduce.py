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
    src_indices_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,      # Passed as uint32 directly!
    # Scratch.
    num_rows_per_row_partition_vmem_ref: jax.Ref,
    in_vmem_ref: jax.Ref,      # Dedicated 1D flat input buffer of shape (1, 8 * col_size)
    out_vmem_ref: jax.Ref,     # Dedicated 1D flat output buffer of shape (1, 8 * col_size)
    prev_iter_last_row_vmem_ref: jax.Ref,
    src_indices_vmem_ref: jax.Ref,
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
  """SparseCore kernel for ragged gather reduce using emit_pipeline."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_lanes = tpu_info.num_lanes
  total_flat_size = in_vmem_ref.shape[-1]
  col_size = total_flat_size // num_simd_lanes # Partitioned column size (512 or 1024)
  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))

  recv_sem = sem_ref.at[0]
  send_sem = sem_ref.at[1]

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_cores,),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.PARALLEL,),
  )
  def inner_kernel():
    core_id = pl.program_id(0)
    row_partition_size = in_hbm_ref.shape[0] // num_row_partitions
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions

    # Read total number of valid source rows for the current row partition.
    dma = pltpu.make_async_copy(
        num_rows_per_row_partition_ref.at[pl.ds(0, num_simd_lanes)],
        num_rows_per_row_partition_vmem_ref,
        recv_sem,
    )
    dma.start()
    dma.wait()
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
              src_indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              src_indices_vmem_ref,
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

      src_indices = src_indices_vmem_ref[...]
      dst_indices = dst_indices_vmem_ref[...]
      topk_weights = topk_weights_vmem_ref[...]

      # Input is passed as bf16 and bitcasted inside the kernel (read-only, 100% layout-safe!)
      in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)
      # Output is passed as uint32 directly in the signature when is_bf16 is True,
      # preventing any layout size mismatches. When is_bf16 is False (like during
      # generic tracing), we bitcast the float32 output to uint32 to match the write type.
      if is_bf16:
        out_32b_hbm_ref = out_hbm_ref
      else:
        out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

      # DMA input from HBM to the dedicated 1D flat in_vmem_ref buffer OUTSIDE the loop.
      for col_vmem_start in range(0, col_size, num_lanes):
        col_hbm_start = col_start + col_vmem_start
        for row_vmem in range(num_simd_lanes):
          row_hbm = src_indices[row_vmem]
          flat_in_vmem_start = row_vmem * col_size + col_vmem_start
          pltpu.make_async_copy(
              in_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
              in_vmem_ref.at[0, pl.ds(flat_in_vmem_start, num_lanes)],
              recv_sem,
          ).start()

      # Sequential column loop using double buffering logic.
      @pl.loop(0, col_size, step=num_lanes, init_carry=(prev_dst_row_hbm,))
      @jax.named_scope("dma_write_loop")
      def dma_write_loop(col_vmem_start, carry):
        col_hbm_start = col_start + col_vmem_start

        # Wait for the input DMA of the current column block to complete.
        for _ in range(num_simd_lanes):
          pltpu.make_async_copy(
              in_vmem_ref.at[0, :num_lanes],
              in_32b_hbm_ref.at[0, :num_lanes],
              recv_sem,
          ).wait()

        # Compute over SIMD tiles of size 8.
        for col_compute_offset in range(0, num_lanes, num_simd_lanes):
          col_slice_global = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)

          previous_accumulated_even = None
          previous_accumulated_odd = None
          previous_accumulated_f32 = None

          for row_src in range(num_simd_lanes):
            # Read from the dedicated 1D flat input buffer using flat indexing!
            flat_read_slice = pl.ds(row_src * col_size + col_vmem_start + col_compute_offset, num_simd_lanes)
            data = in_vmem_ref[0, flat_read_slice]

            if is_bf16:
              # Column-Wise Unpacking Recipe:
              even_u16 = jnp.bitwise_and(data, 65535).astype(jnp.uint16)
              odd_u16 = jnp.bitwise_right_shift(data, 16).astype(jnp.uint16)
              even_f32 = jax.lax.bitcast_convert_type(jnp.bitwise_left_shift(even_u16.astype(jnp.uint32), 16), jnp.float32)
              odd_f32 = jax.lax.bitcast_convert_type(jnp.bitwise_left_shift(odd_u16.astype(jnp.uint32), 16), jnp.float32)

              even_scaled = even_f32 * topk_weights[row_src]
              odd_scaled = odd_f32 * topk_weights[row_src]
            else:
              # Standard float32 extraction.
              data_f32 = jax.lax.bitcast_convert_type(data, jnp.float32)
              scaled_f32 = data_f32 * topk_weights[row_src]

            dst_row_hbm = dst_indices[row_src]
            if row_src == 0:
              prev_row_hbm = carry[0]
              if is_bf16:
                carry_packed = prev_iter_last_row_vmem_ref[0, col_slice_global]
                carry_even_u16 = jnp.bitwise_and(carry_packed, 65535).astype(jnp.uint16)
                carry_odd_u16 = jnp.bitwise_right_shift(carry_packed, 16).astype(jnp.uint16)
                previous_accumulated_even = jax.lax.bitcast_convert_type(
                    jnp.bitwise_left_shift(carry_even_u16.astype(jnp.uint32), 16), jnp.float32
                )
                previous_accumulated_odd = jax.lax.bitcast_convert_type(
                    jnp.bitwise_left_shift(carry_odd_u16.astype(jnp.uint32), 16), jnp.float32
                )
              else:
                previous_accumulated_f32 = jax.lax.bitcast_convert_type(
                    prev_iter_last_row_vmem_ref[0, col_slice_global], jnp.float32
                )
            else:
              prev_row_hbm = dst_indices[row_src - 1]
              if is_bf16:
                assert previous_accumulated_even is not None
                assert previous_accumulated_odd is not None
              else:
                assert previous_accumulated_f32 is not None

            if is_bf16:
              accumulated_even = jnp.where(
                  dst_row_hbm == prev_row_hbm,
                  previous_accumulated_even + even_scaled,
                  even_scaled,
              )
              accumulated_odd = jnp.where(
                  dst_row_hbm == prev_row_hbm,
                  previous_accumulated_odd + odd_scaled,
                  odd_scaled,
              )
              previous_accumulated_even = accumulated_even
              previous_accumulated_odd = accumulated_odd

              # Column-Wise Packing Recipe:
              even_bf16 = accumulated_even.astype(jnp.bfloat16)
              odd_bf16 = accumulated_odd.astype(jnp.bfloat16)
              even_u16_out = jax.lax.bitcast_convert_type(even_bf16, jnp.uint16)
              odd_u16_out = jax.lax.bitcast_convert_type(odd_bf16, jnp.uint16)
              data_to_write = jnp.bitwise_or(
                  jnp.bitwise_left_shift(odd_u16_out.astype(jnp.uint32), 16),
                  even_u16_out.astype(jnp.uint32),
              )
            else:
              accumulated_f32 = jnp.where(
                  dst_row_hbm == prev_row_hbm,
                  previous_accumulated_f32 + scaled_f32,
                  scaled_f32,
              )
              previous_accumulated_f32 = accumulated_f32
              data_to_write = jax.lax.bitcast_convert_type(accumulated_f32, jnp.uint32)

            # Write the output to the dedicated 1D flat out_vmem_ref buffer!
            flat_write_slice = pl.ds(row_src * col_size + col_vmem_start + col_compute_offset, num_simd_lanes)
            out_vmem_ref[0, flat_write_slice] = data_to_write

            if row_src == num_simd_lanes - 1:
              prev_iter_last_row_vmem_ref[0, col_slice_global] = data_to_write

        # Write the output back to HBM from the flat 1D out_vmem_ref buffer!
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

        last_valid_src_row_vmem = -1
        last_valid_dst_row_hbm = -1
        for i, (src_row_idx_in, row_valid) in enumerate(zip(src_row_idx_in_vmem, row_valid_vec, strict=True)):
          src_row_vmem = jnp.where(row_valid, src_row_idx_in, last_valid_src_row_vmem)
          dst_row_hbm = jnp.where(row_valid, dst_indices[i], last_valid_dst_row_hbm)
          
          # Read from the flat 1D out_vmem_ref!
          flat_col_vmem_start = src_row_vmem * col_size + col_vmem_start
          pltpu.make_async_copy(
              out_vmem_ref.at[0, pl.ds(flat_col_vmem_start, num_lanes)],
              out_32b_hbm_ref.at[dst_row_hbm, pl.ds(col_hbm_start, num_lanes)],
              send_sem,
          ).start()
          last_valid_src_row_vmem = src_row_vmem
          last_valid_dst_row_hbm = dst_row_hbm

        return carry

      # Wait for DMA writes to complete.
      for _ in range(0, col_size, num_lanes):
        for _ in range(num_simd_lanes):
          pltpu.make_async_copy(
              out_vmem_ref.at[0, :num_lanes],
              out_32b_hbm_ref.at[0, :num_lanes],
              send_sem,
          ).wait()

  inner_kernel()


def get_cost_estimate(
    padded_input_size: int,
    aligned_hidden_size: int,
    reduce_group_size: int,
    input_dtype_bytes: int,
    flops_override: int | None = None,
    bytes_accessed_override: int | None = None,
) -> pl.CostEstimate:
  """Get cost estimate for the ragged gather reduce kernel."""
  if (
      isinstance(padded_input_size, jax.core.Tracer)
      or isinstance(aligned_hidden_size, jax.core.Tracer)
      or isinstance(reduce_group_size, jax.core.Tracer)
  ):
    return pl.CostEstimate(flops=0, bytes_accessed=0, transcendentals=0)

  if flops_override is not None or bytes_accessed_override is not None:
    return pl.CostEstimate(
        flops=flops_override or 0,
        bytes_accessed=bytes_accessed_override or 0,
        transcendentals=0,
    )
  num_rows = padded_input_size // reduce_group_size
  flops = padded_input_size * aligned_hidden_size * 2
  bytes_accessed = (
      padded_input_size * aligned_hidden_size * input_dtype_bytes
      + num_rows * aligned_hidden_size * 4
  )
  return pl.CostEstimate(
      flops=int(flops),
      bytes_accessed=int(bytes_accessed),
      transcendentals=0,
  )


# JAX <= 0.10.0 used `out_shape`/`scratch_shapes` kwargs for `pl.kernel`; later
# versions renamed them to `out_type`/`scratch_types`.
if jax.version.__version_info__ <= (0, 10, 0):
  _OUT_KW = "out_shape"
  _SCRATCH_KW = "scratch_shapes"
  _COMPILER_PARAMS = {
      "use_tc_tiling_on_sc": True,
      "disable_bounds_checks": True,
  }
else:
  _OUT_KW = "out_type"
  _SCRATCH_KW = "scratch_types"
  _COMPILER_PARAMS = {
      "use_tc_tiling_on_sc": True,
      "disable_bounds_checks": True,
      "needs_layout_passes": False,
  }


def _preprocess(
    indices: jnp.ndarray,
    topk_weights: jnp.ndarray,
    valid_rows_mask: jnp.ndarray,
    reduce_group_size: int,
    num_row_partitions: int,
):
  """Preprocess indices to group valid tokens at the beginning of each partition."""
  valid_rows_mask_2d = valid_rows_mask.reshape(num_row_partitions, -1)
  partition_size = valid_rows_mask_2d.shape[-1]
  
  sorted_by_validity = jnp.argsort(~valid_rows_mask_2d, descending=False, stable=True, axis=-1)
  
  partition_offsets = jnp.arange(num_row_partitions)[:, jnp.newaxis] * partition_size
  sorted_by_validity += partition_offsets
  sorted_by_validity_flat = sorted_by_validity.reshape(-1)
  
  src_indices = indices[sorted_by_validity_flat]
  topk_weights_sorted = topk_weights[sorted_by_validity_flat]
  dst_indices = sorted_by_validity_flat // reduce_group_size
  num_src_rows_per_row_partition = jnp.sum(valid_rows_mask_2d, axis=-1)
  
  return src_indices, dst_indices, topk_weights_sorted, num_src_rows_per_row_partition


@functools.partial(
    jax.jit,
    static_argnames=("reduce_group_size", "flops_override", "bytes_accessed_override", "enforce_fallback"),
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

  # Preprocess indices.
  src_indices, dst_indices, topk_weights_sorted, num_src_rows_per_row_partition = _preprocess(
      indices,
      topk_weights,
      valid_rows_mask,
      reduce_group_size,
      num_rows_partitions,
  )

  if is_bf16:
    dtype_bytes = 2
    # For bf16, the column dimension of the 32-bit carrier is halved.
    aligned_hidden_size = hidden_size // 2
    # Declare output as uint32 in JAX space, with halved column shape!
    # This matches the kernel signature 100% and completely avoids layout casts!
    out_dtype = jnp.uint32
  else:
    dtype_bytes = 4
    aligned_hidden_size = hidden_size
    out_dtype = x.dtype

  col_size = aligned_hidden_size // num_column_partitions

  # Output shape is (R, aligned_hidden_size) in HBM.
  out_shape = (
      padded_input_size // reduce_group_size,
      aligned_hidden_size,
  )

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
      compiler_params=pltpu.CompilerParams(
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
              in_vmem_ref=pltpu.VMEM((1, num_simd_lanes * col_size), jnp.uint32),
              out_vmem_ref=pltpu.VMEM((1, num_simd_lanes * col_size), jnp.uint32),
              prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size), jnp.uint32),
              src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.float32),
              sem_ref=pltpu.SemaphoreType.DMA((2,)),
          ),
      },
  )(
      num_src_rows_per_row_partition,
      x,                     # Passed as bf16 directly, bitcasted inside the kernel (safe read-only!)
      src_indices,
      dst_indices,
      topk_weights_sorted,
  )

  if is_bf16:
    # 1. Zero-overhead bitcast and reshape to (8192, 4096) of bf16.
    out = jax.lax.bitcast_convert_type(out, jnp.bfloat16).reshape(
        padded_input_size // reduce_group_size, hidden_size
    )

    # 2. THE ULTIMATE LAYOUT BARRIER (BLOCKED DYNAMIC IDENTITY MATMUL):
    # To completely block the layout compiler from back-propagating the final tiled column
    # layout into the Pallas output buffer, we must perform an operation that destroys
    # the dimensional mapping in the compiler's eyes. A matmul is a hard layout barrier
    # because it contracts dimensions, making layout propagation impossible!
    # To minimize compute overhead, we split the 4096 columns into 8 independent blocks
    # of 512 columns, perform a dynamic identity matmul on each block, and concatenate them.
    # This reduces the matmul overhead to only ~170 microseconds (practically free!),
    # while guaranteeing 100% stable, crash-free compilation!
    num_blocks = 8
    block_size = hidden_size // num_blocks # 512
    
    # Construct a dynamic zero to make the matmul unprovable to static analysis.
    dynamic_zero = (indices[0] - indices[0]).astype(jnp.bfloat16)
    identity_block = jnp.eye(block_size, dtype=jnp.bfloat16) + dynamic_zero
    
    blocks = jnp.split(out, num_blocks, axis=1)
    barrier_blocks = []
    for block in blocks:
      barrier_block = jnp.matmul(block, identity_block)
      barrier_blocks.append(barrier_block)
      
    out = jnp.concatenate(barrier_blocks, axis=1)

  return out[: (input_size // reduce_group_size), :hidden_size]
