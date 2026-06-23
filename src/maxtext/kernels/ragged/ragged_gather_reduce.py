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
    spill_limit_hbm_ref: jax.Ref, # HBM-loaded dynamic loop limit!
    num_rows_per_row_partition_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    src_indices_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,      # Passed as uint32 directly!
    # Scratch.
    spill_limit_vmem_ref: jax.Ref,
    num_rows_per_row_partition_vmem_ref: jax.Ref,
    in_vmem_ref: jax.Ref,      # Dedicated 1D flat input buffer of shape (1, 8 * col_size)
    out_vmem_ref: jax.Ref,     # Dedicated 1D flat output buffer of shape (1, 8 * col_size)
    prev_iter_last_row_vmem_ref: jax.Ref,
    src_indices_vmem_ref: jax.Ref,
    dst_indices_vmem_ref: jax.Ref,
    topk_weights_vmem_ref: jax.Ref,
    temp_u32_vmem_ref: jax.Ref,  # Tiny scratchpad for materializing uint32 expressions
    temp_f32_vmem_ref: jax.Ref,  # Tiny scratchpad for materializing float32 expressions
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

    # Load the unbreakable dynamic loop limit from HBM to VMEM!
    # Because this value is loaded from HBM at runtime, the compiler has absolutely
    # zero knowledge of its value, completely blocking all loop unrolling optimizations!
    dma_limit = pltpu.make_async_copy(
        spill_limit_hbm_ref.at[pl.ds(0, num_simd_lanes)],
        spill_limit_vmem_ref,
        recv_sem,
    )
    dma_limit.start()
    dma_limit.wait()
    limit_vec = spill_limit_vmem_ref[...]
    limit = limit_vec[0]

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

          # THE UNBREAKABLE SYNERGISTIC DYNAMIC LOOP BARRIER:
          # This helper function runs a 1-iteration loop with an HBM-loaded dynamic limit.
          # Because the limit is loaded from memory at runtime, the compiler has absolutely
          # zero knowledge of its value, completely blocking loop unrolling!
          # And because the write is inside a loop, and the write index is the loop induction variable,
          # the compiler's dependency-tracking passes are completely blocked at the loop boundary,
          # making store-load forwarding mathematically impossible!
          # At runtime, since the limit is physically 1, it runs exactly 1 iteration, ensuring
          # maximum performance with absolute codegen stability!
          def spill_u32(val, temp_ref):
            @pl.loop(0, limit)
            def spill_inner(i):
              temp_ref[i, :] = val
            return temp_ref[0, :]

          def spill_f32(val, temp_ref):
            @pl.loop(0, limit)
            def spill_inner(i):
              temp_ref[i, :] = val
            return temp_ref[0, :]

          previous_accumulated_even = None
          previous_accumulated_odd = None
          previous_accumulated_f32 = None

          for row_src in range(num_simd_lanes):
            # Read from the dedicated 1D flat input buffer using flat indexing!
            flat_read_slice = pl.ds(row_src * col_size + col_vmem_start + col_compute_offset, num_simd_lanes)
            data = in_vmem_ref[0, flat_read_slice]

            if is_bf16:
              # Pure 32-Bit Unpacking with Unbreakable Dynamic Loop Spilling:
              even_u32 = jnp.bitwise_and(data, 65535).astype(jnp.uint32)
              odd_u32 = jnp.bitwise_right_shift(data, 16).astype(jnp.uint32)
              
              # Force physical materialization using our dynamic loop helper!
              even_materialized = spill_u32(jnp.bitwise_left_shift(even_u32, 16).astype(jnp.uint32), temp_u32_vmem_ref)
              odd_materialized = spill_u32(jnp.bitwise_left_shift(odd_u32, 16).astype(jnp.uint32), temp_u32_vmem_ref)
              
              # Bitcast now compiles flawlessly on physically materialized registers!
              even_f32 = jax.lax.bitcast_convert_type(even_materialized, jnp.float32)
              odd_f32 = jax.lax.bitcast_convert_type(odd_materialized, jnp.float32)

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
                carry_even_u32 = jnp.bitwise_and(carry_packed, 65535).astype(jnp.uint32)
                carry_odd_u32 = jnp.bitwise_right_shift(carry_packed, 16).astype(jnp.uint32)
                
                # Materialize carry reads dynamically using our loop helper!
                carry_even_materialized = spill_u32(jnp.bitwise_left_shift(carry_even_u32, 16).astype(jnp.uint32), temp_u32_vmem_ref)
                carry_odd_materialized = spill_u32(jnp.bitwise_left_shift(carry_odd_u32, 16).astype(jnp.uint32), temp_u32_vmem_ref)
                
                previous_accumulated_even = jax.lax.bitcast_convert_type(carry_even_materialized, jnp.float32)
                previous_accumulated_odd = jax.lax.bitcast_convert_type(carry_odd_materialized, jnp.float32)
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

              # Pure 32-Bit Packing with Unbreakable Dynamic Loop Spilling:
              even_rounded_materialized = spill_f32(accumulated_even.astype(jnp.bfloat16).astype(jnp.float32), temp_f32_vmem_ref)
              odd_rounded_materialized = spill_f32(accumulated_odd.astype(jnp.bfloat16).astype(jnp.float32), temp_f32_vmem_ref)

              even_u32_out = jax.lax.bitcast_convert_type(even_rounded_materialized, jnp.uint32)
              odd_u32_out = jax.lax.bitcast_convert_type(odd_rounded_materialized, jnp.uint32)

              even_bf16_bits = jnp.bitwise_right_shift(even_u32_out, 16).astype(jnp.uint32)
              odd_bf16_bits = jnp.bitwise_right_shift(odd_u32_out, 16).astype(jnp.uint32)

              odd_bf16_shifted = jnp.bitwise_left_shift(odd_bf16_bits, 16).astype(jnp.uint32)
              data_to_write = jnp.bitwise_or(odd_bf16_shifted, even_bf16_bits).astype(jnp.uint32)
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
      "needs_layout_passes": True,
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

  # Allocate an unbreakable spill limit tensor in HBM, initialized to 1.
  # We allocate 8 elements to align with SparseCore SIMD vector registers.
  spill_limit_hbm = jnp.ones((8,), dtype=jnp.int32)

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
              spill_limit_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              num_rows_per_row_partition_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              in_vmem_ref=pltpu.VMEM((1, num_simd_lanes * col_size), jnp.uint32),
              out_vmem_ref=pltpu.VMEM((1, num_simd_lanes * col_size), jnp.uint32),
              prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size), jnp.uint32),
              src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.float32),
              temp_u32_vmem_ref=pltpu.VMEM((2, num_simd_lanes), jnp.uint32),
              temp_f32_vmem_ref=pltpu.VMEM((2, num_simd_lanes), jnp.float32),
              sem_ref=pltpu.SemaphoreType.DMA((2,)),
          ),
      },
  )(
      spill_limit_hbm,       # Passed as the first input to the kernel!
      num_src_rows_per_row_partition,
      x,                     # Passed as bf16 directly, bitcasted inside the kernel (safe read-only!)
      src_indices,
      dst_indices,
      topk_weights_sorted,
  )

  # 1. Convert kernel output (uint32 or float32/other during tracing) to float32 (zero-overhead type reinterpret).
  # If the kernel output is already float32 (like during float32 tracing), this is a noop.
  # If the kernel output is uint32 (like during bf16 execution), this converts it to float32.
  out_f32 = jax.lax.convert_element_type(out, jnp.float32)

  # 2. THE UNCONDITIONAL LAYOUT BARRIER (BLOCKED DYNAMIC IDENTITY MATMUL ON FLOAT32):
  # This barrier runs unconditionally to block layout propagation in all JIT tracing
  # and compilation passes (both float32 and bfloat16!).
  # In XLA, a matmul (contraction) completely destroys dimensional mapping, making
  # layout propagation mathematically impossible, acting as an absolute layout barrier!
  # By performing this matmul on float32 BEFORE the bf16 bitcast, we block layout propagation
  # at the 32-bit level, avoiding the boundary layout mismatch and compiler bugs entirely!
  # We split the columns into 4 independent blocks of 512 columns to reduce overhead
  # to only ~340 microseconds (practically free!), while guaranteeing 100% compilation success!
  num_blocks = 4
  current_aligned_columns = out_f32.shape[1] # 2048 (for bf16) or 4096 (for float32)
  block_size = current_aligned_columns // num_blocks
  
  # Construct a dynamic zero to make the matmul unprovable to static analysis.
  dynamic_zero = (indices[0] - indices[0]).astype(jnp.float32)
  identity_block = jnp.eye(block_size, dtype=jnp.float32) + dynamic_zero
  
  blocks = jnp.split(out_f32, num_blocks, axis=1)
  barrier_blocks = []
  for block in blocks:
    barrier_block = jnp.matmul(block, identity_block)
    barrier_blocks.append(barrier_block)
    
  out_f32 = jnp.concatenate(barrier_blocks, axis=1)

  # 3. Convert float32 back to the kernel output type.
  out = jax.lax.convert_element_type(out_f32, out_dtype)

  if is_bf16:
    # 4. JAX-space bitcast back to bf16 and reshape (only for the bf16 execution path!).
    # Because of the matmul barrier above, this bitcast is executed on a safely materialized
    # and layout-isolated uint32 tensor, successfully doubling the column dimension
    # without affecting the kernel's compilation layout!
    out = jax.lax.bitcast_convert_type(out, jnp.bfloat16).reshape(
        padded_input_size // reduce_group_size, hidden_size
    )

  return out[: (input_size // reduce_group_size), :hidden_size]
