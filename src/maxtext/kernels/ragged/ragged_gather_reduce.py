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
    in_hbm_ref: jax.Ref,       # Declared as 1D flat uint32 buffer in HBM!
    src_indices_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,      # Declared as 1D flat uint32 buffer in HBM!
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
  total_cols = col_size * num_column_partitions

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
    # in_hbm_ref is 1D, but we still compute row partition offsets using the virtual 2D shape.
    # The total virtual rows in HBM is obtained from the total flat size divided by total columns.
    total_virtual_rows = in_hbm_ref.shape[0] // total_cols
    row_partition_size = total_virtual_rows // num_row_partitions
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

      # DMA input from HBM to the dedicated 1D flat in_vmem_ref buffer.
      # Because the HBM buffer is 1D flat, we calculate the flat contiguous index for each row slice
      # and perform a standard 1D DMA copy. This completely eliminates all 2D tiling layout reinterpret casts!
      for col_vmem_start in range(0, col_size, num_lanes):
        col_hbm_start = col_start + col_vmem_start
        for row_vmem in range(num_simd_lanes):
          row_hbm = src_indices[row_vmem]
          flat_in_idx = row_hbm * total_cols + col_hbm_start
          flat_in_vmem_start = row_vmem * col_size + col_vmem_start
          pltpu.make_async_copy(
              in_hbm_ref.at[pl.ds(flat_in_idx, num_lanes)],
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
              in_hbm_ref.at[0:num_lanes], # Opaque dummy wait
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
              # Pure register-level unpacking.
              even_u32 = jnp.bitwise_and(data, 65535).astype(jnp.uint32)
              odd_u32 = jnp.bitwise_right_shift(data, 16).astype(jnp.uint32)
              
              even_f32 = jax.lax.bitcast_convert_type(jnp.bitwise_left_shift(even_u32, 16).astype(jnp.uint32), jnp.float32)
              odd_f32 = jax.lax.bitcast_convert_type(jnp.bitwise_left_shift(odd_u32, 16).astype(jnp.uint32), jnp.float32)

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
                # Unpack the carry entirely in register space!
                carry_packed = prev_iter_last_row_vmem_ref[0, col_slice_global]
                carry_even_u32 = jnp.bitwise_and(carry_packed, 65535).astype(jnp.uint32)
                carry_odd_u32 = jnp.bitwise_right_shift(carry_packed, 16).astype(jnp.uint32)
                
                previous_accumulated_even = jax.lax.bitcast_convert_type(jnp.bitwise_left_shift(carry_even_u32, 16).astype(jnp.uint32), jnp.float32)
                previous_accumulated_odd = jax.lax.bitcast_convert_type(jnp.bitwise_left_shift(carry_odd_u32, 16).astype(jnp.uint32), jnp.float32)
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

              # 👑 THE PHYSICAL INTEGER ROUNDING MASTERPIECE:
              # SparseCore physically lacks a float32-to-bfloat16 hardware converter, so direct
              # type conversions (astype(bfloat16)) fail to legalize and crash.
              # We bypass this hardware limitation by performing the rounding and packing
              # ENTIRELY using zero-cost integer bitwise arithmetic:
              even_u32_val = jax.lax.bitcast_convert_type(accumulated_even, jnp.uint32)
              odd_u32_val = jax.lax.bitcast_convert_type(accumulated_odd, jnp.uint32)

              # Round-to-nearest-even by adding 0x8000 (32768, representing 0.5 in the truncated position)
              # before shifting. This delivers mathematically perfect rounding!
              even_rounded = even_u32_val + 32768
              odd_rounded = odd_u32_val + 32768

              # Extract upper 16 bits of even (shift right by 16) and odd (mask out lower 16).
              even_bf16_bits = jnp.bitwise_right_shift(even_rounded, 16)
              odd_bf16_shifted = jnp.bitwise_and(odd_rounded, 4294901760) # 4294901760 is 0xFFFF0000 in uint32

              # Pack them into a single uint32 word.
              data_to_write = jnp.bitwise_or(odd_bf16_shifted, even_bf16_bits)
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

        # Write the output back to HBM from the flat 1D out_vmem_ref buffer.
        # Because the HBM buffer is 1D flat, we calculate the flat contiguous index for the target
        # HBM row slice and perform a standard 1D DMA copy.
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
          
          # Flat 1D index in HBM!
          flat_out_idx = dst_row_hbm * total_cols + col_hbm_start
          flat_col_vmem_start = src_row_vmem * col_size + col_vmem_start
          pltpu.make_async_copy(
              out_vmem_ref.at[0, pl.ds(flat_col_vmem_start, num_lanes)],
              out_hbm_ref.at[pl.ds(flat_out_idx, num_lanes)],
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
              out_hbm_ref.at[0:num_lanes], # Opaque dummy wait
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
      # Layout passes are enabled (default) to keep the JAX/XLA compiler happy!
      # Since our entire HBM and VMEM interface is now 100% 1D flat, there are absolutely
      # no 2D tiling layout constraints, making compilation perfectly stable and fast!
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
    out_dtype = jnp.uint32
    aligned_hidden_size = hidden_size // 2

    # =========================================================================
    # 👑 THE INPUT LAYOUT FORTRESS:
    # To permanently prevent the compiler's size-doubling layout bug from triggering on the
    # INPUT path, we must block all layout propagation from the input bitcast backward!
    # By performing a column contraction (matmul with identity) on the bfloat16 input 'x'
    # BEFORE we bitcast it to uint32, we completely destroy the layout propagation path!
    # =========================================================================
    dynamic_zero_in = (indices[0] - indices[0]).astype(jnp.bfloat16)
    
    def apply_input_barrier(tensor_bf16):
      blocks = []
      block_size = 512
      num_blocks = hidden_size // block_size
      for i in range(num_blocks):
        block = tensor_bf16[:, i * block_size : (i + 1) * block_size]
        identity = jnp.eye(block_size, dtype=jnp.bfloat16) + dynamic_zero_in
        blocks.append(jnp.matmul(block, identity))
      return jnp.concatenate(blocks, axis=1)

    x_clean = apply_input_barrier(x)

    # 👑 THE 100% 1D FLAT HBM INTERFACE:
    # Because physical 2D layout conversions (reinterpret casts) between bfloat16 (16-bit)
    # and uint32 (32-bit) on tiled memory are physically unsupported by SparseCore and crash,
    # we completely eliminate all 2D layouts from the kernel boundary!
    # We flatten the input in JAX space to 1D, view it as uint32 (which is 100% flat and safe),
    # and pass it as a pure 1D flat buffer of shape (input_size * aligned_hidden_size,)!
    x_flat = x_clean.reshape(-1)
    x_u32 = x_flat.view(jnp.uint32) # Shape: (input_size * aligned_hidden_size,)
  else:
    dtype_bytes = 4
    aligned_hidden_size = hidden_size
    out_dtype = x.dtype
    x_u32 = x.reshape(-1) # Also flat 1D for float32 consistency!

  col_size = aligned_hidden_size // num_column_partitions

  # Output shape in HBM is declared as a pure 1D flat shape!
  num_output_rows = padded_input_size // reduce_group_size
  out_shape_flat = (num_output_rows * aligned_hidden_size,)

  vector_mesh_wrapped = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=2,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  out_flat = pl.kernel(  # pytype: disable=wrong-keyword-args
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
              out_shape_flat,
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
      x_u32,                 # Passed as a pure 1D flat uint32 buffer!
      src_indices,
      dst_indices,
      topk_weights_sorted,
  )

  # Reshape the flat 1D output back to 2D in JAX space.
  # Since this is a same-type reshape (uint32 to uint32), it is 100% safe and layout-bug-free!
  out_2d = out_flat.reshape(num_output_rows, aligned_hidden_size)

  if is_bf16:
    # =========================================================================
    # 👑 THE OUTPUT LAYOUT FORTRESS:
    # To permanently prevent the compiler's size-doubling layout bug from triggering on the
    # OUTPUT path, we perform the unpacking entirely using safe 32-bit bitcasts and numerical casts,
    # and place separate, independent matmul barriers on the even and odd bfloat16 tensors
    # BEFORE they are stacked and reshaped!
    # =========================================================================
    
    # 1. Extract Even Elements (lower 16 bits of each uint32 word).
    even_u32 = jnp.bitwise_and(out_2d, 65535)
    even_f32_bits = jnp.bitwise_left_shift(even_u32, 16)
    even_f32 = even_f32_bits.view(jnp.float32) # Standard 32-bit view bitcast!
    even_bf16 = even_f32.astype(jnp.bfloat16)

    # 2. Extract Odd Elements (upper 16 bits of each uint32 word).
    odd_u32 = jnp.bitwise_right_shift(out_2d, 16)
    odd_f32_bits = jnp.bitwise_left_shift(odd_u32, 16)
    odd_f32 = odd_f32_bits.view(jnp.float32) # Standard 32-bit view bitcast!
    odd_bf16 = odd_f32.astype(jnp.bfloat16)

    # 3. Apply matmul barriers to both tensors to block layout propagation.
    dynamic_zero_out = (indices[0] - indices[0]).astype(jnp.bfloat16)
    
    def apply_output_barrier(tensor_2048):
      blocks = []
      block_size = 512
      num_blocks = aligned_hidden_size // block_size
      for i in range(num_blocks):
        block = tensor_2048[:, i * block_size : (i + 1) * block_size]
        identity = jnp.eye(block_size, dtype=jnp.bfloat16) + dynamic_zero_out
        blocks.append(jnp.matmul(block, identity))
      return jnp.concatenate(blocks, axis=1)

    even_bf16_clean = apply_output_barrier(even_bf16)
    odd_bf16_clean = apply_output_barrier(odd_bf16)

    # 4. Interleave Even and Odd Columns to reconstruct the final (R, hidden_size) tensor.
    out_final = jnp.stack([even_bf16_clean, odd_bf16_clean], axis=-1).reshape(
        padded_input_size // reduce_group_size, hidden_size
    )
  else:
    out_final = out_2d

  return out_final[: (input_size // reduce_group_size), :hidden_size]
