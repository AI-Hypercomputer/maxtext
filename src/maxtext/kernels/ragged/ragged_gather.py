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

"""Ragged gather kernel implementation from tpu-inference."""
# Source from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/sparse_core/ragged_gather.py

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from packaging.version import Version


# JAX <= 0.10.0 used `out_shape`/`scratch_shapes` kwargs for `pl.kernel`; later
# versions renamed them to `out_type`/`scratch_types`.
if Version(jax.__version__) <= Version("0.10.0"):
  _OUT_KW = "out_shape"
  _SCRATCH_KW = "scratch_shapes"
else:
  _OUT_KW = "out_type"
  _SCRATCH_KW = "scratch_types"


def main_kernel(
    # Inputs.
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    weights_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    indices_vmem_ref: jax.Ref,
    weights_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    has_weights: bool,
):
  """Core ragged gather operation with per-row weighting."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_lanes = tpu_info.num_lanes
  hidden_size = in_hbm_ref.shape[-1]
  col_size = out_vmem_ref.shape[-1]
  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
  block_size = num_simd_lanes * num_cores

  recv_sem = sem_ref.at[0]
  send_sem = sem_ref.at[1]

  # Read start and end tensor values.
  dma_list = []
  dma = pltpu.make_async_copy(start_ref.at[:1], start_vmem_ref.at[:1], recv_sem)
  dma_list.append(dma)
  dma = pltpu.make_async_copy(end_ref.at[:1], end_vmem_ref.at[:1], recv_sem)
  dma_list.append(dma)

  jax.tree.map(lambda x: x.start(), dma_list)
  jax.tree.map(lambda x: x.wait(), dma_list)

  # Calculate number of tiles to visit using start and end arrays.
  start = start_vmem_ref[...][0]
  end = end_vmem_ref[...][0]

  block_start = start // block_size
  block_end = pl.cdiv(end, block_size)
  num_blocks = block_end - block_start
  num_blocks = jnp.where(end == start, 0, num_blocks)
  aligned_start = block_start * block_size

  num_cols = pl.cdiv(hidden_size, col_size)

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_blocks, num_cores, num_cols),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL, pltpu.ARBITRARY),
  )
  def inner_kernel():
    block_id = pl.program_id(0)
    core_id = pl.program_id(1)
    col_id = pl.program_id(2)

    row_tile_start = aligned_start + block_id * block_size + core_id * num_simd_lanes
    col_tile_start = col_id * col_size

    @pl.when(col_id == 0)
    def _():
      pltpu.sync_copy(
          indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
          indices_vmem_ref,
      )
      if has_weights:
        pltpu.sync_copy(
            weights_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
            weights_vmem_ref,
        )

    # HBM to VMEM transfer.
    indices = indices_vmem_ref[...]
    if has_weights:
      weights = weights_vmem_ref[...]

    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits

    # To fetch only one sublane at a time, we need to use (packing, 128) layout.
    # But, the inputs are in (8, 128) layout and thus we need to perform
    # relayout. For 32-bits, this can be done with a simple reinterpretation,
    # but for other bitwidths, this is not possible. Therefore, we bitcast data
    # into 32-bits first to fetch packing number of rows per dma and later
    # perform bitwise unpacking / packing to obtain desired results.
    in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)
    out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

    for col_vmem_start in range(0, col_size, num_lanes):
      col_hbm_start = col_tile_start + col_vmem_start
      for row_vmem in range(num_simd_lanes):
        row_hbm = indices[row_vmem] // packing
        pltpu.make_async_copy(
            in_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
            out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
            recv_sem,
        ).start()

    # VMEM to HBM transfer.
    # Use dynamic loop to minimize register spills.
    @pl.loop(0, col_size, step=num_lanes)
    @jax.named_scope("dma_write_loop")
    def dma_write_loop(col_vmem_start):
      col_hbm_start = col_tile_start + col_vmem_start

      # Wait for data to be received.
      # NOTE: Because a single semaphore was used for all dma calls, we need
      # to make sure the order of the wait is the same as order of start.
      # Otherwise, a dma finish can trigger wrong dma wait to exit.
      for _ in range(num_simd_lanes):
        pltpu.make_async_copy(
            in_32b_hbm_ref.at[0, :num_lanes],
            out_vmem_ref.at[0, :num_lanes],
            recv_sem,
        ).wait()

      # If multiple elements are packed in single 32-bits, extract the desired
      # elements and reorder them.
      if packing > 1:
        for col_compute_offset in range(0, num_lanes, num_simd_lanes):
          col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)

          out = None
          for row_src in range(num_simd_lanes):
            row_src_pack = indices[row_src] % packing
            row_dst_pack = row_src % packing

            rightshift_bits = row_src_pack * dtype_bits
            leftshift_bits = row_dst_pack * dtype_bits

            # Load data from vmem.
            data = out_vmem_ref[row_src, col_slice]

            # Right shift to make first n bits stores target data.
            data = jnp.bitwise_right_shift(data, rightshift_bits)
            # Mask out unwanted bits.
            data = jnp.bitwise_and(data, 2**dtype_bits - 1)
            # Left shift data into the target bit location.
            data = jnp.bitwise_left_shift(data, leftshift_bits)

            if row_dst_pack == 0:
              out = data
            else:
              assert out is not None
              out = jnp.bitwise_or(out, data)

            if row_dst_pack == packing - 1:
              # Store packed data into correct position.
              row_dst = row_src // packing
              out_vmem_ref[row_dst, col_slice] = out

      # Apply per-row weights after unpacking if needed.
      # For packing == 1 (float32), we can apply directly.
      # For packing > 1 (bf16), the data is already packed; we apply below.
      if has_weights:
        if packing == 1:
          # float32 path: data is already in float32 layout, one row per sublane.
          for col_compute_offset in range(0, num_lanes, num_simd_lanes):
            col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)
            for row_vmem in range(num_simd_lanes):
              data = out_vmem_ref[row_vmem, col_slice]
              data_f32 = jax.lax.bitcast_convert_type(data, jnp.float32)
              data_f32 = data_f32 * weights[row_vmem]
              out_vmem_ref[row_vmem, col_slice] = jax.lax.bitcast_convert_type(data_f32, jnp.uint32)
        else:
          # bf16 path: data is packed, packing=2. Each packed row contains 2
          # bf16 values from consecutive source rows. We need to unpack each
          # bf16 to float32, multiply by its weight, then repack.
          for col_compute_offset in range(0, num_lanes, num_simd_lanes):
            col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)
            for row_dst in range(num_simd_lanes // packing):
              packed_data = out_vmem_ref[row_dst, col_slice]
              result = jnp.zeros_like(packed_data)
              for sub in range(packing):
                row_src = row_dst * packing + sub
                # Extract the sub-element.
                shift_right = sub * dtype_bits
                shift_left = sub * dtype_bits
                elem = jnp.bitwise_right_shift(packed_data, shift_right)
                elem = jnp.bitwise_and(elem, 2**dtype_bits - 1)
                # Convert bf16 bits to float32 for weighting.
                # bf16 is stored in the lower 16 bits; shift to upper 16 for
                # bitcast to float32.
                elem_f32 = jnp.bitwise_left_shift(elem, 16)
                elem_f32 = jax.lax.bitcast_convert_type(elem_f32, jnp.float32)
                elem_f32 = elem_f32 * weights[row_src]
                # Convert back: bitcast float32 -> uint32, shift right 16 to
                # get bf16 bits, then shift left to target position.
                elem_u32 = jax.lax.bitcast_convert_type(elem_f32, jnp.uint32)
                elem_bf16 = jnp.bitwise_right_shift(elem_u32, 16)
                elem_bf16 = jnp.bitwise_and(elem_bf16, 2**dtype_bits - 1)
                elem_bf16 = jnp.bitwise_left_shift(elem_bf16, shift_left)
                result = jnp.bitwise_or(result, elem_bf16)
              out_vmem_ref[row_dst, col_slice] = result

      # Start dma write.
      for row_vmem in range(num_simd_lanes // packing):
        row_hbm = row_tile_start // packing + row_vmem
        pltpu.make_async_copy(
            out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
            out_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
            send_sem,
        ).start()

    # Wait for dma write to finish.
    for _ in range(0, col_size, num_lanes):
      for _ in range(num_simd_lanes // packing):
        pltpu.make_async_copy(
            out_vmem_ref.at[0, :num_lanes],
            out_32b_hbm_ref.at[0, :num_lanes],
            send_sem,
        ).wait()

  inner_kernel()


def calculate_col_size(hidden_size: int) -> int:
  """Calculate col size for ragged gather kernel."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_lanes = tpu_info.num_lanes
  num_simd_lanes = sc_info.num_lanes

  match tpu_info.chip_version:
    case 6:
      target_bytes = (256 * 1024) * 0.9
    case 7:
      target_bytes = (512 * 1024) * 0.9
    case _:
      target_bytes = (128 * 1024) * 0.9

  base_bytes = num_simd_lanes * hidden_size * (32 // 8)
  num_cols = 1

  while pl.cdiv(base_bytes, num_cols * num_lanes) * num_lanes > target_bytes:
    num_cols += 1
  return pl.cdiv(hidden_size, (num_cols * num_lanes)) * num_lanes


@functools.partial(jax.jit, static_argnames=("has_weights",))
def ragged_gather(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    weights: jax.Array | None = None,
    has_weights: bool = False,
) -> jax.Array:
  """Perform gather on indices within dynamic array start and end.

  Args:
    x: 2D input array of shape ``(input_size, hidden_size)``.
    indices: 1D array of gather indices.
    start: Scalar or 1D array indicating the start of the valid range.
    end: Scalar or 1D array indicating the end of the valid range.
    weights: Optional 1D array of per-row weights. When provided, each
      gathered row is multiplied by its corresponding weight inside the
      kernel, avoiding an extra HBM read-write pass.
    has_weights: Static bool flag indicating whether ``weights`` should be
      applied. Must be ``True`` when ``weights`` is not ``None``.

  Returns:
    Gathered output of shape ``(indices_size, hidden_size)``.
  """

  assert x.ndim == 2, "Ragged gather only supports 2d inputs."
  assert indices.ndim == 1, "Ragged gather only supports 1d indices."

  # TODO(b/522341181): this is a workaround to avoid unexpected type mismatch
  if jnp.ndim(start) == 0:
    start = jnp.repeat(start, 2)
  elif start.shape == (1,):
    start = jnp.pad(start, (0, 1))

  if jnp.ndim(end) == 0:
    end = jnp.repeat(end, 2)
  elif end.shape == (1,):
    end = jnp.pad(end, (0, 1))

  dtype = x.dtype

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    # Sparse core is not available. Fallback to regular gather.
    out = x[indices]
    if has_weights:
      out = out * weights[:, None]
    return out

  hidden_size = x.shape[-1]
  out_size = indices.size

  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores
  block_size = num_simd_lanes * num_cores
  col_size = calculate_col_size(hidden_size)

  # Pad to align to the block size.
  out_pad_size = pl.cdiv(out_size, block_size) * block_size - out_size
  indices = jnp.pad(indices, ((0, out_pad_size)))

  if has_weights:
    weights = jnp.pad(weights, ((0, out_pad_size)), constant_values=1.0)
  else:
    # Provide a dummy weights array; the kernel won't use it.
    weights = jnp.ones((out_size + out_pad_size,), dtype=jnp.float32)

  aligned_hidden_size = pl.cdiv(hidden_size, col_size) * col_size

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  return pl.kernel(  # pytype: disable=wrong-keyword-args
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          has_weights=has_weights,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather",
      **{
          _OUT_KW: jax.ShapeDtypeStruct((out_size + out_pad_size, aligned_hidden_size), dtype),
          _SCRATCH_KW: [
              pltpu.VMEM((num_simd_lanes,), jnp.int32),
              pltpu.VMEM((num_simd_lanes,), jnp.int32),
              pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
              pltpu.VMEM((num_simd_lanes,), jnp.int32),
              pltpu.VMEM((num_simd_lanes,), jnp.float32),
              pltpu.SemaphoreType.DMA((2,)),
          ],
      },
  )(start, end, x, indices, weights)[:out_size, :hidden_size]
