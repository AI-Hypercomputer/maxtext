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

"""Ragged scatter kernel implementation from tpu-inference."""
# Source from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/sparse_core/ragged_scatter.py

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
    total_num_rows_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    src_indices_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    total_num_rows_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    src_indices_vmem_ref: jax.Ref,
    dst_indices_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
):
  """Core ragged scatter operations"""
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

  # Read total number of valid rows tensor values.
  dma = pltpu.make_async_copy(total_num_rows_ref, total_num_rows_vmem_ref.at[:1], recv_sem)
  dma.start()
  dma.wait()
  total_num_rows = total_num_rows_vmem_ref[...][0]

  # Calculate number of tiles to visit.
  num_blocks = jnp.where(total_num_rows == 0, 0, pl.cdiv(total_num_rows, block_size))
  num_cols = pl.cdiv(hidden_size, col_size)

  def inner_kernel(block_id, core_id, col_id):
    row_tile_start = block_id * block_size + core_id * num_simd_lanes
    col_tile_start = col_id * col_size

    @pl.when(col_id == 0)
    def _():
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
      jax.tree.map(lambda x: x.start(), dma_list)
      jax.tree.map(lambda x: x.wait(), dma_list)

    # HBM to VMEM transfer.
    src_indices = src_indices_vmem_ref[...]
    dst_indices = dst_indices_vmem_ref[...]

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
        row_hbm = src_indices[row_vmem] // packing
        # Since we have changed layout from (8, 128) to (1, 128), continuous
        # memory address does not yield desired values anymore. Therefore,
        # we break up a dmas into multiple num_lanes sized requests.
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

          previous_data = None
          for row_src in range(num_simd_lanes):
            row_src_pack = src_indices[row_src] % packing
            row_dst_pack = dst_indices[row_src] % packing

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
            row_hbm = dst_indices[row_src] // packing

            if row_src == 0:
              prev_row_hbm = -1
              previous_data = data  # This won't be actually used.
            else:
              prev_row_hbm = dst_indices[row_src - 1] // packing
              assert previous_data is not None

            # We guarantee source rows with the same destination sublane
            # are adjacent to each other in the order of being processed,
            # and must be in the same row tile. If the current row goes to
            # the same destination sublane as the previous row, we merge the
            # data with the previous data.
            data_to_write = jnp.where(
                row_hbm == prev_row_hbm,
                jnp.bitwise_or(previous_data, data),
                data,
            )
            out_vmem_ref[row_src, col_slice] = data_to_write
            previous_data = data_to_write

      # Start dma write.
      # There must be at least one valid row to write if we are here.
      last_valid_row_vmem = -1
      last_valid_row_hbm = -1
      for row_vmem in range(num_simd_lanes):
        row_valid = (row_tile_start + row_vmem) < total_num_rows
        row_hbm = dst_indices[row_vmem] // packing
        if row_vmem < num_simd_lanes - 1:
          next_row_hbm = dst_indices[row_vmem + 1] // packing
          next_row_valid = (row_tile_start + row_vmem + 1) < total_num_rows
        else:
          next_row_hbm = -1
          next_row_valid = False

        # If the current row and the next row are going to the same
        # destination sublane, the merged data for all source rows going
        # to that sublane will be stored in the last vmem row for that sublane,
        # i.e `row_vmem // packing * packing + packing - 1`.
        # Logically, we could skip all the rows that are not the last row for
        # each destination sublane, but we want to avoid using `pl.when` for
        # efficiency.
        #
        # If the current row is out of bounds, we just repeat the last valid
        # write to avoid valid data in hbm being overwritten.
        merged_data_vmem_row = (row_vmem // packing) * packing + packing - 1
        src_row_vmem = jnp.where(
            jnp.logical_and(row_hbm == next_row_hbm, next_row_valid),
            merged_data_vmem_row,
            jnp.where(row_valid, row_vmem, last_valid_row_vmem),
        )
        dst_row_hbm = jnp.where(
            jnp.logical_and(row_hbm == next_row_hbm, next_row_valid),
            row_hbm,
            jnp.where(row_valid, row_hbm, last_valid_row_hbm),
        )
        pltpu.make_async_copy(
            out_vmem_ref.at[src_row_vmem, pl.ds(col_vmem_start, num_lanes)],
            out_32b_hbm_ref.at[dst_row_hbm, pl.ds(col_hbm_start, num_lanes)],
            send_sem,
        ).start()
        last_valid_row_vmem = jnp.where(row_valid, src_row_vmem, last_valid_row_vmem)
        last_valid_row_hbm = jnp.where(row_valid, dst_row_hbm, last_valid_row_hbm)

    # Wait for dma write to finish.
    for _ in range(0, col_size, num_lanes):
      for _ in range(num_simd_lanes):
        pltpu.make_async_copy(
            out_vmem_ref.at[0, :num_lanes],
            out_32b_hbm_ref.at[0, :num_lanes],
            send_sem,
        ).wait()

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_blocks, num_cores, num_cols),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL, pltpu.ARBITRARY),
  )
  def kernel_wrapper():
    block_id = pl.program_id(0)
    core_id = pl.program_id(1)
    col_id = pl.program_id(2)
    row_tile_start = block_id * block_size + core_id * num_simd_lanes

    # Only execute the kernel instance when there is a least one valid row
    # to process.
    @pl.when(row_tile_start < total_num_rows)
    def _():
      inner_kernel(block_id, core_id, col_id)

  kernel_wrapper()


def calculate_col_size(hidden_size: int) -> int:
  """Calculate col size for ragged gather kernel."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_lanes = tpu_info.num_lanes
  num_simd_lanes = sc_info.num_lanes

  match tpu_info.generation:
    case 6:
      target_bytes = (256 * 1024) * 0.8
    case 7:
      target_bytes = (512 * 1024) * 0.8
    case _:
      target_bytes = (128 * 1024) * 0.8

  base_bytes = num_simd_lanes * hidden_size * (32 // 8)
  num_cols = 1

  while pl.cdiv(base_bytes, num_cols * num_lanes) * num_lanes > target_bytes:
    num_cols += 1
  return pl.cdiv(hidden_size, (num_cols * num_lanes)) * num_lanes


def _preprocess_indices(
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    out_pad_size: int,
    packing: int,
    row_tile_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Preprocesses indices for ragged scatter kernel."""
  assert indices.ndim == 1, "Ragged scatter only supports 1d indices."
  assert row_tile_size % packing == 0

  # When packing > 1, there could be multiple source rows destined for the same
  # destination physical 32bits-words sublane (which is the basic unit of DMA).
  # In order to avoid conflict of those writes, we'll assign all writes for
  # the same destination sublane to one SparseCore core.
  # Further, if all writes to a destination sublane are assigned to the same
  # row tile, one kernel invocation can handle the merged write to that sublane.
  #
  # We could achieve this by the following steps:
  # 1. For each destination sublane, count total number of valid writes to it.
  #    If it's greater than 1 and not equal to packing, we'll mark all writes to
  #    that destination sublane as valid so that the total number of valid
  #    writes for each destination sublane are either 0, 1 or packing.
  # 2. Sort the source rows based on the number of valid writes to its
  #    associated destination sublane (from step 1) in descending order.
  #    This will put all source rows with the same destination sublane adjacent
  #    to each other. As long as the row tile size is multiple of packing,
  #    we can guarantee that all writes to the same destination sublane are
  #    assigned to same core, same row tile.

  src_indices = jnp.where(jnp.logical_and(indices >= start, indices < end), indices, -1)
  src_indices = jnp.pad(src_indices, ((0, out_pad_size)), constant_values=-1)
  src_indices = src_indices.reshape(-1, packing)
  is_valid_src_row = src_indices != -1
  num_sublanes = src_indices.shape[0]
  num_valid_src_rows_per_dst_sublane = jnp.sum(is_valid_src_row, axis=-1, keepdims=False)
  num_valid_src_rows_per_dst_sublane = jnp.broadcast_to(
      num_valid_src_rows_per_dst_sublane[:, None], (num_sublanes, packing)
  )
  # For each destination sublane that has more than one valid writes, we
  # consider all writes to that sublane as valid so that total number of
  # writes for those sublanes are always equal to packing.
  cnts = jnp.where(
      num_valid_src_rows_per_dst_sublane > 1,
      packing,
      jnp.where(is_valid_src_row, 1, 0),
  ).reshape(-1)
  sorted_by_cnts = jnp.argsort(cnts, descending=True, stable=True)
  src_indices = (jnp.pad(indices, ((0, out_pad_size)), constant_values=0))[sorted_by_cnts]
  dst_indices = sorted_by_cnts

  # Due to possible considering more source rows as valid, the total number of
  # source rows may be larger than end - start.
  total_num_valid_source_rows = jnp.sum(cnts > 0)[None]
  return src_indices, dst_indices, total_num_valid_source_rows


@jax.jit
def ragged_scatter(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array) -> jax.Array:
  """Gathers rows from `x` according to `indices` within a specified range.

  This function performs a gather operation equivalent to `x[indices]` for
  indices that fall within the range `[start, end)`. For indices outside this
  range, the behavior is undefined.

  Args:
    x: A 2D JAX array to gather data from, with shape `(num_rows, hidden_size)`.
    indices: A 1D JAX array of indices to gather, with shape `(output_size,)`.
    start: A scalar or 1D array of size 1 containing the start index (inclusive)
      to process.
    end: A scalar or 1D array of size 1 containing the end index (exclusive) to
      process.

  Returns:
    A 2D JAX array of gathered data with shape `(output_size, hidden_size)`.

  The typical usage of this kernel is "unpermute" after GMM of the MOE layers.
  That is, replace `gmm2_res[topk_argsort_revert_indices]` with
  `ragged_scatter(x, topk_argsort_revert_indices, ..)` in
  tpu_inference/layers/common/fused_moe_gmm.py.
  """

  assert x.ndim == 2, "Ragged scatter only supports 2d inputs."
  assert indices.ndim == 1, "Ragged scatter only supports 1d indices."

  if jnp.isscalar(start):
    start = start[None]
  if jnp.isscalar(end):
    end = end[None]

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    # Sparse core is not available. Fallback to regular gather.
    return x[indices]

  dtype = x.dtype
  dtype_bits = jax.dtypes.itemsize_bits(dtype)
  packing = 32 // dtype_bits
  dtype_bytes = dtype_bits // 8

  # Heuristic threshold on whether to fallback to xla gather.
  if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
    # For small {input + output}, it's likely that both can be put in TC VMEM,
    # so it's likely faster to run TC-based gather on it than going through SC,
    # without data movement to/from HBM.
    return x[indices]

  hidden_size = x.shape[-1]
  out_size = indices.size

  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores
  block_size = num_simd_lanes * num_cores
  col_size = calculate_col_size(hidden_size)

  # Pad to align to the block size.
  out_pad_size = pl.cdiv(out_size, block_size) * block_size - out_size
  aligned_hidden_size = pl.cdiv(hidden_size, col_size) * col_size

  src_indices, dst_indices, total_num_rows = _preprocess_indices(
      indices, start, end, out_pad_size, packing, row_tile_size=num_simd_lanes
  )

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  return pl.kernel(
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_ragged_scatter",
      **{
          _OUT_KW: jax.ShapeDtypeStruct((out_size + out_pad_size, aligned_hidden_size), dtype),
          _SCRATCH_KW: [
              pltpu.VMEM((num_simd_lanes,), jnp.int32),  # total_num_rows
              pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
              pltpu.VMEM((num_simd_lanes,), jnp.int32),  # src_indices
              pltpu.VMEM((num_simd_lanes,), jnp.int32),  # dst_indices
              pltpu.SemaphoreType.DMA((2,)),
          ],
      },
  )(total_num_rows, x, src_indices, dst_indices)[:out_size, :hidden_size]
