# Copyright 2023–2026 Google LLC
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

"""SparseCore gather-reduce kernel implementation.

This module contains a kernel implementation for performing a gather-reduce
operation on TPU SparseCore. It groups rows of an operand based on provided
indices, sums them up, and scatters the results.
"""

import array
import functools
from typing import Any

import jax
from jax import core
from jax.experimental import mosaic
from jax.experimental.mosaic.dialects import tpu
import jax.experimental.pallas.tpu as pltpu
from jax.interpreters import mlir
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector


class VectorTypeHelper:
  """Helper to create VectorType with a specific element type."""

  def __init__(self, element_type_fn):
    self.element_type_fn = element_type_fn

  def __getitem__(self, shape):
    if isinstance(shape, int):
      shape = [shape]
    return ir.VectorType.get(shape, self.element_type_fn())


_I32 = VectorTypeHelper(functools.partial(ir.IntegerType.get_signless, 32))
_F32 = VectorTypeHelper(ir.F32Type.get)
_BF16 = VectorTypeHelper(ir.BF16Type.get)


@jax.jit(
    static_argnames=[
        "reduce_group_size",
        "single_sc",
        "col_chunk_size",
        "loop_unroll_factor_1",
        "loop_unroll_factor_2",
        "loop_unroll_factor_3",
        "loop_parallel_access_1",
        "loop_parallel_access_2",
        "loop_parallel_access_3",
        "topk_wgt_zero_nan",
    ],
)
def sc_gather_reduce(
    op: jax.Array,
    idx: jax.Array,
    topk_weights: jax.Array | None = None,
    *,
    reduce_group_size: int,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 16,  # writing back 2 rows given reduce size of 8
    loop_unroll_factor_1: int = 2,
    loop_unroll_factor_2: int = 2,
    loop_unroll_factor_3: int = 8,
    loop_parallel_access_1: bool = True,
    loop_parallel_access_2: bool = False,
    loop_parallel_access_3: bool = False,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
  """Performs a gather-reduce operation on SparseCore.

  This kernel groups rows of the operand `op` based on `idx`, sums them up,
  and scatters the results. The gather and add operations are performed in fp32,
  and the results are written back in bf16.

  Equivalent jax numpy code:
  ```
    gathered = op[idx, :]
    if topk_wgt_local is not None:
      flat_weights = topk_wgt_local.flatten()
      gathered = gathered * flat_weights[:, None].astype(acc_dtype)
    gathered = jnp.reshape(gathered, (-1, reduce_group_size, op.shape[1]))
    output = jnp.sum(gathered.astype(acc_dtype), axis=1).astype(jnp.bfloat16)
    ```

  Args:
    op: The operand matrix in fp32 [B, K] to reduce.
    idx: The indices in int32[M,] guiding the reduction and scatter.
    topk_weights: Optional weights to apply to the gathered operands.
    reduce_group_size: The size of the groups to reduce.
    single_sc: Whether to use a single SparseCore.
    col_chunk_size: The size of column chunks to process.
    row_chunk_size: The size of row chunks for internal processing.
    loop_unroll_factor_1: Unroll factor for the main loop over column chunks.
    loop_unroll_factor_2: Unroll factor for the loop over row chunks in offset
      calculation.
    loop_unroll_factor_3: Unroll factor for the inner loop within offset
      calculation.
    loop_parallel_access_1: Enables parallel access for the main column chunk
      loop.
    loop_parallel_access_2: Enables parallel access for the row chunk loop in
      offset calculation.
    loop_parallel_access_3: Enables parallel access for the inner loop within
      offset calculation.
    topk_wgt_zero_nan: If true, treat zero topk_weights as indicators of NaN
      during multiplication, resulting in zero output.

  Returns:
    The result of operation, bf16 matrix [M/reduce_group_size, K].
  """

  assert op.dtype in (
      jnp.float32,
      jnp.bfloat16,
  ), f"op.dtype must be f32 or bf16, but got {op.dtype}"
  is_bf16 = op.dtype == jnp.bfloat16
  assert op.shape[0] % reduce_group_size == 0, (
      "op.shape[0] must be divisible by reduce_group_size, but got" f" {op.shape[0]} and {reduce_group_size}"
  )
  assert row_chunk_size / reduce_group_size == 2, (  # writing back in bf16 (2 rows at once)
      f"row_chunk_size must be 2 * reduce_group_size, but got {row_chunk_size=}" f" and {reduce_group_size=}"
  )

  if topk_weights is not None:
    assert topk_weights.ndim == 2
    assert topk_weights.shape[0] * 128 == idx.shape[0]
    assert topk_weights.shape[1] == 128

  tpu_info = pltpu.get_tpu_info()
  if tpu_info.sparse_core is None:
    raise ValueError("SparseCore is not available on this TPU version.")

  used_sc_cores = 1 if single_sc else 2
  num_sc_per_core = tpu_info.sparse_core.num_subcores
  num_sc = num_sc_per_core * used_sc_cores

  vreg_size = tpu_info.sparse_core.num_lanes

  with mlir.make_ir_context() as ctx, ir.Location.unknown():
    # The TPU dialect is required for its TPU_DimensionSemantics
    tpu.register_dialect(ctx)

    bf16 = ir.BF16Type.get()
    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()
    index = ir.IndexType.get()

    memoryspace_tilespmem = ir.Attribute.parse("#tpu.memory_space<vmem>")
    memoryspace_hbm = ir.Attribute.parse("#tpu.memory_space<hbm>")

    memory_space_semaphore = ir.Attribute.parse("#tpu.memory_space<semaphore_mem>")
    dma_semaphore_type = ir.Type.parse("!tpu.dma_semaphore")

    # A mask of all True values to enable all sublanes in vector operations.
    enable_all_sublanes_mask = ir.DenseBoolArrayAttr.get([True] * vreg_size)

    assert row_chunk_size == vreg_size

    def _kernel_impl(
        current_sc_core,
        current_local_core,
        idx_ref,
        op_ref,
        weights_ref,
        out_ref,
        func_op,
    ):
      constants: dict[tuple[Any, Any], Any] = {}

      def const_lut(val, ty=None):
        ty = index if ty is None else ty
        if (val, ty) not in constants:
          with ir.InsertionPoint.at_block_begin(func_op.entry_block):
            constants[(val, ty)] = arith.constant(ty, ir.IntegerAttr.get(ty, val))
        return constants[(val, ty)]

      def fill_load_offset_tile(offset_tile_local, idx_tile_local, col_pos):
        """Fills the offset tile for indirect DMA gather.

        This function calculates the HBM offsets from which to gather rows
        based on the indices in idx_tile_local, for a given column chunk.
        The offsets are calculated to correctly index into the operand `op`
        in HBM, considering the memory layout and the current column chunk
        being processed. The calculated offsets are stored in
        offset_tile_local, which is later used by tpu.enqueue_indirect_dma.

        Args:
          offset_tile_local: The destination memref in TileSpMem to store
            calculated offsets.
          idx_tile_local: A memref in TileSpMem containing a chunk of indices of
            rows to gather from `op`.
          col_pos: The index of the current column chunk being processed.

        Returns:
          The offset_tile_local memref filled with offsets for DMA gather.
        """
        idx_loaded = tpu.load(
            _I32[row_chunk_size],
            idx_tile_local,
            [const_lut(0)],
            enable_all_sublanes_mask,
        )
        parity = arith.remui(
            idx_loaded,
            vector.broadcast(_I32[row_chunk_size], const_lut(2, i32)),
        )
        if is_bf16:
          mul_mem_layout = 4
        else:
          mul_mem_layout = 8

        iota = arith.constant(
            _I32[row_chunk_size],
            ir.DenseIntElementsAttr.get(
                array.array("i", [i * mul_mem_layout for i in range(row_chunk_size)]),
                type=_I32[row_chunk_size],
            ),
        )
        loop_over_idx = scf.ForOp(
            lower_bound=const_lut(0),
            upper_bound=const_lut(row_chunk_size + 1),
            step=const_lut(1),
        )
        loop_over_idx.attributes["sc.loop_unroll_factor"] = ir.IntegerAttr.get(i32, loop_unroll_factor_2)
        if loop_parallel_access_2:
          loop_over_idx.attributes["sc.parallel_access"] = ir.UnitAttr.get()
        with ir.InsertionPoint(loop_over_idx.body):
          i = loop_over_idx.induction_variable
          base_idx = vector.extract(
              idx_loaded,
              dynamic_position=[i],
              static_position=ir.DenseI64ArrayAttr.get([-9223372036854775808]),
          )

          # align base_idx with memory layout
          base_idx_div = arith.divui(base_idx, const_lut(8, i32))
          if is_bf16:
            base_idx_rem = arith.divui(arith.remui(base_idx, const_lut(8, i32)), const_lut(2, i32))
          else:
            base_idx_rem = arith.remui(base_idx, const_lut(8, i32))
          base_idx = arith.addi(
              arith.muli(  # NOTYPO
                  base_idx_div,
                  const_lut(op.shape[1] // 128 * mul_mem_layout, i32),
              ),
              base_idx_rem,
          )

          # consider col chunk
          base_idx = arith.addi(
              base_idx,
              arith.muli(  # NOTYPO
                  arith.index_cast(i32, col_pos),
                  const_lut(col_chunk_size // 128 * mul_mem_layout, i32),
              ),
          )

          start_vec = arith.addi(vector.broadcast(_I32[row_chunk_size], base_idx), iota)
          loop_j = scf.ForOp(
              const_lut(0),
              const_lut((col_chunk_size // 128) // vreg_size),
              const_lut(1),
              iter_args=[start_vec],
          )
          loop_j.attributes["sc.loop_unroll_factor"] = ir.IntegerAttr.get(i32, loop_unroll_factor_3)
          if loop_parallel_access_3:
            loop_j.attributes["sc.parallel_access"] = ir.UnitAttr.get()
          with ir.InsertionPoint(loop_j.body):
            vec = loop_j.inner_iter_args[0]

            idx_local = arith.addi(
                arith.muli(i, const_lut((col_chunk_size // 128))),  # NOTYPO
                arith.muli(loop_j.induction_variable, const_lut(vreg_size)),  # NOTYPO
            )
            tpu.store(
                vec,
                offset_tile_local,
                [idx_local],
                enable_all_sublanes_mask,
            )
            next_vec = arith.addi(
                vec,
                vector.broadcast(
                    _I32[row_chunk_size],
                    const_lut(mul_mem_layout * vreg_size, i32),
                ),
            )
            scf.YieldOp([next_vec])

          rem_vreg = col_chunk_size / 128 % vreg_size / vreg_size

          last_full_vec = tpu.load(
              _I32[vreg_size],
              offset_tile_local,
              [
                  arith.subi(
                      arith.muli(const_lut((col_chunk_size // 128)), i),  # NOTYPO
                      const_lut(int(vreg_size * (1 + rem_vreg))),
                  )
              ],
              enable_all_sublanes_mask,
          )
          add_vec = vector.broadcast(
              _I32[vreg_size],
              const_lut(int(mul_mem_layout * rem_vreg * vreg_size), i32),
          )  # 4 comes from bf16

          last_full_vec = arith.addi(last_full_vec, add_vec)
          tpu.store(
              last_full_vec,
              offset_tile_local,
              [
                  arith.subi(
                      arith.muli(const_lut((col_chunk_size // 128)), i),  # NOTYPO
                      const_lut(int(vreg_size)),
                  )
              ],
              enable_all_sublanes_mask,
          )

          scf.YieldOp([])
        return offset_tile_local, parity

      def load_weights(lin_idx, dst_tile, sflag):
        """Loads weights from HBM to TileSpMem.

        This function calculates offsets for reading weights from HBM based on
        a linear index `lin_idx`. It handles a specific packed memory layout
        of weights where weights for two rows are packed together (bf16). It
        calculates parity to determine which part of the packed data to use
        later in `perform_add`. It uses indirect DMA to load weights into
        dst_tile.

        Args:
          lin_idx: Linear index used to calculate weight offsets in HBM.
          dst_tile: The destination memref in TileSpMem to load weights into.
          sflag: The semaphore to use for the DMA operation.

        Returns:
          Parity bit (0 or 1) indicating which set of weights to use from
          the loaded packed data.
        """
        lin_idx = arith.divui(lin_idx, const_lut(16))  # reading in increments of 16

        # parity
        # (i >> 3) & 1
        parity = arith.andi(arith.shrui(lin_idx, const_lut(3)), const_lut(1))

        # address
        # (i & 7) | ((i >> 4) << 3)
        lin_idx = arith.ori(
            arith.andi(lin_idx, const_lut(7)),
            arith.shli(arith.shrui(lin_idx, const_lut(4)), const_lut(3)),
        )

        # lin_idx is index type, cast to i32
        lin_idx_i32 = arith.index_cast(i32, lin_idx)
        lin_idx_base = arith.muli(lin_idx_i32, const_lut(16, i32))  # NOTYPO
        lin_idx_base_vec = vector.broadcast(_I32[row_chunk_size], lin_idx_base)

        iota = arith.constant(
            _I32[row_chunk_size],
            ir.DenseIntElementsAttr.get(
                array.array("i", list(range(16))),
                type=_I32[row_chunk_size],
            ),
        )
        offsets = arith.addi(lin_idx_base_vec, iota)

        tpu.enqueue_indirect_dma(
            source=weights_ref,
            target=dst_tile,
            offsets=offsets,
            semaphore=sflag,
        )
        tpu.wait_indirect_dma(semaphore=sflag, src=weights_ref, dst=dst_tile)

        return parity

      def perform_add(
          scratch_local,
          scratch_out_local,
          idx_parity,
          weights_local=None,
          parity=None,
      ):
        """Performs reduction (summation) of rows in scratchpad.

        This function reduces `row_chunk_size` rows stored in scratch_local
        into 2 rows by summing rows in groups of `reduce_group_size`. If
        weights_local is provided, rows are multiplied by weights before
        summation. The result is packed into bf16 format and stored in
        scratch_out_local. With row_chunk_size=16 and reduce_group_size=8,
        this reduces 16 rows to 2 rows (rows 0-7 summed to one row, rows 8-15
        summed to another).

        Args:
          scratch_local: Memref in TileSpMem containing rows gathered from `op`.
          scratch_out_local: Memref in TileSpMem to store the 2 reduced rows in
            bf16 format.
          idx_parity: Parity information for each index in the gathered rows.
          weights_local: Optional memref in TileSpMem containing weights to
            apply before reduction.
          parity: Optional parity bit indicating which weights to use from
            packed weight data in weights_local.

        Returns:
          The scratch_out_local memref containing reduced rows.
        """
        weights_vecs = None
        if weights_local is not None:
          f32_zero = arith.constant(f32, ir.FloatAttr.get(f32, 0.0))
          zero_vec_f32 = vector.broadcast(_F32[vreg_size], f32_zero)
          # Load 32 weights (64B) -> 16 weights for R0 + 16 weights for R1 from
          # packed layout. We check parity to decide which one to take. We need
          # to extract the even or odd elements
          weights_local_2x16 = memref.reinterpret_cast(
              ir.MemRefType.get(
                  (2, row_chunk_size),
                  bf16,
                  ir.Attribute.parse("#tpu.tiled<,[" + str(row_chunk_size) + ",1]>"),
                  memory_space=memoryspace_tilespmem,
              ),
              weights_local,
              [],
              [],
              [],
              static_offsets=[0],
              static_sizes=[2, row_chunk_size],
              static_strides=[row_chunk_size, 1],
          )
          raw_weights = tpu.load(
              _BF16[2, row_chunk_size],
              weights_local_2x16,
              [const_lut(0), const_lut(0)],
              enable_all_sublanes_mask,
          )

          # Reminder Mosaic SC compiler: flips [d0, d1] to [d1, d0]
          # part=1 corresponds to sc_tpu.unpackf result 1 (Even indices)
          weights_evens = tpu.unpack_subelements(
              _F32[16],
              raw_weights,
              0,
              ir.Attribute.parse("#tpu.pack_format<interleaved>"),
          )
          # part=0 corresponds to sc_tpu.unpackf result 0 (Odd indices)
          weights_odds = tpu.unpack_subelements(
              _F32[16],
              raw_weights,
              1,
              ir.Attribute.parse("#tpu.pack_format<interleaved>"),
          )

          is_odd = arith.cmpi(
              arith.CmpIPredicate.eq,
              parity,
              const_lut(0),
          )
          raw_weights_f32 = arith.select(is_odd, weights_odds, weights_evens)

          weights_vecs = [
              vector.broadcast(
                  _F32[vreg_size],
                  vector.extract(
                      raw_weights_f32,
                      dynamic_position=[],
                      static_position=ir.DenseI64ArrayAttr.get([i]),
                  ),
              )
              for i in range(16)
          ]

        # reinterpret cast to correct shape to read from it
        if not is_bf16:
          new_scratch_shape = (row_chunk_size, col_chunk_size)
          new_scratch_layout = ir.Attribute.parse("#tpu.tiled<,[" + str(col_chunk_size) + ",1]>")
          new_scratch_ref_ty = ir.MemRefType.get(
              new_scratch_shape,
              f32,
              new_scratch_layout,
              memory_space=memoryspace_tilespmem,
          )
          scratch_local = memref.reinterpret_cast(
              new_scratch_ref_ty,
              scratch_local,
              [],
              [],
              [],
              static_offsets=[0],
              static_sizes=new_scratch_shape,
              static_strides=[col_chunk_size, 1],
          )

        loop_pack_scratch = scf.ForOp(const_lut(0), const_lut(col_chunk_size), const_lut(vreg_size))
        loop_pack_scratch.attributes["sc.loop_unroll_factor"] = ir.IntegerAttr.get(i32, loop_unroll_factor_1)
        if loop_parallel_access_1:
          loop_pack_scratch.attributes["sc.parallel_access"] = ir.UnitAttr.get()
        with ir.InsertionPoint(loop_pack_scratch.body):
          col_offset = loop_pack_scratch.induction_variable

          # map col_offset to idx and col inside 128
          if is_bf16:
            row_add = arith.divui(col_offset, const_lut(128))
            col_pos = arith.remui(col_offset, const_lut(128))
          else:
            row_add = None
            col_pos = None

          def get_row_val(row_idx):
            if is_bf16:
              row_idx_l = row_idx * (col_chunk_size // 128)
              vec_bf16_2x16 = tpu.load(
                  _BF16[2, 16],
                  scratch_local,
                  [
                      arith.addi(const_lut(row_idx_l), row_add),
                      arith.muli(col_pos, const_lut(2)),  # NOTYPO
                  ],
                  enable_all_sublanes_mask,
              )
              vec_f32_evens = tpu.unpack_subelements(
                  _F32[16],
                  vec_bf16_2x16,
                  0,
                  ir.Attribute.parse("#tpu.pack_format<interleaved>"),
              )
              vec_f32_odds = tpu.unpack_subelements(
                  _F32[16],
                  vec_bf16_2x16,
                  1,
                  ir.Attribute.parse("#tpu.pack_format<interleaved>"),
              )
              parity_of_row = vector.extract(
                  idx_parity,
                  dynamic_position=[],
                  static_position=ir.DenseI64ArrayAttr.get([row_idx]),
              )
              is_odd_scalar = arith.cmpi(arith.CmpIPredicate.eq, parity_of_row, const_lut(0, i32))
              return arith.select(is_odd_scalar, vec_f32_odds, vec_f32_evens)
            else:
              return tpu.load(
                  _F32[vreg_size],
                  scratch_local,
                  [const_lut(row_idx), col_offset],
                  enable_all_sublanes_mask,
              )

          row0 = get_row_val(0)
          if weights_local is not None:
            row0 = arith.mulf(row0, weights_vecs[0])
            if topk_wgt_zero_nan:
              row0 = arith.select(
                  arith.cmpf(arith.CmpFPredicate.OEQ, weights_vecs[0], zero_vec_f32),
                  zero_vec_f32,
                  row0,
              )

          row8 = get_row_val(8)
          if weights_local is not None:
            row8 = arith.mulf(row8, weights_vecs[8])
            if topk_wgt_zero_nan:
              row8 = arith.select(
                  arith.cmpf(arith.CmpFPredicate.OEQ, weights_vecs[8], zero_vec_f32),
                  zero_vec_f32,
                  row8,
              )

          for sum_idx in range(7):
            tmp_row0 = get_row_val(sum_idx + 1)
            if weights_local is not None:
              tmp_row0 = arith.mulf(tmp_row0, weights_vecs[sum_idx + 1])
              if topk_wgt_zero_nan:
                tmp_row0 = arith.select(
                    arith.cmpf(
                        arith.CmpFPredicate.OEQ,
                        weights_vecs[sum_idx + 1],
                        zero_vec_f32,
                    ),
                    zero_vec_f32,
                    tmp_row0,
                )

            row0 = arith.addf(row0, tmp_row0)

            tmp_row8 = get_row_val(8 + sum_idx + 1)
            if weights_local is not None:
              tmp_row8 = arith.mulf(tmp_row8, weights_vecs[8 + sum_idx + 1])
              if topk_wgt_zero_nan:
                tmp_row8 = arith.select(
                    arith.cmpf(
                        arith.CmpFPredicate.OEQ,
                        weights_vecs[8 + sum_idx + 1],
                        zero_vec_f32,
                    ),
                    zero_vec_f32,
                    tmp_row8,
                )

            row8 = arith.addf(row8, tmp_row8)

          packed = tpu.pack_subelements(
              _BF16[2, vreg_size],
              [row8, row0],
              [0, 1],
              ir.Attribute.parse("#tpu.pack_format<interleaved>"),
          )

          tpu.store(
              packed,
              scratch_out_local,
              [const_lut(0), arith.muli(col_offset, const_lut(2))],  # NOTYPO
              enable_all_sublanes_mask,
          )
          scf.YieldOp([])

        # undo reshape cast on scratch_local
        if not is_bf16:
          mem_num = 128
          new_scratch_shape = (
              row_chunk_size * col_chunk_size // mem_num,
              mem_num,
          )
          new_scratch_layout = ir.Attribute.parse("#tpu.tiled<,[" + str(mem_num) + ",1]>")
          new_scratch_ref_ty = ir.MemRefType.get(
              new_scratch_shape,
              f32,
              new_scratch_layout,
              memory_space=memoryspace_tilespmem,
          )

          scratch_local = memref.reinterpret_cast(
              new_scratch_ref_ty,
              scratch_local,
              [],
              [],
              [],
              static_offsets=[0],
              static_sizes=new_scratch_shape,
              static_strides=[mem_num, 1],
          )

        return scratch_out_local

      def fill_out_offset_tile(offset_tile_out_local, col_pos, row_pos=None):
        """Fills the offset tile for indirect DMA scatter for outputs (bf16).

        This function calculates the HBM offsets to scatter the reduced rows
        to. The offsets are calculated to correctly index into the output
        tensor in HBM, based on the row index of the reduction group
        `row_pos` and the current column chunk `col_pos`. The calculated
        offsets are stored in offset_tile_out_local, which is later used by
        tpu.enqueue_indirect_dma to scatter results from TileSpMem to HBM.

        Args:
          offset_tile_out_local: The destination memref in TileSpMem to store
            calculated offsets.
          col_pos: The index of the current column chunk being processed.
          row_pos: The starting row index for the reduction group in the output
            tensor. If None, computes offsets for prologue.

        Returns:
          The offset_tile_out_local memref filled with offsets for DMA scatter.
        """
        if row_pos is not None:
          rest_row_pos = arith.remui(row_pos, const_lut(8))
          rest_row_pos = arith.divui(rest_row_pos, const_lut(2))
          tmp = arith.divui(row_pos, const_lut(8))
          tmp = arith.muli(tmp, const_lut(4 * op.shape[1] // 128))  # NOTYPO
          row_vec_offset = arith.addi(tmp, rest_row_pos)

          row_vec_offset = arith.index_cast(i32, row_vec_offset)

          row_vec_offset = vector.broadcast(_I32[row_chunk_size], row_vec_offset)

        iota = arith.constant(
            _I32[row_chunk_size],
            ir.DenseIntElementsAttr.get(
                array.array("i", [i * 4 for i in range(row_chunk_size)]),  # 4 is key here for bf16
                type=_I32[row_chunk_size],
            ),
        )

        iota = arith.addi(
            iota,
            vector.broadcast(
                _I32[row_chunk_size],
                arith.muli(  # NOTYPO
                    arith.index_cast(i32, col_pos),
                    const_lut(col_chunk_size // 128 * 4, i32),
                ),
            ),
        )

        loop_over_idx = scf.ForOp(
            lower_bound=const_lut(0),
            upper_bound=const_lut(row_chunk_size // reduce_group_size),
            step=const_lut(2),
        )
        loop_over_idx.attributes["sc.loop_unroll_factor"] = ir.IntegerAttr.get(i32, loop_unroll_factor_2)
        if loop_parallel_access_2:
          loop_over_idx.attributes["sc.parallel_access"] = ir.UnitAttr.get()
        with ir.InsertionPoint(loop_over_idx.body):
          loop_i = loop_over_idx.induction_variable
          start_vec = arith.addi(
              vector.broadcast(_I32[row_chunk_size], arith.index_cast(i32, loop_i)),
              iota,
          )
          loop_j = scf.ForOp(
              const_lut(0),
              const_lut((col_chunk_size // 128) // vreg_size),
              const_lut(1),
              iter_args=[start_vec],
          )
          loop_j.attributes["sc.loop_unroll_factor"] = ir.IntegerAttr.get(i32, loop_unroll_factor_3)
          if loop_parallel_access_3:
            loop_j.attributes["sc.parallel_access"] = ir.UnitAttr.get()
          with ir.InsertionPoint(loop_j.body):
            vec = loop_j.inner_iter_args[0]

            idx_local = arith.addi(
                arith.muli(loop_i, const_lut((op.shape[1] // 128))),  # NOTYPO
                arith.muli(loop_j.induction_variable, const_lut(vreg_size)),  # NOTYPO
            )

            if row_pos is not None:
              local_vec = arith.addi(vec, row_vec_offset)
            else:
              local_vec = vec

            tpu.store(
                local_vec,
                offset_tile_out_local,
                [idx_local],
                enable_all_sublanes_mask,
            )
            add_vec = vector.broadcast(_I32[vreg_size], const_lut(4 * vreg_size, i32))
            next_vec = arith.addi(vec, add_vec)
            scf.YieldOp([next_vec])
          scf.YieldOp([])

        rem_vreg = col_chunk_size / 128 % vreg_size / vreg_size

        last_full_vec = tpu.load(
            _I32[vreg_size],
            offset_tile_out_local,
            [const_lut(offset_sizes_out - int(vreg_size * (1 + rem_vreg)))],
            enable_all_sublanes_mask,
        )
        add_vec = vector.broadcast(_I32[vreg_size], const_lut(int(4 * rem_vreg * vreg_size), i32))  # 4 comes from bf16

        last_full_vec = arith.addi(last_full_vec, add_vec)
        tpu.store(
            last_full_vec,
            offset_tile_out_local,
            [const_lut(offset_sizes_out - vreg_size)],
            enable_all_sublanes_mask,
        )
        return offset_tile_out_local

      offset_sizes_out = (
          (row_chunk_size // reduce_group_size) * (col_chunk_size // 128) // 2
      )  # two because doing two rows at once
      offset_tile_out_0 = memref.alloca(
          ir.MemRefType.get(
              shape=(offset_sizes_out,),
              element_type=i32,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      offset_tile_out_1 = memref.alloca(
          ir.MemRefType.get(
              shape=(offset_sizes_out,),
              element_type=i32,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      if is_bf16:
        scratch_0 = memref.alloca(
            ir.MemRefType.get(
                shape=(row_chunk_size * 2, col_chunk_size),
                element_type=bf16,
                memory_space=memoryspace_tilespmem,
            ),
            [],
            [],
        )
        scratch_1 = memref.alloca(
            ir.MemRefType.get(
                shape=(row_chunk_size * 2, col_chunk_size),
                element_type=bf16,
                memory_space=memoryspace_tilespmem,
            ),
            [],
            [],
        )
        mem_num = 128 * 2
        new_scratch_shape = (
            row_chunk_size * col_chunk_size * 2 // mem_num,
            mem_num,
        )
        new_scratch_layout = ir.Attribute.parse("#tpu.tiled<,[" + str(mem_num) + ",1]>")
        new_scratch_ref_ty = ir.MemRefType.get(
            new_scratch_shape,
            bf16,
            new_scratch_layout,
            memory_space=memoryspace_tilespmem,
        )
      else:
        scratch_0 = memref.alloca(
            ir.MemRefType.get(
                shape=(row_chunk_size, col_chunk_size),
                element_type=f32,
                memory_space=memoryspace_tilespmem,
            ),
            [],
            [],
        )
        scratch_1 = memref.alloca(
            ir.MemRefType.get(
                shape=(row_chunk_size, col_chunk_size),
                element_type=f32,
                memory_space=memoryspace_tilespmem,
            ),
            [],
            [],
        )
        mem_num = 128
        new_scratch_shape = (
            row_chunk_size * col_chunk_size // mem_num,
            mem_num,
        )
        new_scratch_layout = ir.Attribute.parse("#tpu.tiled<,[" + str(mem_num) + ",1]>")
        new_scratch_ref_ty = ir.MemRefType.get(
            new_scratch_shape,
            f32,
            new_scratch_layout,
            memory_space=memoryspace_tilespmem,
        )

      scratch_0 = memref.reinterpret_cast(
          new_scratch_ref_ty,
          scratch_0,
          [],
          [],
          [],
          static_offsets=[0],
          static_sizes=new_scratch_shape,
          static_strides=[mem_num, 1],
      )

      scratch_1 = memref.reinterpret_cast(
          new_scratch_ref_ty,
          scratch_1,
          [],
          [],
          [],
          static_offsets=[0],
          static_sizes=new_scratch_shape,
          static_strides=[mem_num, 1],
      )

      scratch_out_0 = memref.alloca(
          ir.MemRefType.get(
              shape=(2, col_chunk_size),
              element_type=bf16,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      scratch_out_1 = memref.alloca(
          ir.MemRefType.get(
              shape=(2, col_chunk_size),
              element_type=bf16,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      mem_num = 128 * 2
      new_scratch_shape = (2 * col_chunk_size // mem_num, mem_num)
      new_scratch_layout = ir.Attribute.parse("#tpu.tiled<,[" + str(mem_num) + ",1]>")
      new_scratch_ref_ty = ir.MemRefType.get(
          new_scratch_shape,
          bf16,
          new_scratch_layout,
          memory_space=memoryspace_tilespmem,
      )

      scratch_out_0 = memref.reinterpret_cast(
          new_scratch_ref_ty,
          scratch_out_0,
          [],
          [],
          [],
          static_offsets=[0],
          static_sizes=new_scratch_shape,
          static_strides=[mem_num, 1],
      )

      scratch_out_1 = memref.reinterpret_cast(
          new_scratch_ref_ty,
          scratch_out_1,
          [],
          [],
          [],
          static_offsets=[0],
          static_sizes=new_scratch_shape,
          static_strides=[mem_num, 1],
      )

      new_input_shape = (
          idx.shape[0] // reduce_group_size * op.shape[1] // mem_num,
          mem_num,
      )
      new_input_ref_ty = ir.MemRefType.get(
          new_input_shape,
          bf16,
          layout=ir.Attribute.parse(f"#tpu.tiled<,[{mem_num}, 1]>"),
          memory_space=memoryspace_hbm,
      )
      out_ref = tpu.reinterpret_cast(
          new_input_ref_ty,
          out_ref,
      )

      if topk_weights is not None:
        new_weights_shape = (topk_weights.size // 2, 2)
        new_weights_ref_ty = ir.MemRefType.get(
            new_weights_shape, bf16, layout=ir.Attribute.parse("#tpu.tiled<,[2, 1]>"), memory_space=memoryspace_hbm
        )
        weights_ref = tpu.reinterpret_cast(new_weights_ref_ty, weights_ref)

      sflag_0 = tpu.sem_alloc(ir.MemRefType.get((), dma_semaphore_type, memory_space=memory_space_semaphore))
      sflag_1 = tpu.sem_alloc(ir.MemRefType.get((), dma_semaphore_type, memory_space=memory_space_semaphore))
      sflag_out_0 = tpu.sem_alloc(ir.MemRefType.get((), dma_semaphore_type, memory_space=memory_space_semaphore))
      sflag_out_1 = tpu.sem_alloc(ir.MemRefType.get((), dma_semaphore_type, memory_space=memory_space_semaphore))

      idx_tile_0 = memref.alloca(
          ir.MemRefType.get(
              shape=(row_chunk_size,),
              element_type=i32,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      idx_tile_1 = memref.alloca(
          ir.MemRefType.get(
              shape=(row_chunk_size,),
              element_type=i32,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      weights_tile_0 = None
      weights_tile_1 = None
      sflag_weights_0 = None
      sflag_weights_1 = None
      if topk_weights is not None:
        weights_tile_0 = memref.alloca(
            ir.MemRefType.get(
                shape=(row_chunk_size, 2),  # matches Offsets shape (16)
                element_type=bf16,
                memory_space=memoryspace_tilespmem,
            ),
            [],
            [],
        )
        weights_tile_1 = memref.alloca(
            ir.MemRefType.get(
                shape=(row_chunk_size, 2),  # matches Offsets shape (16)
                element_type=bf16,
                memory_space=memoryspace_tilespmem,
            ),
            [],
            [],
        )
        tiled_weights_ref_ty = ir.MemRefType.get(
            (row_chunk_size, 2),
            bf16,
            layout=ir.Attribute.parse("#tpu.tiled<,[2, 1]>"),
            memory_space=memoryspace_tilespmem,
        )
        weights_tile_0 = memref.reinterpret_cast(
            tiled_weights_ref_ty,
            weights_tile_0,
            [],
            [],
            [],
            static_offsets=[0],
            static_sizes=[row_chunk_size, 2],
            static_strides=[2, 1],
        )
        weights_tile_1 = memref.reinterpret_cast(
            tiled_weights_ref_ty,
            weights_tile_1,
            [],
            [],
            [],
            static_offsets=[0],
            static_sizes=[row_chunk_size, 2],
            static_strides=[2, 1],
        )

        sflag_weights_0 = tpu.sem_alloc(ir.MemRefType.get((), dma_semaphore_type, memory_space=memory_space_semaphore))
        sflag_weights_1 = tpu.sem_alloc(ir.MemRefType.get((), dma_semaphore_type, memory_space=memory_space_semaphore))

      offset_sizes = row_chunk_size * (col_chunk_size // 128)
      offset_tile_0 = memref.alloca(
          ir.MemRefType.get(
              shape=(offset_sizes,),
              element_type=i32,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      offset_tile_1 = memref.alloca(
          ir.MemRefType.get(
              shape=(offset_sizes,),
              element_type=i32,
              memory_space=memoryspace_tilespmem,
          ),
          [],
          [],
      )

      global_chunk_to_process = arith.addi(
          arith.muli(current_sc_core, const_lut(num_sc_per_core, i32)),  # NOTYPO
          current_local_core,
      )
      global_row_idx_start = arith.index_cast(
          index,
          arith.muli(  # NOTYPO
              global_chunk_to_process,
              const_lut(idx.shape[0] // num_sc, i32),
          ),
      )

      loop_col_chunk = scf.ForOp(
          const_lut(0),
          const_lut(op.shape[1] // col_chunk_size),
          const_lut(1),
      )
      with ir.InsertionPoint(loop_col_chunk.body):
        col_chunk_ij = loop_col_chunk.induction_variable
        offset_tile_out_0 = fill_out_offset_tile(offset_tile_out_0, col_chunk_ij)

        # #############
        # # prologue
        # #############

        # setup stream for #0
        loop_row_chunk_idx = const_lut(0)

        base_idx_val = arith.index_cast(i32, arith.addi(global_row_idx_start, loop_row_chunk_idx))
        base_idx_val = tpu.assume_multiple(base_idx_val, 8)
        source_slice = tpu.memref_slice(
            ir.MemRefType.get((row_chunk_size,), i32, memory_space=memoryspace_hbm),
            idx_ref,
            base_idx=[base_idx_val],
            dynamic_sizes=[],
        )

        tpu.enqueue_dma(source_slice, idx_tile_0, sflag_0)
        tpu.wait_dma2(semaphore=sflag_0, src=source_slice, dst=idx_tile_0)

        if topk_weights is not None:
          # The linear index is global_row_idx_start + loop_row_chunk_idx
          lin_idx = arith.addi(global_row_idx_start, loop_row_chunk_idx)
          parity_0 = load_weights(lin_idx, weights_tile_0, sflag_weights_0)
          parity_1 = const_lut(0)
        else:
          parity_0 = const_lut(0)
          parity_1 = const_lut(0)

        offset_tile_0, idx_parity_0 = fill_load_offset_tile(offset_tile_0, idx_tile_0, col_chunk_ij)

        if is_bf16:
          mem_num = 128 * 2
          new_input_shape = (op.shape[0] * op.shape[1] // mem_num, mem_num)
          new_input_layout = ir.Attribute.parse("#tpu.tiled<,[" + str(mem_num) + ",1]>")
          new_input_ref_ty = ir.MemRefType.get(
              new_input_shape,
              bf16,
              new_input_layout,
              memory_space=memoryspace_hbm,
          )
        else:
          mem_num = 128
          new_input_shape = (op.shape[0] * op.shape[1] // mem_num, mem_num)
          new_input_layout = ir.Attribute.parse("#tpu.tiled<,[" + str(mem_num) + ",1]>")
          new_input_ref_ty = ir.MemRefType.get(
              new_input_shape,
              f32,
              new_input_layout,
              memory_space=memoryspace_hbm,
          )
        op_ref = memref.reinterpret_cast(
            new_input_ref_ty,
            op_ref,
            [],
            [],
            [],
            static_offsets=[0],
            static_sizes=new_input_shape,
            static_strides=[mem_num, 1],
        )

        tpu.enqueue_indirect_dma(
            source=op_ref,
            target=scratch_0,
            offsets=offset_tile_0,
            semaphore=sflag_0,
        )

        #############
        # corpus
        #############
        loop_over_row_chunks = scf.ForOp(
            lower_bound=const_lut(row_chunk_size * 1),
            upper_bound=const_lut(row_chunk_size * ((idx.shape[0] // num_sc // row_chunk_size) - 1)),
            step=const_lut(row_chunk_size * 2),
            iter_args=[
                scratch_0,
                scratch_out_0,
                sflag_0,
                sflag_out_0,
                idx_tile_0,
                offset_tile_0,
                offset_tile_out_0,
                scratch_1,
                scratch_out_1,
                sflag_1,
                sflag_out_1,
                idx_tile_1,
                offset_tile_1,
                offset_tile_out_1,
                weights_tile_0 if topk_weights is not None else scratch_0,
                weights_tile_1 if topk_weights is not None else scratch_0,
                sflag_weights_0 if topk_weights is not None else sflag_0,
                sflag_weights_1 if topk_weights is not None else sflag_0,
                parity_0 if topk_weights is not None else const_lut(0),
                parity_1 if topk_weights is not None else const_lut(0),
                idx_parity_0,
                idx_parity_0,  # idx_parity_1 is not initialized yet
            ],
        )
        with ir.InsertionPoint(loop_over_row_chunks.body):
          loop_row_chunk_idx = loop_over_row_chunks.induction_variable
          (
              scratch_0,
              scratch_out_0,
              sflag_0,
              sflag_out_0,
              idx_tile_0,
              offset_tile_0,
              offset_tile_out_0,
              scratch_1,
              scratch_out_1,
              sflag_1,
              sflag_out_1,
              idx_tile_1,
              offset_tile_1,
              offset_tile_out_1,
              weights_tile_0,
              weights_tile_1,
              sflag_weights_0,
              sflag_weights_1,
              parity_0,
              parity_1,
              idx_parity_0,
              idx_parity_1,
          ) = loop_over_row_chunks.inner_iter_args

          # setup stream for #1
          base_idx_val = arith.index_cast(
              i32,
              arith.addi(
                  global_row_idx_start,
                  arith.addi(loop_row_chunk_idx, const_lut(0)),
              ),
          )
          base_idx_val = tpu.assume_multiple(base_idx_val, 8)
          source_slice = tpu.memref_slice(
              ir.MemRefType.get((row_chunk_size,), i32, memory_space=memoryspace_hbm),
              idx_ref,
              base_idx=[base_idx_val],
              dynamic_sizes=[],
          )

          tpu.enqueue_dma(source_slice, idx_tile_1, sflag_1)
          tpu.wait_dma2(semaphore=sflag_1, src=source_slice, dst=idx_tile_1)

          if topk_weights is not None:
            # The linear index is global_row_idx_start + loop_row_chunk_idx
            lin_idx = arith.addi(global_row_idx_start, loop_row_chunk_idx)
            parity_1 = load_weights(lin_idx, weights_tile_1, sflag_weights_1)

          offset_tile_1, idx_parity_1 = fill_load_offset_tile(offset_tile_1, idx_tile_1, col_chunk_ij)

          tpu.enqueue_indirect_dma(
              source=op_ref,
              target=scratch_1,
              offsets=offset_tile_1,
              semaphore=sflag_1,
          )

          # wait stream #0
          tpu.wait_indirect_dma(semaphore=sflag_0, src=op_ref, dst=scratch_0)

          # process #0
          scratch_out_0 = perform_add(
              scratch_0,
              scratch_out_0,
              idx_parity_0,
              weights_local=weights_tile_0 if topk_weights is not None else None,
              parity=parity_0 if topk_weights is not None else None,
          )

          offset_tile_out_0 = fill_out_offset_tile(
              offset_tile_out_0,
              col_chunk_ij,
              arith.divui(
                  arith.addi(
                      global_row_idx_start,
                      arith.subi(loop_row_chunk_idx, const_lut(row_chunk_size)),
                  ),
                  const_lut(reduce_group_size),
              ),
          )
          tpu.enqueue_indirect_dma(
              source=scratch_out_0,
              target=out_ref,
              offsets=offset_tile_out_0,
              semaphore=sflag_out_0,
          )

          if topk_weights is not None:
            # The linear index is:
            # global_row_idx_start + loop_row_chunk_idx + row_chunk_size
            lin_idx = arith.addi(
                global_row_idx_start,
                arith.addi(loop_row_chunk_idx, const_lut(row_chunk_size)),
            )
            parity_0 = load_weights(lin_idx, weights_tile_0, sflag_weights_0)

          base_idx_val = arith.index_cast(
              i32,
              arith.addi(
                  global_row_idx_start,
                  arith.addi(loop_row_chunk_idx, const_lut(row_chunk_size)),
              ),
          )
          base_idx_val = tpu.assume_multiple(base_idx_val, 8)
          source_slice = tpu.memref_slice(
              ir.MemRefType.get((row_chunk_size,), i32, memory_space=memoryspace_hbm),
              idx_ref,
              base_idx=[base_idx_val],
              dynamic_sizes=[],
          )
          tpu.enqueue_dma(source_slice, idx_tile_0, sflag_0)
          tpu.wait_dma2(semaphore=sflag_0, src=source_slice, dst=idx_tile_0)

          offset_tile_0, idx_parity_0 = fill_load_offset_tile(offset_tile_0, idx_tile_0, col_chunk_ij)

          tpu.enqueue_indirect_dma(
              source=op_ref,
              target=scratch_0,
              offsets=offset_tile_0,
              semaphore=sflag_0,
          )

          # wait stream #1
          tpu.wait_indirect_dma(semaphore=sflag_1, src=op_ref, dst=scratch_1)

          # # process #1
          scratch_out_1 = perform_add(
              scratch_1,
              scratch_out_1,
              idx_parity_1,
              weights_local=weights_tile_1 if topk_weights is not None else None,
              parity=parity_1 if topk_weights is not None else None,
          )

          offset_tile_out_1 = fill_out_offset_tile(
              offset_tile_out_1,
              col_chunk_ij,
              arith.divui(
                  arith.addi(global_row_idx_start, loop_row_chunk_idx),
                  const_lut(reduce_group_size),
              ),
          )
          tpu.enqueue_indirect_dma(
              source=scratch_out_1,
              target=out_ref,
              offsets=offset_tile_out_1,
              semaphore=sflag_out_1,
          )

          # wait stream #0 out
          tpu.wait_indirect_dma(semaphore=sflag_out_0, src=scratch_out_0, dst=out_ref)
          # wait stream #1 out
          tpu.wait_indirect_dma(semaphore=sflag_out_1, src=scratch_out_1, dst=out_ref)

          # return #0 and #1
          scf.YieldOp(
              [
                  scratch_0,
                  scratch_out_0,
                  sflag_0,
                  sflag_out_0,
                  idx_tile_0,
                  offset_tile_0,
                  offset_tile_out_0,
                  scratch_1,
                  scratch_out_1,
                  sflag_1,
                  sflag_out_1,
                  idx_tile_1,
                  offset_tile_1,
                  offset_tile_out_1,
                  weights_tile_0 if topk_weights is not None else scratch_0,
                  weights_tile_1 if topk_weights is not None else scratch_0,
                  sflag_weights_0 if topk_weights is not None else sflag_0,
                  sflag_weights_1 if topk_weights is not None else sflag_0,
                  parity_0 if topk_weights is not None else const_lut(0),
                  parity_1 if topk_weights is not None else const_lut(0),
                  idx_parity_0,
                  idx_parity_1,
              ]
          )

        #############
        # epilogue
        #############
        (
            scratch_0,
            scratch_out_0,
            sflag_0,
            sflag_out_0,
            _,
            _,
            offset_tile_out_0,
            scratch_1,
            scratch_out_1,
            sflag_1,
            sflag_out_1,
            idx_tile_1,
            offset_tile_1,
            offset_tile_out_1,
            weights_tile_0,
            weights_tile_1,
            _,
            sflag_weights_1,
            parity_0,
            parity_1,
            idx_parity_0,
            idx_parity_1,
        ) = loop_over_row_chunks.results_

        epi_idx_loc = row_chunk_size * ((idx.shape[0] // num_sc // row_chunk_size) - 1)
        add_f = arith.divui(global_row_idx_start, const_lut(reduce_group_size))

        # setup stream for #1
        base_idx_val = arith.index_cast(i32, arith.addi(global_row_idx_start, const_lut(epi_idx_loc)))
        base_idx_val = tpu.assume_multiple(base_idx_val, 8)
        source_slice = tpu.memref_slice(
            ir.MemRefType.get((row_chunk_size,), i32, memory_space=memoryspace_hbm),
            idx_ref,
            base_idx=[base_idx_val],
            dynamic_sizes=[],
        )
        tpu.enqueue_dma(source_slice, idx_tile_1, sflag_1)
        tpu.wait_dma2(semaphore=sflag_1, src=source_slice, dst=idx_tile_1)

        if topk_weights is not None:
          lin_idx = arith.addi(global_row_idx_start, const_lut(epi_idx_loc))
          parity_1 = load_weights(lin_idx, weights_tile_1, sflag_weights_1)

        offset_tile_1, idx_parity_1 = fill_load_offset_tile(offset_tile_1, idx_tile_1, col_chunk_ij)

        tpu.enqueue_indirect_dma(
            source=op_ref,
            target=scratch_1,
            offsets=offset_tile_1,
            semaphore=sflag_1,
        )

        # wait stream #0
        tpu.wait_indirect_dma(semaphore=sflag_0, src=op_ref, dst=scratch_0)

        # process #0
        scratch_out_0 = perform_add(
            scratch_0,
            scratch_out_0,
            idx_parity_0,
            weights_local=weights_tile_0 if topk_weights is not None else None,
            parity=parity_0 if topk_weights is not None else None,
        )

        offset_tile_out_0 = fill_out_offset_tile(
            offset_tile_out_0,
            col_chunk_ij,
            arith.addi(
                add_f,
                const_lut((idx.shape[0] // num_sc // reduce_group_size) - 4),
            ),
        )
        tpu.enqueue_indirect_dma(
            source=scratch_out_0,
            target=out_ref,
            offsets=offset_tile_out_0,
            semaphore=sflag_out_0,
        )

        # wait stream #1
        tpu.wait_indirect_dma(semaphore=sflag_1, src=op_ref, dst=scratch_1)

        # process #1
        scratch_out_1 = perform_add(
            scratch_1,
            scratch_out_1,
            idx_parity_1,
            weights_local=weights_tile_1 if topk_weights is not None else None,
            parity=parity_1 if topk_weights is not None else None,
        )

        offset_tile_out_1 = fill_out_offset_tile(
            offset_tile_out_1,
            col_chunk_ij,
            arith.addi(
                add_f,
                const_lut((idx.shape[0] // num_sc // reduce_group_size) - 2),
            ),
        )
        tpu.enqueue_indirect_dma(
            source=scratch_out_1,
            target=out_ref,
            offsets=offset_tile_out_1,
            semaphore=sflag_out_1,
        )
        tpu.wait_indirect_dma(semaphore=sflag_out_0, src=scratch_out_0, dst=out_ref)
        tpu.wait_indirect_dma(semaphore=sflag_out_1, src=scratch_out_1, dst=out_ref)

        scf.YieldOp([])

    input_types = [
        i32,
        i32,
        ir.MemRefType.get(
            idx.shape,
            i32,
            memory_space=memoryspace_hbm,
        ),
        ir.MemRefType.get(
            op.shape,
            bf16 if is_bf16 else f32,
            memory_space=memoryspace_hbm,
        ),
        ir.MemRefType.get(
            (idx.shape[0] // reduce_group_size, op.shape[1]),
            bf16,
            memory_space=memoryspace_hbm,
        ),
    ]
    if topk_weights is not None:
      input_types.insert(
          4,
          ir.MemRefType.get(
              topk_weights.shape,
              bf16,
              memory_space=memoryspace_hbm,
          ),
      )

    # Configure the wrappers
    if topk_weights is not None:

      @func.FuncOp.from_py_func(*input_types, name="main")
      def kernel_main(
          current_sc_core,
          current_local_core,
          idx_ref,
          op_ref,
          weights_ref,
          out_ref,
          func_op,
      ):
        return _kernel_impl(
            current_sc_core,
            current_local_core,
            idx_ref,
            op_ref,
            weights_ref,
            out_ref,
            func_op,
        )

    else:

      @func.FuncOp.from_py_func(*input_types, name="main")
      def kernel_main(
          current_sc_core,
          current_local_core,
          idx_ref,
          op_ref,
          out_ref,
          func_op,
      ):
        return _kernel_impl(
            current_sc_core,
            current_local_core,
            idx_ref,
            op_ref,
            None,
            out_ref,
            func_op,
        )

    # Configure the Mosaic iteration space
    f = kernel_main.func_op
    f.attributes["iteration_bounds"] = ir.DenseI64ArrayAttr.get([used_sc_cores, num_sc_per_core])
    f.attributes["dimension_semantics"] = ir.ArrayAttr.get(
        [
            ir.Attribute.parse("#tpu.dimension_semantics<core_parallel>"),
            ir.Attribute.parse("#tpu.dimension_semantics<parallel>"),
        ]
    )
    args_attributes = [
        ir.DictAttr.get({}),  # i
        ir.DictAttr.get({}),  # j
        ir.DictAttr.get({"sc.persistent": ir.UnitAttr.get()}),  # idx
        ir.DictAttr.get({"sc.persistent": ir.UnitAttr.get()}),  # op
        ir.DictAttr.get({"sc.persistent": ir.UnitAttr.get()}),  # out
    ]
    window_params = [
        ir.DictAttr.get(
            {
                "transform_indices": ir.Attribute.parse("affine_map<(n,m) -> (0)>"),
            }
        ),
        ir.DictAttr.get(
            {
                "transform_indices": ir.Attribute.parse("affine_map<(n,m) -> (0,0)>"),
            }
        ),
        ir.DictAttr.get(
            {
                "transform_indices": ir.Attribute.parse("affine_map<(n,m) -> (0,0)>"),
            }
        ),
    ]

    if topk_weights is not None:
      # Insert weights before output - we append here because it's the same
      # attribute as output.
      args_attributes.append(ir.DictAttr.get({"sc.persistent": ir.UnitAttr.get()}))
      window_params.append(
          ir.DictAttr.get(
              {
                  "transform_indices": ir.Attribute.parse("affine_map<(n,m) -> (0,0)>"),
              }
          ),
      )

    f.attributes["window_params"] = ir.ArrayAttr.get(window_params)
    f.arg_attrs = args_attributes
    f.attributes["tpu.core_type"] = ir.Attribute.parse("#tpu.core_type<sc_vector_subcore>")
    assert f.verify(), f
    m = ir.Module.create()
    m.body.append(f)
    ir.SymbolTable(m.operation).insert(f)

    return mosaic.as_tpu_kernel(
        m,
        out_type=core.ShapedArray((idx.shape[0] // reduce_group_size, op.shape[1]), jnp.bfloat16),
    )(
        *(
            [
                idx,
                op,
                topk_weights,
            ]
            if topk_weights is not None
            else [
                idx,
                op,
            ]
        )
    )
