# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused TensorCore (Pallas) MoE `combine` and `permute` kernels.

Used on TPU v6e to run MoE token `permute` and `combine` on TensorCore so
SparseCore remains free to pipeline/hide weight all-gathers and reduce-scatters.

Operations (`T = B * S` tokens, `D` hidden dim, `G` experts, top-`K` routing)
-----------------------------------------------------------------------------
Let `x` `[R = T * K, D]` be stored in expert-sorted order, where row `r` holds
flat token-slot `sort_idx[r]` (`token_id = sort_idx[r] // K`,
`topk_slot = sort_idx[r] % K`), and `inv = argsort(sort_idx)`:

  * `combine(x, sort_idx, w, group_sizes)` -> `y` `[T, D]` (with f32 accumulation):
      Forward:
        y[t] = sum_{k=0}^{K-1} w[t, k] * x[inv[t * K + k]]
      Backward (fused single-pass custom VJP, no `[T, K, D]` HBM intermediate):
        grad_x[inv[t * K + k]] = w[t, k] * dy[t]
        grad_w[t, k]           = <dy[t], x[inv[t * K + k]]>   (in f32)

  * `permute(x, sort_idx, group_sizes, k)` -> `y` `[T * K, D]`:
      Exact transpose of unweighted `combine` (`w = 1`):
      Forward (scatter to expert order):
        y[r] = x[sort_idx[r] // K]
      Backward (gather + sum K copies per token, reuses `_gather_kernel`):
        grad_x[t] = sum_{k=0}^{K-1} dy[inv[t * K + k]]

Algorithm: Contiguous Block-Window DMAs + MXU Routing
-----------------------------------------------------
A naive per-row TensorCore gather is bound by ~20 ns per 1-row DMA descriptor
(~5 ms for `R = 262,144` rows). Instead, tokens are processed in blocks of `C`
tokens (`C = 256` for `combine`, `C = 512` for `permute`, `T // C` grid steps).

Because `jnp.argsort` is **stable**, within each expert `g in [0, G)` the sorted
rows appear in strictly increasing token order. Thus, all rows belonging to
token block `b` (`tokens in [b * C, (b + 1) * C)`) that routed to expert `g`
form **one contiguous HBM row range** `[range_start[b, g], range_end[b, g])` in
`x`. Each grid step `b` issues **at most `G` coarse block-window DMAs** into a
VMEM buffer `window_x_vmem` `[J, D]`, and performs reordering + top-K weighted
reduction as a dense **MXU matrix multiplication**:

  * `combine` forward (gather + weighted reduction):
      out_block[C, D]     = routing_matrix[C, J] @ window_x_vmem[J, D]
  * `combine` backward (fused `grad_weights` + `grad_x` scatter):
      grad_w_per_row[j]   = <dy[tok(j)], window_x_vmem[j]>
      grad_weights[C, K]  = exact 0/1 MXU routing of grad_w_per_row[j] to (tok(j), k(j))
      window_grad_x[J, D] = routing_matrix^T[J, C] @ dy_block[C, D]  -> DMA'd to HBM
  * `permute` forward (unweighted scatter) / backward (unweighted gather-sum):
      window_out[J, D]    = routing_matrix^T[J, C] @ x_block[C, D]   -> DMA'd to HBM
      grad_x_block[C, D]  = routing_matrix[C, J] @ window_dy_vmem[J, D]

Self-Describing Row Metadata (`row_info`) & One-Hot MXU Histogram
-----------------------------------------------------------------
No per-row gathers, scatters, or secondary `argsort` calls are performed in XLA:
  * `_build_row_info(sort_idx, k)` packs each sorted row's `(token % 256,
    token // 256, topk_slot)` elementwise from `sort_idx` into a `[R, 128]` bf16
    array.
  * Each grid step DMAs `row_info` using the exact same `G` windows as `x`
    (filling unused VMEM rows with sentinel `topk_slot = 255`), so every VMEM
    buffer row `j` carries its `(row_token_in_block, row_topk_slot)`.
  * Inside the kernel, `routing_matrix` is constructed on the VPU with one
    compare + select per element, with weights selected from `w_block[C, K]` via
    an exact 0/1 MXU matmul (`w_block @ row_topk_onehot`).
  * Outside the kernel, `_build_dma_table` computes the `(block, expert)` row
    counts via a single one-hot MXU histogram matmul.

Mosaic Tile Alignment & Window Coalescing
-----------------------------------------
Mosaic TPU requires every HBM DMA and dynamic VMEM slice along the row
(2nd-minor) dimension to use offsets and lengths that are multiples of `align`
rows (`A = 8` for 32-bit `(8, 128)` tiles or 2-packed `bf16` `(8, 128)(2, 1)`
memref tiles; `A = 16` for 16-sublane `(16, 128)` `bf16` tiling). Each expert
range `[range_start, range_end)` is widened outward to `[range_start & -A,
round_up(range_end, A))`. Adjacent expert windows in the same token block that
share a granule or abut in HBM are coalesced into a single contiguous VMEM
segment (`vmem_row = hbm_row + row_delta`, with `row_delta` a multiple of `A`),
ensuring every sorted row `r` has a unique buffer position within a block.

Scatter Partial-Granule Preloading & VMEM Carry (`combine` bwd, `permute` fwd)
------------------------------------------------------------------------------
In the scatter direction, output windows are written back to HBM as whole
`align`-row granules. A boundary granule ("partial granule") at the head or tail
of `[range_start, range_end)` may contain rows owned by another token block and
must be read-modify-written. The output buffer `window_out_vmem` is
double-buffered across consecutive grid steps (`[2, max_buf_rows, D]`), and
partial granules holding rows owned by *earlier* blocks are preloaded (rows of
*later* blocks are overwritten by their owner step later):

  * **VMEM Carry (from step `b - 1`)**: Because rows within an expert are in
    block order, the head granule of `(b, g)` is typically the tail granule just
    written by `(b - 1, g)`. Step `b`'s prologue vector-copies that granule
    directly from step `b - 1`'s `window_out_vmem` slot (`_HEAD_CARRY_SRC ->
    _HEAD_CARRY_DST`, `_TAIL_CARRY_SRC -> _TAIL_CARRY_DST`).
  * **Lookahead HBM Stage (from blocks `< b - 1`)**: Any needed partial granule
    not touched by step `b - 1` is asynchronously DMA'd from HBM into step `b`'s
    slot during step `b - 1`'s epilogue (`_HEAD_STAGE_HBM`, `_TAIL_STAGE_HBM`),
    overlapping with step `b - 1`'s compute.

During compute, each row selects between the newly computed value (for rows
owned by block `b`) and the preloaded granule value (for neighbor rows), and the
windows are written back to HBM after waiting for step `b - 1`'s writes.
"""

import functools
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

__all__ = ["combine", "combine_reference", "kernel_supported", "permute"]

_LANES = 128
_COMBINE_BLOCK_TOKENS = 256  # combine: tokens per grid step
_PERMUTE_BLOCK_TOKENS = 512  # permute: no x window in VMEM, so larger blocks fit
_ROW_CHUNK = 256  # buffer rows processed per routing-matrix chunk
_COL_CHUNK = 256  # hidden-dimension columns processed per MXU matmul chunk
_FWD_BUCKET_ROWS = 512  # forward MXU contraction-size bucket in buffer rows
_SCATTER_ROWS_PER_CHUNK = 1024  # scatter direction: buffer rows per compute chunk
_VMEM_LIMIT = 120 * 1024 * 1024

# Fields of the per-block SMEM DMA schedule table (`num_groups` entries per field, one per expert):
_WIN_HBM_START = 0  # HBM start row of this expert's (coalesced) DMA window
_WIN_NUM_ROWS = 1  # number of rows to DMA for this expert (multiple of `align`, 0 if empty)
_WIN_BUF_OFFSET = 2  # destination start row in the VMEM window buffer (`window_x_vmem` / `window_out_vmem`)
_WIN_ROW_DELTA = 3  # `vmem_row - hbm_row` for every row in this expert's coalesced segment
_HEAD_STAGE_HBM = 4  # HBM start row of the head partial granule to preload from HBM (-1 if none)
_TAIL_STAGE_HBM = 5  # HBM start row of the tail partial granule to preload from HBM (-1 if none)
_HEAD_CARRY_SRC = 6  # VMEM row in previous slot to copy head partial granule from (-1 if none)
_HEAD_CARRY_DST = 7  # VMEM row in current slot to copy head partial granule into (-1 if none)
_TAIL_CARRY_SRC = 8  # VMEM row in previous slot to copy tail partial granule from (-1 if none)
_TAIL_CARRY_DST = 9  # VMEM row in current slot to copy tail partial granule into (-1 if none)
_NUM_DMA_TABLE_FIELDS = 10

# Row-info sentinel for buffer rows that hold no valid row of this step:
# topk_slot = 255 (> any valid K <= 128) marks unused filler rows.
_INFO_FILL = 255.0
_INFO_FILL_K = 255
_MAX_TOKENS = 256 * 256


class _Config(NamedTuple):
  """Static kernel configuration (hashable: the non-differentiable custom_vjp argument)."""

  k: int  # experts per token
  block_tokens: int  # tokens per grid step
  num_groups: int  # number of experts G
  split_weights: bool  # True when routing weights are f32 (split into bf16 hi + lo parts)
  interpret: Any  # False or pltpu.InterpretParams
  align: int  # DMA / VMEM row alignment A


# ---------------------------------------------------------------------------
# shapes and preconditions
# ---------------------------------------------------------------------------


def _round_up(x, multiple):
  return (x + multiple - 1) // multiple * multiple


def _dma_table_stride(num_groups):
  """Per-block SMEM table length: a valid rank-1 Pallas TPU block size >= _NUM_DMA_TABLE_FIELDS * num_groups."""
  min_len = _NUM_DMA_TABLE_FIELDS * num_groups
  if min_len > 1024:
    return _round_up(min_len, 1024)
  stride = _LANES
  while stride < min_len:
    stride *= 2
  return stride


def _buffer_rows(block_tokens, k, num_groups, align=8):
  """Static upper bound on VMEM window-buffer rows per token block.

  A token block owns `block_tokens * k` rows across at most `num_groups`
  experts, and aligning both ends of each expert's range to `align` adds at
  most `2 * (align - 1)` padding rows per non-empty expert range.
  """
  block_rows = block_tokens * k
  return _round_up(block_rows + (2 * align - 2) * min(num_groups, block_rows), _ROW_CHUNK)


def kernel_supported(
    x_shape: tuple[int, ...],
    x_dtype: jax.typing.DTypeLike,
    num_tokens: int,
    k: int,
    block_tokens: int,
    align: int = 8,
) -> bool:
  """Returns whether the Pallas kernels support this problem shape and dtype."""
  if len(x_shape) != 2:
    return False
  num_rows, hidden_dim = x_shape
  return (
      jnp.dtype(x_dtype) == jnp.bfloat16
      and hidden_dim % _LANES == 0
      and k <= _LANES
      and align >= 8
      and (align & (align - 1)) == 0
      and num_rows == num_tokens * k
      and num_rows % align == 0
      and block_tokens % _LANES == 0
      and num_tokens % block_tokens == 0
      and num_tokens <= _MAX_TOKENS
  )


def _block_tokens(num_tokens, default):
  """Largest power-of-two divisor of num_tokens that is <= the tuned default (>= 128)."""
  block_tokens = default
  while block_tokens > _LANES and num_tokens % block_tokens != 0:
    block_tokens //= 2
  return block_tokens


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------


def combine_reference(x: jax.Array, sort_idx: jax.Array, weights: jax.Array) -> jax.Array:
  """Computes `sum_k w * x[argsort(sort_idx)]` in pure JAX with f32 accumulation."""
  num_tokens, k = weights.shape
  inv_sort_idx = jnp.argsort(sort_idx)
  gathered = x[inv_sort_idx].reshape(num_tokens, k, x.shape[1]).astype(jnp.float32)
  out = jnp.einsum("tkd,tk->td", gathered, weights.astype(jnp.float32), precision=lax.Precision.HIGHEST)
  return out.astype(x.dtype)


# ---------------------------------------------------------------------------
# metadata (plain XLA: cumsums, one small one-hot matmul, elementwise row info)
# ---------------------------------------------------------------------------


def _build_dma_table(sort_idx, group_sizes, num_tokens, k, block_tokens, align=8):
  """Builds the per-block DMA schedule table, flat int32 [num_blocks * _dma_table_stride(num_groups)].

  No gathers / scatters / sorts: the (block, expert) row counts are a one-hot
  histogram on the MXU (exact because counts < 2^24).
  """
  num_groups = group_sizes.shape[0]
  num_blocks = num_tokens // block_tokens
  num_rows = num_tokens * k

  # 1. Contiguous row range [range_start, range_end) in `x` for each (block, expert) pair.
  group_sizes = group_sizes.astype(jnp.int32)
  group_ends = jnp.cumsum(group_sizes)
  group_starts = group_ends - group_sizes
  row_block = sort_idx.astype(jnp.int32) // (k * block_tokens)
  row_idx = jnp.arange(num_rows, dtype=jnp.int32)[:, None]
  in_group = ((row_idx >= group_starts[None, :]) & (row_idx < group_ends[None, :])).astype(jnp.bfloat16)
  in_block = (row_block[:, None] == jnp.arange(num_blocks, dtype=jnp.int32)[None, :]).astype(jnp.bfloat16)
  rows_per_block_and_expert = jnp.einsum("rb,rg->bg", in_block, in_group, preferred_element_type=jnp.float32).astype(
      jnp.int32
  )
  range_start = group_starts[None, :] + jnp.cumsum(rows_per_block_and_expert, axis=0) - rows_per_block_and_expert
  range_end = range_start + rows_per_block_and_expert
  nonempty = rows_per_block_and_expert > 0

  # 2. Aligned DMA windows [aligned_win_start, aligned_win_end) and within-block coalescing.
  # Consecutive expert windows in the same block are non-decreasing and can
  # share at most one alignment granule; each range only DMAs the suffix not yet covered.
  aligned_win_start = jnp.bitwise_and(range_start, -align)
  aligned_win_end = jnp.bitwise_and(range_end + (align - 1), -align)
  active_win_end = jnp.where(nonempty, aligned_win_end, -1)
  prev_covered_end = jnp.pad(lax.cummax(active_win_end, axis=1)[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
  dma_hbm_start = jnp.maximum(aligned_win_start, prev_covered_end)
  dma_num_rows = jnp.where(nonempty, jnp.maximum(aligned_win_end - dma_hbm_start, 0), 0)
  dma_vmem_offset = jnp.cumsum(dma_num_rows, axis=1) - dma_num_rows
  hbm_to_vmem_row_delta = dma_vmem_offset - dma_hbm_start  # vmem_row = hbm_row + delta

  # 3. Scatter staging for partial boundary granules.
  # A granule is preloaded (read-modify-written) iff it holds rows owned by an
  # earlier block: rows in [group_starts[g'], range_start[b, g')) for some expert g'.
  head_partial_granule_hbm = jnp.where(nonempty & (jnp.bitwise_and(range_start, align - 1) != 0), aligned_win_start, -1)
  tail_partial_granule_hbm = jnp.where(
      nonempty & (jnp.bitwise_and(range_end, align - 1) != 0), aligned_win_end - align, -1
  )

  def has_earlier_block_rows(granule_hbm_start):
    granule = granule_hbm_start[:, :, None]
    overlap = jnp.maximum(granule, group_starts[None, None, :]) < jnp.minimum(granule + align, range_start[:, None, :])
    return (granule_hbm_start >= 0) & jnp.any(overlap, axis=2)

  # A tail granule that is also the head granule of a later expert in this block
  # is already staged by that head -- never preload the same granule twice.
  valid_heads = jnp.where(nonempty, head_partial_granule_hbm, -2)
  shared_with_head = jnp.any(tail_partial_granule_hbm[:, :, None] == valid_heads[:, None, :], axis=2)
  need_head = has_earlier_block_rows(head_partial_granule_hbm)
  need_tail = has_earlier_block_rows(tail_partial_granule_hbm) & ~shared_with_head

  # If a needed granule lies in a window written by the previous block, copy
  # it from the previous step's VMEM output slot; otherwise preload it from HBM.
  prev_dma_start = jnp.pad(
      jnp.where(dma_num_rows > 0, dma_hbm_start, 1 << 30)[:-1], ((1, 0), (0, 0)), constant_values=1 << 30
  )
  prev_dma_end = jnp.pad(
      jnp.where(dma_num_rows > 0, dma_hbm_start + dma_num_rows, -1)[:-1], ((1, 0), (0, 0)), constant_values=-1
  )
  prev_row_delta = jnp.pad(hbm_to_vmem_row_delta[:-1], ((1, 0), (0, 0)), constant_values=0)

  def find_in_prev_block(granule_hbm_start):
    granule = granule_hbm_start[:, :, None]
    hit = (granule >= prev_dma_start[:, None, :]) & (granule < prev_dma_end[:, None, :])
    in_prev = jnp.any(hit, axis=2) & (granule_hbm_start >= 0)
    prev_vmem_row = jnp.sum(jnp.where(hit, granule + prev_row_delta[:, None, :], 0), axis=2)
    return in_prev, prev_vmem_row

  head_in_prev, head_prev_vmem_row = find_in_prev_block(head_partial_granule_hbm)
  tail_in_prev, tail_prev_vmem_row = find_in_prev_block(tail_partial_granule_hbm)
  head_stage_hbm = jnp.where(need_head & ~head_in_prev, head_partial_granule_hbm, -1)
  tail_stage_hbm = jnp.where(need_tail & ~tail_in_prev, tail_partial_granule_hbm, -1)
  head_carry_src = jnp.where(need_head & head_in_prev, head_prev_vmem_row, -1)
  head_carry_dst = jnp.where(need_head & head_in_prev, head_partial_granule_hbm + hbm_to_vmem_row_delta, -1)
  tail_carry_src = jnp.where(need_tail & tail_in_prev, tail_prev_vmem_row, -1)
  tail_carry_dst = jnp.where(need_tail & tail_in_prev, tail_partial_granule_hbm + hbm_to_vmem_row_delta, -1)

  table_fields = [
      dma_hbm_start,
      dma_num_rows,
      dma_vmem_offset,
      hbm_to_vmem_row_delta,
      head_stage_hbm,
      tail_stage_hbm,
      head_carry_src,
      head_carry_dst,
      tail_carry_src,
      tail_carry_dst,
  ]
  table_width = _NUM_DMA_TABLE_FIELDS * num_groups
  dma_table = jnp.stack(table_fields, axis=1).astype(jnp.int32).reshape(num_blocks, table_width)
  dma_table = jnp.pad(dma_table, ((0, 0), (0, _dma_table_stride(num_groups) - table_width)))
  return dma_table.reshape(-1)


def _build_row_info(sort_idx, k):
  """Builds [R, 128] bf16 per-row metadata: lanes 0..2 hold (token % 256, token // 256, topk_slot).

  All values are integers < 256, exact in bf16. DMA'd by the kernels using the
  exact same row windows as `sorted_x`, so every row in `window_x_vmem` has its
  matching `(token_id, topk_slot)` tag at the same row in `window_info_vmem`.
  """
  flat_token_topk = sort_idx.astype(jnp.int32)
  token_id = flat_token_topk // k
  topk_slot = flat_token_topk % k
  info_cols = jnp.stack([token_id % 256, token_id // 256, topk_slot], axis=1).astype(jnp.bfloat16)
  return jnp.pad(info_cols, ((0, 0), (0, _LANES - 3)))


# ---------------------------------------------------------------------------
# in-kernel building blocks: window DMAs
# ---------------------------------------------------------------------------


def _wait_dma_rows(vmem_buf, slot, dma_sem, num_rows, align):
  """Waits until `num_rows` (multiple of align, may be 0) rows of `vmem_buf[slot]` DMAs have completed on `dma_sem`."""

  @pl.when(num_rows > 0)
  def _():
    rows = pl.ds(0, pl.multiple_of(num_rows, align))
    pltpu.make_async_copy(vmem_buf.at[slot, rows], vmem_buf.at[slot, rows], dma_sem).wait()


def _fill_both_slots(vmem_buf, buf_rows, fill_value):
  """Fills rows `[0, buf_rows)` of both double-buffer slots of `vmem_buf` with `fill_value`."""
  tile = jnp.full((_LANES, vmem_buf.shape[-1]), fill_value, vmem_buf.dtype)

  def fill_tile(tile_idx, carry):
    rows = pl.ds(pl.multiple_of(tile_idx * _LANES, _LANES), _LANES)
    vmem_buf[0, rows, :] = tile
    vmem_buf[1, rows, :] = tile
    return carry

  lax.fori_loop(0, buf_rows // _LANES, fill_tile, 0)


def _num_packed_rows(dma_table_ref, num_groups):
  """Total VMEM buffer rows packed for the token block described by `dma_table_ref`."""
  last_group = num_groups - 1
  return dma_table_ref[_WIN_BUF_OFFSET * num_groups + last_group] + (
      dma_table_ref[_WIN_NUM_ROWS * num_groups + last_group]
  )


class _WindowReadStream(NamedTuple):
  """One HBM array gathered window-by-window into a double-buffered VMEM scratch buffer."""

  hbm_source: Any  # [R, cols] source array in HBM
  vmem_buf: Any  # VMEM [2, buf_rows, cols] destination buffer (2 slots for double-buffering)
  dma_sem: Any  # DMA semaphores [2] (one per slot)
  init_fill: Any  # value both slots are initialized with at step 0, or None


def _start_expert_window_dmas(dma_table_ref, streams, dst_slot, num_groups, align):
  """Issues async HBM -> VMEM DMAs for all `num_groups` expert windows into `dst_slot`."""

  def issue_expert_window(group_idx, carry):
    num_rows = dma_table_ref[_WIN_NUM_ROWS * num_groups + group_idx]

    @pl.when(num_rows > 0)
    def _():
      hbm_start = dma_table_ref[_WIN_HBM_START * num_groups + group_idx]
      vmem_offset = dma_table_ref[_WIN_BUF_OFFSET * num_groups + group_idx]
      hbm_slice = pl.ds(pl.multiple_of(hbm_start, align), pl.multiple_of(num_rows, align))
      vmem_slice = pl.ds(pl.multiple_of(vmem_offset, align), pl.multiple_of(num_rows, align))
      for stream in streams:
        pltpu.make_async_copy(
            stream.hbm_source.at[hbm_slice],
            stream.vmem_buf.at[dst_slot, vmem_slice],
            stream.dma_sem.at[dst_slot],
        ).start()

    return carry

  lax.fori_loop(0, num_groups, issue_expert_window, 0)


def _prefetch_expert_windows(cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align):
  """Double-buffered window prefetch: starts next block's window DMAs and waits for current block's.

  Returns the active double-buffer `slot` (0 or 1) holding this block's rows.
  At block 0, any stream with `init_fill is not None` initializes both VMEM slots
  so unused tail rows hold safe values (0.0 for activations, `_INFO_FILL` for row_info).
  """
  block_idx = pl.program_id(0)
  slot = block_idx % 2

  @pl.when(block_idx == 0)
  def _():
    for stream in streams:
      if stream.init_fill is not None:
        _fill_both_slots(stream.vmem_buf, buf_rows, stream.init_fill)
    _start_expert_window_dmas(cur_dma_table_ref, streams, 0, num_groups, align)

  @pl.when(block_idx + 1 < pl.num_programs(0))
  def _():
    _start_expert_window_dmas(next_dma_table_ref, streams, 1 - slot, num_groups, align)

  packed_rows = _num_packed_rows(cur_dma_table_ref, num_groups)
  for stream in streams:
    _wait_dma_rows(stream.vmem_buf, slot, stream.dma_sem.at[slot], packed_rows, align)
  return slot


# ---------------------------------------------------------------------------
# in-kernel building blocks: row-info decoding and bf16 splitting
# ---------------------------------------------------------------------------


def _split_into_bf16_parts(val, dtype, num_parts):
  """Splits `val` into `num_parts` bf16 tensors whose f32 sum equals `val` (exact for num_parts=3)."""
  parts = []
  for _ in range(num_parts):
    part = val.astype(dtype)
    parts.append(part)
    val = val - part.astype(jnp.float32)
  return parts


def _decode_row_info(
    window_info_vmem, slot, row_offset, num_rows, block_tokens, block_start_token, packed_rows, *, transpose=False
):
  """Decodes `(row_token_in_block, row_topk_slot)` for buffer rows `[row_offset, row_offset + num_rows)`.

  Buffer rows outside `[0, packed_rows)` or belonging to an adjacent token block
  (brought in by 8/16-row DMA alignment padding) receive `row_token_in_block = -1`.

  When `transpose=False`, returns shapes `[num_rows, 1]`.
  When `transpose=True`, transposes the full `[num_rows, 128]` tile before slicing
  lanes 0..2 (satisfying Mosaic's 128-lane transpose alignment) and returns `[1, num_rows]`.
  """
  info_tile = window_info_vmem[slot, pl.ds(row_offset, num_rows), :].astype(jnp.float32)
  if transpose:
    info_tile = info_tile.T  # [128, num_rows]
    token_lo, token_hi, row_topk_slot = info_tile[0:1, :], info_tile[1:2, :], info_tile[2:3, :].astype(jnp.int32)
    buf_row_idx = lax.broadcasted_iota(jnp.int32, (1, num_rows), 1) + row_offset
  else:
    token_lo, token_hi, row_topk_slot = info_tile[:, 0:1], info_tile[:, 1:2], info_tile[:, 2:3].astype(jnp.int32)
    buf_row_idx = lax.broadcasted_iota(jnp.int32, (num_rows, 1), 0) + row_offset

  row_token_in_block = (token_lo + 256.0 * token_hi).astype(jnp.int32) - block_start_token
  belongs_to_current_block = (
      (row_token_in_block >= 0)
      & (row_token_in_block < block_tokens)
      & (buf_row_idx < packed_rows)
      & (row_topk_slot < _INFO_FILL_K)
  )
  return jnp.where(belongs_to_current_block, row_token_in_block, -1), row_topk_slot


def _dma_table_block_specs(num_blocks, num_groups):
  """SMEM BlockSpecs for the current token block's DMA table and the next block's (clamped)."""
  stride = _dma_table_stride(num_groups)
  return [
      pl.BlockSpec((stride,), lambda b: (b,), memory_space=pltpu.SMEM),
      pl.BlockSpec((stride,), lambda b: (jnp.minimum(b + 1, num_blocks - 1),), memory_space=pltpu.SMEM),
  ]


# ---------------------------------------------------------------------------
# gather direction: out_ref = routing_matrix @ window_x_vmem
# (used by combine forward with weights, and permute backward with unit weights)
# ---------------------------------------------------------------------------


def _gather_kernel(
    weights_ref,
    cur_dma_table_ref,
    next_dma_table_ref,
    sorted_x_hbm,
    row_info_hbm,
    out_ref,
    window_x_vmem,
    window_info_vmem,
    routing_matrix_vmem,
    x_dma_sem,
    info_dma_sem,
    *,
    num_groups,
    buf_rows,
    hidden_dim,
    align,
):
  """Gathers and sums the expert-sorted rows belonging to one token block `[block_tokens, hidden_dim]`.

  Step 1: Prefetch this token block's `num_groups` contiguous expert windows from
          `sorted_x_hbm` into `window_x_vmem[slot]` (`[buf_rows, hidden_dim]`) and
          their `(token_id, topk_slot)` tags into `window_info_vmem[slot]`.
  Step 2: Build `routing_matrix_vmem` (`[num_parts, block_tokens, contract_rows]`) where
          entry `(c, j)` is `weights_ref[c, k_j]` (or `1.0` if unweighted) when
          buffer row `j` belongs to token `c`, else `0.0`.
  Step 3: Compute `out_ref = routing_matrix_vmem @ window_x_vmem[slot]` on the MXU.
  """
  block_tokens = out_ref.shape[0]
  num_parts = routing_matrix_vmem.shape[0]  # 1 for bf16 weights / unweighted; 2 (hi + lo) for f32 weights

  # Step 1: DMA the expert windows for this token block into VMEM (and prefetch next block).
  streams = (
      _WindowReadStream(sorted_x_hbm, window_x_vmem, x_dma_sem, init_fill=0),
      _WindowReadStream(row_info_hbm, window_info_vmem, info_dma_sem, init_fill=_INFO_FILL),
  )
  slot = _prefetch_expert_windows(cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align)
  packed_rows = _num_packed_rows(cur_dma_table_ref, num_groups)

  # Round `packed_rows` up to the smallest static multiple of `_FWD_BUCKET_ROWS` (512)
  # so the MXU matmul only needs to compile for a few fixed contraction sizes.
  buckets = tuple(range(_FWD_BUCKET_ROWS, buf_rows, _FWD_BUCKET_ROWS)) + (buf_rows,)
  contract_rows = jnp.int32(buckets[-1])
  for bucket_rows in reversed(buckets[:-1]):
    contract_rows = jnp.where(packed_rows <= bucket_rows, jnp.int32(bucket_rows), contract_rows)

  # Step 2: Build `routing_matrix_vmem[:, :, :contract_rows]` in chunks of `_ROW_CHUNK` (256) columns.
  weight_bf16_parts = (
      None if weights_ref is None else _split_into_bf16_parts(weights_ref[...], routing_matrix_vmem.dtype, num_parts)
  )
  token_in_block_iota = lax.broadcasted_iota(jnp.int32, (block_tokens, _ROW_CHUNK), 0)
  topk_slot_iota = lax.broadcasted_iota(jnp.int32, (_LANES, _ROW_CHUNK), 0)
  block_start_token = pl.program_id(0) * block_tokens

  def build_routing_matrix_chunk(chunk_idx, carry):
    row_offset = pl.multiple_of(chunk_idx * _ROW_CHUNK, _ROW_CHUNK)
    buf_cols = pl.ds(row_offset, _ROW_CHUNK)
    row_token_in_block, row_topk_slot = _decode_row_info(
        window_info_vmem, slot, row_offset, _ROW_CHUNK, block_tokens, block_start_token, packed_rows, transpose=True
    )
    # `is_target_token[c, j]` is True iff VMEM buffer row `j` belongs to token `c` of this block.
    is_target_token = token_in_block_iota == row_token_in_block  # [block_tokens, _ROW_CHUNK]
    if weight_bf16_parts is None:
      routing_matrix_vmem[0, :, buf_cols] = is_target_token.astype(routing_matrix_vmem.dtype)
    else:
      # Select `weights[c, row_topk_slot[j]]` via MXU dot with a one-hot matrix:
      # `w_part` [block_tokens, 128] @ `row_topk_onehot` [128, _ROW_CHUNK] -> [block_tokens, _ROW_CHUNK].
      row_topk_onehot = (topk_slot_iota == row_topk_slot).astype(routing_matrix_vmem.dtype)
      for part_idx, w_part in enumerate(weight_bf16_parts):
        token_row_weights = jnp.dot(w_part, row_topk_onehot, preferred_element_type=jnp.float32)
        routing_matrix_vmem[part_idx, :, buf_cols] = jnp.where(is_target_token, token_row_weights, 0.0).astype(
            routing_matrix_vmem.dtype
        )
    return carry

  lax.fori_loop(0, contract_rows // _ROW_CHUNK, build_routing_matrix_chunk, 0)

  # Step 3: Compute `out_ref[:, cols] = routing_matrix_vmem @ window_x_vmem` across column chunks.
  col_chunk = _COL_CHUNK if hidden_dim % _COL_CHUNK == 0 else _LANES
  for bucket_rows in buckets:

    @pl.when(contract_rows == bucket_rows)
    def _(bucket_rows=bucket_rows):
      def accumulate_output_col_chunk(col_idx, carry):
        hidden_cols = pl.ds(pl.multiple_of(col_idx * col_chunk, col_chunk), col_chunk)
        x_chunk = window_x_vmem[slot, pl.ds(0, bucket_rows), hidden_cols]
        acc = jnp.dot(routing_matrix_vmem[0, :, pl.ds(0, bucket_rows)], x_chunk, preferred_element_type=jnp.float32)
        for part_idx in range(1, num_parts):
          acc = acc + jnp.dot(
              routing_matrix_vmem[part_idx, :, pl.ds(0, bucket_rows)], x_chunk, preferred_element_type=jnp.float32
          )
        out_ref[:, hidden_cols] = acc.astype(out_ref.dtype)
        return carry

      lax.fori_loop(0, hidden_dim // col_chunk, accumulate_output_col_chunk, 0)


def _gather_call(sorted_x, padded_weights, dma_table, row_info, *, num_tokens, cfg: _Config, name):
  """Launches `_gather_kernel` to compute `[num_tokens, D]` from expert-sorted `sorted_x` `[R, D]`."""
  num_rows, hidden_dim = sorted_x.shape
  block_tokens, num_groups, align = cfg.block_tokens, cfg.num_groups, cfg.align
  num_blocks = num_tokens // block_tokens
  buf_rows = _buffer_rows(block_tokens, num_rows // num_tokens, num_groups, align)
  num_parts = 2 if (padded_weights is not None and cfg.split_weights) else 1

  kernel_kwargs = {"num_groups": num_groups, "buf_rows": buf_rows, "hidden_dim": hidden_dim, "align": align}
  table_and_hbm_specs = _dma_table_block_specs(num_blocks, num_groups) + [
      pl.BlockSpec(memory_space=pl.ANY),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  if padded_weights is not None:
    kernel = functools.partial(_gather_kernel, **kernel_kwargs)
    operands = (padded_weights, dma_table, dma_table, sorted_x, row_info)
    in_specs = [pl.BlockSpec((block_tokens, _LANES), lambda b: (b, 0)), *table_and_hbm_specs]
  else:
    kernel = functools.partial(_gather_kernel, None, **kernel_kwargs)
    operands = (dma_table, dma_table, sorted_x, row_info)
    in_specs = table_and_hbm_specs

  itemsize = jnp.dtype(sorted_x.dtype).itemsize
  vmem_bytes = (
      2 * buf_rows * (hidden_dim + _LANES) * itemsize  # window_x_vmem + window_info_vmem
      + 2 * block_tokens * hidden_dim * itemsize  # out_ref double-buffered blocks
      + num_parts * block_tokens * buf_rows * itemsize  # routing_matrix_vmem
      + (2 * block_tokens * _LANES * 4 if padded_weights is not None else 0)  # weights_ref blocks
      + 16 * 1024 * 1024  # dot operands / temporaries, compiler scratch
  )
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(
          (num_tokens, hidden_dim), sorted_x.dtype, manual_axis_type=jax.typeof(sorted_x).manual_axis_type
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_blocks,),
          in_specs=in_specs,
          out_specs=pl.BlockSpec((block_tokens, hidden_dim), lambda b: (b, 0)),
          scratch_shapes=[
              pltpu.VMEM((2, buf_rows, hidden_dim), sorted_x.dtype),  # window_x_vmem
              pltpu.VMEM((2, buf_rows, _LANES), row_info.dtype),  # window_info_vmem
              pltpu.VMEM((num_parts, block_tokens, buf_rows), sorted_x.dtype),  # routing_matrix_vmem
              pltpu.SemaphoreType.DMA((2,)),
              pltpu.SemaphoreType.DMA((2,)),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",), vmem_limit_bytes=min(vmem_bytes, _VMEM_LIMIT)
      ),
      interpret=cfg.interpret,
      name=name,
  )(*operands)


# ---------------------------------------------------------------------------
# scatter direction: output windows written back by DMA (combine bwd, permute fwd)
# ---------------------------------------------------------------------------


class _ScatterWindowState(NamedTuple):
  """Output state for kernels that scatter token-block results back into expert-sorted HBM windows.

  `window_out_vmem[slot]` holds the output rows of the token block processed in
  grid step `slot` at the same VMEM buffer positions as its input rows
  (`vmem_row = hbm_row + delta`), and its windows are DMA'd back to `out_hbm`.
  Partial boundary granules that also contain rows from earlier token blocks are
  preloaded into the slot one step ahead so non-owned rows are preserved.
  """

  out_hbm: Any  # [R, D] destination array in HBM
  window_out_vmem: Any  # VMEM [2, buf_rows, D] double-buffered output window buffer
  writeback_dma_sem: Any  # DMA semaphores [2]: window writebacks issued from each slot
  preload_dma_sem: Any  # DMA semaphores [2]: partial granules preloaded from HBM into each slot
  written_rows_smem: Any  # SMEM [2] int32: total rows written from each slot (wait count)
  preloaded_rows_smem: Any  # SMEM [2] int32: total rows preloaded into each slot (wait count)
  num_groups: int  # number of experts G
  align: int  # row alignment A


def _preload_partial_granules_from_hbm(scatter: _ScatterWindowState, dma_table_ref, dst_slot, group_idx):
  """Starts HBM -> VMEM DMAs for expert `group_idx`'s head/tail partial granules into `window_out_vmem[dst_slot]`.

  Returns the number of rows issued on `preload_dma_sem[dst_slot]`.
  """
  num_groups, align = scatter.num_groups, scatter.align
  row_delta = dma_table_ref[_WIN_ROW_DELTA * num_groups + group_idx]
  preloaded_rows = jnp.int32(0)
  for table_field in (_HEAD_STAGE_HBM, _TAIL_STAGE_HBM):
    granule_hbm_row = dma_table_ref[table_field * num_groups + group_idx]

    @pl.when(granule_hbm_row >= 0)
    def _(granule_hbm_row=granule_hbm_row):
      pltpu.make_async_copy(
          scatter.out_hbm.at[pl.ds(pl.multiple_of(granule_hbm_row, align), align)],
          scatter.window_out_vmem.at[dst_slot, pl.ds(pl.multiple_of(granule_hbm_row + row_delta, align), align)],
          scatter.preload_dma_sem.at[dst_slot],
      ).start()

    preloaded_rows = preloaded_rows + align * (granule_hbm_row >= 0).astype(jnp.int32)
  return preloaded_rows


def _copy_shared_granule_from_prev_slot(
    scatter: _ScatterWindowState, dma_table_ref, src_slot, dst_slot, src_field, dst_field, group_idx
):
  """Vector-copies one shared boundary granule of expert `group_idx` from `src_slot` to `dst_slot`."""
  num_groups, align = scatter.num_groups, scatter.align
  src_vmem_row = dma_table_ref[src_field * num_groups + group_idx]

  @pl.when(src_vmem_row >= 0)
  def _():
    dst_vmem_row = dma_table_ref[dst_field * num_groups + group_idx]
    scatter.window_out_vmem[dst_slot, pl.ds(pl.multiple_of(dst_vmem_row, align), align), :] = scatter.window_out_vmem[
        src_slot, pl.ds(pl.multiple_of(src_vmem_row, align), align), :
    ]


def _prepare_scatter_output_slot(scatter: _ScatterWindowState, cur_dma_table_ref, slot):
  """Ensures `window_out_vmem[slot]` holds all preloaded partial granules before compute starts.

  Block 0 preloads its own partial granules from HBM. Every block waits for the
  HBM preloads into `slot` (started one step ahead) to finish, then vector-copies
  any boundary granules shared with the immediately preceding block (`1 - slot`).
  """
  block_idx = pl.program_id(0)

  @pl.when(block_idx == 0)
  def _():
    scatter.preloaded_rows_smem[0] = lax.fori_loop(
        0,
        scatter.num_groups,
        lambda gi, total: total + _preload_partial_granules_from_hbm(scatter, cur_dma_table_ref, 0, gi),
        jnp.int32(0),
    )

  _wait_dma_rows(
      scatter.window_out_vmem, slot, scatter.preload_dma_sem.at[slot], scatter.preloaded_rows_smem[slot], scatter.align
  )

  @pl.when(block_idx > 0)
  def _():
    def copy_expert_carries(group_idx, carry):
      _copy_shared_granule_from_prev_slot(
          scatter, cur_dma_table_ref, 1 - slot, slot, _HEAD_CARRY_SRC, _HEAD_CARRY_DST, group_idx
      )
      _copy_shared_granule_from_prev_slot(
          scatter, cur_dma_table_ref, 1 - slot, slot, _TAIL_CARRY_SRC, _TAIL_CARRY_DST, group_idx
      )
      return carry

    lax.fori_loop(0, scatter.num_groups, copy_expert_carries, 0)


def _write_back_windows_and_preload_next(scatter: _ScatterWindowState, cur_dma_table_ref, next_dma_table_ref, slot):
  """DMAs `window_out_vmem[slot]` windows back to `out_hbm` and preloads the next block into `1 - slot`.

  Consecutive blocks may write to the same boundary granule, so the previous
  block's HBM writes (`1 - slot`) must complete before this block issues its
  writes. HBM preloads for the next block never overlap this block's writes
  (any overlapping granule is carried in VMEM instead), so both are issued
  together in a single loop over the experts.
  """
  block_idx = pl.program_id(0)
  num_blocks = pl.num_programs(0)
  num_groups, align = scatter.num_groups, scatter.align

  @pl.when(block_idx > 0)
  def _():
    _wait_dma_rows(
        scatter.window_out_vmem,
        1 - slot,
        scatter.writeback_dma_sem.at[1 - slot],
        scatter.written_rows_smem[1 - slot],
        align,
    )

  def start_expert_window_write(group_idx):
    num_rows = cur_dma_table_ref[_WIN_NUM_ROWS * num_groups + group_idx]

    @pl.when(num_rows > 0)
    def _():
      vmem_offset = cur_dma_table_ref[_WIN_BUF_OFFSET * num_groups + group_idx]
      hbm_start = cur_dma_table_ref[_WIN_HBM_START * num_groups + group_idx]
      vmem_slice = pl.ds(pl.multiple_of(vmem_offset, align), pl.multiple_of(num_rows, align))
      hbm_slice = pl.ds(pl.multiple_of(hbm_start, align), pl.multiple_of(num_rows, align))
      pltpu.make_async_copy(
          scatter.window_out_vmem.at[slot, vmem_slice],
          scatter.out_hbm.at[hbm_slice],
          scatter.writeback_dma_sem.at[slot],
      ).start()

    return num_rows

  @pl.when(block_idx + 1 < num_blocks)
  def _():
    def issue_write_and_preload(group_idx, counts):
      written, preloaded = counts
      return (
          written + start_expert_window_write(group_idx),
          preloaded + _preload_partial_granules_from_hbm(scatter, next_dma_table_ref, 1 - slot, group_idx),
      )

    scatter.written_rows_smem[slot], scatter.preloaded_rows_smem[1 - slot] = lax.fori_loop(
        0, num_groups, issue_write_and_preload, (jnp.int32(0), jnp.int32(0))
    )

  @pl.when(block_idx == num_blocks - 1)
  def _():
    total_written = lax.fori_loop(0, num_groups, lambda gi, nw: nw + start_expert_window_write(gi), jnp.int32(0))
    _wait_dma_rows(scatter.window_out_vmem, slot, scatter.writeback_dma_sem.at[slot], total_written, align)


# ---------------------------------------------------------------------------
# combine backward kernel
# ---------------------------------------------------------------------------


def _bwd_compute(
    weights_ref,
    grad_out_ref,
    grad_weights_ref,
    window_x_vmem,
    window_info_vmem,
    window_grad_x_vmem,
    slot,
    packed_rows,
    block_start_token,
    *,
    block_tokens,
    hidden_dim,
    buf_rows,
    split_weights,
):
  """Computes `grad_weights_ref` `[block_tokens, 128]` and `window_grad_x_vmem` rows for one token block.

  Processes `window_x_vmem[slot]` in chunks of `chunk_rows` (up to 1024) buffer rows:
    1. `row_token_dot_products = x_chunk @ grad_out_ref^T` (`[chunk_rows, block_tokens]` on MXU)
    2. `grad_weight_per_buf_row[j] = row_token_dot_products[j, token(j)]` (masked reduction across tokens)
    3. Route `grad_weight_per_buf_row[j]` into `grad_weights_ref[token(j), k(j)]` via exact 3-part bf16 MXU dot
    4. Build transposed routing matrix `routing_chunk[j, c] = weights[c, k(j)]` if `token(j) == c`
    5. Compute `grad_x_chunk = routing_chunk @ grad_out_ref` and write owned rows into `window_grad_x_vmem[slot]`
       (preserving preloaded rows belonging to other token blocks).
  """
  dtype = grad_out_ref.dtype
  col_chunk = _COL_CHUNK if hidden_dim % _COL_CHUNK == 0 else _LANES
  chunk_rows = min(_SCATTER_ROWS_PER_CHUNK, buf_rows)
  weights_t_bf16_parts = _split_into_bf16_parts(weights_ref[...].T, dtype, 2 if split_weights else 1)  # [128, C]
  token_col_iota = lax.broadcasted_iota(jnp.int32, (chunk_rows, block_tokens), 1)
  topk_col_iota = lax.broadcasted_iota(jnp.int32, (chunk_rows, _LANES), 1)
  chunk_row_iota = lax.broadcasted_iota(jnp.int32, (chunk_rows, 1), 0)
  token_row_iota = lax.broadcasted_iota(jnp.int32, (block_tokens, chunk_rows), 0)
  num_chunks = (packed_rows + (chunk_rows - 1)) // chunk_rows

  def process_row_chunk(chunk_idx, grad_weights_acc):
    chunk_start = chunk_idx * chunk_rows
    # Clamp the last chunk so `[row_offset, row_offset + chunk_rows)` stays inside `[0, buf_rows)`.
    row_offset = pl.multiple_of(jnp.minimum(chunk_start, buf_rows - chunk_rows), _ROW_CHUNK)
    buf_rows_slice = pl.ds(row_offset, chunk_rows)
    row_token_in_block, row_topk_slot = _decode_row_info(
        window_info_vmem, slot, row_offset, chunk_rows, block_tokens, block_start_token, packed_rows
    )
    is_target_token = token_col_iota == row_token_in_block  # [chunk_rows, block_tokens]

    # 1-3. Weight gradient `dW[c, k] = <dy[c], x_j>` for buffer rows `j` belonging to `(c, k)`.
    row_token_dot_products = lax.dot_general(
        window_x_vmem[slot, buf_rows_slice, :],
        grad_out_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    grad_weight_per_buf_row = jnp.sum(
        jnp.where(is_target_token, row_token_dot_products, 0.0), axis=1, keepdims=True
    )  # [chunk_rows, 1]
    not_revisited_by_clamped_chunk = chunk_row_iota + row_offset >= chunk_start
    grad_weight_by_topk_slot = jnp.where(
        (topk_col_iota == row_topk_slot) & not_revisited_by_clamped_chunk, grad_weight_per_buf_row, 0.0
    )  # [chunk_rows, 128]
    row_token_in_block_t, _ = _decode_row_info(
        window_info_vmem, slot, row_offset, chunk_rows, block_tokens, block_start_token, packed_rows, transpose=True
    )
    is_target_token_t = (token_row_iota == row_token_in_block_t).astype(dtype)  # [block_tokens, chunk_rows]
    for dw_part in _split_into_bf16_parts(grad_weight_by_topk_slot, dtype, 3):
      grad_weights_acc = grad_weights_acc + jnp.dot(is_target_token_t, dw_part, preferred_element_type=jnp.float32)

    # 4-5. Input activation gradient `dx_j = W[token(j), k(j)] * dy[token(j)]` for owned rows.
    row_topk_onehot = (topk_col_iota == row_topk_slot).astype(dtype)  # [chunk_rows, 128]
    routing_chunk_parts = [
        jnp.where(is_target_token, jnp.dot(row_topk_onehot, wt_part, preferred_element_type=jnp.float32), 0.0).astype(
            dtype
        )
        for wt_part in weights_t_bf16_parts
    ]  # each [chunk_rows, block_tokens]
    belongs_to_current_block = row_token_in_block >= 0  # [chunk_rows, 1]

    def compute_grad_x_col_chunk(col_idx, carry):
      hidden_cols = pl.ds(pl.multiple_of(col_idx * col_chunk, col_chunk), col_chunk)
      dy_cols = grad_out_ref[:, hidden_cols]
      grad_x_chunk = jnp.dot(routing_chunk_parts[0], dy_cols, preferred_element_type=jnp.float32)
      for routing_part in routing_chunk_parts[1:]:
        grad_x_chunk = grad_x_chunk + jnp.dot(routing_part, dy_cols, preferred_element_type=jnp.float32)
      window_grad_x_vmem[slot, buf_rows_slice, hidden_cols] = jnp.where(
          belongs_to_current_block,
          grad_x_chunk.astype(window_grad_x_vmem.dtype),
          window_grad_x_vmem[slot, buf_rows_slice, hidden_cols],
      )
      return carry

    lax.fori_loop(0, hidden_dim // col_chunk, compute_grad_x_col_chunk, 0)
    return grad_weights_acc

  grad_weights_ref[...] = lax.fori_loop(0, num_chunks, process_row_chunk, jnp.zeros((block_tokens, _LANES), jnp.float32))


def _bwd_kernel(
    cur_dma_table_ref,
    next_dma_table_ref,
    weights_ref,
    grad_out_ref,
    sorted_x_hbm,
    row_info_hbm,
    grad_weights_ref,
    grad_x_hbm,
    window_x_vmem,
    window_info_vmem,
    window_grad_x_vmem,
    x_dma_sem,
    info_dma_sem,
    writeback_dma_sem,
    preload_dma_sem,
    written_rows_smem,
    preloaded_rows_smem,
    *,
    num_groups,
    buf_rows,
    hidden_dim,
    split_weights,
    align,
):
  """Combine backward kernel: computes `grad_weights_ref` and scatters `grad_x` windows to `grad_x_hbm`."""
  block_tokens = grad_out_ref.shape[0]
  streams = (
      _WindowReadStream(sorted_x_hbm, window_x_vmem, x_dma_sem, init_fill=None),
      _WindowReadStream(row_info_hbm, window_info_vmem, info_dma_sem, init_fill=_INFO_FILL),
  )
  slot = _prefetch_expert_windows(cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align)
  scatter = _ScatterWindowState(
      grad_x_hbm,
      window_grad_x_vmem,
      writeback_dma_sem,
      preload_dma_sem,
      written_rows_smem,
      preloaded_rows_smem,
      num_groups,
      align,
  )
  _prepare_scatter_output_slot(scatter, cur_dma_table_ref, slot)
  packed_rows = _num_packed_rows(cur_dma_table_ref, num_groups)
  block_start_token = pl.program_id(0) * block_tokens
  _bwd_compute(
      weights_ref,
      grad_out_ref,
      grad_weights_ref,
      window_x_vmem,
      window_info_vmem,
      window_grad_x_vmem,
      slot,
      packed_rows,
      block_start_token,
      block_tokens=block_tokens,
      hidden_dim=hidden_dim,
      buf_rows=buf_rows,
      split_weights=split_weights,
  )
  _write_back_windows_and_preload_next(scatter, cur_dma_table_ref, next_dma_table_ref, slot)


def _bwd_call(sorted_x, padded_weights, dma_table, row_info, grad_out, cfg: _Config):
  """Launches `_bwd_kernel` to compute `(padded_grad_weights [T, 128] f32, grad_x [R, D])`."""
  num_rows, hidden_dim = sorted_x.shape
  num_tokens = padded_weights.shape[0]
  block_tokens, num_groups, align = cfg.block_tokens, cfg.num_groups, cfg.align
  num_blocks = num_tokens // block_tokens
  buf_rows = _buffer_rows(block_tokens, cfg.k, num_groups, align)
  in_specs = _dma_table_block_specs(num_blocks, num_groups) + [
      pl.BlockSpec((block_tokens, _LANES), lambda b: (b, 0)),
      pl.BlockSpec((block_tokens, hidden_dim), lambda b: (b, 0)),
      pl.BlockSpec(memory_space=pl.ANY),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  itemsize = jnp.dtype(sorted_x.dtype).itemsize
  vmem_bytes = (
      4 * buf_rows * hidden_dim * itemsize  # window_x_vmem + window_grad_x_vmem
      + 2 * buf_rows * _LANES * itemsize  # window_info_vmem
      + 2 * block_tokens * hidden_dim * itemsize  # grad_out_ref blocks
      + 4 * block_tokens * _LANES * 4  # weights_ref / grad_weights_ref blocks
      + 24 * 1024 * 1024  # chunk temporaries, compiler scratch
  )
  axis_type = jax.typeof(sorted_x).manual_axis_type
  return pl.pallas_call(
      functools.partial(
          _bwd_kernel,
          num_groups=num_groups,
          buf_rows=buf_rows,
          hidden_dim=hidden_dim,
          split_weights=cfg.split_weights,
          align=align,
      ),
      out_shape=(
          jax.ShapeDtypeStruct((num_tokens, _LANES), jnp.float32, manual_axis_type=axis_type),
          jax.ShapeDtypeStruct((num_rows, hidden_dim), sorted_x.dtype, manual_axis_type=axis_type),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_blocks,),
          in_specs=in_specs,
          out_specs=[pl.BlockSpec((block_tokens, _LANES), lambda b: (b, 0)), pl.BlockSpec(memory_space=pl.ANY)],
          scratch_shapes=[
              pltpu.VMEM((2, buf_rows, hidden_dim), sorted_x.dtype),  # window_x_vmem
              pltpu.VMEM((2, buf_rows, _LANES), row_info.dtype),  # window_info_vmem
              pltpu.VMEM((2, buf_rows, hidden_dim), sorted_x.dtype),  # window_grad_x_vmem
              pltpu.SemaphoreType.DMA((2,)),  # x reads
              pltpu.SemaphoreType.DMA((2,)),  # row-info reads
              pltpu.SemaphoreType.DMA((2,)),  # window writes
              pltpu.SemaphoreType.DMA((2,)),  # partial-granule preloads
              pltpu.SMEM((2,), jnp.int32),  # written_rows_smem
              pltpu.SMEM((2,), jnp.int32),  # preloaded_rows_smem
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",), vmem_limit_bytes=min(vmem_bytes, _VMEM_LIMIT)
      ),
      interpret=cfg.interpret,
      name="moe_combine_bwd",
  )(dma_table, dma_table, padded_weights, grad_out, sorted_x, row_info)


# ---------------------------------------------------------------------------
# permute forward kernel (unit-weight scatter)
# ---------------------------------------------------------------------------


def _permute_fwd_kernel(
    cur_dma_table_ref,
    next_dma_table_ref,
    tokens_x_ref,
    row_info_hbm,
    sorted_out_hbm,
    window_info_vmem,
    window_out_vmem,
    info_dma_sem,
    writeback_dma_sem,
    preload_dma_sem,
    written_rows_smem,
    preloaded_rows_smem,
    *,
    num_groups,
    buf_rows,
    hidden_dim,
    align,
):
  """Scatters token block `tokens_x_ref` `[block_tokens, hidden_dim]` into `sorted_out_hbm` `[R, hidden_dim]`."""
  block_tokens = tokens_x_ref.shape[0]
  streams = (_WindowReadStream(row_info_hbm, window_info_vmem, info_dma_sem, init_fill=_INFO_FILL),)
  slot = _prefetch_expert_windows(cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align)
  scatter = _ScatterWindowState(
      sorted_out_hbm,
      window_out_vmem,
      writeback_dma_sem,
      preload_dma_sem,
      written_rows_smem,
      preloaded_rows_smem,
      num_groups,
      align,
  )
  _prepare_scatter_output_slot(scatter, cur_dma_table_ref, slot)

  packed_rows = _num_packed_rows(cur_dma_table_ref, num_groups)
  col_chunk = _COL_CHUNK if hidden_dim % _COL_CHUNK == 0 else _LANES
  chunk_rows = min(_SCATTER_ROWS_PER_CHUNK, buf_rows)
  num_chunks = (packed_rows + (chunk_rows - 1)) // chunk_rows
  block_start_token = pl.program_id(0) * block_tokens
  token_col_iota = lax.broadcasted_iota(jnp.int32, (chunk_rows, block_tokens), 1)

  # Build 0/1 selection matrix `token_select_chunk[j, c] = (token(j) == c)` and multiply by `tokens_x_ref`.
  def process_row_chunk(chunk_idx, carry):
    row_offset = pl.multiple_of(jnp.minimum(chunk_idx * chunk_rows, buf_rows - chunk_rows), _ROW_CHUNK)
    buf_rows_slice = pl.ds(row_offset, chunk_rows)
    row_token_in_block, _ = _decode_row_info(
        window_info_vmem, slot, row_offset, chunk_rows, block_tokens, block_start_token, packed_rows
    )
    token_select_chunk = (token_col_iota == row_token_in_block).astype(tokens_x_ref.dtype)  # [chunk_rows, block_tokens]
    belongs_to_current_block = row_token_in_block >= 0

    def scatter_col_chunk(col_idx, carry2):
      hidden_cols = pl.ds(pl.multiple_of(col_idx * col_chunk, col_chunk), col_chunk)
      selected_rows = jnp.dot(token_select_chunk, tokens_x_ref[:, hidden_cols], preferred_element_type=jnp.float32)
      window_out_vmem[slot, buf_rows_slice, hidden_cols] = jnp.where(
          belongs_to_current_block,
          selected_rows.astype(window_out_vmem.dtype),
          window_out_vmem[slot, buf_rows_slice, hidden_cols],
      )
      return carry2

    lax.fori_loop(0, hidden_dim // col_chunk, scatter_col_chunk, 0)
    return carry

  lax.fori_loop(0, num_chunks, process_row_chunk, 0)
  _write_back_windows_and_preload_next(scatter, cur_dma_table_ref, next_dma_table_ref, slot)


def _permute_fwd_call(x, dma_table, row_info, cfg: _Config):
  """Launches `_permute_fwd_kernel` to produce `[T*K, D]` expert-sorted rows from `x` `[T, D]`."""
  num_tokens, hidden_dim = x.shape
  block_tokens, num_groups, align = cfg.block_tokens, cfg.num_groups, cfg.align
  num_rows = num_tokens * cfg.k
  num_blocks = num_tokens // block_tokens
  buf_rows = _buffer_rows(block_tokens, cfg.k, num_groups, align)
  in_specs = _dma_table_block_specs(num_blocks, num_groups) + [
      pl.BlockSpec((block_tokens, hidden_dim), lambda b: (b, 0)),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  itemsize = jnp.dtype(x.dtype).itemsize
  vmem_bytes = (
      2 * buf_rows * hidden_dim * itemsize  # window_out_vmem
      + 2 * buf_rows * _LANES * itemsize  # window_info_vmem
      + 2 * block_tokens * hidden_dim * itemsize  # tokens_x_ref blocks
      + 20 * 1024 * 1024  # chunk temporaries, compiler scratch
  )
  return pl.pallas_call(
      functools.partial(
          _permute_fwd_kernel, num_groups=num_groups, buf_rows=buf_rows, hidden_dim=hidden_dim, align=align
      ),
      out_shape=jax.ShapeDtypeStruct((num_rows, hidden_dim), x.dtype, manual_axis_type=jax.typeof(x).manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(num_blocks,),
          in_specs=in_specs,
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          scratch_shapes=[
              pltpu.VMEM((2, buf_rows, _LANES), row_info.dtype),  # window_info_vmem
              pltpu.VMEM((2, buf_rows, hidden_dim), x.dtype),  # window_out_vmem
              pltpu.SemaphoreType.DMA((2,)),  # row-info reads
              pltpu.SemaphoreType.DMA((2,)),  # window writes
              pltpu.SemaphoreType.DMA((2,)),  # partial-granule preloads
              pltpu.SMEM((2,), jnp.int32),  # written_rows_smem
              pltpu.SMEM((2,), jnp.int32),  # preloaded_rows_smem
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",), vmem_limit_bytes=min(vmem_bytes, _VMEM_LIMIT)
      ),
      interpret=cfg.interpret,
      name="moe_permute_fwd",
  )(dma_table, dma_table, x, row_info)


# ---------------------------------------------------------------------------
# custom VJPs and public entry points
# ---------------------------------------------------------------------------


def _resolve_config(
    rows_shape, dtype, num_tokens, k, group_sizes, *, block_tokens, interpret, align, split_weights, default_block_tokens
):
  """Returns `_Config`, or None if the kernel preconditions do not hold (`rows_shape`: the [T*K, D] array)."""
  if block_tokens is None:
    block_tokens = _block_tokens(num_tokens, default_block_tokens)
  if interpret is True:  # dynamic-size DMAs require the Mosaic TPU interpreter
    interpret = pltpu.InterpretParams()
  if group_sizes is None or not kernel_supported(rows_shape, dtype, num_tokens, k, block_tokens, align):
    return None
  return _Config(k, block_tokens, int(group_sizes.shape[0]), split_weights, interpret, align)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _combine_vjp(x, weights, sort_idx, dma_table, cfg):
  padded_weights = jnp.pad(weights.astype(jnp.float32), ((0, 0), (0, _LANES - cfg.k)))
  return _gather_call(
      x,
      padded_weights,
      dma_table,
      _build_row_info(sort_idx, cfg.k),
      num_tokens=weights.shape[0],
      cfg=cfg,
      name="moe_combine_fwd",
  )


def _combine_fwd(x, weights, sort_idx, dma_table, cfg):
  """Forward pass for `_combine_vjp`, saving inputs as residuals for backward."""
  return _combine_vjp(x, weights, sort_idx, dma_table, cfg), (x, weights, sort_idx, dma_table)


def _combine_bwd(cfg, residuals, grad_out):
  """Backward pass for `_combine_vjp`, computing `grad_x` and `grad_weights`."""
  x, weights, sort_idx, dma_table = residuals
  padded_weights = jnp.pad(weights.astype(jnp.float32), ((0, 0), (0, _LANES - cfg.k)))
  padded_grad_weights, grad_x = _bwd_call(
      x,
      padded_weights,
      dma_table,
      _build_row_info(sort_idx, cfg.k),
      grad_out.astype(x.dtype),
      cfg,
  )
  return grad_x, padded_grad_weights[:, : cfg.k].astype(weights.dtype), None, None


_combine_vjp.defvjp(_combine_fwd, _combine_bwd)


def combine(
    x: jax.Array,
    sort_idx: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array | None,
    *,
    block_tokens: int | None = None,
    interpret: bool | pltpu.InterpretParams = False,
    align: int = 8,
) -> jax.Array:
  """Combines expert-sorted rows `x` into token outputs `y[t] = sum_k weights[t, k] * x[argsort(sort_idx)[t*K + k]]`.

  Args:
    x: `[T*K, D]` expert outputs in expert-sorted order (bf16 for the kernel path).
    sort_idx: `[T*K]` int, stable argsort of the flat (token-major) expert ids.
    weights: `[T, K]` routing weights (any float dtype; non-bf16 weights use an exact hi + lo split).
    group_sizes: `[G]` int, rows per expert group of `x` (must sum to `T*K`), or None to fall back.
    block_tokens: tokens per grid step (default: 256 or the largest power-of-two divisor of `T >= 128`).
    interpret: run under the Mosaic TPU interpreter (off-TPU testing); True means `pltpu.InterpretParams()`.
    align: DMA / VMEM row alignment, a power of two >= 8.

  Returns:
    `[T, D]` array in `x.dtype` (f32 accumulation). Differentiable w.r.t. `x` and `weights`.
    Falls back to `combine_reference` when the kernel preconditions (see `kernel_supported`) do not hold.
  """
  num_tokens, k = weights.shape
  cfg = _resolve_config(
      x.shape,
      x.dtype,
      num_tokens,
      k,
      group_sizes,
      block_tokens=block_tokens,
      interpret=interpret,
      align=align,
      split_weights=jnp.dtype(weights.dtype) != jnp.bfloat16,
      default_block_tokens=_COMBINE_BLOCK_TOKENS,
  )
  if cfg is None:
    return combine_reference(x, sort_idx, weights)
  dma_table = _build_dma_table(sort_idx, group_sizes, num_tokens, k, cfg.block_tokens, align)
  return _combine_vjp(x, weights, sort_idx, dma_table, cfg)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _permute_vjp(x, sort_idx, dma_table, cfg):
  return _permute_fwd_call(x, dma_table, _build_row_info(sort_idx, cfg.k), cfg)


def _permute_fwd(x, sort_idx, dma_table, cfg):
  """Forward pass for `_permute_vjp`, saving `sort_idx` and `dma_table` for backward."""
  return _permute_vjp(x, sort_idx, dma_table, cfg), (sort_idx, dma_table)


def _permute_bwd(cfg, residuals, grad_out):
  """Backward pass for `_permute_vjp`, summing the `K` expert gradients per token via `_gather_call`."""
  sort_idx, dma_table = residuals
  num_tokens = sort_idx.shape[0] // cfg.k
  grad_x = _gather_call(
      grad_out,
      None,
      dma_table,
      _build_row_info(sort_idx, cfg.k),
      num_tokens=num_tokens,
      cfg=cfg,
      name="moe_permute_bwd",
  )
  return grad_x, None, None


_permute_vjp.defvjp(_permute_fwd, _permute_bwd)


def permute(
    x: jax.Array,
    sort_idx: jax.Array,
    group_sizes: jax.Array | None,
    num_experts_per_tok: int,
    *,
    block_tokens: int | None = None,
    interpret: bool | pltpu.InterpretParams = False,
    align: int = 8,
) -> jax.Array | None:
  """Permutes token activations `x` `[T, D]` into expert-sorted order `[T*K, D]` on the TensorCore.

  Forward: `y[r] = x[sort_idx[r] // K]` built with the same block / row-window DMA
  scheme as `combine` but with unit weights (one-hot selection on the MXU).
  Backward: `dx[t] = sum` over the `K` rows of `t` in `dy`, i.e. the combine forward
  with unit weights.

  Intended configuration: `moe_permute_kernel="tc"` together with the XLA flags
  `--xla_tpu_offload_gather_to_sparsecore=false`
  `--xla_tpu_offload_all_supported_gathers_to_sparsecore=false`, so that neither
  the permute nor the remaining small gathers are serialized on SparseCore
  behind the expert-weight all-gathers.

  Args:
    x: `[T, D]` bf16 token activations.
    sort_idx: `[T*K]` int, stable argsort of the flat (token-major) expert ids.
    group_sizes: `[G]` int, rows per expert group (must sum to `T*K`), or None to return None.
    num_experts_per_tok: `K`.
    block_tokens: tokens per grid step (default: 512 or the largest power-of-two divisor of `T >= 128`).
    interpret: run under the Mosaic TPU interpreter (off-TPU testing); True means `pltpu.InterpretParams()`.
    align: DMA / VMEM row alignment, a power of two >= 8.

  Returns:
    `[T*K, D]` array in `x.dtype`, differentiable w.r.t. `x`; or None when the kernel
    preconditions (see `kernel_supported`) do not hold so the caller can fall back.
  """
  num_tokens, hidden_dim = x.shape
  k = int(num_experts_per_tok)
  cfg = _resolve_config(
      (num_tokens * k, hidden_dim),
      x.dtype,
      num_tokens,
      k,
      group_sizes,
      block_tokens=block_tokens,
      interpret=interpret,
      align=align,
      split_weights=False,
      default_block_tokens=_PERMUTE_BLOCK_TOKENS,
  )
  if cfg is None:
    return None
  dma_table = _build_dma_table(sort_idx, group_sizes, num_tokens, k, cfg.block_tokens, align)
  return _permute_vjp(x, sort_idx, dma_table, cfg)
