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

"""Experimental fused TensorCore (Pallas) MoE `combine` and `permute` kernels.

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
  * `_build_row_info(sort_idx, k)` packs each sorted row's token id (in base
    256) and `topk_slot` elementwise from `sort_idx` into a `[R, 128]` bf16
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

Expert Chunks & Column Chunks (large `G` / large `D`)
-----------------------------------------------------
The VMEM window buffers hold `C * K + (2A - 2) * G` rows of `D` columns. By
default, each kernel runs in a single pass (`expert_chunks = col_chunks = 1`),
which gives the best speedup. For large `G` or `D` that exceed `vmem_limit_bytes`
in a single pass, callers can configure `expert_chunks` and `col_chunks`:

  * **Expert chunks** (`expert_chunks`): each token block is processed in chunks
    of `ceil(G / expert_chunks)` experts, shrinking the alignment-padding term.
    Rows carry their expert chunk in `row_info` so a boundary granule shared by
    two chunks is only used by its owner. Gather kernels iterate
    `(col_chunk, block, expert_chunk)` and accumulate the token block in f32
    VMEM; scatter kernels iterate `(col_chunk, expert_chunk, block)`, i.e. one
    full pass over the token blocks per chunk, and boundary granules holding
    rows of an earlier chunk are preloaded from HBM.
  * **Column chunks** (`col_chunks`): each pass only DMAs and computes
    `D / col_chunks` hidden columns (a multiple of 128) of the HBM arrays.

`combine` backward writes one partial `grad_weights` per pass, summed in XLA.
If the configured `(block_tokens, expert_chunks, col_chunks)` still exceeds
`vmem_limit_bytes` (or preconditions are not met), `combine` and `permute` fail
fast by raising `ValueError`.

Padding
-------
The public entry points pad `T` up to a multiple of `block_tokens` with tokens
that own no rows (only the small `[T, D]` / `[T, K]` arrays are padded) and
zero-pad `D` up to a multiple of 128 (one extra copy of the `[T * K, D]` array).
Token ids above 65,535 use a third `row_info` lane.
"""

import functools
from typing import Any, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

__all__ = [
    "DEFAULT_COMBINE_BLOCK_TOKENS",
    "DEFAULT_PERMUTE_BLOCK_TOKENS",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "combine",
    "combine_reference",
    "default_vmem_limit_bytes",
    "kernel_supported",
    "permute",
]

_LANES = 128
DEFAULT_COMBINE_BLOCK_TOKENS = 256  # combine: default tokens per grid step
DEFAULT_PERMUTE_BLOCK_TOKENS = 512  # permute: default tokens per grid step
# Default per-kernel VMEM budget: this fraction of the chip's VMEM (`pltpu.get_tpu_info().vmem_capacity_bytes`, e.g.
# 128 MiB on v5e / v6e, 64 MiB on v5p / TPU7x), like the grouped-matmul kernels in kernels/megablox (which use 90%;
# 95% keeps e.g. D=4096, K=8, 128 experts, which needs 116 MiB on v6e, in a single pass).
_VMEM_CAPACITY_FRACTION = 0.95
DEFAULT_VMEM_LIMIT_BYTES = 120 * 1024 * 1024  # budget when the chip is unknown (off-TPU interpret mode)
# Tile VPU masks and MXU dots into 256x256 blocks so f32 accumulators stay in VREGs without spilling to VMEM.
_ROW_CHUNK = 256
_COL_CHUNK = 256
# Dynamic loop bounds over the MXU contraction axis prevent Mosaic from pipelining matmuls, while contracting
# all `buf_rows` wastes MXU work when windows are smaller; static buckets of 512 give both pipelining and pruning.
_FWD_BUCKET_ROWS = 512
# Cap scatter-direction VMEM temporaries (`[chunk_rows, block_tokens]` f32 dot products) to <= 2 MiB per chunk.
_SCATTER_ROWS_PER_CHUNK = 1024

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

# Unused VMEM buffer rows are initialized to -1.0 in all lanes so their decoded token id and top-K slot are negative
# and never match any valid token `[0, block_tokens)` or top-K slot `[0, K)`.
_INFO_FILL = -1.0
_MAX_EXPERT_CHUNKS = 256  # integers in `[0, 256]` are exact in bf16
# Row-info lanes: token id in base 256 (lo, mid, hi), top-k slot and expert chunk.
_INFO_TOKEN_LO = 0
_INFO_TOKEN_MID = 1
_INFO_TOPK = 2
_INFO_TOKEN_HI = 3  # only decoded when num_tokens > _MAX_NARROW_TOKENS
_INFO_EXPERT_CHUNK = 4  # only decoded when expert_chunks > 1
_NUM_INFO_LANES = 5
_MAX_NARROW_TOKENS = 256 * 256  # token ids below this fit in the (lo, mid) lanes
_MAX_TOKENS = 256 * 256 * 256
_MAX_HISTOGRAM_ONEHOT_BYTES = 256 * 1024 * 1024  # one-hot temporaries of the (block, expert) histogram


class _Config(NamedTuple):
  """Static kernel configuration (hashable: the non-differentiable custom_vjp argument)."""

  k: int  # experts per token
  block_tokens: int  # tokens per grid step
  num_groups: int  # number of experts G
  split_weights: bool  # True when routing weights are f32 (split into bf16 hi + lo parts)
  interpret: Any  # False or pltpu.InterpretParams
  align: int  # DMA / VMEM row alignment A
  vmem_limit_bytes: int  # max per-kernel VMEM budget in bytes
  num_tokens: int  # tokens seen by the kernels (padded up to a multiple of block_tokens)
  expert_chunks: int = 1  # each token block is processed in this many contiguous chunks of experts
  col_chunks: int = 1  # the hidden dimension is processed in this many contiguous column chunks

  @property
  def experts_per_chunk(self) -> int:
    return -(-self.num_groups // self.expert_chunks)

  @property
  def wide_tokens(self) -> bool:
    return self.num_tokens > _MAX_NARROW_TOKENS


# ---------------------------------------------------------------------------
# shapes and preconditions
# ---------------------------------------------------------------------------


def default_vmem_limit_bytes() -> int:
  """Returns the per-kernel VMEM budget for the active TPU generation, or `DEFAULT_VMEM_LIMIT_BYTES` off-TPU.

  Reserves `1 - _VMEM_CAPACITY_FRACTION` (5%) of hardware VMEM for compiler-allocated spill slots and scalar/vector
  register live ranges that are not part of the explicit Pallas `scratch_shapes`.
  """
  try:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_CAPACITY_FRACTION)
  except (ValueError, NotImplementedError):  # not a TPU device (e.g. interpret mode on CPU)
    return DEFAULT_VMEM_LIMIT_BYTES


def _round_up(x: int, multiple: int) -> int:
  return (x + multiple - 1) // multiple * multiple


def _dma_table_stride(num_groups: int) -> int:
  """Returns the padded SMEM block length for one step's DMA table.

  Mosaic TPU requires rank-1 SMEM `BlockSpec` shapes to be a power of two >= 128 (up to 1024) or a multiple of 1024.
  """
  min_len = _NUM_DMA_TABLE_FIELDS * num_groups
  if min_len > 1024:
    return _round_up(min_len, 1024)
  stride = _LANES
  while stride < min_len:
    stride *= 2
  return stride


def _buffer_rows(block_tokens: int, k: int, num_groups: int, align: int = 8) -> int:
  """Returns the worst-case VMEM window-buffer height required for one token block.

  A token block owns `block_tokens * k` routed rows distributed across at most `min(num_groups, block_tokens * k)`
  non-empty experts. Widening each expert's `[start, end)` HBM slice outward to `align`-row DMA boundaries can add
  at most `align - 1` rows at the head and `align - 1` rows at the tail (`2 * align - 2` rows per expert). Rounding
  up to `_ROW_CHUNK` (256) allows fixed-size 256-row VPU/MXU loops without tail bounds checks.
  """
  block_rows = block_tokens * k
  return _round_up(block_rows + (2 * align - 2) * min(num_groups, block_rows), _ROW_CHUNK)


def _gather_vmem_bytes(
    buf_rows: int,
    block_tokens: int,
    hidden_dim: int,
    itemsize: int,
    *,
    weighted: bool,
    split_weights: bool,
    accumulate: bool = False,
) -> int:
  """Estimated VMEM footprint in bytes for `_gather_call` (`accumulate`: f32 accumulator over expert chunks)."""
  num_parts = 2 if (weighted and split_weights) else 1
  return (
      2 * buf_rows * (hidden_dim + _LANES) * itemsize  # window_x_vmem + window_info_vmem
      + 2 * block_tokens * hidden_dim * itemsize  # out_ref double-buffered blocks
      + num_parts * block_tokens * buf_rows * itemsize  # routing_matrix_vmem
      + (2 * block_tokens * _LANES * 4 if weighted else 0)  # weights_ref blocks
      + (block_tokens * hidden_dim * 4 if accumulate else 0)  # acc_vmem
      + 16 * 1024 * 1024  # dot operands / temporaries, compiler scratch
  )


def _combine_bwd_vmem_bytes(buf_rows: int, block_tokens: int, hidden_dim: int, itemsize: int) -> int:
  """Estimated VMEM footprint in bytes for `_bwd_call`."""
  return (
      4 * buf_rows * hidden_dim * itemsize  # window_x_vmem + window_grad_x_vmem
      + 2 * buf_rows * _LANES * itemsize  # window_info_vmem
      + 2 * block_tokens * hidden_dim * itemsize  # grad_out_ref blocks
      + 4 * block_tokens * _LANES * 4  # weights_ref / grad_weights_ref blocks
      + 24 * 1024 * 1024  # chunk temporaries, compiler scratch
  )


def _permute_fwd_vmem_bytes(buf_rows: int, block_tokens: int, hidden_dim: int, itemsize: int) -> int:
  """Estimated VMEM footprint in bytes for `_permute_fwd_call`."""
  return (
      2 * buf_rows * hidden_dim * itemsize  # window_out_vmem
      + 2 * buf_rows * _LANES * itemsize  # window_info_vmem
      + 2 * block_tokens * hidden_dim * itemsize  # tokens_x_ref blocks
      + 20 * 1024 * 1024  # chunk temporaries, compiler scratch
  )


def _required_vmem_bytes(
    block_tokens: int,
    k: int,
    num_groups: int,
    hidden_dim: int,
    itemsize: int,
    *,
    split_weights: bool = False,
    align: int = 8,
    is_combine: bool = True,
    accumulate: bool = False,
) -> int:
  """Returns the peak VMEM bytes required across forward and backward passes.

  `num_groups` and `hidden_dim` are the experts and hidden columns handled per grid step, i.e. per expert chunk
  and per column chunk; `accumulate` adds the f32 accumulator the gather kernels need with several expert chunks.
  """
  buf_rows = _buffer_rows(block_tokens, k, num_groups, align)
  if is_combine:
    fwd_bytes = _gather_vmem_bytes(
        buf_rows,
        block_tokens,
        hidden_dim,
        itemsize,
        weighted=True,
        split_weights=split_weights,
        accumulate=accumulate,
    )
    bwd_bytes = _combine_bwd_vmem_bytes(buf_rows, block_tokens, hidden_dim, itemsize)
    return max(fwd_bytes, bwd_bytes)
  fwd_bytes = _permute_fwd_vmem_bytes(buf_rows, block_tokens, hidden_dim, itemsize)
  bwd_bytes = _gather_vmem_bytes(
      buf_rows, block_tokens, hidden_dim, itemsize, weighted=False, split_weights=False, accumulate=accumulate
  )
  return max(fwd_bytes, bwd_bytes)


def kernel_supported(
    x_shape: tuple[int, ...],
    x_dtype: jax.typing.DTypeLike,
    num_tokens: int,
    k: int,
    block_tokens: int,
    align: int = 8,
    *,
    num_groups: int | None = None,
    split_weights: bool = False,
    is_combine: bool = True,
    vmem_limit_bytes: int | None = None,
    expert_chunks: int = 1,
    col_chunks: int = 1,
) -> bool:
  """Returns whether the Pallas kernels support this problem shape, dtype, and VMEM budget.

  Shapes are the ones the kernels see, after the padding done by the public entry points: `x_shape` is the
  `[rows, D]` expert-sorted array with `D` a multiple of 128, and `num_tokens` is a multiple of `block_tokens`
  with `rows <= num_tokens * k` (padding tokens own no rows).
  """
  if len(x_shape) != 2:
    return False
  if vmem_limit_bytes is None:
    vmem_limit_bytes = default_vmem_limit_bytes()
  num_rows, hidden_dim = x_shape
  dtype = jnp.dtype(x_dtype)
  shape_ok = (
      dtype == jnp.bfloat16
      and hidden_dim > 0
      and hidden_dim % _LANES == 0
      and 0 < k <= _LANES
      and align >= 8
      and (align & (align - 1)) == 0
      and block_tokens >= _LANES
      and (block_tokens & (block_tokens - 1)) == 0
      and num_tokens >= block_tokens
      and num_tokens % block_tokens == 0
      and num_tokens <= _MAX_TOKENS
      and 0 < num_rows <= num_tokens * k
      and num_rows % align == 0
      and expert_chunks >= 1
      and col_chunks >= 1
      and hidden_dim % (col_chunks * _LANES) == 0
  )
  if not shape_ok:
    return False
  if num_groups is None:
    return expert_chunks == 1
  if num_groups <= 0 or expert_chunks > min(num_groups, _MAX_EXPERT_CHUNKS):
    return False
  vmem_bytes = _required_vmem_bytes(
      block_tokens,
      k,
      -(-num_groups // expert_chunks),
      hidden_dim // col_chunks,
      dtype.itemsize,
      split_weights=split_weights,
      align=align,
      is_combine=is_combine,
      accumulate=expert_chunks > 1,
  )
  return vmem_bytes <= vmem_limit_bytes


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


def _block_expert_histogram(row_block, group_starts, group_ends, num_blocks):
  """Computes the `[num_blocks, num_groups]` row count per `(token_block, expert)` on the MXU without scatters.

  Computing a 2D histogram via `jnp.bincount` or `scatter_add` lowers to SparseCore scatters or O(R log R) sorts,
  which contend with FSDP weight all-gathers on SparseCore. Instead, we contract binary `{0, 1}` bf16 indicator
  matrices on the MXU with f32 accumulation, which is exact for counts `< 2^24` and runs entirely on TensorCore.
  """
  num_rows = row_block.shape[0]
  num_groups = group_starts.shape[0]
  row_idx = jnp.arange(num_rows, dtype=jnp.int32)

  def histogram(row_block, row_idx):
    row_idx = row_idx[:, None]
    in_group = ((row_idx >= group_starts[None, :]) & (row_idx < group_ends[None, :])).astype(jnp.bfloat16)
    in_block = (row_block[:, None] == jnp.arange(num_blocks, dtype=jnp.int32)[None, :]).astype(jnp.bfloat16)
    return jnp.einsum("rb,rg->bg", in_block, in_group, preferred_element_type=jnp.float32).astype(jnp.int32)

  # Bound the temporary `[R_chunk, num_blocks]` and `[R_chunk, num_groups]` one-hot HBM tensors on large shapes.
  num_row_chunks = 1
  while (
      num_rows // num_row_chunks * (num_blocks + num_groups) * 2 > _MAX_HISTOGRAM_ONEHOT_BYTES
      and num_rows % (2 * num_row_chunks) == 0
  ):
    num_row_chunks *= 2
  if num_row_chunks == 1:
    return histogram(row_block, row_idx)
  counts = lax.map(
      lambda chunk: histogram(*chunk),
      (row_block.reshape(num_row_chunks, -1), row_idx.reshape(num_row_chunks, -1)),
  )
  return jnp.sum(counts, axis=0)


def _build_dma_table(sort_idx, group_sizes, num_tokens, k, block_tokens, align=8, expert_chunks=1):
  """Builds the flat `int32 [expert_chunks * num_blocks * stride]` SMEM DMA schedule table.

  Because `jnp.argsort` is stable, rows within each expert `g` are ordered monotonically by token id, so token block
  `b` occupies a single contiguous slice `[range_start[b, g], range_end[b, g])`. Widening each slice outward to
  `align`-row DMA boundaries can cause adjacent small experts to overlap on a shared boundary granule. We coalesce
  overlapping windows so that every HBM row in a `(block, expert_chunk)` step is DMA'd into VMEM at most once
  (preventing duplicate-row double-counting in gather matmuls and racing writebacks in scatter kernels), and so that
  `vmem_row - hbm_row` (`_WIN_ROW_DELTA`) stays constant across each coalesced segment.
  """
  num_groups = group_sizes.shape[0]
  experts_per_chunk = -(-num_groups // expert_chunks)
  padded_groups = expert_chunks * experts_per_chunk
  num_blocks = num_tokens // block_tokens

  def per_chunk(values):
    return values.reshape(num_blocks, expert_chunks, experts_per_chunk)

  # 1. Contiguous row range [range_start, range_end) in `x` for each (block, expert) pair.
  # Experts are padded with empty ones up to a whole number of chunks.
  group_sizes = jnp.pad(group_sizes.astype(jnp.int32), (0, padded_groups - num_groups))
  group_ends = jnp.cumsum(group_sizes)
  group_starts = group_ends - group_sizes
  row_block = sort_idx.astype(jnp.int32) // (k * block_tokens)
  rows_per_block_and_expert = _block_expert_histogram(row_block, group_starts, group_ends, num_blocks)
  flat_range_start = (
      group_starts[None, :] + jnp.cumsum(rows_per_block_and_expert, axis=0) - rows_per_block_and_expert
  )  # [num_blocks, padded_groups]
  range_start = per_chunk(flat_range_start)
  range_end = range_start + per_chunk(rows_per_block_and_expert)
  nonempty = range_end > range_start

  # 2. Aligned DMA windows [aligned_win_start, aligned_win_end) and coalescing within a (block, expert chunk).
  # Consecutive expert windows in the same block are non-decreasing and can share a boundary alignment granule;
  # clamping `dma_hbm_start` against `prev_covered_end` ensures each expert only DMAs the uncovered suffix, while
  # `prev_covered_end` and `dma_vmem_offset` advance in lockstep so `hbm_to_vmem_row_delta` stays constant.
  aligned_win_start = jnp.bitwise_and(range_start, -align)
  aligned_win_end = jnp.bitwise_and(range_end + (align - 1), -align)
  active_win_end = jnp.where(nonempty, aligned_win_end, -1)
  prev_covered_end = jnp.pad(
      lax.cummax(active_win_end, axis=2)[:, :, :-1], ((0, 0), (0, 0), (1, 0)), constant_values=-1
  )
  dma_hbm_start = jnp.maximum(aligned_win_start, prev_covered_end)
  dma_num_rows = jnp.where(nonempty, jnp.maximum(aligned_win_end - dma_hbm_start, 0), 0)
  dma_vmem_offset = jnp.cumsum(dma_num_rows, axis=2) - dma_num_rows
  hbm_to_vmem_row_delta = dma_vmem_offset - dma_hbm_start  # vmem_row = hbm_row + delta

  # 3. Scatter staging for partial boundary granules.
  # Because scatter kernels write whole `align`-row granules back to HBM, any boundary granule that also holds rows
  # written by an *earlier* step (`chunk_of_group < this_chunk`, or same chunk and earlier token block) must be
  # preloaded into VMEM before compute so those earlier rows are preserved during writeback. Granules shared only
  # with *later* steps do not need preloading here because those later steps will preload and preserve our rows.
  head_partial_granule_hbm = jnp.where(nonempty & (jnp.bitwise_and(range_start, align - 1) != 0), aligned_win_start, -1)
  tail_partial_granule_hbm = jnp.where(
      nonempty & (jnp.bitwise_and(range_end, align - 1) != 0), aligned_win_end - align, -1
  )
  chunk_of_group = (jnp.arange(padded_groups, dtype=jnp.int32) // experts_per_chunk)[None, None, :]
  this_chunk = jnp.arange(expert_chunks, dtype=jnp.int32)[None, :, None]
  earlier_rows_end = jnp.where(
      chunk_of_group < this_chunk,
      group_ends[None, None, :],
      jnp.where(chunk_of_group == this_chunk, flat_range_start[:, None, :], group_starts[None, None, :]),
  )  # [num_blocks, expert_chunks, padded_groups]

  def has_earlier_rows(granule_hbm_start):
    granule = granule_hbm_start[..., None]
    overlap = jnp.maximum(granule, group_starts[None, None, None, :]) < jnp.minimum(
        granule + align, earlier_rows_end[:, :, None, :]
    )
    return (granule_hbm_start >= 0) & jnp.any(overlap, axis=-1)

  # Deduplicate boundary granules shared by >= 2 experts in the same step so we never issue duplicate async DMAs
  # targeting the same VMEM slice (which would race on the preload semaphore).
  valid_heads = jnp.where(nonempty, head_partial_granule_hbm, -2)
  same_head_as_earlier = jnp.any(
      jnp.tril(valid_heads[..., :, None] == valid_heads[..., None, :], k=-1),
      axis=-1,
  )
  shared_with_head = jnp.any(tail_partial_granule_hbm[..., :, None] == valid_heads[..., None, :], axis=-1)
  need_head = has_earlier_rows(head_partial_granule_hbm) & ~same_head_as_earlier
  need_tail = has_earlier_rows(tail_partial_granule_hbm) & ~shared_with_head

  # If the needed granule was written in the immediately preceding block `b - 1` of the same expert chunk, its HBM
  # writeback may still be in flight while the data is already sitting in `window_out_vmem[1 - slot]`; copy it
  # VMEM-to-VMEM (`_HEAD_CARRY_*` / `_TAIL_CARRY_*`) instead of round-tripping through HBM.
  def of_prev_block(values, fill):
    return jnp.pad(values[:-1], ((1, 0), (0, 0), (0, 0)), constant_values=fill)

  prev_dma_start = of_prev_block(jnp.where(dma_num_rows > 0, dma_hbm_start, 1 << 30), 1 << 30)
  prev_dma_end = of_prev_block(jnp.where(dma_num_rows > 0, dma_hbm_start + dma_num_rows, -1), -1)
  prev_row_delta = of_prev_block(hbm_to_vmem_row_delta, 0)

  def find_in_prev_block(granule_hbm_start):
    granule = granule_hbm_start[..., None]
    hit = (granule >= prev_dma_start[:, :, None, :]) & (granule < prev_dma_end[:, :, None, :])
    in_prev = jnp.any(hit, axis=-1) & (granule_hbm_start >= 0)
    prev_vmem_row = jnp.sum(jnp.where(hit, granule + prev_row_delta[:, :, None, :], 0), axis=-1)
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
  table_width = _NUM_DMA_TABLE_FIELDS * experts_per_chunk
  # [num_blocks, expert_chunks, fields, experts_per_chunk] -> [expert_chunks * num_blocks, table_width]
  dma_table = jnp.stack(table_fields, axis=2).astype(jnp.int32).transpose(1, 0, 2, 3)
  dma_table = dma_table.reshape(expert_chunks * num_blocks, table_width)
  dma_table = jnp.pad(dma_table, ((0, 0), (0, _dma_table_stride(experts_per_chunk) - table_width)))
  return dma_table.reshape(-1)


def _build_row_info(sort_idx, k, group_sizes=None, expert_chunks=1):
  """Encodes per-row `(token_id, topk_slot, expert_chunk)` metadata into a `[R, 128]` bf16 tensor.

  Because Mosaic TPU requires 2D VMEM buffers to have a minor dimension of `_LANES` (128), we store each metadata
  field as a base-256 digit (`< 256`, exact in `bf16`) in lanes `0..4`. This halves the DMA bandwidth and VMEM
  footprint compared to `int32` and lets the kernel DMA `row_info` using the exact same row windows as `sorted_x` so
  `window_info_vmem[j]` always describes `window_x_vmem[j]`.
  """
  flat_token_topk = sort_idx.astype(jnp.int32)
  token_id = flat_token_topk // k
  topk_slot = flat_token_topk % k
  expert_chunk = jnp.zeros_like(token_id)
  if expert_chunks > 1:
    num_groups = group_sizes.shape[0]
    experts_per_chunk = -(-num_groups // expert_chunks)
    group_ends = jnp.cumsum(jnp.pad(group_sizes.astype(jnp.int32), (0, expert_chunks * experts_per_chunk - num_groups)))
    chunk_starts = group_ends[experts_per_chunk - 1 : (expert_chunks - 1) * experts_per_chunk : experts_per_chunk]
    row_idx = jnp.arange(flat_token_topk.shape[0], dtype=jnp.int32)
    expert_chunk = jnp.sum(row_idx[:, None] >= chunk_starts[None, :], axis=1, dtype=jnp.int32)
  info_cols = [token_id % 256, token_id // 256 % 256, topk_slot, token_id // 65536, expert_chunk]
  info_cols = jnp.stack(info_cols, axis=1).astype(jnp.bfloat16)
  return jnp.pad(info_cols, ((0, 0), (0, _LANES - _NUM_INFO_LANES)))


# ---------------------------------------------------------------------------
# in-kernel building blocks: window DMAs
# ---------------------------------------------------------------------------


def _wait_dma_rows(vmem_buf, slot, dma_sem, num_rows, align):
  """Waits until `num_rows` (multiple of `align`, may be 0) rows of DMAs on `dma_sem` have completed.

  Pallas TPU semaphore waits require a descriptor whose byte count matches the total bytes started on `dma_sem`
  across all expert windows in the step; constructing a single `num_rows`-row descriptor on `vmem_buf[slot]` waits
  for all of them in one call without issuing a transfer.
  """

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


def _hbm_col_chunk(col_chunk_idx, chunk_cols, col_chunks):
  """Hidden-column slice of the HBM arrays for column chunk `col_chunk_idx`; None when there is a single chunk."""
  if col_chunks == 1:
    return None
  return pl.ds(pl.multiple_of(col_chunk_idx * chunk_cols, _LANES), chunk_cols)


def _hbm_window(hbm_ref, rows, cols):
  """Rows `rows` of `hbm_ref`, restricted to the hidden columns `cols` unless `cols` is None."""
  return hbm_ref.at[rows] if cols is None else hbm_ref.at[rows, cols]


class _WindowReadStream(NamedTuple):
  """One HBM array gathered window-by-window into a double-buffered VMEM scratch buffer."""

  hbm_source: Any  # [R, cols] source array in HBM
  vmem_buf: Any  # VMEM [2, buf_rows, cols] destination buffer (2 slots for double-buffering)
  dma_sem: Any  # DMA semaphores [2] (one per slot)
  init_fill: Any  # value both slots are initialized with at step 0, or None
  hbm_cols: Any = None  # hidden-column slice of `hbm_source` to read (see `_hbm_col_chunk`), or None for all


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
            _hbm_window(stream.hbm_source, hbm_slice, stream.hbm_cols),
            stream.vmem_buf.at[dst_slot, vmem_slice],
            stream.dma_sem.at[dst_slot],
        ).start()

    return carry

  lax.fori_loop(0, num_groups, issue_expert_window, 0)


def _prefetch_expert_windows(
    cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align, step_idx, num_steps
):
  """Double-buffered window prefetch: starts the next step's window DMAs and waits for the current step's.

  `step_idx` counts the grid steps of the current pass (`num_steps` of them); the tables describe this step
  and the next one. Returns the active double-buffer `slot` (0 or 1) holding this step's rows.
  At step 0, any stream with `init_fill is not None` initializes both VMEM slots
  so unused tail rows hold safe values (0.0 for activations, `_INFO_FILL` for row_info).
  """
  slot = step_idx % 2

  @pl.when(step_idx == 0)
  def _():
    for stream in streams:
      if stream.init_fill is not None:
        _fill_both_slots(stream.vmem_buf, buf_rows, stream.init_fill)
    _start_expert_window_dmas(cur_dma_table_ref, streams, 0, num_groups, align)

  @pl.when(step_idx + 1 < num_steps)
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
    window_info_vmem,
    slot,
    row_offset,
    num_rows,
    block_tokens,
    block_start_token,
    packed_rows,
    *,
    transpose=False,
    wide_tokens=False,
    expert_chunk=None,
):
  """Decodes `(row_token_in_block, row_topk_slot)` for buffer rows `[row_offset, row_offset + num_rows)`.

  Buffer rows outside `[0, packed_rows)`, belonging to an adjacent token block or (when `expert_chunk` is given)
  to another expert chunk (brought in by 8/16-row DMA alignment padding) receive `row_token_in_block = -1`.
  `wide_tokens` also decodes the third base-256 digit of the token id (`num_tokens > 65536`).

  When `transpose=False`, returns shapes `[num_rows, 1]`.
  When `transpose=True`, transposes the full `[num_rows, 128]` tile before slicing
  the lanes (satisfying Mosaic's 128-lane transpose alignment) and returns `[1, num_rows]`.
  """
  info_tile = window_info_vmem[slot, pl.ds(row_offset, num_rows), :].astype(jnp.float32)
  if transpose:
    info_tile = info_tile.T  # [128, num_rows]
    buf_row_idx = lax.broadcasted_iota(jnp.int32, (1, num_rows), 1) + row_offset

    def lane(idx):
      return info_tile[idx : idx + 1, :]

  else:
    buf_row_idx = lax.broadcasted_iota(jnp.int32, (num_rows, 1), 0) + row_offset

    def lane(idx):
      return info_tile[:, idx : idx + 1]

  token = lane(_INFO_TOKEN_LO) + 256.0 * lane(_INFO_TOKEN_MID)
  if wide_tokens:
    token = token + 65536.0 * lane(_INFO_TOKEN_HI)
  row_topk_slot = lane(_INFO_TOPK).astype(jnp.int32)
  row_token_in_block = token.astype(jnp.int32) - block_start_token
  belongs_to_current_block = (
      (row_token_in_block >= 0) & (row_token_in_block < block_tokens) & (buf_row_idx < packed_rows)
  )
  if expert_chunk is not None:
    belongs_to_current_block &= lane(_INFO_EXPERT_CHUNK).astype(jnp.int32) == expert_chunk
  return jnp.where(belongs_to_current_block, row_token_in_block, -1), row_topk_slot


def _dma_table_block_specs(num_blocks, experts_per_chunk, expert_chunks, *, expert_chunk_innermost):
  """Returns SMEM `BlockSpec`s for the current grid step's DMA table and the next step's (clamped at the end).

  Gather kernels (`expert_chunk_innermost=True`, grid `(col_chunk, block, expert_chunk)`) write `[block_tokens, D]`
  output tiles in token order, so keeping `expert_chunk` innermost accumulates all expert chunks of one token block
  in `acc_vmem` (f32) before advancing `out_ref` to the next token block. Conversely, scatter kernels
  (`expert_chunk_innermost=False`, grid `(col_chunk, expert_chunk, block)`) write expert-sorted windows to HBM where
  each expert's rows are contiguous across token blocks `0..num_blocks-1`, so keeping `block` innermost lets adjacent
  token blocks forward shared boundary granules directly in VMEM (`1 - slot -> slot`).
  """
  stride = _dma_table_stride(experts_per_chunk)
  if expert_chunk_innermost:

    def cur_index(_, block, chunk):
      return (chunk * num_blocks + block,)

    def next_index(_, block, chunk):
      wrapped = (chunk + 1) // expert_chunks
      return ((chunk + 1) % expert_chunks * num_blocks + jnp.minimum(block + wrapped, num_blocks - 1),)

  else:

    def cur_index(_, chunk, block):
      return (chunk * num_blocks + block,)

    def next_index(_, chunk, block):
      return (chunk * num_blocks + jnp.minimum(block + 1, num_blocks - 1),)

  return [
      pl.BlockSpec((stride,), cur_index, memory_space=pltpu.SMEM),
      pl.BlockSpec((stride,), next_index, memory_space=pltpu.SMEM),
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
    *acc_vmem,
    num_groups,
    buf_rows,
    hidden_dim,
    align,
    expert_chunks,
    col_chunks,
    wide_tokens,
):
  """Gathers and sums the expert-sorted rows belonging to one token block `[block_tokens, hidden_dim]`.

  Grid: `(col_chunk, block, expert_chunk)`. One step handles the `num_groups` experts of one expert chunk and
  the `hidden_dim` columns of one column chunk; with several expert chunks the token block is accumulated in
  f32 in `acc_vmem` over the consecutive steps of the block.

  Step 1: Prefetch this step's `num_groups` contiguous expert windows from
          `sorted_x_hbm` into `window_x_vmem[slot]` (`[buf_rows, hidden_dim]`) and
          their `(token_id, topk_slot)` tags into `window_info_vmem[slot]`.
  Step 2: Build `routing_matrix_vmem` (`[num_parts, block_tokens, contract_rows]`) where
          entry `(c, j)` is `weights_ref[c, k_j]` (or `1.0` if unweighted) when
          buffer row `j` belongs to token `c`, else `0.0`.
  Step 3: Compute `out_ref = routing_matrix_vmem @ window_x_vmem[slot]` on the MXU.
  """
  block_tokens = out_ref.shape[0]
  num_parts = routing_matrix_vmem.shape[0]  # 1 for bf16 weights / unweighted; 2 (hi + lo) for f32 weights
  col_chunk_idx, block_idx, expert_chunk_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  step_idx = block_idx * expert_chunks + expert_chunk_idx
  num_steps = pl.num_programs(1) * expert_chunks
  decode_kwargs = {"wide_tokens": wide_tokens, "expert_chunk": expert_chunk_idx if expert_chunks > 1 else None}

  # Step 1: DMA the expert windows for this step into VMEM (and prefetch the next step's).
  hbm_cols = _hbm_col_chunk(col_chunk_idx, hidden_dim, col_chunks)
  streams = (
      _WindowReadStream(sorted_x_hbm, window_x_vmem, x_dma_sem, init_fill=0, hbm_cols=hbm_cols),
      _WindowReadStream(row_info_hbm, window_info_vmem, info_dma_sem, init_fill=_INFO_FILL),
  )
  slot = _prefetch_expert_windows(
      cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align, step_idx, num_steps
  )
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
  block_start_token = block_idx * block_tokens

  def build_routing_matrix_chunk(chunk_idx, carry):
    row_offset = pl.multiple_of(chunk_idx * _ROW_CHUNK, _ROW_CHUNK)
    buf_cols = pl.ds(row_offset, _ROW_CHUNK)
    row_token_in_block, row_topk_slot = _decode_row_info(
        window_info_vmem,
        slot,
        row_offset,
        _ROW_CHUNK,
        block_tokens,
        block_start_token,
        packed_rows,
        transpose=True,
        **decode_kwargs,
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
        if acc_vmem:
          # Several expert chunks: keep the f32 running sum of this token block across its consecutive steps.
          # `out_ref` keeps the same block for all of them, so the last step's write is the one that lands.
          acc = acc + jnp.where(expert_chunk_idx == 0, 0.0, acc_vmem[0][:, hidden_cols])
          acc_vmem[0][:, hidden_cols] = acc
        out_ref[:, hidden_cols] = acc.astype(out_ref.dtype)
        return carry

      lax.fori_loop(0, hidden_dim // col_chunk, accumulate_output_col_chunk, 0)


def _gather_call(sorted_x, padded_weights, dma_table, row_info, *, cfg: _Config, name):
  """Launches `_gather_kernel` to compute `[cfg.num_tokens, D]` from expert-sorted `sorted_x` `[R, D]`."""
  hidden_dim = sorted_x.shape[1]
  num_tokens, block_tokens, align = cfg.num_tokens, cfg.block_tokens, cfg.align
  expert_chunks, experts_per_chunk, col_chunks = cfg.expert_chunks, cfg.experts_per_chunk, cfg.col_chunks
  num_blocks = num_tokens // block_tokens
  chunk_cols = hidden_dim // col_chunks
  buf_rows = _buffer_rows(block_tokens, cfg.k, experts_per_chunk, align)
  num_parts = 2 if (padded_weights is not None and cfg.split_weights) else 1

  kernel_kwargs = {
      "num_groups": experts_per_chunk,
      "buf_rows": buf_rows,
      "hidden_dim": chunk_cols,
      "align": align,
      "expert_chunks": expert_chunks,
      "col_chunks": col_chunks,
      "wide_tokens": cfg.wide_tokens,
  }
  table_and_hbm_specs = _dma_table_block_specs(
      num_blocks, experts_per_chunk, expert_chunks, expert_chunk_innermost=True
  ) + [
      pl.BlockSpec(memory_space=pl.ANY),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  if padded_weights is not None:
    kernel = functools.partial(_gather_kernel, **kernel_kwargs)
    operands = (padded_weights, dma_table, dma_table, sorted_x, row_info)
    in_specs = [pl.BlockSpec((block_tokens, _LANES), lambda col, block, chunk: (block, 0)), *table_and_hbm_specs]
  else:
    kernel = functools.partial(_gather_kernel, None, **kernel_kwargs)
    operands = (dma_table, dma_table, sorted_x, row_info)
    in_specs = table_and_hbm_specs

  itemsize = jnp.dtype(sorted_x.dtype).itemsize
  vmem_bytes = _gather_vmem_bytes(
      buf_rows,
      block_tokens,
      chunk_cols,
      itemsize,
      weighted=padded_weights is not None,
      split_weights=cfg.split_weights,
      accumulate=expert_chunks > 1,
  )
  scratch_shapes = [
      pltpu.VMEM((2, buf_rows, chunk_cols), sorted_x.dtype),  # window_x_vmem
      pltpu.VMEM((2, buf_rows, _LANES), row_info.dtype),  # window_info_vmem
      pltpu.VMEM((num_parts, block_tokens, buf_rows), sorted_x.dtype),  # routing_matrix_vmem
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
  ]
  if expert_chunks > 1:
    scratch_shapes.append(pltpu.VMEM((block_tokens, chunk_cols), jnp.float32))  # acc_vmem
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(
          (num_tokens, hidden_dim), sorted_x.dtype, manual_axis_type=jax.typeof(sorted_x).manual_axis_type
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(col_chunks, num_blocks, expert_chunks),
          in_specs=in_specs,
          out_specs=pl.BlockSpec((block_tokens, chunk_cols), lambda col, block, chunk: (block, col)),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",) * 3, vmem_limit_bytes=min(vmem_bytes, cfg.vmem_limit_bytes)
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
  num_groups: int  # experts per grid step (per expert chunk)
  align: int  # row alignment A
  hbm_cols: Any  # hidden-column slice of `out_hbm` to write (see `_hbm_col_chunk`), or None for all
  block_idx: Any  # token block of this grid step within the current (column chunk, expert chunk) pass
  num_blocks: Any  # token blocks per pass


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
          _hbm_window(scatter.out_hbm, pl.ds(pl.multiple_of(granule_hbm_row, align), align), scatter.hbm_cols),
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


def _prepare_scatter_output_slot(scatter: _ScatterWindowState, cur_dma_table_ref, slot, buf_rows):
  """Ensures `window_out_vmem[slot]` holds all preloaded partial granules before compute starts.

  Block 0 of each pass zeroes both output slots and preloads its own partial granules from HBM.
  Every block waits for the HBM preloads into `slot` (started one step ahead) to
  finish, then vector-copies any boundary granules shared with the immediately
  preceding block (`1 - slot`).
  """
  block_idx = scatter.block_idx

  @pl.when(block_idx == 0)
  def _():
    _fill_both_slots(scatter.window_out_vmem, buf_rows, 0)
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
  block_idx, num_blocks = scatter.block_idx, scatter.num_blocks
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
          _hbm_window(scatter.out_hbm, hbm_slice, scatter.hbm_cols),
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
    decode_kwargs,
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
    # Pallas VMEM slices require a static length (`chunk_rows`), so when `buf_rows` is not a multiple of `chunk_rows`
    # the final slice shifts back to `buf_rows - chunk_rows`, overlapping the previous chunk. `grad_x` assignment (`=`)
    # is idempotent on the overlap, whereas `grad_weights` accumulates (`+=`) and masks out `row < chunk_start` below.
    row_offset = pl.multiple_of(jnp.minimum(chunk_start, buf_rows - chunk_rows), _ROW_CHUNK)
    buf_rows_slice = pl.ds(row_offset, chunk_rows)
    row_token_in_block, row_topk_slot = _decode_row_info(
        window_info_vmem, slot, row_offset, chunk_rows, block_tokens, block_start_token, packed_rows, **decode_kwargs
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
        window_info_vmem,
        slot,
        row_offset,
        chunk_rows,
        block_tokens,
        block_start_token,
        packed_rows,
        transpose=True,
        **decode_kwargs,
    )
    is_target_token_t = (token_row_iota == row_token_in_block_t).astype(dtype)  # [block_tokens, chunk_rows]
    # Split the f32 dot products into 3 bf16 parts (3 * 8 = 24 mantissa bits) so the MXU (`bf16 @ bf16 -> f32`)
    # routes `grad_weight_by_topk_slot` into `[block_tokens, 128]` with full f32 precision and no VPU scatter.
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

  grad_weights = lax.fori_loop(0, num_chunks, process_row_chunk, jnp.zeros((block_tokens, _LANES), jnp.float32))
  if len(grad_weights_ref.shape) == 3:  # one partial result per (column chunk, expert chunk) pass
    grad_weights_ref[0, :, :] = grad_weights
  else:
    grad_weights_ref[...] = grad_weights


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
    expert_chunks,
    col_chunks,
    wide_tokens,
):
  """Combine backward kernel: computes `grad_weights_ref` and scatters `grad_x` windows to `grad_x_hbm`.

  Grid: `(col_chunk, expert_chunk, block)`. Each `(col_chunk, expert_chunk)` pass walks all token blocks for
  `num_groups` experts and `hidden_dim` columns and writes its own partial `grad_weights_ref`.
  """
  block_tokens = grad_out_ref.shape[0]
  col_chunk_idx, expert_chunk_idx, block_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  num_blocks = pl.num_programs(2)
  hbm_cols = _hbm_col_chunk(col_chunk_idx, hidden_dim, col_chunks)
  streams = (
      _WindowReadStream(sorted_x_hbm, window_x_vmem, x_dma_sem, init_fill=None, hbm_cols=hbm_cols),
      _WindowReadStream(row_info_hbm, window_info_vmem, info_dma_sem, init_fill=_INFO_FILL),
  )
  slot = _prefetch_expert_windows(
      cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align, block_idx, num_blocks
  )
  scatter = _ScatterWindowState(
      grad_x_hbm,
      window_grad_x_vmem,
      writeback_dma_sem,
      preload_dma_sem,
      written_rows_smem,
      preloaded_rows_smem,
      num_groups,
      align,
      hbm_cols,
      block_idx,
      num_blocks,
  )
  _prepare_scatter_output_slot(scatter, cur_dma_table_ref, slot, buf_rows)
  packed_rows = _num_packed_rows(cur_dma_table_ref, num_groups)
  block_start_token = block_idx * block_tokens
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
      decode_kwargs={"wide_tokens": wide_tokens, "expert_chunk": expert_chunk_idx if expert_chunks > 1 else None},
  )
  _write_back_windows_and_preload_next(scatter, cur_dma_table_ref, next_dma_table_ref, slot)


def _bwd_call(sorted_x, padded_weights, dma_table, row_info, grad_out, cfg: _Config):
  """Launches `_bwd_kernel` to compute `(padded_grad_weights [cfg.num_tokens, 128] f32, grad_x [R, D])`."""
  num_rows, hidden_dim = sorted_x.shape
  num_tokens, block_tokens, align = cfg.num_tokens, cfg.block_tokens, cfg.align
  expert_chunks, experts_per_chunk, col_chunks = cfg.expert_chunks, cfg.experts_per_chunk, cfg.col_chunks
  num_blocks = num_tokens // block_tokens
  chunk_cols = hidden_dim // col_chunks
  num_passes = col_chunks * expert_chunks
  buf_rows = _buffer_rows(block_tokens, cfg.k, experts_per_chunk, align)
  in_specs = _dma_table_block_specs(num_blocks, experts_per_chunk, expert_chunks, expert_chunk_innermost=False) + [
      pl.BlockSpec((block_tokens, _LANES), lambda col, chunk, block: (block, 0)),
      pl.BlockSpec((block_tokens, chunk_cols), lambda col, chunk, block: (block, col)),
      pl.BlockSpec(memory_space=pl.ANY),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  itemsize = jnp.dtype(sorted_x.dtype).itemsize
  vmem_bytes = _combine_bwd_vmem_bytes(buf_rows, block_tokens, chunk_cols, itemsize)
  axis_type = jax.typeof(sorted_x).manual_axis_type
  if num_passes == 1:
    grad_weights_shape = (num_tokens, _LANES)
    grad_weights_spec = pl.BlockSpec((block_tokens, _LANES), lambda col, chunk, block: (block, 0))
  else:
    # Each (column chunk, expert chunk) pass visits every token block once: one partial result per pass.
    grad_weights_shape = (num_passes, num_tokens, _LANES)
    grad_weights_spec = pl.BlockSpec(
        (1, block_tokens, _LANES), lambda col, chunk, block: (col * expert_chunks + chunk, block, 0)
    )
  padded_grad_weights, grad_x = pl.pallas_call(
      functools.partial(
          _bwd_kernel,
          num_groups=experts_per_chunk,
          buf_rows=buf_rows,
          hidden_dim=chunk_cols,
          split_weights=cfg.split_weights,
          align=align,
          expert_chunks=expert_chunks,
          col_chunks=col_chunks,
          wide_tokens=cfg.wide_tokens,
      ),
      out_shape=(
          jax.ShapeDtypeStruct(grad_weights_shape, jnp.float32, manual_axis_type=axis_type),
          jax.ShapeDtypeStruct((num_rows, hidden_dim), sorted_x.dtype, manual_axis_type=axis_type),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(col_chunks, expert_chunks, num_blocks),
          in_specs=in_specs,
          out_specs=[grad_weights_spec, pl.BlockSpec(memory_space=pl.ANY)],
          scratch_shapes=[
              pltpu.VMEM((2, buf_rows, chunk_cols), sorted_x.dtype),  # window_x_vmem
              pltpu.VMEM((2, buf_rows, _LANES), row_info.dtype),  # window_info_vmem
              pltpu.VMEM((2, buf_rows, chunk_cols), sorted_x.dtype),  # window_grad_x_vmem
              pltpu.SemaphoreType.DMA((2,)),  # x reads
              pltpu.SemaphoreType.DMA((2,)),  # row-info reads
              pltpu.SemaphoreType.DMA((2,)),  # window writes
              pltpu.SemaphoreType.DMA((2,)),  # partial-granule preloads
              pltpu.SMEM((2,), jnp.int32),  # written_rows_smem
              pltpu.SMEM((2,), jnp.int32),  # preloaded_rows_smem
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",) * 3, vmem_limit_bytes=min(vmem_bytes, cfg.vmem_limit_bytes)
      ),
      interpret=cfg.interpret,
      name="moe_combine_bwd",
  )(dma_table, dma_table, padded_weights, grad_out, sorted_x, row_info)
  if num_passes > 1:
    padded_grad_weights = jnp.sum(padded_grad_weights, axis=0)
  return padded_grad_weights, grad_x


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
    expert_chunks,
    col_chunks,
    wide_tokens,
):
  """Scatters token block `tokens_x_ref` `[block_tokens, hidden_dim]` into `sorted_out_hbm` `[R, D]`.

  Grid: `(col_chunk, expert_chunk, block)`, one pass over the token blocks per (column chunk, expert chunk).
  """
  block_tokens = tokens_x_ref.shape[0]
  col_chunk_idx, expert_chunk_idx, block_idx = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  num_blocks = pl.num_programs(2)
  decode_kwargs = {"wide_tokens": wide_tokens, "expert_chunk": expert_chunk_idx if expert_chunks > 1 else None}
  streams = (_WindowReadStream(row_info_hbm, window_info_vmem, info_dma_sem, init_fill=_INFO_FILL),)
  slot = _prefetch_expert_windows(
      cur_dma_table_ref, next_dma_table_ref, streams, num_groups, buf_rows, align, block_idx, num_blocks
  )
  scatter = _ScatterWindowState(
      sorted_out_hbm,
      window_out_vmem,
      writeback_dma_sem,
      preload_dma_sem,
      written_rows_smem,
      preloaded_rows_smem,
      num_groups,
      align,
      _hbm_col_chunk(col_chunk_idx, hidden_dim, col_chunks),
      block_idx,
      num_blocks,
  )
  _prepare_scatter_output_slot(scatter, cur_dma_table_ref, slot, buf_rows)

  packed_rows = _num_packed_rows(cur_dma_table_ref, num_groups)
  col_chunk = _COL_CHUNK if hidden_dim % _COL_CHUNK == 0 else _LANES
  chunk_rows = min(_SCATTER_ROWS_PER_CHUNK, buf_rows)
  num_chunks = (packed_rows + (chunk_rows - 1)) // chunk_rows
  block_start_token = block_idx * block_tokens
  token_col_iota = lax.broadcasted_iota(jnp.int32, (chunk_rows, block_tokens), 1)

  # Build 0/1 selection matrix `token_select_chunk[j, c] = (token(j) == c)` and multiply by `tokens_x_ref`.
  def process_row_chunk(chunk_idx, carry):
    row_offset = pl.multiple_of(jnp.minimum(chunk_idx * chunk_rows, buf_rows - chunk_rows), _ROW_CHUNK)
    buf_rows_slice = pl.ds(row_offset, chunk_rows)
    row_token_in_block, _ = _decode_row_info(
        window_info_vmem, slot, row_offset, chunk_rows, block_tokens, block_start_token, packed_rows, **decode_kwargs
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
  """Launches `_permute_fwd_kernel` to produce `[R, D]` expert-sorted rows from `x` `[cfg.num_tokens, D]`."""
  hidden_dim = x.shape[1]
  num_rows = row_info.shape[0]
  block_tokens, align = cfg.block_tokens, cfg.align
  expert_chunks, experts_per_chunk, col_chunks = cfg.expert_chunks, cfg.experts_per_chunk, cfg.col_chunks
  num_blocks = cfg.num_tokens // block_tokens
  chunk_cols = hidden_dim // col_chunks
  buf_rows = _buffer_rows(block_tokens, cfg.k, experts_per_chunk, align)
  in_specs = _dma_table_block_specs(num_blocks, experts_per_chunk, expert_chunks, expert_chunk_innermost=False) + [
      pl.BlockSpec((block_tokens, chunk_cols), lambda col, chunk, block: (block, col)),
      pl.BlockSpec(memory_space=pl.ANY),
  ]
  itemsize = jnp.dtype(x.dtype).itemsize
  vmem_bytes = _permute_fwd_vmem_bytes(buf_rows, block_tokens, chunk_cols, itemsize)
  return pl.pallas_call(
      functools.partial(
          _permute_fwd_kernel,
          num_groups=experts_per_chunk,
          buf_rows=buf_rows,
          hidden_dim=chunk_cols,
          align=align,
          expert_chunks=expert_chunks,
          col_chunks=col_chunks,
          wide_tokens=cfg.wide_tokens,
      ),
      out_shape=jax.ShapeDtypeStruct((num_rows, hidden_dim), x.dtype, manual_axis_type=jax.typeof(x).manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=(col_chunks, expert_chunks, num_blocks),
          in_specs=in_specs,
          out_specs=pl.BlockSpec(memory_space=pl.ANY),
          scratch_shapes=[
              pltpu.VMEM((2, buf_rows, _LANES), row_info.dtype),  # window_info_vmem
              pltpu.VMEM((2, buf_rows, chunk_cols), x.dtype),  # window_out_vmem
              pltpu.SemaphoreType.DMA((2,)),  # row-info reads
              pltpu.SemaphoreType.DMA((2,)),  # window writes
              pltpu.SemaphoreType.DMA((2,)),  # partial-granule preloads
              pltpu.SMEM((2,), jnp.int32),  # written_rows_smem
              pltpu.SMEM((2,), jnp.int32),  # preloaded_rows_smem
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",) * 3, vmem_limit_bytes=min(vmem_bytes, cfg.vmem_limit_bytes)
      ),
      interpret=cfg.interpret,
      name="moe_permute_fwd",
  )(dma_table, dma_table, x, row_info)


# ---------------------------------------------------------------------------
# custom VJPs and public entry points
# ---------------------------------------------------------------------------


def _resolve_config(
    rows_shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    num_tokens: int,
    k: int,
    group_sizes: jax.Array | None,
    *,
    sort_idx_shape: tuple[int, ...] | None = None,
    block_tokens: int | None = None,
    interpret: bool | pltpu.InterpretParams | None = None,
    align: int = 8,
    split_weights: bool = False,
    default_block_tokens: int = DEFAULT_COMBINE_BLOCK_TOKENS,
    is_combine: bool = True,
    op_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    expert_chunks: int = 1,
    col_chunks: int = 1,
) -> _Config:
  """Validates kernel preconditions and VMEM budget and returns `_Config` (raises `ValueError` on failure).

  `rows_shape` is the unpadded `[T*K, D]` array and `num_tokens` the unpadded `T`. The config describes the
  padded problem the kernels run on: `T` rounded up to a multiple of `block_tokens` (padding tokens own no rows)
  and `D` rounded up to a multiple of 128 (zero columns).
  """
  if len(rows_shape) != 2 or rows_shape[1] <= 0:
    raise ValueError(f"Expected 2D array [num_tokens * k, hidden_dim] with hidden_dim > 0, got shape {rows_shape}.")
  if group_sizes is None or group_sizes.ndim != 1 or group_sizes.shape[0] <= 0:
    raise ValueError("group_sizes must be a non-empty 1D array.")
  if not 1 <= k <= _LANES:
    raise ValueError(f"num_experts_per_tok (k) must be in [1, {_LANES}], got {k}.")
  if num_tokens <= 0 or rows_shape[0] != num_tokens * k:
    raise ValueError(f"Expected rows_shape[0] == num_tokens * k ({num_tokens * k}), got shape {rows_shape}.")
  if sort_idx_shape is not None and sort_idx_shape != (rows_shape[0],):
    raise ValueError(f"sort_idx shape {sort_idx_shape} must match ({rows_shape[0]},).")
  if jnp.dtype(dtype) != jnp.bfloat16:
    raise ValueError(f"MoE TensorCore kernels require bfloat16 activations, got {jnp.dtype(dtype)}.")
  if align < 8 or (align & (align - 1)) != 0:
    raise ValueError(f"align must be a power of two >= 8, got {align}.")
  if rows_shape[0] % align != 0:
    raise ValueError(f"Total routed rows num_tokens * k ({rows_shape[0]}) must be a multiple of align ({align}).")

  block_tokens = default_block_tokens if block_tokens is None else int(block_tokens)
  if block_tokens < _LANES or (block_tokens & (block_tokens - 1)) != 0:
    raise ValueError(f"block_tokens must be a power of two >= {_LANES}, got {block_tokens}.")
  padded_tokens = _round_up(num_tokens, block_tokens)
  if padded_tokens > _MAX_TOKENS:
    raise ValueError(f"Padded token count {padded_tokens} exceeds maximum supported tokens {_MAX_TOKENS}.")

  num_groups = int(group_sizes.shape[0])
  expert_chunk_limit = min(num_groups, _MAX_EXPERT_CHUNKS)
  if not 1 <= expert_chunks <= expert_chunk_limit:
    raise ValueError(f"expert_chunks must be in [1, {expert_chunk_limit}], got {expert_chunks}.")
  padded_dim = _round_up(rows_shape[1], _LANES)
  if col_chunks < 1 or padded_dim % (col_chunks * _LANES) != 0:
    raise ValueError(
        f"col_chunks ({col_chunks}) must be >= 1 and divide padded hidden_dim // {_LANES} ({padded_dim // _LANES})."
    )

  if vmem_limit_bytes is None:
    vmem_limit_bytes = default_vmem_limit_bytes()
  vmem_bytes = _required_vmem_bytes(
      block_tokens,
      k,
      -(-num_groups // expert_chunks),
      padded_dim // col_chunks,
      jnp.dtype(dtype).itemsize,
      split_weights=split_weights,
      align=align,
      is_combine=is_combine,
      accumulate=expert_chunks > 1,
  )
  if vmem_bytes > vmem_limit_bytes:
    kernel_name = op_name or ("combine" if is_combine else "permute")
    raise ValueError(
        f"MoE TensorCore {kernel_name} kernel requires {vmem_bytes / (1024 * 1024):.1f} MiB of VMEM, exceeding limit"
        f" {vmem_limit_bytes / (1024 * 1024):.1f} MiB (block_tokens={block_tokens}, expert_chunks={expert_chunks},"
        f" col_chunks={col_chunks}, num_experts={num_groups}, top_k={k}, hidden_dim={rows_shape[1]}); reduce"
        " block_tokens or increase expert_chunks / col_chunks."
    )
  if not kernel_supported(
      (rows_shape[0], padded_dim),
      dtype,
      padded_tokens,
      k,
      block_tokens,
      align,
      num_groups=num_groups,
      split_weights=split_weights,
      is_combine=is_combine,
      vmem_limit_bytes=vmem_limit_bytes,
      expert_chunks=expert_chunks,
      col_chunks=col_chunks,
  ):
    raise ValueError(
        f"Unsupported MoE TensorCore kernel configuration for shape={rows_shape}, dtype={dtype},"
        f" num_tokens={num_tokens}, k={k}, block_tokens={block_tokens}, expert_chunks={expert_chunks},"
        f" col_chunks={col_chunks}."
    )
  if interpret is None:
    abstract_mesh = jax.sharding.get_abstract_mesh()
    mesh_platform = getattr(getattr(abstract_mesh, "abstract_device", None), "platform", None)
    on_tpu = jax.default_backend() == "tpu" or mesh_platform == "tpu"
    interpret = False if on_tpu else pltpu.InterpretParams()
  elif interpret is True:  # dynamic-size DMAs require the Mosaic TPU interpreter
    interpret = pltpu.InterpretParams()
  return _Config(
      k,
      block_tokens,
      num_groups,
      split_weights,
      interpret,
      align,
      vmem_limit_bytes,
      padded_tokens,
      expert_chunks,
      col_chunks,
  )


def _pad_trailing(x, num_rows, num_cols):
  """Zero-pads the 2D array `x` at the end of each axis up to `[num_rows, num_cols]`."""
  if x.shape == (num_rows, num_cols):
    return x
  return jnp.pad(x, ((0, num_rows - x.shape[0]), (0, num_cols - x.shape[1])))


def _row_info(sort_idx, group_sizes, cfg: _Config):
  """Recomputes `[R, 128]` bf16 `row_info` on the fly in fwd/bwd to avoid saving a `[R, 128]` residual in HBM."""
  return _build_row_info(sort_idx, cfg.k, group_sizes, cfg.expert_chunks)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _combine_vjp(x, weights, sort_idx, group_sizes, dma_table, cfg):
  padded_weights = jnp.pad(weights.astype(jnp.float32), ((0, 0), (0, _LANES - cfg.k)))
  return _gather_call(
      x, padded_weights, dma_table, _row_info(sort_idx, group_sizes, cfg), cfg=cfg, name="moe_combine_fwd"
  )


def _combine_fwd(x, weights, sort_idx, group_sizes, dma_table, cfg):
  """Forward pass for `_combine_vjp`, saving inputs as residuals for backward."""
  out = _combine_vjp(x, weights, sort_idx, group_sizes, dma_table, cfg)
  return out, (x, weights, sort_idx, group_sizes, dma_table)


def _combine_bwd(cfg, residuals, grad_out):
  """Backward pass for `_combine_vjp`, computing `grad_x` and `grad_weights`."""
  x, weights, sort_idx, group_sizes, dma_table = residuals
  padded_weights = jnp.pad(weights.astype(jnp.float32), ((0, 0), (0, _LANES - cfg.k)))
  padded_grad_weights, grad_x = _bwd_call(
      x,
      padded_weights,
      dma_table,
      _row_info(sort_idx, group_sizes, cfg),
      grad_out.astype(x.dtype),
      cfg,
  )
  return grad_x, padded_grad_weights[:, : cfg.k].astype(weights.dtype), None, None, None


_combine_vjp.defvjp(_combine_fwd, _combine_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _unpermute_sum_vjp(x, sort_idx, group_sizes, dma_table, cfg):
  return _gather_call(
      x, None, dma_table, _row_info(sort_idx, group_sizes, cfg), cfg=cfg, name="moe_unpermute_sum_fwd"
  )


def _unpermute_sum_fwd(x, sort_idx, group_sizes, dma_table, cfg):
  """Forward pass for `_unpermute_sum_vjp`, saving the routing metadata for backward."""
  return _unpermute_sum_vjp(x, sort_idx, group_sizes, dma_table, cfg), (sort_idx, group_sizes, dma_table)


def _unpermute_sum_bwd(cfg, residuals, grad_out):
  """Backward pass for `_unpermute_sum_vjp`: every row gets its token's gradient, i.e. the permute forward."""
  sort_idx, group_sizes, dma_table = residuals
  grad_x = _permute_fwd_call(grad_out, dma_table, _row_info(sort_idx, group_sizes, cfg), cfg)
  return grad_x, None, None, None


_unpermute_sum_vjp.defvjp(_unpermute_sum_fwd, _unpermute_sum_bwd)


def combine(
    x: jax.Array,
    sort_idx: jax.Array,
    weights: jax.Array | None,
    group_sizes: jax.Array,
    *,
    num_experts_per_tok: int | None = None,
    block_tokens: int | None = None,
    expert_chunks: int = 1,
    col_chunks: int = 1,
    interpret: bool | pltpu.InterpretParams | None = None,
    align: int = 8,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Combines expert-sorted rows `x` into token outputs `y[t] = sum_k weights[t, k] * x[argsort(sort_idx)[t*K + k]]`.

  When `weights` is None (e.g. Llama4, which applies router scores before the experts), computes the unweighted
  sum `y[t] = sum_k x[argsort(sort_idx)[t*K + k]]` using `num_experts_per_tok`.

  Args:
    x: `[T*K, D]` bf16 expert outputs in expert-sorted order.
    sort_idx: `[T*K]` int, stable argsort of the flat (token-major) expert ids.
    weights: `[T, K]` routing weights (any float dtype; non-bf16 weights are split into bf16 hi + lo
      parts, ~16 mantissa bits), or None for unit weights (requires `num_experts_per_tok`).
    group_sizes: `[G]` int, rows per expert group of `x` (must sum to `T*K`).
    num_experts_per_tok: `K` (required when `weights` is None; optional otherwise).
    block_tokens: tokens per grid step (default: 256, power of two >= 128).
    expert_chunks: process the experts in this many chunks per token block (default: 1 = single pass).
    col_chunks: process the hidden dimension in this many column chunks (default: 1 = single pass).
    interpret: run under the Mosaic TPU interpreter (default: False on TPU, `pltpu.InterpretParams()` off-TPU).
    align: DMA / VMEM row alignment, a power of two >= 8.
    vmem_limit_bytes: maximum per-kernel VMEM footprint in bytes (default: 95% of the chip's VMEM, see
      `default_vmem_limit_bytes`).

  `T` that is not a multiple of `block_tokens` is padded with tokens that own no rows, and `D` that is not a
  multiple of 128 is zero-padded (one extra `[T*K, D]` copy of `x`); both are sliced off the result.

  Returns:
    `[T, D]` array in `x.dtype` (f32 accumulation), differentiable w.r.t. `x` (and `weights` when provided).

  Raises:
    ValueError: If the inputs or hyperparameters do not satisfy the kernel preconditions or VMEM budget.
  """
  if weights is None:
    if num_experts_per_tok is None or num_experts_per_tok <= 0:
      raise ValueError("num_experts_per_tok must be a positive integer when weights is None.")
    k = int(num_experts_per_tok)
    if x.ndim != 2 or x.shape[0] % k != 0:
      raise ValueError(f"Expected 2D x with rows divisible by k={k}, got shape {x.shape}.")
    num_rows, hidden_dim = x.shape
    num_tokens = num_rows // k
    split_weights = False
    is_combine = False
  else:
    if weights.ndim != 2:
      raise ValueError(f"Expected 2D weights [num_tokens, k], got shape {weights.shape}.")
    num_tokens, k = weights.shape
    if num_experts_per_tok is not None and int(num_experts_per_tok) != k:
      raise ValueError(f"num_experts_per_tok={num_experts_per_tok} does not match weights.shape[1]={k}.")
    if x.ndim != 2:
      raise ValueError(f"Expected 2D x [num_tokens * k, hidden_dim], got shape {x.shape}.")
    num_rows, hidden_dim = x.shape
    split_weights = jnp.dtype(weights.dtype) != jnp.bfloat16
    is_combine = True

  cfg = _resolve_config(
      x.shape,
      x.dtype,
      num_tokens,
      k,
      group_sizes,
      sort_idx_shape=sort_idx.shape,
      block_tokens=block_tokens,
      interpret=interpret,
      align=align,
      split_weights=split_weights,
      default_block_tokens=DEFAULT_COMBINE_BLOCK_TOKENS,
      is_combine=is_combine,
      op_name="combine",
      vmem_limit_bytes=vmem_limit_bytes,
      expert_chunks=expert_chunks,
      col_chunks=col_chunks,
  )
  dma_table = _build_dma_table(sort_idx, group_sizes, cfg.num_tokens, k, cfg.block_tokens, align, cfg.expert_chunks)
  padded_x = _pad_trailing(x, num_rows, _round_up(hidden_dim, _LANES))
  if weights is None:
    out = _unpermute_sum_vjp(padded_x, sort_idx, group_sizes, dma_table, cfg)
  else:
    out = _combine_vjp(
        padded_x,
        _pad_trailing(weights, cfg.num_tokens, k),
        sort_idx,
        group_sizes,
        dma_table,
        cfg,
    )
  return out if out.shape == (num_tokens, hidden_dim) else out[:num_tokens, :hidden_dim]


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _permute_vjp(x, sort_idx, group_sizes, dma_table, cfg):
  return _permute_fwd_call(x, dma_table, _row_info(sort_idx, group_sizes, cfg), cfg)


def _permute_fwd(x, sort_idx, group_sizes, dma_table, cfg):
  """Forward pass for `_permute_vjp`, saving the routing metadata for backward."""
  return _permute_vjp(x, sort_idx, group_sizes, dma_table, cfg), (sort_idx, group_sizes, dma_table)


def _permute_bwd(cfg, residuals, grad_out):
  """Backward pass for `_permute_vjp`, summing the `K` expert gradients per token via `_gather_call`."""
  sort_idx, group_sizes, dma_table = residuals
  grad_x = _gather_call(
      grad_out, None, dma_table, _row_info(sort_idx, group_sizes, cfg), cfg=cfg, name="moe_permute_bwd"
  )
  return grad_x, None, None, None


_permute_vjp.defvjp(_permute_fwd, _permute_bwd)


def permute(
    x: jax.Array,
    sort_idx: jax.Array,
    group_sizes: jax.Array,
    num_experts_per_tok: int,
    *,
    block_tokens: int | None = None,
    expert_chunks: int = 1,
    col_chunks: int = 1,
    interpret: bool | pltpu.InterpretParams | None = None,
    align: int = 8,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
  """Permutes token activations `x` `[T, D]` into expert-sorted order `[T*K, D]` on the TensorCore.

  Forward: `y[r] = x[sort_idx[r] // K]` built with the same block / row-window DMA
  scheme as `combine` but with unit weights (one-hot selection on the MXU).
  Backward: `dx[t] = sum` over the `K` rows of `t` in `dy`, i.e. the combine forward
  with unit weights.

  Args:
    x: `[T, D]` bf16 token activations.
    sort_idx: `[T*K]` int, stable argsort of the flat (token-major) expert ids.
    group_sizes: `[G]` int, rows per expert group (must sum to `T*K`).
    num_experts_per_tok: `K`.
    block_tokens: tokens per grid step (default: 512, power of two >= 128).
    expert_chunks: process the experts in this many chunks per token block (default: 1 = single pass).
    col_chunks: process the hidden dimension in this many column chunks (default: 1 = single pass).
    interpret: run under the Mosaic TPU interpreter (default: False on TPU, `pltpu.InterpretParams()` off-TPU).
    align: DMA / VMEM row alignment, a power of two >= 8.
    vmem_limit_bytes: maximum per-kernel VMEM footprint in bytes (default: 95% of the chip's VMEM, see
      `default_vmem_limit_bytes`).

  `T` that is not a multiple of `block_tokens` is padded with tokens that own no rows; `D` that is not a multiple
  of 128 is zero-padded and the padding sliced off the result (one extra `[T*K, D]` copy).

  Returns:
    `[T*K, D]` array in `x.dtype`, differentiable w.r.t. `x`.

  Raises:
    ValueError: If the inputs or hyperparameters do not satisfy the kernel preconditions or VMEM budget.
  """
  if x.ndim != 2:
    raise ValueError(f"Expected 2D x [num_tokens, hidden_dim], got shape {x.shape}.")
  num_tokens, hidden_dim = x.shape
  k = int(num_experts_per_tok)
  cfg = _resolve_config(
      (num_tokens * k, hidden_dim),
      x.dtype,
      num_tokens,
      k,
      group_sizes,
      sort_idx_shape=sort_idx.shape,
      block_tokens=block_tokens,
      interpret=interpret,
      align=align,
      split_weights=False,
      default_block_tokens=DEFAULT_PERMUTE_BLOCK_TOKENS,
      is_combine=False,
      vmem_limit_bytes=vmem_limit_bytes,
      expert_chunks=expert_chunks,
      col_chunks=col_chunks,
  )
  dma_table = _build_dma_table(sort_idx, group_sizes, cfg.num_tokens, k, cfg.block_tokens, align, cfg.expert_chunks)
  out = _permute_vjp(
      _pad_trailing(x, cfg.num_tokens, _round_up(hidden_dim, _LANES)), sort_idx, group_sizes, dma_table, cfg
  )
  return out if out.shape[1] == hidden_dim else out[:, :hidden_dim]
