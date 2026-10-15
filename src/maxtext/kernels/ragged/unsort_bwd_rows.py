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

"""The row-wise part of the ring unsort backward, on the TensorCore, over the valid rows of the buffer only.

Only rows ``[0, limit)`` of the packed ``[buffer_size, hidden]`` expert buffer hold tokens; the rest is
ragged-buffer padding. XLA's fusion of the same three lines streams the whole buffer. This kernel visits the
row blocks up to the one holding row ``limit - 1``: the later blocks map to that block, so they are neither
fetched nor written, and those output rows are left as they are. No consumer reads them: the gmm_v2 /
tgmm_v2 kernels read rows up to the sublane tile of row ``limit - 1`` (tgmm masks them), and the weight-dot
scatter drops rows ``>= limit``.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Rows per grid step: one (8, 128) tile of the lane-dense [rows / 128, 128] weights and dot. A partial last block
# is fine: Pallas drops its out-of-range writes, and its rows are past `limit`.
_BLOCK_ROWS = 1024
# Hidden columns per grid step, at most. A (1024, 2048) bf16 block is 4 MiB; 3 double-buffered fit v7x VMEM.
_BLOCK_COLS = 2048
_LANES = 128


def unsort_bwd_rows_reference(gathered, sorted_tokens, weights, limit):
  """XLA version: the weighted gradient and the f32 row dot of every row, zeroing rows ``>= limit``."""
  mask = jnp.arange(gathered.shape[0]) < limit
  gathered = jnp.where(mask[:, None], gathered, 0.0)
  grad = (gathered * weights[:, None]).astype(gathered.dtype)
  dot = jnp.sum(gathered.astype(jnp.float32) * sorted_tokens.astype(jnp.float32), axis=-1)
  return grad, dot


def _block_cols(hidden):
  return hidden if hidden <= _BLOCK_COLS else _BLOCK_COLS


def _kernel_supported(gathered, sorted_tokens, weights):
  rows, hidden = gathered.shape
  return (
      gathered.dtype == jnp.bfloat16
      and sorted_tokens.dtype == jnp.bfloat16
      and sorted_tokens.shape == gathered.shape
      and weights.shape == (rows,)
      and rows % _LANES == 0
      and hidden % _LANES == 0
      and hidden % _block_cols(hidden) == 0
  )


def _last_block(limit):
  """The row block holding row ``limit - 1`` (block 0 if there are no valid rows)."""
  return jnp.maximum(pl.cdiv(limit, _BLOCK_ROWS) - 1, 0)


def _unsort_bwd_rows_kernel(limit_ref, g_ref, y_ref, w_ref, grad_ref, dot_ref, *, num_col_blocks):  # pylint: disable=too-many-positional-arguments
  """One (1024, cols) block: grad = bf16(f32(g) * w) and the partial f32 row dot, in 128 x 128 tiles."""
  i, j = pl.program_id(0), pl.program_id(1)
  limit = limit_ref[0]
  row0 = i * _BLOCK_ROWS
  num_col_tiles = g_ref.shape[1] // _LANES

  def row_tile(masked, r, carry):
    rows = pl.ds(pl.multiple_of(r * _LANES, _LANES), _LANES)
    # Row a of the tile gets weight w[row0 + 128 r + a], broadcast along the lanes.
    w_col = jnp.transpose(jnp.broadcast_to(w_ref[pl.ds(r, 1), :], (_LANES, _LANES)))
    if masked:
      row_ids = row0 + r * _LANES + lax.broadcasted_iota(jnp.int32, (_LANES, _LANES), 0)
      valid = row_ids < limit
    acc = jnp.zeros((_LANES, _LANES), jnp.float32)
    for c in range(num_col_tiles):
      cols = pl.ds(c * _LANES, _LANES)
      g = g_ref[rows, cols].astype(jnp.float32)
      if masked:
        g = jnp.where(valid, g, 0.0)
      grad_ref[rows, cols] = (g * w_col).astype(grad_ref.dtype)
      acc = acc + g * y_ref[rows, cols].astype(jnp.float32)
    # Lane-dense: entry a of the row sum over the transposed tile is row a's dot over this block's columns.
    part = jnp.sum(jnp.transpose(acc), axis=0, keepdims=True)
    if num_col_blocks > 1:
      part = part + jnp.where(j == 0, 0.0, dot_ref[pl.ds(r, 1), :])
    dot_ref[pl.ds(r, 1), :] = part
    return carry

  num_row_tiles = _BLOCK_ROWS // _LANES

  @pl.when(row0 + _BLOCK_ROWS <= limit)
  def _full():
    lax.fori_loop(0, num_row_tiles, functools.partial(row_tile, False), 0)

  @pl.when(jnp.logical_and(row0 + _BLOCK_ROWS > limit, i == _last_block(limit)))
  def _partial():
    lax.fori_loop(0, num_row_tiles, functools.partial(row_tile, True), 0)


def _unsort_bwd_rows_pallas(gathered, sorted_tokens, weights, limit, *, interpret=False):
  """The Pallas TensorCore kernel; see `unsort_bwd_rows`."""
  rows, hidden = gathered.shape
  block_cols = _block_cols(hidden)
  num_row_blocks = pl.cdiv(rows, _BLOCK_ROWS)
  num_col_blocks = hidden // block_cols

  def data_map(i, j, limit_ref):
    last = _last_block(limit_ref[0])
    return jnp.minimum(i, last), jnp.where(i <= last, j, num_col_blocks - 1)

  def row_map(i, j, limit_ref):
    del j
    return jnp.minimum(i, _last_block(limit_ref[0])), 0

  data_spec = pl.BlockSpec((_BLOCK_ROWS, block_cols), data_map)
  lane_dense_spec = pl.BlockSpec((_BLOCK_ROWS // _LANES, _LANES), row_map)
  grad, dot = pl.pallas_call(
      functools.partial(_unsort_bwd_rows_kernel, num_col_blocks=num_col_blocks),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(num_row_blocks, num_col_blocks),
          in_specs=[data_spec, data_spec, lane_dense_spec],
          out_specs=[data_spec, lane_dense_spec],
      ),
      out_shape=[
          jax.ShapeDtypeStruct((rows, hidden), gathered.dtype),
          jax.ShapeDtypeStruct((rows // _LANES, _LANES), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary"),
          vmem_limit_bytes=48 * 1024 * 1024,
      ),
      cost_estimate=pl.CostEstimate(
          flops=4 * rows * hidden,
          transcendentals=0,
          bytes_accessed=3 * rows * hidden * gathered.dtype.itemsize + 8 * rows,
      ),
      interpret=interpret,
      name="unsort_bwd_rows",
  )(
      jnp.reshape(limit.astype(jnp.int32), (1,)),
      gathered,
      sorted_tokens,
      jnp.reshape(weights.astype(jnp.float32), (rows // _LANES, _LANES)),
  )
  return grad, jnp.reshape(dot, (rows,))


def unsort_bwd_rows(gathered, sorted_tokens, weights, limit, *, interpret=None):
  """The weighted gradient and the routing-weight row dot of the valid rows ``[0, limit)`` of the expert buffer.

  Args:
    gathered: ``[rows, hidden]`` bf16 output gradient, gathered into the packed expert buffer order.
    sorted_tokens: ``[rows, hidden]`` bf16 expert outputs in the same order.
    weights: ``[rows]`` f32 routing weight of each buffer row.
    limit: int32 scalar, the number of valid rows.
    interpret: Pallas interpret mode (tests). None runs the kernel on a TPU target and the XLA version elsewhere.

  Returns:
    ``(grad, dot)``: for rows ``< limit``, ``grad = bf16(f32(gathered) * weights)`` (bit-identical to the XLA
    version) and ``dot = sum(f32(gathered) * f32(sorted_tokens), -1)`` (the summation order differs). Rows from
    ``limit`` to the end of its 1024-row block are computed as zero rows; later rows are not written.
  """
  use_kernel = interpret if interpret is not None else pltpu.is_tpu_device()
  if not use_kernel or not _kernel_supported(gathered, sorted_tokens, weights):
    return unsort_bwd_rows_reference(gathered, sorted_tokens, weights, limit)
  return _unsort_bwd_rows_pallas(gathered, sorted_tokens, weights, jnp.asarray(limit), interpret=bool(interpret))
