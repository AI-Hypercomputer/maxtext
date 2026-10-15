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

"""SiLU-GLU of the prefused gate/up GEMM output, and its gradient, over the valid rows of the expert buffer only.

Only rows ``[0, limit)`` of the packed ``[buffer_size, 2 * mlp]`` buffer hold tokens. XLA's fusions stream the
whole buffer, and the backward writes the two half gradients and then concatenates them, one more pass. These
kernels visit the row blocks up to the one holding row ``limit - 1`` (later blocks map to it: no fetch, no
write, those output rows are left as they are), and the backward writes both halves of its block at once.
Consumers never read the rows left unwritten: gmm_v2 reads rows up to the sublane tile of row ``limit - 1``,
tgmm_v2 masks them.

Both kernels run the op sequence of XLA's fusions for ``silu(gate) * up`` and its autodiff gradient (negate,
exp, + 1, divide, multiply) the way XLA runs those bf16 fusions on a TPU: in f32, rounding only the outputs.
"""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Rows per grid step. A (1024, 1024) bf16 block is 2 MiB. A partial last block is fine: Pallas drops its
# out-of-range writes.
_BLOCK_ROWS = 1024
# Rows per inner-loop tile.
_TILE_ROWS = 64


def _sigmoid(x):
  """XLA's expansion of logistic: 1 / (exp(-x) + 1)."""
  return 1.0 / (jnp.exp(-x) + 1.0)


def _glu_fwd_tile(gate, up):
  """`silu(gate) * up` as XLA's fusion computes it: in f32, rounded once to the output dtype."""
  g = gate.astype(jnp.float32)
  return ((g * _sigmoid(g)) * up.astype(jnp.float32)).astype(gate.dtype)


def _glu_bwd_tile(gate, up, d_out):
  """The autodiff gradient of `_glu_fwd_tile` as XLA's fusion computes it (f32, rounded once), -0 as +0."""
  g, u, d = gate.astype(jnp.float32), up.astype(jnp.float32), d_out.astype(jnp.float32)
  sig = _sigmoid(g)
  d_up_in = d * u
  d_gate = d_up_in * sig + (g * d_up_in) * (sig * (1.0 - sig))
  d_up = (g * sig) * d
  d_gate = jnp.where(d_gate == 0, 0.0, d_gate)
  d_up = jnp.where(d_up == 0, 0.0, d_up)
  return d_gate.astype(gate.dtype), d_up.astype(gate.dtype)


def kernel_supported(fused):
  rows, width = fused.shape
  return fused.dtype == jnp.bfloat16 and rows % 16 == 0 and width % 256 == 0


def _last_block(limit):
  """The row block holding row ``limit - 1`` (block 0 if there are no valid rows)."""
  return jnp.maximum(pl.cdiv(limit, _BLOCK_ROWS) - 1, 0)


def _row_map(column_block):
  def index_map(i, limit_ref):
    return jnp.minimum(i, _last_block(limit_ref[0])), column_block

  return index_map


def _run(kernel, operands, in_specs, out_spec, out_shape, *, limit, flops, transcendentals, interpret, name):
  rows = out_shape.shape[0]
  bytes_accessed = sum(x.size * x.dtype.itemsize for x in operands) + out_shape.size * out_shape.dtype.itemsize
  return pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(pl.cdiv(rows, _BLOCK_ROWS),),
          in_specs=in_specs,
          out_specs=out_spec,
      ),
      out_shape=out_shape,
      compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=48 * 1024 * 1024),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=transcendentals, bytes_accessed=bytes_accessed),
      interpret=interpret,
      name=name,
  )(jnp.reshape(jnp.asarray(limit).astype(jnp.int32), (1,)), *operands)


def _visit_rows(i, limit, tile_fn):
  """Runs `tile_fn(rows)` over the row tiles of block i if the block holds valid rows (or is block 0)."""

  @pl.when(i <= _last_block(limit))
  def _():
    def body(t, carry):
      tile_fn(pl.ds(pl.multiple_of(t * _TILE_ROWS, _TILE_ROWS), _TILE_ROWS))
      return carry

    lax.fori_loop(0, _BLOCK_ROWS // _TILE_ROWS, body, 0)


def _glu_fwd_kernel(limit_ref, gate_ref, up_ref, out_ref):
  def tile(rows):
    out_ref[rows, :] = _glu_fwd_tile(gate_ref[rows, :], up_ref[rows, :])

  _visit_rows(pl.program_id(0), limit_ref[0], tile)


def _glu_bwd_kernel(limit_ref, gate_ref, up_ref, d_ref, out_ref):
  mlp = gate_ref.shape[1]

  def tile(rows):
    d_gate, d_up = _glu_bwd_tile(gate_ref[rows, :], up_ref[rows, :], d_ref[rows, :])
    out_ref[rows, :mlp] = d_gate
    out_ref[rows, mlp:] = d_up

  _visit_rows(pl.program_id(0), limit_ref[0], tile)


def glu_fwd_rows(fused, limit, *, interpret=False):
  """``silu(gate) * up`` of rows ``[0, limit)`` of ``fused = [gate | up]``, ``[rows, 2 * mlp]`` -> ``[rows, mlp]``."""
  rows, width = fused.shape
  mlp = width // 2
  return _run(
      _glu_fwd_kernel,
      (fused, fused),
      [pl.BlockSpec((_BLOCK_ROWS, mlp), _row_map(0)), pl.BlockSpec((_BLOCK_ROWS, mlp), _row_map(1))],
      pl.BlockSpec((_BLOCK_ROWS, mlp), _row_map(0)),
      jax.ShapeDtypeStruct((rows, mlp), fused.dtype),
      limit=limit,
      flops=4 * rows * mlp,
      transcendentals=2 * rows * mlp,
      interpret=interpret,
      name="glu_fwd_rows",
  )


def glu_bwd_rows(fused, d_out, limit, *, interpret=False):
  """The ``[rows, 2 * mlp]`` gradient ``[d_gate | d_up]`` of `glu_fwd_rows` at rows ``[0, limit)``, -0 as +0."""
  rows, width = fused.shape
  mlp = width // 2
  return _run(
      _glu_bwd_kernel,
      (fused, fused, d_out),
      [
          pl.BlockSpec((_BLOCK_ROWS, mlp), _row_map(0)),
          pl.BlockSpec((_BLOCK_ROWS, mlp), _row_map(1)),
          pl.BlockSpec((_BLOCK_ROWS, mlp), _row_map(0)),
      ],
      pl.BlockSpec((_BLOCK_ROWS, width), _row_map(0)),
      jax.ShapeDtypeStruct((rows, width), fused.dtype),
      limit=limit,
      flops=12 * rows * mlp,
      transcendentals=2 * rows * mlp,
      interpret=interpret,
      name="glu_bwd_rows",
  )
