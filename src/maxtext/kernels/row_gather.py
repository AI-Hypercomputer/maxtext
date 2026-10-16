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

"""Pallas TPU row gather: out[i, :] = x[idx[i], :].

MoE token routing gathers whole activation rows (for example 4 KiB bf16 rows of
width 2048). XLA lowers `x[idx]` to a gather fusion that reaches only a small
fraction of HBM bandwidth on TPU v6e. This kernel issues one DMA per row from
HBM straight into the VMEM output block. A block of indices sits in SMEM, all
row copies of a block are in flight together, and Pallas pipelines the output
write-back with the next block's row copies.

The kernel is not differentiable on its own; callers use it inside custom VJPs
whose backward passes are gathers too.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_LANES = 128


def _gather_kernel(idx_ref, x_hbm, o_ref, buf, sem, *, rows, d, unroll):
  """Pallas kernel that gathers rows from HBM into VMEM via async DMA."""

  def start(r, carry):
    pltpu.make_async_copy(x_hbm.at[pl.ds(idx_ref[r], 1)], buf.at[pl.ds(r, 1)], sem.at[0]).start()
    return carry

  jax.lax.fori_loop(0, rows, start, 0, unroll=unroll)

  def wait(r, carry):
    # Every copy has the same size, so any same-shaped descriptor waits for one.
    pltpu.make_async_copy(x_hbm.at[pl.ds(0, 1)], buf.at[pl.ds(r, 1)], sem.at[0]).wait()
    return carry

  jax.lax.fori_loop(0, rows, wait, 0, unroll=unroll)
  o_ref[...] = buf[...].reshape(rows, d)


def _block_rows(m, cap):
  r = min(cap, m)
  while m % r:
    r -= 1
  return r


@functools.partial(jax.jit, static_argnames=("block_rows", "interpret"))
def take_rows(x: jax.Array, idx: jax.Array, *, block_rows: int = 512, interpret: bool = False) -> jax.Array:
  """Returns x[idx] for x of shape (N, ...) and integer idx of shape (M,)."""
  n = x.shape[0]
  tail = x.shape[1:]
  d = x.size // n
  if d % _LANES:
    raise ValueError(f"row size {d} must be a multiple of {_LANES}")
  # Rows sit on the leading (untiled) axis, so a single-row DMA has no tile
  # alignment constraint. A (2048,) bf16 row is exactly one (16, 128) tile.
  x3 = x.reshape(n, d // _LANES, _LANES)
  m = idx.shape[0]
  rows = _block_rows(m, block_rows)
  idx = jnp.clip(idx.astype(jnp.int32), 0, n - 1)
  out = pl.pallas_call(
      functools.partial(_gather_kernel, rows=rows, d=d, unroll=8 if rows % 8 == 0 else 1),
      out_shape=jax.ShapeDtypeStruct((m, d), x.dtype),
      grid=(m // rows,),
      in_specs=[
          pl.BlockSpec((rows,), lambda i: (i,), memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=pl.BlockSpec((rows, d), lambda i: (i, 0)),
      scratch_shapes=[
          pltpu.VMEM((rows, d // _LANES, _LANES), x.dtype),
          pltpu.SemaphoreType.DMA((1,)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="row_gather",
  )(idx, x3)
  return out.reshape((m,) + tail)


def gather_rows(x: jax.Array, idx: jax.Array, use_pallas: bool) -> jax.Array:
  """x[idx, ...] via the Pallas kernel on TPU when use_pallas, else XLA."""
  if use_pallas and x.ndim >= 2:
    return take_rows(x, idx, interpret=jax.default_backend() != "tpu")
  return x[idx, ...]
