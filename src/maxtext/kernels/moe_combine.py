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

"""Pallas TPU fused MoE unsort + combine (forward and backward).

    out[t, :] = sum_j w[t, j] * x[idx[t * k + j], :]

`x` holds the expert outputs in expert-sorted order (shape (N, D)), `idx` maps
each (token, slot) pair to its sorted row, and `w` holds the router weights.
XLA computes this as a gather of all T*k rows followed by a weighted sum; the
gather fusion reaches only a small fraction of HBM bandwidth on TPU v6e. This
kernel DMAs the k rows of each token straight into VMEM and accumulates them
in fp32, so the (T*k, D) unsorted tensor is never written to HBM.

Backward (`combine_bwd`):
  dx[s, :] = w_flat[p[s]] * dy[p[s] // k, :]   with p the inverse of idx
  dw_sorted[s] = <x[s, :], dy[p[s] // k, :]>
  dw[t, j] = dw_sorted[idx[t * k + j]]
`combine_bwd` DMAs `dy[p[s] // k, :]` directly into VMEM scratch and computes
both `dx` (2D `(N, D)`) and `dw_sorted` (1D `(N,)` fp32) in a single fused
kernel pass without materializing the `(N, D)` intermediate `g` tensor in HBM.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_LANES = 128


def _combine_kernel(idx_ref, w_ref, x_hbm, o_ref, buf, o_buf, sem, *, tb, k, d):
  """One grid step: tb tokens, tb * k row copies, fp32 weighted sum."""
  n = tb * k

  def start(r, carry):
    pltpu.make_async_copy(x_hbm.at[pl.ds(idx_ref[r], 1)], buf.at[pl.ds(r, 1)], sem.at[0]).start()
    return carry

  jax.lax.fori_loop(0, n, start, 0, unroll=8)

  def wait(r, carry):
    pltpu.make_async_copy(x_hbm.at[pl.ds(0, 1)], buf.at[pl.ds(r, 1)], sem.at[0]).wait()
    return carry

  jax.lax.fori_loop(0, n, wait, 0, unroll=8)

  def token(t, carry):
    base = t * k
    acc = buf[pl.ds(base, 1)].astype(jnp.float32) * w_ref[base]
    for j in range(1, k):
      acc = acc + buf[pl.ds(base + j, 1)].astype(jnp.float32) * w_ref[base + j]
    o_buf[pl.ds(t, 1)] = acc.astype(o_ref.dtype)
    return carry

  jax.lax.fori_loop(0, tb, token, 0, unroll=4)
  o_ref[...] = o_buf[...].reshape(tb, d)


def _block_tokens(t, cap):
  r = min(cap, t)
  while t % r:
    r -= 1
  return r


@functools.partial(jax.jit, static_argnames=("block_tokens", "interpret"))
def combine_fwd(
    x: jax.Array, idx: jax.Array, w: jax.Array, *, block_tokens: int = 64, interpret: bool = False
) -> jax.Array:
  """Returns sum_j w[t, j] * x[idx[t * k + j]] with shape (T, D) and x.dtype."""
  if x.ndim == 3:
    n, d_blocks, lanes = x.shape
    if lanes != _LANES:
      raise ValueError(f"last dim of 3D x ({lanes}) must be {_LANES}")
    d = d_blocks * lanes
    x3 = x
  else:
    n, d = x.shape
    if d % _LANES:
      raise ValueError(f"row size {d} must be a multiple of {_LANES}")
    x3 = x.reshape(n, d // _LANES, _LANES)
  t, k = w.shape
  if idx.shape != (t * k,):
    raise ValueError(f"idx shape {idx.shape} must be ({t * k},)")
  tb = _block_tokens(t, block_tokens)
  idx = jnp.clip(idx.astype(jnp.int32), 0, n - 1)
  w_flat = w.astype(jnp.float32).reshape(t * k)
  out = pl.pallas_call(
      functools.partial(_combine_kernel, tb=tb, k=k, d=d),
      out_shape=jax.ShapeDtypeStruct((t, d), x.dtype),
      grid=(t // tb,),
      in_specs=[
          pl.BlockSpec((tb * k,), lambda i: (i,), memory_space=pltpu.SMEM),
          pl.BlockSpec((tb * k,), lambda i: (i,), memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=pl.BlockSpec((tb, d), lambda i: (i, 0)),
      scratch_shapes=[
          pltpu.VMEM((tb * k, d // _LANES, _LANES), x.dtype),
          pltpu.VMEM((tb, d // _LANES, _LANES), x.dtype),
          pltpu.SemaphoreType.DMA((1,)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="moe_combine",
  )(idx, w_flat, x3)
  return out


def _bwd_block_rows(n, cap):
  if n <= cap:
    return n
  r = (min(cap, n) // _LANES) * _LANES
  while r > _LANES and n % r:
    r -= _LANES
  return r if n % r == 0 else n


def _combine_bwd_kernel(p_ref, w_sorted_ref, x_ref, dy_hbm, dx_ref, dw_ref, g_buf, sem, *, rows, k, d, unroll):
  """One grid step: `rows` sorted rows, DMA gather `dy[p // k]`, compute `dx` and `dw_sorted`."""

  def start(r, carry):
    pltpu.make_async_copy(dy_hbm.at[pl.ds(p_ref[r] // k, 1)], g_buf.at[pl.ds(r, 1)], sem.at[0]).start()
    return carry

  jax.lax.fori_loop(0, rows, start, 0, unroll=unroll)

  def wait(r, carry):
    pltpu.make_async_copy(dy_hbm.at[pl.ds(0, 1)], g_buf.at[pl.ds(r, 1)], sem.at[0]).wait()
    return carry

  jax.lax.fori_loop(0, rows, wait, 0, unroll=unroll)

  g = g_buf[...].astype(jnp.float32)
  x_raw = x_ref[...]
  x = (x_raw if x_raw.ndim == 3 else x_raw.reshape(rows, d // _LANES, _LANES)).astype(jnp.float32)
  w = jax.lax.broadcast_in_dim(w_sorted_ref[...], (rows, 1, 1), (0,))
  dx_ref[...] = (g * w).astype(dx_ref.dtype).reshape(rows, d)
  dw_ref[...] = jnp.sum(x * g, axis=(1, 2))


@functools.partial(jax.jit, static_argnames=("k", "block_rows", "interpret"))
def combine_bwd(
    x: jax.Array,
    dy: jax.Array,
    p: jax.Array,
    w_sorted: jax.Array,
    *,
    k: int,
    block_rows: int = 512,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Fused backward kernel returning `(dx, dw_sorted)` without materializing `g` in HBM."""
  t = dy.shape[0]
  if x.ndim == 3:
    n, d_blocks, lanes = x.shape
    if lanes != _LANES:
      raise ValueError(f"last dim of 3D x ({lanes}) must be {_LANES}")
    d = d_blocks * lanes
  else:
    n, d = x.shape
    if d % _LANES:
      raise ValueError(f"row size {d} must be a multiple of {_LANES}")
  dy3 = dy.astype(x.dtype).reshape(t, d // _LANES, _LANES)
  rows = _bwd_block_rows(n, block_rows)
  x_spec = (
      pl.BlockSpec((rows, d // _LANES, _LANES), lambda i: (i, 0, 0))
      if x.ndim == 3
      else pl.BlockSpec((rows, d), lambda i: (i, 0))
  )
  p_clipped = jnp.clip(p.astype(jnp.int32), 0, n - 1)
  w_sorted_f32 = w_sorted.astype(jnp.float32)
  unroll = 8 if rows % 8 == 0 else 1
  dx, dw_sorted = pl.pallas_call(
      functools.partial(_combine_bwd_kernel, rows=rows, k=k, d=d, unroll=unroll),
      out_shape=[
          jax.ShapeDtypeStruct((n, d), x.dtype),
          jax.ShapeDtypeStruct((n,), jnp.float32),
      ],
      grid=(n // rows,),
      in_specs=[
          pl.BlockSpec((rows,), lambda i: (i,), memory_space=pltpu.SMEM),
          pl.BlockSpec((rows,), lambda i: (i,)),
          x_spec,
          pl.BlockSpec(memory_space=pl.ANY),
      ],
      out_specs=[
          pl.BlockSpec((rows, d), lambda i: (i, 0)),
          pl.BlockSpec((rows,), lambda i: (i,)),
      ],
      scratch_shapes=[
          pltpu.VMEM((rows, d // _LANES, _LANES), x.dtype),
          pltpu.SemaphoreType.DMA((1,)),
      ],
      compiler_params=None if interpret else pltpu.CompilerParams(dimension_semantics=("parallel",)),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="moe_combine_bwd",
  )(p_clipped, w_sorted_f32, x, dy3)
  return dx, dw_sorted


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def unsort_combine(x, idx, w, inv_idx, interpret):
  """Fused `einsum('tkd,tk->td', x[idx].reshape(t, k, d), w)`.

  Args:
    x: (N, D) or (N, D // 128, 128) expert outputs in sorted order.
    idx: (T * k,) int sorted-row index of each (token, slot) pair.
    w: (T, k) router weights.
    inv_idx: (N,) int inverse permutation of idx (p[idx[r]] = r), or None.
    interpret: run the Pallas kernels in interpret mode (CPU tests).

  Returns:
    (T, D) combined output in x.dtype, accumulated in fp32.
  """
  del inv_idx
  return combine_fwd(x, idx, w, interpret=interpret)


def _unsort_combine_fwd(x, idx, w, inv_idx, interpret):
  return combine_fwd(x, idx, w, interpret=interpret), (x, idx, w, inv_idx)


def _unsort_combine_bwd(interpret, res, dy):
  """Backward pass for custom_vjp of unsort_combine."""
  x, idx, w, inv_idx = res
  k = w.shape[1]
  p = inv_idx if inv_idx is not None else jnp.argsort(idx)
  # Permuting a 524,288-element 1D vector in VMEM via `jnp.take` lowers to XLA's
  # `gather_fusion` (~3.53 ms/call = 268 ms/step across 40 MoE layers on v6e),
  # whereas `lax.sort_key_val` by the inverse permutation uses the hardware VPU
  # sort (~0.25-0.35 ms/call = ~24 ms/step across 40 layers).
  _, w_sorted = jax.lax.sort_key_val(idx, w.reshape(-1).astype(jnp.float32), is_stable=False)
  dx, dw_sorted = combine_bwd(x, dy, p, w_sorted, k=k, interpret=interpret)
  _, dw_flat = jax.lax.sort_key_val(p, dw_sorted.astype(w.dtype), is_stable=False)
  dw = dw_flat.reshape(w.shape)
  return dx.reshape(x.shape), None, dw, None


unsort_combine.defvjp(_unsort_combine_fwd, _unsort_combine_bwd)


def fused_unsort_combine(x, idx, w, inv_idx=None):
  """Entry point: Pallas on TPU, interpret mode elsewhere."""
  return unsort_combine(x, idx, w, inv_idx, jax.default_backend() != "tpu")
