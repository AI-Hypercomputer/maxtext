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

"""Pallas GPU (Triton) kernel for the RWKV-7 WKV7 recurrence.

The same recurrence as `maxtext.kernels.rwkv7_wkv` against the same reference
(`maxtext.models.rwkv7.wkv7_naive`), laid out for a GPU rather than the MXU,
following BlinkDL's production CUDA kernel
(`RWKV-LM/RWKV-v7/train_temp/cuda/wkv7_cuda_fp32.cu`):

  * one program per (batch, head) walks the sequence token by token with the
    N x N state in registers, so the state never goes to HBM per step;
  * the forward saves each step's `sa = S_{t-1} a_t` and the state every
    `chunk_size` steps;
  * the backward starts from a chunk's end state and *inverts* steps,
    S_{t-1} = (S_t - v k^T - sa b^T) / w, rather than recomputing them. RWKV-7
    bounds its decay from below -- w = exp(-exp(-0.5) sigmoid(x)) > 0.545, so
    1/w < 1.84 -- which is what makes that inversion usable at all. (The TPU
    kernel recomputes from a checkpoint instead: it has the VMEM to hold a
    chunk's states, and registers on a GPU are scarcer.)

The chunk size is what bounds the inversion's error: it multiplies float32
rounding by up to (1/w)^chunk_size, i.e. 1.834^chunk_size. Measured worst-case
relative gradient deviation from `wkv7_naive` at the strongest decay, over 128
steps: 4.4e-6 at 8 steps per chunk, 4.6e-4 at 16, 7.0 at 32. So the maximum is
8, not BlinkDL's 16, which buys two orders of magnitude of gradient accuracy;
on an RTX 3050 it costs under 2% of forward+backward time, because the chunk
size only sets how much state reaches HBM.

Arrays are read and written in the model's own (B, T, H, N) layout and dtype,
and `a = -kk` and `b = kk * iclr` are formed in registers, so no transposes or
elementwise passes are needed around the kernel. The state and all arithmetic
are float32; gradients come back in each input's dtype.

Packed-sequence resets are not supported: a step whose entering state was
zeroed cannot be inverted back past the reset. Use `pallas_chunked` or
`naive` with `rwkv7_segment_resets`.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton
import jax.numpy as jnp

DEFAULT_CHUNK_SIZE = 8
MAX_CHUNK_SIZE = 8  # the inversion backward's conditioning; see the module docstring


def _outer(x, y):
  """Outer product of two (N,) vectors."""
  return x[:, None] * y[None, :]


def _fwd_kernel(r_ref, w_ref, k_ref, v_ref, a_ref, kk_ref, s0_ref, y_ref, sa_ref, s_end_ref, *, num_chunks, heads, chunk):
  """One (batch, head) program of the forward pass."""
  b_idx, h = pl.program_id(0) // heads, pl.program_id(0) % heads
  f32 = jnp.float32

  def chunk_body(c, s):
    def step(i, s):
      t = c * chunk + i

      def load(ref):
        return ref[b_idx, t, h, :].astype(f32)

      r, w, k, v, iclr, kk = (load(ref) for ref in (r_ref, w_ref, k_ref, v_ref, a_ref, kk_ref))
      sa = jnp.sum(s * (-kk)[None, :], axis=1)
      s = s * w[None, :] + _outer(sa, kk * iclr) + _outer(v, k)
      sa_ref[b_idx, t, h, :] = sa
      y_ref[b_idx, t, h, :] = jnp.sum(s * r[None, :], axis=1).astype(y_ref.dtype)
      return s

    s = lax.fori_loop(0, chunk, step, s)
    s_end_ref[b_idx, h, c, :, :] = s
    return s

  lax.fori_loop(0, num_chunks, chunk_body, s0_ref[b_idx, h, :, :].astype(f32))


def _bwd_kernel(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    a_ref,
    kk_ref,
    sa_ref,
    s_end_ref,
    dy_ref,
    ds_ref,
    dr_ref,
    dw_ref,
    dk_ref,
    dv_ref,
    da_ref,
    dkk_ref,
    ds0_ref,
    *,
    num_chunks,
    heads,
    chunk,
):
  """One (batch, head) program of the backward pass; chunks are walked last-first."""
  b_idx, h = pl.program_id(0) // heads, pl.program_id(0) % heads
  f32 = jnp.float32

  def chunk_body(j, ds):
    c = num_chunks - 1 - j

    def step(i, carry):
      s, ds = carry  # S_t and dL/dS_t
      t = c * chunk + chunk - 1 - i

      def load(ref):
        return ref[b_idx, t, h, :].astype(f32)

      def store(ref, value):
        ref[b_idx, t, h, :] = value.astype(ref.dtype)

      r, w, k, v, iclr, kk, sa, dy = (load(ref) for ref in (r_ref, w_ref, k_ref, v_ref, a_ref, kk_ref, sa_ref, dy_ref))
      a, b = -kk, kk * iclr
      store(dr_ref, jnp.sum(s * dy[:, None], axis=0))
      ds = ds + _outer(dy, r)
      s_prev = (s - _outer(v, k) - _outer(sa, b)) / w[None, :]
      store(dw_ref, jnp.sum(s_prev * ds, axis=0))
      store(dk_ref, jnp.sum(ds * v[:, None], axis=0))
      store(dv_ref, jnp.sum(ds * k[None, :], axis=1))
      db = jnp.sum(ds * sa[:, None], axis=0)
      dsa = jnp.sum(ds * b[None, :], axis=1)
      da_vec = jnp.sum(s_prev * dsa[:, None], axis=0)
      # a = -kk and b = kk * iclr, so d(iclr) = kk . db and d(kk) = -da + iclr * db.
      store(da_ref, kk * db)
      store(dkk_ref, iclr * db - da_vec)
      return s_prev, ds * w[None, :] + _outer(dsa, a)

    _, ds = lax.fori_loop(0, chunk, step, (s_end_ref[b_idx, h, c, :, :], ds))
    return ds

  ds0_ref[b_idx, h, :, :] = lax.fori_loop(0, num_chunks, chunk_body, ds_ref[b_idx, h, :, :].astype(f32))


def _call(kernel, args, out_shapes, num_programs, interpret):
  return pl.pallas_call(
      kernel,
      grid=(num_programs,),
      out_shape=out_shapes,
      compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=1),
      interpret=interpret,
  )(*args)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def _core(r, w, k, v, a, kk, s0, chunk, interpret):
  return _core_fwd(r, w, k, v, a, kk, s0, chunk, interpret)[0]


def _core_fwd(r, w, k, v, a, kk, s0, chunk, interpret):
  """Forward pass, keeping every residual the backward needs."""
  batch, seq_len, heads, n = r.shape
  num_chunks = seq_len // chunk
  y, sa, s_end = _call(
      functools.partial(_fwd_kernel, num_chunks=num_chunks, heads=heads, chunk=chunk),
      (r, w, k, v, a, kk, s0),
      (
          jax.ShapeDtypeStruct((batch, seq_len, heads, n), jnp.float32),
          jax.ShapeDtypeStruct((batch, seq_len, heads, n), jnp.float32),
          jax.ShapeDtypeStruct((batch, heads, num_chunks, n, n), jnp.float32),
      ),
      batch * heads,
      interpret,
  )
  return (y, s_end[:, :, -1]), (r, w, k, v, a, kk, sa, s_end)


def _core_bwd(chunk, interpret, residuals, cotangents):
  """Backward pass; returns gradients for (r, w, k, v, a, kk, state)."""
  r, w, k, v, a, kk, sa, s_end = residuals
  dy, ds_final = cotangents
  batch, seq_len, heads, n = r.shape
  return _call(
      functools.partial(_bwd_kernel, num_chunks=seq_len // chunk, heads=heads, chunk=chunk),
      (r, w, k, v, a, kk, sa, s_end, dy, ds_final),
      tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (r, w, k, v, a, kk))
      + (jax.ShapeDtypeStruct((batch, heads, n, n), jnp.float32),),
      batch * heads,
      interpret,
  )


_core.defvjp(_core_fwd, _core_bwd)


def wkv7_pallas_gpu(
    r: jax.Array,
    w: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    kk: jax.Array,
    state: jax.Array,
    *,
    reset: jax.Array | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    interpret: bool | str | None = None,
) -> tuple[jax.Array, jax.Array]:
  """WKV7 on a GPU; a drop-in for `maxtext.models.rwkv7.wkv7_naive` without resets.

  Args:
    r: `(B, T, H, N)` receptance.
    w: `(B, T, H, N)` per-channel decay in (0, 1).
    k: `(B, T, H, N)` key.
    v: `(B, T, H, N)` value.
    a: `(B, T, H, N)` in-context learning rate.
    kk: `(B, T, H, N)` normalized key.
    state: `(B, H, N, N)` entering state.
    reset: unsupported here; must be None.
    chunk_size: steps between stored states, the backward's starting points.
      At most `MAX_CHUNK_SIZE`: it bounds the inversion backward's error.
    interpret: run the Pallas interpreter instead of Triton. Defaults to
      interpreting off-GPU; any truthy value interprets.

  Returns:
    `(out (B, T, H, N), final_state (B, H, N, N))`, both float32.
  """
  if reset is not None:
    raise NotImplementedError(
        "the RWKV-7 GPU kernel has no packed-sequence resets (a step whose state was zeroed cannot be "
        "inverted); set rwkv7_segment_resets=false, or use rwkv7_wkv_impl=pallas_chunked or naive"
    )
  if chunk_size > MAX_CHUNK_SIZE:
    raise ValueError(
        f"chunk_size must be at most {MAX_CHUNK_SIZE} for the RWKV-7 GPU kernel, got {chunk_size}: its backward "
        "inverts steps, which multiplies rounding by up to (1/w)^chunk_size"
    )
  if interpret is None:
    interpret = jax.default_backend() != "gpu"
  seq_len = r.shape[1]
  chunk = min(chunk_size, seq_len)
  padded_len = -(-seq_len // chunk) * chunk
  if padded_len != seq_len:
    # Padding steps are identities: w = 1 keeps the state, and zero k/v/a/kk add nothing to it.
    def pad(x, pad_value=0.0):
      return jnp.pad(x, ((0, 0), (0, padded_len - seq_len), (0, 0), (0, 0)), constant_values=pad_value)

    r, w, k, v, a, kk = pad(r), pad(w, 1.0), pad(k), pad(v), pad(a), pad(kk)
  y, final_state = _core(r, w, k, v, a, kk, state.astype(jnp.float32), chunk, bool(interpret))
  return y[:, :seq_len], final_state
