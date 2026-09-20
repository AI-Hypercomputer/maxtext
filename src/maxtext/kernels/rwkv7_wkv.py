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

"""Pallas TPU kernels for the RWKV-7 WKV7 recurrence (forward and backward).

Per batch element and head, with an N x N state S (rows index the value
dimension, columns the key dimension):

    S_t = S_{t-1} diag(w_t) + (S_{t-1} a_t) b_t^T + v_t k_t^T
    y_t = S_t r_t

where, in terms of the model's projections, `a_t = -kk_t` and
`b_t = kk_t * iclr_t`. `maxtext.models.rwkv7.wkv7_naive` is the reference this
kernel is tested against.

Two algorithms share the same grid, checkpointing and custom VJP:
"recurrent" (described here) and "chunked", which evaluates each chunk as a
handful of matmuls (derivation in the comment block above `_chunk_forward`).

Design ("recurrent"). This is a step-by-step recurrence over time, the same
algorithm as BlinkDL's production CUDA kernel (`RWKV-LM/RWKV-v7/train_temp/cuda/
wkv7_cuda_fp32.cu`), laid out for Pallas TPU:

  * grid `(B, H, T // chunk)`: batch and head are independent ("parallel");
    the chunk axis is sequential ("arbitrary") and the state is carried across
    it in a VMEM scratch buffer.
  * The forward pass also writes the state entering every chunk
    (`(B, H, T // chunk, N, N)` checkpoints) for the backward pass.
  * The backward pass walks the chunks in reverse. Inside a chunk it first
    recomputes that chunk's states forward from its checkpoint, then walks the
    timesteps backwards accumulating dL/dS. The CUDA kernel instead
    reconstructs S_{t-1} from S_t by dividing by w_t, which loses precision
    as w_t gets small; recomputing from a checkpoint avoids that.

The state, all arithmetic and the output are float32 regardless of input dtype
(matching `wkv7_naive`, whose output is promoted by the float32 state).

Packed sequences: an optional per-step `reset` zeroes the state entering that
step, S_{t-1} <- 0, before it is used (document boundaries in a packed row).

Interpret mode (`interpret=True`, the default off-TPU) proves correctness only;
nothing about this kernel's TPU performance has been measured yet.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_CHUNK_SIZE = 16

_HIGHEST = lax.Precision.HIGHEST


def _row_times_mat(x_row, m):
  """(1, N) @ (N, N) -> (1, N): out[j] = sum_i x[i] m[i, j]."""
  return lax.dot_general(x_row, m, (((1,), (0,)), ((), ())), precision=_HIGHEST, preferred_element_type=jnp.float32)


def _row_times_mat_t(x_row, m):
  """(1, N) @ (N, N)^T -> (1, N): out[i] = sum_j m[i, j] x[j]."""
  return lax.dot_general(x_row, m, (((1,), (1,)), ((), ())), precision=_HIGHEST, preferred_element_type=jnp.float32)


def _outer(x_row, y_row):
  """Outer product of two (1, N) rows: out[i, j] = x[i] y[j]."""
  return jnp.transpose(x_row) * y_row


def _load_row(ref, t):
  return ref[pl.ds(t, 1), :].astype(jnp.float32)


def _step(s, w, k, v, a, b):
  """One recurrence step on an (N, N) state; all vectors are (1, N) rows."""
  sa = _row_times_mat_t(a, s)  # (S a)^T
  return s * w + _outer(sa, b) + _outer(v, k), sa


def _fwd_kernel(r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, reset_ref, s0_ref, y_ref, s_out_ref, ckpt_ref, s_scr, *, chunk):
  """One (batch, head, chunk) program of the forward pass."""

  @pl.when(pl.program_id(2) == 0)
  def _():
    s_scr[...] = s0_ref[...].astype(jnp.float32)

  ckpt_ref[...] = s_scr[...]

  def body(t, s):
    s = s * (1.0 - _load_row(reset_ref, t))  # a reset zeroes the state entering step t
    s, _ = _step(
        s, _load_row(w_ref, t), _load_row(k_ref, t), _load_row(v_ref, t), _load_row(a_ref, t), _load_row(b_ref, t)
    )
    y_ref[pl.ds(t, 1), :] = _row_times_mat_t(_load_row(r_ref, t), s).astype(y_ref.dtype)
    return s

  s = lax.fori_loop(0, chunk, body, s_scr[...])
  s_scr[...] = s
  s_out_ref[...] = s


def _bwd_kernel(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    a_ref,
    b_ref,
    reset_ref,
    dy_ref,
    ckpt_ref,
    ds_out_ref,
    dr_ref,
    dw_ref,
    dk_ref,
    dv_ref,
    da_ref,
    db_ref,
    ds0_ref,
    ds_scr,
    states_scr,
    *,
    chunk,
):
  """One (batch, head, chunk) program of the backward pass; chunks arrive last-first."""

  @pl.when(pl.program_id(2) == 0)
  def _():
    ds_scr[...] = ds_out_ref[...].astype(jnp.float32)

  # Recompute this chunk's states from its checkpoint: states_scr[t] is the
  # state entering step t (before any reset), states_scr[t + 1] the state after it.
  states_scr[0] = ckpt_ref[...]

  def recompute(t, s):
    s = s * (1.0 - _load_row(reset_ref, t))
    s, _ = _step(
        s, _load_row(w_ref, t), _load_row(k_ref, t), _load_row(v_ref, t), _load_row(a_ref, t), _load_row(b_ref, t)
    )
    states_scr[t + 1] = s
    return s

  lax.fori_loop(0, chunk, recompute, states_scr[0])

  def body(i, ds):
    t = chunk - 1 - i
    keep = 1.0 - _load_row(reset_ref, t)
    s_prev, s_cur = states_scr[t] * keep, states_scr[t + 1]
    r, w, k, v = _load_row(r_ref, t), _load_row(w_ref, t), _load_row(k_ref, t), _load_row(v_ref, t)
    a, b, dy = _load_row(a_ref, t), _load_row(b_ref, t), _load_row(dy_ref, t)

    # ds is dL/dS_t from here on: later steps' contribution plus y_t's.
    ds = ds + _outer(dy, r)
    sa = _row_times_mat_t(a, s_prev)
    dsa = _row_times_mat_t(b, ds)

    def store(ref, value):
      ref[pl.ds(t, 1), :] = value.astype(ref.dtype)

    store(dr_ref, _row_times_mat(dy, s_cur))
    store(dk_ref, _row_times_mat(v, ds))
    store(dv_ref, _row_times_mat_t(k, ds))
    store(db_ref, _row_times_mat(sa, ds))
    store(dw_ref, jnp.sum(s_prev * ds, axis=0, keepdims=True))
    store(da_ref, _row_times_mat(dsa, s_prev))

    # Propagate to dL/dS_{t-1} (nothing crosses a reset).
    return (ds * w + _outer(dsa, a)) * keep

  ds = lax.fori_loop(0, chunk, body, ds_scr[...])
  ds_scr[...] = ds
  ds0_ref[...] = ds


# -----------------------------------------------------------------------------
# Chunked (matmul) form.
#
# Within a chunk of L steps entering with state S0, write c for the inclusive
# cumulative log-decay (c_t = sum_{tau<=t} log w_tau, per channel), c- = c -
# log w for the exclusive one, and sa_t = S_{t-1} a_t. Unrolling only the
# diagonal decay,
#
#   S_t = S0 G_t + sum_{s<=t} (sa_s b_s^T + v_s k_s^T) G_t G_s^-1,
#
# with G_t = diag(exp(c_t)), so with time along rows:
#
#   (I - M_ab) SA = A~ S0^T + M_ak V                  (the delta-rule coupling)
#   Y             = R~ S0^T + M_rb SA + M_rk V
#   S_L           = S0 G_L + SA^T B_ + V^T K_
#
# where A~ = A exp(c-), R~ = R exp(c), B_ = B exp(c_L - c), K_ = K exp(c_L - c)
# and the L x L interaction matrices are decayed dot products,
#
#   M_xy[t, s] = sum_j x_t[j] y_s[j] exp(p_t[j] - c_s[j])   (masked to s < t
#                for x = a, where p = c-; to s <= t for x = r, where p = c).
#
# Each M is formed as one matmul of rescaled factors x exp(p - ref) and
# y exp(ref - c), with ref the cumulative decay at the middle of the chunk, so
# neither factor over- or underflows as long as the decay accumulated over half
# a chunk stays well inside float32's exponent range. For RWKV-7 that always
# holds: -log w = exp(-softplus(-x) - 0.5) < e^-0.5, so even a 128-step chunk
# stays under e^39. Beyond ~290 steps it overflows float32 (exp limit ~88), so
# `wkv7_pallas` rejects chunked chunk sizes above MAX_CHUNKED_CHUNK_SIZE, the
# largest size tested at the strongest decay. The factors that don't use ref (A~, R~, B_, K_, G_L) have
# nonpositive exponents and are always safe.
#
# Resets (packed sequences) only delete terms, so they are masks on the same
# factorization: a term from step s reaches row t iff no reset lies in (s, t];
# S0 reaches row t iff none lies in [0, t]; step s reaches the chunk's end state
# iff none lies in (s, L-1]; S0 reaches it iff the chunk has none. The decays
# are unchanged, and the backward applies the same (constant) masks.
#
# (I - M_ab)^-1 is formed by forward substitution, one row at a time from
# already-final rows (2 L^2 work per row). Do not replace it with the
# log-depth product (I + M)(I + M^2)(I + M^4)...: it is exact in exact
# arithmetic (M is nilpotent) but the powers of M grow like binomial
# coefficients, so it cancels catastrophically. With M = -(strictly lower
# ones), the aligned-kk / iclr -> 1 limit, whose inverse is just bidiagonal,
# it is off by ~4e9 relative at L = 64 in float32. (Horner iteration
# T <- I + M T for L steps is accurate but costs L^3 per step.)
# -----------------------------------------------------------------------------


def _mm(x, y):
  """x @ y."""
  return lax.dot_general(x, y, (((1,), (0,)), ((), ())), precision=_HIGHEST, preferred_element_type=jnp.float32)


def _mm_nt(x, y):
  """x @ y^T."""
  return lax.dot_general(x, y, (((1,), (1,)), ((), ())), precision=_HIGHEST, preferred_element_type=jnp.float32)


def _mm_tn(x, y):
  """x^T @ y."""
  return lax.dot_general(x, y, (((0,), (0,)), ((), ())), precision=_HIGHEST, preferred_element_type=jnp.float32)


def _tri_masks(length):
  rows = lax.broadcasted_iota(jnp.int32, (length, length), 0)
  cols = lax.broadcasted_iota(jnp.int32, (length, length), 1)
  return rows, cols, rows > cols, rows >= cols


def _inverse_block_size(length):
  """Block size for `_unit_lower_inverse`: the divisor minimizing its matmul count."""
  divisors = [b for b in (8, 16, 32) if length % b == 0] or [length]
  return min(divisors, key=lambda b: b + 2 * (length // b))


def _unit_lower_inverse(m):
  """(I - m)^-1 for strictly lower-triangular (L, L) m, by block forward substitution.

  Splitting m into its block diagonal m_bd (blocks of `block` rows) and the
  rest m_off, the row recurrence Inv = I + m Inv restricted to block row i is
  (I - m_ii) Inv_i = E_i + m_off_i Inv, i.e.

      Inv_i = D_i (E_i + m_off_i Inv),     D = (I - m_bd)^-1 = blockdiag(D_i),

  so one dependent step per block row instead of per row. D is `block` Horner
  steps T <- I + m_bd T, exact because m_bd is nilpotent of index `block`
  (and Horner, unlike the log-depth product warned about above, does not
  cancel). That is `block` + 2 L / `block` full (L, L) matmuls in place of
  2 L single-row ones -- 24 instead of 128 at L = 64 -- which matters on the
  MXU, where a one-row operand wastes the array. It is the same inverse to
  float32 rounding: the largest relative difference from building it a row at
  a time is 1.5e-8 over chunks of 16, 64 and 128, and exactly zero in the
  aligned-kk / iclr -> 1 limit, where the inverse is worst conditioned.
  """
  length = m.shape[0]
  block = _inverse_block_size(length)
  rows, cols, _, _ = _tri_masks(length)
  eye = (rows == cols).astype(jnp.float32)
  # `lax.div`, not `//`: floor division of signed integers lowers through
  # `sign`, which Mosaic cannot lower. Both operands here are iotas, so
  # truncating division is the same thing.
  row_block, col_block = lax.div(rows, block), lax.div(cols, block)
  m_bd = jnp.where(row_block == col_block, m, 0.0)
  m_off = m - m_bd
  d = lax.fori_loop(0, block, lambda _, t: eye + _mm(m_bd, t), eye)

  def body(i, inv):
    return jnp.where(row_block == i, _mm(d, eye + _mm(m_off, inv)), inv)

  return lax.fori_loop(0, length // block, body, eye)


class _ChunkTerms:
  """Every intermediate of one chunk's forward pass (recomputed in the backward)."""

  def __init__(self, s0, r, w, k, v, a, b, reset):
    length = r.shape[0]
    rows, cols, strict, incl = _tri_masks(length)
    incl_f = incl.astype(jnp.float32)
    # Reset masks, from the (L, 1) reset column using only matmuls and iotas.
    # crossings[t, s] = number of resets r with s < r <= t.
    crossings = _mm(incl_f, reset * strict.astype(jnp.float32))
    same_segment = crossings == 0
    self.strict, self.incl = strict & same_segment, incl & same_segment
    self.from_s0 = (_mm(incl_f, reset) == 0).astype(jnp.float32)  # (L, 1): no reset in [0, t]
    self.to_end = (_mm((cols > rows).astype(jnp.float32), reset) == 0).astype(jnp.float32)  # (L, 1): none in (s, L-1]
    s0_to_end = (jnp.sum(reset, axis=0, keepdims=True) == 0).astype(jnp.float32)  # (1, 1)

    self.log_w = jnp.log(w)
    self.c = _mm(incl_f, self.log_w)  # inclusive cumsum over time
    mid = (length - 1) // 2
    ref = self.c[mid : mid + 1, :]  # outputs don't depend on ref; no gradient

    # All six decay factors are products of exp(c - ref), its reciprocal,
    # exp(ref) and 1 / w, so two exponentials (one (L, N), one (1, N)) and two
    # reciprocals replace six (L, N) exponentials. Every intermediate is still
    # centered on `ref`, so the half-a-chunk range argument above is unchanged:
    # `p` and `inv_p` stay inside exp(+-39) and the products below are <= 1.
    p = jnp.exp(self.c - ref)
    inv_p, inv_w, e_ref = 1.0 / p, 1.0 / w, jnp.exp(ref)
    p_last = p[length - 1 :, :]
    exp_incl = p * e_ref  # exp(c)
    self.exp_excl, self.exp_incl = exp_incl * inv_w * self.from_s0, exp_incl * self.from_s0
    self.exp_to_end = p_last * inv_p * self.to_end
    self.g_last = p_last * e_ref * s0_to_end
    self.scale_a = p * inv_w
    self.scale_r = p
    self.scale_key = inv_p
    self.a_s, self.r_s = a * self.scale_a, r * self.scale_r
    self.b_s, self.k_s = b * self.scale_key, k * self.scale_key

    zero = jnp.zeros((length, length), jnp.float32)
    self.m_ab = jnp.where(self.strict, _mm_nt(self.a_s, self.b_s), zero)
    self.m_ak = jnp.where(self.strict, _mm_nt(self.a_s, self.k_s), zero)
    self.m_rb = jnp.where(self.incl, _mm_nt(self.r_s, self.b_s), zero)
    self.m_rk = jnp.where(self.incl, _mm_nt(self.r_s, self.k_s), zero)

    self.a_t, self.r_t = a * self.exp_excl, r * self.exp_incl
    self.b_e, self.k_e = b * self.exp_to_end, k * self.exp_to_end
    self.inv = _unit_lower_inverse(self.m_ab)
    self.rhs = _mm_nt(self.a_t, s0) + _mm(self.m_ak, v)
    self.sa = _mm(self.inv, self.rhs)


def _chunk_forward(s0, r, w, k, v, a, b, reset):
  """One chunk: `(L, N)` float32 rows, `(L, 1)` resets, `(N, N)` entering state -> (y, end state)."""
  t = _ChunkTerms(s0, r, w, k, v, a, b, reset)
  y = _mm_nt(t.r_t, s0) + _mm(t.m_rb, t.sa) + _mm(t.m_rk, v)
  s_end = s0 * t.g_last + _mm_tn(t.sa, t.b_e) + _mm_tn(v, t.k_e)
  return y, s_end


def _chunk_backward(s0, r, w, k, v, a, b, reset, dy, ds_end):
  """VJP of `_chunk_forward` (resets are constants): returns (dr, dw, dk, dv, da, db, ds0)."""
  t = _ChunkTerms(s0, r, w, k, v, a, b, reset)
  zero = jnp.zeros_like(t.m_ab)

  # S_L = S0 G_L + SA^T B_ + V^T K_ (G_L, B_, K_ carry the reset masks)
  g_last = t.g_last
  ds0 = ds_end * g_last
  dc_last = jnp.sum(s0 * ds_end, axis=0, keepdims=True) * g_last
  dsa = _mm_nt(t.b_e, ds_end)
  db_e = _mm(t.sa, ds_end)
  dv = _mm_nt(t.k_e, ds_end)
  dk_e = _mm(v, ds_end)

  # Y = R~ S0^T + M_rb SA + M_rk V
  dr_t = _mm(dy, s0)
  ds0 += _mm_tn(dy, t.r_t)
  dm_rb = jnp.where(t.incl, _mm_nt(dy, t.sa), zero)
  dm_rk = jnp.where(t.incl, _mm_nt(dy, v), zero)
  dsa += _mm_tn(t.m_rb, dy)
  dv += _mm_tn(t.m_rk, dy)

  # SA = (I - M_ab)^-1 RHS: dRHS = inv^T dSA, dM_ab = dRHS SA^T.
  drhs = _mm_tn(t.inv, dsa)
  dm_ab = jnp.where(t.strict, _mm_nt(drhs, t.sa), zero)

  # RHS = A~ S0^T + M_ak V
  da_t = _mm(drhs, s0)
  ds0 += _mm_tn(drhs, t.a_t)
  dm_ak = jnp.where(t.strict, _mm_nt(drhs, v), zero)
  dv += _mm_tn(t.m_ak, drhs)

  # M_xy = x_s y_s^T (masks already applied to the dM's).
  da_s = _mm(dm_ab, t.b_s) + _mm(dm_ak, t.k_s)
  dr_s = _mm(dm_rb, t.b_s) + _mm(dm_rk, t.k_s)
  db_s = _mm_tn(dm_ab, t.a_s) + _mm_tn(dm_rb, t.r_s)
  dk_s = _mm_tn(dm_ak, t.a_s) + _mm_tn(dm_rk, t.r_s)

  # Undo the per-row decay factors; collect gradients of the cumulative decays.
  da = da_t * t.exp_excl + da_s * t.scale_a
  dr = dr_t * t.exp_incl + dr_s * t.scale_r
  db = db_e * t.exp_to_end + db_s * t.scale_key
  dk = dk_e * t.exp_to_end + dk_s * t.scale_key
  dc_excl = da_t * t.a_t + da_s * t.a_s
  dc = dr_t * t.r_t + dr_s * t.r_s - db_e * t.b_e - db_s * t.b_s - dk_e * t.k_e - dk_s * t.k_s
  dc_last += jnp.sum(db_e * t.b_e + dk_e * t.k_e, axis=0, keepdims=True)

  # c- = c - log w, c_last = c[L-1], c = cumsum(log w).
  length = r.shape[0]
  last_row = lax.broadcasted_iota(jnp.int32, (length, 1), 0) == length - 1
  dc = dc + dc_excl + jnp.where(last_row, dc_last, 0.0)
  dlog_w = _mm_tn(t.incl.astype(jnp.float32), dc) - dc_excl  # reverse cumsum
  return dr, dlog_w / w, dk, dv, da, db, ds0


def _chunked_fwd_kernel(r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, reset_ref, s0_ref, y_ref, s_out_ref, ckpt_ref, s_scr):
  """One (batch, head, chunk) program of the chunked forward pass."""

  @pl.when(pl.program_id(2) == 0)
  def _():
    s_scr[...] = s0_ref[...].astype(jnp.float32)

  s0 = s_scr[...]
  ckpt_ref[...] = s0
  rows = [ref[...].astype(jnp.float32) for ref in (r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, reset_ref)]
  y, s_end = _chunk_forward(s0, *rows)
  y_ref[...] = y.astype(y_ref.dtype)
  s_scr[...] = s_end
  s_out_ref[...] = s_end


def _chunked_bwd_kernel(
    r_ref,
    w_ref,
    k_ref,
    v_ref,
    a_ref,
    b_ref,
    reset_ref,
    dy_ref,
    ckpt_ref,
    ds_out_ref,
    dr_ref,
    dw_ref,
    dk_ref,
    dv_ref,
    da_ref,
    db_ref,
    ds0_ref,
    ds_scr,
):
  """One (batch, head, chunk) program of the chunked backward pass; chunks arrive last-first."""

  @pl.when(pl.program_id(2) == 0)
  def _():
    ds_scr[...] = ds_out_ref[...].astype(jnp.float32)

  rows = [ref[...].astype(jnp.float32) for ref in (r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, reset_ref, dy_ref)]
  *grads, ds0 = _chunk_backward(ckpt_ref[...], *rows, ds_scr[...])
  for ref, grad in zip((dr_ref, dw_ref, dk_ref, dv_ref, da_ref, db_ref), grads):
    ref[...] = grad.astype(ref.dtype)
  ds_scr[...] = ds0
  ds0_ref[...] = ds0


ALGORITHMS = ("recurrent", "chunked")

# Largest chunk the "chunked" algorithm accepts: its rescaling factors reach
# exp(decay accumulated over half a chunk), which at RWKV-7's strongest decay
# overflows float32 for chunks of roughly 290 steps or more. 128 is the largest
# size tested at that decay (`test_strongest_decay_large_chunk`).
MAX_CHUNKED_CHUNK_SIZE = 128


def _specs(num_chunks, chunk, n, reverse):
  """BlockSpecs for per-timestep (B, H, T, N) arrays, per-chunk checkpoints, and per-(B, H) states."""

  def chunk_index(c):
    return num_chunks - 1 - c if reverse else c

  seq = pl.BlockSpec((None, None, chunk, n), lambda bi, hi, c: (bi, hi, chunk_index(c), 0))
  ckpt = pl.BlockSpec((None, None, None, n, n), lambda bi, hi, c: (bi, hi, chunk_index(c), 0, 0))
  state = pl.BlockSpec((None, None, n, n), lambda bi, hi, c: (bi, hi, 0, 0))
  reset = pl.BlockSpec((None, chunk, 1), lambda bi, hi, c: (bi, chunk_index(c), 0))  # (B, T, 1), shared by heads
  return seq, ckpt, state, reset


def _compiler_params():
  return pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"))


def _interpret_arg(interpret):
  if isinstance(interpret, pltpu.InterpretParams):
    return interpret
  if interpret == "discharge":
    return True  # Pallas's generic discharge interpreter
  return pltpu.InterpretParams() if interpret else False


def _forward(r, w, k, v, a, b, reset, s0, *, chunk, interpret, algorithm):
  """Kernel-layout forward: (B, H, T, N) inputs and (B, T, 1) resets with T a multiple of `chunk`."""
  batch, heads, seq_len, n = r.shape
  num_chunks = seq_len // chunk
  seq, ckpt, state, reset_spec = _specs(num_chunks, chunk, n, reverse=False)
  if algorithm == "chunked":
    kernel, name = _chunked_fwd_kernel, "rwkv7_wkv_chunked_fwd"
  else:
    kernel, name = functools.partial(_fwd_kernel, chunk=chunk), "rwkv7_wkv_fwd"
  return pl.pallas_call(
      kernel,
      grid=(batch, heads, num_chunks),
      in_specs=[seq] * 6 + [reset_spec, state],
      out_specs=[seq, state, ckpt],
      out_shape=[
          jax.ShapeDtypeStruct(r.shape, jnp.float32),
          jax.ShapeDtypeStruct((batch, heads, n, n), jnp.float32),
          jax.ShapeDtypeStruct((batch, heads, num_chunks, n, n), jnp.float32),
      ],
      scratch_shapes=[pltpu.VMEM((n, n), jnp.float32)],
      compiler_params=_compiler_params(),
      interpret=_interpret_arg(interpret),
      name=name,
  )(r, w, k, v, a, b, reset, s0)


def _backward(r, w, k, v, a, b, reset, dy, ckpts, ds_out, *, chunk, interpret, algorithm):
  """Kernel-layout backward; returns grads for (r, w, k, v, a, b, s0)."""
  batch, heads, seq_len, n = r.shape
  num_chunks = seq_len // chunk
  seq, ckpt, state, reset_spec = _specs(num_chunks, chunk, n, reverse=True)
  if algorithm == "chunked":
    kernel, name = _chunked_bwd_kernel, "rwkv7_wkv_chunked_bwd"
    scratch = [pltpu.VMEM((n, n), jnp.float32)]
  else:
    kernel, name = functools.partial(_bwd_kernel, chunk=chunk), "rwkv7_wkv_bwd"
    scratch = [pltpu.VMEM((n, n), jnp.float32), pltpu.VMEM((chunk + 1, n, n), jnp.float32)]
  grads = pl.pallas_call(
      kernel,
      grid=(batch, heads, num_chunks),
      in_specs=[seq] * 6 + [reset_spec, seq, ckpt, state],
      out_specs=[seq] * 6 + [state],
      out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (r, w, k, v, a, b)]
      + [jax.ShapeDtypeStruct((batch, heads, n, n), jnp.float32)],
      scratch_shapes=scratch,
      compiler_params=_compiler_params(),
      interpret=_interpret_arg(interpret),
      name=name,
  )(r, w, k, v, a, b, reset, dy, ckpts, ds_out)
  return tuple(grads)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10))
def _wkv7_core(r, w, k, v, a, b, reset, s0, chunk, interpret, algorithm):
  y, s_final, _ = _forward(r, w, k, v, a, b, reset, s0, chunk=chunk, interpret=interpret, algorithm=algorithm)
  return y, s_final


def _wkv7_core_fwd(r, w, k, v, a, b, reset, s0, chunk, interpret, algorithm):
  y, s_final, ckpts = _forward(r, w, k, v, a, b, reset, s0, chunk=chunk, interpret=interpret, algorithm=algorithm)
  return (y, s_final), (r, w, k, v, a, b, reset, ckpts)


def _wkv7_core_bwd(chunk, interpret, algorithm, residuals, cotangents):
  r, w, k, v, a, b, reset, ckpts = residuals
  dy, ds_final = cotangents
  dr, dw, dk, dv, da, db, ds0 = _backward(
      r, w, k, v, a, b, reset, dy, ckpts, ds_final, chunk=chunk, interpret=interpret, algorithm=algorithm
  )
  return dr, dw, dk, dv, da, db, jnp.zeros_like(reset), ds0


_wkv7_core.defvjp(_wkv7_core_fwd, _wkv7_core_bwd)


def wkv7_pallas(
    r: jax.Array,
    w: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    kk: jax.Array,
    state: jax.Array,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    interpret: bool | str | pltpu.InterpretParams | None = None,
    algorithm: str = "recurrent",
    reset: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """WKV7 recurrence as a Pallas kernel; drop-in for `rwkv7.wkv7_naive`.

  Differentiable with respect to every input, including `state`.

  Args:
    r, w, k, v, a, kk: `(B, T, H, N)` per-timestep projections. `w` is the
      already-exponentiated decay in (0, 1) and `a` the in-context learning
      rate; the kernel's removal/replacement vectors are `-kk` and `kk * a`.
    state: `(B, H, N, N)` incoming recurrent state (rows = output dim).
    chunk_size: timesteps per grid step, a multiple of 8 (the TPU sublane
      tile), and at most MAX_CHUNKED_CHUNK_SIZE for the chunked algorithm.
      Sequences longer than this are padded up to a multiple of it with
      identity steps; shorter ones run as a single chunk of their own length
      (so single-token decode does no extra work).
    interpret: run in Pallas TPU interpret mode (True, or `InterpretParams` for
      e.g. race detection). Defaults to True unless the default JAX backend is
      a TPU. "discharge" selects Pallas's generic discharge interpreter
      instead: it doesn't model TPU memory semantics as strictly, but it has
      no IO callbacks, so it can be differentiated inside `jax.checkpoint`
      (which the TPU interpreter's ordered effects cannot).
    algorithm: "recurrent" steps through each chunk one token at a time
      (vector ops); "chunked" evaluates each chunk as a few matmuls (see the
      chunked-form notes above `_chunk_forward`). Same results, same
      checkpointing; which one is faster on TPU is unmeasured.
    reset: optional `(B, T)` bool; True at step t zeroes the state entering it
      (a document boundary in a packed row). Not differentiated.

  Returns:
    `(out, final_state)`: `out` is `(B, T, H, N)` and `final_state` is
    `(B, H, N, N)`, both float32.
  """
  if algorithm not in ALGORITHMS:
    raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
  if chunk_size % 8:
    raise ValueError(f"chunk_size must be a multiple of 8 (TPU sublane tile), got {chunk_size}")
  if algorithm == "chunked" and chunk_size > MAX_CHUNKED_CHUNK_SIZE:
    raise ValueError(
        f"chunk_size must be at most {MAX_CHUNKED_CHUNK_SIZE} for the chunked algorithm (its decay rescaling"
        f" overflows float32 for much larger chunks), got {chunk_size}"
    )
  if interpret is None:
    interpret = jax.default_backend() != "tpu"
  _, seq_len, _, _ = r.shape
  if algorithm == "recurrent":
    # The recurrent kernel loads and stores one timestep row at a time, at a
    # dynamic offset. Mosaic can prove that aligned for float32 (8 rows per
    # sublane tile) but not for packed 16-bit refs (16 rows per tile), so it
    # rejects bf16 inputs (E2003 CompileTimeMosaicUnprovenMemoryAccessAlignment,
    # found by compiling for v5e with libtpu). The kernel computes in float32
    # anyway; autodiff casts these inputs' gradients back to their dtype.
    r, v = r.astype(jnp.float32), v.astype(jnp.float32)

  chunk = min(chunk_size, seq_len)
  padded_len = -(-seq_len // chunk) * chunk

  def to_kernel_layout(x, pad_value=0.0):
    x = jnp.swapaxes(x, 1, 2)  # (B, H, T, N)
    if padded_len != seq_len:
      # Padding steps are identities: w = 1 keeps the state, and zero
      # k/v/a/b add nothing to it.
      x = jnp.pad(x, ((0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)), constant_values=pad_value)
    return x

  batch = r.shape[0]
  reset_col = jnp.zeros((batch, seq_len), jnp.float32) if reset is None else reset.astype(jnp.float32)
  reset_col = jnp.pad(reset_col, ((0, 0), (0, padded_len - seq_len)))[..., None]  # (B, T, 1); padding never resets
  y, final_state = _wkv7_core(
      to_kernel_layout(r),
      to_kernel_layout(w, pad_value=1.0),
      to_kernel_layout(k),
      to_kernel_layout(v),
      to_kernel_layout(-kk),
      to_kernel_layout(kk * a),
      reset_col,
      state.astype(jnp.float32),
      chunk,
      interpret,
      algorithm,
  )
  return jnp.swapaxes(y[:, :, :seq_len], 1, 2), final_state
