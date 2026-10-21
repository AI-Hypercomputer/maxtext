# Copyright 2023-2026 Google LLC
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

"""Fused Pallas TPU kernels for the sub-block KDA core: one kernel per direction for the whole chunk pipeline.

Per (batch x head, chunk) grid step the forward kernel forms, in VMEM,

    scores, kk         intra-chunk matrices (`intra.intra_fwd_rows`, sub-block pivots + exact diagonal)
    m = mask(kk) beta  and its inverse X = (I + m)^-1   (exact block doubling, f32 matmuls at `precision`)
    vb, kb, qg, kc     the WY operands from q, k, v, g, beta and the segment layout
    delta = X vb - (X kb) S,   out = qg S + scores delta,   S <- dl S + kc^T delta

and the backward kernel walks the chunks in reverse (the VJP's forward also saves X, kk and the scores, so the
bwd recomputes only the cheap elementwise operands) and returns the gradients of q, k, v, g and beta summed
over every use, so no per-chunk intermediate or partial gradient goes through HBM. The math is that of
`models/olmoe3._delta_rule_chunked_subblock`; the intermediate cotangents stay f32. The per-chunk decay `dl` is
an input (computed in XLA, which also differentiates it).

Residuals of the VJP (all per batch x head and chunk): the chunk states S (`[dk, dv]`, f32 with `state_f32`,
else bf16), delta (`[C, dv]` bf16), X and kk packed into one `[C, C]` f32 matrix (`_pack_x_kk`: kk is strictly
lower, X is unit lower, so kk + (X - I)^T keeps both exactly) and the scores (`[C, C]` bf16). With
`residual_name` they (and the output) are tagged with `jax.ad_checkpoint.checkpoint_name`, so a remat policy
that saves the name keeps them and the remat recompute skips this kernel entirely. `state_stride` controls the
state residual: 1 saves every chunk's state; 2 saves every other one (even chunks), and the bwd kernel rebuilds
an odd chunk's state from the saved even one and the previous chunk's k, decay and delta (one `[dk, C] x [C, dv]`
matmul); 0 saves none, and `_states_call`, a separate light forward pass, rebuilds them all from the deltas. All
three use the forward's own arithmetic for the recurrence, so they are bit-identical.
"""

import functools

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxtext.kernels.kda import intra as kda_intra

_SUB_BLOCK = 8


def _view(ref):
  return ref.at[(0,) * (len(ref.shape) - 2)]


def _nt(a, b):
  return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _tn(a, b):
  return jax.lax.dot_general(a, b, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _mm(a, b, precision, contract=((1,), (0,))):
  return jax.lax.dot_general(a, b, (contract, ((), ())), preferred_element_type=jnp.float32, precision=precision)


def _inverse(m, eye, precision):
  """(I + m)^-1 for strictly lower `[C, C]` m, by block doubling from 1 x 1 blocks.

  With X_s the block-diagonal inverse over s x s blocks, X_2s = X_s - X_s L_s X_s, where L_s keeps the entries
  of m in the lower-left s x s block of each 2s x 2s diagonal block. This is the exact block inverse at every
  step (the power-series product (I - m)(I + m^2)... is algebraically the same but loses ~3x accuracy in f32,
  because the powers of m grow large before they vanish). The first step needs no matmul: X_2 = I - L_1.
  """
  c = m.shape[0]
  row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  x = eye - jnp.where(((row // 2) == (col // 2)) & (row % 2 == 1) & (col % 2 == 0), m, 0.0)
  s = 2
  while s < c:
    lower_left = ((row // (2 * s)) == (col // (2 * s))) & (((row // s) % 2) == 1) & (((col // s) % 2) == 0)
    x = x - _mm(_mm(x, jnp.where(lower_left, m, 0.0), precision), x, precision)
    s *= 2
  return x


def _tril_ones(c):
  row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  return (row >= col).astype(jnp.float32)


def _tril_cumsum(x, reverse=False):
  """Cumulative sum of `[C, d]` f32 rows (reverse: from the last row), as tril(1) matmuls.

  tril(1) is exact in bf16, so splitting x into three bf16 parts gives the f32 result in three bf16 passes
  (HIGHEST would also split the 0/1 matrix and spend six).
  """
  bf16, f32 = jnp.bfloat16, jnp.float32
  tri = _tril_ones(x.shape[0]).astype(bf16)
  hi = x.astype(bf16)
  rest = x - hi.astype(f32)
  mid = rest.astype(bf16)
  lo = (rest - mid.astype(f32)).astype(bf16)
  contract = ((0,), (0,)) if reverse else ((1,), (0,))
  mm = lambda part: jax.lax.dot_general(tri, part, (contract, ((), ())), preferred_element_type=f32)
  return mm(hi) + mm(mid) + mm(lo)


def _masks(segc_ref, segr_ref, c):
  row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  same = segc_ref[...] == segr_ref[...]
  return row, (row >= col) & same, (row > col) & same


def _chunk_matrices(
    q_ref, k_ref, g_ref, beta_ref, segc_ref, segr_ref, sc_s, kk_s, right_s, precision, rin_s=None, outer=None
):
  """Masked intra-chunk scores and kk, and X = (I + kk beta)^-1, of one chunk (two-level intra with `outer`)."""
  f32 = jnp.float32
  c = g_ref.shape[0]
  if rin_s is not None:
    kda_intra.intra_fwd_rows_two_level(
        q_ref, k_ref, g_ref, sc_s, kk_s, right_s, rin_s, _SUB_BLOCK, outer, jnp.bfloat16
    )
  else:
    kda_intra.intra_fwd_rows(q_ref, k_ref, g_ref, sc_s, kk_s, right_s, _SUB_BLOCK, jnp.bfloat16)
  row, causal, strict = _masks(segc_ref, segr_ref, c)
  scores = jnp.where(causal, sc_s[...], 0.0)
  kk = jnp.where(strict, kk_s[...], 0.0)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  x = _inverse(kk * beta_ref[...], (row == col).astype(f32), precision)
  return scores, kk, x


def _carry_keys(kf, g, seg_c):
  """(e^(g_last - g), kc = k e^(g_last - g)) on the chunk's last segment, zero before it: the state writers."""
  c = g.shape[0]
  last = seg_c == seg_c[c - 1 : c, :]
  ecar = jnp.exp(jnp.where(last, g[c - 1 : c, :] - g, -jnp.inf))
  return ecar, (kf * ecar).astype(jnp.bfloat16)


def _chunk_operands(q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref):
  """The WY operands of one chunk from q, k, v, g, beta and its segment layout."""
  f32, bf16 = jnp.float32, jnp.bfloat16
  seg_c = segc_ref[...]
  beta = beta_ref[...]
  g = g_ref[...]
  eg = jnp.exp(g)
  fp = (seg_c == 0).astype(f32)
  qf, kf = q_ref[...].astype(f32), k_ref[...].astype(f32)
  ecar, kc = _carry_keys(kf, g, seg_c)
  return dict(
      beta=beta,
      eg=eg,
      fp=fp,
      ecar=ecar,
      qf=qf,
      kf=kf,
      vb=(v_ref[...].astype(f32) * beta).astype(bf16),
      kb=(kf * eg * beta * fp).astype(bf16),
      qg=(qf * eg * fp).astype(bf16),
      kc=kc,
  )


def _pack_x_kk(x, kk, row, col):
  """One `[C, C]` f32 matrix holding the strictly lower kk and, transposed into the strict upper part, X - I.

  Exact: X is unit lower triangular (its block-doubling iterates keep a diagonal of exactly 1 and an upper part
  of exact zeros), so the two triangles never overlap and the add only ever adds zero.
  """
  return jnp.where(row > col, kk, 0.0) + jnp.transpose(jnp.where(row > col, x, 0.0))


def _unpack_x_kk(packed, row, col):
  """(X, kk) from `_pack_x_kk`."""
  kk = jnp.where(row > col, packed, 0.0)
  x = jnp.transpose(jnp.where(row < col, packed, 0.0)) + (row == col).astype(packed.dtype)
  return x, kk


def _fwd_kernel(*refs, precision, save_wy, state_stride, state_f32, cumsum, outer, mode="full"):
  """Forward per chunk.

  Outputs: the chunk output, then with `state_stride` the incoming state of every `state_stride`-th chunk, then
  with `save_wy` (the VJP's forward) delta, X and kk (packed) and the scores for the bwd. With `cumsum` the g
  input is the per-step log-decay and the kernel takes its in-chunk cumulative sum. With `outer` the intra-chunk
  matrices use two block levels (`intra.intra_fwd_rows_two_level`).

  `mode` splits the kernel in two: 'wy' computes only the chunk matrices and writes X|kk and the scores (its
  only outputs); 'from_wy' takes them as two more inputs instead of computing them, and with `save_wy` writes
  only delta. Run back to back they compute exactly what 'full' does.
  """
  n_in = 8 + (2 if mode == "from_wy" else 0)
  if mode == "wy":
    n_out = 2
  else:
    n_out = 1 + int(bool(state_stride)) + (1 if mode == "from_wy" else 3) * int(save_wy)
  ins, outs, scratch = refs[:n_in], refs[n_in : n_in + n_out], refs[n_in + n_out :]
  q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref = (_view(r) for r in ins[:8])
  outs = [_view(r) for r in outs]
  s_ref, sc_s, kk_s, right_s = scratch[:4]
  extra = list(scratch[4:])

  if cumsum:
    g_s = extra.pop(0)
    g_s[...] = _tril_cumsum(g_ref[...])
    g_ref = g_s

  f32, bf16 = jnp.float32, jnp.bfloat16
  rin_s = extra.pop(0) if outer else None
  c = g_ref.shape[0]
  row = jax.lax.broadcasted_iota(jnp.int32, (c, 1), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  if mode == "from_wy":
    xk_in, sc_in = (_view(r) for r in ins[8:])
    x, kk = _unpack_x_kk(xk_in[...], row, col)
    scores_bf = sc_in[...]
  else:
    scores, kk, x = _chunk_matrices(
        q_ref, k_ref, g_ref, beta_ref, segc_ref, segr_ref, sc_s, kk_s, right_s, precision, rin_s, outer
    )
    scores_bf = scores.astype(bf16)
  if mode == "wy":
    xk_out, sc_out = outs
    xk_out[...] = _pack_x_kk(x, kk, row, col)
    sc_out[...] = scores_bf
    return
  out_ref = outs.pop(0)
  st_ref = outs.pop(0) if state_stride else None

  @pl.when(pl.program_id(1) == 0)
  def _():
    s_ref[...] = jnp.zeros_like(s_ref)

  o = _chunk_operands(q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref)
  x_bf = x.astype(bf16)
  s = s_ref[...]
  s_bf = s.astype(bf16)
  if state_stride == 1 and st_ref is not None:
    st_ref[...] = s if state_f32 else s_bf
  elif state_stride and st_ref is not None:  # the block stays resident over the stride's chunks and is written back once

    @pl.when(pl.program_id(1) % state_stride == 0)
    def _():
      assert st_ref is not None
      st_ref[...] = s if state_f32 else s_bf

  u = jnp.dot(x_bf, o["vb"], preferred_element_type=f32)
  w = jnp.dot(x_bf, o["kb"], preferred_element_type=f32)
  delta_bf = (u - jnp.dot(w.astype(bf16), s_bf, preferred_element_type=f32)).astype(bf16)
  if save_wy:
    outs.pop(0)[...] = delta_bf
    if mode == "full":
      xk_out, sc_out = outs
      xk_out[...] = _pack_x_kk(x, kk, row, col)
      sc_out[...] = scores_bf
  out = jnp.dot(o["qg"], s_bf, preferred_element_type=f32) + jnp.dot(scores_bf, delta_bf, preferred_element_type=f32)
  out_ref[...] = out.astype(out_ref.dtype)
  new = s * dl_ref[...] + _tn(o["kc"], delta_bf)
  # The bf16 state rounds S between chunks; the f32 state keeps it (its matmuls round their operands, as
  # default-precision f32 matmuls do on TPU).
  s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)


def _states_kernel(*refs, state_f32, cumsum):
  """Chunk states S_c from the saved deltas: the forward's recurrence alone, with the same arithmetic."""
  k_ref, g_ref, segc_ref, dl_ref, de_ref, st_ref = (_view(r) for r in refs[:6])
  s_ref = refs[6]

  @pl.when(pl.program_id(1) == 0)
  def _():
    s_ref[...] = jnp.zeros_like(s_ref)

  if cumsum:
    g_s = refs[7]
    g_s[...] = _tril_cumsum(g_ref[...])
    g_ref = g_s

  f32, bf16 = jnp.float32, jnp.bfloat16
  _, kc = _carry_keys(k_ref[...].astype(f32), g_ref[...], segc_ref[...])
  s = s_ref[...]
  st_ref[...] = s if state_f32 else s.astype(bf16)
  new = s * dl_ref[...] + _tn(kc, de_ref[...])
  s_ref[...] = new if state_f32 else new.astype(bf16).astype(f32)


def _rebuilt_state(st_ref, kp_ref, gp_ref, segp_ref, dlp_ref, dep_ref, st_s, gp_s, state_f32):
  """State of an odd chunk c from the saved state of chunk c - 1 and that chunk's k, decay and delta.

  Same arithmetic as the forward's recurrence step, so the result is bit-identical to the state it carried.
  Leaves the state of the current chunk (saved or rebuilt) in `st_s`.
  """
  f32, bf16 = jnp.float32, jnp.bfloat16
  c_idx = pl.num_programs(1) - 1 - pl.program_id(1)
  st_s[...] = st_ref[...].astype(f32)

  @pl.when(c_idx % 2 == 1)
  def _():
    g = gp_ref
    if gp_s is not None:
      gp_s[...] = _tril_cumsum(gp_ref[...])
      g = gp_s
    _, kc = _carry_keys(kp_ref[...].astype(f32), g[...], segp_ref[...])
    new = st_s[...] * dlp_ref[...] + _tn(kc, dep_ref[...])
    st_s[...] = new if state_f32 else new.astype(bf16).astype(f32)


def _bwd_kernel(*refs, precision, cumsum, outer, state_stride, state_f32):
  """Backward per chunk, last chunk first; with `cumsum` g is the per-step log-decay and so is its gradient.

  With `state_stride` 2 the states input holds only even chunks' states and five more inputs carry the previous
  chunk's k, g, segment column, decay and delta, from which `_rebuilt_state` makes an odd chunk's state.
  """
  n_in = 13 + (5 if state_stride == 2 else 0)
  ins, outs, scratch = refs[:n_in], refs[n_in : n_in + 6], refs[n_in + 6 :]
  q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref, segr_ref, dl_ref, st_ref, de_ref, xk_ref, sc_ref, do_ref = (
      _view(r) for r in ins[:13]
  )
  dq_ref, dk_ref, dv_ref, dg_ref, dbeta_ref, ddl_ref = (_view(r) for r in outs)
  r_ref, ds_s, dkk_s, rexp_s, col_s, dq_s, row_s, piv_s = scratch[:8]
  extra = list(scratch[8:])

  @pl.when(pl.program_id(1) == 0)
  def _():
    r_ref[...] = jnp.zeros_like(r_ref)

  if cumsum:
    g_s = extra.pop(0)
    g_s[...] = _tril_cumsum(g_ref[...])
    g_ref = g_s
  rin_s = extra.pop(0) if outer else None
  if state_stride == 2:
    st_s = extra.pop(0)
    gp_s = extra.pop(0) if cumsum else None
    kp_ref, gp_ref, segp_ref, dlp_ref, dep_ref = (_view(r) for r in ins[13:18])
    _rebuilt_state(st_ref, kp_ref, gp_ref, segp_ref, dlp_ref, dep_ref, st_s, gp_s, state_f32)
    st_ref = st_s

  f32, bf16 = jnp.float32, jnp.bfloat16
  c = g_ref.shape[0]
  row, causal, strict = _masks(segc_ref, segr_ref, c)
  o = _chunk_operands(q_ref, k_ref, v_ref, g_ref, beta_ref, segc_ref)
  x, kk = _unpack_x_kk(xk_ref[...], row, jax.lax.broadcasted_iota(jnp.int32, (1, c), 1))
  x_bf, scores_bf = x.astype(bf16), sc_ref[...]
  states_f, delta = st_ref[...].astype(f32), de_ref[...]
  states = states_f.astype(bf16)
  d_out = do_ref[...].astype(bf16)
  # Readout out = qg S + scores delta.
  dqg = _nt(d_out, states)
  dsc = _nt(d_out, delta)
  g_states = _tn(o["qg"], d_out)
  g_delta = _tn(scores_bf, d_out)
  # Recurrence, carrying r = dL/dS_{c+1} backwards.
  r = r_ref[...]
  r_bf = r.astype(bf16)
  dd = g_delta + jnp.dot(o["kc"], r_bf, preferred_element_type=f32)
  dd_bf = dd.astype(bf16)
  w_bf = jnp.dot(x_bf, o["kb"], preferred_element_type=f32).astype(bf16)
  dw_bf = (-_nt(dd_bf, states)).astype(bf16)
  dkc = _nt(delta, r_bf)
  ddl_ref[...] = jnp.sum(states_f * r, axis=-1, keepdims=True)
  r_ref[...] = g_states + r * dl_ref[...] - _tn(w_bf, dd_bf)
  # u = X vb, w = X kb.
  dx = _nt(dd_bf, o["vb"]) + _nt(dw_bf, o["kb"])
  dvb = _tn(x_bf, dd_bf)
  dkb = _tn(x_bf, dw_bf)
  # X = (I + m)^-1: dm = -X^T dX X^T on the strictly lower part; m = kk * beta.
  dm = -_mm(_mm(x, dx, precision, ((0,), (0,))), x, precision, ((1,), (1,)))
  dm = jnp.where(strict, dm, 0.0)
  dbeta = jnp.sum(dm * kk, axis=1, keepdims=True)
  dkk_s[...] = dm * o["beta"]
  ds_s[...] = jnp.where(causal, dsc, 0.0)
  if outer:
    dq_i, dk_i, dg_i = kda_intra.intra_bwd_compute_two_level(
        q_ref, k_ref, g_ref, ds_s, dkk_s, rexp_s, col_s, dq_s, row_s, piv_s, rin_s, _SUB_BLOCK, outer, bf16
    )
  else:
    dq_i, dk_i, dg_i = kda_intra.intra_bwd_compute(
        q_ref, k_ref, g_ref, ds_s, dkk_s, rexp_s, col_s, dq_s, row_s, piv_s, _SUB_BLOCK, bf16
    )
  # WY operands: vb = v beta, kb = k e^g beta fp, qg = q e^g fp, kc = k e^(g_last - g) on the last segment.
  eg, fp, ecar, qf, kf, beta = o["eg"], o["fp"], o["ecar"], o["qf"], o["kf"], o["beta"]
  kegfp = kf * eg * fp
  dv_ref[...] = (dvb * beta).astype(dv_ref.dtype)
  dbeta_ref[...] = dbeta + jnp.sum(dvb * v_ref[...].astype(f32), axis=1, keepdims=True) + jnp.sum(
      dkb * kegfp, axis=1, keepdims=True
  )
  dq_ref[...] = (dq_i + dqg * eg * fp).astype(dq_ref.dtype)
  dk_ref[...] = (dk_i + dkb * (eg * beta * fp) + dkc * ecar).astype(dk_ref.dtype)
  carry = dkc * kf * ecar
  dg = dg_i + dkb * kegfp * beta + dqg * (qf * eg * fp) - carry
  dg = dg + jnp.where(row == c - 1, jnp.sum(carry, axis=0, keepdims=True), 0.0)
  if cumsum:  # through g = tril(1) ld: dld = tril(1)^T dg, a reverse cumulative sum
    dg = _tril_cumsum(dg, reverse=True)
  dg_ref[...] = dg


def _chunk_of(n, reverse, chunk_map=None):
  """Grid step j -> chunk index (reverse: last chunk first), optionally mapped by `chunk_map`."""
  base = (lambda j: n - 1 - j) if reverse else (lambda j: j)
  return base if chunk_map is None else (lambda j: chunk_map(base(j)))


def _specs(n, heads, reverse, chunk_map=None):
  ch = _chunk_of(n, reverse, chunk_map)
  per = lambda *s: pl.BlockSpec((1, 1) + s, lambda i, j: (i, ch(j), 0, 0))
  seg = lambda *s: pl.BlockSpec((1, 1) + s, lambda i, j: (i // heads, ch(j), 0, 0))
  return per, seg


def _tok_spec(n, c, heads, reverse, chunk_map=None):
  """Chunk (b, n, head h) of a token-major `[B, T, H * d]` array as a `[1, C, d]` block, for grid step (b * H + h, n)."""
  ch = _chunk_of(n, reverse, chunk_map)
  return lambda d: pl.BlockSpec((1, c, d), lambda i, j: (i // heads, ch(j), i % heads))


def _states_rows(n, state_stride):
  return -(-n // state_stride)


_PARAMS = pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))
_PARALLEL = pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))  # no state carried across chunks


def _dims(q, v, beta, heads, token_major):
  """(bh, n, c, dk, dv) from chunked `[BH, N, C, d]` or token-major `[B, T, H * d]` q and v."""
  bh, n, c = beta.shape[:3]
  if token_major:
    return bh, n, c, q.shape[-1] // heads, v.shape[-1] // heads
  return bh, n, c, q.shape[-1], v.shape[-1]


def _fwd_call(
    q, k, v, g, beta, seg_col, seg_row, dl, heads, precision, interpret, save=False, state_f32=False, token_major=False,
    outer=None, out_dtype=jnp.float32, state_stride=None, mode="full", wy=None,
):  # fmt: skip
  """The forward kernel. Returns [out] + [states] (if `state_stride`) + [delta, X|kk, scores] (if `save`).

  `state_stride` defaults to 1 with `save` (every chunk's state for the bwd) and 0 without. `mode='wy'` returns
  only [X|kk, scores]; `mode='from_wy'` takes them as `wy` and returns [out] + [states] + [delta] (if `save`).
  """
  if state_stride is None:
    state_stride = 1 if save else 0
  bh, n, c, dk, dv = _dims(q, v, beta, heads, token_major)
  per, seg = _specs(n, heads, reverse=False)
  tok = _tok_spec(n, c, heads, reverse=False) if token_major else (lambda d: per(c, d))
  in_specs = [tok(dk), tok(dk), tok(dv), tok(dk), per(c, 1), seg(c, 1), seg(1, c), per(dk, 1)]
  args = [q, k, v, g, beta, seg_col, seg_row, dl]
  if mode == "wy":
    out_specs = [per(c, c), per(c, c)]
    out_shape = [jax.ShapeDtypeStruct((bh, n, c, c), dt) for dt in (jnp.float32, jnp.bfloat16)]
    state_stride, save = 0, False
  else:
    out_specs = [tok(dv)]
    out_shape = [
        jax.ShapeDtypeStruct((q.shape[0], q.shape[1], heads * dv) if token_major else (bh, n, c, dv), out_dtype)
    ]
  if mode == "from_wy":
    assert wy is not None
    in_specs += [per(c, c), per(c, c)]
    args += list(wy)
  if state_stride:
    st_per, _ = _specs(n, heads, reverse=False, chunk_map=lambda ci: ci // state_stride)
    out_specs.append(st_per(dk, dv))
    rows = _states_rows(n, state_stride)
    out_shape.append(jax.ShapeDtypeStruct((bh, rows, dk, dv), jnp.float32 if state_f32 else jnp.bfloat16))
  if save:  # delta (bf16), X and kk packed (f32) and the scores (bf16) for the bwd
    out_specs += [per(c, dv)]
    out_shape += [jax.ShapeDtypeStruct((bh, n, c, dv), jnp.bfloat16)]
    if mode == "full":
      out_specs += [per(c, c), per(c, c)]
      out_shape += [jax.ShapeDtypeStruct((bh, n, c, c), dt) for dt in (jnp.float32, jnp.bfloat16)]
  return pl.pallas_call(
      functools.partial(
          _fwd_kernel,
          precision=precision,
          save_wy=save,
          state_stride=state_stride,
          state_f32=state_f32,
          cumsum=token_major,
          outer=outer,
          mode=mode,
      ),
      grid=(bh, n),
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shape,
      scratch_shapes=[
          pltpu.VMEM((dk, dv), jnp.float32),
          pltpu.VMEM((c, c), jnp.float32),
          pltpu.VMEM((c, c), jnp.float32),
          pltpu.VMEM((c, dk), jnp.float32),
      ]
      + [pltpu.VMEM((c, dk), jnp.float32) for _ in range(bool(token_major) + bool(outer))],
      compiler_params=_PARAMS if mode != "wy" else _PARALLEL,
      interpret=interpret,
  )(*args)


def _states_call(k, g, seg_col, dl, delta, heads, interpret, state_f32=False, token_major=False):
  """Chunk states `[BH, N, dk, dv]` from the deltas the forward saved, with the forward's recurrence arithmetic."""
  bh, n, c, dv = delta.shape
  dk = dl.shape[2]
  per, seg = _specs(n, heads, reverse=False)
  tok = _tok_spec(n, c, heads, reverse=False) if token_major else (lambda d: per(c, d))
  return pl.pallas_call(
      functools.partial(_states_kernel, state_f32=state_f32, cumsum=token_major),
      grid=(bh, n),
      in_specs=[tok(dk), tok(dk), seg(c, 1), per(dk, 1), per(c, dv)],
      out_specs=per(dk, dv),
      out_shape=jax.ShapeDtypeStruct((bh, n, dk, dv), jnp.float32 if state_f32 else jnp.bfloat16),
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32)] + [pltpu.VMEM((c, dk), jnp.float32)] * bool(token_major),
      compiler_params=_PARAMS,
      interpret=interpret,
  )(k, g, seg_col, dl, delta)


def _bwd_call(
    q, k, v, g, beta, seg_col, seg_row, dl, states, delta, x_kk, scores, d_out, heads, precision, interpret,
    token_major=False, outer=None, state_stride=1, state_f32=True,
):  # fmt: skip
  """The backward kernel; `states` holds every chunk's state (`state_stride` 1) or every other one's (2)."""
  bh, n, c, dk, dv = _dims(q, v, beta, heads, token_major)
  per, seg = _specs(n, heads, reverse=True)
  tok = _tok_spec(n, c, heads, reverse=True) if token_major else (lambda d: per(c, d))
  shp = lambda t: jax.ShapeDtypeStruct(t.shape, t.dtype)
  st_per, _ = _specs(n, heads, reverse=True, chunk_map=lambda ci: ci // state_stride)
  extra_specs, extra_args, extra_scratch = [], [], []
  if state_stride == 2:
    # Chunk 2 * (c // 2): the previous chunk for odd c, and for even c the chunk the previous grid step already
    # fetched (an unchanged block index is not copied again).
    pick = lambda ci: 2 * (ci // 2)
    p_per, p_seg = _specs(n, heads, reverse=True, chunk_map=pick)
    p_tok = _tok_spec(n, c, heads, reverse=True, chunk_map=pick) if token_major else (lambda d: p_per(c, d))
    extra_specs = [p_tok(dk), p_tok(dk), p_seg(c, 1), p_per(dk, 1), p_per(c, dv)]
    extra_args = [k, g, seg_col, dl, delta]
    extra_scratch = [pltpu.VMEM((dk, dv), jnp.float32)] + [pltpu.VMEM((c, dk), jnp.float32)] * bool(token_major)
  elif state_stride != 1:
    raise ValueError(f"state_stride must be 1 or 2 in the bwd kernel, got {state_stride}")
  return pl.pallas_call(
      functools.partial(
          _bwd_kernel, precision=precision, cumsum=token_major, outer=outer, state_stride=state_stride,
          state_f32=state_f32,
      ),  # fmt: skip
      grid=(bh, n),
      in_specs=[
          tok(dk),
          tok(dk),
          tok(dv),
          tok(dk),
          per(c, 1),
          seg(c, 1),
          seg(1, c),
          per(dk, 1),
          st_per(dk, dv),
          per(c, dv),
          per(c, c),
          per(c, c),
          tok(dv),
      ]
      + extra_specs,
      out_specs=[tok(dk), tok(dk), tok(dv), tok(dk), per(c, 1), per(dk, 1)],
      out_shape=[shp(q), shp(k), shp(v), shp(g), shp(beta), shp(dl)],
      # r, ds, dkk, rexp, col, dq, row, piv (and the cumulative decay)
      scratch_shapes=[pltpu.VMEM((dk, dv), jnp.float32), pltpu.VMEM((c, c), jnp.float32), pltpu.VMEM((c, c), jnp.float32)]
      + [pltpu.VMEM((c, dk), jnp.float32) for _ in range(5 + bool(token_major) + bool(outer))]
      + extra_scratch,
      compiler_params=_PARAMS,
      interpret=interpret,
  )(q, k, v, g, beta, seg_col, seg_row, dl, states, delta, x_kk, scores, d_out, *extra_args)


def _vjp_fwd(args, heads, precision, interpret, state_f32, outer, out_dtype, residual_name, state_stride, token_major):
  """The VJP forward shared by both layouts: run the kernel saving the residuals, and name them for remat.

  A `residual_name` ending in ':wy' names only X|kk and the scores, computed by a separate first kernel
  (`mode='wy'`); the second kernel (`mode='from_wy'`, no intra-chunk matrices or inverse) makes the output,
  delta and states, so under remat only that cheaper kernel runs again.
  """
  q, k, v, g, beta, seg_col, seg_row, dl = args
  if residual_name and residual_name.endswith(":wy"):
    x_kk, scores = (
        checkpoint_name(t, residual_name[:-3])
        for t in _fwd_call(
            q, k, v, g, beta, seg_col, seg_row, dl, heads, precision, interpret,
            mode="wy", state_f32=state_f32, token_major=token_major, outer=outer,
        )
    )
    outs = _fwd_call(
        q, k, v, g, beta, seg_col, seg_row, dl, heads, precision, interpret,
        save=True, out_dtype=out_dtype, state_stride=state_stride,
        mode="from_wy", wy=(x_kk, scores), state_f32=state_f32, token_major=token_major, outer=outer,
    )  # fmt: skip
    out, rest = outs[0], list(outs[1:])
    states = rest.pop(0) if state_stride else None
    return out, (*args, states, rest[0], x_kk, scores)
  outs = _fwd_call(
      q, k, v, g, beta, seg_col, seg_row, dl, heads, precision, interpret,
      save=True, out_dtype=out_dtype, state_stride=state_stride,
      state_f32=state_f32, token_major=token_major, outer=outer,
  )
  out, rest = outs[0], list(outs[1:])
  states = rest.pop(0) if state_stride else None
  delta, x_kk, scores = rest
  if residual_name:
    out, delta, x_kk, scores = (checkpoint_name(t, residual_name) for t in (out, delta, x_kk, scores))
    if states is not None:
      states = checkpoint_name(states, residual_name)
  return out, (*args, states, delta, x_kk, scores)


def _vjp_bwd(heads, precision, interpret, state_f32, outer, state_stride, token_major, res, d_out):
  q, k, v, g, beta, seg_col, seg_row, dl, states, delta, x_kk, scores = res
  if not state_stride:
    states = _states_call(k, g, seg_col, dl, delta, heads, interpret, state_f32=state_f32, token_major=token_major)
    state_stride = 1
  dq, dk, dv, dg, dbeta, ddl = _bwd_call(
      q, k, v, g, beta, seg_col, seg_row, dl, states, delta, x_kk, scores, d_out, heads, precision, interpret,
      token_major=token_major, outer=outer, state_stride=state_stride, state_f32=state_f32,
  )  # fmt: skip
  zero_seg = lambda t: jnp.zeros(t.shape, jax.dtypes.float0)
  return dq, dk, dv, dg, dbeta, zero_seg(seg_col), zero_seg(seg_row), ddl


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14, 15))
def kda_fused(
    q, k, v, g, beta, seg_col, seg_row, dl, heads, precision=jax.lax.Precision.HIGHEST, interpret=False, state_f32=False,
    outer_block=None, out_dtype=jnp.float32, residual_name=None, state_stride=1,
):  # fmt: skip
  """Sub-block KDA output per chunk, `[BH, N, C, dv]` (`out_dtype`), in one kernel.

  Args:
    q, k: `[BH, N, C, dk]` (q scaled, both L2-normalized), bf16 or f32 (the kernel rounds the matmul operands
      to bf16 itself).   v: `[BH, N, C, dv]`.
    g: `[BH, N, C, dk]` f32 cumulative log-decay within each chunk.   beta: `[BH, N, C, 1]` f32.
    seg_col, seg_row: `[B, N, C, 1]` and `[B, N, 1, C]` int32 segment index within each chunk (0 = the
      document carried in from the previous chunk), shared by the heads of a batch row.
    dl: `[BH, N, dk, 1]` f32 per-channel decay over the chunk, 0 where the chunk holds a reset.
    heads: heads per batch row (BH = B * heads).
    precision: matmul precision of the triangular inverse and its VJP.
    interpret: run in interpret mode (CPU tests).
    state_f32: keep the chunk state in f32 between chunks (the default rounds it to bf16, gdn_state_dtype=bfloat16).
    outer_block: rows per outer block of a two-level intra-chunk factorization (None: one level of
      `_SUB_BLOCK` rows); the columns before an outer block share its first row as pivot.
    out_dtype: dtype of the output (the kernel accumulates in f32 and rounds once on the store; bf16 halves the
      output write and the cotangent read, and equals an f32 output cast to bf16 afterwards).
    residual_name: if set, `checkpoint_name` of the output and the VJP residuals (for a remat policy).
    state_stride: which chunk states the VJP forward keeps for the bwd: 1 every chunk's, 2 every other chunk's
      (the bwd kernel rebuilds the rest), 0 none (a separate kernel rebuilds them from the saved deltas).
  """
  del residual_name, state_stride
  return _fwd_call(
      q, k, v, g, beta, seg_col, seg_row, dl, heads, precision, interpret, state_f32=state_f32, outer=outer_block,
      out_dtype=out_dtype,
  )[0]  # fmt: skip


def _kda_fused_fwd(
    q, k, v, g, beta, seg_col, seg_row, dl, heads, precision, interpret, state_f32, outer_block, out_dtype,
    residual_name, state_stride,
):  # fmt: skip
  return _vjp_fwd(
      (q, k, v, g, beta, seg_col, seg_row, dl), heads, precision, interpret, state_f32, outer_block, out_dtype,
      residual_name, state_stride, token_major=False,
  )  # fmt: skip


def _kda_fused_bwd(
    heads, precision, interpret, state_f32, outer_block, out_dtype, residual_name, state_stride, res, d_out
):
  del out_dtype, residual_name
  return _vjp_bwd(heads, precision, interpret, state_f32, outer_block, state_stride, False, res, d_out)


kda_fused.defvjp(_kda_fused_fwd, _kda_fused_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11, 12, 13, 14, 15))
def kda_fused_token_major(
    q, k, v, log_decay, beta, seg_col, seg_row, dl, heads, precision=jax.lax.Precision.HIGHEST, interpret=False,
    state_f32=False, outer_block=None, out_dtype=jnp.float32, residual_name=None, state_stride=1,
):  # fmt: skip
  """`kda_fused` reading and writing the token-major layout: no transposes and no XLA cumulative sum.

  Args:
    q, k: `[B, T, H * dk]`.   v: `[B, T, H * dv]`.
    log_decay: `[B, T, H * dk]` f32 per-step log-decay; the kernel takes its cumulative sum within each chunk.
    beta, seg_col, seg_row, dl, heads, precision, interpret, state_f32, outer_block, out_dtype, residual_name,
      state_stride: as in `kda_fused`.

  Returns:
    `[B, T, H * dv]` in `out_dtype`.
  """
  del residual_name, state_stride
  return _fwd_call(
      q, k, v, log_decay, beta, seg_col, seg_row, dl, heads, precision, interpret, state_f32=state_f32,
      token_major=True, outer=outer_block, out_dtype=out_dtype,
  )[0]  # fmt: skip


def _kda_fused_tm_fwd(
    q, k, v, log_decay, beta, seg_col, seg_row, dl, heads, precision, interpret, state_f32, outer_block, out_dtype,
    residual_name, state_stride,
):  # fmt: skip
  return _vjp_fwd(
      (q, k, v, log_decay, beta, seg_col, seg_row, dl), heads, precision, interpret, state_f32, outer_block,
      out_dtype, residual_name, state_stride, token_major=True,
  )  # fmt: skip


def _kda_fused_tm_bwd(
    heads, precision, interpret, state_f32, outer_block, out_dtype, residual_name, state_stride, res, d_out
):
  del out_dtype, residual_name
  return _vjp_bwd(heads, precision, interpret, state_f32, outer_block, state_stride, True, res, d_out)


kda_fused_token_major.defvjp(_kda_fused_tm_fwd, _kda_fused_tm_bwd)
