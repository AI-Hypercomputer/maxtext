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

"""Pallas TPU kernels for the KDA intra-chunk score and key-key matrices (forward and backward).

For one chunk of C rows with cumulative log-decay g (non-increasing in time), it forms

    scores_ij = sum_d q_id k_jd exp(g_id - g_jd),   kk_ij = sum_d k_id k_jd exp(g_id - g_jd)   (j <= i)

as `models/olmoe3._kda_intra_chunk` does, with the same sub-block scheme: rows go in blocks of `sub_block`;
against earlier columns a row's decay factors through its block's first row p, which turns those entries into
matmuls whose exponents are all <= 0, and only the diagonal blocks need the exact per-pair factor. XLA
materializes the per-pair tensor and the decayed right operand in HBM; here both live in VMEM, the right operand
is updated block to block (one per-channel factor, no exp per entry), and the diagonal blocks run on 8-row
tiles. Upper-triangle entries come out as zero, like the XLA version. The per-pair products are formed in f32,
so the result is closer to an f32 reference than XLA's (which rounds the pair tensor to bf16).

The backward computes the gradients of `models/olmoe3._kda_intra_chunk_vjp` (three shared contractions per
chunk, dg = q*DQ + k*(DKrow - DKcol)) with the same block scheme, keeping the column accumulators in VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _nt(a, b):  # a b^T, contracting the trailing axis of both
  return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _tn(a, b):  # a^T b, contracting the leading axis of both
  return jax.lax.dot_general(a, b, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _chunk_view(ref):
  """The `[C, d]` (or `[C, C]`) chunk of a block, whatever its leading unit dims."""
  return ref.at[(0,) * (len(ref.shape) - 2)]


def intra_fwd_rows(q_ref, k_ref, g_ref, s_ref, kk_ref, right_ref, sub_block, compute_dtype):
  """Writes the intra-chunk scores and kk of one chunk into `s_ref`, `kk_ref` (`[C, C]`); 2-D refs throughout.

  `right_ref` is a `[C, d]` f32 VMEM scratch.
  """
  f32, cd = jnp.float32, compute_dtype
  c = g_ref.shape[0]
  sb = sub_block
  irow = jax.lax.broadcasted_iota(jnp.int32, (sb, 1), 0)
  jcol = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  # right_ref row j holds k_j exp(g_p - g_j) for the current block's pivot p, zero for j >= p.
  right_ref[...] = jnp.zeros_like(right_ref)
  for r in range(c // sb):
    p = r * sb
    gr = g_ref[p : p + sb, :]
    qr = q_ref[p : p + sb, :].astype(f32)
    kr = k_ref[p : p + sb, :].astype(f32)
    gp = gr[0:1, :]
    if r > 0:
      pp = p - sb
      if r > 1:  # move the pivot from pp to p: one factor per channel, all <= 1
        right_ref[0:pp, :] = right_ref[0:pp, :] * jnp.exp(gp - g_ref[pp : pp + 1, :])
      right_ref[pp:p, :] = k_ref[pp:p, :].astype(f32) * jnp.exp(gp - g_ref[pp:p, :])
      left = jnp.exp(gr - gp)
      right = right_ref[...].astype(cd)
      row_q = _nt((qr * left).astype(cd), right)
      row_k = _nt((kr * left).astype(cd), right)
    else:
      row_q = jnp.zeros((sb, c), f32)
      row_k = jnp.zeros((sb, c), f32)
    # Diagonal block, exact per pair; masked before the exp because upper pairs can be +inf.
    for jj in range(sb):
      e = jnp.exp(jnp.where(irow >= jj, gr - gr[jj : jj + 1, :], -jnp.inf))
      t = e * kr[jj : jj + 1, :]
      sel = jcol == (p + jj)
      row_q = jnp.where(sel, jnp.sum(qr * t, axis=1, keepdims=True), row_q)
      row_k = jnp.where(sel, jnp.sum(kr * t, axis=1, keepdims=True), row_k)
    s_ref[p : p + sb, :] = row_q
    kk_ref[p : p + sb, :] = row_k


def intra_fwd_rows_two_level(q_ref, k_ref, g_ref, s_ref, kk_ref, right_ref, rin_ref, sub_block, outer, compute_dtype):
  """`intra_fwd_rows` with the two block levels of `intra_bwd_compute_two_level`; `rin_ref` is one more
  `[C, d]` f32 scratch."""
  f32, cd = jnp.float32, compute_dtype
  c = g_ref.shape[0]
  sb = sub_block
  irow = jax.lax.broadcasted_iota(jnp.int32, (sb, 1), 0)
  jcol = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  right_ref[...] = jnp.zeros_like(right_ref)
  rin_ref[...] = jnp.zeros_like(rin_ref)
  for ob in range(c // outer):
    P = ob * outer
    g_o = g_ref[P : P + outer, :]
    gP = g_o[0:1, :]
    if ob > 0:  # columns before the block, through pivot P
      PP = P - outer
      if ob > 1:
        right_ref[0:PP, :] = right_ref[0:PP, :] * jnp.exp(gP - g_ref[PP : PP + 1, :])
      right_ref[PP:P, :] = k_ref[PP:P, :].astype(f32) * jnp.exp(gP - g_ref[PP:P, :])
      left = jnp.exp(g_o - gP)
      right = right_ref[...].astype(cd)
      s_ref[P : P + outer, :] = _nt((q_ref[P : P + outer, :].astype(f32) * left).astype(cd), right)
      kk_ref[P : P + outer, :] = _nt((k_ref[P : P + outer, :].astype(f32) * left).astype(cd), right)
    else:
      s_ref[P : P + outer, :] = jnp.zeros((outer, c), f32)
      kk_ref[P : P + outer, :] = jnp.zeros((outer, c), f32)
    if ob > 0:
      rin_ref[P - outer : P, :] = jnp.zeros((outer, rin_ref.shape[1]), f32)
    for r in range(outer // sb):
      p = P + r * sb
      gr = g_ref[p : p + sb, :]
      qr = q_ref[p : p + sb, :].astype(f32)
      kr = k_ref[p : p + sb, :].astype(f32)
      gp = gr[0:1, :]
      row_q = s_ref[p : p + sb, :]
      row_k = kk_ref[p : p + sb, :]
      if r > 0:  # columns inside the block, through the sub-block's pivot p
        pp = p - sb
        if r > 1:
          rin_ref[P:pp, :] = rin_ref[P:pp, :] * jnp.exp(gp - g_ref[pp : pp + 1, :])
        rin_ref[pp:p, :] = k_ref[pp:p, :].astype(f32) * jnp.exp(gp - g_ref[pp:p, :])
        left = jnp.exp(gr - gp)
        rin = rin_ref[...].astype(cd)  # zero outside [P, p)
        row_q = row_q + _nt((qr * left).astype(cd), rin)
        row_k = row_k + _nt((kr * left).astype(cd), rin)
      for jj in range(sb):
        e = jnp.exp(jnp.where(irow >= jj, gr - gr[jj : jj + 1, :], -jnp.inf))
        t = e * kr[jj : jj + 1, :]
        sel = jcol == (p + jj)
        row_q = jnp.where(sel, jnp.sum(qr * t, axis=1, keepdims=True), row_q)
        row_k = jnp.where(sel, jnp.sum(kr * t, axis=1, keepdims=True), row_k)
      s_ref[p : p + sb, :] = row_q
      kk_ref[p : p + sb, :] = row_k


def _fwd_kernel(q_ref, k_ref, g_ref, s_ref, kk_ref, right_ref, *, sub_block, compute_dtype):
  q_ref, k_ref, g_ref, s_ref, kk_ref = (_chunk_view(r) for r in (q_ref, k_ref, g_ref, s_ref, kk_ref))
  intra_fwd_rows(q_ref, k_ref, g_ref, s_ref, kk_ref, right_ref, sub_block, compute_dtype)


@functools.partial(jax.jit, static_argnames=("sub_block", "compute_dtype", "interpret"))
def kda_intra_fwd(q, k, g, sub_block=8, compute_dtype=jnp.bfloat16, interpret=False):
  """Intra-chunk (scores, kk), each `[BH, N, C, C]` f32, from q, k `[BH, N, C, d]` and g `[BH, N, C, d]` f32.

  `compute_dtype` is the operand dtype of the below-diagonal matmuls (the decayed q/k rows and right operand).
  """
  bh, n, c, d = q.shape
  spec_in = pl.BlockSpec((1, 1, c, d), lambda i, j: (i, j, 0, 0))
  spec_out = pl.BlockSpec((1, 1, c, c), lambda i, j: (i, j, 0, 0))
  return pl.pallas_call(
      functools.partial(_fwd_kernel, sub_block=sub_block, compute_dtype=compute_dtype),
      grid=(bh, n),
      in_specs=[spec_in, spec_in, spec_in],
      out_specs=[spec_out, spec_out],
      out_shape=[jax.ShapeDtypeStruct((bh, n, c, c), jnp.float32)] * 2,
      scratch_shapes=[pltpu.VMEM((c, d), jnp.float32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(q, k, g)


def intra_bwd_compute(
    q_ref, k_ref, g_ref, ds_ref, dk_ref, rexp_ref, col_ref, dq_s, row_s, piv_s, sub_block, compute_dtype
):
  """(dq, dk, dg), f32 `[C, d]`, of one chunk's intra matrices for cotangents in `ds_ref`, `dk_ref` (`[C, C]`).

  2-D refs; only the lower triangle of the cotangents is read. The last five refs are `[C, d]` f32 scratch.
  Each below-diagonal block's row and column terms of dg cancel exactly but not once rounded; the difference
  goes on the block's pivot row, as autodiff puts it (through e^(+-g_p)), so dg keeps summing to zero over the
  chunk and the log-decay gradient (a reverse cumsum of dg) does not accumulate a bias.
  """
  f32, cd = jnp.float32, compute_dtype
  c = g_ref.shape[0]
  sb = sub_block
  irow = jax.lax.broadcasted_iota(jnp.int32, (sb, 1), 0)
  jcol = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  # rexp_ref row j holds exp(g_p - g_j) for the current pivot p, zero for j >= p; col_ref accumulates DKcol.
  rexp_ref[...] = jnp.zeros_like(rexp_ref)
  col_ref[...] = jnp.zeros_like(col_ref)
  piv_s[...] = jnp.zeros_like(piv_s)
  for r in range(c // sb):
    p = r * sb
    gr = g_ref[p : p + sb, :]
    qr = q_ref[p : p + sb, :].astype(f32)
    kr = k_ref[p : p + sb, :].astype(f32)
    ds_r = ds_ref[p : p + sb, :]
    dk_r = dk_ref[p : p + sb, :]
    gp = gr[0:1, :]
    if r > 0:
      pp = p - sb
      if r > 1:
        rexp_ref[0:pp, :] = rexp_ref[0:pp, :] * jnp.exp(gp - g_ref[pp : pp + 1, :])
      rexp_ref[pp:p, :] = jnp.exp(gp - g_ref[pp:p, :])
      left = jnp.exp(gr - gp)
      rexp = rexp_ref[...]
      kr_all = (k_ref[...].astype(f32) * rexp).astype(cd)
      ds_c, dk_c = ds_r.astype(cd), dk_r.astype(cd)
      dq_r = left * jnp.dot(ds_c, kr_all, preferred_element_type=f32)
      row_r = left * jnp.dot(dk_c, kr_all, preferred_element_type=f32)
      col_blk = rexp * (_tn(ds_c, (qr * left).astype(cd)) + _tn(dk_c, (kr * left).astype(cd)))
      col_ref[...] += col_blk
      row_tot = jnp.sum(qr * dq_r + kr * row_r, axis=0, keepdims=True)
      col_tot = jnp.sum(k_ref[...].astype(f32) * col_blk, axis=0, keepdims=True)
      piv_s[p : p + 1, :] = col_tot - row_tot
    else:
      dq_r = jnp.zeros_like(qr)
      row_r = jnp.zeros_like(qr)
    col_d = jnp.zeros_like(qr)
    for jj in range(sb):
      e = jnp.exp(jnp.where(irow >= jj, gr - gr[jj : jj + 1, :], -jnp.inf))
      sel = jcol == (p + jj)
      cs = jnp.sum(jnp.where(sel, ds_r, 0.0), axis=1, keepdims=True)
      ck = jnp.sum(jnp.where(sel, dk_r, 0.0), axis=1, keepdims=True)
      t = e * kr[jj : jj + 1, :]
      dq_r = dq_r + cs * t
      row_r = row_r + ck * t
      col_d = jnp.where(irow == jj, jnp.sum((cs * qr + ck * kr) * e, axis=0, keepdims=True), col_d)
    dq_s[p : p + sb, :] = dq_r
    row_s[p : p + sb, :] = row_r
    col_ref[p : p + sb, :] += col_d
  dq, row, col = dq_s[...], row_s[...], col_ref[...]
  return dq, row + col, q_ref[...].astype(f32) * dq + k_ref[...].astype(f32) * (row - col) + piv_s[...]




def intra_bwd_compute_two_level(
    q_ref, k_ref, g_ref, ds_ref, dk_ref, rexp_ref, col_ref, dq_s, row_s, piv_s, rin_ref, sub_block, outer, compute_dtype
):
  """`intra_bwd_compute` with two block levels: rows go in blocks of `outer`, whose columns before the block
  factor through one pivot (the block's first row) in an `outer`-row matmul; only the columns inside the block
  use the `sub_block` pivots, so the full-width rescaling runs once per `outer` rows instead of per `sub_block`.

  `rin_ref` is one more `[C, d]` f32 scratch (the inner pivots' factors, rows inside the current outer block).
  """
  f32, cd = jnp.float32, compute_dtype
  c = g_ref.shape[0]
  sb = sub_block
  irow = jax.lax.broadcasted_iota(jnp.int32, (sb, 1), 0)
  jcol = jax.lax.broadcasted_iota(jnp.int32, (1, c), 1)
  rexp_ref[...] = jnp.zeros_like(rexp_ref)
  rin_ref[...] = jnp.zeros_like(rin_ref)
  col_ref[...] = jnp.zeros_like(col_ref)
  piv_s[...] = jnp.zeros_like(piv_s)
  kf_all = k_ref[...].astype(f32)
  for ob in range(c // outer):
    P = ob * outer
    g_o = g_ref[P : P + outer, :]
    q_o = q_ref[P : P + outer, :].astype(f32)
    k_o = k_ref[P : P + outer, :].astype(f32)
    ds_o = ds_ref[P : P + outer, :]
    dk_o = dk_ref[P : P + outer, :]
    gP = g_o[0:1, :]
    if ob > 0:  # columns before the block, through pivot P
      PP = P - outer
      if ob > 1:
        rexp_ref[0:PP, :] = rexp_ref[0:PP, :] * jnp.exp(gP - g_ref[PP : PP + 1, :])
      rexp_ref[PP:P, :] = jnp.exp(gP - g_ref[PP:P, :])
      left = jnp.exp(g_o - gP)
      rexp = rexp_ref[...]
      kr_all = (kf_all * rexp).astype(cd)
      ds_c, dk_c = ds_o.astype(cd), dk_o.astype(cd)
      dq_o = left * jnp.dot(ds_c, kr_all, preferred_element_type=f32)
      row_o = left * jnp.dot(dk_c, kr_all, preferred_element_type=f32)
      col_blk = rexp * (_tn(ds_c, (q_o * left).astype(cd)) + _tn(dk_c, (k_o * left).astype(cd)))
      col_ref[...] += col_blk
      row_tot = jnp.sum(q_o * dq_o + k_o * row_o, axis=0, keepdims=True)
      col_tot = jnp.sum(kf_all * col_blk, axis=0, keepdims=True)
      piv_s[P : P + 1, :] = col_tot - row_tot
    else:
      dq_o = jnp.zeros_like(q_o)
      row_o = jnp.zeros_like(q_o)
    dq_s[P : P + outer, :] = dq_o
    row_s[P : P + outer, :] = row_o
    # Inside the block: rows in sub-blocks; rin rows [P, p) hold exp(g_p - g_j) for the sub-block's pivot p.
    if ob > 0:
      rin_ref[P - outer : P, :] = jnp.zeros((outer, rin_ref.shape[1]), f32)
    for r in range(outer // sb):
      p = P + r * sb
      gr = g_ref[p : p + sb, :]
      qr = q_ref[p : p + sb, :].astype(f32)
      kr = k_ref[p : p + sb, :].astype(f32)
      ds_r = ds_ref[p : p + sb, :]
      dk_r = dk_ref[p : p + sb, :]
      gp = gr[0:1, :]
      dq_r = dq_s[p : p + sb, :]
      row_r = row_s[p : p + sb, :]
      if r > 0:
        pp = p - sb
        if r > 1:
          rin_ref[P:pp, :] = rin_ref[P:pp, :] * jnp.exp(gp - g_ref[pp : pp + 1, :])
        rin_ref[pp:p, :] = jnp.exp(gp - g_ref[pp:p, :])
        left = jnp.exp(gr - gp)
        rin_blk = rin_ref[P : P + outer, :]
        kin = (k_o * rin_blk).astype(cd)  # [outer, d], zero outside [P, p)
        ds_c = ds_r[:, P : P + outer].astype(cd) if outer < c else ds_r.astype(cd)
        dk_c = dk_r[:, P : P + outer].astype(cd) if outer < c else dk_r.astype(cd)
        dq_in = left * jnp.dot(ds_c, kin, preferred_element_type=f32)
        row_in = left * jnp.dot(dk_c, kin, preferred_element_type=f32)
        col_in = rin_blk * (_tn(ds_c, (qr * left).astype(cd)) + _tn(dk_c, (kr * left).astype(cd)))
        col_ref[P : P + outer, :] += col_in
        row_tot = jnp.sum(qr * dq_in + kr * row_in, axis=0, keepdims=True)
        col_tot = jnp.sum(k_o * col_in, axis=0, keepdims=True)
        piv_s[p : p + 1, :] = piv_s[p : p + 1, :] + col_tot - row_tot
        dq_r = dq_r + dq_in
        row_r = row_r + row_in
      col_d = jnp.zeros_like(qr)
      for jj in range(sb):
        e = jnp.exp(jnp.where(irow >= jj, gr - gr[jj : jj + 1, :], -jnp.inf))
        sel = jcol == (p + jj)
        cs = jnp.sum(jnp.where(sel, ds_r, 0.0), axis=1, keepdims=True)
        ck = jnp.sum(jnp.where(sel, dk_r, 0.0), axis=1, keepdims=True)
        t = e * kr[jj : jj + 1, :]
        dq_r = dq_r + cs * t
        row_r = row_r + ck * t
        col_d = jnp.where(irow == jj, jnp.sum((cs * qr + ck * kr) * e, axis=0, keepdims=True), col_d)
      dq_s[p : p + sb, :] = dq_r
      row_s[p : p + sb, :] = row_r
      col_ref[p : p + sb, :] += col_d
  dq, row, col = dq_s[...], row_s[...], col_ref[...]
  return dq, row + col, q_ref[...].astype(f32) * dq + k_ref[...].astype(f32) * (row - col) + piv_s[...]


def _bwd_kernel(
    q_ref, k_ref, g_ref, ds_ref, dk_ref, dq_out, dk_out, dg_out, rexp_ref, col_ref, dq_s, row_s, piv_s, *, sub_block,
    compute_dtype
):
  q_ref, k_ref, g_ref, ds_ref, dk_ref, dq_out, dk_out, dg_out = (
      _chunk_view(r) for r in (q_ref, k_ref, g_ref, ds_ref, dk_ref, dq_out, dk_out, dg_out)
  )
  dq, dk, dg = intra_bwd_compute(
      q_ref, k_ref, g_ref, ds_ref, dk_ref, rexp_ref, col_ref, dq_s, row_s, piv_s, sub_block, compute_dtype
  )
  dq_out[...] = dq.astype(dq_out.dtype)
  dk_out[...] = dk.astype(dk_out.dtype)
  dg_out[...] = dg


@functools.partial(jax.jit, static_argnames=("sub_block", "compute_dtype", "interpret"))
def kda_intra_bwd(q, k, g, d_scores, d_kk, sub_block=8, compute_dtype=jnp.bfloat16, interpret=False):
  """(dq, dk, dg) of the intra-chunk matrices for cotangents `[BH, N, C, C]`; only their lower triangle is read."""
  bh, n, c, d = q.shape
  spec_cd = pl.BlockSpec((1, 1, c, d), lambda i, j: (i, j, 0, 0))
  spec_cc = pl.BlockSpec((1, 1, c, c), lambda i, j: (i, j, 0, 0))
  return pl.pallas_call(
      functools.partial(_bwd_kernel, sub_block=sub_block, compute_dtype=compute_dtype),
      grid=(bh, n),
      in_specs=[spec_cd, spec_cd, spec_cd, spec_cc, spec_cc],
      out_specs=[spec_cd, spec_cd, spec_cd],
      out_shape=[
          jax.ShapeDtypeStruct(q.shape, q.dtype),
          jax.ShapeDtypeStruct(k.shape, k.dtype),
          jax.ShapeDtypeStruct(g.shape, g.dtype),
      ],
      scratch_shapes=[pltpu.VMEM((c, d), jnp.float32) for _ in range(5)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
      interpret=interpret,
  )(q, k, g, d_scores.astype(jnp.float32), d_kk.astype(jnp.float32))
