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

"""The Gated DeltaNet's chunked delta rule as Triton kernels (`Qwen35Config.gdn_scan_impl='triton'`, GPU).

`gdn.chunk_gated_delta_rule` runs the recurrence over the state `S [Dk, Dv]` as a 64-step `lax.scan` (4096 tokens,
chunks of 64) whose every step is a handful of small fp32 matmuls, elementwise ops and slice updates: on a GB200 at a
Qwen3.8 device's shapes (B=2, 128 value heads, Dk = Dv = 128) 10.0 ms forward and 27.4 ms forward + backward a layer
a microbatch.  Here the part that is sequential -- per (batch, head) and chunk n,

    o_n = (q_n e^{g_n}) S;   vn_n = u_n - w_n S;   S <- a_n S + Kd_n^T vn_n,      Kd_n = k_n * e_n[:, None]

(u_n, w_n the chunk's WY factors, a_n = exp(g_n[-1]), e_n = exp(g_n[-1] - g_n), the state entering the chunk on
the right-hand sides) -- runs as ONE kernel that keeps S on chip and loops over the chunks (a program per
(batch * head, Dv tile): every op is linear in Dv, so Dv tiles are independent).  The rest -- the WY factors
before it, the intra-chunk term `A_n vn_n` of the output after it -- runs on per-chunk kernels too (FUSED, below),
or, unfused, stays in XLA, batched over all chunks, at HIGHEST as before.  The exps (e, a, e^g) are XLA's, computed
as the reference computes them; the kernels only multiply.

The backward is the reverse recurrence, also one kernel: with lam = 0 after the last chunk and, for n = N-1 .. 0,

    dvt_n = dvn_n + Kd_n lam;      lam <- (q_n e^{g_n})^T do_n + a_n lam - w_n^T dvt_n

(do_n, dvn_n the cotangents of o_n and vn_n), it writes dvt_n and the lam after chunk n; then, batched over the
chunks (in XLA; FUSED: its `prod` kernel), du_n = dvt_n, dw_n = -dvt_n h_n^T, d(q e^g)_n = do_n h_n^T,
dKd_n = vn_n lam_n^T (-> dk_n, de_n) and da_n = <h_n, lam_n>, h_n the state entering chunk n -- which the
differentiated forward stores (the undifferentiated one never writes it).  XLA's autodiff (FUSED: `chunk_rule`'s
VJP) carries de, da and d(e^g) on to g through the exps.

Every kernel matmul pairs an operand whose values are exact in the compute dtype (q, k, w: bf16) with an fp32 one
(the state, its cotangent, vn scaled by a decay); the expressions are arranged so (`(q e^g) S = e^g (q S)`,
`(k e)^T vn = k^T (e vn)`), and the fp32 side is split into three bf16 parts -- three bf16 tensor-core products,
fp32's 24 bits ('bf16x3e', DEFAULT_PRECISION); fp32 compute uses Triton's 'tf32x3'.  Triton's 'ieee' fp32 dot runs
on the FMA units, several times slower than the scan it replaces.  Summed in another order than cuBLAS's, the
result matches the scan to fp32 rounding, not bitwise; in bf16 compute it is as far from an fp32-compute run as the
scan is (gdn_kernels_test).

FUSED (`chunk` / `chunk_rule`: the default under 'triton' where it applies, chunks of 16, 32 or 64 -- the model's
64): the rest of the rule on kernels as well, a program per (batch, chunk, head), so that XLA keeps only the layout
changes, the exps and the decays' shares of dg --

    wy_fwd    smat = (k beta) k^T decay-masked, A = (I + smat)^-1 (its 16-row diagonal blocks by forward
              substitution, then the block rows X_i = X_ii (E_i - sum_{k<i} S_ik X_k)), u = A (v beta),
              w = (A e^g) (k beta)
              -- the recurrence's forward kernel --
    out       o = oi + causal(q k^T e^{g_i - g_j}) vn                      (in the output's dtype)
  and back
    out_bwd   that term's backward: dvn = P^T do, d(q k^T) -> dq, dk, and the decay's share of dg
              -- the recurrence's backward kernel --
    prod      the per-chunk products: dw = -dvt h^T, d(q e^g) = do h^T, dKd = vn lam^T, da = <h, lam> (dq whole)
    wy_bwd    the WY factors' backward: dA = du (v beta)^T + (dw (k beta)^T) e^{g_j}, dS = -A^T dA A^T -> dk, dv,
              dbeta, dg (dk, dv whole)

Operands pair as in the recurrence: one exact in bf16 (q, k, v, beta; the output's cotangent when the output is in
the compute dtype) against an fp32 one split three ways.  The products of two fp32 operands (A against the
cotangents, P against vn, the states against theirs) split both into two bf16 parts and take three products
('bf16x3': the dropped terms are 2^-16 of the operands' scale, under the bf16 rounding of what bf16 compute
returns; TWO_SIDED); in fp32 compute those are 'tf32x3' too.  The 16 x 16 solves and the block rows run at 'ieee'.

On a GB200 at a Qwen3.8 device's shapes (bf16 compute), forward and forward + backward a layer a microbatch, and
XLA's temporaries for each:
    the scan                       9.9 ms   27.4 ms    2.0 GiB   8.3 GiB
    the recurrence's kernels       5.5      14.5       2.5       6.5
    FUSED                          3.7      10.7       1.8       5.8
(the kernels of FUSED's forward + backward, ms: wy_fwd 1.55, recurrence 1.46, out 0.36 | out_bwd 1.04, recurrence
1.70, prod 1.90, wy_bwd 2.08; the rest, ~0.6 ms, XLA's layout changes, l2norm and casts).  Against the scan in fp32
compute on the same bf16 inputs, FUSED's output and five gradients are as close as the bf16 scan's or closer (rms
4.3e-3 .. 5.9e-3 of each tensor's, the scan's 4.9e-3 .. 7.2e-3).

Memory: h [B, N, H, Dk, Dv] fp32 (1.07 GB at the shapes above) and vn (0.54 GB) are the backward's residuals (FUSED
adds A, 0.27 GB), lam (1.07 GB) and dvt (0.54 GB) its temporaries -- the scan's own VJP keeps its per-step carries
(the same 1.07 GB) and more.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

_HI = jax.lax.Precision.HIGHEST
PRECISIONS = ("ieee", "tf32x3", "tf32", "tf32x2e", "bf16x3e")
EXACT_PRECISIONS = ("tf32x2e", "bf16x3e")  # these need q, k, w exact in the split's head: bf16 or f16 for
# 'tf32x2e' (10 mantissa bits), bf16 alone for 'bf16x3e' (7)
# The kernels' dots in bf16 / f16 compute: 'bf16x3e' -- the fp32 operand split into three bf16 parts against the
# exact-in-bf16 one (q, k or w), three bf16 tensor-core products: fp32's 24 bits.  In fp32 compute 'tf32x3' (three
# tf32 products, Triton's split of both operands: 9e-7 of HIGHEST's result at these shapes).  Triton's 'ieee' fp32
# dot runs on the FMA units, 4x slower than the scan it replaces on a GB200.  On a GB200 at a Qwen3.8 device's
# shapes, the rule's forward + backward unfused: bf16x3e 14.5 ms, tf32x2e 14.4, tf32x3 16.1, ieee 120; the scan 27.4.
DEFAULT_PRECISION = "bf16x3e"


def supported(chunk, dk, dv):
  """The kernels' shape limits: the chunk and both head dims powers of two, at least 16 (tl.arange and tl.dot)."""
  return all(x >= 16 and x & (x - 1) == 0 for x in (chunk, dk, dv))


def default_block_v(dv):
  """The Dv tile a program carries (64 on a GB200: 128 with 8 warps ties it, 32 is 3% slower)."""
  return min(dv, 64)


@functools.cache
def _kernels():
  import triton
  import triton.language as tl

  @triton.jit
  def mm(a, b, PREC: tl.constexpr):
    # a @ b with fp32 accumulation.  'tf32x2e' takes an `a` whose values are exact in tf32 (bf16 or f16 q, k, w),
    # 'bf16x3e' one exact in bf16: b is split into a tf32 head and tail (two tf32 products) or into three bf16 parts
    # (three bf16 products, b to fp32's 24 bits); the others are Triton's own precisions on fp32 operands.
    if PREC == "tf32x2e":
      af = a.to(tl.float32)
      hi = (b.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
      return tl.dot(af, b - hi, acc=tl.dot(af, hi, input_precision="tf32"), input_precision="tf32")
    elif PREC == "bf16x3e":
      ab = a.to(tl.bfloat16)
      b1 = b.to(tl.bfloat16)
      r = b - b1.to(tl.float32)
      b2 = r.to(tl.bfloat16)
      b3 = (r - b2.to(tl.float32)).to(tl.bfloat16)
      return tl.dot(ab, b3, acc=tl.dot(ab, b2, acc=tl.dot(ab, b1)))
    else:
      return tl.dot(a.to(tl.float32), b, input_precision=PREC)

  @triton.jit
  def fwd_kernel(
      u_ptr,
      w_ptr,
      k_ptr,
      e_ptr,
      a_ptr,
      q_ptr,
      eg_ptr,
      o_ptr,
      vn_ptr,
      h_ptr,
      N: tl.constexpr,
      H: tl.constexpr,
      C: tl.constexpr,
      DK: tl.constexpr,
      DV: tl.constexpr,
      BV: tl.constexpr,
      PREC: tl.constexpr,
      STORE_H: tl.constexpr,
  ):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    hh = (pid_bh % H).to(tl.int64)
    offs_c = tl.arange(0, C)
    offs_k = tl.arange(0, DK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    s = tl.zeros((DK, BV), dtype=tl.float32)
    for n in range(N):
      bnh = (b * N + n) * H + hh  # [B, N, H, ...] layouts throughout
      rows = bnh * C + offs_c
      ck = rows[:, None] * DK + offs_k[None, :]
      cv = rows[:, None] * DV + offs_v[None, :]
      w = tl.load(w_ptr + ck)  # [C, DK], the compute dtype
      k = tl.load(k_ptr + ck)
      q = tl.load(q_ptr + ck)
      u = tl.load(u_ptr + cv).to(tl.float32)  # [C, BV]
      e = tl.load(e_ptr + rows)  # [C]
      eg = tl.load(eg_ptr + rows)
      a = tl.load(a_ptr + bnh)
      if STORE_H:  # h_n = S, for the backward
        tl.store(h_ptr + (bnh * DK + offs_k)[:, None] * DV + offs_v[None, :], s)
      tl.store(o_ptr + cv, eg[:, None] * mm(q, s, PREC))  # (q e^g) h_n
      vn = u - mm(w, s, PREC)
      tl.store(vn_ptr + cv, vn)
      s = s * a + mm(tl.trans(k), vn * e[:, None], PREC)  # (k e)^T vn

  @triton.jit
  def bwd_kernel(
      w_ptr,
      k_ptr,
      e_ptr,
      a_ptr,
      q_ptr,
      eg_ptr,
      do_ptr,
      dvn_ptr,
      dvt_ptr,
      lam_ptr,
      N: tl.constexpr,
      H: tl.constexpr,
      C: tl.constexpr,
      DK: tl.constexpr,
      DV: tl.constexpr,
      BV: tl.constexpr,
      PREC: tl.constexpr,
  ):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    hh = (pid_bh % H).to(tl.int64)
    offs_c = tl.arange(0, C)
    offs_k = tl.arange(0, DK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    lam = tl.zeros((DK, BV), dtype=tl.float32)  # lam_{n+1}: the cotangent of the state after chunk n
    for i in range(N):
      n = N - 1 - i
      bnh = (b * N + n) * H + hh
      rows = bnh * C + offs_c
      ck = rows[:, None] * DK + offs_k[None, :]
      cv = rows[:, None] * DV + offs_v[None, :]
      tl.store(lam_ptr + (bnh * DK + offs_k)[:, None] * DV + offs_v[None, :], lam)
      w = tl.load(w_ptr + ck)
      k = tl.load(k_ptr + ck)
      q = tl.load(q_ptr + ck)
      e = tl.load(e_ptr + rows)
      eg = tl.load(eg_ptr + rows)
      a = tl.load(a_ptr + bnh)
      do = tl.load(do_ptr + cv)
      dvn = tl.load(dvn_ptr + cv)
      dvt = dvn + e[:, None] * mm(k, lam, PREC)  # + (k e) lam
      tl.store(dvt_ptr + cv, dvt)
      dh = mm(tl.trans(q), do * eg[:, None], PREC)  # the cotangent of h_n
      lam = dh + lam * a - mm(tl.trans(w), dvt, PREC)

  return fwd_kernel, bwd_kernel


# The launch: warps a program and the software-pipelining depth of the chunk loop (a GB200 at a Qwen3.8 device's
# shapes: 4 warps, no pipelining, at Dv tiles of 64; deeper pipelines ran slower).
LAUNCH = dict(num_warps=4, num_stages=1)


def _launch(kernel, args, out_type, b, h, n, c, dk, dv, block_v, precision, name, **meta):
  import jax_triton as jt

  if dv % block_v:
    raise ValueError(f"gdn_triton: Dv {dv} is not a multiple of the block {block_v}")
  return jt.triton_call(
      *args,
      kernel=kernel,
      out_type=out_type,
      grid=(b * h, dv // block_v),
      name=name,
      N=n,
      H=h,
      C=c,
      DK=dk,
      DV=dv,
      BV=block_v,
      PREC=precision,
      **meta,
      **LAUNCH,
  )


def _fwd_call(u, w, k, e, a, q, eg, block_v, precision, store_h):
  """The forward kernel -> (o_inter, vn, h): o_inter [B, N, H, C, Dv] = (q e^g) h_n, vn [B, N, H, C, Dv], h
  [B, N, H, Dk, Dv] the state entering each chunk (store_h; else a placeholder never written).  All fp32."""
  b, n, h, c, dk = w.shape
  dv = u.shape[-1]
  f32 = jnp.float32
  out = (
      jax.ShapeDtypeStruct((b, n, h, c, dv), f32),
      jax.ShapeDtypeStruct((b, n, h, c, dv), f32),
      jax.ShapeDtypeStruct((b, n, h, dk, dv) if store_h else (1,), f32),
  )
  fwd, _ = _kernels()
  return _launch(
      fwd,
      (u, w, k, e.astype(f32), a.astype(f32), q, eg.astype(f32)),
      out,
      b,
      h,
      n,
      c,
      dk,
      dv,
      block_v,
      precision,
      "gdn_scan_fwd",
      STORE_H=bool(store_h),
  )


def _bwd_call(w, k, e, a, q, eg, do, dvn, block_v, precision):
  """The reverse recurrence -> (dvt, lam): dvt_i the total cotangent of vn_i, lam_i that of the state after
  chunk i (lam_{N-1} = 0); the cotangent of h_i, (q_i e^{g_i})^T do_i, is formed inside."""
  b, n, h, c, dk = w.shape
  dv = do.shape[-1]
  f32 = jnp.float32
  out = (jax.ShapeDtypeStruct((b, n, h, c, dv), f32), jax.ShapeDtypeStruct((b, n, h, dk, dv), f32))
  _, bwd = _kernels()
  return _launch(
      bwd,
      (w, k, e.astype(f32), a.astype(f32), q, eg.astype(f32), do.astype(f32), dvn.astype(f32)),
      out,
      b,
      h,
      n,
      c,
      dk,
      dv,
      block_v,
      precision,
      "gdn_scan_bwd",
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def scan(u, w, k, e, a, q, eg, block_v, precision):
  """The inter-chunk recurrence (module docstring) -> (o_inter, vn), both fp32.

  u [B, N, H, C, Dv], w, k, q [B, N, H, C, Dk] (any float dtype: read as fp32), e, eg [B, N, H, C] and
  a [B, N, H] fp32; o_inter = (q e^g) h_n with h_n the state entering chunk n (zero for the first), vn = u - w h_n.
  Undifferentiated, h is never written; differentiated, it is the backward's residual."""
  o, vn, _ = _fwd_call(u, w, k, e, a, q, eg, block_v, precision, store_h=False)
  return o, vn


def _scan_fwd(u, w, k, e, a, q, eg, block_v, precision):
  o, vn, h = _fwd_call(u, w, k, e, a, q, eg, block_v, precision, store_h=True)
  return (o, vn), (jnp.zeros((0,), u.dtype), w, k, e, a, q, eg, h, vn)


def _scan_bwd(block_v, precision, res, cts):
  u0, w, k, e, a, q, eg, h, vn = res
  do, dvn = cts
  f32 = jnp.float32
  dvt, lam = _bwd_call(w, k, e, a, q, eg, do, dvn, block_v, precision)
  ht = h.swapaxes(-1, -2)
  dw = -jnp.matmul(dvt, ht, precision=_HI)
  dqg = jnp.matmul(do.astype(f32), ht, precision=_HI)
  dkd = jnp.matmul(vn, lam.swapaxes(-1, -2), precision=_HI)
  da = jnp.sum(h * lam, axis=(-2, -1))
  ef, egf = e.astype(f32), eg.astype(f32)
  dk = dkd * ef[..., None]
  de = jnp.sum(dkd * k.astype(f32), axis=-1)
  dq = dqg * egf[..., None]
  deg = jnp.sum(dqg * q.astype(f32), axis=-1)
  return (
      dvt.astype(u0.dtype),
      dw.astype(w.dtype),
      dk.astype(k.dtype),
      de.astype(e.dtype),
      da.astype(a.dtype),
      dq.astype(q.dtype),
      deg.astype(eg.dtype),
  )


scan.defvjp(_scan_fwd, _scan_bwd)


# ------------------------------------------------------------------ the whole rule on kernels (FUSED)
#
# The kernels above take the recurrence; the rest of the rule -- the WY factors (the intra-chunk triangular solve
# among them), the output's intra-chunk term and every backward product -- ran in XLA, fp32 at HIGHEST, which on a
# GPU is CUDA-core SGEMMs and a batched trsm (on a GB200 ~5 + 1.7 s of a 65 s Qwen3.8 step on a stage with two GDN
# layers).  With FUSED, `chunk_rule` runs them as per-chunk kernels too (the module docstring: wy_fwd, out, out_bwd,
# prod, wy_bwd), one program a (batch, chunk, head), every product on the tensor cores.  False: the unfused path.
FUSED = True
FUSED_CHUNKS = (16, 32, 64)  # wy_fwd's 16-row blocks, at most 4 of them


def fused_supported(chunk, dk, dv):
  """Shapes `chunk_rule` takes: `supported`'s and a chunk of 16, 32 or 64 rows."""
  return supported(chunk, dk, dv) and chunk in FUSED_CHUNKS


@functools.cache
def _chunk_kernels():
  import triton
  import triton.language as tl

  @triton.jit
  def mm(a, b, acc, PREC: tl.constexpr):
    # acc + a @ b, `a` exact in the compute dtype against `b` fp32 split three ways (bf16x3e) or two (tf32x2e).
    # Every helper accumulates into the `acc` it is handed (a loop's carried sum stays the MMAs' accumulator)
    if PREC == "tf32x2e":
      af = a.to(tl.float32)
      hi = (b.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
      return tl.dot(af, b - hi, acc=tl.dot(af, hi, acc=acc, input_precision="tf32"), input_precision="tf32")
    elif PREC == "bf16x3e":
      ab = a.to(tl.bfloat16)
      b1 = b.to(tl.bfloat16)
      r = b - b1.to(tl.float32)
      b2 = r.to(tl.bfloat16)
      b3 = (r - b2.to(tl.float32)).to(tl.bfloat16)
      return tl.dot(ab, b3, acc=tl.dot(ab, b2, acc=tl.dot(ab, b1, acc=acc)))
    else:
      return tl.dot(a.to(tl.float32), b, acc=acc, input_precision=PREC)

  @triton.jit
  def mme(a, b, acc, PREC: tl.constexpr):
    # acc + a @ b, both exact in the compute dtype (bf16 / f16: their products are exact in fp32): one native product
    if PREC == "bf16x3e" or PREC == "tf32x2e":
      return tl.dot(a, b, acc=acc)
    else:
      return tl.dot(a.to(tl.float32), b.to(tl.float32), acc=acc, input_precision=PREC)

  @triton.jit
  def mml(a, b, acc, PREC: tl.constexpr):
    # acc + a @ b, `a` fp32 split three ways (bf16x3e) or two (tf32x2e) against a `b` exact in the compute dtype
    if PREC == "bf16x3e":
      a1 = a.to(tl.bfloat16)
      r = a - a1.to(tl.float32)
      a2 = r.to(tl.bfloat16)
      a3 = (r - a2.to(tl.float32)).to(tl.bfloat16)
      bb = b.to(tl.bfloat16)
      return tl.dot(a3, bb, acc=tl.dot(a2, bb, acc=tl.dot(a1, bb, acc=acc)))
    elif PREC == "tf32x2e":
      bf = b.to(tl.float32)
      hi = (a.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
      return tl.dot(a - hi, bf, acc=tl.dot(hi, bf, acc=acc, input_precision="tf32"), input_precision="tf32")
    else:
      return tl.dot(a, b.to(tl.float32), acc=acc, input_precision=PREC)

  @triton.jit
  def mmf(a, b, acc, PREC: tl.constexpr):
    # acc + a @ b, both fp32 and neither exact.  'bf16x3': each split in two bf16 parts, three bf16 products (the
    # dropped terms 2^-16 of |a||b|: below the bf16 rounding of what bf16 compute returns); else Triton's own
    # ('tf32x3': three tf32 products, ~fp32's precision)
    if PREC == "bf16x3":
      a1 = a.to(tl.bfloat16)
      a2 = (a - a1.to(tl.float32)).to(tl.bfloat16)
      b1 = b.to(tl.bfloat16)
      b2 = (b - b1.to(tl.float32)).to(tl.bfloat16)
      return tl.dot(a2, b1, acc=tl.dot(a1, b2, acc=tl.dot(a1, b1, acc=acc)))
    else:
      return tl.dot(a, b, acc=acc, input_precision=PREC)

  @triton.jit
  def wy_fwd_kernel(
      k_ptr,
      v_ptr,
      beta_ptr,
      g_ptr,
      a_ptr,
      u_ptr,
      w_ptr,
      C: tl.constexpr,
      DK: tl.constexpr,
      DV: tl.constexpr,
      NB: tl.constexpr,
      BD: tl.constexpr,
      PREC: tl.constexpr,
      FPREC: tl.constexpr,
  ):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * C
    o16 = tl.arange(0, 16)
    oc = tl.arange(0, C)
    ok = tl.arange(0, DK)
    od = tl.arange(0, BD)
    eye = (o16[:, None] == o16[None, :]).to(tl.float32)
    for i in tl.static_range(NB):
      ri = base + i * 16 + o16
      k_i = tl.load(k_ptr + ri[:, None] * DK + ok[None, :])
      g_i = tl.load(g_ptr + ri)
      kb_i = (k_i.to(tl.float32) * tl.load(beta_ptr + ri).to(tl.float32)[:, None]).to(k_i.dtype)
      strict = o16[:, None] > o16[None, :]
      s_ii = mme(kb_i, tl.trans(k_i), tl.zeros((16, 16), dtype=tl.float32), PREC)
      s_ii = tl.where(strict, s_ii * tl.exp(tl.where(strict, g_i[:, None] - g_i[None, :], 0.0)), 0.0)
      x_ii = eye  # (I + S_ii)^-1, row by row: x_r = e_r - sum_{j<r} S_rj x_j
      for r in tl.static_range(1, 16):
        s_r = tl.sum(tl.where(o16[:, None] == r, s_ii, 0.0), axis=0)
        x_ii = tl.where(o16[:, None] == r, x_ii - tl.sum(s_r[:, None] * x_ii, axis=0)[None, :], x_ii)
      t = tl.zeros((16, C), dtype=tl.float32)  # sum_{k<i} S_ik X_k, over the block rows already written
      for kk in tl.static_range(i):
        rk = base + kk * 16 + o16
        k_k = tl.load(k_ptr + rk[:, None] * DK + ok[None, :])
        s_ik = mme(kb_i, tl.trans(k_k), tl.zeros((16, 16), dtype=tl.float32), PREC)
        s_ik = s_ik * tl.exp(g_i[:, None] - tl.load(g_ptr + rk)[None, :])
        t += tl.dot(s_ik, tl.load(a_ptr + rk[:, None] * C + oc[None, :]), input_precision="ieee")
      e_i = (oc[None, :] == (i * 16 + o16)[:, None]).to(tl.float32)
      tl.store(a_ptr + ri[:, None] * C + oc[None, :], tl.dot(x_ii, e_i - t, input_precision="ieee"))
      tl.debug_barrier()  # block row i is read back for the block rows after it
    rows = base + oc
    a = tl.load(a_ptr + rows[:, None] * C + oc[None, :])
    beta = tl.load(beta_ptr + rows).to(tl.float32)
    ag = a * tl.exp(tl.load(g_ptr + rows))[None, :]
    for t in tl.range(0, DV // BD):  # u = A vb, a Dv tile at a time
      cv = rows[:, None] * DV + (t * BD + od)[None, :]
      v = tl.load(v_ptr + cv)
      vb = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
      tl.store(u_ptr + cv, mml(a, vb, tl.zeros((C, BD), dtype=tl.float32), PREC).to(u_ptr.dtype.element_ty))
    for t in tl.range(0, DK // BD):  # w = (A e^{g_j}) kb, a Dk tile at a time
      ck = rows[:, None] * DK + (t * BD + od)[None, :]
      k = tl.load(k_ptr + ck)
      kb = (k.to(tl.float32) * beta[:, None]).to(k.dtype)
      tl.store(w_ptr + ck, mml(ag, kb, tl.zeros((C, BD), dtype=tl.float32), PREC).to(w_ptr.dtype.element_ty))

  @triton.jit
  def qk_tiles(q_ptr, k_ptr, rows, DK: tl.constexpr, BD: tl.constexpr, C: tl.constexpr, PREC: tl.constexpr):
    od = tl.arange(0, BD)
    qk = tl.zeros((C, C), dtype=tl.float32)
    for t in tl.range(0, DK // BD):
      ck = rows[:, None] * DK + (t * BD + od)[None, :]
      qk = mme(tl.load(q_ptr + ck), tl.trans(tl.load(k_ptr + ck)), qk, PREC)
    return qk

  @triton.jit
  def out_kernel(
      oi_ptr,
      q_ptr,
      k_ptr,
      g_ptr,
      vn_ptr,
      o_ptr,
      C: tl.constexpr,
      DK: tl.constexpr,
      DV: tl.constexpr,
      BD: tl.constexpr,
      PREC: tl.constexpr,
      FPREC: tl.constexpr,
  ):
    pid = tl.program_id(0).to(tl.int64)
    oc = tl.arange(0, C)
    od = tl.arange(0, BD)
    rows = pid * C + oc
    g = tl.load(g_ptr + rows)
    causal = oc[:, None] >= oc[None, :]
    p = qk_tiles(q_ptr, k_ptr, rows, DK, BD, C, PREC)
    p = tl.where(causal, p * tl.exp(tl.where(causal, g[:, None] - g[None, :], 0.0)), 0.0)
    for t in tl.range(0, DV // BD):
      cv = rows[:, None] * DV + (t * BD + od)[None, :]
      o = tl.load(oi_ptr + cv) + mmf(p, tl.load(vn_ptr + cv), tl.zeros((C, BD), dtype=tl.float32), FPREC)
      tl.store(o_ptr + cv, o.to(o_ptr.dtype.element_ty))

  @triton.jit
  def out_bwd_kernel(
      q_ptr,
      k_ptr,
      g_ptr,
      do_ptr,
      vn_ptr,
      dvn_ptr,
      dq_ptr,
      dk_ptr,
      dg_ptr,
      C: tl.constexpr,
      DK: tl.constexpr,
      DV: tl.constexpr,
      BD: tl.constexpr,
      PREC: tl.constexpr,
      FPREC: tl.constexpr,
      DO_EXACT: tl.constexpr,
  ):
    pid = tl.program_id(0).to(tl.int64)
    oc = tl.arange(0, C)
    od = tl.arange(0, BD)
    rows = pid * C + oc
    g = tl.load(g_ptr + rows)
    causal = oc[:, None] >= oc[None, :]
    dm = tl.where(causal, tl.exp(tl.where(causal, g[:, None] - g[None, :], 0.0)), 0.0)
    qk = qk_tiles(q_ptr, k_ptr, rows, DK, BD, C, PREC)
    pt = tl.trans(qk * dm)
    dp = tl.zeros((C, C), dtype=tl.float32)
    for t in tl.range(0, DV // BD):
      cv = rows[:, None] * DV + (t * BD + od)[None, :]
      do = tl.load(do_ptr + cv)
      if DO_EXACT:  # the cotangent of an output in the compute dtype: exact there
        tl.store(dvn_ptr + cv, mml(pt, do, tl.zeros((C, BD), dtype=tl.float32), PREC))  # P^T do
        dp = mm(do, tl.trans(tl.load(vn_ptr + cv)), dp, PREC)  # do vn^T
      else:
        tl.store(dvn_ptr + cv, mmf(pt, do.to(tl.float32), tl.zeros((C, BD), dtype=tl.float32), FPREC))
        dp = mmf(do.to(tl.float32), tl.trans(tl.load(vn_ptr + cv)), dp, FPREC)
    dqk = dp * dm
    dqkt = tl.trans(dqk)
    for t in tl.range(0, DK // BD):
      ck = rows[:, None] * DK + (t * BD + od)[None, :]
      tl.store(dq_ptr + ck, mml(dqk, tl.load(k_ptr + ck), tl.zeros((C, BD), dtype=tl.float32), PREC))
      tl.store(dk_ptr + ck, mml(dqkt, tl.load(q_ptr + ck), tl.zeros((C, BD), dtype=tl.float32), PREC))
    dl = dqk * qk  # d log D: + row i, - column j
    tl.store(dg_ptr + rows, tl.sum(dl, axis=1) - tl.sum(dl, axis=0))

  @triton.jit
  def prod_kernel(
      dvt_ptr,
      h_ptr,
      do_ptr,
      vn_ptr,
      lam_ptr,
      k_ptr,
      q_ptr,
      e_ptr,
      eg_ptr,
      dqo_ptr,
      dko_ptr,
      dw_ptr,
      dq_ptr,
      dk_ptr,
      deg_ptr,
      de_ptr,
      da_ptr,
      C: tl.constexpr,
      DK: tl.constexpr,
      DV: tl.constexpr,
      BK: tl.constexpr,
      BD: tl.constexpr,
      PREC: tl.constexpr,
      FPREC: tl.constexpr,
      DO_EXACT: tl.constexpr,
  ):
    pid = tl.program_id(0).to(tl.int64)
    pk = tl.program_id(1)
    nk = DK // BK
    oc = tl.arange(0, C)
    obk = pk * BK + tl.arange(0, BK)
    od = tl.arange(0, BD)
    rows = pid * C + oc
    hrow = (pid * DK + obk)[:, None] * DV  # this K tile's rows of h / lam
    ndw = tl.zeros((C, BK), dtype=tl.float32)
    dqg = tl.zeros((C, BK), dtype=tl.float32)
    dkd = tl.zeros((C, BK), dtype=tl.float32)
    da = tl.zeros((BK, BD), dtype=tl.float32)
    for t in tl.range(0, DV // BD):
      ov = t * BD + od
      cv = rows[:, None] * DV + ov[None, :]
      ht = tl.trans(tl.load(h_ptr + hrow + ov[None, :]))
      lam = tl.load(lam_ptr + hrow + ov[None, :])
      ndw = mmf(tl.load(dvt_ptr + cv), ht, ndw, FPREC)  # -dw
      if DO_EXACT:
        dqg = mm(tl.load(do_ptr + cv), ht, dqg, PREC)
      else:
        dqg = mmf(tl.load(do_ptr + cv).to(tl.float32), ht, dqg, FPREC)
      dkd = mmf(tl.load(vn_ptr + cv), tl.trans(lam), dkd, FPREC)
      da += tl.trans(ht) * lam
    ckb = rows[:, None] * DK + obk[None, :]
    tl.store(dw_ptr + ckb, -ndw)
    dq = dqg * tl.load(eg_ptr + rows)[:, None] + tl.load(dqo_ptr + ckb)  # + the intra-chunk term's
    tl.store(dq_ptr + ckb, dq.to(dq_ptr.dtype.element_ty))
    tl.store(deg_ptr + (pid * nk + pk) * C + oc, tl.sum(dqg * tl.load(q_ptr + ckb).to(tl.float32), axis=1))
    tl.store(dk_ptr + ckb, dkd * tl.load(e_ptr + rows)[:, None] + tl.load(dko_ptr + ckb))
    tl.store(de_ptr + (pid * nk + pk) * C + oc, tl.sum(dkd * tl.load(k_ptr + ckb).to(tl.float32), axis=1))
    tl.store(da_ptr + pid * nk + pk, tl.sum(tl.sum(da, axis=1), axis=0))

  @triton.jit
  def wy_bwd_kernel(
      k_ptr,
      v_ptr,
      beta_ptr,
      g_ptr,
      a_ptr,
      du_ptr,
      dw_ptr,
      dkin_ptr,
      dk_ptr,
      dv_ptr,
      dbeta_ptr,
      dg_ptr,
      C: tl.constexpr,
      DK: tl.constexpr,
      DV: tl.constexpr,
      BD: tl.constexpr,
      PREC: tl.constexpr,
      FPREC: tl.constexpr,
  ):
    pid = tl.program_id(0).to(tl.int64)
    oc = tl.arange(0, C)
    od = tl.arange(0, BD)
    rows = pid * C + oc
    beta = tl.load(beta_ptr + rows).to(tl.float32)
    g = tl.load(g_ptr + rows)
    eg = tl.exp(g)
    a = tl.load(a_ptr + rows[:, None] * C + oc[None, :])
    at = tl.trans(a)
    # dA = du vb^T + (dw kb^T) e^{g_j}  (u = A vb, w = (A e^g) kb), and kb k^T for S's decay gradient
    da_u = tl.zeros((C, C), dtype=tl.float32)
    for t in tl.range(0, DV // BD):
      cv = rows[:, None] * DV + (t * BD + od)[None, :]
      v = tl.load(v_ptr + cv)
      da_u = mml(tl.load(du_ptr + cv), tl.trans((v.to(tl.float32) * beta[:, None]).to(v.dtype)), da_u, PREC)
    da_w = tl.zeros((C, C), dtype=tl.float32)
    kk = tl.zeros((C, C), dtype=tl.float32)
    for t in tl.range(0, DK // BD):
      ck = rows[:, None] * DK + (t * BD + od)[None, :]
      k = tl.load(k_ptr + ck)
      kb = (k.to(tl.float32) * beta[:, None]).to(k.dtype)
      da_w = mml(tl.load(dw_ptr + ck), tl.trans(kb), da_w, PREC)
      kk = mme(kb, tl.trans(k), kk, PREC)
    zc = tl.zeros((C, C), dtype=tl.float32)
    ds = -mmf(mmf(at, da_u + da_w * eg[None, :], zc, FPREC), at, zc, FPREC)  # A = (I + S)^-1
    strict = oc[:, None] > oc[None, :]
    dkk = tl.where(strict, ds * tl.exp(tl.where(strict, g[:, None] - g[None, :], 0.0)), 0.0)
    dl = dkk * kk
    dkkt = tl.trans(dkk)
    dbeta = tl.zeros((C,), dtype=tl.float32)
    dgw = tl.zeros((C,), dtype=tl.float32)
    for t in tl.range(0, DK // BD):
      ck = rows[:, None] * DK + (t * BD + od)[None, :]
      k = tl.load(k_ptr + ck)
      kf = k.to(tl.float32)
      kb = (kf * beta[:, None]).to(k.dtype)
      dkbe = mmf(at, tl.load(dw_ptr + ck), tl.zeros((C, BD), dtype=tl.float32), FPREC)  # w = A (kb e^g)
      dkb = mml(dkk, k, dkbe * eg[:, None], PREC)
      dk = mml(dkkt, kb, dkb * beta[:, None], PREC) + tl.load(dkin_ptr + ck)
      tl.store(dk_ptr + ck, dk.to(dk_ptr.dtype.element_ty))
      dbeta += tl.sum(dkb * kf, axis=1)
      dgw += tl.sum(dkbe * kb.to(tl.float32), axis=1)
    for t in tl.range(0, DV // BD):
      cv = rows[:, None] * DV + (t * BD + od)[None, :]
      dvb = mmf(at, tl.load(du_ptr + cv), tl.zeros((C, BD), dtype=tl.float32), FPREC)
      tl.store(dv_ptr + cv, (dvb * beta[:, None]).to(dv_ptr.dtype.element_ty))
      dbeta += tl.sum(dvb * tl.load(v_ptr + cv).to(tl.float32), axis=1)
    tl.store(dbeta_ptr + rows, dbeta)
    tl.store(dg_ptr + rows, tl.sum(dl, axis=1) - tl.sum(dl, axis=0) + eg * dgw)

  return dict(wy_fwd=wy_fwd_kernel, out=out_kernel, out_bwd=out_bwd_kernel, prod=prod_kernel, wy_bwd=wy_bwd_kernel)


# Each per-chunk kernel's launch: warps a program, the width of the D tiles its loops take, the software-pipelining
# depth of those loops -- the best of BD 16 / 32 / 64 / 128 x 4 / 8 warps x 1 / 2 / 3 stages on a GB200 at a
# Qwen3.8 device's shapes (ms: wy_fwd 1.54, out 0.36, out_bwd 1.05, prod 1.91, wy_bwd 2.08; the worst of the
# 4-warp configurations 10-60% slower, 2 warps several times so)
CHUNK_LAUNCH = dict(
    wy_fwd=dict(num_warps=4, BD=128, num_stages=1),
    out=dict(num_warps=4, BD=32, num_stages=2),
    out_bwd=dict(num_warps=4, BD=128, num_stages=1),
    prod=dict(num_warps=8, BD=64, num_stages=1),
    wy_bwd=dict(num_warps=4, BD=64, num_stages=1),
)


# The precision of the products whose operands are both fp32 (neither exact in the compute dtype), by the compute
# precision: in bf16 / f16 compute (whose results are rounded to it) 'bf16x3', two bf16 parts a side, three bf16
# products; in fp32 compute its own precision ('tf32x3', or the tests' 'ieee').
TWO_SIDED = {"bf16x3e": "bf16x3", "tf32x2e": "bf16x3"}


def _call(name, args, out_type, grid, **meta):
  import jax_triton as jt

  launch = CHUNK_LAUNCH[name]
  bd = min(launch["BD"], meta.get("DK", 1 << 30), meta.get("DV", 1 << 30))
  fprec = TWO_SIDED.get(meta["PREC"], meta["PREC"])
  return jt.triton_call(
      *args,
      kernel=_chunk_kernels()[name],
      out_type=out_type,
      grid=grid,
      name=f"gdn_{name}",
      num_warps=launch["num_warps"],
      num_stages=launch["num_stages"],
      BD=bd,
      FPREC=fprec,
      **meta,
  )


def _wy_fwd_call(k, v, beta, g, precision):
  """-> (A [B, N, H, C, C] fp32, u [.., C, Dv], w [.., C, Dk] in k's dtype): the WY factors of every chunk."""
  b, n, h, c, dk = k.shape
  dv = v.shape[-1]
  f32 = jnp.float32
  out = (
      jax.ShapeDtypeStruct((b, n, h, c, c), f32),
      jax.ShapeDtypeStruct((b, n, h, c, dv), k.dtype),
      jax.ShapeDtypeStruct((b, n, h, c, dk), k.dtype),
  )
  return _call("wy_fwd", (k, v, beta, g.astype(f32)), out, (b * n * h,), C=c, DK=dk, DV=dv, NB=c // 16, PREC=precision)


def _out_call(oi, q, k, g, vn, precision, out_dtype=jnp.float32):
  """-> o = oi + causal(q k^T e^{g_i - g_j}) vn [B, N, H, C, Dv] in out_dtype."""
  b, n, h, c, dk = q.shape
  dv = vn.shape[-1]
  return _call(
      "out",
      (oi, q, k, g.astype(jnp.float32), vn),
      jax.ShapeDtypeStruct(oi.shape, out_dtype),
      (b * n * h,),
      C=c,
      DK=dk,
      DV=dv,
      PREC=precision,
  )


def _do_exact(do, q, precision):
  # the output's cotangent, when the output is in the compute dtype, is exact in it: one-sided splits apply
  return bool(precision in EXACT_PRECISIONS and do.dtype == q.dtype)


def _out_bwd_call(q, k, g, do, vn, precision):
  """The intra-chunk term's backward -> (dvn, dq, dk, dg), fp32."""
  b, n, h, c, dk = q.shape
  dv = vn.shape[-1]
  f32 = jnp.float32
  out = (
      jax.ShapeDtypeStruct(vn.shape, f32),
      jax.ShapeDtypeStruct(q.shape, f32),
      jax.ShapeDtypeStruct(q.shape, f32),
      jax.ShapeDtypeStruct(g.shape, f32),
  )
  return _call(
      "out_bwd",
      (q, k, g.astype(f32), do, vn),
      out,
      (b * n * h,),
      C=c,
      DK=dk,
      DV=dv,
      PREC=precision,
      DO_EXACT=_do_exact(do, q, precision),
  )


def _prod_call(dvt, h, do, vn, lam, k, q, e, eg, dq_o, dk_o, precision, block_k=64, dq_dtype=jnp.float32):
  """The recurrence's per-chunk products -> (dw, dq, dk, deg, de, da), fp32 but dq (dq_dtype): dq, dk with their
  decays applied and the intra-chunk term's dq_o, dk_o added -- dq whole, dk without the WY factors' share."""
  b, n, hh, c, dk = k.shape
  dv = vn.shape[-1]
  bk = min(block_k, dk)
  nk = dk // bk
  f32 = jnp.float32
  out = (
      jax.ShapeDtypeStruct(k.shape, f32),
      jax.ShapeDtypeStruct(k.shape, dq_dtype),
      jax.ShapeDtypeStruct(k.shape, f32),
      jax.ShapeDtypeStruct((b, n, hh, nk, c), f32),
      jax.ShapeDtypeStruct((b, n, hh, nk, c), f32),
      jax.ShapeDtypeStruct((b, n, hh, nk), f32),
  )
  dw, dq, dkk, deg, de, da = _call(
      "prod",
      (dvt, h, do, vn, lam, k, q, e.astype(f32), eg.astype(f32), dq_o.astype(f32), dk_o.astype(f32)),
      out,
      (b * n * hh, nk),
      C=c,
      DK=dk,
      DV=dv,
      BK=bk,
      PREC=precision,
      DO_EXACT=_do_exact(do, q, precision),
  )
  return dw, dq, dkk, deg.sum(-2), de.sum(-2), da.sum(-1)


def _wy_bwd_call(k, v, beta, g, a, du, dw, dk_in, precision, dk_dtype=jnp.float32, dv_dtype=jnp.float32):
  """The WY factors' backward -> (dk + dk_in in dk_dtype, dv in dv_dtype, dbeta, dg fp32)."""
  b, n, h, c, dk = k.shape
  dv = v.shape[-1]
  f32 = jnp.float32
  out = (
      jax.ShapeDtypeStruct(k.shape, dk_dtype),
      jax.ShapeDtypeStruct(v.shape, dv_dtype),
      jax.ShapeDtypeStruct(g.shape, f32),
      jax.ShapeDtypeStruct(g.shape, f32),
  )
  return _call(
      "wy_bwd",
      (k, v, beta, g.astype(f32), a, du.astype(f32), dw.astype(f32), dk_in.astype(f32)),
      out,
      (b * n * h,),
      C=c,
      DK=dk,
      DV=dv,
      PREC=precision,
  )


def _decays(g):
  return jnp.exp(g[..., -1:] - g), jnp.exp(g[..., -1]), jnp.exp(g)


def _rule_fwd_impl(q, k, v, beta, g, block_v, precision, out_dtype, store):
  a_inv, u, w = _wy_fwd_call(k, v, beta, g, precision)
  e, a, eg = _decays(g)
  oi, vn, h = _fwd_call(u, w, k, e, a, q, eg, block_v, precision, store_h=store)
  return _out_call(oi, q, k, g, vn, precision, out_dtype), (a_inv, w, h, vn)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def chunk_rule(q, k, v, beta, g, block_v, precision, out_dtype):
  """The chunked rule on kernels -> o [B, N, H, C, Dv] in out_dtype.  q, k [B, N, H, C, Dk], v [.., C, Dv], beta
  [.., C] in the compute dtype, g [B, N, H, C] fp32 the decays' cumulative sum within each chunk (the reference's
  g_cum).  Its cotangent arrives in out_dtype: exact in the compute dtype when the two agree, which the backward's
  products of it then take one-sided."""
  return _rule_fwd_impl(q, k, v, beta, g, block_v, precision, out_dtype, store=False)[0]


def _chunk_rule_fwd(q, k, v, beta, g, block_v, precision, out_dtype):
  o, (a_inv, w, h, vn) = _rule_fwd_impl(q, k, v, beta, g, block_v, precision, out_dtype, store=True)
  return o, (q, k, v, beta, g, a_inv, w, h, vn)


def _chunk_rule_bwd(block_v, precision, out_dtype, res, do):
  q, k, v, beta, g, a_inv, w, h, vn = res
  e, a, eg = _decays(g)
  dvn, dq_o, dk_o, dg_o = _out_bwd_call(q, k, g, do, vn, precision)
  dvt, lam = _bwd_call(w, k, e, a, q, eg, do, dvn, block_v, precision)
  dw, dq, dk_os, deg, de, da = _prod_call(dvt, h, do, vn, lam, k, q, e, eg, dq_o, dk_o, precision, dq_dtype=q.dtype)
  dk, dv, dbeta, dg_w = _wy_bwd_call(k, v, beta, g, a_inv, dvt, dw, dk_os, precision, dk_dtype=k.dtype, dv_dtype=v.dtype)
  dg = dg_o + dg_w + deg * eg - de * e  # e^g, and e = e^{g_last - g} (its -g side)
  dg = dg.at[..., -1].add(jnp.sum(de * e, axis=-1) + da * a)  # e's g_last side and a = e^{g_last}
  return dq, dk, dv, dbeta.astype(beta.dtype), dg.astype(g.dtype)


chunk_rule.defvjp(_chunk_rule_fwd, _chunk_rule_bwd)


def _precision_for(dtype, precision, operands):
  # bf16 compute: 'bf16x3e' (q, k, w exact in bf16); f16: 'tf32x2e' (f16's 11 bits fit tf32's head, not bf16's 8);
  # fp32: 'tf32x3'
  precision = precision or {jnp.dtype(jnp.bfloat16): DEFAULT_PRECISION, jnp.dtype(jnp.float16): "tf32x2e"}.get(
      jnp.dtype(dtype), "tf32x3"
  )
  if precision not in PRECISIONS:
    raise ValueError(f"gdn_triton precision must be one of {PRECISIONS}, got {precision!r}")
  exact = (jnp.bfloat16,) if precision == "bf16x3e" else (jnp.bfloat16, jnp.float16)
  if precision in EXACT_PRECISIONS and not all(x.dtype in exact for x in operands):
    raise ValueError(
        f"gdn_triton precision {precision!r} needs q, k, w in "
        f'{" / ".join(jnp.dtype(t).name for t in exact)}, got '
        f'{" / ".join(str(x.dtype) for x in operands)}'
    )
  return precision


def chunk(q_c, k_c, v_c, beta_c, g_cum, *, block_v=None, precision=None, out_dtype=jnp.float32):
  """`chunk_gated_delta_rule` from its chunked operands on the kernels alone (`chunk_rule`) -> o [B, N, H, C, Dv]
  in out_dtype.  q_c, k_c [B, N, H, C, Dk], v_c [.., C, Dv], beta_c [.., C] (compute dtype), g_cum [.., C] fp32."""
  precision = _precision_for(q_c.dtype, precision, (q_c, k_c, v_c))
  c, dk, dv = q_c.shape[-2], q_c.shape[-1], v_c.shape[-1]
  if not fused_supported(c, dk, dv):
    raise ValueError(
        f"gdn_triton.chunk needs the head dims powers of two >= 16 and a chunk of {FUSED_CHUNKS}, got "
        f"chunk {c}, Dk {dk}, Dv {dv}"
    )
  return chunk_rule(
      q_c, k_c, v_c, beta_c, g_cum.astype(jnp.float32), block_v or default_block_v(dv), precision, jnp.dtype(out_dtype)
  )


def inter_chunk(q_c, k_c, u, w, g_cum, *, block_v=None, precision=None):
  """`chunk_gated_delta_rule`'s inter-chunk part on the kernels -> o [B, N, H, C, Dv] fp32, the scan's outputs.

  q_c, k_c [B, N, H, C, Dk] (compute dtype), u [B, N, H, C, Dv], w [B, N, H, C, Dk] (the WY factors, compute
  dtype), g_cum [B, N, H, C] fp32 -- the reference's operands before it transposes them for its scan."""
  # bf16 compute: 'bf16x3e' (q, k, w exact in bf16); f16: 'tf32x2e' (f16's 11 bits fit tf32's head, not bf16's 8);
  # fp32: 'tf32x3'
  precision = precision or {jnp.dtype(jnp.bfloat16): DEFAULT_PRECISION, jnp.dtype(jnp.float16): "tf32x2e"}.get(
      jnp.dtype(q_c.dtype), "tf32x3"
  )
  if precision not in PRECISIONS:
    raise ValueError(f"gdn_triton precision must be one of {PRECISIONS}, got {precision!r}")
  exact = (jnp.bfloat16,) if precision == "bf16x3e" else (jnp.bfloat16, jnp.float16)
  if precision in EXACT_PRECISIONS and not all(x.dtype in exact for x in (q_c, k_c, w)):
    raise ValueError(
        f"gdn_triton precision {precision!r} needs q, k, w in "
        f'{" / ".join(jnp.dtype(t).name for t in exact)}, got {q_c.dtype} / {k_c.dtype} / {w.dtype}'
    )
  c, dk, dv = q_c.shape[-2], q_c.shape[-1], u.shape[-1]
  if not supported(c, dk, dv):
    raise ValueError(
        f"gdn_scan_impl 'triton' needs the chunk and head dims powers of two >= 16, got chunk {c}, " f"Dk {dk}, Dv {dv}"
    )
  f32 = jnp.float32
  e = jnp.exp(g_cum[..., -1, None] - g_cum)  # the reference's k_dec factor
  a = jnp.exp(g_cum[..., -1])  # its state decay
  eg = jnp.exp(g_cum)  # and its q_g factor
  attn_inter, vn = scan(u, w, k_c, e, a, q_c, eg, block_v or default_block_v(dv), precision)
  causal = jnp.tril(jnp.ones((c, c), dtype=bool))
  attn = jnp.matmul(q_c, k_c.swapaxes(-1, -2), precision=_HI).astype(f32)
  gd = jnp.where(causal, g_cum[..., :, None] - g_cum[..., None, :], -1e30)
  attn = jnp.where(causal, attn * jnp.exp(gd), 0.0)
  return attn_inter + jnp.matmul(attn, vn, precision=_HI)
