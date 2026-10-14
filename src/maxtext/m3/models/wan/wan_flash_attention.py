# Copyright 2023–2025 Google LLC
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

"""Wan flash attention: fixed-row-max Pallas TPU kernel with a fused backward (MaxKernel loop, iteration 5).

Source: diffusion/mk_root/third_party/py/accelerator_agents/MaxKernel/workspace/iter5/optimized.py. Changes for the
model: residuals are (q, k, v, o, lse hi/lo) with o and lse tagged by `residual_checkpoint_name` (remat offload, like
splash's), and the backward rebuilds its augmented operands from them.

Public API
----------
  o = wan_flash_attention(q, k, v, *, seq_real=None, scale=None, residual_checkpoint_name=None)

  q, k, v: [B, H, S, D] (D = 128; any B, H, S; bf16 recommended, other dtypes are cast to bf16 and o is cast back).
  seq_real: static Python int, 1 <= seq_real <= S (default S). Tokens with index >= seq_real are padding.
  scale: optional softmax scale applied to q before the kernel (default None: q is already scaled by the caller,
    as in the Wan model / the splash baseline).
  Differentiable w.r.t. q, k, v (jax.custom_vjp, fused Pallas backward).

Padding semantics (segment-id free, identical to splash with segment ids seg = (arange(S) < seq_real)):
  * real rows (i < seq_real) attend only to real keys; padding keys get exactly zero weight, so real-row outputs and
    every gradient flowing into real tokens are exact;
  * padding rows attend only to padding keys (their outputs are well defined but meaningless; mask them in the loss).
  The kernel needs a length S_pad that is a multiple of 128 with a good block divisor. If S_pad > S, zero tokens are
  appended internally (they join the padding segment, so they can change the *padding* rows' outputs but never the
  real ones) and sliced off again. S_pad = argmin over 128-multiples in [round_up(S, 128), round_up(S, 2048)] of
  S_pad^2 * q(block), where block = the largest multiple of 128 <= 4736 dividing S_pad (q = 1.0 for block >= 3072,
  1.04 >= 2048, 1.12 >= 1024, 1.3 >= 512, 1.6 otherwise). Examples: S = 75,600 or 75,776 -> 75,776 (block
  4736 = 37 * 128, the benchmark config); 32,760 -> 32,768 (4096); 4,096 -> 4,096; 1,024 -> 1,024.
  Limits: the backward keeps the whole-head f32 dq (S_pad * 512 B) resident in VMEM, so S_pad <= ~150k.

Algorithm (unchanged from iterations 3-4)
  * Forward with a fixed per-row max m_i = min(Cauchy-Schwarz bound_i, est_i + EST_MARGIN), est_i = exact max over
    NS = 512 sampled real keys: no running max / accumulator rescale. The row sum l_i >= exp(-EST_MARGIN) by
    construction; an exact detector (log2 l outside [-100, 100], non-finite accumulator, NaN) switches to an exact
    online-softmax fallback (lax.cond) otherwise. o is normalised by the sum of the same bf16 p that multiplied V; lse
    keeps the f32 row sum.
  * Masking, -m / -lse and segment bookkeeping are folded into an augmented K = 256 contraction (head_dim 128
    half-fills the 256-wide v6e MXU anyway): s = [q, -m, 0, a, 1-a] . [k, 1, 1, -BIG(1-b), -BIG b].
  * Forward: per grid step the chunk loop writes bf16 P^T into VMEM, then O^T += [V^T; 1] P^T is one long-K matmul.
    Backward: fused dK/dV/dQ, whole-head f32 dq resident in VMEM, long-K dV^T, chunked dK^T and dQ.
  Both kernels are MXU bound; 100 MiB scoped VMEM.

Benchmark shape: (1, 10, 75776, 128) bf16 with seq_real = 75,600 (Wan 2.1 14B 720p, Ulysses 4 -> 10 heads/chip).
"""

import contextlib
import contextvars
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HEAD_DIM = 128

# Trace-time number of real tokens when the model padded its token sequence for this kernel (see seq_real_scope).
_SEQ_REAL = contextvars.ContextVar("wan_flash_seq_real", default=None)


@contextlib.contextmanager
def seq_real_scope(seq_real):
  """While tracing inside this scope, attention calls treat tokens >= seq_real as padding (None: no override)."""
  token = _SEQ_REAL.set(seq_real)
  try:
    yield
  finally:
    _SEQ_REAL.reset(token)


def current_seq_real():
  """The seq_real set by the innermost seq_real_scope, or None."""
  return _SEQ_REAL.get()

MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)
NUM_LANES = 128
AUG = 2 * HEAD_DIM  # augmented contraction width
VROWS = HEAD_DIM + 8  # rows of the augmented V^T: [v^T; 1; 1; 0 x 6]
BIG = 16384.0  # 2^14, exact in bf16; exp(-BIG) == 0
# Fall back to the online-softmax forward if any row's log2(sum p) under the fixed max is below this (or above its
# negation, or the O accumulator is not finite).
UNDERFLOW_LOG2 = -100.0
# m_i = min(Cauchy-Schwarz bound_i, est_i + EST_MARGIN): l_i >= exp(true_max_i - m_i) >= exp(-EST_MARGIN), so
# log2 l >= -57.7 (underflow impossible); log2 l <= 100 needs true_max_i - est_i < 58 + EST_MARGIN = 98 nats.
EST_MARGIN = 40.0
NS = 512  # sampled real keys per head: per strided group, the largest-norm key and the one most aligned with mu
USE_EST = True  # False: Cauchy-Schwarz bound only (iteration-3 behaviour)
FORCE_FALLBACK = False  # testing only: always run the online-softmax forward
O_BF16_NORM = True  # o = (sum bf16(p) v) / (sum bf16(p)) (consistent rounding; see the forward _finalize)

# Block-size caps: each block is the largest multiple of 128 that divides S_pad and is <= the cap. At S_pad = 75,776
# (= 2^11 * 37) this gives 4736 = 37 * 128 for the forward and the backward q blocks and 2048 for the backward kv.
BQ = 4736  # forward q block
BKV = 4736  # forward kv block
BKV_COMPUTE = 512  # forward chunk (static loop, last chunk may be shorter)
BQ_B = 4736  # backward q block
BKV_B = 2048  # backward kv block
BQ_COMPUTE_B = 768  # backward chunk
PREP_BLOCK = 4736  # prep kernels (v^T, do augmentation)
FB_BQ, FB_BKV, FB_BKVC = 1024, 2048, 1024  # online-softmax fallback

VMEM_MIB = 100
VMEM_LIMIT = VMEM_MIB * 1024 * 1024
# Long-K dV^T needs a (bq_b, bkv_b) bf16 P scratch next to the whole-head f32 dq; drop it (chunked dV^T, ~1 % slower)
# when both would exceed this budget; error out if dq alone exceeds DQ_VMEM_MAX.
BIG_DV_VMEM_MAX = 60 * 1024 * 1024
DQ_VMEM_MAX = 76 * 1024 * 1024

NT_DIMS = (((1,), (1,)), ((), ()))  # a @ b.T
NN_DIMS = (((1,), (0,)), ((), ()))  # a @ b


# ----------------------------------------------------------------------------
# Static per-shape configuration
# ----------------------------------------------------------------------------
class Config(NamedTuple):
  seq_real: int
  s: int  # padded sequence length (multiple of 128)
  bq: int
  bkv: int
  bkv_compute: int
  bq_b: int
  bkv_b: int
  bq_compute_b: int
  prep: int
  fb_bq: int
  fb_bkv: int
  fb_bkvc: int
  big_dv: bool


def _round_up(x, m):
  return (x + m - 1) // m * m


def _block(s, cap):
  """Largest multiple of 128 that divides s (a multiple of 128) and is <= cap (at least 128)."""
  for b in range(max(min(cap, s) // 128, 1) * 128, 0, -128):
    if s % b == 0:
      return b
  raise ValueError(f"no 128-multiple block divides {s}")


def _block_quality(b):
  if b >= 3072:
    return 1.0
  if b >= 2048:
    return 1.04
  if b >= 1024:
    return 1.12
  if b >= 512:
    return 1.3
  return 1.6


def padded_len(s):
  """Padded kernel length for a caller length s (see the module docstring)."""
  s0 = _round_up(s, 128)
  hi = _round_up(s0, 2048)
  return min(range(s0, hi + 1, 128), key=lambda p: (p * p * _block_quality(_block(p, BQ)), p))


def make_config(s, seq_real):
  """Static config for caller length s and seq_real real tokens (reads the module tunables at call time)."""
  if not 1 <= seq_real <= s:
    raise ValueError(f"seq_real={seq_real} must be in [1, {s}]")
  sp = padded_len(s)
  if sp * HEAD_DIM * 4 > DQ_VMEM_MAX:
    raise ValueError(f"padded sequence {sp} too long: the backward keeps the whole-head f32 dq in VMEM")
  bq_b = _block(sp, BQ_B)
  bkv_b = _block(sp, BKV_B)
  fb_bkv = _block(sp, FB_BKV)
  return Config(
      seq_real=seq_real, s=sp, bq=_block(sp, BQ), bkv=_block(sp, BKV), bkv_compute=BKV_COMPUTE, bq_b=bq_b,
      bkv_b=bkv_b, bq_compute_b=BQ_COMPUTE_B, prep=_block(sp, PREP_BLOCK), fb_bq=_block(sp, FB_BQ), fb_bkv=fb_bkv,
      fb_bkvc=_block(fb_bkv, FB_BKVC),
      big_dv=sp * HEAD_DIM * 4 + bq_b * bkv_b * 2 <= BIG_DV_VMEM_MAX)


def _kv_tail(kv_start, rows, cols, kv_axis, seq_real):
  """Augmented-k tail: [1, 1, -BIG*(1-b_j), -BIG*b_j, 0...] along the non-kv axis (b_j = kv j is padding)."""
  shape = (rows, cols)
  c = lax.broadcasted_iota(jnp.int32, shape, 1 - kv_axis)
  j = kv_start + lax.broadcasted_iota(jnp.int32, shape, kv_axis)
  b = (j >= seq_real).astype(jnp.float32)
  t = jnp.where(c < 2, 1.0, 0.0)
  t = jnp.where(c == 2, -BIG * (1.0 - b), t)
  t = jnp.where(c == 3, -BIG * b, t)
  return t.astype(jnp.bfloat16)


# ----------------------------------------------------------------------------
# Forward kernel
# ----------------------------------------------------------------------------
def kernel(kst_ref, mu_ref, ks_ref, mpad_ref, q_ref, k_ref, vt_ref, o_ref, qb_ref, qt_ref, flag_ref, qat_scr,
           ka_scr, pt_scr, acc_scr, l_scr, *, seq_real, bq, bkv, bkv_compute):
  """Forward for one (head, q block, kv block) grid step.

  kst_ref: (H, 4) f32 SMEM: [max ||k_j||, max ||k_j - mu||] over the real keys, then the same over the pad keys.
  mu_ref: (8, D) bf16, row 0 = mean real key mu, row 1 = mean pad key mu' (rows 2..7 zero).
  ks_ref: (NS, D) bf16 sampled real keys: est_i = max_j q_i.ks_j is a lower bound of the real rows' true max.
  mpad_ref: (1, bq) f32: exact row max (+1) for the pad rows (pad rows only see the pad keys, so it is computed
    exactly outside); unused for real rows.
  q_ref: (bq, D) bf16; k_ref: (bkv, D) bf16; vt_ref: (136, bkv) bf16 = [v^T; 1; 1; 0...]
    (Transfer: HBM -> VMEM via BlockSpec).
  Outputs: o_ref (bq, D) bf16; qb_ref (bq, 256) bf16 = backward q operand [q, -lse_hi, -lse_lo, a, 1-a,
    0...]; qt_ref (D, bq) bf16 = q^T; flag_ref (8, 128) f32 = min / max_i log2(l_i) (fallback detector).
  qat_scr: (256, bq) bf16 = [q^T; -m; 0; a; 1-a; 0...] built at kv block 0.
  ka_scr: (bkv, 256) bf16 = [k, 1, 1, -BIG(1-b), -BIG b, 0...]. pt_scr: (bkv, bq) bf16 = P^T of this step.
  acc_scr: (136, bq) f32 = [O^T; l; l; ...]. l_scr: (8, bq) f32 partial f32 row sums.
  """
  h = pl.program_id(0)
  qi = pl.program_id(1)
  ki = pl.program_id(2)
  nk = pl.num_programs(2)

  @pl.when(ki == 0)
  def _init():
    qst = q_ref[...].astype(jnp.float32).T.astype(jnp.bfloat16)  # (D, bq)
    qsf = qst.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(qsf * qsf, axis=0, keepdims=True))  # (1, bq)
    qmus = lax.dot_general(mu_ref[...], qst, NN_DIMS, preferred_element_type=jnp.float32)  # (8, bq)
    i1 = qi * bq + lax.broadcasted_iota(jnp.int32, (1, bq), 1)
    pad_row = i1 >= seq_real  # pad rows only see pad keys, real rows only real keys
    qmu = jnp.where(pad_row, qmus[1:2, :], qmus[0:1, :])
    b1 = norm * jnp.where(pad_row, kst_ref[h, 2], kst_ref[h, 0])  # ||q|| max ||k||
    rn = norm * jnp.where(pad_row, kst_ref[h, 3], kst_ref[h, 1])
    b2 = qmu + rn  # q.mu + ||q|| max ||k - mu||
    # Both are upper bounds of every logit of the row; pad each by 2^-6 of its magnitude (rounding) and +1.
    bound = jnp.minimum(b1 * (1.0 + 2.0**-6), b2 + (jnp.abs(qmu) + rn) * 2.0**-6) + 1.0
    if USE_EST:
      # est = max over the sampled real keys of the exact logits (lower bound of the row max); m only has to be
      # within the exp range of the true max, so m = min(bound, est + EST_MARGIN) (see EST_MARGIN).
      est = None
      for c in range(0, NS, 128):
        sc = lax.dot_general(ks_ref[c:c + 128, :], qst, NN_DIMS, preferred_element_type=jnp.float32)  # (128, bq)
        mc = jnp.max(sc, axis=0, keepdims=True)
        est = mc if est is None else jnp.maximum(est, mc)
      bound = jnp.minimum(bound, est + EST_MARGIN)
    neg_m = jnp.where(pad_row, -mpad_ref[...], -bound)  # (1, bq)
    r = lax.broadcasted_iota(jnp.int32, (HEAD_DIM, bq), 0)
    i = qi * bq + lax.broadcasted_iota(jnp.int32, (HEAD_DIM, bq), 1)
    a = (i >= seq_real).astype(jnp.float32)
    tail = jnp.where(r == 0, neg_m, 0.0)
    tail = jnp.where(r == 2, a, tail)
    tail = jnp.where(r == 3, 1.0 - a, tail)
    qat_scr[:HEAD_DIM, :] = qst
    qat_scr[HEAD_DIM:, :] = tail.astype(jnp.bfloat16)
    acc_scr[...] = jnp.zeros_like(acc_scr)
    l_scr[...] = jnp.zeros_like(l_scr)

  # The kv tail depends on the kv block only through its padding: write it at block 0 (fresh q block) and for every
  # block that contains padding (padding is a suffix, so all blocks after the first padded one are rewritten).
  @pl.when(jnp.logical_or(ki == 0, (ki + 1) * bkv > seq_real))
  def _write_tail():
    ka_scr[:, HEAD_DIM:] = _kv_tail(ki * bkv, bkv, HEAD_DIM, 0, seq_real)

  ka_scr[:, :HEAD_DIM] = k_ref[...]

  # Static (unrolled) chunk loop; the last chunk may be shorter when bkv_compute does not divide bkv
  # (e.g. bkv = 4736 = 37 * 128).
  for start in range(0, bkv, bkv_compute):
    size = min(bkv_compute, bkv - start)
    ka = ka_scr[start:start + size, :]  # (size, 256)
    s_t = lax.dot_general(ka, qat_scr[...], NN_DIMS, preferred_element_type=jnp.float32)  # (size, bq)
    p_f = jnp.exp(s_t)
    # Row sum of the f32 p (sum over kv = sublane axis here: plain vreg adds, the VPU has slack).
    l_scr[...] += jnp.sum(p_f.reshape(size // 8, 8, p_f.shape[1]), axis=0)
    pt_scr[start:start + size, :] = p_f.astype(jnp.bfloat16)

  acc_scr[...] += lax.dot_general(vt_ref[...], pt_scr[...], NN_DIMS,
                                  preferred_element_type=jnp.float32)  # (136, bq), K = bkv

  @pl.when(ki == nk - 1)
  def _finalize():
    l = jnp.sum(l_scr[...], axis=0, keepdims=True)  # (1, bq), sum of f32 p (as the baseline's l)
    # Normalise O by the sum of the SAME bf16-rounded p that multiplied V (MXU ones row of [V^T; 1]): with m != row
    # max, bf16(p_max) is not exactly 1, and dividing by the f32 sum would leave a 2^-9 per-row scale error that the
    # backward's di = rowsum(o * do) amplifies into dq / dk. lse keeps the f32 l (exact p in the bwd).
    l_o = acc_scr[HEAD_DIM:HEAD_DIM + 1, :] if O_BF16_NORM else l
    o_ref[...] = (acc_scr[:HEAD_DIM, :] * (1.0 / l_o)).T.astype(o_ref.dtype)
    log2l = jnp.log2(l)
    neg_m = qat_scr[HEAD_DIM:HEAD_DIM + 1, :].astype(jnp.float32)  # exactly the -m used
    lse = jnp.log(l) - neg_m  # (1, bq)
    hi = lse.astype(jnp.bfloat16).astype(jnp.float32)
    lo = (lse - hi).astype(jnp.bfloat16).astype(jnp.float32)
    tail = qat_scr[HEAD_DIM:, :].astype(jnp.float32)  # (128, bq)
    r = lax.broadcasted_iota(jnp.int32, tail.shape, 0)
    tail = jnp.where(r == 0, -hi, jnp.where(r == 1, -lo, tail))
    qb_ref[:, :HEAD_DIM] = qat_scr[:HEAD_DIM, :].astype(jnp.float32).T.astype(qb_ref.dtype)
    qb_ref[:, HEAD_DIM:] = tail.T.astype(qb_ref.dtype)
    qt_ref[...] = q_ref[...].astype(jnp.float32).T.astype(qt_ref.dtype)
    rows = lax.broadcasted_iota(jnp.int32, flag_ref.shape, 0)
    i1 = qi * bq + lax.broadcasted_iota(jnp.int32, log2l.shape, 1)
    min_real = jnp.min(jnp.where(i1 < seq_real, log2l, jnp.inf))
    # Sublane 0: min_i log2(l_i); 1: max_i; 2..7: min over the real rows only (diagnostics).
    # The accumulator must be finite too (huge |v| with a large l could overflow O; NaN compares False -> inf).
    acc_ok = jnp.max(jnp.abs(acc_scr[...])) < jnp.inf
    max_l = jnp.where(acc_ok, jnp.max(log2l), jnp.inf)
    flag_ref[...] = jnp.where(rows == 0, jnp.min(log2l), jnp.where(rows == 1, max_l, min_real))


def _flash_fwd_fixed(kst, mu, ks, mpad, q, k, vt, cfg):
  h, s, d = q.shape
  bq, bkv = cfg.bq, cfg.bkv
  nq = s // bq
  return pl.pallas_call(
      functools.partial(kernel, seq_real=cfg.seq_real, bq=bq, bkv=bkv, bkv_compute=cfg.bkv_compute),
      grid=(h, nq, s // bkv),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec((None, 8, d), lambda hi, qi, ki: (hi, 0, 0)),
          pl.BlockSpec((None, NS, d), lambda hi, qi, ki: (hi, 0, 0)),
          pl.BlockSpec((None, 1, bq), lambda hi, qi, ki: (hi, 0, qi)),
          pl.BlockSpec((None, bq, d), lambda hi, qi, ki: (hi, qi, 0)),
          pl.BlockSpec((None, bkv, d), lambda hi, qi, ki: (hi, ki, 0)),
          pl.BlockSpec((None, VROWS, bkv), lambda hi, qi, ki: (hi, 0, ki)),
      ],
      out_specs=[
          pl.BlockSpec((None, bq, d), lambda hi, qi, ki: (hi, qi, 0)),
          pl.BlockSpec((None, bq, AUG), lambda hi, qi, ki: (hi, qi, 0)),
          pl.BlockSpec((None, d, bq), lambda hi, qi, ki: (hi, 0, qi)),
          pl.BlockSpec((None, 8, NUM_LANES), lambda hi, qi, ki: (hi, qi, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((h, s, d), q.dtype),
          jax.ShapeDtypeStruct((h, s, AUG), q.dtype),
          jax.ShapeDtypeStruct((h, d, s), q.dtype),
          jax.ShapeDtypeStruct((h, nq * 8, NUM_LANES), jnp.float32),
      ],
      scratch_shapes=[
          pltpu.VMEM((AUG, bq), jnp.bfloat16),
          pltpu.VMEM((bkv, AUG), jnp.bfloat16),
          pltpu.VMEM((bkv, bq), jnp.bfloat16),
          pltpu.VMEM((VROWS, bq), jnp.float32),
          pltpu.VMEM((8, bq), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=VMEM_LIMIT,
      ),
  )(kst, mu, ks, mpad, q, k, vt)


# ----------------------------------------------------------------------------
# Fallback forward (online softmax), used only if the fixed max would under/overflow.
# ----------------------------------------------------------------------------
def _segment_mask(q_start, kv_start, shape, q_axis, seq_real):
  q_ids = q_start + lax.broadcasted_iota(jnp.int32, shape, q_axis)
  kv_ids = kv_start + lax.broadcasted_iota(jnp.int32, shape, 1 - q_axis)
  return (q_ids < seq_real) == (kv_ids < seq_real)


def _online_kernel(q_ref, k_ref, v_ref, o_ref, lse_ref, m_scr, l_scr, acc_scr, *, seq_real, bq, bkv, bkv_compute):
  """Online-softmax forward; same logits bf16(q) @ k^T as the fixed-max kernel."""
  qi = pl.program_id(1)
  ki = pl.program_id(2)
  nk = pl.num_programs(2)

  @pl.when(ki == 0)
  def _init():
    m_scr[...] = jnp.full_like(m_scr, MASK_VALUE)
    l_scr[...] = jnp.zeros_like(l_scr)
    acc_scr[...] = jnp.zeros_like(acc_scr)

  repeats = bkv_compute // NUM_LANES

  def body(c, carry):
    start = pl.multiple_of(c * bkv_compute, bkv_compute)
    k = k_ref[pl.ds(start, bkv_compute), :]
    v = v_ref[pl.ds(start, bkv_compute), :]
    s = lax.dot_general(q_ref[...], k, NT_DIMS, preferred_element_type=jnp.float32)
    mask = _segment_mask(qi * bq, ki * bkv + start, s.shape, 0, seq_real)
    s = jnp.where(mask, s, MASK_VALUE)
    m_prev = m_scr[...]
    l_prev = l_scr[...]
    m_next = jnp.maximum(m_prev, s.max(axis=-1)[:, None])
    p = jnp.exp(s - jnp.tile(m_next, (1, repeats)))
    alpha = jnp.exp(m_prev - m_next)
    l_scr[...] = lax.broadcast_in_dim(p.sum(axis=-1), l_prev.shape, (0,)) + alpha * l_prev
    m_scr[...] = m_next
    pv = lax.dot_general(p.astype(v.dtype), v, NN_DIMS, preferred_element_type=jnp.float32)
    acc_scr[...] = alpha * acc_scr[...] + pv
    return carry

  lax.fori_loop(0, bkv // bkv_compute, body, None, unroll=True)

  @pl.when(ki == nk - 1)
  def _finalize():
    l = l_scr[...]
    o_ref[...] = (acc_scr[...] * (1.0 / l)).astype(o_ref.dtype)
    lse_ref[...] = m_scr[...] + jnp.log(l)


def _flash_fwd_online(q, k, v, cfg):
  h, s, d = q.shape
  bq, bkv, bkvc = cfg.fb_bq, cfg.fb_bkv, cfg.fb_bkvc
  o, lse = pl.pallas_call(
      functools.partial(_online_kernel, seq_real=cfg.seq_real, bq=bq, bkv=bkv, bkv_compute=bkvc),
      grid=(h, s // bq, s // bkv),
      in_specs=[
          pl.BlockSpec((None, bq, d), lambda hi, qi, ki: (hi, qi, 0)),
          pl.BlockSpec((None, bkv, d), lambda hi, qi, ki: (hi, ki, 0)),
          pl.BlockSpec((None, bkv, d), lambda hi, qi, ki: (hi, ki, 0)),
      ],
      out_specs=[
          pl.BlockSpec((None, bq, d), lambda hi, qi, ki: (hi, qi, 0)),
          pl.BlockSpec((None, bq, NUM_LANES), lambda hi, qi, ki: (hi, qi, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((h, s, d), q.dtype),
          jax.ShapeDtypeStruct((h, s, NUM_LANES), jnp.float32),
      ],
      scratch_shapes=[
          pltpu.VMEM((bq, NUM_LANES), jnp.float32),
          pltpu.VMEM((bq, NUM_LANES), jnp.float32),
          pltpu.VMEM((bq, d), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
          vmem_limit_bytes=VMEM_LIMIT,
      ),
  )(q, k, v)
  return o, lse[:, :, :1]  # (H, S, 1) natural-log logsumexp


# ----------------------------------------------------------------------------
# Fused backward kernel
# ----------------------------------------------------------------------------
def bwd_kernel(qb_ref, qt_ref, doa_ref, dot_ref, k_ref, vt_ref, dq_hbm, dk_ref, dv_ref,
               kat_scr, vat_scr, dq_scr, dkt_scr, dvt_scr, p_scr, dq_stage, sem, *, seq_real, bq, bkv, bq_compute,
               big_dv):
  """Fused backward for one (head, kv block, q block) grid step.

  Scores are (q, kv). The chunk loop (bq_compute q rows at a time) computes S = qb @ kat (= s - lse - mask),
  P = exp(S), dP - di = doa @ vat and dS = P (dP - di), and does dK^T += Q^T dS and dQ += dS K per chunk; with
  big_dv the bf16 P of the whole (bq, bkv) step is kept in VMEM and dV^T += dO^T P is one K = bq matmul.
  qb_ref: (bq, 256) bf16 [q, -lse_hi, -lse_lo, a, 1-a, 0...]; qt_ref: (D, bq) bf16 q^T;
  doa_ref: (bq, 256) bf16 [do, -di_hi, -di_lo, 0...]; dot_ref: (D, bq) bf16 do^T;
  k_ref: (bkv, D) bf16; vt_ref: (136, bkv) bf16 [v^T; 1; 1; 0...]  (Transfer: HBM -> VMEM via BlockSpec).
  dq_hbm: (H, S, D) bf16 in HBM (memory_space=ANY), written by manual DMA on the last kv block.
  dk_ref, dv_ref: (bkv, D) bf16 outputs written at the last q block.
  kat_scr: (256, bkv) bf16 = [k^T; 1; 1; -BIG(1-b); -BIG b; 0...]; vat_scr: (256, bkv) bf16 = [v^T; 1; 1; 0...].
  dq_scr: (S, D) f32 whole-head dq accumulator resident in VMEM. dkt_scr, dvt_scr: (D, bkv) f32.
  p_scr: (bq, bkv) bf16 P of this step (big_dv). dq_stage: (bq, D) bf16 DMA staging buffer.
  """
  h = pl.program_id(0)
  ki = pl.program_id(1)
  qi = pl.program_id(2)
  nk = pl.num_programs(1)
  nq = pl.num_programs(2)

  @pl.when(qi == 0)
  def _init():
    dkt_scr[...] = jnp.zeros_like(dkt_scr)
    dvt_scr[...] = jnp.zeros_like(dvt_scr)
    kat_scr[:HEAD_DIM, :] = k_ref[...].astype(jnp.float32).T.astype(kat_scr.dtype)
    kat_scr[HEAD_DIM:, :] = _kv_tail(ki * bkv, HEAD_DIM, bkv, 1, seq_real)
    vat_scr[:VROWS, :] = vt_ref[...]
    vat_scr[VROWS:, :] = jnp.zeros((AUG - VROWS, bkv), vat_scr.dtype)

  first_kv = ki == 0

  # Static (unrolled) chunk loop; the last chunk may be shorter when bq_compute does not divide bq.
  for start in range(0, bq, bq_compute):
    size = min(bq_compute, bq - start)
    qa = qb_ref[start:start + size, :]  # (size, 256)
    da = doa_ref[start:start + size, :]  # (size, 256)
    s = lax.dot_general(qa, kat_scr[...], NN_DIMS, preferred_element_type=jnp.float32)  # s - lse - mask
    p = jnp.exp(s)  # (size, bkv) f32
    p_b = p.astype(jnp.bfloat16)
    if big_dv:
      p_scr[start:start + size, :] = p_b
    else:
      dvt_scr[...] += lax.dot_general(dot_ref[:, start:start + size], p_b, NN_DIMS,
                                      preferred_element_type=jnp.float32)  # (D, bkv)
    dp = lax.dot_general(da, vat_scr[...], NN_DIMS, preferred_element_type=jnp.float32)  # dP - di
    ds_b = (p * dp).astype(jnp.bfloat16)  # (size, bkv)
    dkt_scr[...] += lax.dot_general(qt_ref[:, start:start + size], ds_b, NN_DIMS,
                                    preferred_element_type=jnp.float32)  # (D, bkv)
    dq_c = lax.dot_general(ds_b, k_ref[...], NN_DIMS, preferred_element_type=jnp.float32)  # (size, D)
    row = pl.multiple_of(qi * bq + start, 128 if start % 128 == 0 else 8)
    prev = dq_scr[pl.ds(row, size), :]
    dq_scr[pl.ds(row, size), :] = jnp.where(first_kv, 0.0, prev) + dq_c

  if big_dv:
    dvt_scr[...] += lax.dot_general(dot_ref[...], p_scr[...], NN_DIMS,
                                    preferred_element_type=jnp.float32)  # (D, bkv), K = bq

  @pl.when(qi == nq - 1)
  def _write_dkv():
    dk_ref[...] = dkt_scr[...].T.astype(dk_ref.dtype)
    dv_ref[...] = dvt_scr[...].T.astype(dv_ref.dtype)

  @pl.when(ki == nk - 1)
  def _write_dq():
    row = pl.multiple_of(qi * bq, bq)
    dq_stage[...] = dq_scr[pl.ds(row, bq), :].astype(dq_stage.dtype)  # (bq, D)
    copy = pltpu.make_async_copy(dq_stage, dq_hbm.at[h, pl.ds(row, bq), :], sem)
    copy.start()
    copy.wait()


def _flash_bwd(qb, qt, doa, dot, k, vt, cfg):
  h, s, d = k.shape
  bq, bkv = cfg.bq_b, cfg.bkv_b
  return pl.pallas_call(
      functools.partial(bwd_kernel, seq_real=cfg.seq_real, bq=bq, bkv=bkv, bq_compute=cfg.bq_compute_b,
                        big_dv=cfg.big_dv),
      grid=(h, s // bkv, s // bq),
      in_specs=[
          pl.BlockSpec((None, bq, AUG), lambda hi, ki, qi: (hi, qi, 0)),
          pl.BlockSpec((None, d, bq), lambda hi, ki, qi: (hi, 0, qi)),
          pl.BlockSpec((None, bq, AUG), lambda hi, ki, qi: (hi, qi, 0)),
          pl.BlockSpec((None, d, bq), lambda hi, ki, qi: (hi, 0, qi)),
          pl.BlockSpec((None, bkv, d), lambda hi, ki, qi: (hi, ki, 0)),
          pl.BlockSpec((None, VROWS, bkv), lambda hi, ki, qi: (hi, 0, ki)),
      ],
      out_specs=[
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec((None, bkv, d), lambda hi, ki, qi: (hi, ki, 0)),
          pl.BlockSpec((None, bkv, d), lambda hi, ki, qi: (hi, ki, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((h, s, d), k.dtype),
          jax.ShapeDtypeStruct((h, s, d), k.dtype),
          jax.ShapeDtypeStruct((h, s, d), k.dtype),
      ],
      scratch_shapes=[
          pltpu.VMEM((AUG, bkv), jnp.bfloat16),
          pltpu.VMEM((AUG, bkv), jnp.bfloat16),
          pltpu.VMEM((s, d), jnp.float32),
          pltpu.VMEM((d, bkv), jnp.float32),
          pltpu.VMEM((d, bkv), jnp.float32),
          pltpu.VMEM((bq, bkv) if cfg.big_dv else (16, 128), jnp.bfloat16),
          pltpu.VMEM((bq, d), k.dtype),
          pltpu.SemaphoreType.DMA(()),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
          vmem_limit_bytes=VMEM_LIMIT,
      ),
  )(qb, qt, doa, dot, k, vt)


# ----------------------------------------------------------------------------
# Prep kernels and statistics
# ----------------------------------------------------------------------------
def _hi_lo(x):
  """Splits f32 x into bf16 hi + lo. hi truncates via bit masking: XLA's excess-precision simplification would
  otherwise fold f32 -> bf16 -> f32 into a no-op and make lo == 0."""
  hi_f = lax.bitcast_convert_type(lax.bitcast_convert_type(x, jnp.uint32) & jnp.uint32(0xFFFF0000), jnp.float32)
  hi = hi_f.astype(jnp.bfloat16)  # exact
  lo = (x - hi_f).astype(jnp.bfloat16)
  return hi, lo


def _vt_kernel(v_ref, vt_ref):
  """v (blk, D) -> vt (136, blk) = [v^T; 1; 1; 0 x 6]."""
  vt_ref[:HEAD_DIM, :] = v_ref[...].astype(jnp.float32).T.astype(vt_ref.dtype)
  r = lax.broadcasted_iota(jnp.int32, (VROWS - HEAD_DIM, vt_ref.shape[1]), 0)
  vt_ref[HEAD_DIM:, :] = jnp.where(r < 2, 1.0, 0.0).astype(vt_ref.dtype)


def _vt_aug(v, cfg):
  """(H, S, D) -> (H, 136, S) = [v^T; 1; 1; 0 x 6] (Pallas, avoids XLA relayout copies)."""
  h, s, d = v.shape
  blk = cfg.prep
  return pl.pallas_call(
      _vt_kernel,
      grid=(h, s // blk),
      in_specs=[pl.BlockSpec((None, blk, d), lambda hi, i: (hi, i, 0))],
      out_specs=pl.BlockSpec((None, VROWS, blk), lambda hi, i: (hi, 0, i)),
      out_shape=jax.ShapeDtypeStruct((h, VROWS, s), v.dtype),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
  )(v)


def _do_kernel(o_ref, do_ref, doa_ref, dot_ref):
  """o, do (blk, D) -> doa (blk, 256) = [do, -di_hi, -di_lo, 0...], dot (D, blk) = do^T; di = rowsum(o*do)."""
  do = do_ref[...]
  dof = do.astype(jnp.float32)
  di = jnp.sum(o_ref[...].astype(jnp.float32) * dof, axis=1, keepdims=True)  # (blk, 1)
  hi = di.astype(jnp.bfloat16).astype(jnp.float32)
  lo = (di - hi).astype(jnp.bfloat16).astype(jnp.float32)
  lane = lax.broadcasted_iota(jnp.int32, dof.shape, 1)
  tail = jnp.where(lane == 0, -hi, jnp.where(lane == 1, -lo, 0.0))
  doa_ref[:, :HEAD_DIM] = do
  doa_ref[:, HEAD_DIM:] = tail.astype(doa_ref.dtype)
  dot_ref[...] = dof.T.astype(dot_ref.dtype)


def _do_aug(o, do, cfg):
  h, s, d = do.shape
  blk = cfg.prep
  return pl.pallas_call(
      _do_kernel,
      grid=(h, s // blk),
      in_specs=[pl.BlockSpec((None, blk, d), lambda hi, i: (hi, i, 0))] * 2,
      out_specs=[
          pl.BlockSpec((None, blk, AUG), lambda hi, i: (hi, i, 0)),
          pl.BlockSpec((None, d, blk), lambda hi, i: (hi, 0, i)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((h, s, AUG), do.dtype),
          jax.ShapeDtypeStruct((h, d, s), do.dtype),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
  )(o, do)


def _q_kernel(q_ref, nl_ref, qb_ref, qt_ref, *, seq_real, blk):
  """q (blk, D), -lse hi/lo (blk, 2) -> qb (blk, 256) = [q, -lse_hi, -lse_lo, a, 1-a, 0...], qt (D, blk) = q^T."""
  q = q_ref[...]
  nl = nl_ref[...].astype(jnp.float32)  # (blk, 2)
  lane = lax.broadcasted_iota(jnp.int32, (blk, HEAD_DIM), 1)
  row = pl.program_id(1) * blk + lax.broadcasted_iota(jnp.int32, (blk, HEAD_DIM), 0)
  a = (row >= seq_real).astype(jnp.float32)
  tail = jnp.where(lane == 0, nl[:, 0:1], jnp.where(lane == 1, nl[:, 1:2], 0.0))
  tail = jnp.where(lane == 2, a, jnp.where(lane == 3, 1.0 - a, tail))
  qb_ref[:, :HEAD_DIM] = q
  qb_ref[:, HEAD_DIM:] = tail.astype(qb_ref.dtype)
  qt_ref[...] = q.astype(jnp.float32).T.astype(qt_ref.dtype)


def _q_aug(q, neg_lse, cfg):
  """Rebuilds the backward's q operands from the residuals (Pallas, replaces an XLA concat + transpose)."""
  h, s, d = q.shape
  blk = cfg.prep
  return pl.pallas_call(
      functools.partial(_q_kernel, seq_real=cfg.seq_real, blk=blk),
      grid=(h, s // blk),
      in_specs=[
          pl.BlockSpec((None, blk, d), lambda hi, i: (hi, i, 0)),
          pl.BlockSpec((None, blk, 2), lambda hi, i: (hi, i, 0)),
      ],
      out_specs=[
          pl.BlockSpec((None, blk, AUG), lambda hi, i: (hi, i, 0)),
          pl.BlockSpec((None, d, blk), lambda hi, i: (hi, 0, i)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((h, s, AUG), q.dtype),
          jax.ShapeDtypeStruct((h, d, s), q.dtype),
      ],
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
  )(q, neg_lse)


def _pad_row_max(qp, kp):
  """(H, n, D) x (H, n, D) bf16 -> (H, n) f32 exact row max of qp @ kp^T, in row chunks if the scores are large."""
  h, n, d = qp.shape
  if h * n * n * 4 <= (1 << 28):
    return jnp.max(jnp.einsum("hqd,hkd->hqk", qp, kp, preferred_element_type=jnp.float32), axis=-1)
  c = 1024
  npad = _round_up(n, c)
  qc = jnp.pad(qp, ((0, 0), (0, npad - n), (0, 0))).reshape(h, npad // c, c, d).transpose(1, 0, 2, 3)
  mc = lax.map(lambda x: jnp.max(jnp.einsum("hqd,hkd->hqk", x, kp, preferred_element_type=jnp.float32), axis=-1),
               qc)  # (npad / c, H, c)
  return mc.transpose(1, 0, 2).reshape(h, npad)[:, :n]


def _key_stats(q, k, cfg):
  """Per-head statistics for the fixed row max.

  Real rows only see the real keys: kst (H, 4) f32 = [max ||k_j||, max ||k_j - mu||] over the real keys (+ the same
  over the pad keys), inflated by 2^-7 relative to absorb rounding; mu8 (H, 8, D) bf16 with row 0 = mu (real mean
  key), row 1 = mu' (pad mean). ks (H, NS, D): sampled real keys. Pad rows only see the pad keys, so their row max is
  computed exactly here: mpad (H, 1, S) f32 = max_j s_ij + 1 at the pad rows (0 at the real rows).
  """
  h, s, d = k.shape
  sr = cfg.seq_real
  n_pad = s - sr
  # Reads bf16 k directly (no f32 copy of k): ||k_j - mu||^2 = ||k_j||^2 - 2 k_j.mu + ||mu||^2 in f32, with a
  # 2^-12 ||k||^2 absolute pad for the cancellation (all products of bf16 values are exact in f32).
  mu = jnp.mean(k[:, :sr, :], axis=1, dtype=jnp.float32).astype(jnp.bfloat16)  # (H, D)
  if n_pad:
    mup = jnp.mean(k[:, sr:, :], axis=1, dtype=jnp.float32).astype(jnp.bfloat16)
  else:
    mup = jnp.zeros((h, d), jnp.bfloat16)
  mu8 = jnp.concatenate([mu[:, None, :], mup[:, None, :], jnp.zeros((h, 6, d), jnp.bfloat16)], axis=1)
  k2 = jnp.einsum("hsd,hsd->hs", k, k, preferred_element_type=jnp.float32)  # (H, S)
  kmu = jnp.einsum("hsd,hd->hs", k, mu, preferred_element_type=jnp.float32)  # (H, S)

  def stats(k2s, kmus, mu_b):
    mu2 = jnp.sum(jnp.square(mu_b.astype(jnp.float32)), axis=-1, keepdims=True)
    k2max = jnp.max(k2s, axis=-1)
    kmax = jnp.sqrt(k2max)  # (H,)
    r2 = jnp.max(k2s - 2.0 * kmus + mu2, axis=-1)
    rmax = jnp.sqrt(jnp.maximum(r2, 0.0) + k2max * 2.0**-12)
    return kmax, rmax

  km, rm = stats(k2[:, :sr], kmu[:, :sr], mu)
  if n_pad:
    kmup = jnp.einsum("hsd,hd->hs", k[:, sr:, :], mup, preferred_element_type=jnp.float32)
    kmp, rmp = stats(k2[:, sr:], kmup, mup)
  else:
    kmp = rmp = jnp.zeros((h,), jnp.float32)
  kst = jnp.stack([km, rm, kmp, rmp], axis=-1) * (1.0 + 2.0**-7)  # (H, 4)
  # Sampled real keys for the row-max estimate: split the keys into NS/2 strided groups (key j in group j % ng, which
  # keeps the reduction lane-dense) and take, per group, the largest-norm key and the key with the largest k.mu (pad
  # keys excluded). Any real keys give a valid lower bound; these pick the likely maximisers (large-norm outlier /
  # sink keys, keys along the shared mean direction). Indices of groups without real keys are clamped to a real key.
  ng = NS // 2
  sg = _round_up(s, ng)
  real = (jnp.arange(s) < sr)[None, :]
  idx = []
  for score in (k2, kmu):
    sc = jnp.where(real, score, -jnp.inf)
    if sg > s:
      sc = jnp.pad(sc, ((0, 0), (0, sg - s)), constant_values=-jnp.inf)
    idx.append(jnp.argmax(sc.reshape(h, sg // ng, ng), axis=1) * ng + jnp.arange(ng)[None, :])  # (H, ng)
  idx = jnp.minimum(jnp.concatenate(idx, axis=1), sr - 1)  # (H, NS)
  ks = jnp.take_along_axis(k, idx[:, :, None], axis=1)  # (H, NS, D) bf16
  if n_pad:
    mp = _pad_row_max(q[:, sr:, :], k[:, sr:, :]) + 1.0  # (H, n_pad)
    mpad = jnp.concatenate([jnp.zeros((h, sr), jnp.float32), mp], axis=1)[:, None, :]
  else:
    mpad = jnp.zeros((h, 1, s), jnp.float32)
  return kst, mu8, ks, mpad


def _flag_stats(flag):
  """flag (H, nq*8, 128): per q block sublane 0 = min_i log2(l_i), 1 = max_i, 2 = min over real rows.
  Returns (min, max, min over real rows)."""
  f = flag.reshape(flag.shape[0], -1, 8, NUM_LANES)
  return jnp.min(f[:, :, 0, :]), jnp.max(f[:, :, 1, :]), jnp.min(f[:, :, 2, :])


# ----------------------------------------------------------------------------
# Forward driver and custom_vjp wiring (inputs (B*H, S_pad, D) bf16)
# ----------------------------------------------------------------------------
def _forward(q, k, v, cfg):
  """Returns o (H,S,D) bf16 and the residuals (qb, qt, vt)."""
  h, s, d = q.shape
  vt = _vt_aug(v, cfg)
  stats = _key_stats(q, k, cfg)
  o, qb, qt, flag = _flash_fwd_fixed(*stats, q, k, vt, cfg)
  fmin, fmax, _ = _flag_stats(flag)
  # Underflow (row sum too small for the fixed max), overflow (m too far below the true max, or a non-finite O
  # accumulator) or NaN -> exact online-softmax fallback.
  bad = jnp.logical_not(jnp.logical_and(fmin >= UNDERFLOW_LOG2, fmax <= -UNDERFLOW_LOG2))
  if FORCE_FALLBACK:
    bad = jnp.array(True)

  def fallback(args):
    q, k, v, _, _ = args
    o2, lse = _flash_fwd_online(q, k, v, cfg)
    hi, lo = _hi_lo(lse)
    a = jnp.broadcast_to((jnp.arange(s) >= cfg.seq_real).astype(jnp.bfloat16)[None, :, None], (h, s, 1))
    qb2 = jnp.concatenate([q, -hi, -lo, a, 1 - a, jnp.zeros((h, s, d - 4), jnp.bfloat16)], axis=-1)
    return o2, qb2

  o, qb = lax.cond(bad, fallback, lambda args: (args[3], args[4]), (q, k, v, o, qb))
  return o, (qb, qt, vt)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _attention(q, k, v, cfg, residual_checkpoint_name):
  del residual_checkpoint_name
  o, _ = _forward(q, k, v, cfg)
  return o


def _attention_fwd(q, k, v, cfg, residual_checkpoint_name):
  o, (qb, _, _) = _forward(q, k, v, cfg)
  # Keep only (q, k, v, o, -lse hi/lo): the augmented backward operands are rebuilt in the backward (cheap), so a
  # remat policy that saves / offloads the named residuals moves about one activation's worth instead of six.
  neg_lse = qb[:, :, HEAD_DIM:HEAD_DIM + 2]
  if residual_checkpoint_name is not None:
    o = checkpoint_name(o, residual_checkpoint_name)
    neg_lse = checkpoint_name(neg_lse, residual_checkpoint_name)
  return o, (q, k, v, o, neg_lse)


def _attention_bwd(cfg, residual_checkpoint_name, res, do):
  del residual_checkpoint_name
  q, k, v, o, neg_lse = res
  qb, qt = _q_aug(q, neg_lse, cfg)
  vt = _vt_aug(v, cfg)
  doa, dot = _do_aug(o, do, cfg)
  return _flash_bwd(qb, qt, doa, dot, k, vt, cfg)


_attention.defvjp(_attention_fwd, _attention_bwd)


def _prepare(q, k, v, seq_real):
  """[B, H, S, D] -> (B*H, S_pad, D) bf16 (zero-padded), config."""
  b, h, s, d = q.shape
  if d != HEAD_DIM:
    raise ValueError("head_dim must be %d, got %d" % (HEAD_DIM, d))
  if k.shape != q.shape or v.shape != q.shape:
    raise ValueError(f"q, k, v shapes must match: {q.shape} {k.shape} {v.shape}")
  cfg = make_config(s, s if seq_real is None else int(seq_real))
  xs = []
  for x in (q, k, v):
    x = x.astype(jnp.bfloat16).reshape(b * h, s, d)
    if cfg.s > s:
      x = jnp.pad(x, ((0, 0), (0, cfg.s - s), (0, 0)))
    xs.append(x)
  return xs, cfg


def wan_flash_attention(q, k, v, *, seq_real=None, scale=None, residual_checkpoint_name=None):
  """Non-causal flash attention over [B, H, S, 128] with a padding suffix (tokens >= seq_real); see module doc.

  Args:
    q, k, v: [B, H, S, 128] arrays (bf16; other dtypes are cast to bf16, o is returned in q's dtype).
    seq_real: static int, number of real (non-padding) tokens; default S.
    scale: optional softmax scale multiplied into q (default None: q already scaled).
    residual_checkpoint_name: optional `checkpoint_name` for the residuals o and lse (remat save / offload).
  Returns:
    o: [B, H, S, 128]. Differentiable w.r.t. q, k, v (fused Pallas backward).
  """
  b, h, s, d = q.shape
  dtype = q.dtype
  if scale is not None:
    q = (q.astype(jnp.float32) * scale).astype(jnp.bfloat16)
  (qf, kf, vf), cfg = _prepare(q, k, v, seq_real)
  o = _attention(qf, kf, vf, cfg, residual_checkpoint_name)
  if cfg.s > s:
    o = o[:, :s, :]
  return o.reshape(b, h, s, d).astype(dtype)


def fallback_flag(q, k, v, seq_real=None):
  """Diagnostics: (min_i log2(l_i), max_i log2(l_i), min over real rows) under the fixed max; the fallback runs
  iff min < UNDERFLOW_LOG2 or max > -UNDERFLOW_LOG2 (or either is NaN)."""
  (q, k, v), cfg = _prepare(q, k, v, seq_real)
  stats = _key_stats(q, k, cfg)
  _, _, _, flag = _flash_fwd_fixed(*stats, q, k, _vt_aug(v, cfg), cfg)
  return _flag_stats(flag)
