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

"""Fused Pallas TPU kernel for the Gated DeltaNet chunked delta rule (training).

Design Note:
- Grid: (batch, head_group, chunk) for fwd, (batch, head_group, 2 * chunk) for bwd.
- VMEM budget per step: v6e has ~128 MiB physical VMEM. The default scoped limit is
  ~16-32 MiB. The forward state scratch is (hb, K, V). Backward needs state scratches
  hs_scr (num_chunks, hb, K, V) ~ 1-2 MB which easily fits. We use default compiler params.
- Recomputation: To avoid large HBM traffic, we recompute the chunk-local variables
  (S, A, u, w) from the base inputs (q, k, v, g, beta) in the backward pass. Stage 1
  math is fully batched across heads (using hb as batch dimension) so the 256x256 MXU
  is highly utilized.
- Why it beats the XLA path: The XLA fast path separates Stage 1 (chunk-local S, u, w
  and (I+S)^-1 via block-diagonal Neumann and off-diagonal fix) and Stage 3 (inter-chunk
  recurrence). Intermediate tensors (S, A, u, w, and their gradients) are written to HBM.
  This fused kernel keeps all chunk-local intermediates in VMEM, uses batched MXU math
  across heads (hb), and pipelines the chunk sequence. The batched 1-pass bf16 inverse
  reuses the tril_inverse math but fills the MXU efficiently.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_TN = (((1,), (1,)), ((0,), (0,)))
_NT = (((2,), (2,)), ((0,), (0,)))
_NN = (((2,), (1,)), ((0,), (0,)))

_VMEM_LIMIT_BYTES = 110 * 1024 * 1024
_BWD_STATE_SCRATCH_BUDGET = 64 * 1024 * 1024


def _bmm(a, b, dims):
  """Batched bf16 x bf16 -> fp32 matmul. a and b are (hb, M, K) and (hb, N, K) etc."""
  return jax.lax.dot_general(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), dims, preferred_element_type=jnp.float32)


def _inverse_batched(s, eye, diag_block, num_blocks, block):
  """Returns (I + S)^-1 for strictly lower-triangular S of shape (hb, C, C), fp32."""
  n = jnp.where(diag_block, -s, 0.0)
  x = eye + n
  p = n
  step = 1
  while step * 2 < block:
    p = _bmm(p, p, _NN)
    x = _bmm(x, eye + p, _NN)
    step *= 2
  dinv = x
  if num_blocks == 1:
    return dinv
  m = _bmm(dinv, jnp.where(diag_block, 0.0, s), _NN)
  t = eye - m
  q = -m
  step = 1
  while step * 2 < num_blocks:
    q = _bmm(q, q, _NN)
    t = _bmm(t, eye + q, _NN)
    step *= 2
  return _bmm(t, dinv, _NN)


def _chunk_cumsum(g):
  """Inclusive cumsum over the last axis of (hb, C) fp32 via a triangular matmul.

  Mosaic has no lowering for cumsum, so use g @ U with U[j, i] = (j <= i).
  """
  c = g.shape[-1]
  r = jax.lax.broadcasted_iota(jnp.int32, (c, c), 0)
  col = jax.lax.broadcasted_iota(jnp.int32, (c, c), 1)
  upper = (r <= col).astype(jnp.float32)
  return jax.lax.dot_general(
      g.astype(jnp.float32),
      upper,
      (((1,), (0,)), ((), ())),
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )


def _fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    o_ref,
    h_scr,
    *,
    hb,
    repeats,
    chunk_size,
    inverse_block_size,
    use_qk_norm,
    scale_q,
    eps,
):
  """Pallas forward kernel for fused Gated DeltaNet stage 2 and recurrence."""

  @pl.when(pl.program_id(2) == 0)
  def _():
    h_scr[...] = jnp.zeros_like(h_scr)

  q = q_ref[0].swapaxes(0, 1)
  k = k_ref[0].swapaxes(0, 1)
  if use_qk_norm:
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    inv_q = jax.lax.rsqrt(jnp.sum(q_f32 * q_f32, axis=-1, keepdims=True) + eps)
    inv_k = jax.lax.rsqrt(jnp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + eps)
    q = (q_f32 * inv_q).astype(q_ref.dtype)
    k = (k_f32 * inv_k).astype(k_ref.dtype)
  if scale_q:
    scale = jax.lax.rsqrt(jnp.array(q.shape[-1], dtype=jnp.float32)).astype(q_ref.dtype)
    q = q * scale
  if repeats > 1:
    q = jnp.repeat(q, repeats, axis=0)
    k = jnp.repeat(k, repeats, axis=0)
  v = v_ref[0].swapaxes(0, 1)
  g = g_ref[0, 0].swapaxes(0, 1)
  beta = beta_ref[0, 0].swapaxes(0, 1)

  gcc = _chunk_cumsum(g)
  k_beta = k * beta[..., None]
  v_beta = v * beta[..., None]

  S = _bmm(k_beta, k, _NT)

  r = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  c = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  mask = r > c
  causal_mask = r >= c

  g_diff = jnp.where(causal_mask, gcc[..., :, None] - gcc[..., None, :], -1e30)

  S = S * jnp.exp(g_diff)
  S = jnp.where(mask, S, 0.0)

  eye = (r == c).astype(jnp.float32)
  diag_block = (r // inverse_block_size) == (c // inverse_block_size)

  A = _inverse_batched(S, eye, diag_block, chunk_size // inverse_block_size, inverse_block_size)

  u = _bmm(A, v_beta, _NN)
  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(gcc)[..., None]
  w = _bmm(A, k_beta_g, _NN)

  gl = gcc[:, chunk_size - 1 : chunk_size]
  h = h_scr[...]

  w_h = _bmm(w, h, _NN)
  vn = u - w_h

  qk = _bmm(q, k, _NT)
  p = qk * jnp.exp(g_diff)
  p = jnp.where(causal_mask, p, 0.0)

  q_g = q.astype(jnp.float32) * jnp.exp(gcc)[..., None]
  qg_h = _bmm(q_g, h, _NN)
  p_vn = _bmm(p, vn, _NN)
  o = qg_h + p_vn

  o_ref[0] = o.swapaxes(0, 1).astype(o_ref.dtype)

  kd = k.astype(jnp.float32) * jnp.exp(gl - gcc)[..., None]
  h_scr[...] = jnp.exp(gl)[..., None] * h + _bmm(kd, vn, _TN)


def _bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    do_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dg_ref,
    dbeta_ref,
    hs_scr,
    a_scr,
    h_scr,
    dh_scr,
    *,
    hb,
    repeats,
    chunk_size,
    inverse_block_size,
    num_chunks,
    use_qk_norm,
    scale_q,
    eps,
):
  """Pallas backward kernel for fused Gated DeltaNet stage 2 and recurrence."""
  s = pl.program_id(2)

  r = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 0)
  c = jax.lax.broadcasted_iota(jnp.int32, (chunk_size, chunk_size), 1)
  mask = r > c
  causal_mask = r >= c
  eye = (r == c).astype(jnp.float32)
  diag_block = (r // inverse_block_size) == (c // inverse_block_size)
  causal_f32 = causal_mask.astype(jnp.float32)

  @pl.when(s == 0)
  def _():
    h_scr[...] = jnp.zeros_like(h_scr)

  @pl.when(s < num_chunks)
  def _():
    k = k_ref[0].swapaxes(0, 1)
    if use_qk_norm:
      k_f32 = k.astype(jnp.float32)
      inv_k = jax.lax.rsqrt(jnp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + eps)
      k = (k_f32 * inv_k).astype(k_ref.dtype)
    if repeats > 1:
      k = jnp.repeat(k, repeats, axis=0)
    v = v_ref[0].swapaxes(0, 1)
    g = g_ref[0, 0].swapaxes(0, 1)
    beta = beta_ref[0, 0].swapaxes(0, 1)

    gcc = _chunk_cumsum(g)
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]

    S = _bmm(k_beta, k, _NT)
    g_diff = jnp.where(causal_mask, gcc[..., :, None] - gcc[..., None, :], -1e30)
    S = S * jnp.exp(g_diff)
    S = jnp.where(mask, S, 0.0)

    A = _inverse_batched(S, eye, diag_block, chunk_size // inverse_block_size, inverse_block_size)
    a_scr[s] = A.astype(a_scr.dtype)

    u = _bmm(A, v_beta, _NN)
    k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(gcc)[..., None]
    w = _bmm(A, k_beta_g, _NN)

    hs_scr[s] = h_scr[...].astype(hs_scr.dtype)

    gl = gcc[:, chunk_size - 1 : chunk_size]
    h = h_scr[...]
    w_h = _bmm(w, h, _NN)
    vn = u - w_h
    kd = k.astype(jnp.float32) * jnp.exp(gl - gcc)[..., None]
    h_scr[...] = jnp.exp(gl)[..., None] * h + _bmm(kd, vn, _TN)

  @pl.when(s == num_chunks)
  def _():
    dh_scr[...] = jnp.zeros_like(dh_scr)

  @pl.when(s >= num_chunks)
  def _():
    i = 2 * num_chunks - 1 - s

    q_raw = q_ref[0].swapaxes(0, 1)
    k_raw = k_ref[0].swapaxes(0, 1)
    if use_qk_norm:
      q_f32 = q_raw.astype(jnp.float32)
      k_f32 = k_raw.astype(jnp.float32)
      inv_q = jax.lax.rsqrt(jnp.sum(q_f32 * q_f32, axis=-1, keepdims=True) + eps)
      inv_k = jax.lax.rsqrt(jnp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + eps)
      q_hat = q_f32 * inv_q
      k_hat = k_f32 * inv_k
      q = q_hat.astype(q_ref.dtype)
      k = k_hat.astype(k_ref.dtype)
    else:
      q = q_raw
      k = k_raw
    if scale_q:
      scale_val = jax.lax.rsqrt(jnp.array(q.shape[-1], dtype=jnp.float32)).astype(q_ref.dtype)
      q = q * scale_val
    if repeats > 1:
      q = jnp.repeat(q, repeats, axis=0)
      k = jnp.repeat(k, repeats, axis=0)
    v = v_ref[0].swapaxes(0, 1)
    g = g_ref[0, 0].swapaxes(0, 1)
    beta = beta_ref[0, 0].swapaxes(0, 1)
    do = do_ref[0].swapaxes(0, 1)

    gcc = _chunk_cumsum(g)
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]

    S = _bmm(k_beta, k, _NT)
    g_diff = jnp.where(causal_mask, gcc[..., :, None] - gcc[..., None, :], -1e30)
    S = S * jnp.exp(g_diff)
    S = jnp.where(mask, S, 0.0)

    A = a_scr[i]

    u = _bmm(A, v_beta, _NN)
    k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(gcc)[..., None]
    w = _bmm(A, k_beta_g, _NN)

    h = hs_scr[i].astype(jnp.float32)
    dh = dh_scr[...]

    gl = gcc[:, chunk_size - 1 : chunk_size]
    eg = jnp.exp(gcc)[..., None]
    ek = jnp.exp(gl - gcc)[..., None]
    egl = jnp.exp(gl)[..., None]

    w_h = _bmm(w, h, _NN)
    vn = u - w_h
    qk = _bmm(q, k, _NT)
    p = qk * jnp.exp(g_diff)
    p = jnp.where(causal_mask, p, 0.0)

    q_g = q.astype(jnp.float32) * eg
    kd = k.astype(jnp.float32) * ek

    dvn = _bmm(p, do, _TN) + _bmm(kd, dh, _NN)
    dp = _bmm(do, vn, _NT) * jnp.exp(g_diff)
    dp = jnp.where(causal_mask, dp, 0.0)
    do_h = _bmm(do, h, _NT)
    vn_dh = _bmm(vn, dh, _NT)

    dq_rec = _bmm(dp, k, _NN) + do_h * eg
    dk_rec = _bmm(dp, q, _TN) + vn_dh * ek
    dw = -_bmm(dvn, h, _NT)
    du = dvn

    dpq = dp * qk
    t_kd = vn_dh * kd
    s_tkd = jnp.sum(t_kd, axis=-1)

    dg_rec = jnp.sum(do_h * q_g, axis=-1) + jnp.sum(dpq, axis=-1) - jnp.sum(dpq, axis=-2) - s_tkd
    dg_last = jnp.sum(s_tkd, axis=-1, keepdims=True) + jnp.exp(gl) * jnp.sum(
        jnp.sum(dh * h, axis=-1), axis=-1, keepdims=True
    )

    dh_scr[...] = egl * dh + _bmm(q_g, do, _TN) - _bmm(w, du, _TN)

    dv_beta = _bmm(A, du, _TN)
    dk_beta_g = _bmm(A, dw, _TN)

    dA = _bmm(du, v_beta, _NT) + _bmm(dw, k_beta_g, _NT)
    t_A = _bmm(A, dA, _TN)
    dS = -_bmm(t_A, A, _NT)
    dS = jnp.where(mask, dS, 0.0)

    dS_pre = dS * jnp.exp(g_diff)
    dk_beta_S = _bmm(dS_pre, k, _NN)
    dk_S = _bmm(dS_pre, k_beta, _TN)

    dk_beta = dk_beta_g * eg + dk_beta_S

    dg_w = jnp.sum(dk_beta_g * k_beta_g, axis=-1)
    dg_S_diff = dS * S
    dg_S_i = jnp.sum(dg_S_diff, axis=-1)
    dg_S_j = -jnp.sum(dg_S_diff, axis=-2)

    dg_stage1 = dg_w + dg_S_i + dg_S_j
    dg_cc = dg_rec + dg_stage1

    dg = (
        jax.lax.dot_general(
            dg_cc.astype(jnp.float32),
            causal_f32,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        + dg_last
    )

    dv = dv_beta * beta[..., None]
    dbeta_v = jnp.sum(dv_beta * v, axis=-1)
    dbeta_k = jnp.sum(dk_beta * k, axis=-1)
    dbeta = dbeta_v + dbeta_k

    dk = dk_rec + dk_beta * beta[..., None] + dk_S
    if repeats > 1:
      hb_k = hb // repeats
      dq_rec = dq_rec.reshape(hb_k, repeats, chunk_size, -1).sum(axis=1)
      dk = dk.reshape(hb_k, repeats, chunk_size, -1).sum(axis=1)
    if scale_q:
      dq_rec = dq_rec * scale_val.astype(jnp.float32)
    if use_qk_norm:
      dq_rec = inv_q * (dq_rec - q_hat * jnp.sum(dq_rec * q_hat, axis=-1, keepdims=True))
      dk = inv_k * (dk - k_hat * jnp.sum(dk * k_hat, axis=-1, keepdims=True))

    dq_ref[0] = dq_rec.swapaxes(0, 1).astype(dq_ref.dtype)
    dk_ref[0] = dk.swapaxes(0, 1).astype(dk_ref.dtype)
    dv_ref[0] = dv.swapaxes(0, 1).astype(dv_ref.dtype)
    dg_ref[0, 0] = dg.swapaxes(0, 1).astype(dg_ref.dtype)
    dbeta_ref[0, 0] = dbeta.swapaxes(0, 1).astype(dbeta_ref.dtype)


def _largest_divisor_at_most(n, cap):
  """Returns the largest divisor of n that is at most cap."""
  cap = max(1, min(cap, n))
  while n % cap:
    cap -= 1
  return cap


def _params():
  """Returns TPU compiler parameters for the fused GDN Pallas kernel."""
  return pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"), vmem_limit_bytes=_VMEM_LIMIT_BYTES
  )


def _in_specs(hb_k, hb, chunk_size, dims, chunk_of):
  """Builds Pallas input BlockSpecs for q, k, v, g, and beta."""
  d_q, d_k, vdim = dims
  specs = [
      pl.BlockSpec((1, chunk_size, hb_k, d_q), lambda b, g, s: (b, chunk_of(s), g, 0)),
      pl.BlockSpec((1, chunk_size, hb_k, d_k), lambda b, g, s: (b, chunk_of(s), g, 0)),
      pl.BlockSpec((1, chunk_size, hb, vdim), lambda b, g, s: (b, chunk_of(s), g, 0)),
  ]
  # g and beta use the (B, H // hb, T, hb) layout from _gb_layout, so the
  # last two block dims are (chunk_size, hb) = (multiple of 8, full dim).
  specs.append(_gb_spec(hb, chunk_size, chunk_of))
  specs.append(_gb_spec(hb, chunk_size, chunk_of))
  return specs


def _gb_spec(hb, chunk_size, chunk_of):
  """Builds a BlockSpec for g or beta in (B, H // hb, T, hb) layout."""
  return pl.BlockSpec((1, 1, chunk_size, hb), lambda b, g, s: (b, g, chunk_of(s), 0))


def _gb_layout(x, hb):
  """(B, T, H) -> (B, H // hb, T, hb)."""
  b, t, h = x.shape
  return x.reshape(b, t, h // hb, hb).transpose(0, 2, 1, 3)


def _gb_unlayout(x):
  """(B, H // hb, T, hb) -> (B, T, H)."""
  b, ng, t, hb = x.shape
  return x.transpose(0, 2, 1, 3).reshape(b, t, ng * hb)


def _forward(q, k, v, g, beta, chunk_size, heads_per_block, inverse_block_size, use_qk_norm, scale_q, eps, interpret):
  """Dispatches the forward Pallas kernel for fused Gated DeltaNet."""
  b, t, h_k, d = q.shape
  h_v = v.shape[2]
  vdim = v.shape[-1]
  repeats = h_v // h_k
  n = t // chunk_size
  hb_k = _largest_divisor_at_most(h_k, max(1, heads_per_block // repeats))
  hb = hb_k * repeats

  def ident(s):
    return s

  return pl.pallas_call(
      functools.partial(
          _fwd_kernel,
          hb=hb,
          repeats=repeats,
          chunk_size=chunk_size,
          inverse_block_size=inverse_block_size,
          use_qk_norm=use_qk_norm,
          scale_q=scale_q,
          eps=eps,
      ),
      out_shape=jax.ShapeDtypeStruct((b, t, h_v, vdim), q.dtype),
      grid=(b, h_v // hb, n),
      in_specs=_in_specs(hb_k, hb, chunk_size, [d, d, vdim], ident),
      out_specs=pl.BlockSpec((1, chunk_size, hb, vdim), lambda bi, g, s: (bi, s, g, 0)),
      scratch_shapes=[pltpu.VMEM((hb, d, vdim), jnp.float32)],
      compiler_params=None if interpret else _params(),
      interpret=interpret,
      name="gdn_fused_fwd",
  )(q, k, v, _gb_layout(g, hb), _gb_layout(beta, hb))


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11))
def _gdn_fused(q, k, v, g, beta, chunk_size, heads_per_block, inverse_block_size, use_qk_norm, scale_q, eps, interpret):
  """Custom VJP wrapper for the fused Gated DeltaNet kernel."""
  return _forward(q, k, v, g, beta, chunk_size, heads_per_block, inverse_block_size, use_qk_norm, scale_q, eps, interpret)


def _gdn_fused_fwd(
    q, k, v, g, beta, chunk_size, heads_per_block, inverse_block_size, use_qk_norm, scale_q, eps, interpret
):
  """Forward pass for custom_vjp saving inputs as residuals."""
  return (
      _forward(q, k, v, g, beta, chunk_size, heads_per_block, inverse_block_size, use_qk_norm, scale_q, eps, interpret),
      (q, k, v, g, beta),
  )


def _gdn_fused_bwd(chunk_size, heads_per_block, inverse_block_size, use_qk_norm, scale_q, eps, interpret, res, do):
  """Backward pass for custom_vjp using the 2N-step Pallas kernel."""
  q, k, v, g, beta = res
  b, t, h_k, d = q.shape
  h_v = v.shape[2]
  vdim = v.shape[-1]
  repeats = h_v // h_k
  n = t // chunk_size
  state_bytes = n * d * vdim * 2
  bwd_budget = _BWD_STATE_SCRATCH_BUDGET if repeats > 1 else _BWD_STATE_SCRATCH_BUDGET // 2
  bwd_cap = min(heads_per_block, max(1, bwd_budget // state_bytes))
  hb_k = _largest_divisor_at_most(h_k, max(1, bwd_cap // repeats))
  hb = hb_k * repeats

  def in_chunk(s):
    return jnp.where(s < n, s, 2 * n - 1 - s)

  def out_chunk(s):
    # Outputs are written only in phase 2. During phase 1 keep the block index
    # on chunk N-1 (the first chunk phase 2 writes), so nothing is flushed early.
    return jnp.where(s < n, n - 1, 2 * n - 1 - s)

  in_specs = _in_specs(hb_k, hb, chunk_size, [d, d, vdim], in_chunk)
  in_specs.append(pl.BlockSpec((1, chunk_size, hb, vdim), lambda bi, g, s: (bi, in_chunk(s), g, 0)))

  out_specs = [
      pl.BlockSpec((1, chunk_size, hb_k, d), lambda bi, g, s: (bi, out_chunk(s), g, 0)),
      pl.BlockSpec((1, chunk_size, hb_k, d), lambda bi, g, s: (bi, out_chunk(s), g, 0)),
      pl.BlockSpec((1, chunk_size, hb, vdim), lambda bi, g, s: (bi, out_chunk(s), g, 0)),
  ]
  out_specs.append(_gb_spec(hb, chunk_size, out_chunk))
  out_specs.append(_gb_spec(hb, chunk_size, out_chunk))

  out_shape = [
      jax.ShapeDtypeStruct((b, t, h_k, d), q.dtype),
      jax.ShapeDtypeStruct((b, t, h_k, d), k.dtype),
      jax.ShapeDtypeStruct((b, t, h_v, vdim), v.dtype),
      jax.ShapeDtypeStruct((b, h_v // hb, t, hb), g.dtype),
      jax.ShapeDtypeStruct((b, h_v // hb, t, hb), beta.dtype),
  ]
  dq, dk, dv, dg, dbeta = pl.pallas_call(
      functools.partial(
          _bwd_kernel,
          hb=hb,
          repeats=repeats,
          chunk_size=chunk_size,
          inverse_block_size=inverse_block_size,
          num_chunks=n,
          use_qk_norm=use_qk_norm,
          scale_q=scale_q,
          eps=eps,
      ),
      out_shape=out_shape,
      grid=(b, h_v // hb, 2 * n),
      in_specs=in_specs,
      out_specs=out_specs,
      scratch_shapes=[
          pltpu.VMEM((n, hb, d, vdim), jnp.bfloat16),
          pltpu.VMEM((n, hb, chunk_size, chunk_size), jnp.bfloat16),
          pltpu.VMEM((hb, d, vdim), jnp.float32),
          pltpu.VMEM((hb, d, vdim), jnp.float32),
      ],
      compiler_params=None if interpret else _params(),
      interpret=interpret,
      name="gdn_fused_bwd",
  )(q, k, v, _gb_layout(g, hb), _gb_layout(beta, hb), do.astype(q.dtype))

  return dq, dk, dv, _gb_unlayout(dg), _gb_unlayout(dbeta)


_gdn_fused.defvjp(_gdn_fused_fwd, _gdn_fused_bwd)


def _resolve_interpret(interpret):
  return jax.default_backend() != "tpu" if interpret is None else interpret


def gdn_fused_bhtd(
    q,
    k,
    v,
    g,
    beta,
    *,
    chunk_size: int,
    heads_per_block: int = 4,
    inverse_block_size: int = 16,
    use_qk_norm: bool = False,
    scale_q: bool = False,
    eps: float = 1e-6,
    interpret: bool | None = None,
) -> jax.Array:
  """q, k, v, g, beta are all in (B, T, H, D) or (B, T, H) format."""
  if q.shape[1] % chunk_size:
    raise ValueError(f"T={q.shape[1]} must be a multiple of chunk_size={chunk_size}")
  resolved_interpret = _resolve_interpret(interpret)
  h_k, h_v = q.shape[2], v.shape[2]
  if h_v > h_k:
    if h_v % h_k:
      raise ValueError(f"v heads {h_v} must be a multiple of q/k heads {h_k}")
    repeats = h_v // h_k
    n = q.shape[1] // chunk_size
    state_bytes = n * q.shape[-1] * v.shape[-1] * 2
    bwd_cap = min(heads_per_block, max(1, _BWD_STATE_SCRATCH_BUDGET // state_bytes))
    hb_k_fwd = _largest_divisor_at_most(h_k, max(1, heads_per_block // repeats))
    hb_k_bwd = _largest_divisor_at_most(h_k, max(1, bwd_cap // repeats))
    if not resolved_interpret and ((hb_k_fwd < h_k and hb_k_fwd % 8 != 0) or (hb_k_bwd < h_k and hb_k_bwd % 8 != 0)):
      q = jnp.repeat(q, repeats, axis=2)
      k = jnp.repeat(k, repeats, axis=2)
  return _gdn_fused(
      q, k, v, g, beta, chunk_size, heads_per_block, inverse_block_size, use_qk_norm, scale_q, eps, resolved_interpret
  )
