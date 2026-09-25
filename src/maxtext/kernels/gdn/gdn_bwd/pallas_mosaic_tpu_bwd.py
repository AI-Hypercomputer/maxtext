# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas emit_pipeline GDN backward kernel on Mosaic TPU."""

import dataclasses
import functools
from typing import Any, Optional

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .. import compute_conv1d as local_compute_conv1d
from .bwd_memory_ref import make_bwd_block_specs
from .runtime_utils import ensure_cpu_interpret_registered
from .runtime_utils import pallas_unsupported_reason
from .runtime_utils import warn_gdn_pallas_fallback_once

# Default number of value heads per grid step, by activation dtype.
_F32_HEAD_TILE = 16
_BF16_HEAD_TILE = 32


def _gdn_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    keep_lhs_fp32: bool = False,
    precise: bool = False,
) -> jax.Array:
  """Batched MXU matmul with FP32 accumulation for the GDN backward kernel.

  For bf16 activations the default is a single-pass bf16 MXU matmul. With
  `precise=True` both operands stay f32 and the matmul runs at
  `Precision.HIGH` (3-pass bf16). The intra-chunk matmuls need this: the
  gating adjoint sums `q * dq - k * dk` over products that nearly cancel, and
  bf16-rounded operands leave the A_log / dt_bias gradients with the wrong
  direction (cosine ~0.3 against an fp32 reference). For f32 activations every
  matmul runs at `Precision.HIGHEST`.
  """
  if jnp.dtype(compute_dtype) == jnp.bfloat16:
    if precise:
      lhs_dtype = rhs_dtype = jnp.float32
      precision = jax.lax.Precision.HIGH
    else:
      lhs_dtype = jnp.float32 if keep_lhs_fp32 else jnp.bfloat16
      rhs_dtype = jnp.bfloat16
      precision = jax.lax.Precision.DEFAULT
    return jax.lax.dot_general(
        lhs.astype(lhs_dtype),
        rhs.astype(rhs_dtype),
        dimension_numbers=(
            ((lhs.ndim - 1,), (rhs.ndim - 2,)),
            (tuple(range(lhs.ndim - 2)), tuple(range(rhs.ndim - 2))),
        ),
        precision=precision,
        preferred_element_type=jnp.float32,
    )
  return jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GDNBackwardConfig:
  """Configuration dataclass for GDN backward kernel dimensions and tiling."""

  chunk_size: int = 64
  dim_size: int
  num_kq_heads: int
  num_v_heads: int
  kq_head_dim: int
  v_head_dim: int
  num_chunks: int = 1
  vmem_limit_mb: Optional[int] = None
  use_qk_norm_in_gdn: bool = False
  has_seg_ids: bool = False

  @property
  def repeats(self) -> int:
    return self.num_v_heads // self.num_kq_heads

  @property
  def padded_num_v_heads(self) -> int:
    extra = 1 if self.has_seg_ids else 0
    return ((self.num_v_heads + extra + 127) // 128) * 128


def _bwd_gdn_pipeline_body(
    *refs: Any,
    cfg: GDNBackwardConfig,
    has_dht: bool = False,
    has_dh0: bool = False,
) -> None:
  """Inner kernel executed per (batch, group, chunk) by emit_pipeline with manual GDN backward."""
  if len(refs) == 17:
    has_dht = True
    has_dh0 = True
  idx = 0
  # pylint: disable=unbalanced-tuple-unpacking
  (
      qkv_conv_ref,
      b_ref,
      a_ref,
      do_ref,
      chunk_states_ref,
      t_inv_ref,
      a_log_ref,
      dt_bias_ref,
      reset_ref,
  ) = refs[idx : idx + 9]
  idx += 9
  if has_dht:
    dht_ref = refs[idx]
    idx += 1
  else:
    dht_ref = None
  (
      dy_conv_ref,
      d_b_ref,
      d_a_ref,
      d_a_log_ref,
      d_dt_bias_ref,
  ) = refs[idx : idx + 5]
  # pylint: enable=unbalanced-tuple-unpacking
  idx += 5
  if has_dh0:
    dh0_ref = refs[idx]
    idx += 1
  else:
    dh0_ref = None
  d_state_scr = refs[idx]

  c = pl.program_id(2)
  chunk_size = cfg.chunk_size
  num_kq_heads = cfg.num_kq_heads
  num_v_heads = cfg.num_v_heads
  padded_num_v_heads = cfg.padded_num_v_heads
  kq_head_dim = cfg.kq_head_dim
  v_head_dim = cfg.v_head_dim
  repeats = cfg.repeats
  q_size = num_kq_heads * kq_head_dim
  k_size = num_kq_heads * kq_head_dim
  v_size = num_v_heads * v_head_dim

  @pl.when(c == 0)
  def _init():
    if has_dht and dht_ref is not None:
      d_state_scr[...] = dht_ref[0, ...].astype(jnp.float32)
    else:
      d_state_scr[...] = jnp.zeros_like(d_state_scr)

  is_reset = reset_ref[...][0, 0] > 0.5
  d_state = jnp.where(is_reset, 0.0, d_state_scr[...])
  compute_dtype = qkv_conv_ref.dtype
  y_c = qkv_conv_ref[...]

  # Slice chunk inputs for this head group
  q_orig = y_c[:, :q_size].reshape((chunk_size, num_kq_heads, kq_head_dim)).astype(jnp.float32)
  k_orig = y_c[:, q_size : q_size + k_size].reshape((chunk_size, num_kq_heads, kq_head_dim)).astype(jnp.float32)
  v = (
      y_c[:, q_size + k_size : q_size + k_size + v_size]
      .reshape((chunk_size, num_v_heads, v_head_dim))
      .astype(jnp.float32)
  )

  b_val = b_ref[...][:, :num_v_heads].astype(jnp.float32)
  a_val = a_ref[...][:, :num_v_heads].astype(jnp.float32)
  do_val = do_ref[...].astype(jnp.float32)
  state_prev = chunk_states_ref[...].astype(jnp.float32)
  t_inv_val = t_inv_ref[...].astype(jnp.float32)
  a_log_val = a_log_ref[...][0, :num_v_heads].astype(jnp.float32)
  dt_bias_val = dt_bias_ref[...][0, :num_v_heads].astype(jnp.float32)

  scale = 1.0 / jnp.sqrt(kq_head_dim)
  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  mask_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=0)

  valid_c = None
  m_in = None
  m_out = None
  m_keep = None
  if cfg.has_seg_ids:
    seg_c = b_ref[...][:, num_v_heads : num_v_heads + 1].astype(jnp.float32)
    seg_prev = reset_ref[...][0, 1].astype(jnp.float32)
    valid_c = seg_c > 0.5
    active_c = jnp.abs(seg_c)
    q_orig = jnp.where(valid_c[:, :, None], q_orig, 0.0)
    k_orig = jnp.where(valid_c[:, :, None], k_orig, 0.0)
    v = jnp.where(valid_c[:, :, None], v, 0.0)
    do_val = jnp.where(valid_c[:, :, None], do_val, 0.0)

    same_active = (jnp.abs(active_c - active_c.T) < 0.5) & (active_c > 0.5)
    same_valid = (jnp.abs(seg_c - seg_c.T) < 0.5) & valid_c
    mask_cumsum = mask_cumsum * same_active.astype(jnp.float32)
    mask_strict = mask_strict * same_valid.astype(jnp.float32)
    mask_causal = mask_causal * same_valid.astype(jnp.float32)
    m_in = ((jnp.abs(seg_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
    m_out = ((jnp.abs(seg_c - active_c[-1:]) < 0.5) & (active_c[-1:] > 0.5)).astype(jnp.float32)
    m_keep = ((jnp.abs(active_c[-1:] - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)

  if cfg.use_qk_norm_in_gdn:
    q_norm_sq = jnp.sum(q_orig**2, axis=-1, keepdims=True)
    q_is_zero = q_norm_sq == 0.0
    inv_r_q = jnp.where(q_is_zero, 0.0, jax.lax.rsqrt(q_norm_sq + 1e-6))
    q_unit = q_orig * inv_r_q
    q_scaled = q_unit * scale

    k_norm_sq = jnp.sum(k_orig**2, axis=-1, keepdims=True)
    k_is_zero = k_norm_sq == 0.0
    inv_r_k = jnp.where(k_is_zero, 0.0, jax.lax.rsqrt(k_norm_sq + 1e-6))
    k_unit = k_orig * inv_r_k
    k_scaled_val = k_unit
  else:
    q_scaled = q_orig * scale
    k_scaled_val = k_orig

  if valid_c is not None:
    q_scaled = jnp.where(valid_c[:, :, None], q_scaled, 0.0)
    k_scaled_val = jnp.where(valid_c[:, :, None], k_scaled_val, 0.0)

  q_rep = jnp.repeat(q_scaled, repeats, axis=1)
  k_rep = jnp.repeat(k_scaled_val, repeats, axis=1)
  beta = jax.nn.sigmoid(b_val)

  sp_input = a_val + dt_bias_val
  sp_val = jax.nn.softplus(sp_input)
  exp_a_log = jnp.exp(a_log_val)
  log_g = -exp_a_log * sp_val

  if valid_c is not None:
    beta = jnp.where(valid_c, beta, 0.0)
    log_g = jnp.where(valid_c, log_g, 0.0)

  cumsum_log_g = jnp.dot(mask_cumsum, log_g, precision=jax.lax.Precision.HIGHEST)

  q_h = jnp.transpose(q_rep, (1, 0, 2))
  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))
  do_h = jnp.transpose(do_val, (1, 0, 2))

  diff = cumsum_h[:, :, None] - cumsum_h[:, None, :]
  safe_diff_strict = jnp.where(mask_strict[None, :, :] > 0.5, diff, -1e4)
  g_mat_strict = jnp.exp(safe_diff_strict) * mask_strict[None, :, :]

  safe_diff_causal = jnp.where(mask_causal[None, :, :] > 0.5, diff, -1e4)
  g_mat_causal = jnp.exp(safe_diff_causal) * mask_causal[None, :, :]

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  if m_in is not None and m_out is not None and m_keep is not None:
    gating_forward = gating_forward * m_in[None, :, :]
    gating_last = gating_last * m_keep[None, :, :]
    bwd_diff = jnp.where(m_out.T > 0.5, cumsum_h[:, -1:] - cumsum_h, -1e4)
    gating_backward = jnp.exp(bwd_diff)[:, :, None] * m_out[None, :, :]
  else:
    gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  k_beta = k_h * beta_h[:, :, None]
  k_h_T = jnp.swapaxes(k_h, -1, -2)

  A = t_inv_val

  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward

  # Precision split for bf16 activations (see `_gdn_matmul`): the intra-chunk
  # matmuls (attention, Gram / dS, and the t_inv products) feed the closed-form
  # gating adjoint below and run with f32 operands (`precise=True`). The
  # state-path matmuls (state_prev / d_state operands and the state recurrence)
  # stay single-pass bf16. On the Qwen3.5-397B layer (S=8192) this matches the
  # all-precise variant's A_log / dt_bias accuracy at about a third of its cost.

  # Forward state contribution & v_new in 2 GEMMs instead of 3 (eliminates u, w, ws)
  # Safeguard 1: keep_lhs_fp32=True on A preserves triangular inverse precision
  v_new = _gdn_matmul(
      A,
      v_beta - _gdn_matmul(k_beta_g, state_prev, compute_dtype),
      compute_dtype,
      keep_lhs_fp32=True,
      precise=True,
  )

  q_g = q_h * gating_forward
  attn_unmasked = _gdn_matmul(q_h, k_h_T, compute_dtype, precise=True)
  attn = attn_unmasked * g_mat_causal

  k_scaled_bwd = k_h * gating_backward

  dv_attn = _gdn_matmul(jnp.swapaxes(attn, -1, -2), do_h, compute_dtype, precise=True)
  dv_new = dv_attn + _gdn_matmul(k_scaled_bwd, d_state, compute_dtype)
  d_attn = _gdn_matmul(do_h, jnp.swapaxes(v_new, -1, -2), compute_dtype, precise=True)

  A_T = jnp.swapaxes(A, -1, -2)
  d_v_beta = _gdn_matmul(A_T, dv_new, compute_dtype, keep_lhs_fp32=True, precise=True)

  # Packed weight-stationary GEMM: computes [-d_k_beta_g, d_q_g] in one matmul
  neg_d_k_beta_g, d_q_g = jnp.split(
      _gdn_matmul(
          jnp.concatenate([d_v_beta, do_h], axis=1),
          jnp.swapaxes(state_prev, -1, -2),
          compute_dtype,
      ),
      2,
      axis=1,
  )
  d_k_beta_g = -neg_d_k_beta_g

  # State recurrence using k_beta_g^T @ d_v_beta (eliminates w^T @ dv_new)
  d_state_prev = (
      d_state * gating_last
      + _gdn_matmul(jnp.swapaxes(q_g, -1, -2), do_h, compute_dtype)
      - _gdn_matmul(jnp.swapaxes(k_beta_g, -1, -2), d_v_beta, compute_dtype)
  )

  # Single-GEMM Gram adjoint dS (replaces 4 GEMMs and eliminates dw and dA)
  dS = jnp.tril(
      -_gdn_matmul(d_v_beta, jnp.swapaxes(v_new, -1, -2), compute_dtype, precise=True),
      k=-1,
  )

  d_S_unmasked = dS * g_mat_strict
  d_attn_unmasked = d_attn * g_mat_causal

  # Packed GEMM: d_k_beta_from_S and d_q_h_from_attn sharing RHS k_h
  packed_from_k_h = _gdn_matmul(
      jnp.concatenate([d_S_unmasked, d_attn_unmasked], axis=1),
      k_h,
      compute_dtype,
      precise=True,
  )
  d_k_beta_from_S, d_q_h_from_attn = jnp.split(packed_from_k_h, 2, axis=1)

  d_k_h_from_S = _gdn_matmul(jnp.swapaxes(d_S_unmasked, -1, -2), k_beta, compute_dtype, precise=True)
  d_k_h_from_attn = _gdn_matmul(jnp.swapaxes(d_attn_unmasked, -1, -2), q_h, compute_dtype, precise=True)

  d_k_beta = d_k_beta_g * gating_forward + d_k_beta_from_S

  # Shared inner product <d_k_beta, k_h> used by both d_beta_h and d_cumsum_h
  dk_beta_dot_kh = jnp.sum(d_k_beta * k_h, axis=-1)
  d_beta_h = jnp.sum(d_v_beta * v_h, axis=-1) + dk_beta_dot_kh
  d_v_h = d_v_beta * beta_h[:, :, None]

  d_q_h_from_q_g = d_q_g * gating_forward
  d_k_scaled = _gdn_matmul(v_new, jnp.swapaxes(d_state, -1, -2), compute_dtype)
  d_k_h_from_k_scaled = d_k_scaled * gating_backward

  d_q_h = d_q_h_from_q_g + d_q_h_from_attn
  d_k_h = d_k_h_from_attn + d_k_h_from_S + d_k_beta * beta_h[:, :, None] + d_k_h_from_k_scaled

  # Closed-form vector identity for gating adjoints (eliminates S_unmasked GEMM and d_diff matrices)
  d_cumsum_h = jnp.sum(q_h * d_q_h - k_h * d_k_h, axis=-1) + 2.0 * beta_h * dk_beta_dot_kh
  last_col_addition = (
      jnp.sum(d_state * state_prev, axis=(-1, -2)) * gating_last[:, 0, 0]
      + jnp.sum(d_k_h_from_k_scaled * k_h, axis=(-1, -2))
  )[:, None]
  d_cumsum_h = d_cumsum_h + jnp.pad(last_col_addition, ((0, 0), (chunk_size - 1, 0)))

  d_cumsum_log_g = jnp.transpose(d_cumsum_h, (1, 0))
  d_log_g = jnp.dot(mask_cumsum.T, d_cumsum_log_g, precision=jax.lax.Precision.HIGHEST)
  if valid_c is not None:
    d_log_g = jnp.where(valid_c, d_log_g, 0.0)

  sig_sp = jax.nn.sigmoid(sp_input)
  d_a_val = d_log_g * (-exp_a_log * sig_sp)
  d_a_log_val = jnp.sum(d_log_g * (-exp_a_log * sp_val), axis=0)
  d_dt_bias_val = jnp.sum(d_a_val, axis=0)

  d_b_val = (jnp.transpose(d_beta_h, (1, 0))) * beta * (1.0 - beta)
  d_v_val = jnp.transpose(d_v_h, (1, 0, 2))
  if valid_c is not None:
    d_b_val = jnp.where(valid_c, d_b_val, 0.0)
    d_v_val = jnp.where(valid_c[:, :, None], d_v_val, 0.0)

  # Sublane-aligned GVA 4:1 head reduction summing over repeats before transpose
  d_q_proj_h = jnp.sum(
      d_q_h.reshape(num_kq_heads, repeats, chunk_size, kq_head_dim),
      axis=1,
  )
  d_k_proj_h = jnp.sum(
      d_k_h.reshape(num_kq_heads, repeats, chunk_size, kq_head_dim),
      axis=1,
  )
  d_q_proj = jnp.transpose(d_q_proj_h, (1, 0, 2))
  d_k_proj = jnp.transpose(d_k_proj_h, (1, 0, 2))

  # Immediate rsqrt normalization
  if cfg.use_qk_norm_in_gdn:
    d_q_scaled = d_q_proj * scale
    d_q = (d_q_scaled - q_unit * jnp.sum(d_q_scaled * q_unit, axis=-1, keepdims=True)) * inv_r_q
    d_k = (d_k_proj - k_unit * jnp.sum(d_k_proj * k_unit, axis=-1, keepdims=True)) * inv_r_k
  else:
    d_q = d_q_proj * scale
    d_k = d_k_proj

  if valid_c is not None:
    d_q = jnp.where(valid_c[:, :, None], d_q, 0.0)
    d_k = jnp.where(valid_c[:, :, None], d_k, 0.0)

  # Flatten gradients to chunk_size x dim_size and write to refs
  d_q_flat = d_q.reshape(chunk_size, q_size).astype(dy_conv_ref.dtype)
  d_k_flat = d_k.reshape(chunk_size, k_size).astype(dy_conv_ref.dtype)
  d_v_flat = d_v_val.reshape(chunk_size, v_size).astype(dy_conv_ref.dtype)
  dy_conv_ref[...] = jnp.concatenate([d_q_flat, d_k_flat, d_v_flat], axis=-1)

  if padded_num_v_heads > num_v_heads:
    d_b_ref[...] = jnp.pad(d_b_val, ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(d_b_ref.dtype)
    d_a_ref[...] = jnp.pad(d_a_val, ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(d_a_ref.dtype)
    d_a_log_ref[...] = jnp.pad(d_a_log_val[None, :], ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(
        d_a_log_ref.dtype
    )
    d_dt_bias_ref[...] = jnp.pad(d_dt_bias_val[None, :], ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(
        d_dt_bias_ref.dtype
    )
  else:
    d_b_ref[...] = d_b_val.astype(d_b_ref.dtype)
    d_a_ref[...] = d_a_val.astype(d_a_ref.dtype)
    d_a_log_ref[...] = d_a_log_val[None, :].astype(d_a_log_ref.dtype)
    d_dt_bias_ref[...] = d_dt_bias_val[None, :].astype(d_dt_bias_ref.dtype)

  d_state_scr[...] = d_state_prev.astype(d_state_scr.dtype)

  if has_dh0 and dh0_ref is not None:

    @pl.when(c == cfg.num_chunks - 1)
    def _store_dh0():
      dh0_ref[0, ...] = d_state_prev.astype(dh0_ref.dtype)


def _pallas_gdn_bwd_kernel_single_group(
    qkv_conv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    chunk_states: jax.Array,
    t_inv: jax.Array,
    *,
    cfg: GDNBackwardConfig,
    segment_ids: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
    d_recurrent_state: Optional[jax.Array] = None,
    return_dh0: bool = False,
    interpret: bool | pltpu.InterpretParams | None = None,
    name: str = "gdn_bwd_kernel",
):
  """Executes single head-group Pallas emit_pipeline kernel (delegates to pallas_gdn_bwd_kernel)."""
  return pallas_gdn_bwd_kernel(
      qkv_conv=qkv_conv,
      b=b,
      a=a,
      a_log=a_log,
      dt_bias=dt_bias,
      do=do,
      chunk_states=chunk_states,
      t_inv=t_inv,
      num_v_heads=cfg.num_v_heads,
      kq_head_dim=cfg.kq_head_dim,
      v_head_dim=cfg.v_head_dim,
      chunk_size=cfg.chunk_size,
      use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
      vmem_limit_mb=cfg.vmem_limit_mb,
      head_tile=cfg.num_v_heads,
      segment_ids=segment_ids,
      init_seg=init_seg,
      d_recurrent_state=d_recurrent_state,
      return_dh0=return_dh0,
      interpret=interpret,
      name=name,
  )


def pallas_gdn_bwd_kernel(
    qkv_conv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    chunk_states: jax.Array,
    t_inv: jax.Array,
    *,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    vmem_limit_mb: Optional[int] = None,
    head_tile: Optional[int] = None,
    segment_ids: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
    d_recurrent_state: Optional[jax.Array] = None,
    return_dh0: bool = False,
    interpret: bool | pltpu.InterpretParams | None = None,
    name: str = "gdn_bwd_kernel",
):
  """Executes the Pallas reverse-chunk GDNv3 backward kernel using emit_pipeline.

  Dispatches a single kernel call with 3D grid=(batch_size, num_groups,
  num_chunks).
  """
  if interpret is None:
    on_cpu = jax.default_backend() == "cpu"
    fallback_reason = pallas_unsupported_reason(head_k_dim=kq_head_dim, head_v_dim=v_head_dim, chunk_size=chunk_size)
    if on_cpu or fallback_reason is not None:
      if not on_cpu:
        warn_gdn_pallas_fallback_once("backward", f"{fallback_reason}; running Pallas in interpret mode (very slow)")
      interpret = True
  if interpret:
    ensure_cpu_interpret_registered()

  assert a_log.ndim == 1, f"a_log must be 1D with shape (num_v_heads,), got {a_log.shape}"
  assert dt_bias.ndim == 1, f"dt_bias must be 1D with shape (num_v_heads,), got {dt_bias.shape}"

  batch_size, seq_len, dim_size = qkv_conv.shape
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"GDN backward kernel requires the local sequence length ({seq_len}) to be a multiple of"
        f" chunk_size ({chunk_size}); with sequence-sharded context parallelism this is seq_len / cp."
    )
  num_chunks = seq_len // chunk_size
  num_kq_heads = (dim_size - num_v_heads * v_head_dim) // (kq_head_dim * 2)
  repeats = num_v_heads // num_kq_heads

  has_dht = d_recurrent_state is not None
  has_dh0 = bool(return_dh0)
  if head_tile is not None:
    target_tile = head_tile
  elif jnp.dtype(qkv_conv.dtype) == jnp.float32:
    target_tile = _F32_HEAD_TILE
  else:
    target_tile = _BF16_HEAD_TILE

  max_possible = min(num_v_heads, target_tile)
  tile_v_heads = None
  for candidate in range(max_possible, 0, -1):
    if num_v_heads % candidate == 0 and candidate % repeats == 0:
      tile_v_heads = candidate
      break
  if tile_v_heads is None:
    tile_v_heads = repeats if num_v_heads % repeats == 0 else num_v_heads
  num_groups = num_v_heads // tile_v_heads
  tile_kq_heads = tile_v_heads // repeats

  q_size = num_kq_heads * kq_head_dim
  k_size = num_kq_heads * kq_head_dim
  v_size = num_v_heads * v_head_dim
  tile_q_size = tile_kq_heads * kq_head_dim
  tile_k_size = tile_kq_heads * kq_head_dim
  tile_v_size = tile_v_heads * v_head_dim
  group_dim_size = tile_q_size + tile_k_size + tile_v_size

  has_seg_ids = segment_ids is not None
  cfg = GDNBackwardConfig(
      chunk_size=chunk_size,
      dim_size=group_dim_size,
      num_kq_heads=tile_kq_heads,
      num_v_heads=tile_v_heads,
      kq_head_dim=kq_head_dim,
      v_head_dim=v_head_dim,
      num_chunks=num_chunks,
      vmem_limit_mb=vmem_limit_mb,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      has_seg_ids=has_seg_ids,
  )

  padded_tile_v_heads = cfg.padded_num_v_heads

  # 1. Prepare qkv_conv into (B, G, C, chunk_size, group_dim_size)
  q = qkv_conv[:, :, :q_size]
  k = qkv_conv[:, :, q_size : q_size + k_size]
  v = qkv_conv[:, :, q_size + k_size :]
  q_5d = q.reshape(batch_size, num_chunks, chunk_size, num_groups, tile_q_size).transpose((0, 3, 1, 2, 4))
  k_5d = k.reshape(batch_size, num_chunks, chunk_size, num_groups, tile_k_size).transpose((0, 3, 1, 2, 4))
  v_5d = v.reshape(batch_size, num_chunks, chunk_size, num_groups, tile_v_size).transpose((0, 3, 1, 2, 4))
  qkv_conv_5d = jnp.concatenate([q_5d, k_5d, v_5d], axis=-1)

  # 2. Prepare b and a into (B, G, C, chunk_size, padded_tile_v_heads)
  # With sequence packing, the signed segment IDs are packed into the spare lane
  # of b below. Keep b in f32 then: bf16 rounds integers above 256, so distinct
  # documents would share an ID. The kernel upcasts b to f32 anyway.
  b_packed = b.astype(jnp.float32) if segment_ids is not None else b
  b_5d = b_packed.reshape(batch_size, num_chunks, chunk_size, num_groups, tile_v_heads).transpose((0, 3, 1, 2, 4))
  if padded_tile_v_heads > tile_v_heads:
    b_5d = jnp.pad(
        b_5d,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, padded_tile_v_heads - tile_v_heads),
        ),
    )

  a_5d = a.reshape(batch_size, num_chunks, chunk_size, num_groups, tile_v_heads).transpose((0, 3, 1, 2, 4))
  if padded_tile_v_heads > tile_v_heads:
    a_5d = jnp.pad(
        a_5d,
        (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, padded_tile_v_heads - tile_v_heads),
        ),
    )

  # 3. Prepare do into (B, G, C, chunk_size, tile_v_heads, v_head_dim)
  do_6d = do.reshape(
      batch_size,
      num_chunks,
      chunk_size,
      num_groups,
      tile_v_heads,
      cfg.v_head_dim,
  ).transpose((0, 3, 1, 2, 4, 5))

  # 4. Prepare chunk_states into (B, G, C, tile_v_heads, kq_head_dim, v_head_dim)
  chunk_states_6d = chunk_states.reshape(
      batch_size,
      num_chunks,
      num_groups,
      tile_v_heads,
      cfg.kq_head_dim,
      cfg.v_head_dim,
  ).swapaxes(1, 2)

  # 5. Prepare t_inv into (B, G, C, tile_v_heads, chunk_size, chunk_size)
  t_inv_6d = (
      t_inv.reshape(
          batch_size,
          num_chunks,
          num_groups,
          tile_v_heads,
          chunk_size,
          chunk_size,
      )
      .swapaxes(1, 2)
      .astype(jnp.float32)
  )

  # 6. Prepare a_log and dt_bias into (B, G, 1, padded_tile_v_heads)
  a_log_2d = a_log.reshape(num_groups, tile_v_heads)
  if padded_tile_v_heads > tile_v_heads:
    a_log_2d = jnp.pad(a_log_2d, ((0, 0), (0, padded_tile_v_heads - tile_v_heads)))
  a_log_4d = jnp.broadcast_to(
      a_log_2d[None, :, None, :],
      (batch_size, num_groups, 1, padded_tile_v_heads),
  )

  dt_bias_2d = dt_bias.reshape(num_groups, tile_v_heads)
  if padded_tile_v_heads > tile_v_heads:
    dt_bias_2d = jnp.pad(dt_bias_2d, ((0, 0), (0, padded_tile_v_heads - tile_v_heads)))
  dt_bias_4d = jnp.broadcast_to(
      dt_bias_2d[None, :, None, :],
      (batch_size, num_groups, 1, padded_tile_v_heads),
  )

  # 7. Prepare reset_hbm into (B, G, C, 1, 128) and pack s_enc into b_5d
  if segment_ids is not None:
    s_enc = local_compute_conv1d.encode_segment_ids(segment_ids.reshape(batch_size, seq_len), init_seg=init_seg)
    s_enc_3d = s_enc.reshape(batch_size, num_chunks, chunk_size)
    b_5d = b_5d.at[:, :, :, :, tile_v_heads].set(s_enc_3d[:, None, :, :].astype(b_5d.dtype))
    active_2d = jnp.abs(s_enc)
    active_3d = active_2d.reshape(batch_size, num_chunks, chunk_size)
    active_end = active_3d[:, :, -1]
    if init_seg is not None:
      init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(batch_size, 1))
    else:
      init_active = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    seg_prev_chunks = jnp.concatenate([init_active, active_end[:, :-1]], axis=1)

    if num_chunks > 1:
      end_idx = jnp.arange(1, num_chunks) * chunk_size - 1
      start_next_idx = jnp.arange(1, num_chunks) * chunk_size
      boundaries = active_2d[:, end_idx] != active_2d[:, start_next_idx]
      reset_mask = jnp.pad(boundaries, ((0, 0), (0, 1)), constant_values=False).astype(jnp.float32)
    else:
      reset_mask = jnp.zeros((batch_size, num_chunks), dtype=jnp.float32)

    reset_header = jnp.stack([reset_mask, seg_prev_chunks], axis=-1)[:, :, None, :]
    reset_hbm = jnp.pad(reset_header, ((0, 0), (0, 0), (0, 0), (0, 126)))
  else:
    reset_hbm = jnp.zeros((batch_size, num_chunks, 1, 128), dtype=jnp.float32)
  reset_hbm_5d = jnp.broadcast_to(reset_hbm[:, None, :, :, :], (batch_size, num_groups, num_chunks, 1, 128))

  in_args = [
      qkv_conv_5d,
      b_5d,
      a_5d,
      do_6d,
      chunk_states_6d,
      t_inv_6d,
      a_log_4d,
      dt_bias_4d,
      reset_hbm_5d,
  ]
  if has_dht:
    dht_6d = (
        d_recurrent_state.astype(jnp.float32)
        .reshape(
            batch_size,
            1,
            num_groups,
            tile_v_heads,
            cfg.kq_head_dim,
            cfg.v_head_dim,
        )
        .swapaxes(1, 2)
    )
    in_args.append(dht_6d)

  in_specs, out_specs, num_in, num_out = make_bwd_block_specs(
      num_chunks=num_chunks,
      chunk_size=chunk_size,
      dim_size=group_dim_size,
      num_v_heads=tile_v_heads,
      kq_head_dim=cfg.kq_head_dim,
      v_head_dim=cfg.v_head_dim,
      padded_num_v_heads=padded_tile_v_heads,
      has_dht=has_dht,
      has_dh0=has_dh0,
  )

  out_shapes_list = [
      jax.ShapeDtypeStruct(
          (batch_size, num_groups, num_chunks, chunk_size, group_dim_size),
          qkv_conv.dtype,
      ),
      jax.ShapeDtypeStruct(b_5d.shape, b_5d.dtype),
      jax.ShapeDtypeStruct(a_5d.shape, a_5d.dtype),
      jax.ShapeDtypeStruct(
          (batch_size, num_groups, num_chunks, 1, padded_tile_v_heads),
          a_log_4d.dtype,
      ),
      jax.ShapeDtypeStruct(
          (batch_size, num_groups, num_chunks, 1, padded_tile_v_heads),
          dt_bias_4d.dtype,
      ),
  ]
  if has_dh0:
    out_shapes_list.append(
        jax.ShapeDtypeStruct(
            (
                batch_size,
                num_groups,
                1,
                tile_v_heads,
                cfg.kq_head_dim,
                cfg.v_head_dim,
            ),
            jnp.float32,
        )
    )
  out_shapes = tuple(out_shapes_list)

  body = functools.partial(
      _bwd_gdn_pipeline_body,
      cfg=cfg,
      has_dht=has_dht,
      has_dh0=has_dh0,
  )

  def outer(*refs):
    pltpu.emit_pipeline(
        body,
        grid=(batch_size, num_groups, num_chunks),
        in_specs=in_specs,
        out_specs=out_specs,
    )(*refs[: num_in + num_out], scratches=tuple(refs[num_in + num_out :]))

  if cfg.vmem_limit_mb is not None and cfg.vmem_limit_mb <= 64:
    vmem_limit_bytes = int(cfg.vmem_limit_mb) * 1024 * 1024
  else:
    tpu_info = pltpu.get_tpu_info()
    vmem_limit_bytes = int(0.85 * tpu_info.vmem_capacity_bytes)

  hbm = pltpu.MemorySpace.HBM
  pallas_out = pl.pallas_call(
      outer,
      grid=(),
      out_shape=out_shapes,
      in_specs=[pl.BlockSpec(memory_space=hbm)] * num_in,
      out_specs=[pl.BlockSpec(memory_space=hbm)] * num_out,
      scratch_shapes=[
          pltpu.VMEM((tile_v_heads, cfg.kq_head_dim, cfg.v_head_dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      interpret=interpret,
      name=name,
  )(*in_args)

  if has_dh0:
    (
        dy_conv_chunks,
        d_b_chunks,
        d_a_chunks,
        d_a_log_chunks,
        d_dt_bias_chunks,
        dh0_6d,
    ) = pallas_out
  else:
    (
        dy_conv_chunks,
        d_b_chunks,
        d_a_chunks,
        d_a_log_chunks,
        d_dt_bias_chunks,
    ) = pallas_out
    dh0_6d = None

  # Reconstruct dy_conv: (B, G, C, S, group_dim_size) -> (B, seq_len, dim_size)
  dq_5d = dy_conv_chunks[..., :tile_q_size]
  dk_5d = dy_conv_chunks[..., tile_q_size : tile_q_size + tile_k_size]
  dv_5d = dy_conv_chunks[..., tile_q_size + tile_k_size :]
  dq = dq_5d.transpose((0, 2, 3, 1, 4)).reshape(batch_size, seq_len, q_size)
  dk = dk_5d.transpose((0, 2, 3, 1, 4)).reshape(batch_size, seq_len, k_size)
  dv = dv_5d.transpose((0, 2, 3, 1, 4)).reshape(batch_size, seq_len, v_size)
  dy_conv_flat = jnp.concatenate([dq, dk, dv], axis=-1).astype(qkv_conv.dtype)

  # Reconstruct d_b and d_a: (B, G, C, S, padded_tile_v_heads) -> (B, seq_len, num_v_heads)
  d_b_flat = (
      d_b_chunks[..., :tile_v_heads].transpose((0, 2, 3, 1, 4)).reshape(batch_size, seq_len, num_v_heads).astype(b.dtype)
  )
  d_a_flat = (
      d_a_chunks[..., :tile_v_heads].transpose((0, 2, 3, 1, 4)).reshape(batch_size, seq_len, num_v_heads).astype(a.dtype)
  )

  # Reduce d_a_log and d_dt_bias: (B, G, C, 1, padded_tile_v_heads) -> (num_v_heads,)
  d_a_log_reduced = jnp.sum(d_a_log_chunks[..., 0, :tile_v_heads], axis=(0, 2)).reshape(num_v_heads).astype(a_log.dtype)
  d_dt_bias_reduced = (
      jnp.sum(d_dt_bias_chunks[..., 0, :tile_v_heads], axis=(0, 2)).reshape(num_v_heads).astype(dt_bias.dtype)
  )

  if return_dh0:
    dh0_flat = dh0_6d[:, :, 0, ...].reshape(batch_size, num_v_heads, cfg.kq_head_dim, cfg.v_head_dim)
    return (
        dy_conv_flat,
        d_b_flat,
        d_a_flat,
        d_a_log_reduced,
        d_dt_bias_reduced,
        dh0_flat,
    )

  return (
      dy_conv_flat,
      d_b_flat,
      d_a_flat,
      d_a_log_reduced,
      d_dt_bias_reduced,
  )
