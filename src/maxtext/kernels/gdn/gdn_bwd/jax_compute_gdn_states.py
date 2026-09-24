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

"""Pure JAX mathematical equations, chunk state recurrence, and reference GDN."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from maxtext.layers import normalizations

from .. import compute_conv1d as local_compute_conv1d
from .compute_conv1d_bwd import conv1d_silu_fwd
from .runtime_utils import invert_triangular_matrix


def chunk_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    b_val: jax.Array,
    a_val: jax.Array,
    a_log_val: jax.Array,
    dt_bias_val: jax.Array,
    state_prev: jax.Array,
    *,
    kq_head_dim: int,
    repeats: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
    seg_c: Optional[jax.Array] = None,
    seg_prev: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
  """Computes one chunk forward pass for GDN v3 with WY delta rule."""
  out, state_new, _ = chunk_forward_with_tinv(
      q=q,
      k=k,
      v=v,
      b_val=b_val,
      a_val=a_val,
      a_log_val=a_log_val,
      dt_bias_val=dt_bias_val,
      state_prev=state_prev,
      kq_head_dim=kq_head_dim,
      repeats=repeats,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      seg_c=seg_c,
      seg_prev=seg_prev,
  )
  return out, state_new


def chunk_forward_with_tinv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    b_val: jax.Array,
    a_val: jax.Array,
    a_log_val: jax.Array,
    dt_bias_val: jax.Array,
    state_prev: jax.Array,
    *,
    kq_head_dim: int,
    repeats: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
    seg_c: Optional[jax.Array] = None,
    seg_prev: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Computes one chunk forward pass for GDN v3 and returns (out, state_new, t_inv)."""
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  v = v.astype(jnp.float32)
  if use_qk_norm_in_gdn:
    q = normalizations.l2norm(q, dim=-1, eps=1e-6)
    k = normalizations.l2norm(k, dim=-1, eps=1e-6)
  scale = 1.0 / jnp.sqrt(kq_head_dim)
  q = q * scale
  b_val = b_val.astype(jnp.float32)
  a_val = a_val.astype(jnp.float32)
  a_log_val = a_log_val.astype(jnp.float32)
  dt_bias_val = dt_bias_val.astype(jnp.float32)
  state_prev = state_prev.astype(jnp.float32)

  beta = jax.nn.sigmoid(b_val)
  log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)

  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
  mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  mask_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=0)

  valid_c = None
  m_in = None
  m_out = None
  m_keep = None
  if seg_c is not None:
    seg_col = seg_c.astype(jnp.float32).reshape(chunk_size, 1)
    sp = jnp.abs(jnp.asarray(seg_prev if seg_prev is not None else 0.0, dtype=jnp.float32))
    valid_c = seg_col > 0.5
    active_c = jnp.abs(seg_col)
    q = jnp.where(valid_c[:, :, None], q, 0.0)
    k = jnp.where(valid_c[:, :, None], k, 0.0)
    v = jnp.where(valid_c[:, :, None], v, 0.0)
    beta = jnp.where(valid_c, beta, 0.0)
    log_g = jnp.where(valid_c, log_g, 0.0)

    same_active = (jnp.abs(active_c - active_c.T) < 0.5) & (active_c > 0.5)
    same_valid = (jnp.abs(seg_col - seg_col.T) < 0.5) & valid_c
    mask_cumsum = mask_cumsum * same_active.astype(log_g.dtype)
    mask_strict = mask_strict * same_valid.astype(jnp.float32)
    mask_causal = mask_causal * same_valid.astype(jnp.float32)
    m_in = ((jnp.abs(seg_col - sp) < 0.5) & (sp > 0.5)).astype(jnp.float32)
    m_out = ((jnp.abs(seg_col - active_c[-1:]) < 0.5) & (active_c[-1:] > 0.5)).astype(jnp.float32)
    m_keep = ((jnp.abs(active_c[-1:] - sp) < 0.5) & (sp > 0.5)).astype(jnp.float32)

  q_rep = jnp.repeat(q, repeats, axis=1)
  k_rep = jnp.repeat(k, repeats, axis=1)

  # Fast MXU cumsum replacement
  high_prec = jax.lax.Precision.HIGHEST
  cumsum_log_g = jnp.dot(mask_cumsum, log_g, precision=high_prec)

  # Transpose to head-first: (H, C, D)
  q_h = jnp.transpose(q_rep, (1, 0, 2))
  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))

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

  # WY Representation: T = unit lower-triangular Gram matrix
  k_beta = k_h * beta_h[:, :, None]
  S = jnp.matmul(k_beta, jnp.swapaxes(k_h, -1, -2), precision=high_prec) * g_mat_strict
  identity_mask = jnp.eye(chunk_size, dtype=S.dtype)[None, :, :]
  t = jnp.where(identity_mask == 1.0, 1.0, S)
  A = invert_triangular_matrix(t)

  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward
  u = jnp.matmul(A, v_beta, precision=high_prec)
  w = jnp.matmul(A, k_beta_g, precision=high_prec)

  # Delta error subtraction against recurrent state
  ws = jnp.matmul(w, state_prev, precision=high_prec)
  v_new = u - ws

  # Output: cross-chunk state read + intra-chunk attention with v_new
  q_g = q_h * gating_forward
  out_cross = jnp.matmul(q_g, state_prev, precision=high_prec)

  attn = jnp.matmul(q_h, jnp.swapaxes(k_h, -1, -2), precision=high_prec) * g_mat_causal
  out_intra = jnp.matmul(attn, v_new, precision=high_prec)

  out = out_cross + out_intra
  out = jnp.transpose(out, (1, 0, 2))
  if valid_c is not None:
    out = jnp.where(valid_c[:, :, None], out, 0.0)

  # State update: decayed previous state + rank-1 update from chunk with v_new
  state_prev_decayed = state_prev * gating_last
  k_scaled = k_h * gating_backward
  state_new_intra = jnp.matmul(jnp.swapaxes(k_scaled, -1, -2), v_new, precision=high_prec)
  state_new = state_prev_decayed + state_new_intra

  return out, state_new, A


def chunk_state_forward(
    k: jax.Array,
    v: jax.Array,
    b_val: jax.Array,
    a_val: jax.Array,
    a_log_val: jax.Array,
    dt_bias_val: jax.Array,
    state_prev: jax.Array,
    t_inv: Optional[jax.Array] = None,
    *,
    repeats: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
    seg_c: Optional[jax.Array] = None,
    seg_prev: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
  """Computes next recurrent state and t_inv for one chunk, reusing cached t_inv if provided."""
  v = v.astype(jnp.float32)
  k = k.astype(jnp.float32)
  if use_qk_norm_in_gdn:
    k = normalizations.l2norm(k, dim=-1, eps=1e-6)

  b_val = b_val.astype(jnp.float32)
  a_val = a_val.astype(jnp.float32)
  a_log_val = a_log_val.astype(jnp.float32)
  dt_bias_val = dt_bias_val.astype(jnp.float32)
  state_prev = state_prev.astype(jnp.float32)

  beta = jax.nn.sigmoid(b_val)
  log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)

  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
  mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)

  m_in = None
  m_out = None
  m_keep = None
  if seg_c is not None:
    seg_col = seg_c.astype(jnp.float32).reshape(chunk_size, 1)
    sp = jnp.abs(jnp.asarray(seg_prev if seg_prev is not None else 0.0, dtype=jnp.float32))
    valid_c = seg_col > 0.5
    active_c = jnp.abs(seg_col)
    k = jnp.where(valid_c[:, :, None], k, 0.0)
    v = jnp.where(valid_c[:, :, None], v, 0.0)
    beta = jnp.where(valid_c, beta, 0.0)
    log_g = jnp.where(valid_c, log_g, 0.0)

    same_active = (jnp.abs(active_c - active_c.T) < 0.5) & (active_c > 0.5)
    same_valid = (jnp.abs(seg_col - seg_col.T) < 0.5) & valid_c
    mask_cumsum = mask_cumsum * same_active.astype(log_g.dtype)
    mask_strict = mask_strict * same_valid.astype(jnp.float32)
    m_in = ((jnp.abs(seg_col - sp) < 0.5) & (sp > 0.5)).astype(jnp.float32)
    m_out = ((jnp.abs(seg_col - active_c[-1:]) < 0.5) & (active_c[-1:] > 0.5)).astype(jnp.float32)
    m_keep = ((jnp.abs(active_c[-1:] - sp) < 0.5) & (sp > 0.5)).astype(jnp.float32)

  high_prec = jax.lax.Precision.HIGHEST
  k_rep = jnp.repeat(k, repeats, axis=1)
  cumsum_log_g = jnp.dot(mask_cumsum, log_g, precision=high_prec)

  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  if m_in is not None and m_out is not None and m_keep is not None:
    gating_forward = gating_forward * m_in[None, :, :]
    gating_last = gating_last * m_keep[None, :, :]
    bwd_diff = jnp.where(m_out.T > 0.5, cumsum_h[:, -1:] - cumsum_h, -1e4)
    gating_backward = jnp.exp(bwd_diff)[:, :, None] * m_out[None, :, :]
  else:
    gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  if t_inv is not None:
    A = t_inv.astype(jnp.float32)
  else:
    diff = cumsum_h[:, :, None] - cumsum_h[:, None, :]
    safe_diff_strict = jnp.where(mask_strict[None, :, :] > 0.5, diff, -1e4)
    g_mat_strict = jnp.exp(safe_diff_strict) * mask_strict[None, :, :]

    k_beta = k_h * beta_h[:, :, None]
    S = jnp.matmul(k_beta, jnp.swapaxes(k_h, -1, -2), precision=high_prec) * g_mat_strict
    identity_mask = jnp.eye(chunk_size, dtype=S.dtype)[None, :, :]
    t = jnp.where(identity_mask == 1.0, 1.0, S)
    A = invert_triangular_matrix(t)

  k_beta = k_h * beta_h[:, :, None]
  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward

  u = jnp.matmul(A, v_beta, precision=high_prec)
  w = jnp.matmul(A, k_beta_g, precision=high_prec)

  ws = jnp.matmul(w, state_prev, precision=high_prec)
  v_new = u - ws

  state_prev_decayed = state_prev * gating_last
  k_scaled = k_h * gating_backward
  state_new_intra = jnp.matmul(jnp.swapaxes(k_scaled, -1, -2), v_new, precision=high_prec)
  state_new = state_prev_decayed + state_new_intra

  return state_new, A


def pure_jax_decoupled_conv1d_gdn(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype = jnp.float32,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
  """Pure-JAX composite of Conv1D + GDN used during backward pass autodiff."""
  batch, seq_len, _ = qkv.shape
  key_dim = num_k_heads * head_k_dim

  # Conv1D in FP32
  if conv_state is not None:
    conv_input = jnp.concatenate([conv_state.astype(jnp.float32), qkv.astype(jnp.float32)], axis=1)
  else:
    conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))
  _, qkv_conv = conv1d_silu_fwd(
      qkv=qkv.astype(jnp.float32),
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      kernel_size=conv_kernel_size,
      conv_state=conv_state.astype(jnp.float32) if conv_state is not None else None,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
  )
  qkv_conv = qkv_conv.astype(jnp.float32)

  q_conv, k_conv, v_conv = jnp.split(qkv_conv, [key_dim, 2 * key_dim], axis=-1)

  # Reshape for GDN
  query = q_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
  key = k_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
  value = v_conv.reshape(batch, seq_len, num_v_heads, head_v_dim)

  a_log_cast = jnp.asarray(a_log, dtype=jnp.float32)
  dt_bias_cast = jnp.asarray(dt_bias, dtype=jnp.float32)
  beta = jax.nn.sigmoid(b.astype(jnp.float32))
  g = -jnp.exp(a_log_cast) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias_cast)

  if num_v_heads > num_k_heads and num_v_heads % num_k_heads == 0:
    repeats = num_v_heads // num_k_heads
    query = jnp.repeat(query, repeats, axis=2)
    key = jnp.repeat(key, repeats, axis=2)
  from maxtext.models import qwen3  # pylint: disable=import-outside-toplevel,g-import-not-at-top

  init_rs = (
      recurrent_state.astype(jnp.float32)
      if recurrent_state is not None
      else jnp.zeros((batch, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
  )
  core_attn_out, next_recurrent_state = qwen3.jax_chunk_gated_delta_rule(
      query=query,
      key=key,
      value=value,
      g=g,
      beta=beta,
      chunk_size=chunk_size,
      initial_state=init_rs,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
      segment_ids=segment_ids,
      init_seg=init_seg,
  )

  if segment_ids is not None:
    next_conv_state = local_compute_conv1d.extract_segment_conv_state(
        conv_input, segment_ids, conv_kernel_size, conv_halo_seg
    ).astype(qkv.dtype)
  else:
    next_conv_state = conv_input[:, -(conv_kernel_size - 1) :, :].astype(qkv.dtype)
  if next_recurrent_state is None:
    next_recurrent_state = jnp.zeros((batch, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)

  return core_attn_out.astype(qkv.dtype), (
      next_conv_state.astype(qkv.dtype),
      next_recurrent_state.astype(jnp.float32),
  )


def _compute_forward_conv_and_states(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    recurrent_state: Optional[jax.Array],
    *,
    conv_state: Optional[jax.Array] = None,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.float32,
    cached_t_inv: Optional[jax.Array] = None,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Computes convolved QKV, inter-chunk states, and t_inv matrices in FP32."""
  del compute_dtype
  batch_size, seq_len, _ = qkv.shape
  num_chunks = seq_len // chunk_size

  _, qkv_conv_f32 = conv1d_silu_fwd(
      qkv=qkv.astype(jnp.float32),
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      kernel_size=conv_kernel_size,
      conv_state=conv_state.astype(jnp.float32) if conv_state is not None else None,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
  )
  qkv_conv_f32 = qkv_conv_f32.astype(jnp.float32)

  # Chunk states progression in FP32
  num_kq_heads = num_k_heads
  q_size = num_kq_heads * head_k_dim
  k_size = num_kq_heads * head_k_dim
  repeats = num_v_heads // num_kq_heads

  k = qkv_conv_f32[:, :, q_size : q_size + k_size].reshape(batch_size, num_chunks, chunk_size, num_kq_heads, head_k_dim)
  v = qkv_conv_f32[:, :, q_size + k_size :].reshape(batch_size, num_chunks, chunk_size, num_v_heads, head_v_dim)

  b_4d = b.astype(jnp.float32).reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  a_4d = a.astype(jnp.float32).reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  a_log_f32 = a_log.astype(jnp.float32)
  dt_bias_f32 = dt_bias.astype(jnp.float32)

  if recurrent_state is None:
    init_state = jnp.zeros((batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
  else:
    init_state = recurrent_state.astype(jnp.float32)

  k_chunks = k.swapaxes(0, 1)
  v_chunks = v.swapaxes(0, 1)
  b_chunks = b_4d.swapaxes(0, 1)
  a_chunks = a_4d.swapaxes(0, 1)

  if segment_ids is not None:
    s_enc = local_compute_conv1d.encode_segment_ids(segment_ids.reshape(batch_size, seq_len), init_seg=init_seg)
    seg_c_chunks = s_enc.reshape(batch_size, num_chunks, chunk_size).swapaxes(0, 1)
    active_3d = jnp.abs(s_enc.reshape(batch_size, num_chunks, chunk_size))
    active_end = active_3d[:, :, -1]
    if init_seg is not None:
      init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(batch_size, 1))
    else:
      init_active = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    seg_prev_chunks = jnp.concatenate([init_active, active_end[:, :-1]], axis=1).swapaxes(0, 1)
  else:
    seg_c_chunks = None
    seg_prev_chunks = None

  if cached_t_inv is not None:
    t_inv_chunks = cached_t_inv.astype(jnp.float32).swapaxes(0, 1)

    def chunk_cached_step(
        k_single,
        v_single,
        b_single,
        a_single,
        s_prev,
        t_inv_single,
        seg_c_single=None,
        seg_prev_single=None,
    ):
      next_state, _ = chunk_state_forward(
          k=k_single,
          v=v_single,
          b_val=b_single,
          a_val=a_single,
          a_log_val=a_log_f32,
          dt_bias_val=dt_bias_f32,
          state_prev=s_prev,
          t_inv=t_inv_single,
          repeats=repeats,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=use_qk_norm_in_gdn,
          seg_c=seg_c_single,
          seg_prev=seg_prev_single,
      )
      return next_state

    if seg_c_chunks is not None and seg_prev_chunks is not None:

      def scan_fn_cached_seg(carry_state, chunk_inputs):
        k_i, v_i, b_i, a_i, t_inv_i, sc_i, sp_i = chunk_inputs
        next_state = jax.vmap(chunk_cached_step)(k_i, v_i, b_i, a_i, carry_state, t_inv_i, sc_i, sp_i)
        return next_state, carry_state

      _, chunk_states = jax.lax.scan(
          scan_fn_cached_seg,
          init_state,
          (
              k_chunks,
              v_chunks,
              b_chunks,
              a_chunks,
              t_inv_chunks,
              seg_c_chunks,
              seg_prev_chunks,
          ),
      )
    else:

      def scan_fn_cached(carry_state, chunk_inputs):
        k_i, v_i, b_i, a_i, t_inv_i = chunk_inputs
        next_state = jax.vmap(chunk_cached_step)(k_i, v_i, b_i, a_i, carry_state, t_inv_i)
        return next_state, carry_state

      _, chunk_states = jax.lax.scan(
          scan_fn_cached,
          init_state,
          (k_chunks, v_chunks, b_chunks, a_chunks, t_inv_chunks),
      )
    chunk_states = chunk_states.swapaxes(0, 1)
    t_inv_all = cached_t_inv
  else:

    def chunk_step(
        k_single,
        v_single,
        b_single,
        a_single,
        s_prev,
        seg_c_single=None,
        seg_prev_single=None,
    ):
      return chunk_state_forward(
          k=k_single,
          v=v_single,
          b_val=b_single,
          a_val=a_single,
          a_log_val=a_log_f32,
          dt_bias_val=dt_bias_f32,
          state_prev=s_prev,
          t_inv=None,
          repeats=repeats,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=use_qk_norm_in_gdn,
          seg_c=seg_c_single,
          seg_prev=seg_prev_single,
      )

    if seg_c_chunks is not None and seg_prev_chunks is not None:

      def scan_fn_seg(carry_state, chunk_inputs):
        k_i, v_i, b_i, a_i, sc_i, sp_i = chunk_inputs
        next_state, t_inv_i = jax.vmap(chunk_step)(k_i, v_i, b_i, a_i, carry_state, sc_i, sp_i)
        return next_state, (carry_state, t_inv_i)

      _, (chunk_states, t_inv_all) = jax.lax.scan(
          scan_fn_seg,
          init_state,
          (
              k_chunks,
              v_chunks,
              b_chunks,
              a_chunks,
              seg_c_chunks,
              seg_prev_chunks,
          ),
      )
    else:

      def scan_fn(carry_state, chunk_inputs):
        k_i, v_i, b_i, a_i = chunk_inputs
        next_state, t_inv_i = jax.vmap(chunk_step)(k_i, v_i, b_i, a_i, carry_state)
        return next_state, (carry_state, t_inv_i)

      _, (chunk_states, t_inv_all) = jax.lax.scan(
          scan_fn,
          init_state,
          (k_chunks, v_chunks, b_chunks, a_chunks),
      )
    chunk_states = chunk_states.swapaxes(0, 1)
    t_inv_all = t_inv_all.swapaxes(0, 1)

  return qkv_conv_f32.astype(qkv.dtype), chunk_states, t_inv_all


compute_gdn_states_jax = _compute_forward_conv_and_states
