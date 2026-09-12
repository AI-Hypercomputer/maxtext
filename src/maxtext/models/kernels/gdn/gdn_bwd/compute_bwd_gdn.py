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

try:
  from maxtext.layers import normalizations
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.layers import normalizations
  except (ImportError, ModuleNotFoundError):
    pass

try:
  from maxtext.models.kernels.gdn.gdn_bwd.runtime_utils import invert_triangular_matrix
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd.runtime_utils import invert_triangular_matrix
  except (ImportError, ModuleNotFoundError):
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
  q_rep = jnp.repeat(q, repeats, axis=1)
  k_rep = jnp.repeat(k, repeats, axis=1)

  beta = jax.nn.sigmoid(b_val)

  # EXACT GDN v3 gating formula
  log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)

  # Fast MXU cumsum replacement
  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
  cumsum_log_g = jnp.dot(mask_cumsum, log_g)

  # Transpose to head-first: (H, C, D)
  q_h = jnp.transpose(q_rep, (1, 0, 2))
  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))

  diff = cumsum_h[:, :, None] - cumsum_h[:, None, :]
  mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=diff.dtype), k=-1)
  safe_diff_strict = jnp.where(mask_strict[None, :, :] == 1.0, diff, -1e4)
  g_mat_strict = jnp.exp(safe_diff_strict) * mask_strict[None, :, :]

  mask_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=diff.dtype), k=0)
  safe_diff_causal = jnp.where(mask_causal[None, :, :] == 1.0, diff, -1e4)
  g_mat_causal = jnp.exp(safe_diff_causal) * mask_causal[None, :, :]

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  # WY Representation: T = unit lower-triangular Gram matrix
  k_beta = k_h * beta_h[:, :, None]
  S = jnp.matmul(k_beta, jnp.swapaxes(k_h, -1, -2)) * g_mat_strict
  identity_mask = jnp.eye(chunk_size, dtype=S.dtype)[None, :, :]
  t = jnp.where(identity_mask == 1.0, 1.0, S)
  A = invert_triangular_matrix(t)

  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward
  u = jnp.matmul(A, v_beta)
  w = jnp.matmul(A, k_beta_g)

  # Delta error subtraction against recurrent state
  ws = jnp.matmul(w, state_prev)
  v_new = u - ws

  # Output: cross-chunk state read + intra-chunk attention with v_new
  q_g = q_h * gating_forward
  out_cross = jnp.matmul(q_g, state_prev)

  attn = jnp.matmul(q_h, jnp.swapaxes(k_h, -1, -2)) * g_mat_causal
  out_intra = jnp.matmul(attn, v_new)

  out = out_cross + out_intra
  out = jnp.transpose(out, (1, 0, 2))

  # State update: decayed previous state + rank-1 update from chunk with v_new
  state_prev_decayed = state_prev * gating_last
  k_scaled = k_h * gating_backward
  state_new_intra = jnp.matmul(jnp.swapaxes(k_scaled, -1, -2), v_new)
  state_new = state_prev_decayed + state_new_intra

  return out, state_new, A


def chunk_state_forward_with_cached_tinv(
    k: jax.Array,
    v: jax.Array,
    b_val: jax.Array,
    a_val: jax.Array,
    a_log_val: jax.Array,
    dt_bias_val: jax.Array,
    state_prev: jax.Array,
    t_inv: jax.Array,
    *,
    repeats: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
) -> jax.Array:
  """Computes next recurrent state for one chunk using cached t_inv without triangular inversion."""
  v = v.astype(jnp.float32)
  k = k.astype(jnp.float32)
  if use_qk_norm_in_gdn:
    k = normalizations.l2norm(k, dim=-1, eps=1e-6)
  k_rep = jnp.repeat(k, repeats, axis=1)

  b_val = b_val.astype(jnp.float32)
  a_val = a_val.astype(jnp.float32)
  a_log_val = a_log_val.astype(jnp.float32)
  dt_bias_val = dt_bias_val.astype(jnp.float32)
  state_prev = state_prev.astype(jnp.float32)

  beta = jax.nn.sigmoid(b_val)
  log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)

  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
  cumsum_log_g = jnp.dot(mask_cumsum, log_g)

  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  A = t_inv.astype(jnp.float32)
  k_beta = k_h * beta_h[:, :, None]
  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward

  u = jnp.matmul(A, v_beta)
  w = jnp.matmul(A, k_beta_g)

  ws = jnp.matmul(w, state_prev)
  v_new = u - ws

  state_prev_decayed = state_prev * gating_last
  k_scaled = k_h * gating_backward
  state_new_intra = jnp.matmul(jnp.swapaxes(k_scaled, -1, -2), v_new)
  state_new = state_prev_decayed + state_new_intra

  return state_new


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
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
  """Pure-JAX composite of Conv1D + GDN used during backward pass autodiff."""
  del conv_state
  batch, seq_len, _ = qkv.shape
  key_dim = num_k_heads * head_k_dim

  # Conv1D in FP32
  conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)
  conv_out = sum(conv_input[:, k : k + seq_len, :] * conv_weight_3d[k, 0, :] for k in range(conv_kernel_size))
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(jnp.float32)
  qkv_conv = jax.nn.silu(conv_out).astype(jnp.float32)

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

  try:
    from maxtext.models import qwen3  # pylint: disable=import-outside-toplevel
  except (ImportError, ModuleNotFoundError):
    from maxtext.src.maxtext.models import qwen3  # pylint: disable=import-outside-toplevel

  core_attn_out, next_recurrent_state = qwen3.jax_chunk_gated_delta_rule(
      query=query,
      key=key,
      value=value,
      g=g,
      beta=beta,
      chunk_size=chunk_size,
      initial_state=(recurrent_state.astype(jnp.float32) if recurrent_state is not None else None),
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=jnp.float32,
  )

  next_conv_state = (
      qkv[:, -(conv_kernel_size - 1) :, :]
      if seq_len >= conv_kernel_size - 1
      else jnp.zeros((batch, conv_kernel_size - 1, qkv.shape[-1]), dtype=qkv.dtype)
  )
  if next_recurrent_state is None:
    next_recurrent_state = jnp.zeros((batch, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)

  return core_attn_out.astype(qkv.dtype), (
      next_conv_state.astype(qkv.dtype),
      next_recurrent_state.astype(jnp.float32),
  )


pure_jax_fused_conv1d_gdn = pure_jax_decoupled_conv1d_gdn


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
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.float32,
    cached_t_inv: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Computes convolved QKV, inter-chunk states, and t_inv matrices in FP32."""
  del compute_dtype
  batch_size, seq_len, _ = qkv.shape
  num_chunks = seq_len // chunk_size

  # Conv1D in FP32
  conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)
  conv_out = sum(conv_input[:, k : k + seq_len, :] * conv_weight_3d[k, 0, :] for k in range(conv_kernel_size))
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(jnp.float32)
  qkv_conv_f32 = jax.nn.silu(conv_out).astype(jnp.float32)

  # Chunk states progression in FP32
  num_kq_heads = num_k_heads
  q_size = num_kq_heads * head_k_dim
  k_size = num_kq_heads * head_k_dim
  repeats = num_v_heads // num_kq_heads

  q = qkv_conv_f32[:, :, :q_size].reshape(batch_size, num_chunks, chunk_size, num_kq_heads, head_k_dim)
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

  if cached_t_inv is not None:
    t_inv_chunks = cached_t_inv.astype(jnp.float32).swapaxes(0, 1)

    def chunk_cached_step(k_single, v_single, b_single, a_single, s_prev, t_inv_single):
      return chunk_state_forward_with_cached_tinv(
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
      )

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

    def chunk_step(q_single, k_single, v_single, b_single, a_single, s_prev):
      return chunk_forward_with_tinv(
          q_single,
          k_single,
          v_single,
          b_single,
          a_single,
          a_log_f32,
          dt_bias_f32,
          s_prev,
          kq_head_dim=head_k_dim,
          repeats=repeats,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      )

    def scan_fn(carry_state, chunk_inputs):
      q_i, k_i, v_i, b_i, a_i = chunk_inputs
      _, next_state, t_inv_i = jax.vmap(chunk_step)(q_i, k_i, v_i, b_i, a_i, carry_state)
      return next_state, (carry_state, t_inv_i)

    q_chunks = q.swapaxes(0, 1)
    _, (chunk_states, t_inv_all) = jax.lax.scan(scan_fn, init_state, (q_chunks, k_chunks, v_chunks, b_chunks, a_chunks))
    chunk_states = chunk_states.swapaxes(0, 1)
    t_inv_all = t_inv_all.swapaxes(0, 1)

  return qkv_conv_f32.astype(qkv.dtype), chunk_states, t_inv_all
