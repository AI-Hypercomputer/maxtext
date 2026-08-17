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
# ==============================================================================
"""Reference and helper implementations for GDN attention testing."""

import dataclasses
import enum
import functools
from typing import Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp


def l2norm_chunked(
    x: jnp.ndarray, dim: int = -1, eps: float = 1e-6
) -> jnp.ndarray:
  """Normalizes x along the specified dimension using L2 norm."""
  x_f32 = x.astype(jnp.float32)
  inv_norm = lax.rsqrt((x_f32 * x_f32).sum(axis=dim, keepdims=True) + eps)
  return (x_f32 * inv_norm).astype(x.dtype)


def l2_normalize_ref(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  """L2 normalize along last dimension."""
  x_f32 = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
  return (x_f32 / norm).astype(x.dtype)


def _recurrent_gated_delta_rule_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Single-step recurrent update for decode."""
  batch_size, num_heads, _, d_k = query.shape
  d_v = value.shape[-1]

  if state is None:
    state = jnp.zeros((batch_size, num_heads, d_k, d_v), dtype=query.dtype)

  q = query[:, :, 0]
  k = key[:, :, 0]
  v = value[:, :, 0]
  beta_val = beta[:, :, 0]
  g_val = g[:, :, 0]

  scale = d_k**-0.5
  q = q * scale

  # v_diff = v - e^g * (k @ state)
  k_state = jnp.einsum("bhd, bhdm -> bhm", k, state)
  v_diff = v - jnp.exp(g_val)[..., None] * k_state

  # v_new = beta * v_diff
  v_new = beta_val[..., None] * v_diff

  # out = e^g * (q @ state) + (q . k) * v_new
  q_state = jnp.einsum("bhd, bhdm -> bhm", q, state)
  q_k = jnp.sum(q * k, axis=-1, keepdims=True)

  out = jnp.exp(g_val)[..., None] * q_state + q_k * v_new

  # s_new = state * exp(g) + k outer v_new
  k_v_new = jnp.einsum("bhd, bhm -> bhdm", k, v_new)
  new_state = state * jnp.exp(g_val)[..., None, None] + k_v_new

  return out[:, :, None, :], new_state


def ragged_gated_delta_rule_ref(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
):
  """Applies the gated delta rule over ragged sequences and updates recurrent state."""
  mixed_qkv = jax.nn.silu(mixed_qkv)
  num_tokens = mixed_qkv.shape[0]
  key_dim = n_kq * d_k
  query = mixed_qkv[..., :key_dim]
  key = mixed_qkv[..., key_dim : key_dim * 2]
  value = mixed_qkv[..., key_dim * 2 :]
  max_reqs = state_indices.shape[0]
  token_idx = jnp.arange(num_tokens)

  num_valid_seqs = distribution[2]
  valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
  last_valid_loc = query_start_loc[num_valid_seqs]
  effective_query_start_loc = jnp.where(
      valid_loc_mask, query_start_loc, last_valid_loc
  )

  req_indices = (
      jnp.sum(token_idx[:, None] >= effective_query_start_loc[None, :], axis=1)
      - 1
  )
  req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
  valid_mask = token_idx < last_valid_loc

  gathered_states = recurrent_state[state_indices]
  masked_initial_states = jnp.where(
      has_initial_state[:, None, None, None],
      gathered_states,
      jnp.zeros_like(gathered_states),
  )
  recurrent_state = recurrent_state.at[state_indices].set(masked_initial_states)

  def scan_fn(carry, xs):
    recurrent_state_all = carry
    (
        curr_q,
        curr_k,
        curr_v,
        curr_b,
        curr_a,
        request_index,
        is_valid_token,
    ) = xs

    curr_q = curr_q[None, None, :]
    curr_k = curr_k[None, None, :]
    curr_v = curr_v[None, None, :]
    curr_b = curr_b[None, None, :]
    curr_a = curr_a[None, None, :]

    state_index = state_indices[request_index]
    recurrent_state = recurrent_state_all[state_index][None, ...]

    batch_size, num_steps = 1, 1
    query_reshaped = curr_q.reshape(batch_size, num_steps, n_kq, d_k)
    key_reshaped = curr_k.reshape(batch_size, num_steps, n_kq, d_k)
    value_reshaped = curr_v.reshape(batch_size, num_steps, n_v, d_v)

    beta = jax.nn.sigmoid(curr_b.astype(jnp.float32))
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        curr_a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )

    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
      query_reshaped = jnp.repeat(query_reshaped, repeat_factor, axis=2)
      key_reshaped = jnp.repeat(key_reshaped, repeat_factor, axis=2)

    query_reshaped = jnp.transpose(query_reshaped, (0, 2, 1, 3)).astype(
        jnp.float32
    )
    key_reshaped = jnp.transpose(key_reshaped, (0, 2, 1, 3)).astype(jnp.float32)
    value_reshaped = jnp.transpose(value_reshaped, (0, 2, 1, 3)).astype(
        jnp.float32
    )
    beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
    g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

    query_reshaped = l2_normalize_ref(query_reshaped)
    key_reshaped = l2_normalize_ref(key_reshaped)

    output, new_recurrent_state = _recurrent_gated_delta_rule_step(
        query_reshaped,
        key_reshaped,
        value_reshaped,
        g,
        beta,
        state=recurrent_state,
    )

    output = jnp.transpose(output, (0, 2, 1, 3)).astype(query.dtype)
    output = output.reshape(batch_size, num_steps, -1)

    recurrent_state_all = jnp.where(
        is_valid_token,
        recurrent_state_all.at[state_index].set(
            new_recurrent_state[0].astype(recurrent_state_all.dtype)
        ),
        recurrent_state_all,
    )

    return recurrent_state_all, output[0, 0]

  carry_init = recurrent_state
  xs = (query, key, value, b, a, req_indices, valid_mask)

  new_recurrent_state, output = lax.scan(scan_fn, carry_init, xs)
  return new_recurrent_state, output


def _fix_query_start_loc(query_start_loc, num_valid_seqs):
  """Fixes query_start_loc to be non-decreasing for invalid sequences."""
  last_valid_loc = query_start_loc[num_valid_seqs]
  valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
  return jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)


def _get_boundary_indices(starts, lengths, kernel_size, num_valid_seqs):
  """Computes indices for boundary fixup."""
  valid_mask = jnp.arange(starts.shape[0]) < num_valid_seqs
  starts = jnp.where(valid_mask, starts, 1)[:, None]
  lengths = lengths[:, None]
  k_range = jnp.arange(kernel_size - 1)[None, :]
  gather_indices = starts + jnp.minimum(k_range, lengths - 1)
  scatter_indices = jnp.where(
      (k_range < lengths) & valid_mask[:, None],
      starts + k_range,
      -1,
  )
  return gather_indices, scatter_indices


def _get_state_update_indices(query_start_loc, kernel_size, num_tokens):
  """Computes indices for updating the convolutional state."""
  lengths = query_start_loc[1:] - query_start_loc[:-1]

  k_range = jnp.arange(kernel_size - 1)

  safe_idx_x = (
      query_start_loc[1:, None] - jnp.arange(kernel_size - 1, 0, -1)[None, :]
  )
  safe_idx_x = jnp.clip(safe_idx_x, 0, num_tokens - 1)

  is_from_old_state = k_range[None, :] < (kernel_size - 1 - lengths)[:, None]

  idx_g = k_range[None, :] + lengths[:, None]
  idx_g = jnp.clip(idx_g, 0, kernel_size - 2)

  return safe_idx_x, is_from_old_state, idx_g


def _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias):
  """Depthwise 1D convolution using loops over kernel size."""
  num_tokens = x.shape[0]
  kernel_size = conv_weight.shape[-1]
  out = None

  padded_x = jnp.pad(x, ((kernel_size - 1, 0), (0, 0)))

  for k in range(kernel_size):
    x_slice = padded_x[k : k + num_tokens, :].astype(jnp.float32)
    weight_slice = conv_weight[:, 0, k].astype(jnp.float32)
    if out is None:
      if conv_bias is None:
        out = x_slice * weight_slice
      else:
        out = x_slice * weight_slice + conv_bias[jnp.newaxis, :]
    else:
      out += x_slice * weight_slice

  assert out is not None
  return out.astype(x.dtype)


def ragged_conv1d_mixed_prefill(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
  """Applies 1D convolution, optimized for prefill."""
  num_tokens = x.shape[0]
  max_blocks = state_indices.shape[0]
  num_valid_seqs = distribution[2]

  out = _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias)

  query_start_loc = _fix_query_start_loc(query_start_loc, num_valid_seqs)
  starts = query_start_loc[:-1]
  lengths = query_start_loc[1:] - query_start_loc[:-1]
  gather_indices, scatter_indices = _get_boundary_indices(
      starts, lengths, kernel_size, num_valid_seqs
  )
  x_first = x[gather_indices]

  gathered_state = conv_state[state_indices]

  gathered_state = jnp.where(
      has_initial_state[:, None, None],
      gathered_state,
      jnp.zeros_like(gathered_state),
  )

  combined_tokens = jnp.concatenate([gathered_state, x_first], axis=1)

  b_out = lax.conv_general_dilated(
      combined_tokens,
      conv_weight,
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NWC", "OIW", "NWC"),
      feature_group_count=x.shape[-1],
      precision=lax.Precision.HIGHEST,
  ).reshape(-1, x.shape[-1])
  if conv_bias is not None:
    b_out += conv_bias[jnp.newaxis, :]

  out = out.at[scatter_indices.flatten()].set(
      b_out.astype(out.dtype), mode="drop", wrap_negative_indices=False
  )
  total_valid_tokens = query_start_loc[num_valid_seqs]
  valid_token_mask = jnp.arange(num_tokens) < total_valid_tokens
  out = jnp.where(valid_token_mask[:, jnp.newaxis], out, 0.0)

  true_valid_seq_mask = jnp.arange(max_blocks) < num_valid_seqs
  safe_idx_x, is_from_old_state, idx_g = _get_state_update_indices(
      query_start_loc, kernel_size, num_tokens
  )

  x_tokens = x[safe_idx_x]
  r_grid = jnp.arange(max_blocks)[:, None]
  state_tokens = gathered_state[r_grid, idx_g]

  new_state_extracted = jnp.where(
      is_from_old_state[..., None], state_tokens, x_tokens
  )

  updated_conv_state = conv_state.at[state_indices].set(
      jnp.where(
          true_valid_seq_mask[:, None, None],
          new_state_extracted,
          conv_state[state_indices],
      )
  )

  return out.astype(x.dtype), updated_conv_state


def ragged_conv1d_decode_only(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
  """Apply conv1d for decode-only case."""
  num_tokens = x.shape[0]

  token_idx = jnp.arange(num_tokens)
  req_state_indices = state_indices[token_idx]
  gathered_state = conv_state[req_state_indices]

  lhs = jnp.concatenate([gathered_state, x[:, jnp.newaxis, :]], axis=1)

  out = jnp.einsum(
      "nkd,dk->nd",
      lhs,
      conv_weight[:, 0, :],
      precision=lax.Precision.HIGHEST,
  )

  if conv_bias is not None:
    out = out + conv_bias

  num_valid_seqs = distribution[2]

  new_state_extracted = jnp.concatenate(
      [gathered_state[:, 1:, :], x[:, jnp.newaxis, :]], axis=1
  )

  token_idx = jnp.arange(num_tokens)
  valid_mask = token_idx < num_valid_seqs
  states_to_set = jnp.where(
      valid_mask[:, jnp.newaxis, jnp.newaxis],
      new_state_extracted,
      gathered_state,
  )

  updated_conv_state = conv_state.at[req_state_indices].set(states_to_set)

  out = jnp.where(valid_mask[:, jnp.newaxis], out, 0.0)

  return out.astype(x.dtype), updated_conv_state


def ragged_conv1d_jax(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Applies 1D convolution over ragged sequences and updates state."""
  is_decode_only = distribution[0] == distribution[2]

  def decode_only_branch(_):
    return ragged_conv1d_decode_only(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

  def mixed_prefill_branch(_):
    return ragged_conv1d_mixed_prefill(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

  return jax.lax.cond(
      is_decode_only, decode_only_branch, mixed_prefill_branch, operand=None
  )


class RaggedConv1dImpl(enum.Enum):
  JAX = "ragged_conv1d_jax"


class RaggedGatedDeltaRuleImpl(enum.Enum):
  REF = "ragged_gated_delta_rule_ref"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GdnAttentionConfig:
  ragged_conv1d_impl: RaggedConv1dImpl = RaggedConv1dImpl.JAX
  ragged_gated_delta_rule_impl: RaggedGatedDeltaRuleImpl = (
      RaggedGatedDeltaRuleImpl.REF
  )


def run_jax_gdn_attention_local_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Runs the local JAX GDN attention mechanism with combined QKV tensors."""
  max_reqs = seq_lens.shape[0]
  query_lens = query_start_loc[1 : max_reqs + 1] - query_start_loc[:max_reqs]
  has_initial_state = (seq_lens - query_lens) > 0

  conv_impl = ragged_conv1d_jax

  out_mixed_qkv, new_conv_state = conv_impl(
      qkv,
      conv_state,
      conv_weight,
      conv_bias,
      query_start_loc,
      state_indices,
      distribution,
      has_initial_state,
      kernel_size=kernel_size,
  )

  ragged_gdn_impl = functools.partial(
      ragged_gated_delta_rule_ref,
      has_initial_state=has_initial_state,
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
  )
  new_recurrent_state, output = ragged_gdn_impl(
      out_mixed_qkv,
      b,
      a,
      recurrent_state,
      a_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
  )

  return (new_conv_state, new_recurrent_state), output
