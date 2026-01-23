# Copyright 2023–2025 Google LLC
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

"""Qwen3 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, cast
import math

import jax
import jax.nn
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText import max_utils
from MaxText.common_types import AttentionType, Config, DType, Array, BATCH, LENGTH_NO_EXP, EMBED, MODEL_MODE_TRAIN
from MaxText.layers import attentions
from MaxText.layers import initializers as max_initializers
from MaxText.layers import moe
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.layers.embeddings import Qwen3OmniMoeVisionPosEmbedInterpolate, PositionalEmbedding
from MaxText.layers.normalizations import RMSNorm, l2norm, Qwen3NextRMSNorm, Qwen3NextRMSNormGated
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import DenseGeneral, MlpBlock
from MaxText.layers.moe import RoutedMoE
from MaxText.layers.initializers import nd_dense_init, variable_to_logically_partitioned
# -----------------------------------------
# Qwen3-Next Layer Implementations
# -----------------------------------------


def jax_chunk_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    use_qk_norm_in_gdn: bool = False,
) -> tuple[Array, None | Array]:
  """
  A JAX implementation of the chunked Gated Delta Rule, a parallel scan algorithm.
  This function implements the core recurrent logic of the Gated Delta Network in
  a hardware-efficient way by splitting the sequence into chunks and using
  jax.lax.scan for the recurrent part.

  Tensor Shape Abbreviations:
    B: batch_size, S: sequence_length, H: num_heads,
    D_k: key/query_head_dim, D_v: value_head_dim,
    N: num_chunks, C: chunk_size

  Args:
    query: Query tensor. Shape (B, S, H, D_k)
    key: Key tensor. Shape (B, S, H, D_k)
    value: Value tensor. Shape (B, S, H, D_v)
    g: Log decay tensor. Shape (B, S, H)
    beta: Gate tensor. Shape (B, S, H)
    chunk_size: The size of each chunk for processing.
    initial_state: Optional initial state for the recurrence. Shape (B, H, D_k, D_v)
    use_qk_norm_in_gdn: Whether to apply L2 normalization to query and key.

  Returns:
    Output tensor. Shape (B, S, H, D_v)
    Final recurrent state. Shape (B, H, D_k, D_v) or None
  """

  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype
  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  # Transpose (B, S, H, D) -> (B, H, S, D)
  query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
  key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
  value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
  # Transpose (B, S, H) -> (B, H, S)
  beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
  g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

  batch_size, num_heads, sequence_length, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

  # Padding to make sequence_length divisible by chunk_size
  if pad_size > 0:
    query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))  # (B, H, S_padded, D_k)
    key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))  # (B, H, S_padded, D_k)
    value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))  # (B, H, S_padded, D_v)
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))  # (B, H, S_padded)
    g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))  # (B, H, S_padded)

  total_sequence_length = sequence_length + pad_size
  # query shape: (B, H, S_padded, D_k)
  scale = jax.lax.rsqrt(jnp.array(query.shape[-1]).astype(jnp.float32))
  query = query * scale

  v_beta = value * jnp.expand_dims(beta, -1)  # (B, H, S_padded, D_v)
  k_beta = key * jnp.expand_dims(beta, -1)  # (B, H, S_padded, D_k)

  # Reshape to chunks
  num_chunks = total_sequence_length // chunk_size
  # query_c shape: (B, H, N, C, D_k)
  query_c = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  key_c = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  k_beta_c = k_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  v_beta_c = v_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
  g_c = g.reshape(batch_size, num_heads, num_chunks, chunk_size)  # (B, H, N, C)

  mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)  # (C, C)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK CALCULATION (PARALLEL)
  # =========================================================================
  # g_cumsum shape: (B, H, N, C)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  # g_diff shape: (B, H, N, C, C)
  g_diff = jnp.expand_dims(g_cumsum, -1) - jnp.expand_dims(g_cumsum, -2)

  # Apply tril to zero out the upper triangle of g_diff. This is crucial because
  # the upper triangle contains large positive values that would cause exp() to overflow.
  g_diff_tril = jnp.tril(g_diff)

  # Exponentiate the lower triangular g_diff. Since these values are non-positive,
  # exp() will not overflow and will produce values between 0 and 1.
  g_diff_exp = jnp.exp(g_diff_tril).astype(jnp.float32)

  # The result g_diff_exp is already lower triangular and serves as the decay_mask.
  # decay_mask shape: (B, H, N, C, C)
  decay_mask = g_diff_exp

  # --- Precompute within-chunk attention ---
  # NOTE: Precision set to HIGHEST for numerical accuracy.
  prec = jax.lax.Precision.HIGHEST
  # attn shape: (B, H, N, C, C)
  attn = -jnp.matmul(k_beta_c, jnp.swapaxes(key_c, -1, -2), precision=prec) * decay_mask
  attn = jnp.where(mask, 0.0, attn)

  # Iterative refinement of the intra-chunk attention.
  # This loop is equivalent to inverting (I - A) where A is the lower triangular part of attn.
  def inner_attn_body(i, attn_val):
    # indices: (C,)
    indices = jnp.arange(chunk_size)
    # col_mask: (C,)
    col_mask = indices < i
    # row: (B, H, N, C)
    row = attn_val[..., i, :] * col_mask
    # sub_mask: (C, C)
    sub_mask = jnp.expand_dims(indices < i, -1) & (indices < i)
    # sub: (B, H, N, C, C)
    sub = attn_val * sub_mask
    # row_exp: (B, H, N, C, 1)
    row_exp = jnp.expand_dims(row, -1)
    # term: (B, H, N, C, C)
    term = row_exp * sub
    # summed: (B, H, N, C)
    summed = jnp.sum(term, axis=-2)
    # update_val: (B, H, N, C)
    update_val = row + summed
    # original_row: (B, H, N, C)
    original_row = attn_val[..., i, :]
    # new_row: (B, H, N, C)
    new_row = jnp.where(col_mask, update_val, original_row)
    return attn_val.at[..., i, :].set(new_row)

  attn = jax.lax.fori_loop(1, chunk_size, inner_attn_body, attn)

  attn = attn + jnp.eye(chunk_size, dtype=attn.dtype)  # (B, H, N, C, C)
  # value_intra shape: (B, H, N, C, D_v)
  value_intra = jnp.matmul(attn, v_beta_c, precision=prec)
  # k_cumdecay shape: (B, H, N, C, D_k)
  k_cumdecay = jnp.matmul(attn, (k_beta_c * jnp.expand_dims(jnp.exp(g_cumsum), -1)), precision=prec)
  # --- End Precompute ---

  output_final_state = initial_state is not None
  if initial_state is None:
    # last_recurrent_state shape: (B, H, D_k, D_v)
    last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=value_intra.dtype)
  else:
    last_recurrent_state = initial_state.astype(value_intra.dtype)

  # mask_inter shape: (C, C)
  mask_inter = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

  # Transpose for scan: (B, H, N, C, D) -> (N, B, H, C, D)
  query_scan = jnp.transpose(query_c, (2, 0, 1, 3, 4))
  key_scan = jnp.transpose(key_c, (2, 0, 1, 3, 4))
  value_scan = jnp.transpose(value_intra, (2, 0, 1, 3, 4))
  k_cumdecay_scan = jnp.transpose(k_cumdecay, (2, 0, 1, 3, 4))
  # Transpose for scan: (B, H, N, C) -> (N, B, H, C)
  g_scan = jnp.transpose(g_cumsum, (2, 0, 1, 3))
  decay_mask_scan = jnp.transpose(decay_mask, (2, 0, 1, 3, 4))

  xs = (query_scan, key_scan, value_scan, k_cumdecay_scan, g_scan, decay_mask_scan)

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (SEQUENTIAL VIA SCAN)
  # =========================================================================
  def scan_body(prev_state, x):
    q_i, k_i, v_i, k_cumdecay_i, g_i, decay_mask_i = x
    # prev_state shape: (B, H, D_k, D_v)
    last_recurrent_state = prev_state
    prec = jax.lax.Precision.HIGHEST

    # Intra-chunk attention for the current chunk
    # attn_i shape: (B, H, C, C)
    attn_i = jnp.matmul(q_i, jnp.swapaxes(k_i, -1, -2), precision=prec) * decay_mask_i
    attn_i = jnp.where(mask_inter, 0.0, attn_i)

    # Interaction with the recurrent state
    # v_prime shape: (B, H, C, D_v)
    v_prime = jnp.matmul(k_cumdecay_i, last_recurrent_state, precision=prec)
    # v_new shape: (B, H, C, D_v)
    v_new = v_i - v_prime

    # g_i is cumulative sum, so exp(g_i) is the decay factor
    g_i_exp = jnp.exp(g_i)
    # attn_inter shape: (B, H, C, D_v)
    attn_inter = jnp.matmul(q_i * jnp.expand_dims(g_i_exp, -1), last_recurrent_state, precision=prec)

    # core_attn_out_i shape: (B, H, C, D_v)
    core_attn_out_i = attn_inter + jnp.matmul(attn_i, v_new, precision=prec)

    # Update the recurrent state
    # g_i_last_exp shape: (B, H, 1, 1)
    g_i_last_exp = jnp.exp(g_i[..., -1, None, None])
    # new_last_recurrent_state shape: (B, H, D_k, D_v)
    new_last_recurrent_state = last_recurrent_state * g_i_last_exp

    # g_diff_exp shape: (B, H, C, 1)
    g_diff_exp = jnp.expand_dims(jnp.exp(jnp.expand_dims(g_i[..., -1], -1) - g_i), -1)
    # k_i_g_diff shape: (B, H, C, D_k)
    k_i_g_diff = k_i * g_diff_exp

    # Update term shape: (B, H, D_k, D_v)
    update_term = jnp.matmul(jnp.swapaxes(k_i_g_diff, -1, -2), v_new, precision=prec)
    new_last_recurrent_state = new_last_recurrent_state + update_term

    return new_last_recurrent_state, core_attn_out_i

  # final_state shape: (B, H, D_k, D_v)
  # core_attn_out_stacked shape: (N, B, H, C, D_v)
  final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  # core_attn_out shape: (B, H, N, C, D_v)
  core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))

  # core_attn_out shape: (B, H, S_padded, D_v)
  core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
  # Trim padding: (B, H, S, D_v)
  core_attn_out = core_attn_out[:, :, :sequence_length, :]

  # Transpose back to (B, S, H, D_v)
  core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

  return core_attn_out, final_state if output_final_state else None


class Qwen3NextGatedDeltaNet(nnx.Module):
  """
  This module implements the full end-to-end logic of a Gated Delta Network layer.

  End-to-End Equations Implemented:
  Let `x` be the input `hidden_states`.

  Step A: Input Projections
  1. (q_raw, k_raw, v_raw, z) = Linear_qkvz(x)
  2. (b, a) = Linear_ba(x)

  Step B: 1D Convolution
  1. qkv_conv = silu(Conv1D(concatenate(q_raw, k_raw, v_raw)))
  2. (q, k, v) = split(qkv_conv)

  Step C: Gated Delta Rule (Recurrent Core)
  1. Gates: β=sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
  2. Core Calculation: core_attn_out = jax_chunk_gated_delta_rule(q, k, v, g, β)

  Step D: Final Output Stage
  1. y = RMSNorm(core_attn_out) * silu(z)
  2. output = Linear_out(y)
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    """
    Args:
      config: MaxText configuration object.
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    """
    self.config = config
    cfg = self.config

    in_features = cfg.emb_dim
    self.num_v_heads = cfg.gdn_num_value_heads
    self.num_k_heads = cfg.gdn_num_key_heads
    self.head_k_dim = cfg.gdn_key_head_dim
    self.head_v_dim = cfg.gdn_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    conv_dim = self.key_dim * 2 + self.value_dim
    conv_kernel_size = cfg.gdn_conv_kernel_dim
    self.v_heads_per_k_head = self.num_v_heads // self.num_k_heads

    # Submodule instantiations
    self.in_proj_qkvz = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.key_dim * 2 + self.value_dim * 2),
        dtype=cfg.dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.in_proj_ba = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.num_v_heads * 2),
        dtype=cfg.dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    self.conv1d = nnx.Conv(
        in_features=conv_dim,
        out_features=conv_dim,
        kernel_size=(conv_kernel_size,),
        feature_group_count=conv_dim,  # Depthwise
        padding="CAUSAL",
        use_bias=False,
        dtype=cfg.dtype,
        precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Initialize A_log to match torch.log(torch.uniform(0, 16))
    def a_log_init(key, shape, dtype=jnp.float32):
      # Sample from Uniform(epsilon, 16) to avoid log(0)
      a_vals = jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0)
      return jnp.log(a_vals)

    self.A_log = nnx.Param(a_log_init(rngs.params(), (self.num_v_heads,)))
    self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (self.num_v_heads,)))

    self.norm = Qwen3NextRMSNormGated(
        num_features=self.head_v_dim,  # Normalize over the head dimension (D_v)
        eps=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.out_proj = DenseGeneral(
        in_features_shape=self.value_dim,
        out_features_shape=(in_features,),
        dtype=cfg.dtype,
        kernel_axes=("mlp", "embed"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    # hidden_states: (B, S, E)
    cfg = self.config
    batch, seq_len, _ = hidden_states.shape

    # =========================================================================
    # STEP A: Input Projections
    # =========================================================================
    # qkvz: (B, S, 2 * K_dim + 2 * V_dim)
    qkvz = self.in_proj_qkvz(hidden_states)
    # ba: (B, S, 2 * H_v)
    ba = self.in_proj_ba(hidden_states)

    # QKVZ Reshaping and Splitting
    # Per-K_head group dim: 2 * D_k + 2 * D_v * V_per_K
    new_shape_qkvz = (
        batch,
        seq_len,
        self.num_k_heads,  # H_k
        2 * self.head_k_dim + 2 * self.head_v_dim * self.v_heads_per_k_head,
    )
    # mixed_qkvz: (B, S, H_k, 2*D_k + 2*D_v*V_per_K)
    mixed_qkvz = qkvz.reshape(new_shape_qkvz)

    split_indices_qkvz = [
        self.head_k_dim,  # D_k
        2 * self.head_k_dim,  # 2 * D_k
        2 * self.head_k_dim + (self.v_heads_per_k_head * self.head_v_dim),  # 2 * D_k + V_per_K * D_v
    ]
    # query: (B, S, H_k, D_k)
    # key: (B, S, H_k, D_k)
    # value_raw: (B, S, H_k, V_per_K * D_v)
    # z_raw: (B, S, H_k, V_per_K * D_v)
    query, key, value_raw, z_raw = jnp.split(mixed_qkvz, split_indices_qkvz, axis=3)

    # value: (B, S, H_v, D_v)
    value = value_raw.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
    # z: (B, S, H_v, D_v)
    z = z_raw.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

    # BA Reshaping and Splitting
    new_shape_ba = (
        batch,
        seq_len,
        self.num_k_heads,  # H_k
        2 * self.v_heads_per_k_head,
    )
    # mixed_ba: (B, S, H_k, 2 * V_per_K)
    mixed_ba = ba.reshape(new_shape_ba)

    split_indices_ba = [self.v_heads_per_k_head]
    # b_raw: (B, S, H_k, V_per_K)
    # a_raw: (B, S, H_k, V_per_K)
    b_raw, a_raw = jnp.split(mixed_ba, split_indices_ba, axis=3)

    # b: (B, S, H_v)
    b = b_raw.reshape(batch, seq_len, self.num_v_heads)
    # a: (B, S, H_v)
    a = a_raw.reshape(batch, seq_len, self.num_v_heads)

    # Flatten head dimensions for concatenation before conv
    # q: (B, S, K_dim)
    q = query.reshape(batch, seq_len, -1)
    # k: (B, S, K_dim)
    k = key.reshape(batch, seq_len, -1)
    # v: (B, S, V_dim)
    v = value.reshape(batch, seq_len, -1)

    # =========================================================================
    # STEP B: 1D Convolution
    # =========================================================================
    # conv_dim = 2 * K_dim + V_dim
    # qkv: (B, S, 2 * K_dim + V_dim)
    qkv = jnp.concatenate([q, k, v], axis=-1)

    # TODO(parambole): Implement caching logic for conv_state and recurrent_state

    # Input to conv_layer should be (B, S, C)
    # qkv_conv shape: (B, S, conv_dim)
    conv_out = self.conv1d(qkv)
    qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(cfg.dtype)
    # q_conv shape: (B, S, key_dim), k_conv shape: (B, S, key_dim), v_conv shape: (B, S, value_dim)
    q_conv, k_conv, v_conv = jnp.split(qkv_conv, [self.key_dim, 2 * self.key_dim], axis=-1)

    # Reshape for multi-head processing
    batch, seq_len, _ = hidden_states.shape
    # query shape: (B, S, H_k, D_k)
    query = q_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
    # key shape: (B, S, H_k, D_k)
    key = k_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
    # value shape: (B, S, H_v, D_v)
    value = v_conv.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

    # =========================================================================
    # STEP C: Gated Delta Rule Recurrence
    # =========================================================================
    A_log = self.A_log.value
    dt_bias = self.dt_bias.value
    # beta shape: (B, S, H_v)
    beta = jax.nn.sigmoid(b)
    # g shape: (B, S, H_v)
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias.astype(jnp.float32))
    g = g.astype(cfg.dtype)

    if self.num_v_heads > self.num_k_heads and self.num_v_heads % self.num_k_heads == 0:
      repeats = self.num_v_heads // self.num_k_heads
      # query shape after repeat: (B, S, H_v, D_k)
      query = jnp.repeat(query, repeats, axis=2)
      # key shape after repeat: (B, S, H_v, D_k)
      key = jnp.repeat(key, repeats, axis=2)
    elif self.num_k_heads > self.num_v_heads and self.num_k_heads % self.num_v_heads == 0:
      # This case might occur if key/query heads are more than value heads.
      pass  # No repeating needed for query/key in this case

    # TODO(parambole): Pass and update cache state for jax_chunk_gated_delta_rule
    # core_attn_out shape: (B, S, H_v, D_v)
    core_attn_out, _ = jax_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=cfg.gdn_chunk_size, use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn
    )

    # =========================================================================
    # STEP D: Final Output Stage
    # =========================================================================

    # The normalization and gating is applied per-head on the value dimension.

    # Apply the norm and gate. Output shape: (B, S, H_v, D_v)
    gated_output_reshaped = self.norm(core_attn_out, z)

    # Reshape back to a single feature dimension for the final projection.
    # Shape from (B, S, H_v, D_v) -> (B, S, value_dim)
    gated_output = gated_output_reshaped.reshape(batch, seq_len, -1)

    # Final output shape: (B, S, E)
    output = self.out_proj(gated_output)

    return output


class Qwen3NextFullAttention(nnx.Module):
  """Qwen3-Next Full Attention Layer.

  This module implements the full self-attention mechanism as used in
  Qwen3-Next models for layers that do not use the Gated Delta Network.
  It wraps the main `attentions.Attention` class, which handles the core attention operation,
  including the query, key, value, and output projections.

  Qwen3 Next Attention differs from standard attention by the following features:
    - Query and Gate splitting from a single q projection.
    - Application of a sigmoid gate to the attention output.
    - Usage of `Qwen3NextRMSNorm` for query and key normalization.
    - Usage of `Qwen3NextRotaryEmbedding` for partial rotary position embeddings.
      - Partial ROPE is applied to the first 25% of head dimensions

  Attributes:
    config: MaxText configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer.
    quant: Optional quantization configuration.
    attention: An instance of `attentions.Attention` which contains the
      learnable parameters for query, key, value, and output projections
      (e.g., `attention.query`, `attention.key`, etc.), and performs
      the attention calculation.
  """

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config

    scaling_factor = self.config.head_dim**-0.5
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.attention = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        out_axis_names=(BATCH, LENGTH_NO_EXP, EMBED),
        mesh=self.mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        use_qk_norm=cfg.use_qk_norm,
        query_pre_attn_scalar=scaling_factor,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    attention_output, kv_cache = self.attention(
        inputs_q=inputs,
        inputs_kv=inputs,
        inputs_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    return attention_output, kv_cache


class Qwen3NextSparseMoeBlock(nnx.Module):
  """
  This module encapsulates the unique MoE structure of Qwen3-Next, which includes:
  1. A set of routed experts, where each token is sent to a subset of experts.
  2. A single shared expert, which all tokens pass through.
  3. A learnable gate that determines the contribution of the shared expert.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    quant: Optional quantization configuration.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = self.config

    # 1. Instantiate and apply the routed experts block.
    self.routed_experts = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

    # 2. Instantiate and apply the shared expert.
    self.shared_expert = MlpBlock(
        config=cfg,
        mesh=mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.moe_mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
        model_mode=config.model_call_mode,
        rngs=rngs,
    )

    # 3. Instantiate and apply the gate for the shared expert.
    self.shared_expert_gate = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=1,
        use_bias=False,  # Qwen3-Next shared_expert_gate does not have a bias
        dtype=cfg.dtype,
        kernel_init=max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "vocab"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array, deterministic: bool) -> tuple[Array, Array | None]:
    """
    Applies the sparse MoE block to the input hidden states.

    Args:
      hidden_states: The input array from the previous layer. Shape: (batch, seq, embed_dim)
      deterministic: If True, disables dropout.

    Returns:
      A tuple containing:
        - The output array of the MoE block.
        - The load balancing loss from the routed experts, if applicable during training.
    """
    # 1. Apply the routed experts block.
    routed_output, load_balance_loss, _ = self.routed_experts(hidden_states)

    # 2. Apply the shared expert.
    shared_expert_output = self.shared_expert(hidden_states, deterministic=deterministic)

    # 3. Apply the gate for the shared expert.
    shared_gate_output = self.shared_expert_gate(hidden_states)

    # 4. Combine the outputs.
    final_output = routed_output + jax.nn.sigmoid(shared_gate_output) * shared_expert_output

    return final_output, load_balance_loss


class Qwen3NextScannableBlock(nnx.Module):
  """A scannable block of Qwen3-Next decoder layers.

  This module contains a fixed number of heterogeneous decoder layers that form
  a repeating pattern, as defined by `config.inhomogeneous_layer_cycle_interval`. It is
  intended to be the body of an `nn.scan` transformation to construct the full
  decoder stack efficiently.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    quant: Optional quantization configuration.
  """

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    cfg = self.config

    # Instantiate each layer within the block in __init__
    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer_rngs = self.rngs.fork()  # Fork RNGs for each layer
      layer_name = f"layer_{i}"
      layer = Qwen3NextDecoderLayer(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          layer_idx=i,
          rngs=layer_rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ) -> tuple[Array, None]:
    """Applies the block of decoder layers to the input carry.

    Args:
      carry: The input tensor from the previous scan iteration.
      # ... other arguments are broadcasted to each iteration.

    Returns:
      A tuple containing the output of the block (the new carry) and an empty
      value for the scan's `y` collection.
    """
    cfg = self.config
    x = carry

    # Loop over the number of sub-layers that make up one repeating pattern.
    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      # The second return value is kv_cache, which we ignore here because
      # it is not passed as a carry in scannable layers.
      x, _ = layer(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          page_state,
          slot,
      )

    # The output of the block is the carry for the next scan iteration.
    return x, None


class Qwen3NextDecoderLayer(nnx.Module):
  """
  This layer is a hybrid, capable of functioning as either:
  1. A standard attention + MoE layer.
  2. A linear attention + MoE layer.

  NOTE: This implementation assumes every layer contains a MoE block, which is true for
  models like Qwen3-Next-80B-A3B where `decoder_sparse_step=1`. For models that
  interleave dense and sparse MLP layers, conditional logic would be needed here.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    # First LayerNorm, applied before the attention block.
    self.input_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        eps=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Determine the type of attention mechanism for the current layer.
    is_full_attention_layer = (self.layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0

    # Conditionally instantiate either the Linear Attention or Full Attention block.
    if is_full_attention_layer:
      self.attention = Qwen3NextFullAttention(
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=model_mode,
          layer_idx=self.layer_idx,
          rngs=rngs,
      )
    else:
      self.attention = Qwen3NextGatedDeltaNet(config=cfg, rngs=rngs)

    # Second LayerNorm, applied before the MoE block.
    self.post_attention_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        eps=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Instantiate our `Qwen3NextSparseMoeBlock`.
    self.mlp = Qwen3NextSparseMoeBlock(config=cfg, mesh=self.mesh, quant=self.quant, rngs=rngs)

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    residual = inputs

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Conditionally apply either the Linear Attention or Full Attention block.
    if isinstance(self.attention, Qwen3NextFullAttention):
      attention_output, kv_cache = cast(Qwen3NextFullAttention, self.attention)(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    elif isinstance(self.attention, Qwen3NextGatedDeltaNet):
      attention_output = cast(Qwen3NextGatedDeltaNet, self.attention)(hidden_states)
    else:
      raise TypeError(f"Unexpected type for self.attention: {type(self.attention)}")

    # First residual connection after attention
    hidden_states = residual + attention_output
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Prepare for the MoE block by capturing the new residual
    residual = hidden_states

    # Second LayerNorm, applied before the MoE block.
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Instantiate and call our `Qwen3NextSparseMoeBlock`.
    mlp_output, load_balance_loss = self.mlp(hidden_states, deterministic=deterministic)

    # We sow the load balancing loss so it can be collected and added to the total loss
    # during training.
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    # Final residual connection (after the MoE block)
    layer_output = residual + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        self.activation_axis_names,
    )

    return layer_output, kv_cache


# -----------------------------------------
# The Base Decoder Layer for Qwen3
# -----------------------------------------
class AttentionWithNorm(nnx.Module):
  """Base class with shared common components: self-attention block with normalization."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    # Corresponds to Qwen3's `input_layernorm`
    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Self-attention block
    query_pre_attn_scalar = config.head_dim**-0.5  # Qwen3 specific scaling
    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        use_qk_norm=config.use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode,
        rngs=rngs,
    )

    # Post Attention LayerNorm (corresponds to Qwen3's `post_attention_layernorm`)
    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

  def apply_attention_with_norm(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    """Applies self-attention with pre and post-layer normalization."""
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # Pre attention norm
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)
    # Self attention
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    # Residual connection after attention
    intermediate_inputs = inputs + attention_lnx
    # Post attention norm
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)
    return hidden_states, intermediate_inputs, kv_cache


# -----------------------------------------
# The Dense Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3DecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (dense)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


# -----------------------------------------
# The MoE Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3MoeDecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (MoE)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.moe_block = RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=config.moe_mlp_dim,  # same as config.mlp_dim
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    mlp_lnx, load_balance_loss, _ = self.moe_block(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


class Qwen3OmniMoeVisionPatchMerger(nnx.Module):
  """Vision patch merger that spatially merges patches using an MLP.

  Attributes:
      config: Config containing model parameters
      hidden_size: Hidden dimension after spatial merging
      use_postshuffle_norm: Whether to apply normalization after spatial shuffle
      dtype: Data type for computation
      weight_dtype: Data type for weights
      kernel_init: Initializer for kernel weights
      rngs: RNG state for initialization
      ln_q: LayerNorm before MLP
      mlp_0: First MLP layer
      mlp_2: Second MLP layer
  """

  def __init__(
      self,
      config: Config,
      use_postshuffle_norm: bool = False,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: max_initializers.NdInitializer = max_initializers.nd_dense_init(1.0, "fan_in", "normal"),
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3Omni vision patch merger.

    Args:
        config: Config containing model parameters
        use_postshuffle_norm: Whether to apply normalization after spatial shuffle
        dtype: Data type for computation
        weight_dtype: Data type for weights
        kernel_init: Initializer for kernel weights
        rngs: RNG state for initialization
    """
    self.config = config
    self.use_postshuffle_norm = use_postshuffle_norm
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init
    self.rngs = rngs

    # Calculate hidden_size after spatial merge
    spatial_merge_size = config.spatial_merge_size_for_vit
    base_hidden_size = config.hidden_size_for_vit
    out_hidden_size = config.out_hidden_size_for_vit

    self.hidden_size = base_hidden_size * (spatial_merge_size**2)

    # LayerNorm before MLP
    ln_features = self.hidden_size if use_postshuffle_norm else base_hidden_size
    self.ln_q = nnx.LayerNorm(
        num_features=ln_features,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        rngs=rngs,
    )

    # MLP layers: Linear -> GELU -> Linear
    self.mlp_0 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.mlp_2 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=out_hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden: Array) -> Array:
    """
    Args:
        hidden: Input tensor of shape (batch, seq_len, base_hidden_size) after spatial reordering

    Returns:
        Output tensor of shape (batch, seq_len//merge_size**2, out_hidden_size) - spatially merged
    """
    # Get dimensions
    spatial_merge_size = self.config.spatial_merge_size_for_vit
    base_hidden_size = self.config.hidden_size_for_vit
    tokens_per_block = spatial_merge_size**2

    batch_size = hidden.shape[0]
    seq_len = hidden.shape[1]
    num_blocks = seq_len // tokens_per_block

    hidden = hidden.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    # Apply layer norm
    if self.use_postshuffle_norm:
      hidden = self.ln_q(hidden)
    else:
      hidden_unmerged = hidden.reshape(batch_size, seq_len, base_hidden_size)
      hidden_unmerged = self.ln_q(hidden_unmerged)
      hidden = hidden_unmerged.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    # MLP: Linear -> GELU -> Linear
    hidden = self.mlp_0(hidden)
    hidden = jax.nn.gelu(hidden)
    hidden = self.mlp_2(hidden)

    return hidden


class Qwen3OmniMoeVisionMLP(nnx.Module):
  """Vision MLP block with GELU activation.

  Attributes:
      config: Config containing model parameters
      hidden_size: Hidden dimension size
      intermediate_size: Intermediate dimension size
      dtype: Data type for computation
      weight_dtype: Data type for weights
      kernel_init: Initializer for kernel weights
      rngs: RNG state for initialization
      linear_fc1: First linear layer
      linear_fc2: Second linear layer
  """

  def __init__(
      self,
      config: Config,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: max_initializers.NdInitializer = max_initializers.nd_dense_init(1.0, "fan_in", "normal"),
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3Omni vision MLP.

    Args:
        config: Config containing model parameters
        dtype: Data type for computation
        weight_dtype: Data type for weights
        kernel_init: Initializer for kernel weights
        rngs: RNG state for initialization
    """
    self.config = config
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init
    self.rngs = rngs

    self.hidden_size = config.hidden_size_for_vit
    self.intermediate_size = config.intermediate_size_for_vit

    self.linear_fc1 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.intermediate_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.linear_fc2 = DenseGeneral(
        in_features_shape=self.intermediate_size,
        out_features_shape=self.hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_state: Array) -> Array:
    """
    Args:
        hidden_state: Input tensor of shape (..., hidden_size) - supports packed sequences

    Returns:
        Output tensor of shape (..., hidden_size)
    """
    hidden_state = self.linear_fc1(hidden_state)
    hidden_state = jax.nn.gelu(hidden_state)
    hidden_state = self.linear_fc2(hidden_state)
    return hidden_state


class Qwen3OmniMoeVisionPatchEmbed(nnx.Module):
  """3D convolution-based patch embedding for vision inputs.

  Attributes:
      config: Config containing model parameters
      patch_size: Spatial patch size
      temporal_patch_size: Temporal patch size
      in_channels: Number of input channels
      embed_dim: Embedding dimension
      dtype: Data type for computation
      weight_dtype: Data type for weights
      rngs: RNG state for initialization
      proj: Convolution projection layer
  """

  def __init__(
      self,
      config: Config,
      # Default to float32 for numerical stability in 3D convolutions on image/video inputs
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3Omni vision patch embedding.

    Args:
        config: Config containing model parameters
        dtype: Data type for computation (defaults to float32 for numerical stability)
        weight_dtype: Data type for weights (defaults to float32 for numerical stability)
        rngs: RNG state for initialization
    """
    self.config = config
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rngs = rngs

    self.patch_size = config.patch_size_for_vit
    self.temporal_patch_size = config.temporal_patch_size_for_vit
    self.in_channels = config.num_channels_for_vit
    self.embed_dim = config.hidden_size_for_vit

    kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)

    self.proj = nnx.Conv(
        in_features=self.in_channels,
        out_features=self.embed_dim,
        kernel_size=kernel_size,
        strides=kernel_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    """
    Args:
        hidden_states: Input tensor of shape (batch, in_channels, temporal*patch_size, height*patch_size, width*patch_size)
    Returns:
        Output tensor of shape (batch, T*H*W, embed_dim) where T, H, W are the number of patches
    """
    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    hidden_states = self.proj(hidden_states)
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1] * hidden_states.shape[2] * hidden_states.shape[3]
    hidden_states = hidden_states.reshape(batch_size, seq_len, self.embed_dim)
    return hidden_states


class Qwen3OmniMoeVisionAttention(nnx.Module):
  """Vision attention layer wrapper.

  Attributes:
      config: Config containing model parameters
      attn: Underlying attention module
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision attention layer.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    head_dim = self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit
    # Vision uses full SA, no kv cache
    self.attn = Attention(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=head_dim,
        max_target_length=self.config.num_position_embeddings_for_vit,
        attention_kernel="dot_product",
        inputs_q_shape=(1, 1, self.config.hidden_size_for_vit),
        inputs_kv_shape=(1, 1, self.config.hidden_size_for_vit),
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=mesh,
        dropout_rate=0.0,
        attention_type=AttentionType.FULL,
        is_nope_layer=False,
        use_bias_in_projections=True,
        is_vision=True,
        use_qk_norm=False,
        query_pre_attn_scalar=head_dim ** (-0.5),
        model_mode="train",
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      num_frames: int,
      height: int,
      width: int,
      deterministic: bool = True,
  ) -> Array:
    """
    Args:
        hidden_states: Input tensor of shape (batch, T*H*W, hidden_size)
        num_frames: Number of temporal frames (static)
        height: Height in patches (static)
        width: Width in patches (static)
        deterministic: Whether to use deterministic mode (disable dropout)

    Returns:
        Output tensor of shape (batch, T*H*W, hidden_size)
    """
    # Pass through attention with static dimensions via rope_kwargs
    rope_kwargs = {
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }
    output, _ = self.attn(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
        rope_kwargs=rope_kwargs,
    )

    return output


class Qwen3OmniMoeVisionBlock(nnx.Module):
  """Vision transformer block with attention and MLP.

  Attributes:
      config: Config containing model parameters
      ln1: LayerNorm before attention
      ln2: LayerNorm before MLP
      attn: Attention module
      mlp: First MLP layer
      mlp_out: Second MLP layer
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision transformer block.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    hs = self.config.hidden_size_for_vit
    self.ln1 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.ln2 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.attn = Qwen3OmniMoeVisionAttention(config=config, mesh=mesh, rngs=rngs)
    self.mlp = DenseGeneral(
        in_features_shape=hs,
        out_features_shape=self.config.intermediate_size_for_vit,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.mlp_out = DenseGeneral(
        in_features_shape=self.config.intermediate_size_for_vit,
        out_features_shape=hs,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      x: Array,
      num_frames: int,
      height: int,
      width: int,
  ) -> Array:
    """
    Args:
        x: Input tensor of shape (batch, T*H*W, hidden_size)
        num_frames: Number of temporal frames (static)
        height: Height in patches (static)i
        width: Width in patches (static)

    Returns:
        Output tensor of shape (batch, T*H*W, hidden_size)
    """
    x = x + self.attn(self.ln1(x), num_frames=num_frames, height=height, width=width)
    y = self.ln2(x)
    y = self.mlp(y)
    y = jax.nn.gelu(y)
    y = self.mlp_out(y)
    return x + y


class Qwen3OmniMoeVisionEncoder(nnx.Module):
  """Vision encoder with patch embedding, positional embedding, and transformer blocks.

  Attributes:
      config: Config containing model parameters
      patch_embed: Patch embedding module
      pos_embed_interpolate: Position embedding interpolation module
      blocks: List of transformer blocks
      merger_list: List of patch mergers for deep supervision
      spatial_merge_size: Size of spatial merging
      deep_idx: Indices of layers to extract deep features from
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision encoder.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    self.patch_embed = Qwen3OmniMoeVisionPatchEmbed(config=config, rngs=rngs)

    num_pos = config.num_position_embeddings_for_vit
    hs = config.hidden_size_for_vit
    self.spatial_merge_size = config.spatial_merge_size_for_vit

    self.pos_embed_interpolate = Qwen3OmniMoeVisionPosEmbedInterpolate(
        num_position_embeddings=num_pos,
        hidden_size=hs,
        spatial_merge_size=self.spatial_merge_size,
        rngs=rngs,
    )

    self.depth = config.num_hidden_layers_for_vit

    # Use setattr with string names instead of nnx.List to avoid Orbax integer key bug
    for i in range(self.depth):
      block_name = f"blocks_{i}"
      block = Qwen3OmniMoeVisionBlock(config=config, mesh=mesh, rngs=rngs)
      setattr(self, block_name, block)

    self.deep_idx = tuple(config.deepstack_visual_indexes_for_vit)
    # Use setattr with string names instead of nnx.List to avoid Orbax integer key bug
    for i, _ in enumerate(self.deep_idx):
      merger_name = f"merger_{i}"
      merger = Qwen3OmniMoeVisionPatchMerger(config=config, use_postshuffle_norm=True, rngs=rngs)
      setattr(self, merger_name, merger)

  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = True,
  ):
    """
    Args:
        hidden_states: Input visual tokens of shape (batch, in_channels, T*patch_size, H*patch_size, W*patch_size)
        deterministic: Whether to use deterministic mode

    Returns:
        Tuple of:
        - encoder_output: shape (batch, T*H*W, hidden_size_for_vit)
        - deep_features: List of intermediate features, each of shape (batch, T*H*W, out_hidden_size)
    """
    _, _, num_frames, height, width = hidden_states.shape
    num_frames = num_frames // self.config.temporal_patch_size_for_vit
    height = height // self.config.patch_size_for_vit
    width = width // self.config.patch_size_for_vit

    x = self.patch_embed(hidden_states)
    pos = self.pos_embed_interpolate(num_frames, height, width)

    pos = pos[jnp.newaxis, :, :]
    x = x + pos

    h_traj = []
    for i in range(self.depth):
      block_name = f"blocks_{i}"
      blk = getattr(self, block_name)
      x = blk(x, num_frames=num_frames, height=height, width=width)
      h_traj.append(x)

    deep_feats = []
    for i, idx in enumerate(self.deep_idx):
      h = h_traj[idx]
      merger_name = f"merger_{i}"
      merger = getattr(self, merger_name)
      deep_feat = merger(h)
      deep_feats.append(deep_feat)

    return x, deep_feats


class Qwen3OmniMoeVisionProjector(nnx.Module):
  """Projection layer that converts vision encoder output to model embedding space.

  Attributes:
      config: Config containing model parameters
      merger: Patch merger for spatial reduction
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision projector.

    Args:
        config: Config containing model parameters
        rngs: RNG state for initialization
    """
    self.config = config
    self.merger = Qwen3OmniMoeVisionPatchMerger(config=config, use_postshuffle_norm=False, rngs=rngs)

  def __call__(self, hidden_states: Array) -> Array:
    """
    Args:
        hidden_states: Encoder output of shape (batch, T*H*W, hidden_size_for_vit)

    Returns:
        Projected output of shape (batch, T*H*W//merge_size**2, out_hidden_size_for_vit)
    """
    output = self.merger(hidden_states)
    return output


def qwen3omni_visionencoder_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  """Convert Qwen3OmniMoeVisionEncoder to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniMoeVisionEncoder,
      config=config,
      mesh=mesh,
      name="Qwen3OmniMoeVisionEncoder_0",
      abstract_init=False,
      metadata_fn=max_initializers.variable_to_logically_partitioned,
  )


def qwen3omni_visionprojector_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  """Convert Qwen3OmniMoeVisionProjector to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniMoeVisionProjector,
      config=config,
      name="Qwen3OmniMoeVisionProjector_0",
      abstract_init=False,
      metadata_fn=max_initializers.variable_to_logically_partitioned,
  )


class Qwen3OmniAudioEncoderLayer(nnx.Module):
  """Transformer encoder layer for audio model."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.hidden_states_shape = (
        self.config.per_device_batch_size,
        self.config.max_source_positions_for_audio,
        self.config.d_model_for_audio,
    )

    self.input_layer_norm = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    self.self_attention_audio = Attention(
        config=self.config,
        num_query_heads=self.config.encoder_attention_heads_for_audio,
        num_kv_heads=self.config.encoder_attention_heads_for_audio,
        head_dim=self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio,
        max_target_length=self.config.max_source_positions_for_audio,
        attention_kernel="dot_product",
        inputs_q_shape=self.hidden_states_shape,
        inputs_kv_shape=self.hidden_states_shape,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=self.mesh,
        dropout_rate=self.config.attention_dropout_for_audio,
        name="self_attention_audio",
        attention_type=AttentionType.FULL,
        is_nope_layer=True,  # No rotary position embeddings for audio
        use_bias_in_projections=True,
        use_qk_norm=False,
        query_pre_attn_scalar=1
        / math.sqrt(self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio),
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )

    self.post_attention_layer_norm = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    self.AudioMLP = MlpBlock(
        config=self.config,
        mesh=self.mesh,
        in_features=self.config.d_model_for_audio,
        intermediate_dim=self.config.encoder_ffn_dim_for_audio,
        activations=("gelu",),  # Single GELU activation
        kernel_init=max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        intermediate_dropout_rate=0.0,  # No dropout to match AudioMLP
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        use_bias=True,  # AudioMLP uses bias
        use_pre_norm=False,  # Norm is handled outside
        quant=None,  # No quantization
        model_mode=None,  # Not needed for encoder
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = False,
  ):
    """Apply transformer encoder layer to audio hidden states.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, d_model_for_audio)
        deterministic: Whether to use deterministic mode (disable dropout)

    Returns:
        Output tensor of shape (batch, seq_len, d_model_for_audio)
    """
    residual = hidden_states
    hidden_states = self.input_layer_norm(hidden_states)
    hidden_states, _ = self.self_attention_audio(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layer_norm(hidden_states)
    hidden_states = self.AudioMLP(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class Qwen3OmniAudioEncoder(nnx.Module):
  """Full audio encoder with convs, positional embeddings, and transformer layers.

  Attributes:
      config: Config containing model parameters
      mesh: Mesh, JAX device mesh (used for sharding)
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.positional_embedding = PositionalEmbedding(
        embedding_dims=self.config.d_model_for_audio,
        max_wavelength=self.config.max_timescale_for_audio,
        cast_as_fprop_dtype=True,
        fprop_dtype=self.config.dtype_mm,
    )

    self.layernorm_post = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    # Convolutional downsampling layers
    self.conv2d1 = nnx.Conv(
        in_features=1,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.conv2d2 = nnx.Conv(
        in_features=self.config.downsample_hidden_size_for_audio,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.conv2d3 = nnx.Conv(
        in_features=self.config.downsample_hidden_size_for_audio,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    conv_out_dim = self.config.downsample_hidden_size_for_audio * (
        (((self.config.num_mel_bins_for_audio + 1) // 2 + 1) // 2 + 1) // 2
    )
    self.conv_out = DenseGeneral(
        in_features_shape=conv_out_dim,
        out_features_shape=self.config.d_model_for_audio,
        use_bias=False,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    # Transformer encoder layers
    for lyr in range(self.config.encoder_layers_for_audio):
      layer_name = f"layers_{lyr}"
      layer = Qwen3OmniAudioEncoderLayer(
          config=self.config,
          mesh=self.mesh,
          rngs=self.rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      audio_features: Array,
      deterministic: bool = False,
  ):
    """Process audio features through convs + transformer encoder.

    Args:
        audio_features: Input of shape (batch, num_mel_bins, audio_length)
        deterministic: Whether to use deterministic mode

    Returns:
        Encoded features of shape (batch, seq_len, d_model_for_audio)
    """
    batch_size, num_mel_bins, audio_length = audio_features.shape
    chunk_size = self.config.n_window_for_audio * 2

    # Reshape to chunks
    num_chunks = audio_length // chunk_size
    audio_chunks = audio_features.reshape(batch_size, num_mel_bins, num_chunks, chunk_size)
    audio_chunks = audio_chunks.transpose(0, 2, 1, 3)
    audio_chunks = audio_chunks.reshape(batch_size * num_chunks, num_mel_bins, chunk_size)

    # Add channel dimension
    hidden_states = audio_chunks[:, :, :, jnp.newaxis]

    # Apply convolutional layers
    hidden_states = self.conv2d1(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.conv2d2(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.conv2d3(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)

    # Reshape conv output
    bc, f, t, c = hidden_states.shape
    hidden_states = hidden_states.transpose(0, 2, 3, 1)
    hidden_states = hidden_states.reshape(bc, t, c * f)
    hidden_states = self.conv_out(hidden_states)

    # Add positional embeddings
    seq_len_per_chunk = hidden_states.shape[1]
    pos_emb = self.positional_embedding(seq_len_per_chunk)
    pos_emb = jnp.broadcast_to(
        pos_emb[None, :, :], (batch_size * num_chunks, seq_len_per_chunk, self.config.d_model_for_audio)
    )
    hidden_states = hidden_states + pos_emb

    # Apply transformer encoder layers
    for lyr in range(self.config.encoder_layers_for_audio):
      layer_name = f"layers_{lyr}"
      layer = getattr(self, layer_name)
      hidden_states = layer(
          hidden_states,
          deterministic=deterministic,
      )

    hidden_states = self.layernorm_post(hidden_states)

    # Reshape back: (batch*chunks, seq_len_per_chunk, d_model) -> (batch, chunks*seq_len_per_chunk, d_model)
    hidden_states = hidden_states.reshape(batch_size, num_chunks * seq_len_per_chunk, self.config.d_model_for_audio)

    return hidden_states


class Qwen3OmniAudioProjector(nnx.Module):
  """Projection layer that converts audio encoder output to model embedding space."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    self.config = config
    self.proj1 = DenseGeneral(
        in_features_shape=config.d_model_for_audio,
        out_features_shape=config.d_model_for_audio,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.proj2 = DenseGeneral(
        in_features_shape=config.d_model_for_audio,
        out_features_shape=config.output_dim_for_audio,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    """
    Args:
        hidden_states: Encoder output of shape (num_chunks, seq_len, d_model_for_audio)

    Returns:
        Projected output of shape (num_chunks, seq_len, output_dim_for_audio)
    """
    hidden_states = self.proj1(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.proj2(hidden_states)
    return hidden_states


def qwen3omni_audioencoder_as_linen(config: Config, mesh: Mesh):
  """Convert AudioEncoder (convs + transformer layers, no projector) to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniAudioEncoder,
      config=config,
      mesh=mesh,
      name="Qwen3OmniAudioEncoder_0",
      abstract_init=False,
      metadata_fn=variable_to_logically_partitioned,
  )


def qwen3omni_audioprojector_as_linen(config: Config, mesh: Mesh):
  """Convert AudioProjector to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniAudioProjector,
      config=config,
      name="Qwen3OmniAudioProjector_0",
      abstract_init=False,
      metadata_fn=variable_to_logically_partitioned,
  )


# Vision encoder Linen wrappers
Qwen3OmniMoeVisionPatchMergerToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionPatchMerger,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionMLPToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionMLP,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionPatchEmbedToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionPatchEmbed,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionAttentionToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionAttention,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionEncoderToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionEncoder,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionProjectorToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionProjector,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3MoeDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3MoeDecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3NextDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3NextDecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3NextScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3NextScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

# Audio encoder Linen wrappers
Qwen3OmniAudioEncoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniAudioEncoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniAudioEncoderToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniAudioEncoder,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniAudioProjectorToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniAudioProjector,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
