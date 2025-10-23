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

from typing import cast

import jax
import jax.nn
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx
from flax.linen import initializers as linen_initializers

from MaxText import max_utils
from MaxText.common_types import Config, DType, Array
from MaxText.layers import attentions
from MaxText.layers import initializers as max_initializers
from MaxText.layers import linears
from MaxText.layers import moe
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.layers.normalizations import RMSNorm, l2norm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import MlpBlock
from MaxText.layers.moe import RoutedMoE


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


class Qwen3NextRMSNorm(nnx.Module):
  """
  Used for input and post attention layernorms
  in Qwen3NextDecoderLayer.

  This normalization layer is specific to Qwen3-Next. Key characteristics:
  1.  The learnable scale parameter `weight` is initialized to ZEROS.
  2.  The scale is applied as `(1.0 + self.weight)`, making the initial scale effectively 1.0.
      This matches the PyTorch implementation of Qwen3NextRMSNorm.
  3.  It is NOT a zero-centered normalization (as in the blog); it still uses the root mean square of the inputs.
      The standard `MaxText.layers.normalizations.RMSNorm` also does not center the data.
  4.  This differs from the standard MaxText `RMSNorm`
      (MaxText.layers.normalizations.RMSNorm) which initializes its scale to ONES
      and applies it multiplicatively (`y * scale`).
  """

  def __init__(self, num_features: int, eps: float, dtype: DType, weight_dtype: DType, *, rngs: nnx.Rngs):
    self.num_features = num_features
    self.eps = eps
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    self.weight = nnx.Param(linen_initializers.zeros(rngs.params(), (self.num_features,), self.weight_dtype))

  def __call__(self, x: Array) -> Array:
    """Applies RMSNorm to the input tensor."""
    weight = self.weight.value
    x_dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + self.eps)
    # Add 1.0 to the learnable weight
    x = x * (1.0 + weight.astype(jnp.float32))
    return x.astype(x_dtype)


class Qwen3NextRMSNormGated(nnx.Module):
  """
  This applies RMS Normalization and then a gated activation function (SiLU).
  This is used within the Qwen3NextGatedDeltaNet.

  Attributes:
    num_features: The number of features in the input.
    eps: A small epsilon value to prevent division by zero in RMSNorm.
    dtype: The datatype of the computation.
    weight_dtype: The datatype of the weights.
  """

  def __init__(self, num_features: int, eps: float, dtype: DType, weight_dtype: DType, *, rngs: nnx.Rngs):
    self.num_features = num_features
    self.eps = eps
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    self.weight = nnx.Param(nnx.initializers.ones(rngs.params(), (self.num_features,), self.weight_dtype))

  def __call__(self, hidden_states: Array, gate: Array) -> Array:
    """
    Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized (o). Shape: (..., F)
      gate: The gating array for the activation (z). Shape: (..., F)
            where F is num_features.

    Returns:
      The normalized and gated output array. Shape: (..., F)
    """
    weight = self.weight.value

    # RMS Normalization logic
    hidden_states_f32 = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_states_f32), axis=-1, keepdims=True)
    normalized_states = hidden_states_f32 * jax.lax.rsqrt(variance + self.eps)
    normalized_states = normalized_states * weight.astype(jnp.float32)

    # Gated Activation using SiLU (Sigmoid-weighted Linear Unit)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


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

  Attributes:
    config: MaxText configuration object.
    dtype: The datatype of the computation.
  """

  def __init__(self, config: Config, dtype: DType = jnp.float32, *, rngs: nnx.Rngs):
    self.config = config
    self.dtype = dtype
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

    # Submodule instantiations
    self.in_proj_qkvz = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.key_dim * 2 + self.value_dim * 2),
        dtype=cfg.dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.in_proj_ba = linears.DenseGeneral(
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
    self.out_proj = linears.DenseGeneral(
        in_features_shape=self.value_dim,
        out_features_shape=(in_features,),
        dtype=cfg.dtype,
        kernel_axes=("mlp", "embed"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    cfg = self.config

    # =========================================================================
    # STEP A: Input Projections
    # =========================================================================
    # hidden_states shape: (B, S, E)
    # qkvz shape: (B, S, 2*key_dim + 2*value_dim)
    qkvz = self.in_proj_qkvz(hidden_states)
    # ba shape: (B, S, 2*H_v)
    ba = self.in_proj_ba(hidden_states)

    # q shape: (B, S, key_dim), k shape: (B, S, key_dim), v shape: (B, S, value_dim), z shape: (B, S, value_dim)
    q, k, v, z = jnp.split(qkvz, [self.key_dim, 2 * self.key_dim, 2 * self.key_dim + self.value_dim], axis=-1)
    # b shape: (B, S, H_v), a shape: (B, S, H_v)
    b, a = jnp.split(ba, [self.num_v_heads], axis=-1)

    # =========================================================================
    # STEP B: 1D Convolution
    # =========================================================================
    # qkv shape: (B, S, conv_dim)
    qkv = jnp.concatenate([q, k, v], axis=-1)

    # TODO(parambole): Implement caching logic for conv_state and recurrent_state

    # Input to conv_layer should be (B, S, C)
    # qkv_conv shape: (B, S, conv_dim)
    qkv_conv = jax.nn.silu(self.conv1d(qkv).astype(jnp.float32)).astype(cfg.dtype)
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
    # We first reshape the `z` tensor to match the multi-head structure of `core_attn_out`.
    # z shape from (B, S, value_dim) -> (B, S, H_v, D_v)
    z_reshaped = z.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

    # Apply the norm and gate. Output shape: (B, S, H_v, D_v)
    gated_output_reshaped = self.norm(core_attn_out, z_reshaped)

    # Reshape back to a single feature dimension for the final projection.
    # Shape from (B, S, H_v, D_v) -> (B, S, value_dim)
    gated_output = gated_output_reshaped.reshape(batch, seq_len, -1)

    # Final output shape: (B, S, E)
    output = self.out_proj(gated_output)

    return output


class Qwen3NextFullAttention(nnx.Module):
  """Placeholder for Qwen3-Next full attention."""

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config

    # TODO(rbierneni): Implement the actual Qwen3NextAttention logic.
    # This is a placeholder. The actual Qwen3NextAttention in the PyTorch code has:
    # 1.  q_proj projection: hidden_size -> num_attention_heads * head_dim * 2.
    #     This output is chunked to get query_states and a 'gate'.
    # 2.  Qwen3NextRMSNorm (self.q_norm and self.k_norm) is applied to query_states
    #     and key_states *before* RoPE. These norms are on the head_dim.
    # 3.  The final attention output is gated: attn_output = attn_output * torch.sigmoid(gate).
    # 4.  k_proj and v_proj are standard Linear layers to num_key_value_heads * head_dim.
    # 5.  RoPE is applied to the normed query and key.
    # 6.  The o_proj maps from num_attention_heads * head_dim back to hidden_size.

    # Placeholder call to standard MaxText attention
    # NOTE: This will NOT match the Qwen3Next behavior or weights.

    # Get mode-specific batch size and sequence length for shape
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
    inputs_shape = (batch_size, seq_len, cfg.emb_dim)

    self.attention_layer = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        mesh=self.mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        use_qk_norm=False,
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
      # TODO(parambole): Add cache arguments
  ):

    # TODO(parambole): Add caching in/out
    attention_output = self.attention_layer(
        inputs,
        inputs,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    return attention_output


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
    self.shared_expert = linears.MlpBlock(
        config=cfg,
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
    self.shared_expert_gate = linears.DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=1,
        use_bias=False,  # Qwen3-Next shared_expert_gate does not have a bias
        dtype=cfg.dtype,
        kernel_init=max_initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "vocab"),
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
    routed_output, load_balance_loss = self.routed_experts(hidden_states)

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
      x = layer(
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
      self.attention = Qwen3NextGatedDeltaNet(config=cfg, dtype=cfg.dtype, rngs=rngs)

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
  ):
    residual = inputs

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Conditionally apply either the Linear Attention or Full Attention block.
    if isinstance(self.attention, Qwen3NextFullAttention):
      attention_output = cast(Qwen3NextFullAttention, self.attention)(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
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
    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    # Final residual connection (after the MoE block)
    layer_output = residual + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        self.activation_axis_names,
    )

    return layer_output


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
  ):
    """Applies self-attention with pre and post-layer normalization."""
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # Pre attention norm
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)
    # Self attention
    attention_lnx = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    # Residual connection after attention
    intermediate_inputs = inputs + attention_lnx
    # Post attention norm
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)
    return hidden_states, intermediate_inputs


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
  ):
    hidden_states, intermediate_inputs = self.apply_attention_with_norm(
        inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode
    )

    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output


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
  ):
    hidden_states, intermediate_inputs = self.apply_attention_with_norm(
        inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode
    )

    mlp_lnx, load_balance_loss = self.moe_block(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)
    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output


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
