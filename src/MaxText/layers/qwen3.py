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

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn

from MaxText.common_types import Config, DType, Array
from MaxText.layers import attentions
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import moe
from MaxText.layers import quantizations
from MaxText.layers.normalizations import rms_norm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager

# -----------------------------------------
# Helper functions for Qwen3 layers
# -----------------------------------------


def self_attention_with_norm(
    inputs: jnp.ndarray,
    cfg: Config,
    mesh: Mesh,
    quant: None | Quant,
    decoder_segment_ids: None | jnp.ndarray,
    decoder_positions: None | jnp.ndarray,
    deterministic: bool,
    model_mode: str,
):
  """A helper function for self-attention block with normalization."""

  inputs_checkpoint = checkpoint_name(inputs, "decoder_layer_input")

  # Corresponds to Qwen3's `input_layernorm`
  lnx = rms_norm(
      num_features=inputs.shape[-1],
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      name="pre_self_attention_layer_norm",
      epsilon=cfg.normalization_layer_epsilon,
      kernel_axes=("norm",),
  )(inputs_checkpoint)
  lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

  # Self-attention block
  attention_layer = attentions.attention_as_linen(
      config=cfg,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      attention_kernel=cfg.attention,
      inputs_q_shape=lnx.shape,
      inputs_kv_shape=lnx.shape,
      mesh=mesh,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      dropout_rate=cfg.dropout_rate,
      name="self_attention",
      quant=quant,
      kv_quant=quantizations.configure_kv_quant(cfg),
      use_qk_norm=cfg.use_qk_norm,
      query_pre_attn_scalar=(cfg.head_dim**-0.5),  # Qwen3 specific scaling
      model_mode=model_mode,
  )

  attention_output = attention_layer(
      lnx,  # inputs_q
      lnx,  # inputs_kv
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=deterministic,
      model_mode=model_mode,
  )
  attention_output = nn.with_logical_constraint(
      attention_output, ("activation_batch", "activation_length", "activation_embed")
  )

  # Residual connection after attention
  residual_after_attention = inputs_checkpoint + attention_output

  # Post Attention LayerNorm (corresponds to Qwen3's `post_attention_layernorm`)
  hidden_states = rms_norm(
      num_features=residual_after_attention.shape[-1],
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      name="post_self_attention_layer_norm",
      epsilon=cfg.normalization_layer_epsilon,
      kernel_axes=("norm",),
  )(residual_after_attention)
  hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", "activation_length", "activation_embed"))

  return hidden_states, residual_after_attention

from flax.linen import initializers

# -----------------------------------------
# Qwen3-Next Layer Implementations
# -----------------------------------------

class Qwen3NextRMSNormGated(nn.Module):
  """
  A JAX/Flax implementation of Qwen3NextRMSNormGated.
  This applies RMS Normalization and then a gated activation function (SiLU).
  Matches the behavior of the PyTorch version in modeling_qwen3_next.py.

  Attributes:
    num_features: The number of features in the input.
    eps: A small epsilon value to prevent division by zero in RMSNorm.
    dtype: The datatype of the computation.
  """
  num_features: int
  eps: float = 1e-6
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, hidden_states: Array, gate: Array) -> Array:
    """
    Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized. Shape: (..., num_features)
      gate: The gating array for the activation. Shape: (..., num_features)

    Returns:
      The normalized and gated output array. Shape: (..., num_features)
    """
    weight = self.param('weight', initializers.ones, (self.num_features,), self.dtype)

    # RMS Normalization logic
    # Intermediate calculations are upcasted to float32 for numerical stability,
    # matching PyTorch's default behavior.
    hidden_states_f32 = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_states_f32), axis=-1, keepdims=True)
    normalized_states = hidden_states_f32 * jax.lax.rsqrt(variance + self.eps)
    normalized_states = (normalized_states * weight.astype(jnp.float32))

    # Gated Activation
    gated_states = normalized_states * nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


def l2norm(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
  """L2 normalization function.

  Args:
    x: Input array.
    axis: The axis or axes along which to normalize.
    eps: Small epsilon to prevent division by zero.

  Returns:
    L2 normalized array.
  """
  x_f32 = x.astype(jnp.float32)
  inv_norm = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=axis, keepdims=True) + eps)
  return (x * inv_norm).astype(x.dtype)

def jax_chunk_gated_delta_rule(
    query: Array, key: Array, value: Array, g: Array, beta: Array, chunk_size: int = 64
) -> Array:
    """
    A JAX implementation of the chunked Gated Delta Rule.
    Matches the PyTorch implementation in transformers.models.qwen3_next.modeling_qwen3_next.torch_chunk_gated_delta_rule

    B: Batch size, H: Number of heads, L: Sequence length, N_CHUNKS: Number of chunks
    CS: Chunk size, D_k: Key/Query head dimension, D_v: Value head dimension

    Args:
      query: Query tensor. Shape (B, H, L, D_k)
      key: Key tensor. Shape (B, H, L, D_k)
      value: Value tensor. Shape (B, H, L, D_v)
      g: Decay tensor. Shape (B, H, L)
      beta: Gate tensor. Shape (B, H, L)
      chunk_size: The size of each chunk for processing.

    Returns:
      Output tensor. Shape (B, H, L, D_v)
    """
    orig_dtype = query.dtype
    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Use float32 for most intermediate computations for better precision.
    query_f32, key_f32, value_f32, g_f32, beta_f32 = [x.astype(jnp.float32) for x in (query, key, value, g, beta)]

    # Pad sequences to be divisible by chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    if pad_size > 0:
      query_f32 = jnp.pad(query_f32, ((0, 0), (0, 0), (0, pad_size), (0, 0)), 'constant', constant_values=0.0)
      key_f32 = jnp.pad(key_f32, ((0, 0), (0, 0), (0, pad_size), (0, 0)), 'constant', constant_values=0.0)
      value_f32 = jnp.pad(value_f32, ((0, 0), (0, 0), (0, pad_size), (0, 0)), 'constant', constant_values=0.0)
      beta_f32 = jnp.pad(beta_f32, ((0, 0), (0, 0), (0, pad_size)), 'constant', constant_values=0.0)
      g_f32 = jnp.pad(g_f32, ((0, 0), (0, 0), (0, pad_size)), 'constant', constant_values=0.0)

    total_sequence_length = seq_len + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply L2 norm and scaling
    query_f32 = l2norm(query_f32, axis=-1)
    key_f32 = l2norm(key_f32, axis=-1)
    query_f32 = query_f32 * (k_head_dim**-0.5)

    v_beta = value_f32 * beta_f32[..., None] # (B, H, L, D_v)
    k_beta = key_f32 * beta_f32[..., None]   # (B, H, L, D_k)

    # Reshape for chunked processing: (B, H, N_CHUNKS, CHUNK_SIZE, DIM)
    query_c, key_c, _, k_beta_c, v_beta_c = [
        x.reshape(batch_size, num_heads, num_chunks, chunk_size, -1)
        for x in (query_f32, key_f32, value_f32, k_beta, v_beta)
    ]
    g_c = g_f32.reshape(batch_size, num_heads, num_chunks, chunk_size)

    # Intra-chunk computations (Independent of the scan)
    g_cumsum = jnp.cumsum(g_c, axis=-1) # (B, H, N_CHUNKS, CHUNK_SIZE)
    # decay_mask: (B, H, N_CHUNKS, CHUNK_SIZE, CHUNK_SIZE)
    decay_mask = jnp.exp(g_cumsum[..., :, :, None] - g_cumsum[..., :, None, :])
    decay_mask = jnp.tril(decay_mask)

    mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=0)

    # attn: (B, H, N_CHUNKS, CHUNK_SIZE, CHUNK_SIZE)
    # This is the base intra-chunk attention matrix, influenced by decay
    attn = -((jnp.einsum('...csd,...ctd->...cst', k_beta_c, key_c, precision=jax.lax.Precision.HIGHEST)) * decay_mask)
    attn = jnp.where(mask, 0.0, attn)

    # Iterative refinement of the intra-chunk attention, as in the original PyTorch.
    for i in range(1, chunk_size):
       row = attn[..., i, :i]
       sub = attn[..., :i, :i]
       update = row + jnp.einsum('...ci,...cij->...cj', row, sub, precision=jax.lax.Precision.HIGHEST)
       attn = attn.at[..., i, :i].set(update)

    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)

    # Precompute value transformed by intra-chunk attention and k_cumdecay
    # value_intra: (B, H, N_CHUNKS, CHUNK_SIZE, D_v)
    value_intra = jnp.einsum('...cst,...ctv->...csv', attn, v_beta_c, precision=jax.lax.Precision.HIGHEST)
    # k_cumdecay: (B, H, N_CHUNKS, CHUNK_SIZE, D_k)
    k_cumdecay = jnp.einsum('...cst,...ctd->...csd', attn, k_beta_c * jnp.exp(g_c[..., None]), precision=jax.lax.Precision.HIGHEST)

    # Prepare for lax.scan: Transpose N_CHUNKS to the leading axis.
    query_s, key_s = [x.transpose(2, 0, 1, 3, 4) for x in (query_c, key_c)]
    value_intra_s = value_intra.transpose(2, 0, 1, 3, 4)
    k_cumdecay_s = k_cumdecay.transpose(2, 0, 1, 3, 4)
    g_s = g_c.transpose(2, 0, 1, 3)
    decay_mask_s = decay_mask.transpose(2, 0, 1, 3, 4)

    def chunk_scanner(carry, xs):
        """Performs the recurrent update across chunks."""
        recurrent_state = carry # Shape: (B, H, D_k, D_v)
        q_i, k_i, v_i, g_i, k_cumdecay_i, decay_mask_i = xs

        # Inter-chunk communication: Interaction with the recurrent state
        # v_prime: (B, H, CHUNK_SIZE, D_v)
        v_prime = jnp.einsum('bhsd,bhdv->bhsv', k_cumdecay_i, recurrent_state, precision=jax.lax.Precision.HIGHEST)
        v_new = v_i - v_prime

        # attn_inter: (B, H, CHUNK_SIZE, D_v)
        attn_inter = jnp.einsum('bhsd,bhdv->bhsv', q_i * jnp.exp(g_i[..., None]), recurrent_state, precision=jax.lax.Precision.HIGHEST)

        # Intra-chunk attention within the scan loop
        mask_intra = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)
        # attn_intra: (B, H, CHUNK_SIZE, CHUNK_SIZE)
        attn_intra = jnp.einsum('bhsd,bhtd->bhst', q_i, k_i, precision=jax.lax.Precision.HIGHEST) * decay_mask_i
        attn_intra = jnp.where(mask_intra, 0.0, attn_intra)

        # Combine inter and intra chunk outputs
        # chunk_output: (B, H, CHUNK_SIZE, D_v)
        chunk_output = attn_inter + jnp.einsum('bhst,bhtv->bhsv', attn_intra, v_new, precision=jax.lax.Precision.HIGHEST)

        # Update recurrent state for the next chunk
        g_i_last_val = g_i[..., -1, None, None]
        exp_g_last = jnp.exp(g_i_last_val)
        exp_g_diff = jnp.exp(g_i[..., -1, None] - g_i) # Shape (B, H, CHUNK_SIZE)
        k_i_weighted = k_i * exp_g_diff[..., None]   # Shape (B, H, CHUNK_SIZE, D_k)

        # update_term: (B, H, D_k, D_v)
        update_term = jnp.einsum('bhsd,bhsv->bhdv', k_i_weighted, v_new, precision=jax.lax.Precision.HIGHEST)
        next_recurrent_state = (recurrent_state * exp_g_last + update_term)

        return next_recurrent_state, chunk_output

    initial_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)

    # Run the scan over the chunks
    _, all_chunk_outputs = jax.lax.scan(
        chunk_scanner,
        initial_recurrent_state,
        (query_s, key_s, value_intra_s, g_s, k_cumdecay_s, decay_mask_s)
    )

    # Transpose back and reshape to (B, H, L, D_v)
    output = all_chunk_outputs.transpose(1, 2, 0, 3, 4).reshape(
        batch_size, num_heads, total_sequence_length, v_head_dim)

    return output[:, :, :seq_len, :].astype(orig_dtype)

class Qwen3NextGatedDeltaNet(nn.Module):
  """
  JAX/Flax implementation of Qwen3NextGatedDeltaNet for training.
  This module combines a 1D convolution, projections, and the jax_chunk_gated_delta_rule
  to form a linear attention layer.

  Attributes:
    config: MaxText configuration object.
  """
  config: Config

  @nn.compact
  def __call__(self, hidden_states: Array, deterministic: bool) -> Array:
    cfg = self.config
    in_features = hidden_states.shape[-1]
    num_v_heads = cfg.linear_num_value_heads
    num_k_heads = cfg.linear_num_key_heads
    head_k_dim = cfg.linear_key_head_dim
    head_v_dim = cfg.linear_value_head_dim
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim
    conv_kernel_size = cfg.linear_conv_kernel_dim

    # Input projections for Q, K, V, Z and and B, A
    # Using linears.dense_general wrapper to handle NNX instantiation within Linen
    qkvz = linears.dense_general(
        inputs_shape=hidden_states.shape,
        out_features_shape=(key_dim * 2 + value_dim * 2),
        dtype=cfg.dtype,
        kernel_axes=('embed', 'mlp'),
        matmul_precision=cfg.matmul_precision,
        name='in_proj_qkvz')(hidden_states)
    ba = linears.dense_general(
        inputs_shape=hidden_states.shape,
        out_features_shape=(num_v_heads * 2),
        dtype=cfg.dtype,
        kernel_axes=('embed', 'mlp'),
        matmul_precision=cfg.matmul_precision,
        name='in_proj_ba')(hidden_states)

    # Split the projections
    q, k, v, z = jnp.split(qkvz, [key_dim, 2 * key_dim, 2 * key_dim + value_dim], axis=-1)
    b, a = jnp.split(ba, [num_v_heads], axis=-1)

    # Causal 1D Convolution
    qkv = jnp.concatenate([q, k, v], axis=-1) # Shape: (B, L, conv_dim)

    conv_layer = nn.Conv(
        features=conv_dim,
        kernel_size=(conv_kernel_size,),
        feature_group_count=conv_dim, # Depthwise-like
        padding='CAUSAL',
        dtype=cfg.dtype,
        precision=cfg.matmul_precision,
        name='conv1d')
    # Input to conv_layer should be (B, L, C)
    qkv_conv = nn.silu(conv_layer(qkv).astype(jnp.float32)).astype(cfg.dtype)
    # Output qkv_conv shape: (B, L, conv_dim)

    q_conv, k_conv, v_conv = jnp.split(qkv_conv, [key_dim, 2 * key_dim], axis=-1)

    # Reshape for multi-head processing
    batch, seq_len, _ = hidden_states.shape
    query = q_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
    key = k_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
    value = v_conv.reshape(batch, seq_len, num_v_heads, head_v_dim)

    # Gated Delta Rule parameters
    a_log = self.param("A_log", initializers.uniform(1.0), (num_v_heads,))
    dt_bias = self.param("dt_bias", initializers.ones, (num_v_heads,))
    beta = nn.sigmoid(b)
    g = -jnp.exp(a_log.astype(jnp.float32)) * nn.softplus(a.astype(jnp.float32) + dt_bias.astype(jnp.float32))
    g = g.astype(cfg.dtype)

    # Handle Grouped Query Attention (GQA) where num_v_heads > num_k_heads
    if num_v_heads // num_k_heads > 1:
        query = jnp.repeat(query, num_v_heads // num_k_heads, axis=2)
        key = jnp.repeat(key, num_v_heads // num_k_heads, axis=2)

    # Transpose to (B, H, L, D) for chunked processing
    query, key, value = [jnp.transpose(x, (0, 2, 1, 3)) for x in (query, key, value)]
    g, beta = [jnp.transpose(x, (0, 2, 1)) for x in (g, beta)]

    # Apply the core Gated Delta Rule
    core_attn_out = jax_chunk_gated_delta_rule(query, key, value, g, beta)

    # Transpose back and reshape for output
    core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).reshape(batch, seq_len, -1)

    # Final RMSNormGated and output projection
    norm_gated_layer = Qwen3NextRMSNormGated(num_features=value_dim, name="norm_gated", dtype=cfg.dtype)
    gated_output = norm_gated_layer(core_attn_out, z)

    output = linears.dense_general(
        inputs_shape=gated_output.shape,
        out_features_shape=(in_features,),
        dtype=cfg.dtype,
        kernel_axes=('mlp', 'embed'),
        matmul_precision=cfg.matmul_precision,
        name='out_proj')(gated_output)

    return output

# -----------------------------------------
# The Dense Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3DecoderLayer(nn.Module):
  """Qwen3 Transformer decoder layer (dense)."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
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
    cfg = self.config

    hidden_states, residual_after_attention = self_attention_with_norm(
        inputs,
        cfg,
        self.mesh,
        self.quant,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )

    # Dense MLP block
    mlp_output = linears.mlp_block(
        in_features=hidden_states.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(hidden_states, deterministic=deterministic)

    # Final residual connection
    layer_output = residual_after_attention + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output


# -----------------------------------------
# The MoE Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3MoeDecoderLayer(nn.Module):
  """Qwen3 Transformer decoder layer (MoE)."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
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
    cfg = self.config

    hidden_states, residual_after_attention = self_attention_with_norm(
        inputs,
        cfg,
        self.mesh,
        self.quant,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )

    # Mixture of Experts block
    mlp_output, load_balance_loss = moe.get_routed_moe(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="moe_block",
        quant=self.quant,
    )(hidden_states)

    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    mlp_output = nn.with_logical_constraint(mlp_output, ("activation_batch", "activation_length", "activation_embed"))

    # Final residual connection
    layer_output = residual_after_attention + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
