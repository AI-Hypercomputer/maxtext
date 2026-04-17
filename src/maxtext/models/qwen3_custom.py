# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# limitations under the License.
"""Custom Qwen3 model decoder layer."""

from typing import Any

from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from maxtext.common.common_types import Config
from maxtext.layers import initializers as max_initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import DenseGeneral
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding
from maxtext.inference import page_manager
from maxtext.models.qwen3 import AttentionWithNorm
from maxtext.layers.normalizations import RMSNorm


class Qwen3CustomAttention(Attention):
  """Custom GQA attention that supports sub-dimensional output."""

  def init_out_w(self, output_dim: int) -> nnx.Module:
    """Initializes the output projection."""
    if not self.config.attention_output_dim > 0:
      raise ValueError("attention_output_dim must be set to a positive integer for CustomAttention.")

    in_features = (self.num_query_heads, self.head_dim)
    out_kernel_axis = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("heads", "kv", "embed")
    )
    axis = (-2, -1)

    return DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.config.attention_output_dim,
        axis=axis,
        kernel_init=self.kernel_init,
        kernel_axes=out_kernel_axis,  # trade speed with memory
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        shard_mode=self.config.shard_mode,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )


class Qwen3CustomMoeDecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (Custom MoE)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    """Initializes the instance.

    Args:
      config: The model configuration.
      mesh: The JAX device mesh.
      model_mode: The current mode of the model (e.g., "train", "decode").
      quant: Quantization configuration, if any.
      rngs: PRNG keys for Flax.
    """
    super().__init__(config, mesh, model_mode, quant, rngs)

    query_pre_attn_scalar = config.head_dim**-0.5
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    # Override self_attention with Qwen3CustomAttention
    self.self_attention = Qwen3CustomAttention(
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
        use_mrope=config.use_mrope,
        mrope_section=config.mrope_section,
        rngs=rngs,
    )

    if config.attention_output_dim <= 0 or config.attention_output_dim != config.moe_expert_input_dim:
      raise ValueError(
          "attention_output_dim must be positive and equal to moe_expert_input_dim for Qwen3CustomMoeDecoderLayer."
      )

    self.latent_norm = RMSNorm(
        num_features=config.attention_output_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.moe_block = moe.RoutedMoE(
        config=self.config,
        num_experts=self.config.num_experts,
        num_experts_per_tok=self.config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=max_initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=self.config.moe_mlp_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

    if self.config.attention_output_dim > 0 and self.config.attention_output_dim != self.config.emb_dim:
      out_kernel_axis = (None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("mlp", "embed")
      self.layer_up_projection = DenseGeneral(
          in_features_shape=self.config.attention_output_dim,
          out_features_shape=self.config.emb_dim,
          axis=-1,
          kernel_init=max_initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
          kernel_axes=out_kernel_axis,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          quant=quant,
          shard_mode=self.config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          use_bias=False,
          rngs=rngs,
      )
    else:
      self.layer_up_projection = None

    self.out_sharding = create_sharding(self.mesh, self.activation_axis_names)

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
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)
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
    return inputs, attention_lnx, kv_cache

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
    """Applies the Qwen3CustomMoeDecoderLayer to the inputs.

    Args:
      inputs: Input tensor to the decoder layer.
      decoder_segment_ids: Optional segment IDs for packed sequences.
      decoder_positions: Optional positional information for each token.
      deterministic: Whether to run in deterministic mode (e.g., no dropout).
      model_mode: The current mode of the model (e.g., "train", "decode").
      previous_chunk: Ignored in this implementation.
      page_state: Optional PageState for paged attention.
      slot: Optional slot index for decoding.
      kv_cache: Optional KV cache for self-attention.
      attention_metadata: Optional metadata for attention.

    Returns:
      A tuple containing:
        - hidden_states: The output tensor of the decoder layer.
        - kv_cache: The updated KV cache.
    """
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs, attention_lnx, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    attention_lnx = self.latent_norm(attention_lnx)
    mlp_lnx, load_balance_loss, _ = self.moe_block(attention_lnx, out_sharding=self.out_sharding)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    layer_output = mlp_lnx
    if self.layer_up_projection is not None:
      layer_output = self.layer_up_projection(layer_output)
      layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    layer_output = inputs + layer_output

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


Qwen3CustomMoeDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3CustomMoeDecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
