"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Qwen3 model decoder layer."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Optional

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn

from MaxText.common_types import Config
from MaxText.layers import attentions
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import moe
from MaxText.layers import quantizations
from MaxText.layers.normalizations import rms_norm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager


class Qwen3DecoderLayer(nn.Module):
  """Qwen3 Transformer decoder layer."""

  config: Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray],
      decoder_positions: Optional[jnp.ndarray],
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: Optional[page_manager.PageState] = None,
      slot: Optional[int] = None,
  ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
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
    attention_layer = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        use_qk_norm=cfg.use_qk_norm,
        query_pre_attn_scalar=(cfg.head_dim**-0.5),  # Qwen3 specific scaling
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
    mlp_input = rms_norm(
        num_features=residual_after_attention.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="post_self_attention_layer_norm",  # Standard MaxText naming
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(residual_after_attention)
    mlp_input = nn.with_logical_constraint(mlp_input, ("activation_batch", "activation_length", "activation_embed"))

    # MLP block
    if cfg.num_experts is None or cfg.num_experts <= 1:  # Dense MLP
      mlp_output = linears.mlp_block(
          in_features=mlp_input.shape[-1],
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="mlp",
          config=cfg,
          quant=self.quant,
      )(mlp_input, deterministic=deterministic)
    else:  # Mixture of Experts MLP -- not supported / tested in MaxText
      mlp_output, _ = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=self.mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          intermediate_dim=cfg.mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="moe_block",
          quant=self.quant,
      )(mlp_input)

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
