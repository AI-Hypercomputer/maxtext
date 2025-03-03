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

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module


from typing import Optional
from layers import quantizations
from layers import linears
from layers import initializers
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
from flax import linen as nn
import jax.numpy as jnp
from layers import attentions
from layers import embeddings
from layers import normalizations
from layers import models
import common_types
import max_logging

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization

# -----------------------------------------
# The Decoder Layer for DeepSeek v3
# -----------------------------------------
# Please note: DeepSeek V3 is not fully support at this moment


def self_attention_with_norm(inputs, cfg, mesh, quant, decoder_segment_ids, decoder_positions, deterministic, model_mode):
  # Normalization
  lnx_rms = models.RMSNorm(
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      name="pre_self_attention_layer_norm",
      kernel_axes=("norm",),
      epsilon=cfg.normalization_layer_epsilon,
  )
  lnx = lnx_rms(inputs)
  lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

  attention_layer = attentions.MLA(
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
      quant=quant,
      kv_quant=quantizations.configure_kv_quant(cfg),
      q_lora_rank=cfg.q_lora_rank,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      v_head_dim=cfg.v_head_dim,
      max_seq_len=cfg.max_target_length,
      original_seq_len=cfg.original_seq_len,
      mscale=cfg.mscale,
      rope_factor=cfg.rope_factor,
  )

  attention_lnx = attention_layer(
      lnx,
      lnx,
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=deterministic,
      model_mode=model_mode,
  )

  attention_lnx = nn.with_logical_constraint(
      attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
  )
  intermediate_inputs = inputs + attention_lnx

  # Normalization
  hidden_states = models.RMSNorm(
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      name="post_self_attention_layer_norm",
      kernel_axes=("norm",),
      epsilon=cfg.normalization_layer_epsilon,
  )(intermediate_inputs)
  hidden_states = nn.with_logical_constraint(
      hidden_states, ("activation_batch", "activation_norm_length", "activation_embed")
  )
  return hidden_states, intermediate_inputs


def post_process(cfg, layer_output, sow):
  if cfg.record_internal_nn_metrics:
    sow("intermediates", "activation_mean", jnp.mean(layer_output))
    sow("intermediates", "activation_stdev", jnp.std(layer_output))
    sow(
        "intermediates",
        "activation_fraction_zero",
        jnp.sum(layer_output == 0) / jnp.size(layer_output),
    )

  if cfg.scan_layers:
    return layer_output, None
  else:
    return layer_output


class DeepSeekDenseLayer(nn.Module):
  """DeepSeek-style dense layer with Multi-Head Latent Attention."""

  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
  ):
    cfg = self.config
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    hidden_states, intermediate_inputs = self_attention_with_norm(
        inputs, cfg, self.mesh, self.quant, decoder_segment_ids, decoder_positions, deterministic, model_mode
    )
    mlp_lnx = linears.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )
    return post_process(cfg, layer_output, self.sow)


class DeepSeekMoELayer(nn.Module):
  """DeepSeek-style MoE layer with Multi-Head Latent Attention.
  Supports dropless and dropping base on configs.
  Uses a bias in routing instead of load balancing loss.
  """

  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
  ):
    cfg = self.config
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    hidden_states, intermediate_inputs = self_attention_with_norm(
        inputs, self.config, self.mesh, self.quant, decoder_segment_ids, decoder_positions, deterministic, model_mode
    )

    mlp_lnx = linears.DeepSeekMoeBlock(
        config=cfg,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
    )(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )
    return post_process(cfg, layer_output, self.sow)
