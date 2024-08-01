"""
Copyright 2023 Google LLC

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

from flax import linen as nn
import common_types
import jax.numpy as jnp

from layers import normalizations
from layers import attentions
from layers import initializers
from layers import embeddings
from layers import linears
from layers import quantizations

from typing import Optional

Embed = embeddings.Embed
RMSNorm = normalizations.RMSNorm
NdInitializer = initializers.NdInitializer
Attention = attentions.Attention
MlpBlock = linears.MlpBlock
Config = common_types.Config
AxisNames = common_types.AxisNames
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn
DType = common_types.DType
Array = common_types.Array
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


nd_dense_init = initializers.nd_dense_init
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant


# Decoder and Model definitions
class Gemma2DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
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
    mesh = self.mesh
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="pre_self_attention_norm_local", kernel_axes=("norm",))(
        inputs
    )

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    attention_layer = Attention(
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
        name="self_attention_local",
        float32_qk_product=True,
        float32_logits=True,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        attention_type=attentions.AttentionType.LOCAL_SLIDING,
        sliding_window_size=cfg.sliding_window_size
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    if cfg.use_post_attn_norm:
        attention_lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="post_self_attention_norm_local", kernel_axes=("norm",))(
            attention_lnx
    )
    
    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))
    attention_lnx += inputs
    residual = attention_lnx
    
    attn_output = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="pre_ffw_norm_local", kernel_axes=("norm",))(
        attention_lnx
    )

    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp_local",
        config=cfg,
        quant=self.quant,
    )(attn_output, deterministic=deterministic)

    if cfg.use_post_ffw_norm:
        mlp_lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="post_ffw_norm_local", kernel_axes=("norm",))(
        mlp_lnx
    )
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    next_layer_addition = mlp_lnx + residual

    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )
    
    ### global part
    inputs = nn.with_logical_constraint(layer_output, ("activation_batch", "activation_length", "activation_embed"))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="pre_self_attention_norm_global", kernel_axes=("norm",))(
        inputs
    )

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    attention_layer = Attention(
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
        name="self_attention_global",
        float32_qk_product=True,
        float32_logits=True,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        attention_type=attentions.AttentionType.GLOBAL,
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    if cfg.use_post_attn_norm:
        attention_lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="post_self_attention_norm_global", kernel_axes=("norm",))(
            attention_lnx
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))
    attention_lnx += inputs
    residual = attention_lnx
    
    attn_output = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="pre_ffw_norm_global", kernel_axes=("norm",))(
        attention_lnx
    )

    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp_global",
        config=cfg,
        quant=self.quant,
    )(attn_output, deterministic=deterministic)
    if cfg.use_post_ffw_norm:
        mlp_lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="post_ffw_norm_global", kernel_axes=("norm",))(
        mlp_lnx
    )
        
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    next_layer_addition = mlp_lnx + residual

    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
