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

from flax import linen as nn
from MaxText import common_types
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
import jax

from MaxText.layers import normalizations
from MaxText.layers import attentions
from MaxText.layers import initializers
from MaxText.layers import embeddings
from MaxText.layers import linears
from MaxText.layers import quantizations

from typing import Optional

from MaxText import max_logging

Embed = embeddings.Embed
RMSNorm = normalizations.RMSNorm
NdInitializer = initializers.NdInitializer
Attention = attentions.Attention
AttentionType = attentions.AttentionType
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


nd_dense_init = initializers.nd_dense_init
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant


GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  layer_id %= len(GEMMA3_ATTENTION_PATTERN)
  return GEMMA3_ATTENTION_PATTERN[layer_id]


def get_query_pre_attn_scalar(config) -> float:
  """Returns the scalar to multiply the query by before attention."""
  if config.model_name in ["gemma3-4b", "gemma3-12b"]:
    return config.head_dim**-0.5
  elif config.model_name == "gemma3-27b":
    return (config.base_emb_dim // config.base_num_query_heads) ** -0.5
  else:
    raise ValueError(f"Unsupported model name: {config.model_name}")


class Gemma3VisionEncoderLayer(nn.Module):
  config: Config

  @nn.compact
  def __call__(self, inputs, train=False):
    """ViT model that transforms image inputs to image embeddings.
    Args:
      inputs: jnp.array shaped [B, N, H, W, C], e.g. [4, 1, 896, 896, 3]
    Returns:
      jnp.array for image embeddings, shaped [B, N, P, D], e.g. [4, 1, 256, 2560]
    """
    cfg = self.config
    print(f"Gemma3VisionEncoderLayer input: {inputs.shape}")
    x = inputs.squeeze(axis=1)
    x = nn.Conv(features=1152,
        kernel_size=(14, 14),
        strides=14,
        padding="VALID",
        name="embedding")(x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])
    # TODO(hengtaoguo): finish the ViT with posemb, dropout and transformation layers.
    # Currently it is only a placeholder with one Conv layer.
    # Placeholder x shape (B, 4096, 1152).
    return x


# Gemma3 Decoder Layer
class Gemma3DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
  mesh: Mesh
  quant: Optional[Quant] = None
  attention_type: AttentionType = AttentionType.LOCAL_SLIDING

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
  ):
    cfg = self.config
    mesh = self.mesh
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="pre_self_attention_norm", kernel_axes=("norm",))(
        inputs
    )

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))
    query_pre_attn_scalar = get_query_pre_attn_scalar(cfg)

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
        name="self_attention",
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        attention_type=self.attention_type,
        sliding_window_size=cfg.sliding_window_size,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
        use_qk_norm=True,  # Gemma 3 models use query, key normalizations
        query_pre_attn_scalar=query_pre_attn_scalar,
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
      attention_lnx = RMSNorm(
          dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="post_self_attention_norm", kernel_axes=("norm",)
      )(attention_lnx)

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    attention_lnx += inputs
    residual = attention_lnx

    attn_output = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="pre_ffw_norm", kernel_axes=("norm",))(
        attention_lnx
    )

    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(attn_output, deterministic=deterministic)

    if cfg.use_post_ffw_norm:
      mlp_lnx = RMSNorm(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, name="post_ffw_norm", kernel_axes=("norm",))(mlp_lnx)

    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))
    next_layer_addition = mlp_lnx + residual
    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
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
