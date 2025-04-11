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

"""Llama4 decoder layer definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from flax import linen as nn
from jax.sharding import Mesh
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from MaxText.layers import attentions
from MaxText.layers import embeddings
from MaxText.layers import initializers
from MaxText.layers import moe
from MaxText.layers import normalizations
from MaxText.layers import models
from MaxText.layers import quantizations

from MaxText import common_types
from MaxText.inference import page_manager
from typing import Optional

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM


Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization


def determine_is_nope_layer(layer_id: int, nope_layer_interval: int) -> bool:
  """
  Determines whether the given layer at `layer_id` should use RoPE or not (NoPE).

  Args:
    layer_id: The index of the layer.
    nope_layer_interval: The interval at which layers should use NoPE.

  Returns:
    True if the layer should use NoPE, False otherwise.
  """
  return nope_layer_interval is not None and nope_layer_interval > 0 and (layer_id + 1) % nope_layer_interval == 0


# -----------------------------------------
# The Decoder Layer specific for LLama4
# -----------------------------------------


class Llama4DecoderLayer(nn.Module):
  """Transformer decoder layer for Llama4.

  Attributes:
    config: models.Config, MaxText model config
    mesh: Mesh, JAX device mesh (used for sharding)
    quant: Optional[Quant], quantization config
    is_nope_layer: bool, whether to use RoPE or not on this layer
  """

  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None
  is_nope_layer: bool = False

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      previous_chunk=None,
  ):
    cfg = self.config
    mesh = self.mesh

    assert (
        cfg.num_experts > 1
    ), "Expected the Llama4 config to have num_experts > 1.  Note that MaxText only supports Llama4 Scout (MoE-only) for now!"

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx_rms = models.RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_layer_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )
    lnx = lnx_rms(inputs)

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))
    # Instead of scaling the query values in the checkpoint conversion (`llama_or_mistral_ckpt`)
    # we'll do it dynamically in the forward pass of Attention
    query_pre_attn_scalar = cfg.head_dim**-0.5

    # Self-attention block
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
        prefill_cache_axis_order=tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
        ar_cache_axis_order=tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
        compute_axis_order=tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
        reshape_q=cfg.reshape_q,
        use_ragged_attention=cfg.use_ragged_attention,
        ragged_block_size=cfg.ragged_block_size,
        is_nope_layer=self.is_nope_layer,
        use_qk_norm=cfg.use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        slot=slot,
        page_state=page_state,
        previous_chunk=previous_chunk,
    )

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
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

    load_balance_loss = None
    mlp_lnx = moe.DeepSeekMoeBlock(
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

    # NOTE: this is only needed for dropping MoE
    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

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
