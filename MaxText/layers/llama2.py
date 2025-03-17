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

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from flax import linen as nn
from jax.sharding import Mesh
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations
from layers import models
from layers import quantizations

import common_types
from inference.page_manager import PageState
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

# -----------------------------------------
# The Decoder Layer specific for Llama2
# -----------------------------------------


class LlamaDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

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
      page_state: Optional[PageState] = None,
      layer_idx: Optional[int] = None,
      slot: Optional[int] = None,
      true_length: Optional[int] = None,
  ):
    """Llama decoder layer forward pass."""
    cfg = self.config
    mesh = self.mesh

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
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        page_state=page_state,
        layer_idx=layer_idx,  # Pass explicit layer index
        slot=slot,  # Pass slot
        true_length=true_length,
    )

    # Conditional processing based on attention type and mode
    if self.config.attention == "paged" and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      # Unpack the results from Attention.__call__
      q_input, k_pages, v_pages, lengths, page_indices, _ = attention_lnx

      # The attention calculation is now done *HERE*, inside the layer.
      from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention as jax_paged_attention
      from jax.experimental import shard_map
      from jax.sharding import PartitionSpec as P
      import jax

      # Define sharding specs (same as before)
      q_pspec = P(None, None, None)
      k_pspec = P(None, None, None, None)
      v_pspec = P(None, None, None, None)
      lengths_pspec = P(None)
      page_indices_pspec = P(None, None)

      # Define attention function (same as before)
      def attention_fn(q, k, v, lens, indices):
        return jax_paged_attention(
            q=q,
            k_pages=k,
            v_pages=v,
            lengths=lens,
            page_indices=indices,
            mask_value=-1e7,  # Use a static value
            attn_logits_soft_cap=self.config.attn_logits_soft_cap,
            pages_per_compute_block=self.config.pagedattn_pages_per_compute_block,
        )

      # Create sharded function
      wrapped_attention = shard_map.shard_map(
          attention_fn,
          mesh=self.mesh,
          in_specs=(q_pspec, k_pspec, v_pspec, lengths_pspec, page_indices_pspec),
          out_specs=q_pspec,  # Output sharding
          check_rep=False,
      )

      # Call attention computation, INSIDE the layer, inside mesh context.
      with self.mesh:
        attention_lnx = wrapped_attention(q_input, k_pages, v_pages, lengths, page_indices)
        attention_lnx = jnp.expand_dims(attention_lnx, axis=1)  # Add back the sequence dimension

      # Apply output projection *here*, after the attention calculation.
      attention_lnx = attention_layer.out_projection(cfg.emb_dim, attention_lnx)

    # The rest of the layer is unchanged
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

    # MLP block.
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