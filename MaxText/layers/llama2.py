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
        layer_idx=layer_idx,
        slot=slot,
        true_length=true_length,
    )

    # Special processing for paged attention in autoregressive mode
    # Defensive handling of attention_lnx based on its structure
    if (
        self.config.attention == "paged" 
        and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE
        and isinstance(attention_lnx, tuple)
    ):
        # Check if the tuple has the expected length
        if len(attention_lnx) >= 7:
            # Standard case - we received the expected 7-tuple
            cache, q_input, k_pages, v_pages, lengths, page_indices, pages_used = attention_lnx
            
            # Import necessary libraries
            from jax.experimental import shard_map
            from jax.sharding import PartitionSpec as P
            import jax

            # Define sharding specs
            q_pspec = P(None, None, None)  # [batch, heads, dim]
            k_pspec = P(None, None, None, None)  # [kv_heads, num_pages, page_size, head_dim]
            v_pspec = P(None, None, None, None)  # [kv_heads, num_pages, page_size, head_dim]
            lengths_pspec = P(None)  # [batch]
            page_indices_pspec = P(None, None)  # [batch, max_pages]
            pages_used_pspec = P(None)  # [batch]

            # Use our custom paged attention implementation
            def attention_fn(q, k, v, lens, indices, used):
                # Import our custom implementation
                from inference.paged_attention import fixed_paged_attention
                
                # Use simplified implementation during initialization
                if self.is_initializing():
                    return jnp.zeros_like(q)
                
                return fixed_paged_attention(
                    query=q,
                    key_pages=k,
                    value_pages=v,
                    page_indices=indices,
                    pages_used=used,
                    lengths=lens,
                    attn_logits_soft_cap=self.config.attn_logits_soft_cap,
                )

            # Create sharded function with updated specs
            wrapped_attention = shard_map.shard_map(
                attention_fn,
                mesh=self.mesh,
                in_specs=(q_pspec, k_pspec, v_pspec, lengths_pspec, page_indices_pspec, pages_used_pspec),
                out_specs=q_pspec,
                check_rep=False,
            )

            # Call attention computation
            with self.mesh:
                # During initialization, just return zeros
                if self.is_initializing():
                    attention_output = jnp.zeros_like(q_input)
                else:
                    jax.debug.print("Attention inputs - shape: q={}, k={}, v={}, len={}, idx={}, used={}",
                                  q_input.shape, k_pages.shape, v_pages.shape, 
                                  lengths.shape, page_indices.shape, pages_used.shape)
                    jax.debug.print("Sample values - q[0,0]={}; k[0,0,0]={}; v[0,0,0]={}; len[0]={}; idx[0,0]={}; used[0]={}",
                                  q_input[0,0], k_pages[0,0,0], v_pages[0,0,0],
                                  lengths[0], page_indices[0,0], pages_used[0])

                    attention_output = wrapped_attention(q_input, k_pages, v_pages, lengths, page_indices, pages_used)
                
                # Add back sequence dimension
                attention_lnx = jnp.expand_dims(attention_output, axis=1)

            # Apply output projection
            attention_lnx = attention_layer.out_projection(cfg.emb_dim, attention_lnx)
        
        elif len(attention_lnx) == 1 and "cache" in attention_lnx:
            # Handle case where only cache is returned (during initialization)
            # Just return zeros for now
            attention_lnx = jnp.zeros_like(inputs)

    # Continue with the rest of the layer processing
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

    # MLP block
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