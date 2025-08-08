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

from typing import Any, Optional

import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
# from jax.experimental.pallas.ops.tpu import flash_attention

from flax import linen as nn
from flax import nnx

from MaxText.inference import page_manager
from MaxText.common_types import Config
from MaxText.layers.linears import MlpBlock
from MaxText.layers import initializers
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.layers.attentions import Attention
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.normalizations import RMSNorm
from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE


# -----------------------------------------
# The Decoder Layer specific for Llama2
# -----------------------------------------


class LlamaDecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      quant: Optional[Quant] = None,
      rngs: Optional[nnx.Rngs] = None,
  ):

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs

    dummy_inputs_shape = (int(config.global_batch_size_to_load), config.max_target_length, int(config.emb_dim))
    #dummy_inputs_shape = (int(config.per_device_batch_size), config.max_target_length, int(config.emb_dim))

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

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
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        prefill_cache_axis_order=tuple(map(int, config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, config.compute_axis_order.split(","))),
        reshape_q=config.reshape_q,
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        #model_mode=model_mode,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        #model_mode=MODEL_MODE_TRAIN, # Only supporting training here
        #model_mode=MODEL_MODE_PREFILL,  # This is just for initialization
        rngs=self.rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        quant=self.quant,
        model_mode=model_mode,
        #model_mode=MODEL_MODE_TRAIN, # Only supporting training here
        #model_mode=MODEL_MODE_PREFILL, # This is just for initialization
        rngs=self.rngs,
    )

    self.dropout = nnx.Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)
  

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

    if model_mode == MODEL_MODE_PREFILL:
      activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    inputs = nn.with_logical_constraint(inputs, activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx = self.pre_self_attention_layer_norm(inputs)

    lnx = nn.with_logical_constraint(lnx, activation_axis_names)

    attention_lnx = self.self_attention(
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

    attention_lnx = nn.with_logical_constraint(attention_lnx, activation_axis_names)
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, activation_axis_names)

    # MLP block.
    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, activation_axis_names)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(layer_output, activation_axis_names)

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

LlamaDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    LlamaDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
