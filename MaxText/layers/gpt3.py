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

from jax.sharding import Mesh



from flax import linen as nn


import jax.numpy as jnp
# from jax.experimental.pallas.ops.tpu import flash_attention
from layers import attentions
from layers import linears
from layers import normalizations

from layers import models

import common_types

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh

Gpt3MultiHeadAttention = attentions.Gpt3MultiHeadAttention
LayerNorm = normalizations.LayerNorm


#-----------------------------------------
# The Decoder Layer specific for GPT3
#-----------------------------------------


class Gpt3DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: models.Config
  mesh: Mesh

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               padding_mask,
               deterministic,
               model_mode,
               ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed'))


    lnx_layer_norm = models.LayerNorm(
        dtype=cfg.dtype,
        name='pre_self_attention_layer_norm',
        kernel_axes=('embed',),
        epsilon=cfg.norm_epsilon,
        reductions_in_fp32=False,
        use_bias=True,
        )
    lnx = lnx_layer_norm(inputs)

    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    # Self-attention block
    attention_layer = Gpt3MultiHeadAttention(
      num_heads=cfg.num_heads,
      dtype=cfg.dtype,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      attention_kernel=cfg.attention,
      mesh=mesh,
      dropout_rate=cfg.dropout_rate,
      name='self_attention',
      fused_qkv=cfg.fused_qkv,
      use_bias=True,
      use_int8=cfg.int8_training)

    attention_lnx = attention_layer(
            lnx,
            decoder_segment_ids=decoder_segment_ids,
            model_mode=model_mode,
            deterministic=deterministic)

    attention_lnx = nn.with_logical_constraint(
        attention_lnx,
        ('activation_batch', 'activation_length', 'activation_embed'))
    attention_lnx += inputs

    # MLP block.
    mlp_lnx = linears.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
        use_bias=True,
        use_pre_norm=True,
        apply_padding_mask=True,
        add_skip_connection=True,
        config=cfg,
    )(attention_lnx, padding_mask=padding_mask, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(
        mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
    )

    layer_output = mlp_lnx

    layer_output = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ('activation_batch', 'activation_length', 'activation_embed'),
    )

    if cfg.record_internal_nn_metrics:
      self.sow('intermediates', 'activation_mean', jnp.mean(layer_output))
      self.sow('intermediates', 'activation_stdev', jnp.std(layer_output))
      self.sow(
          'intermediates',
          'activation_fraction_zero',
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
