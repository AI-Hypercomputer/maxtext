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

import jax
from jax.sharding import Mesh



from flax import linen as nn


import jax.numpy as jnp
# from jax.experimental.pallas.ops.tpu import flash_attention
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations

from layers import models

from transformer_engine.common.recipe import Format
from transformer_engine.jax.flax.transformer import TransformerLayerType, TransformerLayer
import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax

import common_types

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm


#-----------------------------------------
# The Decoder Layer specific for TE
#-----------------------------------------

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

class TEDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  def generate_attention_mask(
      self,
      seq_len,
      decoder_segment_ids: Array | None,
      model_mode: str
  ) -> Array | None:
    
    mask = None
    if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      mask = decoder_segment_ids[:, None, None, None, :] == common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    elif decoder_segment_ids is not None:
      mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      mask = mask[:, None, None,:, :]

    causal_mask = None
    # We enforce causality except for AUTOREGRESSION
    if model_mode != common_types.MODEL_MODE_AUTOREGRESSIVE:
      q_seq_len = seq_len
      kv_seq_len = seq_len

      mask_shape = (q_seq_len, kv_seq_len)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      causal_mask = (col_ids <= row_ids)[None, None, None, :, :]

    if (mask is not None) and (causal_mask is not None):
      output_mask = jnp.logical_and(mask, causal_mask)
    elif mask is not None:
      output_mask = mask
    elif causal_mask is not None:
      output_mask = causal_mask
    else:
      output_mask = None
    return jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE) if output_mask is not None else None

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
               ):
    
    cfg = self.config
    mesh = self.mesh
   
    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed'))

    batch, seq, hidden = inputs.shape
    layer_output = TransformerLayer(
                hidden_size = cfg.base_emb_dim,
                mlp_hidden_size = cfg.base_mlp_dim,
                num_attention_heads = cfg.base_num_query_heads,
                hidden_dropout = 0.0,
                attention_dropout = 0.0,
                mlp_activations = cfg.mlp_activations,
                layer_type = TransformerLayerType.ENCODER,
                relative_embedding = False,
                scale_attn_logits = True,
                layernorm_type = 'rmsnorm',
                dtype = cfg.dtype,
                enable_rotary_pos_emb = True,
                enable_relative_embedding = False)(
                    inputs=inputs,
                    attention_mask=jnp.zeros((batch, 1, seq, seq)),
                    deterministic=deterministic
                )

    layer_output = nn.with_logical_constraint(
            layer_output,
            ('activation_batch', 'activation_length', 'activation_embed')
    )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
