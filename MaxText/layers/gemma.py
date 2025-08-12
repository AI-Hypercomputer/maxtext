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

from typing import Optional

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from MaxText.common_types import Config
from MaxText.layers import initializers
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import MlpBlock
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant


# Decoder and Model definitions
class GemmaDecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: Optional[Quant] = None

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs

    inputs_shape = (
      self.config.per_device_batch_size,
      self.config.max_target_length,
      self.config.base_emb_dim,
    )

    self.pre_attention_norm = RMSNorm(
        num_features=inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.attention = Attention(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        mesh=self.mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(self.config),
        use_ragged_attention=self.config.use_ragged_attention,
        ragged_block_size=self.config.ragged_block_size,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

    self.mlp = MlpBlock(
        config=self.config,
        in_features=inputs_shape[-1],
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        use_pre_norm=True,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

    self.dropout = nnx.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_manager=None,
      page_state=None,
      slot=None,
  ):
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = self.pre_attention_norm(inputs)

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    attention_lnx = self.attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        page_state=page_state,
        slot=slot,
    )

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    attention_lnx += inputs
    residual = attention_lnx

    # MLP block with pre-norm.
    mlp_lnx = self.mlp(residual, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    next_layer_addition = mlp_lnx + residual

    next_layer_addition_dropped_out = self.dropout(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    if self.config.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output


GemmaDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    GemmaDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
