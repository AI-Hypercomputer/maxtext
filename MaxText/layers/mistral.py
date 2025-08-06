"""
Copyright 2024 Google LLC

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


from typing import Any

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText.layers import nnx_wrappers, initializers
from MaxText.layers.linears import MlpBlock
from MaxText.layers.models import Config
from MaxText.layers.attentions import Attention
from MaxText.layers import quantizations
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.normalizations import RMSNorm


# -----------------------------------------
# The Decoder Layer for Mistral
# -----------------------------------------


class MistralDecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Quant | None = None,
      rngs: nnx.Rngs | None = None,
      **kwargs: Any,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs if rngs else kwargs.get("rngs", nnx.Rngs(0))

    inputs_shape = (
      config.per_device_batch_size,
      config.max_target_length,
      config.base_emb_dim,
    )

    self.pre_attention_norm = RMSNorm(
        num_features=inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
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
        mesh=self.mesh,
        dtype=self.config.dtype,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(self.config),
        prefill_cache_axis_order=tuple(map(int, self.config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, self.config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, self.config.compute_axis_order.split(","))),
        rngs=self.rngs,
    )
    self.mlp = MlpBlock(
        in_features=inputs_shape[-1],
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        use_pre_norm=True,
        config=self.config,
        quant=self.quant,
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
      page_state=None,
      slot=None,
  ):

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
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
    )

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    intermediate_inputs = inputs + attention_lnx

    mlp_lnx = self.mlp(intermediate_inputs, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)

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


class MistralDecoderLayerWrapper(nn.Module):
  """A Linan wrapper for the NNX MistralDecoderLayer"""

  config: Config
  mesh: Mesh
  quant: Quant | None = None

  def setup(self):
    self.mistral_nnx_layer = nnx_wrappers.to_linen(
      MistralDecoderLayer,
      config=self.config,
      mesh=self.mesh,
      quant=self.quant,
      metadata_fn=initializers.variable_to_logically_partitioned,
    )

  def __call__(self, *args, **kwargs):
    """Call the underlying NNX layer"""
    return self.mistral_nnx_layer(*args, **kwargs)
