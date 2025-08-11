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

"""Decoder layer definition for GPT OSS models."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module


from typing import Optional

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText.layers import initializers, nnx_wrappers, models, moe, quantizations
from MaxText.layers.models import Config
from MaxText.layers.attentions import Attention
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.initializers import variable_to_logically_partitioned


# -----------------------------------------
# The Decoder Layer for GPT OSS models
# -----------------------------------------


class GptOssDecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(self, config: models.Config, mesh: Mesh, model_mode: str, rngs: nnx.Rngs, quant: Optional[Quant] = None):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.quant = quant
    self.activaction_shape = (self.config.per_device_batch_size, self.config.max_target_length, self.config.emb_dim)

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    self.self_attention = Attention(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        inputs_q_shape=self.activaction_shape,
        inputs_kv_shape=self.activaction_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(self.config),
        prefill_cache_axis_order=tuple(map(int, self.config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, self.config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, self.config.compute_axis_order.split(","))),
        model_mode=model_mode,
        rngs=self.rngs,
    )

    self.mlp = moe.RoutedMoE(
        config=self.config,
        num_experts=self.config.num_experts,
        num_experts_per_tok=self.config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=self.config.mlp_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
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

    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    attention_lnx = self.self_attention(
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

    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(
        hidden_states, ("activation_batch", "activation_norm_length", "activation_embed")
    )

    load_balance_loss = None
    mlp_lnx, load_balance_loss = self.mlp(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

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


class GptOssDecoderLayerWrapper(nn.Module):
  """Creates a GptOssDecoderLayer Linen module."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: Quant | None = None

  def setup(self):
    self.gpt_oss = nnx_wrappers.to_linen(
        GptOssDecoderLayer,
        config=self.config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        quant=self.quant,
        metadata_fn=initializers.variable_to_logically_partitioned,
        abstract_init=False,
    )

  def __call__(self, *args, **kwargs):
    return self.gpt_oss(*args, **kwargs)
