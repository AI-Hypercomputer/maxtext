# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText.layers import nnx_wrappers, initializers
from MaxText.layers.linears import Dropout, MlpBlock
from MaxText.layers.models import Config
from MaxText.layers.attentions import Attention
from MaxText.layers import quantizations
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.normalizations import RMSNorm
from maxtext.utils import max_utils


# -----------------------------------------
# The Decoder Layer for Mistral
# -----------------------------------------


class MistralDecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

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
        model_mode=model_mode,
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
        mesh=self.mesh,
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        quant=self.quant,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      page_state: None | int = None,
      slot: None | int = None,
      previous_chunk=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config

    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        slot=slot,
        page_state=page_state,
        previous_chunk=previous_chunk,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # MLP block.
    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

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
      return layer_output, kv_cache


MistralDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    MistralDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
