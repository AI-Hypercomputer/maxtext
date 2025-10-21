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

from typing import Optional

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText.layers import initializers, linears, moe, nnx_wrappers, quantizations
from MaxText.layers import attention_mla
from MaxText.common_types import Config
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.linears import Dropout
from MaxText.inference import page_manager
from MaxText.common_types import MODEL_MODE_PREFILL
from MaxText import max_utils
# -----------------------------------------
# The Decoder Layer for DeepSeek v3
# -----------------------------------------


class DeepSeekGenericLayer(nnx.Module):
  """Generic DeepSeek layer with Multi-Head Latent Attention.

  This is to be used as a base class for DeepSeek layers with dense/sparse MLPs.
  This class follows a pattern of separating module creation from execution.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ) -> None:

    self.config = config
    self.model_mode = model_mode
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.self_attention = attention_mla.MLA(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=self.config.attention_type,
        inputs_q_shape=self.dummy_inputs_shape,
        inputs_kv_shape=self.dummy_inputs_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        q_lora_rank=self.config.q_lora_rank,
        kv_lora_rank=self.config.kv_lora_rank,
        qk_nope_head_dim=self.config.qk_nope_head_dim,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        v_head_dim=self.config.v_head_dim,
        max_position_embeddings=self.config.max_position_embeddings,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        mscale=self.config.mscale,
        rope_factor=self.config.rope_factor,
        model_mode=model_mode,
        rngs=rngs,
    )

    self.dropout = Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

  def __call__(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    x = self.with_logical_constraint(x)
    x = checkpoint_name(x, "decoder_layer_input")

    hidden_states, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )

    mlp_lnx = self.mlp_op(hidden_states, deterministic)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output)

  def mlp_op(self, x, deterministic):
    """Executes the MLP operation. To be implemented by subclasses."""
    raise NotImplementedError()

  def with_logical_constraint(self, x):
    return nn.with_logical_constraint(x, self.logical_axis_names)

  def dropout_op(self, x, deterministic):
    return self.with_logical_constraint(self.dropout(x, deterministic=deterministic))

  def pre_attention_norm_op(self, x):
    return self.with_logical_constraint(self.pre_self_attention_layer_norm(x))

  def post_attention_norm_op(self, x):
    return self.with_logical_constraint(self.post_self_attention_layer_norm(x))

  def attention_op(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    """Executes the attention layer."""
    return self.with_logical_constraint(
        self.self_attention(
            x,
            x,
            decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
            model_mode=self.model_mode,
            previous_chunk=previous_chunk,
            page_state=page_state,
            slot=slot,
        )
    )

  @property
  def logical_axis_names(self):
    if self.model_mode == MODEL_MODE_PREFILL:
      return (
          "activation_batch",
          "prefill_activation_norm_length",
          "activation_embed",
      )
    else:
      return (
          "activation_batch",
          "activation_norm_length",
          "activation_embed",
      )

  def post_process(self, layer_output):
    """postprocessing."""
    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    return layer_output

  def self_attention_with_norm_op(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    """self-attention with normalization"""
    lnx = self.pre_attention_norm_op(inputs)

    attention_lnx = self.attention_op(
        lnx,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )
    intermediate_inputs = inputs + attention_lnx
    # Normalization
    hidden_states = self.post_attention_norm_op(intermediate_inputs)
    return hidden_states, intermediate_inputs


class DeepSeekDenseLayer(DeepSeekGenericLayer):
  """DeepSeek-style dense layer with Multi-Head Latent Attention."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant)
    self.mlp = linears.MlpBlock(
        in_features=self.dummy_inputs_shape[-1],
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=quant,
        model_mode=model_mode,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic):
    return self.with_logical_constraint(self.mlp(x, deterministic))


DeepSeekDenseLayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekMoELayer(DeepSeekGenericLayer):
  """DeepSeek-style MoE layer with Multi-Head Latent Attention.
  Supports dropless and dropping base on configs.
  Uses a bias in routing instead of load balancing loss.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ) -> None:

    super().__init__(config, model_mode, mesh, rngs, quant)
    self.DeepSeekMoeBlock_0 = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=self.rngs,
    )

  def mlp_op(self, x, _):
    return self.with_logical_constraint(self.DeepSeekMoeBlock_0(x))


DeepSeekMoELayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
