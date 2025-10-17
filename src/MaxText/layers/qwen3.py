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

"""Qwen3 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText import max_utils
from MaxText.common_types import Config
from MaxText.layers import initializers
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import MlpBlock
from MaxText.layers.moe import RoutedMoE
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager


# -----------------------------------------
# The Base Decoder Layer for Qwen3
# -----------------------------------------
class AttentionWithNorm(nnx.Module):
  """Base class with shared common components: self-attention block with normalization."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    # Corresponds to Qwen3's `input_layernorm`
    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Self-attention block
    query_pre_attn_scalar = config.head_dim**-0.5  # Qwen3 specific scaling
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
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        use_qk_norm=config.use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode,
        rngs=rngs,
    )

    # Post Attention LayerNorm (corresponds to Qwen3's `post_attention_layernorm`)
    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

  def apply_attention_with_norm(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
  ):
    """Applies self-attention with pre and post-layer normalization."""
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # Pre attention norm
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)
    # Self attention
    attention_lnx = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    # Residual connection after attention
    intermediate_inputs = inputs + attention_lnx
    # Post attention norm
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)
    return hidden_states, intermediate_inputs


# -----------------------------------------
# The Dense Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3DecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (dense)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    hidden_states, intermediate_inputs = self.apply_attention_with_norm(
        inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode
    )

    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output


# -----------------------------------------
# The MoE Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3MoeDecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (MoE)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.moe_block = RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=config.moe_mlp_dim,  # same as config.mlp_dim
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    hidden_states, intermediate_inputs = self.apply_attention_with_norm(
        inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode
    )

    mlp_lnx, load_balance_loss = self.moe_block(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)
    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    else:
      return layer_output


Qwen3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

Qwen3MoeDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3MoeDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
