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

from maxtext.src.maxtext.common_types import Config
from maxtext.src.maxtext.layers import attentions
from maxtext.src.maxtext.layers import initializers
from maxtext.src.maxtext.layers import linears
from maxtext.src.maxtext.layers import moe
from maxtext.src.maxtext.layers import quantizations
from maxtext.src.maxtext.layers.normalizations import rms_norm
from maxtext.src.maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.src.maxtext.inference import page_manager

# -----------------------------------------
# Helper functions for Qwen3 layers
# -----------------------------------------


def self_attention_with_norm(
    inputs: jnp.ndarray,
    cfg: Config,
    mesh: Mesh,
    quant: None | Quant,
    decoder_segment_ids: None | jnp.ndarray,
    decoder_positions: None | jnp.ndarray,
    deterministic: bool,
    model_mode: str,
):
  """A helper function for self-attention block with normalization."""

  inputs_checkpoint = checkpoint_name(inputs, "decoder_layer_input")

  # Corresponds to Qwen3's `input_layernorm`
  lnx = rms_norm(
      num_features=inputs.shape[-1],
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      name="pre_self_attention_layer_norm",
      epsilon=cfg.normalization_layer_epsilon,
      kernel_axes=("norm",),
  )(inputs_checkpoint)
  lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

  # Self-attention block
  attention_layer = attentions.attention_as_linen(
      config=cfg,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      attention_kernel=cfg.attention,
      inputs_q_shape=lnx.shape,
      inputs_kv_shape=lnx.shape,
      mesh=mesh,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      dropout_rate=cfg.dropout_rate,
      name="self_attention",
      quant=quant,
      kv_quant=quantizations.configure_kv_quant(cfg),
      use_qk_norm=cfg.use_qk_norm,
      query_pre_attn_scalar=(cfg.head_dim**-0.5),  # Qwen3 specific scaling
      model_mode=model_mode,
  )

  attention_output = attention_layer(
      lnx,  # inputs_q
      lnx,  # inputs_kv
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=deterministic,
      model_mode=model_mode,
  )
  attention_output = nn.with_logical_constraint(
      attention_output, ("activation_batch", "activation_length", "activation_embed")
  )

  # Residual connection after attention
  residual_after_attention = inputs_checkpoint + attention_output

  # Post Attention LayerNorm (corresponds to Qwen3's `post_attention_layernorm`)
  hidden_states = rms_norm(
      num_features=residual_after_attention.shape[-1],
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      name="post_self_attention_layer_norm",
      epsilon=cfg.normalization_layer_epsilon,
      kernel_axes=("norm",),
  )(residual_after_attention)
  hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", "activation_length", "activation_embed"))

  return hidden_states, residual_after_attention


# -----------------------------------------
# The Dense Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3DecoderLayer(nn.Module):
  """Qwen3 Transformer decoder layer (dense)."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
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
    cfg = self.config

    hidden_states, residual_after_attention = self_attention_with_norm(
        inputs,
        cfg,
        self.mesh,
        self.quant,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )

    # Dense MLP block
    mlp_output = linears.mlp_block(
        in_features=hidden_states.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(hidden_states, deterministic=deterministic)

    # Final residual connection
    layer_output = residual_after_attention + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output


# -----------------------------------------
# The MoE Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3MoeDecoderLayer(nn.Module):
  """Qwen3 Transformer decoder layer (MoE)."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
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
    cfg = self.config

    hidden_states, residual_after_attention = self_attention_with_norm(
        inputs,
        cfg,
        self.mesh,
        self.quant,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )

    # Mixture of Experts block
    mlp_output, load_balance_loss = moe.get_routed_moe(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="moe_block",
        quant=self.quant,
    )(hidden_states)

    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    mlp_output = nn.with_logical_constraint(mlp_output, ("activation_batch", "activation_length", "activation_embed"))

    # Final residual connection
    layer_output = residual_after_attention + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
