# Copyright 2025 Google LLC
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

from typing import Optional

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import nnx

from MaxText.common_types import Config, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText.layers import attentions
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import moe
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager

# -----------------------------------------
# Self-Attention Block for Qwen3
# -----------------------------------------


class Qwen3SelfAttentionWithNorm(nnx.Module):
  """A self-attention block with pre and post normalization."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Optional[Quant],
      model_mode: str,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = self.config

    if model_mode == MODEL_MODE_PREFILL:
      seq_len = cfg.max_prefill_predict_length
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      seq_len = 1
    else:
      seq_len = cfg.max_target_length

    dummy_inputs_shape = (cfg.micro_batch_size_to_train_on, seq_len, cfg.emb_dim)

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_layer_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.self_attention = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
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
        rngs=rngs,
    )
    self.post_self_attention_layer_norm = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="post_self_attention_layer_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray],
      decoder_positions: Optional[jnp.ndarray],
      deterministic: bool,
      model_mode: str,
      activation_axis_names: tuple[str, ...],
  ):
    """Helper function for self-attention block with normalization."""
    inputs_checkpoint = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_layer_norm(inputs_checkpoint)
    lnx = nnx.with_logical_constraint(lnx, activation_axis_names)

    attention_output = self.self_attention(
        lnx,  # inputs_q
        lnx,  # inputs_kv
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    attention_output = nnx.with_logical_constraint(attention_output, activation_axis_names)

    residual_after_attention = inputs_checkpoint + attention_output

    hidden_states = self.post_self_attention_layer_norm(residual_after_attention)
    hidden_states = nnx.with_logical_constraint(hidden_states, activation_axis_names)

    return hidden_states, residual_after_attention


# -----------------------------------------
# The Dense Decoder Layer for Qwen3
# -----------------------------------------


class Qwen3DecoderLayer(nnx.Module):
  """Qwen3 Transformer decoder layer (dense)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[Quant] = None,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    cfg = self.config

    self.attention = Qwen3SelfAttentionWithNorm(
        config=cfg,
        mesh=self.mesh,
        quant=self.quant,
        model_mode=self.model_mode,
        rngs=rngs,
    )

    self.mlp = linears.MlpBlock(
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
        model_mode=model_mode,
        rngs=rngs,
    )

    if self.model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_length", "activation_embed")

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray],
      decoder_positions: Optional[jnp.ndarray],
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: Optional[page_manager.PageState] = None,
      slot: Optional[int] = None,
  ):
    cfg = self.config

    hidden_states, residual_after_attention = self.attention(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        self.activation_axis_names,
    )

    mlp_output = self.mlp(hidden_states, deterministic=deterministic)

    layer_output = residual_after_attention + mlp_output
    layer_output = nnx.with_logical_constraint(layer_output, self.activation_axis_names)

    if cfg.scan_layers:
      return layer_output, None
    return layer_output


# -----------------------------------------
# The MoE Decoder Layer for Qwen3
# -----------------------------------------


class Qwen3MoeDecoderLayer(nnx.Module):
  """Qwen3 Transformer decoder layer (MoE)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[Quant] = None,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    cfg = self.config

    self.attention = Qwen3SelfAttentionWithNorm(
        config=cfg,
        mesh=self.mesh,
        quant=self.quant,
        model_mode=self.model_mode,
        rngs=rngs,
    )

    self.moe_block = moe.MoeBlock(
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
        rngs=rngs,
    )

    if self.model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_length", "activation_embed")

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray],
      decoder_positions: Optional[jnp.ndarray],
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: Optional[page_manager.PageState] = None,
      slot: Optional[int] = None,
  ):
    cfg = self.config

    hidden_states, residual_after_attention = self.attention(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        self.activation_axis_names,
    )

    mlp_output, load_balance_loss = self.moe_block(hidden_states)

    if load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    mlp_output = nnx.with_logical_constraint(mlp_output, self.activation_axis_names)

    # Final residual connection
    layer_output = residual_after_attention + mlp_output
    layer_output = nnx.with_logical_constraint(layer_output, self.activation_axis_names)

    if cfg.scan_layers:
      return layer_output, None
    return layer_output


# Linen wrappers for backward compatibility
Qwen3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
Qwen3MoeDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3MoeDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
