# Copyright 2023–2026 Google LLC
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

"""Specialized layers for Gemma 4."""

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx
from typing import Optional

from maxtext.common.common_types import Config, AttentionType, MODEL_MODE_PREFILL
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import MlpBlock

import jax.sharding
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils


GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  layer_id %= len(GEMMA4_ATTENTION_PATTERN)
  return GEMMA4_ATTENTION_PATTERN[layer_id]


class Gemma4MoE(nnx.Module):
  """Gemma4 specific MoE block containing layer norms and a generic MoE block."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.quant = quant

    self.moe_block = moe.RoutedAndSharedMoE(
        config=config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        quant=self.quant,
        rngs=self.rngs,
    )

    self.pre_forward_scale_2 = nnx.Param(
        jnp.ones((self.config.emb_dim,), dtype=self.config.weight_dtype),
        sharding=("embed",),
    )
    self.pre_feedforward_layernorm_2 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_1 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_2 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.gate_norm = RMSNorm(
        num_features=self.config.emb_dim,
        epsilon=self.config.normalization_layer_epsilon,
        dtype=jnp.float32 if self.config.float32_gate_logits else self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        with_scale=False,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs: jax.Array,
      original_inputs: jax.Array | None = None,
      intermediate_sharding: jax.sharding.NamedSharding | None = None,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    shared_experts = self.moe_block.shared_experts(
        inputs, intermediate_sharding=intermediate_sharding, out_sharding=out_sharding
    )
    shared_experts = self.post_feedforward_layernorm_1(shared_experts)

    # 1. Experts receive standard RMSNorm (with weight)
    routed_inputs = self.pre_feedforward_layernorm_2(original_inputs)

    # 2. Gate receives RMSNorm (without weight) * root_size * router_scale
    gate_dtype = jnp.float32 if self.config.float32_gate_logits else self.config.dtype
    unscaled_norm = self.gate_norm(original_inputs)

    root_size = self.config.emb_dim**-0.5
    router_scale = jnp.asarray(self.pre_forward_scale_2.value, gate_dtype)
    gate_inputs = unscaled_norm * root_size * router_scale

    # 3. Pass both to routed_moe
    routed_experts, load_balance_loss, moe_bias_updates = self.moe_block.routed_moe(
        routed_inputs, gate_inputs=gate_inputs, out_sharding=out_sharding
    )
    routed_experts = self.post_feedforward_layernorm_2(routed_experts)

    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


class Gemma4DecoderLayer(nnx.Module):
  """Transformer decoder layer for Gemma4."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      attention_type: AttentionType = AttentionType.LOCAL_SLIDING,
      layer_idx: int = 0,
  ):
    """Initializes the instance.

    Args:
      config: The Config object with model hyperparameters.
      mesh: The device mesh for distributed training.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: The random number generators for initialization.
      quant: The quantization configuration.
      attention_type: The type of attention to use.
      layer_idx: The index of the layer in the block.
    """

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.attention_type = attention_type
    self.layer_idx = layer_idx

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    query_pre_attn_scalar = 1.0
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    share_kv_projections = False

    if attention_type == AttentionType.GLOBAL:
      if hasattr(config, "global_num_kv_heads") and config.global_num_kv_heads:
        num_kv_heads = config.global_num_kv_heads
      if hasattr(config, "global_head_dim") and config.global_head_dim:
        head_dim = config.global_head_dim
      if getattr(config, "share_kv_projections", False):
        share_kv_projections = True

    if attention_type == AttentionType.GLOBAL:
      partial_rotary_factor = config.global_rope_proportion if hasattr(config, "global_rope_proportion") else 0.25
      max_timescale = (
          config.global_rope_max_timescale
          if hasattr(config, "global_rope_max_timescale") and config.global_rope_max_timescale > 0
          else config.rope_max_timescale
      )
    else:  # LOCAL_SLIDING
      partial_rotary_factor = config.local_rope_proportion if hasattr(config, "local_rope_proportion") else 1.0
      max_timescale = (
          config.local_rope_max_timescale
          if hasattr(config, "local_rope_max_timescale") and config.local_rope_max_timescale > 0
          else config.rope_max_timescale
      )

    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        use_qk_norm=True,  # Gemma 4 models use query, key normalizations
        use_v_norm=True,
        query_pre_attn_scalar=query_pre_attn_scalar,
        share_kv_projections=share_kv_projections,
        rope_max_timescale=max_timescale,
        partial_rotary_factor=partial_rotary_factor,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    if self.config.use_post_attn_norm:
      self.post_self_attention_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_self_attention_norm = None

    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    if getattr(config, "num_experts", 1) > 1:
      self.mlp = Gemma4MoE(
          config=config,
          mesh=mesh,
          rngs=self.rngs,
          quant=self.quant,
      )
    else:
      self.mlp = MlpBlock(
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          config=config,
          quant=self.quant,
          model_mode=model_mode,
          mesh=mesh,
          rngs=self.rngs,
      )

    if self.config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_ffw_norm = None

    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=config.dtype), sharding=(None,))

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

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
      bidirectional_mask=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    # Gemma4 only applies bidirectional attention in sliding (local) layers,
    # not in full (global) attention layers.
    if self.attention_type != AttentionType.LOCAL_SLIDING:
      bidirectional_mask = None

    # Self-attention block
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    if cfg.use_post_attn_norm:
      attention_lnx = self.post_self_attention_norm(attention_lnx)
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)

    attention_lnx += inputs
    residual = attention_lnx
    attn_output = self.pre_ffw_norm(attention_lnx)

    # MLP block.
    if getattr(self.config, "num_experts", 1) > 1:
      mlp_lnx, load_balance_loss, _ = self.mlp(attn_output, original_inputs=attention_lnx)
      if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        self.sow("intermediates", "moe_lb_loss", load_balance_loss)
    else:
      mlp_lnx = self.mlp(attn_output, deterministic=deterministic)

    if cfg.use_post_ffw_norm:
      mlp_lnx = self.post_ffw_norm(mlp_lnx)

    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    next_layer_addition = mlp_lnx + residual
    layer_output = next_layer_addition
    layer_output = layer_output * self.layer_scalar.value

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


Gemma4DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma4DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Gemma4ScannableBlock(nnx.Module):
  """A repeatable block of Gemma4 decoder layers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      num_of_layers: int = 1,
  ):
    """Initializes the instance.

    Args:
      config: The Config object with model hyperparameters.
      mesh: The device mesh for distributed training.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: The random number generators for initialization.
      quant: The quantization configuration.
      num_of_layers: The number of layers in the model.
    """
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.num_of_layers = num_of_layers

    for layer_id in range(self.num_of_layers):
      attention_type = get_attention_type(layer_id)
      layer_name = f"layers_{layer_id}"
      layer = Gemma4DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          attention_type=attention_type,
          layer_idx=layer_id,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot=None,
      page_state=None,
      previous_chunk=None,
      bidirectional_mask=None,
  ):
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    y = inputs

    for layer_id in range(self.num_of_layers):
      y, _ = getattr(self, f"layers_{layer_id}")(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          bidirectional_mask=bidirectional_mask,
      )

    return y, None


Gemma4ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Gemma4ScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
