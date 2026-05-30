# Copyright 2023-2026 Google LLC
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

"""Gemma4 decoder layer with latent MoE (a.k.a. inverted MoE).

The attention output is projected to a smaller ``attention_output_dim`` (the
"latent" dim); the MoE block (routed + shared experts) runs at
``moe_expert_input_dim`` (== ``attention_output_dim``); a per-layer
up-projection maps back to ``emb_dim`` for the residual connection. Mirrors
the structure of ``qwen3_custom``.
"""

from typing import Optional

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp
import jax.sharding

from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import Config, AttentionType, MODEL_MODE_PREFILL
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models.gemma4 import GEMMA4_ATTENTION_PATTERN, get_attention_type
from maxtext.utils import max_utils


class Gemma4LatentMoEAttention(Attention):
  """Gemma4 attention with output projected to ``attention_output_dim``."""

  def init_out_w(self, output_dim: int) -> nnx.Module:
    del output_dim
    if not self.config.attention_output_dim > 0:
      raise ValueError("attention_output_dim must be positive for Gemma4LatentMoEAttention.")

    in_features = (self.num_query_heads, self.head_dim)
    out_kernel_axis = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("heads", "kv", "embed")
    )

    return DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.config.attention_output_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=out_kernel_axis,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        shard_mode=self.config.shard_mode,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )


class Gemma4LatentMoEMoE(nnx.Module):
  """Gemma4 MoE block whose layer-norms operate on the latent dim."""

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

    latent_dim = config.moe_expert_input_dim

    self.moe_block = moe.RoutedAndSharedMoE(
        config=config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        quant=self.quant,
        rngs=self.rngs,
    )

    self.pre_forward_scale_2 = nnx.Param(
        jnp.ones((latent_dim,), dtype=self.config.weight_dtype),
        sharding=("embed",),
    )
    self.pre_feedforward_layernorm_2 = RMSNorm(
        num_features=latent_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_1 = RMSNorm(
        num_features=latent_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_2 = RMSNorm(
        num_features=latent_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.gate_norm = RMSNorm(
        num_features=latent_dim,
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

    # Experts use standard RMSNorm; gate uses scale-free norm * root_size * router_scale.
    routed_inputs = self.pre_feedforward_layernorm_2(original_inputs)
    gate_dtype = jnp.float32 if self.config.float32_gate_logits else self.config.dtype
    unscaled_norm = self.gate_norm(original_inputs)
    # root_size is over the gate input (latent) dim, matching Gemma4MoE.
    root_size = self.config.moe_expert_input_dim**-0.5
    router_scale = jnp.asarray(self.pre_forward_scale_2.value, gate_dtype)
    gate_inputs = unscaled_norm * root_size * router_scale

    routed_experts, load_balance_loss, moe_bias_updates = self.moe_block.routed_moe(
        routed_inputs, gate_inputs=gate_inputs, out_sharding=out_sharding
    )
    routed_experts = self.post_feedforward_layernorm_2(routed_experts)

    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


class Gemma4LatentMoEDecoderLayer(nnx.Module):
  """Gemma4 decoder layer with latent MoE (no mid-layer residual)."""

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
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.attention_type = attention_type
    self.layer_idx = layer_idx

    if config.attention_output_dim <= 0 or config.attention_output_dim != config.moe_expert_input_dim:
      raise ValueError(
          "attention_output_dim must be positive and equal to moe_expert_input_dim for Gemma4LatentMoEDecoderLayer."
      )

    latent_dim = config.moe_expert_input_dim
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

    self.self_attention = Gemma4LatentMoEAttention(
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
        use_qk_norm=True,
        use_v_norm=True,
        query_pre_attn_scalar=query_pre_attn_scalar,
        share_kv_projections=share_kv_projections,
        rope_max_timescale=max_timescale,
        partial_rotary_factor=partial_rotary_factor,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    # Norms after attention live at the latent dim.
    if self.config.use_post_attn_norm:
      self.post_self_attention_norm = RMSNorm(
          num_features=latent_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_self_attention_norm = None

    self.pre_ffw_norm = RMSNorm(
        num_features=latent_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.mlp = Gemma4LatentMoEMoE(
        config=config,
        mesh=mesh,
        rngs=self.rngs,
        quant=self.quant,
    )

    if self.config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          num_features=latent_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_ffw_norm = None

    out_kernel_axis = (None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("mlp", "embed")
    self.layer_up_projection = DenseGeneral(
        in_features_shape=latent_dim,
        out_features_shape=config.emb_dim,
        axis=-1,
        kernel_init=initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=out_kernel_axis,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        use_bias=False,
        rngs=self.rngs,
    )

    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=config.weight_dtype), sharding=(None,))

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

    is_scan_carry = False
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      kv_cache = stacked_kv_cache[layer_idx]
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    # Gemma4 only applies bidirectional attention in local layers.
    if self.attention_type != AttentionType.LOCAL_SLIDING:
      bidirectional_mask = None

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
    # Tag at the latent dim so remat_policy=custom can save it and skip attention recompute.
    attention_lnx = checkpoint_name(attention_lnx, "latent_input")

    # No mid-layer residual: post-attention activations are at the latent dim
    # and cannot be added to the emb_dim residual stream.
    if cfg.use_post_attn_norm:
      attention_lnx = self.post_self_attention_norm(attention_lnx)

    attn_output = self.pre_ffw_norm(attention_lnx)
    mlp_lnx, load_balance_loss, _ = self.mlp(attn_output, original_inputs=attention_lnx)
    if cfg.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    if cfg.use_post_ffw_norm:
      mlp_lnx = self.post_ffw_norm(mlp_lnx)

    layer_output = self.layer_up_projection(mlp_lnx)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    layer_output = inputs + layer_output
    layer_output = layer_output * jnp.asarray(self.layer_scalar.value, cfg.dtype)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if is_scan_carry:

      def update_cache(cache, val):
        if jnp.size(val) > 0:
          return cache.at[layer_idx].set(val)
        return cache

      stacked_kv_cache = jax.tree_util.tree_map(update_cache, stacked_kv_cache, kv_cache)
      return (layer_output, stacked_kv_cache, layer_idx + 1), None
    elif cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


Gemma4LatentMoEDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma4LatentMoEDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Gemma4LatentMoEScannableBlock(nnx.Module):
  """A repeatable block of Gemma4LatentMoEDecoderLayer instances."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      num_of_layers: int = 1,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.num_of_layers = num_of_layers

    for layer_id in range(self.num_of_layers):
      attention_type = get_attention_type(layer_id)
      layer = Gemma4LatentMoEDecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          attention_type=attention_type,
          layer_idx=layer_id,
      )
      setattr(self, f"layers_{layer_id}", layer)

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


Gemma4LatentMoEScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Gemma4LatentMoEScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


__all__ = [
    "GEMMA4_ATTENTION_PATTERN",
    "Gemma4LatentMoEAttention",
    "Gemma4LatentMoEDecoderLayer",
    "Gemma4LatentMoEDecoderLayerToLinen",
    "Gemma4LatentMoEMoE",
    "Gemma4LatentMoEScannableBlock",
    "Gemma4LatentMoEScannableBlockToLinen",
    "get_attention_type",
]
