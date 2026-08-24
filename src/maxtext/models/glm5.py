# Copyright 2026 Google LLC
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

"""GLM model definitions (GLM-5.1 & GLM-5.2 with Cross-Layer IndexShare)."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Optional

import absl.logging
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import AttentionType, Config, HyperConnectionType
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.models import deepseek
from maxtext.utils import index_share_utils


class GLMGenericLayer(deepseek.DeepSeekGenericLayer):  # pylint: disable=abstract-method
  """Generic GLM layer with Multi-Head Latent Attention and IndexShare support."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)

    # GLM-5.2 Cross-Layer IndexShare Role Resolution
    self.is_index_share_enabled = getattr(config, "use_index_share", False)
    self.is_shared_layer = False
    self.served_group_size = 1
    if self.is_index_share_enabled and layer_idx >= 0:
      pattern = index_share_utils.parse_index_share_pattern(config.index_share_pattern, config.num_decoder_layers)
      self.is_shared_layer = index_share_utils.is_shared_layer(layer_idx, pattern)
      self.served_group_size = index_share_utils.get_served_group_sizes(pattern)[layer_idx]
      if layer_idx == 0 and jax.process_index() == 0:
        num_f = pattern.count("F")
        num_s = pattern.count("S")
        absl.logging.info(
            "[GLM-5.2 IndexShare Active] Total layers: %d | Pattern: %s | Full (F) layers with active indexers: %d | "
            "Shared (S) layers with pruned indexers: %d (Pruned %.1f%% indexer compute/parameters)",
            config.num_decoder_layers,
            config.index_share_pattern,
            num_f,
            num_s,
            num_s / config.num_decoder_layers * 100,
        )

    # Re-initialize MLA with GLM-specific IndexShare configuration
    self.self_attention = attention_mla.MLA(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=AttentionType(self.config.attention_type),
        inputs_q_shape=self.dummy_inputs_shape,
        inputs_kv_shape=self.dummy_inputs_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(self.config),
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
        attn_logits_soft_cap=self.config.attn_logits_soft_cap,
        is_shared_layer=self.is_shared_layer,
        served_group_size=self.served_group_size,
    )

  def attention_op(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      cached_indexer_state=None,
      layer_idx=None,
  ):
    """Executes the attention layer and passes cached indexer state."""
    attn_out = self.self_attention(
        x,
        x,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        out_sharding=self.out_sharding,
        previous_chunk=previous_chunk,
        slot=slot,
        cached_indexer_state=cached_indexer_state,
        layer_idx=layer_idx,
    )
    attention_result = attn_out[0]
    if self.is_index_share_enabled:
      new_indexer_state = attn_out[2] if len(attn_out) > 2 else None
      return self.with_logical_constraint(attention_result), new_indexer_state
    else:
      return self.with_logical_constraint(attention_result), None

  def post_process(self, layer_output, load_balance_loss, moe_bias_updates, kv_cache=None, cached_indexer_state=None):
    """Post-processing with IndexShare state pass-through."""
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    if getattr(self.config, "record_internal_nn_metrics", False):
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.is_index_share_enabled:
      if self.config.scan_layers:
        return layer_output, None, cached_indexer_state
      return layer_output, kv_cache, cached_indexer_state

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache

  def self_attention_with_norm_op(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      cached_indexer_state=None,
      layer_idx=None,
  ):
    """Self-attention with normalization and IndexShare caching."""
    if self.is_mhc_enabled:
      intermediate_inputs, _ = self.mhc_attention(
          self.pre_attention_norm_op,
          self.self_attention,
          x=inputs,
          mhc_type=HyperConnectionType.ATTENTION,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          out_sharding=self.out_sharding,
          previous_chunk=previous_chunk,
          slot=slot,
          cached_indexer_state=cached_indexer_state,
      )
      new_indexer_state = None
    else:
      lnx = self.pre_attention_norm_op(inputs)
      attention_lnx, new_indexer_state = self.attention_op(
          lnx,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          slot,
          cached_indexer_state=cached_indexer_state,
          layer_idx=layer_idx,
      )
      intermediate_inputs = inputs + attention_lnx
    # Normalization
    hidden_states = self.post_attention_norm_op(intermediate_inputs)
    return hidden_states, intermediate_inputs, new_indexer_state


class GLMDenseLayer(GLMGenericLayer):
  """GLM dense layer with Multi-Head Latent Attention."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)
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
        mesh=mesh,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic, *args, **kwargs):
    mlp = self.mlp(x, deterministic, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding)
    return self.with_logical_constraint(mlp)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
      cached_indexer_state=None,
      layer_idx=None,
      **kwargs,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    hidden_states, intermediate_inputs, new_indexer_state = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
        cached_indexer_state=cached_indexer_state,
        layer_idx=layer_idx,
    )

    if self.is_mhc_enabled:
      layer_output, _ = self.mhc_mlp(
          self.post_attention_norm_op,
          self.mlp,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_DENSE,
          deterministic=deterministic,
      )
    else:
      mlp_lnx = self.mlp_op(hidden_states, deterministic)
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, None, None, kv_cache, new_indexer_state)


class GLMMoELayer(GLMGenericLayer):
  """GLM MoE layer with Multi-Head Latent Attention and IndexShare support."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)
    self.DeepSeekMoeBlock_0 = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic, *args, **kwargs):
    mlp_lnx, load_balance_loss, moe_bias_updates = self.DeepSeekMoeBlock_0(
        x, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding
    )
    return self.with_logical_constraint(mlp_lnx), load_balance_loss, moe_bias_updates

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
      cached_indexer_state=None,
      layer_idx=None,
      **kwargs,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    hidden_states, intermediate_inputs, new_indexer_state = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
        cached_indexer_state=cached_indexer_state,
        layer_idx=layer_idx,
    )

    if self.is_mhc_enabled:
      layer_output, metadata = self.mhc_mlp(
          self.post_attention_norm_op,
          self.DeepSeekMoeBlock_0,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_MOE,
      )
      load_balance_loss = metadata["load_balance_loss"]
      moe_bias_updates = metadata["moe_bias_updates"]
    else:
      mlp_lnx, load_balance_loss, moe_bias_updates = self.mlp_op(hidden_states, deterministic)
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache, new_indexer_state)


GLMDenseLayerToLinen = nnx_wrappers.to_linen_class(
    GLMDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

GLMMoELayerToLinen = nnx_wrappers.to_linen_class(
    GLMMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
