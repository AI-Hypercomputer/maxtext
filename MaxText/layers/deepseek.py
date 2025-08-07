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
"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Optional, Protocol

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from MaxText.layers import attentions,initializers, linears,moe, nnx_wrappers, quantizations
from MaxText.common_types import Array, Config
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.inference import page_manager
from MaxText.common_types import MODEL_MODE_PREFILL

# -----------------------------------------
# The Decoder Layer for DeepSeek v3
# -----------------------------------------

class DeepSeekCommonAttribute(Protocol):
    pre_attention_norm: RMSNorm
    attention: attentions.MLA
    post_attention_norm: RMSNorm


def self_attention_with_norm(obj: DeepSeekCommonAttribute, inputs: Array, decoder_segment_ids, decoder_positions, deterministic,
    model_mode, previous_chunk=None, page_state: Optional[page_manager.PageState] = None,
    slot: Optional[int] = None)->tuple:

  lnx = obj.pre_attention_norm(inputs)
  if model_mode == MODEL_MODE_PREFILL:
    logical_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
  else:
    logical_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

  lnx = nn.with_logical_constraint(lnx, logical_axis_names)
  attention_lnx = obj.attention(
      lnx,
      lnx,
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=deterministic,
      model_mode=model_mode,
      previous_chunk=previous_chunk,
      page_state=page_state,
      slot=slot,
  )
  attention_lnx = nn.with_logical_constraint(attention_lnx, logical_axis_names)
  intermediate_inputs = inputs + attention_lnx
  
  hidden_states = obj.post_attention_norm(intermediate_inputs)
  hidden_states = nn.with_logical_constraint(hidden_states, logical_axis_names)
  
  return hidden_states, intermediate_inputs


def post_process(cfg, layer_output, sow):
  """postprocessing."""
  if cfg.record_internal_nn_metrics:
    sow("intermediates", "activation_mean", jnp.mean(layer_output))
    sow("intermediates", "activation_stdev", jnp.std(layer_output))
    sow(
        "intermediates",
        "activation_fraction_zero",
        jnp.sum(layer_output == 0) / jnp.size(layer_output),
    )

  if cfg.scan_layers:
    return layer_output, None
  else:
    return layer_output


class BaseDeepSeekLayer(nnx.Module):
  def __init__(
      self,
      *,
      config: Config,
      mesh: Mesh,
      mlp_block: nnx.Module|nn.Module,
      quant: Optional[quantizations.AqtQuantization] = None,
      rngs: Optional[nnx.Rngs] = None,
      **kwargs: Any
  ) -> None:
    super().__init__()
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs if rngs is not None else kwargs.get("rngs", nnx.Rngs(0))
    self.mlp_block = mlp_block
    self.inputs_shape = [
      self.config.per_device_batch_size,
      self.config.max_target_length,
      self.config.base_emb_dim,
    ]

    self.drop_out = nnx.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,),rngs=self.rngs)

    self.pre_attention_norm = RMSNorm(
      num_features=self.inputs_shape[-1],
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      kernel_axes=("norm", ),
      epsilon=self.config.normalization_layer_epsilon,
      rngs=self.rngs
    )

    self.post_attention_norm = RMSNorm(
      num_features=self.inputs_shape[-1],
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      kernel_axes=("norm", ),
      epsilon=self.config.normalization_layer_epsilon,
      rngs=self.rngs
    )

    self.attention = attentions.MLA(
      config=self.config,
      num_query_heads=self.config.num_query_heads,
      num_kv_heads=self.config.num_kv_heads,
      head_dim=self.config.head_dim,
      max_target_length=self.config.max_target_length,
      max_prefill_predict_length=self.config.max_prefill_predict_length,
      attention_kernel=self.config.attention,
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
    )


  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: Optional[page_manager.PageState] = None,
      slot: Optional[int] = None,
  ):
    cfg = self.config
    if model_mode == MODEL_MODE_PREFILL:
      logical_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      logical_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    inputs = nn.with_logical_constraint(inputs, logical_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    hidden_states, intermediate_inputs = self_attention_with_norm(
        self,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        page_state,
        slot,
    )
    
    if isinstance(self.mlp_block, linears.MlpBlock):
      mlp_lnx = self.mlp_block(hidden_states, deterministic=deterministic)
    elif isinstance(self.mlp_block, moe.RoutedAndSharedMoE):
      mlp_lnx = self.mlp_block(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, logical_axis_names)

    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.drop_out(layer_output, deterministic=deterministic)
    layer_output = nn.with_logical_constraint(
        layer_output,
        logical_axis_names,
    )
    return post_process(cfg, layer_output, self.sow)


class DeepSeekDenseLayer(BaseDeepSeekLayer):
  """DeepSeek-style dense layer with Multi-Head Latent Attention."""

  def __init__(
      self,
      *,
      config: Config,
      mesh: Mesh,
      quant: Optional[quantizations.AqtQuantization] = None,
      rngs: Optional[nnx.Rngs] = None,
      **kwargs: Any
  ) -> None:
    
    safe_rngs: nnx.Rngs = rngs if rngs is not None else kwargs.get("rngs", nnx.Rngs(0))

    mlp_block = linears.MlpBlock(
      in_features=config.base_emb_dim,
      intermediate_dim=config.mlp_dim,
      activations=config.mlp_activations,
      intermediate_dropout_rate=config.dropout_rate,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      config=config,
      quant=quant,
      rngs=safe_rngs
    )
    super().__init__(config=config, mesh=mesh,mlp_block=mlp_block, quant=quant, rngs=safe_rngs, **kwargs)


class DeepSeekDenseLayerWrapper(nn.Module):
  """A Linen wrapper for the NNX DeepSeekDenseLayer"""

  config: Config
  mesh: Mesh
  quant: Quant | None = None

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Call the underlying NNX layer"""
    layer = nnx_wrappers.to_linen(
      DeepSeekDenseLayer,
      config=self.config,
      mesh=self.mesh,
      quant=self.quant,
      metadata_fn=initializers.variable_to_logically_partitioned,
    )
    return layer(*args, **kwargs)


class DeepSeekMoELayer(BaseDeepSeekLayer):
  """DeepSeek-style MoE layer with Multi-Head Latent Attention.
  Supports dropless and dropping base on configs.
  Uses a bias in routing instead of load balancing loss.
  """
  def __init__(
      self,
      *,
      config: Config,
      mesh: Mesh,
      quant: Optional[quantizations.AqtQuantization] = None,
      rngs: Optional[nnx.Rngs] = None,
      **kwargs: Any
  ) -> None:
    mlp_block = moe.RoutedAndSharedMoE(
        name="DeepSeekMoeBlock_0",
        config=config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
    )
    super().__init__(config=config, mesh=mesh,mlp_block=mlp_block, quant=quant, rngs=rngs, **kwargs)

class DeepSeekMoELayerWrapper(nn.Module):
  """A Linen wrapper for the NNX DeepSeekMoELayer"""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, *args, **kwargs):
    """Call the underlying NNX layer"""
    layer = nnx_wrappers.to_linen(
      DeepSeekMoELayer,
      config=self.config,
      mesh=self.mesh,
      quant=self.quant,
      metadata_fn=initializers.variable_to_logically_partitioned,
    )
    return layer(*args, **kwargs)