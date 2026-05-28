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

"""Qwen3.5 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, cast

from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import Config, Array
from maxtext.layers import initializers as max_initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers.normalizations import Qwen3NextRMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN

from maxtext.inference import page_manager
from maxtext.utils import max_utils, max_logging

from maxtext.models.qwen3 import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextFullAttention,
    Qwen3NextSparseMoeBlock,
)


# -----------------------------------------
# Qwen3.5 Layer Implementations
# -----------------------------------------


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
  """Qwen3.5 GatedDeltaNet layer that is identical to Qwen3-Next GatedDeltaNet"""


class Qwen3_5FullAttention(Qwen3NextFullAttention):
  """Qwen3.5 Gated Attention layer that is identical to Qwen3-Next"""


class Qwen3_5SparseMoEBlock(Qwen3NextSparseMoeBlock):
  """Shares same MoE code as Qwen3-Next"""


class Qwen3_5ScannableBlock(nnx.Module):
  """Scanned Structure for Text-only Architecture, explicitly invoking Qwen3_5 layers."""

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant=None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    cfg = self.config

    # Explicitly instantiate Qwen3_5DecoderLayer here
    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer_rngs = self.rngs.fork()
      layer_name = f"layer_{i}"
      layer = Qwen3_5DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          layer_idx=i,
          rngs=layer_rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ) -> tuple[Array, None]:
    cfg = self.config
    x = carry

    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      x, _ = layer(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          page_state,
          slot,
      )

    return x, None


class Qwen3_5DecoderLayer(nnx.Module):
  """
  This layer is a hybrid, capable of functioning as either:
  1. A standard attention + MoE layer.
  2. A linear attention + MoE layer.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    # First LayerNorm, applied before the attention block.
    self.input_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        eps=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Determine the type of attention mechanism for the current layer.
    is_full_attention_layer = (self.layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0

    # Conditionally instantiate either the Linear Attention or Full Attention block.
    if is_full_attention_layer:
      self.attention = Qwen3_5FullAttention(
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=model_mode,
          layer_idx=self.layer_idx,
          rngs=rngs,
      )
    else:
      def get_gdn_batch_seq_len(config, model_mode):
        """
        Custom override for GDN cache initialization.
        Bypasses the max_utils assumption that prefill batch size is always 1,
        and properly scales it to the global TPU mesh.
        """
        # Calculate the true global batch size based on the cluster topology
        batch_size = config.micro_batch_size_to_train_on

        if model_mode == MODEL_MODE_PREFILL:
          return batch_size, config.max_prefill_predict_length

        elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
          return batch_size, 1

        elif model_mode == MODEL_MODE_TRAIN:
          return batch_size, config.max_target_length

        else:
          raise ValueError(f"Unknown model_mode: {model_mode}")
      # batch_size, seq_len = get_gdn_batch_seq_len(config, model_mode)
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      dummy_inputs_shape = (
          batch_size,
          seq_len,
          config.emb_dim
      )
      print(f"\n[DEBUG INIT] Layer {self.layer_idx} | Mode: {model_mode}", flush=True)
      print(f"[DEBUG INIT] Calculated batch_size: {batch_size}", flush=True)
      print(f"[DEBUG INIT] Passing inputs_shape to GDN: {dummy_inputs_shape}\n", flush=True)
      self.attention = Qwen3_5GatedDeltaNet(
          config=cfg, 
          inputs_shape=dummy_inputs_shape,
          dtype=cfg.dtype, 
          model_mode=model_mode, 
          rngs=rngs
      )

    # Second LayerNorm, applied before the MoE block.
    self.post_attention_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        eps=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Instantiate our `Qwen3_5SparseMoEBlock`.
    self.mlp = Qwen3_5SparseMoEBlock(config=cfg, mesh=self.mesh, quant=self.quant, rngs=rngs)

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
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    residual = inputs

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Conditionally apply either the Linear Attention or Full Attention block.
    if isinstance(self.attention, Qwen3_5FullAttention):
      attention_output, new_kv_cache = cast(Qwen3_5FullAttention, self.attention)(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      attention_output, new_kv_cache = cast(Qwen3_5GatedDeltaNet, self.attention)(
          hidden_states,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
      )

    # First residual connection after attention
    hidden_states = residual + attention_output
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Prepare for the MoE block by capturing the new residual
    residual = hidden_states

    # Second LayerNorm, applied before the MoE block.
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Instantiate and call our `Qwen3_5SparseMoEBlock`.
    mlp_output, load_balance_loss = self.mlp(hidden_states, deterministic=deterministic)

    # We sow the load balancing loss so it can be collected and added to the total loss
    # during training.
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    # Final residual connection (after the MoE block)
    layer_output = residual + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        self.activation_axis_names,
    )
    return layer_output, new_kv_cache


Qwen3_5DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3_5DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)


Qwen3_5ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3_5ScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
