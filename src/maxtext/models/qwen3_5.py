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
from maxtext.utils import max_utils

from maxtext.models.qwen3 import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextFullAttention,
    Qwen3NextScannableBlock,
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


class Qwen3_5ScannableBlock(Qwen3NextScannableBlock):
  """Scanned Structure for Text-only Architecture, explicitly invoking Qwen3_5 layers.

  Qwen3.5 repeats the same hybrid attention period as Qwen3-Next -- several
  GatedDeltaNet layers followed by one full-attention layer -- so it reuses
  Qwen3-Next's hierarchical nested scans (an inner scan over the homogeneous
  linear-attention layers plus a trip-count-one scan over the full-attention
  layer) and only swaps in the Qwen3.5 decoder layer.
  """

  def _make_decoder_layer(self, *, layer_idx, is_full_attention_layer, rngs):
    return Qwen3_5DecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        quant=self.quant,
        layer_idx=layer_idx,
        is_full_attention_layer=is_full_attention_layer,
        rngs=rngs,
    )


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
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      is_full_attention_layer: bool | None = None,
      rngs: nnx.Rngs,
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
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Determine the type of attention mechanism for the current layer. A scanned block
    # knows each sub-layer's role up front and passes it explicitly, because inside a
    # scan the layer's position is not recoverable from layer_idx.
    if is_full_attention_layer is None:
      is_full_attention_layer = (self.layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    self.is_full_attention_layer = is_full_attention_layer

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
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
      self.attention = Qwen3_5GatedDeltaNet(
          config=cfg, inputs_shape=dummy_inputs_shape, mesh=self.mesh, dtype=cfg.dtype, model_mode=model_mode, rngs=rngs
      )

    # Second LayerNorm, applied before the MoE block.
    self.post_attention_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
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
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
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
          attention_metadata=attention_metadata,
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
    mlp_output, load_balance_loss = self.mlp(
        hidden_states,
        deterministic=deterministic,
        forced_routed_experts=forced_routed_experts,
    )

    # We sow the load balancing loss so it can be collected and added to the total loss
    # during training.
    # Assigned rather than sown: `sow` appends to a tuple, so the layer's Intermediate
    # structure would grow on every call, which a `jax.lax.scan` body cannot express.
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.moe_lb_loss = nnx.Intermediate(load_balance_loss)

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
