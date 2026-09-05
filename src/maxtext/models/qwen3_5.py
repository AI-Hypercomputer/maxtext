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

import functools
from typing import Any, cast

from flax import nnx
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config, ShardMode
from maxtext.layers import initializers as max_initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers.normalizations import Qwen3NextRMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models.qwen3 import (
    Qwen3NextFullAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical

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
      slot: None | int = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ) -> tuple[Array, None]:
    cfg = self.config
    x = carry

    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      # forced_routed_experts, when present, is shaped
      # [inhomogeneous_layer_cycle_interval, batch, seq, top_k]: one slice per
      # sub-layer in this cycle (see nnx_decoders.py's scan wiring).
      layer_forced_routed_experts = forced_routed_experts[i] if forced_routed_experts is not None else None
      x, _ = layer(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          slot,
          forced_routed_experts=layer_forced_routed_experts,
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
    self.mlp_activation_axis_names = (
        "activation_batch",
        "activation_norm_length",
        "activation_mlp",
    )

    # Physical shardings used to pin sublayer outputs under ShardMode.EXPLICIT. In
    # ShardMode.AUTO the callees ignore these and let GSPMD infer the layout.
    if cfg.shard_mode == ShardMode.EXPLICIT:
      self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
      self.mlp_intermediate_sharding = create_sharding(
          mesh, self.mlp_activation_axis_names, rules=get_logical_axis_rules()
      )
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
          extra_stack_level=1,
      )
    else:
      self.out_sharding = None
      self.mlp_intermediate_sharding = None
      self._maybe_shard_with_logical = lambda inputs, *args, **kwargs: inputs

    # First LayerNorm, applied before the attention block.
    self.input_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
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
        shard_mode=cfg.shard_mode,
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
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    residual = inputs

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

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
          out_sharding=self.out_sharding,
      )
    else:
      attention_output, new_kv_cache = cast(Qwen3_5GatedDeltaNet, self.attention)(
          hidden_states,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )

    # First residual connection after attention
    attention_output = self._maybe_shard_with_logical(attention_output, self.activation_axis_names)
    hidden_states = residual + attention_output
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Prepare for the MoE block by capturing the new residual
    residual = hidden_states

    # Second LayerNorm, applied before the MoE block.
    hidden_states = self.post_attention_layernorm(hidden_states, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Instantiate and call our `Qwen3_5SparseMoEBlock`.
    mlp_output, load_balance_loss = self.mlp(
        hidden_states,
        deterministic=deterministic,
        forced_routed_experts=forced_routed_experts,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
    )

    # We sow the load balancing loss so it can be collected and added to the total loss
    # during training.
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    # Final residual connection (after the MoE block)
    mlp_output = self._maybe_shard_with_logical(mlp_output, self.activation_axis_names)
    layer_output = residual + mlp_output
    layer_output = self._maybe_shard_with_logical(
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
