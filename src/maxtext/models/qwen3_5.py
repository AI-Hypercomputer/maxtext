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

from flax import nnx
import jax
from jax import ad_checkpoint
import jax.numpy as jnp
from maxtext.common.common_types import Array, get_weight_dtype
from maxtext.layers.linears import MlpBlock
from maxtext.models.qwen3 import (
    Qwen3NextDecoderLayer,
    Qwen3NextFullAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextScannableBlock,
    Qwen3NextSparseMoeBlock,
)
from maxtext.utils import max_utils

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
  """Scanned Structure for Text-only Architecture, explicitly invoking Qwen3_5 layers."""

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


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
  """Qwen3.5 hybrid decoder layer.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def _norm_weight_dtype(self):
    return get_weight_dtype(self.config, "norm")

  def _make_full_attention(self, *, rngs: nnx.Rngs):
    return Qwen3_5FullAttention(
        config=self.config,
        mesh=self.mesh,
        quant=self.quant,
        model_mode=self.model_mode,
        layer_idx=self.layer_idx,
        rngs=rngs,
    )

  def _make_linear_attention(self, *, rngs: nnx.Rngs):
    cfg = self.config
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, self.model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
    return Qwen3_5GatedDeltaNet(
        config=cfg,
        inputs_shape=dummy_inputs_shape,
        mesh=self.mesh,
        dtype=cfg.dtype,
        model_mode=self.model_mode,
        quant=self.quant,
        rngs=rngs,
    )

  def _make_mlp(self, *, rngs: nnx.Rngs):
    cfg = self.config
    if getattr(cfg, "num_experts", 1) > 1:
      return Qwen3_5SparseMoEBlock(config=cfg, mesh=self.mesh, quant=self.quant, rngs=rngs)
    return MlpBlock(
        config=cfg,
        mesh=self.mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim,
        activations=tuple(cfg.mlp_activations),
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=get_weight_dtype(cfg, "mlp"),
        quant=self.quant,
        model_mode=self.model_mode,
        rngs=rngs,
    )

  def _apply_mlp(self, hidden_states, deterministic, forced_routed_experts):
    if getattr(self.config, "num_experts", 1) > 1:
      return super()._apply_mlp(hidden_states, deterministic, forced_routed_experts)
    # A dense MLP routes nothing, so there is no routing to replay and no load-balance loss.
    mlp_output = self.mlp(
        hidden_states,
        deterministic=deterministic,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
    )
    return mlp_output, None

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray = None,
      decoder_positions: None | jnp.ndarray = None,
      deterministic: bool = False,
      model_mode: str = "train",
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    if not (
        getattr(self.config, "qwen3_5_sublayer_remat", False)
        and getattr(self.config, "num_experts", 1) > 1
        and kv_cache is None
    ):
      return super().__call__(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          forced_routed_experts=forced_routed_experts,
      )

    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = ad_checkpoint.checkpoint_name(inputs, "decoder_layer_input")

    if isinstance(attention_metadata, dict):
      attention_metadata = attention_metadata.get(f"layer.{self.layer_idx}", attention_metadata.get(self.layer_idx))

    attn_graphdef, attn_state = nnx.split((self.input_layernorm, self.attention), nnx.Variable)

    def _run_attn_subblock_impl(x_in, state_in, seg_in, pos_in):
      norm_mod, attn_mod = nnx.merge(attn_graphdef, state_in)
      h = norm_mod(x_in, out_sharding=self.out_sharding)
      h = self._maybe_shard_with_logical(h, self.activation_axis_names)
      if isinstance(attn_mod, Qwen3NextFullAttention):
        attn_out, _ = cast(Qwen3NextFullAttention, attn_mod)(
            h,
            seg_in,
            pos_in,
            deterministic,
            model_mode,
            kv_cache=None,
            attention_metadata=attention_metadata,
            out_sharding=self.out_sharding,
        )
      else:
        attn_out, _ = cast(Qwen3NextGatedDeltaNet, attn_mod)(
            h,
            model_mode=model_mode,
            kv_cache=None,
            decoder_segment_ids=seg_in,
            attention_metadata=attention_metadata,
            out_sharding=self.out_sharding,
        )
      attn_out = self._maybe_shard_with_logical(attn_out, self.activation_axis_names)
      h_out = x_in + attn_out
      return self._maybe_shard_with_logical(h_out, self.activation_axis_names)

    _run_attn_subblock = jax.checkpoint(_run_attn_subblock_impl, policy=None, prevent_cse=False)
    hidden_states = _run_attn_subblock(inputs, attn_state, decoder_segment_ids, decoder_positions)
    hidden_states = ad_checkpoint.checkpoint_name(hidden_states, "attn_subblock_out")
    new_kv_cache = None

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    mlp_output, load_balance_loss = self._apply_mlp(hidden_states, deterministic, forced_routed_experts)
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.moe_lb_loss = nnx.Intermediate(load_balance_loss)

    mlp_output = self._maybe_shard_with_logical(mlp_output, self.activation_axis_names)
    layer_output = residual + mlp_output
    layer_output = self._maybe_shard_with_logical(
        layer_output,
        self.activation_axis_names,
    )
    return layer_output, new_kv_cache
