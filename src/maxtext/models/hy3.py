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

"""Hy3 (Tencent Hunyuan V3) model definition.

Hy3 combines standard GQA attention with QK-Norm (as in Qwen3) with a
DeepSeek-V3-style aux-loss-free sigmoid+bias MoE router that has a shared
expert and a dense first layer (`first_num_dense_layers`). Unlike DeepSeek
V3/V4, Hy3 uses no Multi-Head Latent Attention and no compressed/sparse
attention, so this reuses `DeepSeekGenericLayer`'s generic dense/MoE-split
scaffolding (norms, dropout, sharding, post_process) but overrides
`self_attention` with plain GQA, following the same subclassing pattern
`DeepSeek4DecoderLayer` uses in `deepseek4.py`.
"""

from typing import Optional

from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.models import deepseek


def _build_hy3_attention(config: Config, mesh: Mesh, model_mode: str, quant, dummy_inputs_shape, rngs: nnx.Rngs):
  """Builds the plain GQA + QK-Norm attention module shared by Hy3's dense and MoE layers."""
  return Attention(
      config=config,
      num_query_heads=config.num_query_heads,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
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
      quant=quant,
      kv_quant=quantizations.configure_kv_quant(config),
      use_ragged_attention=config.use_ragged_attention,
      ragged_block_size=config.ragged_block_size,
      use_qk_norm=config.use_qk_norm,
      query_pre_attn_scalar=config.head_dim**-0.5,
      model_mode=model_mode,
      name="self_attention",
      rngs=rngs,
  )


class Hy3DenseLayer(deepseek.DeepSeekGenericLayer):
  """Hy3 dense decoder layer, used only for the first `config.first_num_dense_layers` layers."""

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
    self.self_attention = _build_hy3_attention(self.config, mesh, model_mode, quant, self.dummy_inputs_shape, rngs)
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
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    hidden_states, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
    )

    mlp_lnx = self.mlp_op(hidden_states, deterministic)
    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, None, None, kv_cache)


Hy3DenseLayerToLinen = nnx_wrappers.to_linen_class(
    Hy3DenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Hy3MoELayer(deepseek.DeepSeekGenericLayer):
  """Hy3 MoE decoder layer: GQA+QK-Norm attention with a DeepSeek-V3-style
  sigmoid+bias routed MoE block (1 shared expert)."""

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
    self.self_attention = _build_hy3_attention(self.config, mesh, model_mode, quant, self.dummy_inputs_shape, rngs)
    self.Hy3MoeBlock_0 = moe.RoutedAndSharedMoE(
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
    mlp_lnx, load_balance_loss, moe_bias_updates = self.Hy3MoeBlock_0(
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
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    hidden_states, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
    )

    mlp_lnx, load_balance_loss, moe_bias_updates = self.mlp_op(hidden_states, deterministic)
    layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache)


Hy3MoELayerToLinen = nnx_wrappers.to_linen_class(
    Hy3MoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
