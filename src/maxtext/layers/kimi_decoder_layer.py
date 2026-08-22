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

"""Kimi K3 Decoder Layer in MaxText (NNX)."""

from typing import Any, Optional


from flax import nnx
import jax

from maxtext.common import common_types as ctypes
from maxtext.layers import linears, quantizations
from maxtext.layers.attention_mla import MLA
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.kda import KimiDecoupledAttention
from maxtext.layers.moe import RoutedAndSharedMoE
from maxtext.layers.normalizations import RMSNorm


class KimiDecoderLayer(nnx.Module):
  """Decoder layer for Kimi K3, which can be a KDA (linear attn) or MLA (full attn) layer."""

  def __init__(
      self,
      config: ctypes.Config,
      mesh: jax.sharding.Mesh,
      layer_idx: int,
      model_mode: str = ctypes.MODEL_MODE_TRAIN,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.layer_idx = layer_idx
    self.model_mode = model_mode
    self.quant = quant

    layer_num = layer_idx + 1
    self.is_kda = layer_num in config.kda_layers

    # Pre-attention norm
    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

    # Attention layer: KDA or MLA
    if self.is_kda:
      self.self_attention = KimiDecoupledAttention(
          config=config,
          layer_idx=layer_idx,
          rngs=rngs,
      )
    else:
      self.self_attention = MLA(
          config=config,
          num_query_heads=config.num_query_heads,
          num_kv_heads=config.num_kv_heads,
          head_dim=config.head_dim,
          max_target_length=config.max_target_length,
          mesh=mesh,
          attention_kernel=config.attention,
          inputs_q_shape=(1, 1, config.emb_dim),
          inputs_kv_shape=(1, 1, config.emb_dim),
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=quant,
          model_mode=model_mode,
          rngs=rngs,
      )

    # Pre-MLP norm
    self.pre_mlp_norm = RMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

    # MLP / MoE layer
    if config.num_experts > 1 and layer_idx >= config.first_num_dense_layers:
      self.mlp = RoutedAndSharedMoE(
          config=config,
          mesh=mesh,
          kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
          kernel_axes=("embed_moe", None),
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=quant,
          rngs=rngs,
      )

    else:
      self.mlp = linears.MlpBlock(
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          model_mode=model_mode,
          config=config,
          quant=quant,
          mesh=mesh,
          rngs=rngs,
      )

  def __call__(
      self,
      inputs: jax.Array,
      segment_ids: Optional[jax.Array] = None,
      inputs_positions: Optional[jax.Array] = None,
      deterministic: bool = True,
      model_mode: str = ctypes.MODEL_MODE_TRAIN,
      *args,
      initial_kda_state: Optional[jax.Array] = None,
      kv_cache: Optional[Any] = None,
      **kwargs,
  ) -> tuple[jax.Array, Optional[Any]]:


    # 1. Pre-attention norm & Attention
    normed_inputs = self.pre_self_attention_norm(inputs)

    if self.is_kda:
      attn_out, kda_state = self.self_attention(
          normed_inputs,
          initial_state=initial_kda_state,
      )
    else:
      attn_out, _ = self.self_attention(
          inputs_q=normed_inputs,
          inputs_kv=normed_inputs,
          inputs_positions=inputs_positions,
          decoder_segment_ids=segment_ids,
          model_mode=self.model_mode,
      )
      kda_state = None

    # Residual connection for attention
    hidden_states = inputs + attn_out

    # 2. Pre-MLP norm & MLP / MoE
    normed_hidden = self.pre_mlp_norm(hidden_states)
    if isinstance(self.mlp, RoutedAndSharedMoE):
      mlp_out, _, _ = self.mlp(normed_hidden)
    else:
      mlp_out = self.mlp(normed_hidden, deterministic=deterministic)




    # Residual connection for MLP
    output = hidden_states + mlp_out

    return output, (kda_state if self.is_kda else kv_cache)


