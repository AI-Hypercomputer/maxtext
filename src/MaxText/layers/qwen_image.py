# Copyright 2023–2025 Google LLC
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

"""Qwen Image family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import jax
import jax.nn
from jax.sharding import Mesh
import jax.numpy as jnp
from flax import nnx
from MaxText.common_types import AttentionType, Config, DType, Array
from MaxText.layers.attentions import QwenImageAttention, Qwen3NextRotaryEmbedding
from MaxText.layers.linears import DenseGeneral

class QwenImageImgMod(nnx.Module):

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    """
    Args:
      config: MaxText configuration object.
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    """
    cfg = config

    in_features = config.base_emb_dim

    self.img_mod_1 = DenseGeneral(
      in_features_shape=in_features,
      out_features_shape=(in_features * 6),
      dtype=cfg.dtype,
      use_bias=False,
      kernel_axes=("embed", "mlp"),
      matmul_precision=cfg.matmul_precision,
      rngs=rngs,
    )
  
  def __call__(self, inputs: jnp.ndarray):
    hidden_states = jax.nn.silu(inputs)
    return self.img_mod_1(hidden_states)

class QwenImageTransformerBlock(nnx.Module):

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    """Qwen Image Edit Transformer Block.
    
    This module implements the full Qwen Image transformer block.

    Attributes:
      config: The model configuration object.
      mesh: The device mesh for sharding.
    """
    self.config = config
    self.mesh = mesh
    in_features = config.base_emb_dim

    self.img_mod = QwenImageImgMod(config)
    self.img_norm1 = nnx.LayerNorm(
      num_features=in_features,
      epsilon=config.normalization_layer_epsilon,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
      use_bias=False,
      use_scale=False,
      rngs=rngs,
    )

    self.attn = QwenImageAttention(
      config=self.config,
      num_query_heads=self.config.base_num_query_heads,
      num_kv_heads=self.config.base_num_kv_heads,
      head_dim=self.config.head_dim,
      max_target_length=self.config.base_emb_dim,
      attention_kernel="dot_product",
      inputs_q_shape=(1, 1, self.config.)

    )
  
  def __call__(self, inputs: jnp.ndarray):


# class QwenImageEdit(nnx.Module):

#   def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
#     self.config = config
#     self.mesh = mesh
#     self.rngs = rngs

