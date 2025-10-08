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

""" Simple decoder layers for testing and debugging purposes."""

from jax import numpy as jnp
from jax.sharding import Mesh

from flax import nnx

from MaxText.common_types import Config
from MaxText.layers import nnx_wrappers
from MaxText.layers import initializers
from MaxText.layers.linears import DenseGeneral

# pytype: disable=attribute-error


class SimpleDecoderLayer(nnx.Module):
  """Decoder layer consisting of simple matmul."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None = None,
      *,
      rngs: nnx.Rngs, # NNX modules require rngs in __init__
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant

    self.linear = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=config.emb_dim,
        kernel_axes=('embed', 'mlp'),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      positions,
      segmentation,
      deterministic,
      model_mode,
      slot=None,
      previous_chunk=None,
      page_state=None,
  ):
    output = self.linear(inputs)
    if self.config.scan_layers:
      return output, None
    else:
      return output

SimpleDecoderLayerToLinen = nnx_wrappers.to_linen_class(
  SimpleDecoderLayer,
  base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class SimpleMlpDecoderLayer(nnx.Module):
  """Decoder layer consisting of [embed,mlp] followed by an [mlp,embed] matmul."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None = None,
      *,
      rngs: nnx.Rngs, # NNX modules require rngs in __init__
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant

    self.ff_1 = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=config.mlp_dim,
        kernel_axes=('embed', 'mlp'),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

    self.ff_2 = DenseGeneral(
        in_features_shape=config.mlp_dim,
        out_features_shape=config.emb_dim,
        kernel_axes=('mlp', 'embed'),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      positions,
      segmentation,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=0,
  ):
    intermediate = self.ff_1(inputs)
    output = self.ff_2(intermediate)
    if self.config.scan_layers:
      return output, None
    else:
      return output

SimpleMlpDecoderLayerToLinen = nnx_wrappers.to_linen_class(
  SimpleMlpDecoderLayer,
  base_metadata_fn=initializers.variable_to_logically_partitioned,
)
