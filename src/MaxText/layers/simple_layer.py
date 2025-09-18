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
from MaxText.layers import quantizations, nnx_wrappers
from MaxText.layers.initializers import variable_to_logically_partitioned
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText import max_logging

from typing import Any, Optional
# pytype: disable=attribute-error

class SimpleDecoderLayer(nnx.Module):
  """Decoder layer consisting of a single [embed, embed] weight matrix."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[quantizations.AqtQuantization] = None,
      rngs: Optional[nnx.Rngs] = None,
      weight_dtype: Any = jnp.float32,
  ) -> None:

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.weight_dtype = weight_dtype
    self.model_mode = model_mode
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)

    init_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("embed", "mlp"), mesh=self.mesh)

    self.weights = nnx.Param(init_fn(self.rngs.params(), (self.config.emb_dim, self.config.emb_dim), self.weight_dtype), )

  def __call__(
      self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode, previous_chunk=None, page_state=None
  ):
    if self.config.scan_layers:
      return inputs @ self.weights.astype(inputs.dtype), None
    return inputs @ self.weights.astype(inputs.dtype)

SimpleDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    SimpleDecoderLayer,
    base_metadata_fn=variable_to_logically_partitioned,
)

class SimpleMlpDecoderLayer(nnx.Module):
  """Decoder layer consisting of [embed,mlp] followed by an [mlp,embed] matmul."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[quantizations.AqtQuantization] = None,
      rngs: Optional[nnx.Rngs] = None,
      weight_dtype: Any = jnp.float32,
  ) -> None:

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.weight_dtype = weight_dtype
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)
    self.model_mode = model_mode

    init_ff1_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("embed", "mlp"), mesh=self.mesh)

    self.ff_1 = nnx.Param(init_ff1_fn(self.rngs.params(), (self.config.emb_dim, self.config.mlp_dim), self.weight_dtype), )

    init_ff2_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("mlp", "embed"), mesh=self.mesh)

    self.ff_2 = nnx.Param(init_ff2_fn(self.rngs.params(), (self.config.mlp_dim, self.config.emb_dim), self.weight_dtype), )


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
    intermediate = inputs @ self.ff_1.astype(inputs.dtype)
    output = intermediate @ self.ff_2.astype(inputs.dtype)
    if self.config.scan_layers:
      return output, None
    else:
      return output

SimpleMlpDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    SimpleMlpDecoderLayer,
    base_metadata_fn=variable_to_logically_partitioned,
)