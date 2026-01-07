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

""" Simple decoder layers for testing and debugging purposes."""

from jax import numpy as jnp
from jax.sharding import Mesh

from flax import nnx
from MaxText.common_types import Config, ShardMode
from MaxText.sharding import create_sharding
from MaxText.layers import quantizations, nnx_wrappers
from MaxText.layers.initializers import variable_to_logically_partitioned


from typing import Optional
# pytype: disable=attribute-error


class SimpleDecoderLayer(nnx.Module):
  """Decoder layer consisting of a single [embed, embed] weight matrix."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ) -> None:

    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.quant = quant

    init_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("embed", "mlp"), mesh=self.mesh)

    self.weights = nnx.Param(
        init_fn(self.rngs.params(), (self.config.emb_dim, self.config.emb_dim)),
    )

    activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.out_sharding = (
        create_sharding(self.mesh, activation_axis_names) if config.shard_mode == ShardMode.EXPLICIT else None
    )

  def __call__(
      self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode, previous_chunk=None, page_state=None
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    if self.config.scan_layers:
      return jnp.dot(inputs, self.weights.astype(inputs.dtype), out_sharding=self.out_sharding), None
    return jnp.dot(inputs, self.weights.astype(inputs.dtype), out_sharding=self.out_sharding)


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
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ) -> None:

    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.quant = quant

    init_ff1_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("embed", "mlp"), mesh=self.mesh)

    self.ff_1 = nnx.Param(
        init_ff1_fn(self.rngs.params(), (self.config.emb_dim, self.config.mlp_dim)),
    )

    init_ff2_fn = nnx.with_partitioning(nnx.initializers.lecun_normal(), sharding=("mlp", "embed"), mesh=self.mesh)

    self.ff_2 = nnx.Param(
        init_ff2_fn(self.rngs.params(), (self.config.mlp_dim, self.config.emb_dim)),
    )

    activation_axes_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.activation_sharding = (
        create_sharding(mesh, activation_axes_names) if config.shard_mode == ShardMode.EXPLICIT else None
    )
    mlp_axes_names = ("activation_batch", "activation_norm_length", "activation_mlp")
    self.mlp_sharding = create_sharding(mesh, mlp_axes_names) if config.shard_mode == ShardMode.EXPLICIT else None

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
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    intermediate = jnp.dot(inputs, self.ff_1.astype(inputs.dtype), out_sharding=self.mlp_sharding)
    output = jnp.dot(intermediate, self.ff_2.astype(inputs.dtype), out_sharding=self.activation_sharding)
    if self.config.scan_layers:
      return output, None
    return output


SimpleMlpDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    SimpleMlpDecoderLayer,
    base_metadata_fn=variable_to_logically_partitioned,
)
