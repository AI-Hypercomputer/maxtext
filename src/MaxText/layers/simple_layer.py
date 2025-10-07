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

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, reshard, NamedSharding, PartitionSpec as P

from flax import linen as nn
from flax import nnx

from MaxText.common_types import Config
from MaxText.layers.linears import DenseGeneral
from MaxText.layers import quantizations
from MaxText.layers import nnx_wrappers
from MaxText.layers import initializers

# pytype: disable=attribute-error


class SimpleDecoderLayer(nnx.Module):
  """Decoder layer rewritten as an nnx.Module."""

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
      self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode, slot=None, previous_chunk=None, page_state=None
  ):
    # inputs_spec = NamedSharding(
    #   self.mesh,
    #   nn.logical_to_mesh_axes(("activation_batch", "activation_norm_length", "activation_embed")),
    # )
    # inputs = reshard(inputs, inputs_spec)
    # kernel_spec = NamedSharding(
    #   self.mesh,
    #   nn.logical_to_mesh_axes(("embed", "mlp")),
    # )
    # self.linear.kernel = reshard(self.linear.kernel, kernel_spec)

    # print(f"Inputs type {jax.typeof(inputs)}; Kernel type {jax.typeof(self.linear.kernel)}")
    output = self.linear(inputs)

    if self.config.scan_layers:
      return output, None
    else:
      return output
    
SimpleDecoderLayerToLinen = nnx_wrappers.to_linen_class(
  SimpleDecoderLayer,
  base_metadata_fn=initializers.variable_to_logically_partitioned,
)


# class SimpleDecoderLayer(nn.Module):
#   """Decoder layer consisting of a single [embed, embed] weight matrix."""

#   config: Config
#   mesh: Mesh
#   model_mode: str
#   quant: None | quantizations.AqtQuantization = None

#   def setup(self):
#     self.weight_mat = self.param(
#         "weights",
#         nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
#         (self.config.emb_dim, self.config.emb_dim),
#     )

#   def __call__(
#       self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode, previous_chunk=None, page_state=None
#   ):
#     shard_spec = NamedSharding(
#       self.mesh,
#       nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_embed")),
#     )
#     if self.config.scan_layers:
#       return reshard(inputs @ self.weight_mat.astype(inputs.dtype), shard_spec), None
#     else:
#       return reshard(inputs @ self.weight_mat.astype(inputs.dtype), shard_spec)
    



class SimpleMlpDecoderLayer(nn.Module):
  """Decoder layer consisting of [embed,mlp] followed by an [mlp,embed] matmul."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | quantizations.AqtQuantization = None

  def setup(self):
    self.ff_1 = self.param(
        "ff_1",
        nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        (self.config.emb_dim, self.config.mlp_dim),
    )
    self.ff_2 = self.param(
        "ff_2",
        nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", "embed")),
        (self.config.mlp_dim, self.config.emb_dim),
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
    intermediate = inputs @ self.ff_1.astype(inputs.dtype)
    output = intermediate @ self.ff_2.astype(inputs.dtype)
    shard_spec = NamedSharding(
        self.mesh,
        nn.logical_to_mesh_axes(("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_embed")),
      )
    if self.config.scan_layers:
      return reshard(output, shard_spec), None
    else:
      return reshard(output, shard_spec)
