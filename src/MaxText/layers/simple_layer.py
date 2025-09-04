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

from flax import linen as nn

from MaxText.common_types import Config
from MaxText.layers import quantizations

# pytype: disable=attribute-error


class SimpleDecoderLayer(nn.Module):
  """Decoder layer consisting of a single [embed, embed] weight matrix."""

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | quantizations.AqtQuantization = None

  def setup(self):
    self.weight_mat = self.param(
        "weights",
        nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        (self.config.emb_dim, self.config.emb_dim),
    )

  def __call__(
      self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode, previous_chunk=None, page_state=None
  ):
    if self.config.scan_layers:
      return inputs @ self.weight_mat.astype(inputs.dtype), None
    else:
      return inputs @ self.weight_mat.astype(inputs.dtype)


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
    if self.config.scan_layers:
      return output, None
    else:
      return output
