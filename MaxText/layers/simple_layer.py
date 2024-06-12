"""
Copyright 2024 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" A simple decoder layer consisting of a single [embed, embed] weight matrix for testing and debugging purposes."""

from jax import numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh
from typing import Optional
from layers import quantizations
import common_types

# pytype: disable=attribute-error

class SimpleDecoderLayer(nn.Module):
  config: common_types.Config
  mesh: Mesh
  quant: Optional[quantizations.AqtQuantization] = None

  def setup(self):
    self.weight_mat = self.param(
      'weights',
      nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
      (self.config.emb_dim, self.config.emb_dim)
    )

  def __call__(self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode):
    if self.config.scan_layers:
      return inputs @ self.weight_mat.astype(inputs.dtype), None
    else:
      return inputs @ self.weight_mat.astype(inputs.dtype)
