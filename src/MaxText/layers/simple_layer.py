# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple decoder layer used for testing."""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from flax import linen as nn
from flax import nnx
import jax.numpy as jnp
from MaxText.common_types import Config, MODEL_MODE_TRAIN
from MaxText.layers import initializers


class SimpleDecoderLayer(nn.Module):
  """A simple decoder layer used for testing."""

  config: Config
  mesh: Mesh
  model_mode: str = MODEL_MODE_TRAIN

  @nn.compact
  def __call__(self, inputs, inputs_position, inputs_segmentation, deterministic=False, model_mode=MODEL_MODE_TRAIN):
    """Applies SimpleDecoderLayer module."""
    del inputs_position, inputs_segmentation, deterministic, model_mode
    wo = self.param(
        "wo",
        nn.with_logical_partitioning(nn.initializers.normal(0.02), ("mlp", "embed"), None),
        (self.config.base_mlp_dim, self.config.base_emb_dim),
    )
    x = nn.dot_general(inputs, wo, (((inputs.ndim - 1),), ((wo.ndim - 1),)))
    return x, None


class SimpleDecoderLayerNnx(nnx.Module):
  """An NNX version of a simple decoder layer used for testing."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    self.wo = nnx.Param(
        jax.random.normal(rngs.params(), (self.config.base_emb_dim, self.config.base_emb_dim)) * 0.02,
        sharding=PartitionSpec("mlp", "embed"),
    )

  def __call__(self, inputs, inputs_position, inputs_segmentation, deterministic=False, model_mode=MODEL_MODE_TRAIN):
    """Applies SimpleDecoderLayer module."""
    del inputs_position, inputs_segmentation, deterministic, model_mode
    x = jax.lax.dot_general(
        inputs,
        self.wo.value,
        (((inputs.ndim - 1,), (0,)), ((), ())),  # Contract last dim of input with first dim of weights
    )
    return x, None
