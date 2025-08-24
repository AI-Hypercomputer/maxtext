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

"""Normalization Layers."""

from typing import Any, Tuple, Optional

from flax import linen as nn
from flax import nnx
from jax import lax
import jax
import jax.numpy as jnp
from flax import nnx
from MaxText import max_logging
from MaxText import max_utils
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import Initializer, variable_to_logically_partitioned
from MaxText.sharding import MeshSharding, LogicalAxisRulesSharding


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: Tuple[Optional[str], ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
      sharding: MeshSharding | None = None,
      tensor_name: str = "rms_norm"
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.tensor_name = tensor_name
    self.sharding = sharding if sharding else LogicalAxisRulesSharding()

    self.scale = nnx.Param(
        scale_init(rngs.params(), (num_features,), weight_dtype),
        sharding=self.sharding(t=self.tensor_name, a=kernel_axes),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.scale.value
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


def rms_norm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: Tuple[Optional[str], ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: Optional[str] = None,
    parameter_memory_host_offload: bool = False,
    sharding: MeshSharding | None = None,
):
  """Creates a RMSNorm module."""
  module = nnx_wrappers.to_linen(
      RMSNorm,
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      sharding=sharding,
      tensor_name=name,
  )
  return module
