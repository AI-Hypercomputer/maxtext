#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Normalization Layers."""

from typing import Any, Tuple, Optional

from flax import linen as nn
from flax import nnx
from jax import lax
import jax.numpy as jnp
from MaxText.layers import initializers

Initializer = initializers.Initializer


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
      self,
      features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: Tuple[Optional[str], ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      *,
      rngs: nnx.Rngs
  ):
    self.features = features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init

    self.scale = nnx.Param(
        scale_init(rngs.params(), (features,), weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.scale.value
    scale = jnp.asarray(scale, self.dtype)
    return y * scale
  

def rms_norm(
    features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: Tuple[Optional[str], ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: Optional[str] = None,
):
  module = nnx.bridge.to_linen(
      RMSNorm,
      features=features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      name=name,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
  return module