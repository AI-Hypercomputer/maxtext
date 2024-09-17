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

from typing import Any, Tuple
import dataclasses

from flax import linen as nn
from flax import nnx
from jax import lax
import jax.numpy as jnp
from layers import initializers

Initializer = initializers.Initializer


@dataclasses.dataclass
class RMSNorm(nnx.Module):
  """RMS normalization."""

  features: int 
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[str, ...] = ()
  scale_init: Initializer = nn.initializers.ones
  name: str = "rms_norm"
  rngs: nnx.Rngs | None = None
  
  def __post_init__(self):
    #value = self.scale_init(self.rngs(), (self.features,), self.weight_dtype)
    #self.scale = nnx.Param(value, names=self.kernel_axes)
    self.scale = nnx.Param(
      nnx.with_partitioning(self.scale_init, self.kernel_axes)(
        self.rngs(), (self.features,), self.weight_dtype))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    assert self.features == x.shape[-1], f"{self.features} != {x.shape[-1]}"
    #features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    #scale = self.param(
    #    "scale",
    #    nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
    #    (features,),
    #    self.weight_dtype,
    #)
    scale = jnp.asarray(self.scale.value, self.dtype)
    return y * scale
