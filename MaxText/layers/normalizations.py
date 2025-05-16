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
from jax import lax
import jax
import jax.numpy as jnp
from MaxText import max_logging
from MaxText.layers.initializers import Initializer


class RMSNorm(nn.Module):
  """RMS normalization."""

  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[Optional[str], ...] = ()
  scale_init: Initializer = nn.initializers.ones
  parameter_memory_host_offload: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.param(
        "scale",
        nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
        (features,),
        self.weight_dtype,
    )
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, jax._src.sharding_impls.TransferToMemoryKind("device"))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale
