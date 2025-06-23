"""
Copyright 2023 Google LLC

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

"""GPT3 specific layers. Pulled out of gpt3.py to avoid circular imports.
Specifically, linears.py needs to use Gp3LayerNorm formerly from gpt3.py,
but gpt3.py needs to use MlpBlock, from linears.py """
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import lax

from flax import linen as nn

from MaxText import max_logging
from MaxText.layers import initializers
from MaxText.layers.initializers import Initializer

# -----------------------------------------
# The Normalization Layer specific for GPT3
# -----------------------------------------


class Gpt3LayerNorm(nn.Module):
  """GPT3 Layer normalization operating on the last axis of the input data."""

  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  kernel_axes: Tuple[Optional[str], ...] = ()
  scale_init: Initializer = nn.initializers.zeros
  use_bias: bool = True
  reductions_in_fp32: bool = False
  parameter_memory_host_offload: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    if self.reductions_in_fp32:
      x = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x, axis=[-1], keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=[-1], keepdims=True)
    normed_inputs = (x - mean) * lax.rsqrt(var + self.epsilon)
    if self.reductions_in_fp32:
      normed_inputs = normed_inputs.astype(self.dtype)

    features = x.shape[-1]
    scale = self.param(
        "scale", nn.with_logical_partitioning(self.scale_init, self.kernel_axes), (features,), self.weight_dtype
    )
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("gpt3_layers.py: Moving scale parameter to device")
      scale = jax.device_put(scale, jax._src.sharding_impls.TransferToMemoryKind("device"))  # pylint: disable=protected-access

    scale = jnp.asarray(scale, self.dtype)
    output = normed_inputs * (scale + 1)

    if self.use_bias:
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(initializers.default_bias_init, self.kernel_axes),
          (features,),
          self.weight_dtype,
      )
      bias = jnp.asarray(bias, self.dtype)
      output += bias
    return output
