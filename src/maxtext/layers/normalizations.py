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

"""Normalization Layers."""

from typing import Any

from flax import linen as nn
from flax import nnx
from flax.linen import initializers as linen_initializers
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from MaxText.common_types import Array, DType, ShardMode
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import Initializer, variable_to_logically_partitioned
from maxtext.utils import max_logging
from maxtext.utils import max_utils


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      scale_offset: float = 0.0,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale_offset = scale_offset
    self.scale = nnx.Param(
        scale_init(rngs.params(), (num_features,), weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.scale.value
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())
    # out_sharding must be None in auto shard mode
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    scale = jnp.asarray(scale, self.dtype)
    effective_scale = scale + self.scale_offset  # Apply offset
    return jnp.einsum("i...k,...k->i...k", y, effective_scale, out_sharding=out_sharding)


class GlobalRMSNorm(RMSNorm):
  """
  Applies RMSNorm over the last two dimensions (Heads * HeadDim).
  Used for Olmo3 which normalizes across all heads combined.
  """

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    # x shape: [..., Heads, HeadDim]
    input_shape = x.shape

    # Flatten the last two dimensions: [..., Heads * HeadDim]
    # We use -2 and -1 to ensure we capture the last two dims regardless of rank
    flattened_shape = input_shape[:-2] + (input_shape[-2] * input_shape[-1],)
    x_flat = x.reshape(flattened_shape)

    # Apply standard RMSNorm (which normalizes over the last axis)
    y_flat = super().__call__(x_flat, out_sharding)

    # Reshape back to [..., Heads, HeadDim]
    return y_flat.reshape(input_shape)


def Qwen3NextRMSNorm(num_features: int, eps: float, dtype: DType, weight_dtype: DType, *, rngs: nnx.Rngs):
  """
  Used for input and post attention layernorms
  in Qwen3NextDecoderLayer.

  This normalization layer is specific to Qwen3-Next. Key characteristics:
  1.  The learnable scale parameter `scale` is initialized to ZEROS.
  2.  The scale is applied as `(1.0 + self.scale)`, making the initial scale effectively 1.0.
      This matches the PyTorch implementation of Qwen3NextRMSNorm.
  """
  return nnx.data(
      RMSNorm(
          num_features=num_features,
          epsilon=eps,
          dtype=dtype,
          weight_dtype=weight_dtype,
          scale_init=linen_initializers.zeros,
          scale_offset=1.0,
          rngs=rngs,
      )
  )


class Qwen3NextRMSNormGated(nnx.Module):
  """
  This applies RMS Normalization and then a gated activation function (SiLU).
  This is used within the Qwen3NextGatedDeltaNet.

  The normalization is performed by an internal `RMSNorm` instance (`self.rms_norm`),
  which has its own learnable `scale` parameter, initialized to ONES.

  Attributes:
    num_features: The number of features in the input.
    eps: A small epsilon value to prevent division by zero in RMSNorm.
    dtype: The datatype of the computation.
    weight_dtype: The datatype of the internal RMSNorm scale.
  """

  def __init__(self, num_features: int, eps: float, dtype: DType, weight_dtype: DType, *, rngs: nnx.Rngs):
    self.num_features = num_features
    self.eps = eps
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = nnx.data(
        RMSNorm(
            num_features=num_features,
            epsilon=eps,
            dtype=dtype,
            weight_dtype=weight_dtype,
            scale_init=nnx.initializers.ones,
            rngs=rngs,
        )
    )

  def __call__(self, hidden_states: Array, gate: Array) -> Array:
    """
    Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized (o). Shape: (..., F)
      gate: The gating array for the activation (z). Shape: (..., F)
            where F is num_features.

    Returns:
      The normalized and gated output array. Shape: (..., F)
    """
    normalized_states = self.rms_norm(hidden_states)

    # Gated Activation using SiLU (Sigmoid-weighted Linear Unit)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


def rms_norm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    shard_mode: ShardMode = ShardMode.AUTO,
    kernel_axes: tuple[None | str, ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: None | str = None,
    parameter_memory_host_offload: bool = False,
):
  """Creates a RMSNorm module."""
  module = nnx_wrappers.to_linen(
      RMSNorm,
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      shard_mode=shard_mode,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  """L2 normalization function. Normalizes a vector to have a length of 1.

  Args:
    x: Input array.
    dim: The axis or axes along which to normalize. Defaults to the last axis.
    eps: Small epsilon to prevent division by zero.

  Returns:
    L2 normalized array with the same shape as x.
  """

  inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))
  return x * inv_norm


Qwen3NextRMSNormLinen = nnx_wrappers.to_linen_class(
    RMSNorm,
    base_metadata_fn=variable_to_logically_partitioned,
    scale_init=linen_initializers.zeros,
    scale_offset=1.0,
)
