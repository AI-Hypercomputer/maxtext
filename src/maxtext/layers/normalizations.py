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

from typing import Any, Callable, ClassVar

from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from maxtext.common.common_types import Array, DType, ShardMode, is_fp8_dtype
from maxtext.layers.initializers import Initializer
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import truncate_out_sharding


def align_scale_with_normalized_axis(scale: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Reshards a 1-D per-feature scale to match the sharding of the axis it scales.

  A scale is stored with whatever layout its declared axes give it, which is
  picked for the usual case of scaling the embedding axis (`norm` -> `tensor`).
  A per-head QK norm instead normalizes over `head_dim`, which is unsharded
  while the neighbouring `heads` axis is the one on `tensor`; Gemma 4's router
  scale is declared `embed` (-> `fsdp`) but multiplies an activation whose last
  axis is `activation_embed` (-> `tensor`). Multiplying either pair directly
  under `ShardMode.EXPLICIT` puts one mesh axis on two axes of the result, or
  two different axes on the same one, which JAX rejects -- so align the scale
  with the activation instead.

  This is a no-op whenever the two already agree, which includes every
  auto-sharding-equivalent layout for the ordinary layer norms.
  """
  # Read through `.partitions`: a scale carrying a reduced/unreduced tag (the deferred
  # data-parallel all-reduce under gradient accumulation) rejects direct indexing. The tags
  # are carried over to the new spec so the scale keeps its place on the deferred path.
  activation_axis = jax.typeof(y).sharding.spec.partitions[-1]
  scale_spec = jax.typeof(scale).sharding.spec
  if scale_spec.partitions[-1] == activation_axis:
    return scale
  return jax.sharding.reshard(
      scale,
      jax.sharding.PartitionSpec(activation_axis, unreduced=scale_spec.unreduced, reduced=scale_spec.reduced),
  )


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
      scale_init: Initializer = jax.nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      scale_offset: float = 0.0,
      with_scale: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = dtype if is_fp8_dtype(weight_dtype) else weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale_offset = scale_offset
    self.with_scale = with_scale
    if self.with_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), (num_features,), self.weight_dtype),
          out_sharding=kernel_axes,
      )
    else:
      self.scale = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)

    # out_sharding must be None in auto shard mode
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    if out_sharding is not None:
      out_sharding = truncate_out_sharding(out_sharding, y.ndim)

    if not self.with_scale:
      if out_sharding is not None:
        y = jax.lax.with_sharding_constraint(y, out_sharding)
      return y

    scale = self.scale.get_value()
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, max_utils.device_space())

    scale_fp32 = jnp.asarray(scale, jnp.float32)
    effective_scale = scale_fp32 + self.scale_offset
    if self.shard_mode == ShardMode.EXPLICIT:
      effective_scale = align_scale_with_normalized_axis(effective_scale, y)

    if self.scale_offset != 0.0:
      normed_fp32 = x * lax.rsqrt(mean2 + self.epsilon)
      y = jnp.einsum("...k,k->...k", normed_fp32, effective_scale, out_sharding=out_sharding)
      return jnp.asarray(y, self.dtype)

    effective_scale = jnp.asarray(effective_scale, self.dtype)
    return jnp.einsum("...k,k->...k", y, effective_scale, out_sharding=out_sharding)


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


def Qwen3NextRMSNorm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: DType = None,
    weight_dtype: DType = None,
    shard_mode=None,
    kernel_axes=None,
    parameter_memory_host_offload=None,
    *,
    rngs: nnx.Rngs,
):
  """Used for input and post attention layernorms in Qwen3NextDecoderLayer.

  This normalization layer is specific to Qwen3-Next. Key characteristics:
  1. The learnable scale parameter `scale` is initialized to ZEROS.
  2. The scale is applied as `(1.0 + self.scale)`, making the initial scale effectively 1.0.
     This matches the PyTorch implementation of Qwen3NextRMSNorm.
  """
  return nnx.data(
      RMSNorm(
          num_features=num_features,
          epsilon=epsilon,
          dtype=dtype,
          weight_dtype=weight_dtype,
          shard_mode=shard_mode if shard_mode is not None else ShardMode.AUTO,
          kernel_axes=kernel_axes if kernel_axes is not None else (),
          scale_init=jax.nn.initializers.zeros,
          scale_offset=1.0,
          parameter_memory_host_offload=bool(parameter_memory_host_offload),
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

  def __init__(
      self,
      num_features: int,
      epsilon: float,
      dtype: DType,
      weight_dtype: DType,
      shard_mode: ShardMode = ShardMode.AUTO,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = nnx.data(
        RMSNorm(
            num_features=num_features,
            epsilon=self.epsilon,
            dtype=jnp.float32,
            weight_dtype=weight_dtype,
            shard_mode=shard_mode,
            scale_init=nnx.initializers.ones,
            rngs=rngs,
        )
    )

  def __call__(
      self,
      hidden_states: Array,
      gate: Array,
      out_sharding: NamedSharding | None = None,
  ) -> Array:
    """Applies RMSNorm and then a SiLU gate.

    Args:
      hidden_states: The input array to be normalized (o). Shape: (..., F)
      gate: The gating array for the activation (z). Shape: (..., F) where F is
        num_features.
      out_sharding: Optional layout for the normalized states, honoured only
        under `ShardMode.EXPLICIT`.

    Returns:
      The normalized and gated output array. Shape: (..., F)
    """
    normalized_states = self.rms_norm(hidden_states, out_sharding=out_sharding)

    # Gated Activation using SiLU (Sigmoid-weighted Linear Unit)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))

    return gated_states.astype(self.dtype)


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


class _MuseSparkNormBase(nnx.Module):
  """Shared pure-NNX parameter and sharding initializer matching RMSNorm."""

  _SCALE_INIT: ClassVar[Callable[..., jax.Array]] = staticmethod(jax.nn.initializers.ones)

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      with_scale: bool = True,
      shard_mode: ShardMode = ShardMode.AUTO,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    self.num_features, self.epsilon, self.dtype = num_features, epsilon, dtype
    self.weight_dtype = dtype if is_fp8_dtype(weight_dtype) else weight_dtype
    self.kernel_axes, self.with_scale, self.shard_mode = (kernel_axes, with_scale, shard_mode)
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale = (
        nnx.Param(self._SCALE_INIT(rngs.params(), (num_features,), self.weight_dtype), out_sharding=kernel_axes)
        if with_scale
        else None
    )

  def _load_scale(self, compute_dtype: Any, ref_y: jax.Array, offset: float = 0.0) -> jax.Array:
    scale = self.scale.get_value()
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    if offset:
      scale = jnp.asarray(offset, dtype=scale.dtype) + scale
    w = jnp.asarray(scale, compute_dtype)
    return align_scale_with_normalized_axis(w, ref_y) if self.shard_mode == ShardMode.EXPLICIT else w

  def _finalize_out(self, y: jax.Array, out_dtype: Any, out_sharding: NamedSharding | None) -> jax.Array:
    y = jnp.asarray(y, out_dtype)
    if self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None:
      return jax.sharding.reshard(y, truncate_out_sharding(out_sharding, y.ndim))
    return y


class MuseSparkRMSNorm(_MuseSparkNormBase):
  """Muse Spark RMSNorm: y = (x * (mean(x^2) + eps)^(-0.5)) * w."""

  def __call__(self, x: jax.Array, out_sharding: NamedSharding | None = None) -> jax.Array:
    c = jnp.promote_types(x.dtype, jnp.float32)
    xc = jnp.asarray(x, c)
    y = xc * jnp.power(jnp.mean(lax.square(xc), axis=-1, keepdims=True) + self.epsilon, -0.5)
    if self.with_scale and self.scale is not None:
      y = y * self._load_scale(c, y)
    return self._finalize_out(y, x.dtype, out_sharding)


class CenteredRMSNorm(_MuseSparkNormBase):
  """Centered RMSNorm: y = (x * rsqrt(mean(x^2) + eps)) * (1.0 + w)."""

  _SCALE_INIT: ClassVar[Callable[..., jax.Array]] = staticmethod(jax.nn.initializers.zeros)

  def __call__(self, x: jax.Array, out_sharding: NamedSharding | None = None) -> jax.Array:
    c = jnp.promote_types(x.dtype, jnp.float32)
    xc = jnp.asarray(x, c)
    y = xc * lax.rsqrt(jnp.mean(lax.square(xc), axis=-1, keepdims=True) + self.epsilon)
    if self.with_scale and self.scale is not None:
      y = y * self._load_scale(c, y, offset=1.0)
    return self._finalize_out(y, x.dtype, out_sharding)


class InputScaledRMSNorm(_MuseSparkNormBase):
  """InputScaledRMSNorm: scales x by (1 + w), rounds to x.dtype, then normalizes."""

  _SCALE_INIT: ClassVar[Callable[..., jax.Array]] = staticmethod(jax.nn.initializers.zeros)

  def __call__(self, x: jax.Array, out_sharding: NamedSharding | None = None) -> jax.Array:
    c = jnp.promote_types(x.dtype, jnp.float32)
    scaled = (
        jnp.asarray(jnp.asarray(x, c) * self._load_scale(c, x, offset=1.0), x.dtype)
        if (self.with_scale and self.scale is not None)
        else x
    )
    sc = jnp.asarray(scaled, c)
    y = sc * lax.rsqrt(jnp.mean(lax.square(sc), axis=-1, keepdims=True) + self.epsilon)
    return self._finalize_out(y, x.dtype, out_sharding)
