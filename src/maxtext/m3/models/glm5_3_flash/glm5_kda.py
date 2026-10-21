# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Primitives for Kimi Delta Attention (KDA) in GLM-5.3-Flash."""

import math

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from maxtext.m3.core import sharding as m3_sharding


def l2norm(
    x: jax.Array,
    axis: int = -1,
    eps: float = 1e-6,
) -> jax.Array:
  """L2 normalization along a specified axis: x / sqrt(sum(x^2) + eps).

  Aligns with the l2norm implementation in the FLA library and GLM-5.3-Flash HF
  reference, which uses `+ eps` inside the sqrt rather than `max(norm, eps)`.

  Args:
    x: Input array.
    axis: Axis along which to compute the L2 norm (default -1).
    eps: Epsilon value added inside the square root (default 1e-6).

  Returns:
    Normalized array with the same shape and dtype as x.
  """
  input_dtype = x.dtype
  acc_dtype = jnp.promote_types(input_dtype, jnp.float32)
  x_acc = x.astype(acc_dtype)
  norm = jnp.sqrt(jnp.sum(x_acc * x_acc, axis=axis, keepdims=True) + eps)
  out = x_acc / norm
  return out.astype(input_dtype)


class Glm5RMSNormGated(nnx.Module):
  """RMSNorm with sigmoid gating: (weight * normed) * sigmoid(gate)."""

  def __init__(
      self,
      hidden_size: int,
      eps: float = 1e-5,
      *,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.hidden_size = hidden_size
    self.eps = eps
    self.param_dtype = param_dtype
    self.sharding = sharding

    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(
        jnp.ones((hidden_size,), dtype=param_dtype),
        out_sharding=spec,
    )

  def __call__(self, hidden_states: jax.Array, gate: jax.Array) -> jax.Array:
    """Forward pass for gated RMSNorm.

    Args:
      hidden_states: Input tensor of shape [..., hidden_size].
      gate: Gate tensor of same shape as hidden_states.

    Returns:
      Gated and normalized tensor of shape [..., hidden_size].
    """
    input_dtype = hidden_states.dtype
    h_fp32 = hidden_states.astype(jnp.float32)
    gate_fp32 = gate.astype(jnp.float32)

    variance = jnp.mean(jnp.square(h_fp32), axis=-1, keepdims=True)
    normed = h_fp32 * jax.lax.rsqrt(variance + self.eps)
    w = self.weight[...].astype(jnp.float32)
    out = (w * normed) * jax.nn.sigmoid(gate_fp32)
    return out.astype(input_dtype)


class Glm5ShortConv1d(nnx.Module):
  """1D causal depthwise convolution with SiLU activation."""

  def __init__(
      self,
      dim: int,
      kernel_size: int = 4,
      *,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs | None = None,
  ):
    if rngs is None:
      raise ValueError("Glm5ShortConv1d requires rngs for weight initialization.")
    self.dim = dim
    self.kernel_size = kernel_size
    self.param_dtype = param_dtype
    self.sharding = sharding

    spec = sharding("conv1d", ("heads", None, None)) if sharding else None
    bound = 1.0 / math.sqrt(kernel_size)
    w_init = jax.random.uniform(
        rngs.params(),
        (dim, 1, kernel_size),
        dtype=jnp.float32,
        minval=-bound,
        maxval=bound,
    ).astype(param_dtype)
    self.weight = nnx.Param(w_init, out_sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass for causal depthwise 1D conv + SiLU.

    Args:
      x: Input tensor of shape [batch, seq_len, dim].

    Returns:
      Output tensor of shape [batch, seq_len, dim].
    """
    input_dtype = x.dtype
    b, s, _ = x.shape
    k = self.kernel_size
    x_fp32 = x.astype(jnp.float32)
    pad = jnp.zeros((b, k - 1, self.dim), dtype=jnp.float32)
    x_padded = jnp.concatenate([pad, x_fp32], axis=1)
    w_fp32 = self.weight[...].astype(jnp.float32)
    out = sum(x_padded[:, i : i + s, :] * w_fp32[:, 0, i] for i in range(k))
    return jax.nn.silu(out).astype(input_dtype)
