# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.8-Flash-Next model in m3 format.

Self-contained single-file implementation following the m3 design
(go/maxtext-m3-design): the model owns every component of its architecture,
composed from Flax NNX primitives, with no shared decoder or attention class.

Sections (in dependency order):
  * Normalization layers and 4-stream gated residual (hyper-connection)
"""

from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


# =================================================================================================
# Normalization layers and 4-stream gated residual (hyper-connection)
# =================================================================================================


class Qwen3_8RMSNorm(nnx.Module):
  """RMSNorm with zero-centered scale (1.0 + scale) and optional group size."""

  def __init__(
      self,
      dim: int,
      group_size: int | None = None,
      eps: float = 1e-6,
      *,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.dim, self.group_size, self.eps = dim, group_size, eps
    if group_size is not None and dim % group_size != 0:
      raise ValueError(f"dim ({dim}) not divisible by group_size ({group_size}).")
    self.scale = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))

  def _norm(self, x: jax.Array) -> jax.Array:
    if self.group_size is not None:
      shape = x.shape
      x = x.reshape(*shape[:-1], -1, self.group_size)
      var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
      return (x * jax.lax.rsqrt(var + self.eps)).reshape(shape)
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)

  def __call__(self, x: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    scale_f32 = 1.0 + self.scale[...].astype(jnp.float32)
    return (self._norm(x_f32) * scale_f32).astype(x.dtype)


class Qwen3_8RMSNormGated(nnx.Module):
  """Gated RMSNorm with standard scale (init ones) and configurable gate activation."""

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      activation: str = "silu",
      *,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.dim, self.eps = dim, eps
    self.activation = activation
    if activation == "silu":
      self._act_fn = jax.nn.silu
    elif activation == "sigmoid":
      self._act_fn = jax.nn.sigmoid
    else:
      raise ValueError(f"Unsupported activation: {activation}. Expected 'silu' or 'sigmoid'.")
    self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

  def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)
    hidden = self.scale[...] * x_norm.astype(orig_dtype)
    gate_act = self._act_fn(gate.astype(jnp.float32))
    return (hidden * gate_act).astype(orig_dtype)


def l2norm(x: jax.Array, dim: int = -1, eps: float = 1e-6) -> jax.Array:
  """L2 normalization along a dimension aligning with FLA library."""
  x_f32 = x.astype(jnp.float32)
  return (x_f32 * jax.lax.rsqrt(jnp.sum(jnp.square(x_f32), axis=dim, keepdims=True) + eps)).astype(x.dtype)


def init_hyper_connection_streams(token_embeddings: jax.Array, hc_count: int = 4) -> jax.Array:
  """Initializes residual stream by replicating token embeddings across streams."""
  return jnp.concatenate([token_embeddings] * hc_count, axis=-1)


class Qwen3_8GatedResidual(nnx.Module):
  """4-stream Gated Residual (Hyper-Connection) module."""

  def __init__(
      self,
      emb_dim: int,
      hc_count: int = 4,
      hc_lowrank: int = 320,
      *,
      use_combine: bool = True,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.emb_dim, self.hc_count = emb_dim, hc_count
    self.hc_lowrank, self.use_combine = hc_lowrank, use_combine
    self.dtype, self.weight_dtype = dtype, weight_dtype

    hc_dim = hc_count * emb_dim
    self.hc_norm = Qwen3_8RMSNorm(hc_dim, group_size=emb_dim, eps=eps, rngs=rngs)
    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.input_mix_weight_down = nnx.Linear(hc_dim, hc_lowrank, **kw)
    self.input_mix_weight_up = nnx.Linear(hc_lowrank, hc_dim, **kw)
    self.block_inject_weight = nnx.Linear(hc_dim, hc_count, **kw) if use_combine else None

  def _mix(self, normed: jax.Array, out_dtype: Any) -> jax.Array:
    w_down = jax.nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
    w_up = jax.nn.sigmoid(self.input_mix_weight_up(w_down))
    shape = w_up.shape[:-1]
    w_up_streams = w_up.reshape(*shape, self.hc_count, self.emb_dim)
    normed_streams = normed.reshape(*shape, self.hc_count, self.emb_dim)
    return jnp.mean(w_up_streams * normed_streams, axis=-2).astype(out_dtype)

  def mix_input(self, hyper_input: jax.Array) -> jax.Array:
    """Mixes multiple residual streams into a single representation."""
    if hyper_input.shape[-1] != self.hc_count * self.emb_dim:
      raise ValueError(f"Expected {self.hc_count * self.emb_dim} features, got {hyper_input.shape[-1]}.")
    return self._mix(self.hc_norm(hyper_input), hyper_input.dtype)

  def before_block(self, hyper_input: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Pre-block transformation computing mixed input and injection weights."""
    if hyper_input.shape[-1] != self.hc_count * self.emb_dim:
      raise ValueError(f"Expected {self.hc_count * self.emb_dim} features, got {hyper_input.shape[-1]}.")
    if self.block_inject_weight is None:
      raise ValueError("block_inject_weight is None (use_combine=False).")
    normed = self.hc_norm(hyper_input)
    inject = (2.0 * jax.nn.sigmoid(self.block_inject_weight(normed) / self.hc_count)).astype(hyper_input.dtype)
    return self._mix(normed, hyper_input.dtype), inject

  def combine(
      self,
      hyper_input: jax.Array,
      block_output: jax.Array,
      injection_weights: jax.Array,
  ) -> jax.Array:
    """Injects block output into hyper-connection residual streams."""
    inj = jnp.expand_dims(block_output, axis=-2) * jnp.expand_dims(injection_weights, axis=-1)
    return (hyper_input + inj.reshape(*hyper_input.shape)).astype(hyper_input.dtype)

  def apply_block(
      self,
      hyper_input: jax.Array,
      block_fn: Callable[[jax.Array], jax.Array],
  ) -> jax.Array:
    """Executes before_block, applies block_fn, and combines.

    Args:
      hyper_input: Multi-stream residual representations with shape `(..., hc_count * emb_dim)`.
      block_fn: Callable mapping the mixed pre-block representation `(..., emb_dim)` to the
        single-stream block output `(..., emb_dim)`.

    Returns:
      The combined output `combine(hyper_input, block_fn(mixed), injection_weights)` with shape
      `(..., hc_count * emb_dim)`.
    """
    mixed, inject_gate = self.before_block(hyper_input)
    return self.combine(hyper_input, block_fn(mixed), inject_gate)
