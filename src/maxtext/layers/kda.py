# Copyright 2026 Google LLC
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

"""Kimi Decoupled Attention (KDA) layer for Kimi K3 in MaxText (NNX)."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from maxtext.common.common_types import Config, DType
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm


def kda_recurrent_kernel(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Pure JAX KDA recurrent kernel for autoregressive decoding and sequence processing.

  Args:
    q: [B, T, H, K] - Queries
    k: [B, T, H, K] - Keys
    v: [B, T, HV, V] - Values
    g: [B, T, HV, K] - Decay gates in log-space (<= 0)
    beta: [B, T, HV] - Beta scalars
    scale: Optional scale factor (defaults to 1 / sqrt(K))
    initial_state: Optional initial state [B, HV, K, V]

  Returns:
    o: [B, T, HV, V] - Output tensor
    S_final: [B, HV, K, V] - Final recurrent state
  """
  B, T, H, K = q.shape
  HV, V = v.shape[2], v.shape[3]
  G = HV // H
  if scale is None:
    scale = K**-0.5

  # Repeat interleave q, k to HV if HV != H
  if G > 1:
    q = jnp.repeat(q, G, axis=2)
    k = jnp.repeat(k, G, axis=2)

  q = (q * scale).astype(jnp.float32)
  k = k.astype(jnp.float32)
  v = v.astype(jnp.float32)
  g = g.astype(jnp.float32)
  beta = beta.astype(jnp.float32)

  if initial_state is None:
    S_init = jnp.zeros((B, HV, K, V), dtype=jnp.float32)
  else:
    S_init = initial_state.astype(jnp.float32)

  # Transpose to (T, B, HV, ...) for jax.lax.scan
  q_t = jnp.transpose(q, (1, 0, 2, 3))
  k_t = jnp.transpose(k, (1, 0, 2, 3))
  v_t = jnp.transpose(v, (1, 0, 2, 3))
  g_t = jnp.transpose(g, (1, 0, 2, 3))
  beta_t = jnp.transpose(beta, (1, 0, 2))

  def scan_fn(S, xs):
    q_i, k_i, v_i, g_i, b_i = xs
    # Decay state: g_i is <= 0 in log space, so exp(g_i) is in (0, 1]
    S = S * jnp.exp(g_i[..., None])

    # Compute k_i^T @ S -> [B, HV, V]
    k_S = jnp.sum(k_i[..., None] * S, axis=-2)

    # Compute v_diff = v_i - k_S
    v_diff = v_i - k_S

    # Compute bk = beta_i * k_i -> [B, HV, K]
    bk = b_i[..., None] * k_i

    # Update state: S += bk ^ T @ v_diff
    S = S + bk[..., None] * v_diff[..., None, :]

    # Compute output: o_i = q_i ^ T @ S -> [B, HV, V]
    o_i = jnp.sum(q_i[..., None] * S, axis=-2)
    return S, o_i

  S_final, o_t = jax.lax.scan(scan_fn, S_init, (q_t, k_t, v_t, g_t, beta_t))
  o = jnp.transpose(o_t, (1, 0, 2, 3))
  return o.astype(v.dtype), S_final


class ShortConv1D(nnx.Module):
  """1D Short Convolution with SiLU activation for KDA."""

  def __init__(
      self,
      features: int,
      kernel_size: int = 4,
      *,
      rngs: nnx.Rngs,
  ):
    self.features = features
    self.kernel_size = kernel_size
    # Weight shape: [kernel_size, features] (depthwise 1D conv)
    self.weight = nnx.Param(
        jax.random.normal(rngs.params(), (kernel_size, features)) * 0.02
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """x: [B, T, features] -> [B, T, features]"""
    # Depthwise 1D conv along sequence dimension T
    # Pad left by (kernel_size - 1) to maintain causal alignment
    padded = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
    # Padded shape: [B, T + kernel_size - 1, features]
    # We use jax.lax.conv_general_dilated for depthwise 1D conv:
    # lhs: [B, features, T_padded], rhs: [features, 1, kernel_size]
    lhs = jnp.transpose(padded, (0, 2, 1)) # [B, features, T_padded]
    rhs = jnp.transpose(self.weight[...], (1, 0))[:, None, :] # [features, 1, kernel_size]


    out = jax.lax.conv_general_dilated(
        lhs=lhs,
        rhs=rhs,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=self.features,
    ) # [B, features, T]

    out = jnp.transpose(out, (0, 2, 1)) # [B, T, features]
    return jax.nn.silu(out)


class KimiDecoupledAttention(nnx.Module):
  """Kimi Decoupled Attention (KDA) layer for Kimi K3."""

  def __init__(
      self,
      config: Config,
      layer_idx: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config.emb_dim
    self.num_heads = config.num_query_heads
    self.head_dim = config.head_dim
    self.conv_kernel_size = config.kda_conv_kernel_size
    self.use_full_rank_gate = config.kda_use_full_rank_gate
    self.gate_lower_bound = config.kda_gate_lower_bound

    projection_size = self.num_heads * self.head_dim

    # Projections for Q, K, V
    self.q_proj = DenseGeneral(
        self.hidden_size,
        projection_size,
        use_bias=False,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        rngs=rngs,
    )
    self.k_proj = DenseGeneral(
        self.hidden_size,
        projection_size,
        use_bias=False,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        rngs=rngs,
    )
    self.v_proj = DenseGeneral(
        self.hidden_size,
        projection_size,
        use_bias=False,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        rngs=rngs,
    )

    # 1D Short Convolutions
    self.q_conv1d = ShortConv1D(projection_size, self.conv_kernel_size, rngs=rngs)
    self.k_conv1d = ShortConv1D(projection_size, self.conv_kernel_size, rngs=rngs)
    self.v_conv1d = ShortConv1D(projection_size, self.conv_kernel_size, rngs=rngs)

    # Gate & Beta Projections
    self.f_a_proj = DenseGeneral(
        self.hidden_size,
        self.head_dim,
        use_bias=False,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        rngs=rngs,
    )
    self.f_b_proj = DenseGeneral(
        self.head_dim,
        projection_size,
        use_bias=False,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        rngs=rngs,
    )
    self.b_proj = DenseGeneral(
        self.hidden_size,
        self.num_heads,
        use_bias=False,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        rngs=rngs,
    )

    # Parameters: A_log & dt_bias
    # A_log is initialized uniformly in [1, 16] and stored as log
    a_init = jax.random.uniform(rngs.params(), (self.num_heads,), minval=1.0, maxval=16.0)
    self.A_log = nnx.Param(jnp.log(a_init))
    self.dt_bias = nnx.Param(jnp.zeros((projection_size,)))

    # Output gate projection
    if self.use_full_rank_gate:
      self.g_proj = DenseGeneral(
          self.hidden_size,
          projection_size,
          use_bias=False,
          kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
          rngs=rngs,
      )
    else:
      self.g_a_proj = DenseGeneral(
          self.hidden_size,
          self.head_dim,
          use_bias=False,
          kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
          rngs=rngs,
      )
      self.g_b_proj = DenseGeneral(
          self.head_dim,
          projection_size,
          use_bias=False,
          kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
          rngs=rngs,
      )

    # Output Norm & Projection
    self.o_norm = RMSNorm(
        self.head_dim,
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )
    self.o_proj = DenseGeneral(
        projection_size,
        self.hidden_size,
        use_bias=False,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      *,
      initial_state: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """hidden_states: [B, T, hidden_size] -> [B, T, hidden_size], final_state"""
    B, T, _ = hidden_states.shape

    # 1. Projections & 1D Convolutions
    q = self.q_conv1d(self.q_proj(hidden_states))
    k = self.k_conv1d(self.k_proj(hidden_states))
    v = self.v_conv1d(self.v_proj(hidden_states))

    # 2. Reshape to [B, T, H, D]
    q = q.reshape(B, T, self.num_heads, self.head_dim)
    k = k.reshape(B, T, self.num_heads, self.head_dim)
    v = v.reshape(B, T, self.num_heads, self.head_dim)

    # 3. L2-normalize q and k along head_dim
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-6)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True).clip(min=1e-6)

    # 4. Gate & Beta computation
    # g_raw: [B, T, H, D]
    g_raw = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(B, T, self.num_heads, self.head_dim)
    dt_bias = self.dt_bias[...].reshape(1, 1, self.num_heads, self.head_dim)

    # decay = -exp(A_log) * softplus(g_raw + dt_bias) <= 0
    A_log = self.A_log[...].reshape(1, 1, self.num_heads, 1)
    decay = -jnp.exp(A_log) * jax.nn.softplus(g_raw + dt_bias)


    if self.gate_lower_bound is not None:
      decay = jnp.maximum(decay, self.gate_lower_bound)

    # beta: [B, T, H] -> sigmoid(beta)
    beta = jax.nn.sigmoid(self.b_proj(hidden_states))

    # 5. KDA Recurrent Kernel
    o, final_state = kda_recurrent_kernel(
        q=q,
        k=k,
        v=v,
        g=decay,
        beta=beta,
        initial_state=initial_state,
    ) # o: [B, T, H, D]

    # 6. Output Gate & Norm
    if self.use_full_rank_gate:
      g_out = self.g_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim)
    else:
      g_out = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(B, T, self.num_heads, self.head_dim)

    # FusedRMSNormGated: RMSNorm(o) * sigmoid(g_out)
    o = self.o_norm(o) * jax.nn.sigmoid(g_out)

    # 7. Output Projection
    o = o.reshape(B, T, self.num_heads * self.head_dim)
    o = self.o_proj(o)

    return o, final_state
