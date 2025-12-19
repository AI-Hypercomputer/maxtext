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

"""Kimi Delta Attention Layer."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from MaxText.common_types import (
    Array,
    Config,
    DType,
)
from MaxText.layers.initializers import (
    nd_dense_init,
    NdInitializer,
    default_bias_init,
)
from MaxText.layers.linears import DenseGeneral
from MaxText.layers.normalizations import l2norm, RMSNorm


def chunk_parallel_delta_attention(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    initial_state: Optional[Array] = None,
    scale: Optional[float] = None,
    output_final_state: bool = False,
) -> Tuple[Array, Optional[Array]]:
  """Interface for Kimi Delta Attention chunk-parallel computation.
  Returns dummy all-ones matrix for integration testing.
  """
  batch, seq_len, num_heads, head_dim_k = q.shape
  head_dim_v = v.shape[-1]
  
  # Return ones with shape [B, T, H, V]
  output = jnp.ones((batch, seq_len, num_heads, head_dim_v), dtype=q.dtype)
  
  final_state = None
  if output_final_state:
    # Final state shape is usually [B, H, Dv, Dk]
    final_state = jnp.ones((batch, num_heads, head_dim_v, head_dim_k), dtype=q.dtype)
    
  return output, final_state



class FusedRMSNormGated(nnx.Module):
  """Fused RMSNorm with gating, matching Kimi's o_norm logic."""

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      activation: str = "sigmoid",
      dtype: DType = jnp.float32,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.activation = activation
    self.dtype = dtype
    self.rms_norm = RMSNorm(
        num_features=dim,
        epsilon=eps,
        dtype=dtype,
        rngs=rngs,
    )

  def __call__(self, x: Array, gate: Array) -> Array:
    normalized_x = self.rms_norm(x)
    if self.activation == "sigmoid":
      g = jax.nn.sigmoid(gate.astype(jnp.float32))
    elif self.activation in ("silu", "swish"):
      g = jax.nn.silu(gate.astype(jnp.float32))
    else:
      g = gate
    return (normalized_x * g).astype(self.dtype)


class KimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention Implementation with maximized code reuse."""

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      head_dim: int,
      conv_kernel_size: int = 4,
      normalization_layer_epsilon: float = 1e-5,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.conv_kernel_size = conv_kernel_size
    self.normalization_layer_epsilon = normalization_layer_epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init

    # Projections
    self.q_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.k_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.v_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    # Short convolutions (Match user keys: q_conv1d, k_conv1d, v_conv1d)
    conv_dim = num_heads * head_dim
    conv_kwargs = {
        "in_features": conv_dim,
        "out_features": conv_dim,
        "kernel_size": (conv_kernel_size,),
        "feature_group_count": conv_dim,
        "padding": "CAUSAL",
        "use_bias": False,
        "dtype": dtype,
        "rngs": rngs,
    }
    self.q_conv1d = nnx.Conv(**conv_kwargs)
    self.k_conv1d = nnx.Conv(**conv_kwargs)
    self.v_conv1d = nnx.Conv(**conv_kwargs)

    # Gating and Beta branches
    self.b_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    # Bottleneck gate projections (f and g branches)
    self.f_a_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.f_b_proj = DenseGeneral(
        in_features_shape=(head_dim,), out_features_shape=(num_heads*head_dim),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.g_a_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.g_b_proj = DenseGeneral(
        in_features_shape=(head_dim,), out_features_shape=(num_heads*head_dim),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    # Gate params (Ref: Qwen3NextGatedDeltaNet initialization)
    def a_log_init(key, shape, dtype=jnp.float32):
      return jnp.log(jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0))

    self.A_log = nnx.Param(a_log_init(rngs.params(), (1,1,num_heads,1)))
    self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (num_heads*head_dim), dtype=jnp.float32))

    # Output stage
    self.o_norm = FusedRMSNormGated(
        dim=head_dim, eps=self.normalization_layer_epsilon, activation="sigmoid", dtype=dtype, rngs=rngs,
    )
    self.o_proj = DenseGeneral(
        in_features_shape=(num_heads*head_dim), out_features_shape=(hidden_size,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

  def apply_fused_kda_gate(self, g_linear: Array) -> Array:
    """Computes log-space forget gate."""
    b, s, _ = g_linear.shape
    g = g_linear + self.dt_bias
    sp = jax.nn.softplus(g.astype(jnp.float32)).reshape(b, s, self.num_heads, self.head_dim)
    return (-jnp.exp(self.A_log) * sp).astype(self.dtype).reshape(b, s, -1)

  def __call__(
      self,
      hidden_states: Array,
      initial_state: Optional[Array] = None,
      output_final_state: bool = False,
  ) -> Tuple[Array, Optional[Array]]:
    batch, seq_len, _ = hidden_states.shape

    # 1. Projections and L2 Norm (Reusing normalizations.l2norm)
    q = l2norm(self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1), dim=-1, eps=1e-6)
    k = l2norm(self.k_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1), dim=-1, eps=1e-6)
    v = self.v_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1)

    # 2. Causal Conv (Applied per channel)
    def apply_conv(x, conv_layer):
      # x: [B, T, H, D]
      batch, seq_len, num_heads, head_dim = x.shape
      x_flat = x.reshape(batch, seq_len, -1)
      out = conv_layer(x_flat)
      out = jax.nn.silu(out.astype(jnp.float32)).astype(self.dtype)
      return out.reshape(batch, seq_len, num_heads, head_dim)

    q = apply_conv(q, self.q_conv1d)
    k = apply_conv(k, self.k_conv1d)
    v = apply_conv(v, self.v_conv1d)

    # 3. Gating and Beta
    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32)).astype(self.dtype)
    g_forget = self.apply_fused_kda_gate(self.f_b_proj(self.f_a_proj(hidden_states)))

    # 4. Core Attention Interface
    attn_out, final_state = chunk_parallel_delta_attention(
        q=q, k=k, v=v, g=g_forget, beta=beta,
        initial_state=initial_state, scale=self.head_dim**-0.5, output_final_state=output_final_state,
    )

    # 5. Output stage
    g_output = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(batch, seq_len, self.num_heads, self.head_dim)
    out = self.o_norm(attn_out, g_output)
    out = out.reshape(batch, seq_len, -1)
    
    return self.o_proj(out), final_state