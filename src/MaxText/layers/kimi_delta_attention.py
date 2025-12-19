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
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = 64,
    initial_state: None | jax.Array = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, None | jax.Array]:
  """
  JAX implementation of Chunked KDA.
  Final verified fixes:
  1. Gating Direction: Row - Col (g[i] - g[j])
  2. Stage 2 Mask: Strict Lower (i > j)
  3. Stage 3 Mask: Lower + Diagonal (i >= j)
  4. Beta application order: Rows then Columns
  """
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
  key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
  value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
  g = jnp.transpose(g, (0, 2, 1, 3)).astype(jnp.float32)
  beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)

  batch_size, num_heads, sequence_length, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

  if pad_size > 0:
    pad_config_4d = ((0, 0), (0, 0), (0, pad_size), (0, 0))
    pad_config_3d = ((0, 0), (0, 0), (0, pad_size))
    query = jnp.pad(query, pad_config_4d)
    key = jnp.pad(key, pad_config_4d)
    value = jnp.pad(value, pad_config_4d)
    g = jnp.pad(g, pad_config_4d)
    beta = jnp.pad(beta, pad_config_3d)

  total_sequence_length = sequence_length + pad_size
  scale = k_head_dim ** -0.5
  query = query * scale

  num_chunks = total_sequence_length // chunk_size
  
  def to_chunk(x):
      new_shape = (batch_size, num_heads, num_chunks, chunk_size) + x.shape[3:]
      return x.reshape(new_shape)

  query_c = to_chunk(query)
  key_c = to_chunk(key)
  value_c = to_chunk(value)
  g_c = to_chunk(g)
  beta_c = beta.reshape(batch_size, num_heads, num_chunks, chunk_size)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK CALCULATION (Recursive Dependency)
  # =========================================================================
  g_cumsum = jnp.cumsum(g_c, axis=-2)
  
  def compute_chunk_vars(k_blk, g_blk, beta_blk, v_blk):
      prec = jax.lax.Precision.HIGHEST
      g_diff = jnp.expand_dims(g_blk, -2) - jnp.expand_dims(g_blk, -3)
      decay_full = jnp.exp(g_diff)
      
      idx = jnp.arange(chunk_size)
      
      # [STRICT MASK] Stage 2: i > j (Strict Lower)
      # Matches PyTorch triu(0) masked_fill 0
      mask = idx[:, None] > idx[None, :] 
      decay_mask = jnp.where(jnp.expand_dims(mask, -1), decay_full, 0.0)
      
      A_raw = jnp.einsum('id, jd, ijd -> ij', k_blk, k_blk, decay_mask, precision=prec)

      # [BETA ROW]
      A = A_raw * jnp.expand_dims(beta_blk, -1)
      
      # [INVERT] Matches PyTorch logic A = -A then closure
      A_neg = -A
      
      def invert_body(i, m):
          row = m[i]
          mask_idx = jnp.arange(chunk_size) < i
          row = jnp.where(mask_idx, row, 0.0)
          increment = jnp.dot(row, m, precision=prec)
          increment = jnp.where(mask_idx, increment, 0.0)
          return m.at[i].set(row + increment)

      A_inv = jax.lax.fori_loop(1, chunk_size, invert_body, A_neg)
      
      # [BETA COL] Matches PyTorch (A_inv + I) * beta_col
      T = A_inv + jnp.eye(chunk_size)
      T_final = T * jnp.expand_dims(beta_blk, -2) 
      
      # Compute u, w
      u = jnp.matmul(T_final, v_blk, precision=prec)
      w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
      
      return u, w

  compute_vmap = jax.vmap(jax.vmap(jax.vmap(compute_chunk_vars)))
  u_c, w_c = compute_vmap(key_c, g_cumsum, beta_c, value_c)

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (Local Attention + State Pass)
  # =========================================================================
  
  def to_scan(x): return jnp.transpose(x, (2, 0, 1, 3, 4))
  
  if initial_state is None:
      last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
  else:
      last_recurrent_state = initial_state

  xs = (
      to_scan(query_c), 
      to_scan(key_c), 
      to_scan(u_c), 
      to_scan(w_c), 
      to_scan(g_cumsum)
  )

  def scan_body(prev_state, x):
      q_i, k_i, u_i, w_i, g_i = x
      prec = jax.lax.Precision.HIGHEST
      
      # [FIXED DIRECTION] Row - Col
      g_diff = jnp.expand_dims(g_i, -2) - jnp.expand_dims(g_i, -3)
      decay_full = jnp.exp(g_diff)
      
      idx = jnp.arange(chunk_size)
      
      # [INCLUSIVE MASK] Stage 3: i >= j (Lower + Diagonal)
      # Matches PyTorch triu(1) masked_fill 0
      mask = idx[:, None] >= idx[None, :] 
      g_rel = jnp.where(jnp.expand_dims(mask, -1), decay_full, 0.0)
      
      attn_local = jnp.einsum('...ik, ...jk, ...ijk -> ...ij', q_i, k_i, g_rel)
      
      correction = jnp.matmul(w_i, prev_state, precision=prec)
      v_new = u_i - correction
      
      o_hist = jnp.matmul(q_i * jnp.exp(g_i), prev_state, precision=prec)
      o_intra = jnp.matmul(attn_local, v_new, precision=prec)
      o_block = o_hist + o_intra
      
      decay_last = jnp.exp(g_i[..., -1, :])
      S_decayed = prev_state * jnp.expand_dims(decay_last, -1)
      
      # k_tail: Matches PyTorch exp(G_end - G_cur)
      k_tail = k_i * jnp.exp(jnp.expand_dims(g_i[..., -1, :], -2) - g_i)
      update_term = jnp.matmul(jnp.swapaxes(k_tail, -1, -2), v_new, precision=prec)
      
      new_state = S_decayed + update_term
      
      return new_state, o_block

  final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
  core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
  core_attn_out = core_attn_out[:, :, :sequence_length, :]
  core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

  return core_attn_out, final_state if output_final_state else None


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