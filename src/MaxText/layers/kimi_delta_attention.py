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
    chunk_size: int = 256,
    initial_state: None | jax.Array = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, None | jax.Array]:
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
      # (batch_size, num_heads, num_chunks, chunk_size) -> (batch_size, num_heads, num_chunks, chunk_size, head_dim)
      new_shape = (batch_size, num_heads, num_chunks, chunk_size) + x.shape[3:]
      return x.reshape(new_shape)

  query_c = to_chunk(query)
  key_c = to_chunk(key)
  value_c = to_chunk(value)
  g_c = to_chunk(g)
  beta_c = beta.reshape(batch_size, num_heads, num_chunks, chunk_size)

  g_cumsum = jnp.cumsum(g_c, axis=-2)
  
  #(batch_size, num_heads, num_chunks, chunk_size, head_dim) -> (num_chunks, batch_size, num_heads, chunk_size, head_dim)
  def to_scan(x): return jnp.transpose(x, (2, 0, 1, 3, 4))
  
  if initial_state is None:
      last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
  else:
      last_recurrent_state = initial_state

  # Prepare scan inputs. Note beta_c rank is one less than others.
  xs = (
      to_scan(query_c), 
      to_scan(key_c), 
      to_scan(value_c), 
      to_scan(g_cumsum),
      jnp.transpose(beta_c, (2, 0, 1, 3)) # (chunks, batch, heads, chunk_size)
  )

  def scan_body(prev_state, x):
      q_i, k_i, v_i, g_i, beta_i = x
      prec = jax.lax.Precision.HIGHEST
      
      # --- Fused Local Chunk Computation ---
      def compute_chunk_vars_local(k_blk, g_blk, beta_blk, v_blk):
          # k_blk: (C, D), g_blk: (C, D), beta_blk: (C), v_blk: (C, D)
          
          # Optimization: Compute A_raw row-by-row to avoid (C, C, D) intermediate tensor
          # A_raw_ij = sum_d(k_i * k_j * exp(g_i - g_j))
          
          def compute_A_row(carry, args):
              idx, k_row, g_row = args
              # k_row: (D,), g_row: (D,)
              
              g_diff = g_row[None, :] - g_blk # (C, D)
              
              # Mask: i > j
              mask = idx > jnp.arange(chunk_size)
              safe_g_diff = jnp.where(mask[:, None], g_diff, -float('inf'))
              
              term = (k_row[None, :] * k_blk) * jnp.exp(safe_g_diff) # (C, D)
              row_sum = jnp.sum(term, axis=-1) # (C,)
              return carry, row_sum

          # Scan over rows (i)
          _, A_raw = jax.lax.scan(compute_A_row, None, (jnp.arange(chunk_size), k_blk, g_blk))
          
          A = A_raw * jnp.expand_dims(beta_blk, -1)
          
          eye = jnp.eye(chunk_size, dtype=A.dtype)
          L = eye + A
          T = jax.scipy.linalg.solve_triangular(L, eye, lower=True)
          T_final = T * jnp.expand_dims(beta_blk, -2) 
          
          u = jnp.matmul(T_final, v_blk, precision=prec)
          w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
          return u, w

      # vmap over Batch (0) and Heads (1)
      # k_i: (B, H, C, D)
      u_i, w_i = jax.vmap(jax.vmap(compute_chunk_vars_local))(k_i, g_i, beta_i, v_i)
      # -------------------------------------------------------------
      
      # Stable calculation
      # attn_local_ij = q_i * k_j * exp(g_i - g_j)
      # o_intra = attn_local @ v_new
      
      # Optimization: Compute o_intra row-by-row (fused attention) to avoid (B, H, C, C) intermediate
      
      def compute_o_intra_row(carry, args):
          idx, q_vec, g_vec = args 
          # q_vec: (B, H, D), g_vec: (B, H, D)
          
          # 1. Compute Attention Score Row: (B, H, C)
          g_diff = jnp.expand_dims(g_vec, 2) - g_i # (B, H, C, D)
          
          mask = idx >= jnp.arange(chunk_size) # (C,)
          mask_broad = jnp.expand_dims(mask, (0, 1, 3)) # (1, 1, C, 1)

          # Mask positive exponents (i < j) to avoid overflow
          safe_g_diff = jnp.where(mask_broad, g_diff, -float('inf'))
          
          # term: (B, H, C, D)
          term = jnp.expand_dims(q_vec, 2) * k_i * jnp.exp(safe_g_diff)
          
          attn_row = jnp.sum(term, axis=-1) # (B, H, C)
          
          # 2. Fused Multiply with Value: (B, H, D)
          # attn_row (B, H, C) @ v_new (B, H, C, D) -> (B, H, D)
          # We use simple reduction over C dimension
          o_row = jnp.sum(jnp.expand_dims(attn_row, -1) * v_new, axis=2)
          
          return carry, o_row

      # Prepare scan inputs
      q_scan = jnp.transpose(q_i, (2, 0, 1, 3)) # (C, B, H, D)
      g_scan = jnp.transpose(g_i, (2, 0, 1, 3)) # (C, B, H, D)
      
      correction = jnp.matmul(w_i, prev_state, precision=prec)
      v_new = u_i - correction
      
      _, o_intra_stacked = jax.lax.scan(compute_o_intra_row, None, (jnp.arange(chunk_size), q_scan, g_scan))
      
      # o_intra_stacked: (C, B, H, D) -> (B, H, C, D)
      o_intra = jnp.transpose(o_intra_stacked, (1, 2, 0, 3))
      
      o_hist = jnp.matmul(q_i * jnp.exp(g_i), prev_state, precision=prec)
      # o_intra already computed via fused scan
      o_block = o_hist + o_intra
      
      decay_last = jnp.exp(g_i[..., -1, :])
      S_decayed = prev_state * jnp.expand_dims(decay_last, -1)
      
      k_tail = k_i * jnp.exp(jnp.expand_dims(g_i[..., -1, :], -2) - g_i)
      update_term = jnp.matmul(jnp.swapaxes(k_tail, -1, -2), v_new, precision=prec)
      
      new_state = S_decayed + update_term
      
      return new_state, o_block

  # Use gradient checkpointing to avoid storing O(Chunk^2) intermediates for the entire sequence
  scan_body = jax.checkpoint(scan_body)

  final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

  core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
  core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
  core_attn_out = core_attn_out[:, :, :sequence_length, :]
  core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

  return core_attn_out, final_state if output_final_state else None


class FusedRMSNormGated(nnx.Module):
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
  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      head_dim: int,
      num_v_heads: Optional[int] = None,
      conv_kernel_size: int = 4,
      normalization_layer_epsilon: float = 1e-5,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
    self.head_dim = head_dim
    self.conv_kernel_size = conv_kernel_size
    self.normalization_layer_epsilon = normalization_layer_epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init

    self.q_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.k_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.v_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(self.num_v_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    conv_dim = num_heads * head_dim
    conv_kwargs = {
        "kernel_size": (conv_kernel_size,),
        "padding": "CAUSAL",
        "use_bias": False,
        "dtype": dtype,
        "rngs": rngs,
    }
    self.q_conv1d = nnx.Conv(in_features=conv_dim, out_features=conv_dim, feature_group_count=conv_dim, **conv_kwargs)
    self.k_conv1d = nnx.Conv(in_features=conv_dim, out_features=conv_dim, feature_group_count=conv_dim, **conv_kwargs)
    
    v_conv_dim = self.num_v_heads * head_dim
    self.v_conv1d = nnx.Conv(in_features=v_conv_dim, out_features=v_conv_dim, feature_group_count=v_conv_dim, **conv_kwargs)

    self.b_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

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
        in_features_shape=(head_dim,), out_features_shape=(self.num_v_heads*head_dim),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=True, rngs=rngs,
    )

    def a_log_init(key, shape, dtype=jnp.float32):
      return jnp.log(jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0))

    self.A_log = nnx.Param(a_log_init(rngs.params(), (1,1,num_heads,1)))
    self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (num_heads*head_dim), dtype=jnp.float32))

    self.o_norm = FusedRMSNormGated(
        dim=head_dim, eps=self.normalization_layer_epsilon, activation="sigmoid", dtype=dtype, rngs=rngs,
    )
    self.o_proj = DenseGeneral(
        in_features_shape=(self.num_v_heads*head_dim), out_features_shape=(hidden_size,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

  def apply_fused_kda_gate(self, g_linear: Array) -> Array:
    """Computes log-space forget gate."""
    b, s, _ = g_linear.shape
    g = g_linear + self.dt_bias
    sp = jax.nn.softplus(g.astype(jnp.float32)).reshape(b, s, self.num_heads, self.head_dim)
    return -jnp.exp(self.A_log) * sp

  def __call__(
      self,
      hidden_states: Array,
      chunk_size: int = 64,
      initial_state: Optional[Array] = None,
      output_final_state: bool = False,
  ) -> Tuple[Array, Optional[Array]]:
    batch, seq_len, _ = hidden_states.shape

    q = self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1)
    k = self.k_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1)
    v = self.v_proj(hidden_states).reshape(batch, seq_len, self.num_v_heads, -1)

    def apply_conv(x, conv_layer):
      batch, seq_len, num_heads, head_dim = x.shape
      x_flat = x.reshape(batch, seq_len, -1)
      out = conv_layer(x_flat)
      out = jax.nn.silu(out.astype(jnp.float32))
      return out.reshape(batch, seq_len, num_heads, head_dim)

    q = apply_conv(q, self.q_conv1d)
    k = apply_conv(k, self.k_conv1d)
    v = apply_conv(v, self.v_conv1d)

    q = l2norm(q, dim=-1, eps=1e-6)
    k = l2norm(k, dim=-1, eps=1e-6)

    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))
    g_forget = self.apply_fused_kda_gate(self.f_b_proj(self.f_a_proj(hidden_states)))
    
    # Repeat for MQA/GQA if num_v_heads > num_heads
    if self.num_v_heads > self.num_heads:
        assert self.num_v_heads % self.num_heads == 0
        n_rep = self.num_v_heads // self.num_heads
        q = jnp.repeat(q, n_rep, axis=2)
        k = jnp.repeat(k, n_rep, axis=2)
        g_forget = jnp.repeat(g_forget, n_rep, axis=2)
        beta = jnp.repeat(beta, n_rep, axis=2)

    attn_out, final_state = chunk_parallel_delta_attention(
        query=q, key=k, value=v, g=g_forget, beta=beta,
        chunk_size=chunk_size, initial_state=initial_state, output_final_state=output_final_state
    )

    g_output = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(batch, seq_len, self.num_v_heads, self.head_dim)
    out = self.o_norm(attn_out, g_output)
    out = out.reshape(batch, seq_len, -1)
    
    return self.o_proj(out), final_state