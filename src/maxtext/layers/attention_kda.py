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

"""Kimi Delta Attention (KDA) Linear Attention Layer.

This module implements the Kimi Delta Attention (KDA) mechanism for Kimi-K3 models.
KDA is a linear attention architecture with data-dependent channel-wise decay,
causal short convolutions, chunked WY representations for parallel training/prefill,
and autoregressive recurrent state updating for single-step decoding.

Key operations:
1. 3 separate 1D causal depthwise convolutions on Q, K, V with kernel_size=4 and SiLU activation.
2. Data-dependent channel decay (log space, computed in fp32; see `kda_gate`):
   g_raw = W_{f_b}(W_{f_a} x)  (hidden_size -> head_dim -> projection_size)
   g = gate_lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))   if gate_lower_bound is set
   g = -exp(A_log) * softplus(g_raw + dt_bias)                       otherwise
3. Step size beta:
   beta = sigmoid(W_b x)  (hidden_size -> num_heads)
4. L2-norm on Q, K along head dimension D, scaled by 1/sqrt(D) on Q.
5. Chunked WY scan (jax_chunk_kimi_delta_rule) for parallel chunked solve and inter-chunk scan.
6. Autoregressive recurrent step (jax_ar_kimi_delta_rule) for single-token decode.
7. Output gated RMSNorm:
   y = W_o(RMSNorm(o) * sigmoid(W_g x))
"""

import math
from typing import Optional, Tuple

from flax import nnx
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from maxtext.common.common_types import (
    Array,
    Config,
    DType,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_TRAIN,
    ShardMode,
)
from maxtext.layers.initializers import nd_dense_init, NdInitializer
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm, l2norm


# ==============================================================================
# 1D Causal Depthwise Short Convolution
# ==============================================================================
class ShortConvolution(nnx.Module):
  """1D Causal Depthwise Convolution module."""

  def __init__(
      self,
      hidden_size: int,
      kernel_size: int = 4,
      activation: str = "silu",
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal(),
      kernel_axes: tuple[None | str, ...] = (),
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes ShortConvolution.

    Args:
      hidden_size: Number of feature channels.
      kernel_size: 1D convolution window size (default: 4).
      activation: Activation function name applied after conv ('silu' or 'none').
      dtype: Computation dtype.
      weight_dtype: Weight parameter dtype.
      kernel_init: Kernel weight initializer.
      kernel_axes: Logical partition axes for kernel sharding.
      rngs: Flax NNX random number generators.
    """
    self.hidden_size = hidden_size
    self.kernel_size = kernel_size
    self.activation = activation
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes

    # Kernel shape in MaxText/JAX: [kernel_size, 1, hidden_size]
    # Corresponding PyTorch shape in HF: [hidden_size, 1, kernel_size]
    if rngs is not None:
      kernel_val = kernel_init(rngs.params(), (kernel_size, 1, hidden_size), weight_dtype)
    else:
      kernel_val = jnp.zeros((kernel_size, 1, hidden_size), dtype=weight_dtype)

    self.kernel = nnx.Param(kernel_val, out_sharding=kernel_axes)

  def __call__(
      self,
      x: Array,
      cache: Optional[Array] = None,
      output_final_state: bool = False,
  ) -> Tuple[Array, Optional[Array]]:
    """Applies causal depthwise 1D convolution.

    Args:
      x: Input tensor of shape [batch_size, seq_len, hidden_size].
      cache: Optional convolution history cache of shape [batch_size, kernel_size - 1, hidden_size].
      output_final_state: Whether to return updated cache state.

    Returns:
      Tuple of (convolved_output, next_cache_state).
    """
    _, _, c = x.shape
    k = self.kernel_size
    kernel = self.kernel.get_value().astype(self.dtype)

    if cache is not None:
      x_cat = jnp.concatenate([cache.astype(self.dtype), x.astype(self.dtype)], axis=1)
    else:
      x_cat = jnp.pad(x.astype(self.dtype), ((0, 0), (k - 1, 0), (0, 0)))

    # Compute depthwise 1D convolution
    out = lax.conv_general_dilated(
        lhs=x_cat,
        rhs=kernel,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=c,
    )

    if self.activation == "silu":
      out = jax.nn.silu(out)

    final_state = None
    if output_final_state:
      final_state = x_cat[:, -(k - 1) :, :]

    return out.astype(self.dtype), final_state


# ==============================================================================
# KDA Decay Gate
# ==============================================================================
def kda_gate(
    g_raw: Array,
    a_log: Array,
    dt_bias: Array,
    lower_bound: Optional[float],
) -> Array:
  """Per-channel log-space KDA decay `g` (fp32), matching `fla`'s in-kernel gate.

  With a lower bound (Kimi-K3 ships `gate_lower_bound=-5.0`) `fla` switches the activation
  to a bounded sigmoid; it is NOT the softplus gate clamped at the bound:
      g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))      in [lower_bound, 0)
  Without one it is the original Mamba-style gate:
      g = -exp(A_log) * softplus(g_raw + dt_bias)
  See `fla/ops/kda/gate.py` (`naive_kda_lowerbound_gate`, kernel `USE_LOWER_BOUND` branch).
  Like the `fla` kernel, the gate is computed in fp32 regardless of the activation dtype.

  Args:
    g_raw: Raw gate projection `[..., num_heads, head_dim]`.
    a_log: Per-head log rate `[num_heads]`.
    dt_bias: Per-channel bias `[num_heads * head_dim]`.
    lower_bound: Gate lower bound (negative), or None for the softplus gate.

  Returns:
    fp32 log-decay of the same shape as `g_raw`.
  """
  num_heads, head_dim = g_raw.shape[-2:]
  x = g_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32).reshape(num_heads, head_dim)
  rate = jnp.exp(a_log.astype(jnp.float32)).reshape(num_heads, 1)
  if lower_bound is not None:
    return lower_bound * jax.nn.sigmoid(rate * x)
  return -rate * jax.nn.softplus(x)


# ==============================================================================
# Chunked WY Linear Attention Scan for KDA
# ==============================================================================
def jax_chunk_kimi_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: Optional[Array] = None,
    compute_dtype: DType = jnp.float32,
) -> Tuple[Array, Optional[Array]]:
  """Optimized Chunked WY implementation of Kimi Delta Attention (KDA).

  Handles intra-chunk DPLR / WY solver and inter-chunk state recurrence with
  per-channel decay vector g in R^{H x D}.

  Args:
    query: Query tensor [batch_size, seq_len, num_heads, head_dim].
    key: Key tensor [batch_size, seq_len, num_heads, head_dim].
    value: Value tensor [batch_size, seq_len, num_heads, head_dim].
    g: Channel decay tensor in log space [batch_size, seq_len, num_heads, head_dim].
    beta: Step size tensor [batch_size, seq_len, num_heads].
    chunk_size: Block chunk size for WY representation (default: 64).
    initial_state: Optional initial recurrent state [batch_size, num_heads, head_dim, head_dim].
    compute_dtype: Precision dtype for internal calculations.

  Returns:
    Tuple of (attn_output, final_recurrent_state).
  """
  initial_dtype = query.dtype
  b, seq_len, h, d = key.shape
  v_dim = value.shape[-1]
  prec = lax.Precision.HIGHEST

  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  g = g.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  # Sequence padding to multiple of chunk_size
  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:

    def pad_fn(x, val=0.0):
      return jnp.pad(x, ((0, 0), (0, pad_len)) + ((0, 0),) * (x.ndim - 2), constant_values=val)

    query = pad_fn(query)
    key = pad_fn(key)
    value = pad_fn(value)
    g = pad_fn(g, val=0.0)
    beta = pad_fn(beta)

  total_len = query.shape[1]
  num_chunks = total_len // chunk_size

  # Reshape to (batch, num_chunks, chunk_size, num_heads, dim)
  def to_chunk_vec(x):
    return x.reshape(b, num_chunks, chunk_size, h, -1).transpose(0, 1, 3, 2, 4)

  def to_chunk_scl(x):
    return x.reshape(b, num_chunks, chunk_size, h).transpose(0, 1, 3, 2)

  q_c = to_chunk_vec(query)  # [B, N, H, C, D]
  k_c = to_chunk_vec(key)  # [B, N, H, C, D]
  v_c = to_chunk_vec(value)  # [B, N, H, C, V]
  g_c = to_chunk_vec(g)  # [B, N, H, C, D]
  beta_c = to_chunk_scl(beta)  # [B, N, H, C]

  # =========================================================================
  # Intra-chunk Pre-computation (WY Representation with stable relative decay)
  # =========================================================================
  g_cumsum = jnp.cumsum(g_c, axis=3)  # [B, N, H, C, D]

  # S_mat intra-chunk decay: g_diff[i, j] = g_cumsum[i] - g_cumsum[j] <= 0 for i > j
  g_diff = g_cumsum[:, :, :, :, None, :] - g_cumsum[:, :, :, None, :, :]  # [B, N, H, C, C, D]
  decay_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  exp_g_diff = jnp.where(decay_mask[None, None, None, :, :, None], jnp.exp(g_diff), 0.0)

  kk = k_c[:, :, :, :, None, :] * k_c[:, :, :, None, :, :]  # [B, N, H, C, C, D]
  s_mat = beta_c[:, :, :, :, None] * jnp.sum(kk * exp_g_diff, axis=-1)  # [B, N, H, C, C]

  identity = jnp.eye(chunk_size, dtype=compute_dtype)
  identity_broadcasted = jnp.broadcast_to(identity, s_mat.shape)
  a_mat = jax.scipy.linalg.solve_triangular(identity + s_mat, identity_broadcasted, lower=True, unit_diagonal=True)

  # WY factors
  v_beta = beta_c[..., None] * v_c  # [B, N, H, C, V]
  u = jnp.matmul(a_mat, v_beta, precision=prec)  # [B, N, H, C, V]
  k_beta_decay = beta_c[..., None] * k_c * jnp.exp(g_cumsum)  # [B, N, H, C, D]
  w = jnp.matmul(a_mat, k_beta_decay, precision=prec)  # [B, N, H, C, D]

  # =========================================================================
  # Inter-chunk Recurrence Scan
  # =========================================================================
  w_scan = w.swapaxes(0, 1)  # [N, B, H, C, D]
  u_scan = u.swapaxes(0, 1)  # [N, B, H, C, V]
  q_scan = q_c.swapaxes(0, 1)  # [N, B, H, C, D]
  k_scan = k_c.swapaxes(0, 1)  # [N, B, H, C, D]
  g_cumsum_scan = g_cumsum.swapaxes(0, 1)  # [N, B, H, C, D]

  if initial_state is None:
    h_init = jnp.zeros((b, h, d, v_dim), dtype=compute_dtype)
  else:
    h_init = initial_state.astype(compute_dtype)

  xs = (w_scan, u_scan, q_scan, k_scan, g_cumsum_scan)

  def scan_body(h_prev, args):
    w_i, u_i, q_i, k_i, g_i = args  # each [B, H, C, D/V]

    # Inter-chunk carry subtraction
    v_prime_carry = jnp.matmul(w_i, h_prev, precision=prec)
    v_new = u_i - v_prime_carry  # [B, H, C, V]

    # Inter-chunk output
    q_decay_i = q_i * jnp.exp(g_i)  # [B, H, C, D]
    attn_inter = jnp.matmul(q_decay_i, h_prev, precision=prec)  # [B, H, C, V]

    # Intra-chunk output
    g_diff_i = g_i[:, :, :, None, :] - g_i[:, :, None, :, :]  # [B, H, C, C, D]
    mask_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
    exp_g_diff_i = jnp.where(mask_causal[None, None, :, :, None], jnp.exp(g_diff_i), 0.0)
    qk = q_i[:, :, :, None, :] * k_i[:, :, None, :, :]  # [B, H, C, C, D]
    attn_intra = jnp.sum(qk * exp_g_diff_i, axis=-1)  # [B, H, C, C]

    o_i = attn_inter + jnp.matmul(attn_intra, v_new, precision=prec)  # [B, H, C, V]

    # Recurrent state update to chunk boundary
    g_last = g_i[..., -1, :]  # [B, H, D]
    h_decayed = jnp.exp(g_last[..., :, None]) * h_prev  # [B, H, D, V]
    k_end = k_i * jnp.exp(g_last[..., None, :] - g_i)  # [B, H, C, D]
    h_new = h_decayed + jnp.matmul(k_end.swapaxes(-1, -2), v_new, precision=prec)  # [B, H, D, V]

    return h_new, o_i

  final_h, o_chunks = lax.scan(scan_body, h_init, xs)

  # Reshape chunks back to [B, T_padded, H, V]
  o = o_chunks.transpose(1, 0, 3, 2, 4).reshape(b, total_len, h, v_dim)

  if pad_len > 0:
    o = o[:, :seq_len, :, :]

  return o.astype(initial_dtype), final_h.astype(compute_dtype)


# ==============================================================================
# Autoregressive Single-Step Decoding for KDA
# ==============================================================================
def jax_ar_kimi_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    initial_state: Optional[Array] = None,
    compute_dtype: DType = jnp.float32,
) -> Tuple[Array, Array]:
  """Single-token recurrent step for autoregressive decoding (seq_len == 1).

  Args:
    query: Query tensor [batch_size, 1, num_heads, head_dim] or [batch_size, num_heads, head_dim].
    key: Key tensor [batch_size, 1, num_heads, head_dim] or [batch_size, num_heads, head_dim].
    value: Value tensor [batch_size, 1, num_heads, head_dim] or [batch_size, num_heads, head_dim].
    g: Channel decay tensor [batch_size, 1, num_heads, head_dim] or [batch_size, num_heads, head_dim].
    beta: Step size tensor [batch_size, 1, num_heads] or [batch_size, num_heads].
    initial_state: Recurrent state tensor [batch_size, num_heads, head_dim, head_dim].
    compute_dtype: Precision dtype.

  Returns:
    Tuple of (output, next_state).
  """
  initial_dtype = query.dtype
  has_seq_dim = query.ndim == 4

  if has_seq_dim:
    query = query.squeeze(1)
    key = key.squeeze(1)
    value = value.squeeze(1)
    g = g.squeeze(1)
    beta = beta.squeeze(1)

  b, h, d = key.shape
  v_dim = value.shape[-1]

  query = query.astype(compute_dtype)
  key = key.astype(compute_dtype)
  value = value.astype(compute_dtype)
  g = g.astype(compute_dtype)
  beta = beta.astype(compute_dtype)

  if initial_state is None:
    state = jnp.zeros((b, h, d, v_dim), dtype=compute_dtype)
  else:
    state = initial_state.astype(compute_dtype)

  prec = lax.Precision.HIGHEST
  alpha = jnp.exp(g)  # [B, H, D]
  k_alpha = key * alpha  # [B, H, D]

  # v_prime_carry = (k_alpha)^T @ state -> [B, H, V]
  v_prime_carry = jnp.einsum("bhi,bhiv->bhv", k_alpha, state, precision=prec)
  v_new = beta[..., None] * (value - v_prime_carry)  # [B, H, V]

  # Attention output
  q_alpha = query * alpha  # [B, H, D]
  attn_inter = jnp.einsum("bhi,bhiv->bhv", q_alpha, state, precision=prec)  # [B, H, V]
  attn_intra = jnp.sum(query * key, axis=-1, keepdims=True)  # [B, H, 1]
  core_out = attn_inter + attn_intra * v_new  # [B, H, V]

  # State update: diag(alpha) @ state + key \otimes v_new
  new_state = alpha[..., :, None] * state + jnp.einsum("bhi,bhv->bhiv", key, v_new, precision=prec)

  if has_seq_dim:
    core_out = core_out[:, None, :, :]

  return core_out.astype(initial_dtype), new_state.astype(compute_dtype)


# ==============================================================================
# Top-Level Kimi Delta Attention (KDA) NNX Module
# ==============================================================================
class KimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention (KDA) Linear Attention Module."""

  def __init__(
      self,
      config: Optional[Config] = None,
      hidden_size: int = 7168,
      num_heads: int = 96,
      head_dim: int = 128,
      conv_size: int = 4,
      chunk_size: int = 64,
      rms_norm_eps: float = 1e-5,
      gate_lower_bound: Optional[float] = -5.0,
      compute_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes KimiDeltaAttention module.

    Args:
      config: Optional MaxText Config object.
      hidden_size: Model hidden size (default: 7168).
      num_heads: Number of attention heads (default: 96).
      head_dim: Dimension per head (default: 128).
      conv_size: 1D convolution window size (default: 4).
      chunk_size: Chunk size for WY linear attention scan (default: 64).
      rms_norm_eps: Epsilon for output RMSNorm (default: 1e-5).
      gate_lower_bound: Lower bound of the bounded-sigmoid decay gate (default: -5.0);
        None selects the softplus gate. See `kda_gate`.
      compute_dtype: Precision of the KDA core (q/k L2-norm, beta sigmoid, decay cumsum,
        WY solve and recurrent state). float32 (default) matches fla; the core output is
        cast back to `dtype`. From a config: `kda_compute_dtype`.
      dtype: Computation dtype.
      weight_dtype: Weights parameter dtype.
      kernel_init: Initializer for linear projection weights.
      shard_mode: Sharding mode (AUTO or EXPLICIT).
      matmul_precision: Matrix multiplication precision string.
      rngs: Flax NNX random number generators.
    """
    self.config = config
    if config is not None:
      self.hidden_size = getattr(config, "emb_dim", hidden_size)
      # MaxText names these `num_query_heads` / `normalization_layer_epsilon`; fall back to
      # the HF-style names so the module still works with hand-built config objects.
      self.num_heads = getattr(config, "num_query_heads", getattr(config, "num_heads", num_heads))
      self.head_dim = getattr(config, "head_dim", head_dim)
      self.conv_size = getattr(config, "short_conv_kernel_size", conv_size)
      self.chunk_size = getattr(config, "kda_chunk_size", chunk_size)
      self.rms_norm_eps = getattr(config, "normalization_layer_epsilon", getattr(config, "rms_norm_eps", rms_norm_eps))
      self.gate_lower_bound = getattr(config, "gate_lower_bound", gate_lower_bound)
      self.compute_dtype = jnp.dtype(getattr(config, "kda_compute_dtype", compute_dtype))
      self.dtype = getattr(config, "dtype", dtype)
      self.weight_dtype = getattr(config, "weight_dtype", weight_dtype)
      self.shard_mode = getattr(config, "shard_mode", shard_mode)
      self.matmul_precision = getattr(config, "matmul_precision", matmul_precision)
    else:
      self.hidden_size = hidden_size
      self.num_heads = num_heads
      self.head_dim = head_dim
      self.conv_size = conv_size
      self.chunk_size = chunk_size
      self.rms_norm_eps = rms_norm_eps
      self.gate_lower_bound = gate_lower_bound
      self.compute_dtype = jnp.dtype(compute_dtype)
      self.dtype = dtype
      self.weight_dtype = weight_dtype
      self.shard_mode = shard_mode
      self.matmul_precision = matmul_precision

    self.projection_size = self.num_heads * self.head_dim
    if rngs is None:
      rngs = nnx.Rngs(params=0)
    self.rngs = rngs

    # Q, K, V Linear Projections
    self.q_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )
    self.k_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )
    self.v_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

    # 1D Causal Short Convolutions
    self.q_conv1d = ShortConvolution(
        hidden_size=self.projection_size,
        kernel_size=self.conv_size,
        activation="silu",
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.k_conv1d = ShortConvolution(
        hidden_size=self.projection_size,
        kernel_size=self.conv_size,
        activation="silu",
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.v_conv1d = ShortConvolution(
        hidden_size=self.projection_size,
        kernel_size=self.conv_size,
        activation="silu",
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    # Channel Decay Projections and Parameters
    self.f_a_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", None),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )
    self.f_b_proj = DenseGeneral(
        in_features_shape=self.head_dim,
        out_features_shape=self.projection_size,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=(None, "heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

    # Learnable Log Decay (A_log) and Bias (dt_bias)
    if rngs is not None:
      a_log_val = jnp.log(
          jax.random.uniform(rngs.params(), (self.num_heads,), minval=1.0, maxval=16.0, dtype=self.weight_dtype)
      )
      dt_bias_val = jax.random.uniform(
          rngs.params(), (self.projection_size,), minval=-1.0, maxval=1.0, dtype=self.weight_dtype
      )
    else:
      a_log_val = jnp.zeros((self.num_heads,), dtype=self.weight_dtype)
      dt_bias_val = jnp.zeros((self.projection_size,), dtype=self.weight_dtype)

    self.A_log = nnx.Param(a_log_val, out_sharding=("heads",))
    self.dt_bias = nnx.Param(dt_bias_val, out_sharding=("heads",))

    # Step Size (Beta) Projection: hidden_size -> num_heads
    self.b_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.num_heads,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

    # Output Gate Projection: hidden_size -> projection_size
    self.g_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

    # Output Normalization (Head-wise RMSNorm) & Output Linear Projection
    self.o_norm = RMSNorm(
        num_features=self.head_dim,
        epsilon=self.rms_norm_eps,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        rngs=rngs,
    )
    self.out = DenseGeneral(
        in_features_shape=self.projection_size,
        out_features_shape=self.hidden_size,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("heads", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        shard_mode=self.shard_mode,
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      conv_state: Optional[Tuple[Array, Array, Array]] = None,
      recurrent_state: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      output_final_state: bool = False,
  ) -> Tuple[Array, Optional[Tuple[Array, Array, Array]], Optional[Array]]:
    """Forward pass for KDA Linear Attention.

    Args:
      hidden_states: Input activations [batch_size, seq_len, hidden_size].
      conv_state: Optional tuple of (conv_state_q, conv_state_k, conv_state_v).
      recurrent_state: Optional recurrent memory state [batch_size, num_heads, head_dim, head_dim].
      model_mode: Operational mode ('train', 'prefill', or 'autoregressive').
      output_final_state: Whether to return updated cache and recurrent state.

    Returns:
      Tuple of (layer_output, next_conv_state, next_recurrent_state).
    """
    b, seq_len, _ = hidden_states.shape

    # 1. Linear Projections
    q_proj_states = self.q_proj(hidden_states)
    k_proj_states = self.k_proj(hidden_states)
    v_proj_states = self.v_proj(hidden_states)

    # 2. 1D Causal Short Convolutions
    conv_q, conv_k, conv_v = (None, None, None)
    if conv_state is not None:
      conv_q, conv_k, conv_v = conv_state

    q_conv, next_conv_q = self.q_conv1d(q_proj_states, cache=conv_q, output_final_state=output_final_state)
    k_conv, next_conv_k = self.k_conv1d(k_proj_states, cache=conv_k, output_final_state=output_final_state)
    v_conv, next_conv_v = self.v_conv1d(v_proj_states, cache=conv_v, output_final_state=output_final_state)

    next_conv_state = None
    if output_final_state:
      next_conv_state = (next_conv_q, next_conv_k, next_conv_v)

    # 3. Data-dependent Channel Decay Calculation
    g_raw = self.f_b_proj(self.f_a_proj(hidden_states))
    g_raw = g_raw.reshape(b, seq_len, self.num_heads, self.head_dim)
    g = kda_gate(g_raw, self.A_log.get_value(), self.dt_bias.get_value(), self.gate_lower_bound)

    # 4. Step Size Beta (sigmoid in the core precision; fla computes it in fp32 in-kernel)
    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(self.compute_dtype))

    # 5. Reshape Q, K, V to Head Layout and Apply QK L2-Norm
    # Like fla's in-kernel l2norm + scale, q/k are normalized and scaled in the core precision.
    q = q_conv.reshape(b, seq_len, self.num_heads, self.head_dim).astype(self.compute_dtype)
    k = k_conv.reshape(b, seq_len, self.num_heads, self.head_dim).astype(self.compute_dtype)
    v = v_conv.reshape(b, seq_len, self.num_heads, self.head_dim)

    # L2-Norm along head_dim, scaled by 1/sqrt(head_dim) on Q
    scale = 1.0 / math.sqrt(self.head_dim)
    q = l2norm(q, dim=-1, eps=1e-6) * scale
    k = l2norm(k, dim=-1, eps=1e-6)

    # 6. Core KDA Execution (Chunked vs Autoregressive Recurrent)
    # The core computes in `compute_dtype` (default fp32, matching fla's kernels: decay cumsum,
    # WY solve and recurrent state); the output is returned in the activation dtype.
    is_ar_decode = model_mode == MODEL_MODE_AUTOREGRESSIVE and seq_len == 1

    if is_ar_decode:
      if recurrent_state is None:
        recurrent_state = jnp.zeros((b, self.num_heads, self.head_dim, self.head_dim), dtype=self.compute_dtype)
      o, next_rec_state = jax_ar_kimi_delta_rule(
          query=q,
          key=k,
          value=v,
          g=g,
          beta=beta,
          initial_state=recurrent_state,
          compute_dtype=self.compute_dtype,
      )
    else:
      o, next_rec_state = jax_chunk_kimi_delta_rule(
          query=q,
          key=k,
          value=v,
          g=g,
          beta=beta,
          chunk_size=self.chunk_size,
          initial_state=recurrent_state,
          compute_dtype=self.compute_dtype,
      )
    o = o.astype(v.dtype)

    # 7. Output Gated RMSNorm & Projection
    g_out = self.g_proj(hidden_states).reshape(b, seq_len, self.num_heads, self.head_dim)
    o_normed = self.o_norm(o)
    o_gated = o_normed * jax.nn.sigmoid(g_out)

    # Flatten heads and project to hidden_size
    o_flat = o_gated.reshape(b, seq_len, self.projection_size)
    output = self.out(o_flat)

    next_recurrent_state = next_rec_state if output_final_state else None
    return output, next_conv_state, next_recurrent_state
