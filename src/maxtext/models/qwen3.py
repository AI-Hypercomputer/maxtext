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

"""Qwen3 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
import math
import os
from typing import Any, cast

from flax import linen as nn
from flax import nnx
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import xla_metadata
import jax.nn
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, AttentionType, BATCH, Config, DType, EMBED, LENGTH, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.common.common_types import HyperConnectionType
from maxtext.common.common_types import KV_BATCH, KV_HEAD
from maxtext.inference import kvcache
from maxtext.utils.sharding import (
    get_logical_axis_rules,
    logical_to_mesh_axes,
    remove_incompatible_mesh_axes_from_partition_spec,
)
from maxtext.layers import attentions
from maxtext.layers import initializers as max_initializers
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import nnx_scan
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import PositionalEmbedding, Qwen3OmniMoeVisionPosEmbedInterpolate
from maxtext.layers.initializers import nd_dense_init, variable_to_logically_partitioned
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.moe import RoutedMoE
from maxtext.layers.normalizations import Qwen3NextRMSNorm, Qwen3NextRMSNormGated, RMSNorm, l2norm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import get_logical_axis_rules, logical_to_mesh_axes

# -----------------------------------------
# Qwen3-Next Layer Implementations
# -----------------------------------------


def naive_jax_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None, use_qk_norm_in_gdn=False
):
  """Naive implementation of the Gated Delta Rule in jax."""
  initial_dtype = query.dtype
  if use_qk_norm_in_gdn:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

  query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
  key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
  value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
  beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
  g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

  batch_size, num_heads, sequence_length, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

  if pad_size > 0:
    query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
    g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))

  total_sequence_length = sequence_length + pad_size
  scale = query.shape[-1] ** -0.5
  query = query * scale

  v_beta = value * jnp.expand_dims(beta, -1)
  k_beta = key * jnp.expand_dims(beta, -1)

  num_chunks = total_sequence_length // chunk_size
  query_c = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  key_c = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  k_beta_c = k_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
  v_beta_c = v_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
  g_c = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

  mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

  g_cumsum = jnp.cumsum(g_c, axis=-1)
  g_diff = jnp.expand_dims(g_cumsum, -1) - jnp.expand_dims(g_cumsum, -2)
  g_diff_tril = jnp.tril(g_diff)
  g_diff_exp = jnp.exp(g_diff_tril).astype(jnp.float32)
  decay_mask = g_diff_exp

  prec = jax.lax.Precision.HIGHEST
  attn = -jnp.einsum('...cd,...ed->...ce', k_beta_c, key_c, precision=prec) * decay_mask
  attn = attn * (~mask).astype(attn.dtype)

  def inner_attn_body(i, attn_val):
    indices = jnp.arange(chunk_size)
    col_mask = indices < i
    row = attn_val[..., i, :] * col_mask
    sub_mask = jnp.expand_dims(indices < i, -1) & (indices < i)
    sub = attn_val * sub_mask
    row_exp = jnp.expand_dims(row, -1)
    term = row_exp * sub
    summed = jnp.sum(term, axis=-2)
    update_val = row + summed
    original_row = attn_val[..., i, :]
    new_row = jnp.where(col_mask, update_val, original_row)
    return attn_val.at[..., i, :].set(new_row)

  attn = jax.lax.fori_loop(1, chunk_size, inner_attn_body, attn)
  attn = attn + jnp.eye(chunk_size, dtype=attn.dtype)
  value_intra = jnp.einsum('...cd,...dv->...cv', attn, v_beta_c, precision=prec)
  k_cumdecay = jnp.einsum('...cd,...dv->...cv', attn, (k_beta_c * jnp.expand_dims(jnp.exp(g_cumsum), -1)), precision=prec)

  output_final_state = initial_state is not None
  if initial_state is None:
    last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=value_intra.dtype)
  else:
    last_recurrent_state = initial_state.astype(value_intra.dtype)

  mask_inter = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

  query_scan = jnp.transpose(query_c, (2, 0, 1, 3, 4))
  key_scan = jnp.transpose(key_c, (2, 0, 1, 3, 4))
  value_scan = jnp.transpose(value_intra, (2, 0, 1, 3, 4))
  k_cumdecay_scan = jnp.transpose(k_cumdecay, (2, 0, 1, 3, 4))
  g_scan = jnp.transpose(g_cumsum, (2, 0, 1, 3))
  decay_mask_scan = jnp.transpose(decay_mask, (2, 0, 1, 3, 4))

  xs = (query_scan, key_scan, value_scan, k_cumdecay_scan, g_scan, decay_mask_scan)

  def scan_body(prev_state, x):
    q_i, k_i, v_i, k_cumdecay_i, g_i, decay_mask_i = x
    last_recurrent_state = prev_state
    prec = jax.lax.Precision.HIGHEST

    attn_i = jnp.einsum('...cd,...ed->...ce', q_i, k_i, precision=prec) * decay_mask_i
    attn_i = jnp.where(mask_inter, 0.0, attn_i)

    v_prime = jnp.einsum('...cd,...dv->...cv', k_cumdecay_i, last_recurrent_state, precision=prec)
    v_new = v_i - v_prime

    g_i_exp = jnp.exp(g_i)
    attn_inter = jnp.einsum('...cd,...dv->...cv', q_i * jnp.expand_dims(g_i_exp, -1), last_recurrent_state, precision=prec)

    core_attn_out_i = attn_inter + jnp.einsum('...ce,...ev->...cv', attn_i, v_new, precision=prec)

    g_i_last_exp = jnp.exp(g_i[..., -1, None, None])
    new_last_recurrent_state = last_recurrent_state * g_i_last_exp

    g_diff_exp = jnp.expand_dims(jnp.exp(jnp.expand_dims(g_i[..., -1], -1) - g_i), -1)
    k_i_g_diff = k_i * g_diff_exp

    update_term = jnp.einsum('...cd,...cv->...dv', k_i_g_diff, v_new, precision=prec)
    new_last_recurrent_state = new_last_recurrent_state + update_term

    return new_last_recurrent_state, core_attn_out_i

  final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

  core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
  core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
  core_attn_out = core_attn_out[:, :, :sequence_length, :]
  core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

  return core_attn_out, final_state if output_final_state else None


def jax_chunk_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: None | Array = None,
    pad_size: int = 0,
) -> tuple[Array, None | Array]:
  """Optimized JAX implementation of Gated Delta Rule."""
  # =========================================================================
  # STAGE 1: PREPARATION & PADDING
  # =========================================================================
  initial_dtype = query.dtype

  B, seq_len, H_k, R, K_dim = key.shape
  V_dim = value.shape[-1]

  num_chunks = query.shape[1] // chunk_size

  def to_chunk(x):
    return x.reshape(B, num_chunks, chunk_size, H_k, x.shape[3], -1).transpose(0, 1, 3, 4, 2, 5)

  def to_chunk_scalar(x):
    return x.reshape(B, num_chunks, chunk_size, H_k, x.shape[3]).transpose(0, 1, 3, 4, 2)

  q_c = to_chunk(query)
  k_c = to_chunk(key)
  v_c = to_chunk(value)
  g_c = to_chunk_scalar(g)
  beta_c = to_chunk_scalar(beta)

  # =========================================================================
  # STAGE 2: INTRA-CHUNK PRE-COMPUTATION (Parallel)
  # =========================================================================

  # Cumulative decay (Must be float32)
  g_cumsum = jnp.cumsum(g_c, axis=-1)
  k_beta = k_c * beta_c[..., None]

  # S Matrix Calculation
  S = jnp.einsum('...cd,...ed->...ce', k_beta, k_c, precision=jax.lax.Precision.HIGHEST)
  S = S.astype(jnp.float32)

  # Apply mask BEFORE exp to prevent 'inf' gradients
  g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
  mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = jnp.where(mask, g_diff, -1e30)

  S = S * jnp.exp(g_diff)
  S = S * mask.astype(S.dtype)

  # Inversion (A) - Strictly float32
  identity = jnp.eye(chunk_size, dtype=jnp.float32)
  identity_broadcasted = jnp.broadcast_to(identity, S.shape)

  A = jax.scipy.linalg.solve_triangular(identity + S, identity_broadcasted, lower=True, unit_diagonal=True)

  # 5. WY Factors
  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.einsum('...cd,...dv->...cv', A, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
  u_chunks = u_chunks.astype(compute_dtype)

  k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(g_cumsum)[..., None]
  w_chunks = jnp.einsum('...cd,...dv->...cv', A, k_beta_g, precision=jax.lax.Precision.HIGHEST)
  w_chunks = w_chunks.astype(compute_dtype)

  # =========================================================================
  # STAGE 3: INTER-CHUNK RECURRENCE (Scan)
  # =========================================================================
  scan_perm_vec = (1, 0, 2, 3, 4)
  scan_perm_scl = (1, 0, 2, 3)

  w_scan = w_chunks.transpose(scan_perm_vec)
  u_scan = u_chunks.transpose(scan_perm_vec)
  k_scan = k_c.transpose(scan_perm_vec)
  q_scan = q_c.transpose(scan_perm_vec)
  g_scan = g_cumsum.transpose(scan_perm_scl)

  if initial_state is None:
    h_init = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
  else:
    h_init = initial_state.astype(jnp.float32)

  xs = (w_scan, u_scan, q_scan, k_scan, g_scan)

  def scan_body(h, args):
    w, u, q, k, g = args
    prec = jax.lax.Precision.HIGHEST

    # --- Output Computation ---
    # 1. Inter-chunk: q(dtype) * exp(g)(f32) -> f32
    q_g = q.astype(jnp.float32) * jnp.exp(g)[..., None]
    attn_inter = jnp.einsum('...cd,...dv->...cv', q_g, h, precision=prec)

    # 2. Delta Rule Subtraction (v_prime and v_new)
    # w serves as k_cumdecay, u serves as value_intra
    v_prime = jnp.einsum('...cd,...dv->...cv', w.astype(jnp.float32), h, precision=prec)
    v_new = u.astype(jnp.float32) - v_prime

    # 3. Intra-chunk: q(dtype) @ k(dtype) -> f32
    attn = jnp.einsum('...cd,...ed->...ce', q, k, precision=prec)
    attn = attn.astype(jnp.float32)

    # Mask before exp
    g_diff = g[..., :, None] - g[..., None, :]
    mask_intra = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    g_diff = jnp.where(mask_intra, g_diff, -1e30)

    attn_i = attn * jnp.exp(g_diff)
    attn_i = attn_i * mask_intra.astype(attn_i.dtype)

    # Note: We do NOT multiply attn_i by beta here. The Delta rule mathematically
    # absorbed beta inside v_new (via u).

    # 4. Combine Core Output
    term2 = jnp.matmul(attn_i, v_new, precision=prec)
    o_c = attn_inter + term2

    # --- State Update ---
    g_i_last_exp = jnp.exp(g[..., -1, None, None])
    h_new = h * g_i_last_exp

    # Apply Delta Rule K decay to state
    g_diff_exp_state = jnp.exp(g[..., -1, None] - g)[..., None]
    k_i_g_diff = k.astype(jnp.float32) * g_diff_exp_state

    update_term = jnp.einsum('...cd,...cv->...dv', k_i_g_diff, v_new, precision=prec)
    h_new = h_new + update_term

    return h_new, o_c

  final_h, o_chunks = lax.scan(scan_body, h_init, xs)

  # =========================================================================
  # STAGE 4: FINALIZATION
  # =========================================================================
  o = o_chunks.transpose(0, 1, 4, 2, 3, 5).reshape(B, seq_len, H_k * R, V_dim)

  if pad_size > 0:
    o = o[:, :-pad_size, :, :]

  o = o.astype(initial_dtype)

  return o, (final_h if initial_state is not None else None)


def jax_ar_gated_delta_rule(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    initial_state: Array,
) -> tuple[Array, Array]:
  """Highly optimized step for Autoregressive Decoding (seq_len == 1)."""
  # Shapes: q, k (B, 1, H, K_dim) | v (B, 1, H, V_dim) | g, beta (B, 1, H)
  initial_dtype = query.dtype

  # Strip the seq_len=1 dimension to avoid broadcast overhead
  q = query.squeeze(1)
  k = key.squeeze(1)
  v = value.squeeze(1)
  g = g.squeeze(1)
  beta = beta.squeeze(1)

  g_exp = jnp.exp(g)[..., None]
  beta_exp = beta[..., None]

  k_beta = k * beta_exp
  v_beta = v * beta_exp

  state = initial_state.astype(jnp.float32)
  B_s, H_v, K_dim, V_dim = state.shape
  H_k = q.shape[1]
  R = H_v // H_k
  state = state.reshape(B_s, H_k, R, K_dim, V_dim)

  # v_prime = state @ (k_beta * exp(g))
  k_cumdecay = k_beta.astype(jnp.float32) * g_exp
  v_prime = jnp.einsum('...k,...kv->...v', k_cumdecay, state, precision=jax.lax.Precision.HIGHEST)

  v_new = v_beta.astype(jnp.float32) - v_prime

  # Core Output
  q_g = q.astype(jnp.float32) * g_exp
  attn_inter = jnp.einsum('...k,...kv->...v', q_g, state, precision=jax.lax.Precision.HIGHEST)

  attn_intra = jnp.sum(q.astype(jnp.float32) * k.astype(jnp.float32), axis=-1, keepdims=True)
  core_attn_out = attn_inter + attn_intra * v_new

  # State Update: new_state = state * exp(g) + k^T @ v_new
  update_term = jnp.einsum('...k,...v->...kv', k.astype(jnp.float32), v_new, precision=jax.lax.Precision.HIGHEST)
  new_state = state * g_exp[..., None] + update_term

  # Restore dimensions
  new_state = new_state.reshape(B_s, H_v, K_dim, V_dim)
  core_attn_out = core_attn_out.reshape(B_s, H_v, V_dim)
  core_attn_out = core_attn_out[:, None, :, :].astype(initial_dtype)

  return core_attn_out, new_state


class Qwen3NextGatedDeltaNet(nnx.Module):
  """
  This module implements the full end-to-end logic of a Gated Delta Network layer.

  End-to-End Equations Implemented:
  Let `x` be the input `hidden_states`.

  Step A: Input Projections
  1. (q_raw, k_raw, v_raw, z) = Linear_qkvz(x)
  2. (b, a) = Linear_ba(x)

  Step B: 1D Convolution
  1. qkv_conv = silu(Conv1D(concatenate(q_raw, k_raw, v_raw)))
  2. (q, k, v) = split(qkv_conv)

  Step C: Gated Delta Rule (Recurrent Core)
  1. Gates: β=sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
  2. Core Calculation: core_attn_out = jax_chunk_gated_delta_rule(q, k, v, g, β)

  Step D: Final Output Stage
  1. y = RMSNorm(core_attn_out) * silu(z)
  2. output = Linear_out(y)
  """

  def __init__(
      self,
      config: Config,
      inputs_shape: tuple | None = None,
      mesh=None,
      dtype: DType = jnp.float32,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    """
    Args:
      config: MaxText configuration object.
      mesh: Optional JAX device mesh (required for vLLM paged-state path).
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    """
    self.config = config
    self.mesh = mesh

    self._gdn_replicate_expert = os.environ.get("MAXTEXT_GDN_REPLICATE_EXPERT", "False").lower() == "true"
    cfg = self.config

    in_features = cfg.emb_dim
    self.num_v_heads = cfg.gdn_num_value_heads
    self.num_k_heads = cfg.gdn_num_key_heads
    self.head_k_dim = cfg.gdn_key_head_dim
    self.head_v_dim = cfg.gdn_value_head_dim
    self.key_dim = self.head_k_dim * self.num_k_heads
    self.value_dim = self.head_v_dim * self.num_v_heads
    conv_dim = self.key_dim * 2 + self.value_dim
    conv_kernel_size = cfg.gdn_conv_kernel_dim
    self.v_heads_per_k_head = self.num_v_heads // self.num_k_heads

    if model_mode != MODEL_MODE_TRAIN and inputs_shape is not None:
      runtime_batch_size = inputs_shape[0]

      self.cache = kvcache.KVCache(
          max_prefill_length=cfg.max_prefill_predict_length,
          max_target_length=cfg.max_target_length,
          batch=runtime_batch_size,
          key_seq_len=1,
          value_seq_len=1,
          key_heads=self.num_v_heads,
          value_heads=self.num_v_heads,
          key_head_size=self.head_k_dim,
          value_head_size=self.head_v_dim,
          dtype=dtype,
          is_gdn=True,
          conv_kernel_size=conv_kernel_size,
          conv_dim=conv_dim,
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      self.cache = None  # No cache for train mode or when inputs_shape not provided

    # Submodule instantiations
    self.in_proj_qkvz = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.num_k_heads, 2 * self.head_k_dim + 2 * self.head_v_dim * self.v_heads_per_k_head),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "gdn_head", None),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.in_proj_ba = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=(self.num_k_heads, 2 * self.v_heads_per_k_head),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "gdn_head", None),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    self.conv1d = nnx.Conv(
        in_features=conv_dim,
        out_features=conv_dim,
        kernel_size=(conv_kernel_size,),
        feature_group_count=conv_dim,  # Depthwise
        padding="CAUSAL",
        use_bias=False,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Initialize A_log to match torch.log(torch.uniform(0, 16))
    def a_log_init(key, shape, dtype=jnp.float32):
      # Sample from Uniform(epsilon, 16) to avoid log(0)
      a_vals = jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0)
      return jnp.log(a_vals)

    self.A_log = nnx.Param(a_log_init(rngs.params(), (self.num_v_heads,), dtype=cfg.weight_dtype))
    self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (self.num_v_heads,), dtype=cfg.weight_dtype))

    self.norm = Qwen3NextRMSNormGated(
        num_features=self.head_v_dim,  # Normalize over the head dimension (D_v)
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.out_proj = DenseGeneral(
        in_features_shape=(self.num_v_heads, self.head_v_dim),
        out_features_shape=(in_features,),
        axis=(-2, -1),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("gdn_head", None, "embed"),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      model_mode: str = MODEL_MODE_TRAIN,
      kv_cache=None,
      decoder_segment_ids: None | Array = None,
      attention_metadata=None,
      **kwargs,
  ) -> tuple[Array, Any | None]:
    # hidden_states: (B, S, E)
    cfg = self.config
    batch, seq_len, _ = hidden_states.shape

    active_cache = kv_cache if kv_cache is not None else self.cache

    # When kv_cache is a 2-tuple of paged mamba state arrays from vLLM, use
    # run_jax_gdn_attention from tpu_inference for correct sequential token processing.
    use_paged_state = (
        kv_cache is not None
        and isinstance(kv_cache, tuple)
        and len(kv_cache) == 2
        and attention_metadata is not None
        and getattr(attention_metadata, "mamba_state_indices", None) is not None
        and self.mesh is not None
    )

    # =========================================================================
    # STEP A: Input Projections
    # =========================================================================
    # mixed_qkvz: (B, S, H_k, 2 * D_k + 2 * V_per_K * D_v)
    mixed_qkvz = self.in_proj_qkvz(hidden_states)
    # mixed_ba: (B, S, H_k, 2 * V_per_K)
    mixed_ba = self.in_proj_ba(hidden_states)

    # =========================================================================
    # QKVZ and BA Splitting (shared by both paths)
    # =========================================================================
    if self.mesh is not None:
      logical_rules = (
          None
          if self.config.using_pipeline_parallelism
          else self.config.logical_axis_rules
      )
      qkvz_pspec = logical_to_mesh_axes((KV_BATCH, None, KV_HEAD, None), mesh=self.mesh, rules=logical_rules)
      # Training microbatches can be smaller than the physical KV_BATCH mesh partition.
      qkvz_pspec = remove_incompatible_mesh_axes_from_partition_spec(
          qkvz_pspec,
          mixed_qkvz.shape,
          self.mesh,
          dims=(0,),
          allow_remove_axes=True,
      )
      qkvz_sharding = jax.sharding.NamedSharding(self.mesh, qkvz_pspec)
      mixed_qkvz = jax.lax.with_sharding_constraint(mixed_qkvz, qkvz_sharding)

    D_k = self.head_k_dim
    V_per_k_D_v = self.v_heads_per_k_head * self.head_v_dim

    # query: (B, S, H_k, D_k)
    query = mixed_qkvz[..., :D_k]
    # key: (B, S, H_k, D_k)
    key = mixed_qkvz[..., D_k : 2 * D_k]
    # value: (B, S, H_v, D_v)
    value = mixed_qkvz[..., 2 * D_k : 2 * D_k + V_per_k_D_v].reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
    # z: (B, S, H_v, D_v)
    z = mixed_qkvz[..., 2 * D_k + V_per_k_D_v :].reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

    # b: (B, S, H_v)
    b = mixed_ba[..., :self.v_heads_per_k_head].reshape(batch, seq_len, self.num_v_heads)
    # a: (B, S, H_v)
    a = mixed_ba[..., self.v_heads_per_k_head:].reshape(batch, seq_len, self.num_v_heads)

    if use_paged_state:
      # =========================================================================
      # vLLM PAGED STATE PATH: use tpu_inference fused conv + ragged delta-rule.
      # =========================================================================
      try:
        # pylint: disable=import-outside-toplevel
        # pytype: disable=import-error
        from tpu_inference.layers.common.gdn_attention import GdnAttentionConfig, run_jax_gdn_attention
        from tpu_inference.layers.common.ragged_gated_delta_rule_wrapper import RaggedGatedDeltaRuleImpl
        from tpu_inference.layers.common.sharding import ShardingAxisName
        from tpu_inference.layers.common.utils import (
            reorder_concatenated_tensor_for_sharding,
            truncate_sharded_tensor,
        )
        from tpu_inference.utils import get_mesh_shape_product
        from jax.sharding import PartitionSpec as P_spec
      except ImportError as e:
        raise ImportError(
            "GDN attention kernel require the vllm-tpu package. Please install it with `pip install vllm-tpu`."
        ) from e

      attn_data = ShardingAxisName.ATTN_DATA
      # Head axis for the GDN kernel + the producer-side reshapes. Default ATTN_HEAD
      # (model*expert); the experimental MAXTEXT_GDN_REPLICATE_EXPERT path uses 'model' only
      # so GDN replicates over the expert axis (no expert-axis transpose all-to-all).
      attn_head = ShardingAxisName.MODEL if self._gdn_replicate_expert else ShardingAxisName.ATTN_HEAD
      tp_size = get_mesh_shape_product(self.mesh, attn_head)
      num_tokens = batch * seq_len

      # Build mixed_qkv in the kernel's per-shard layout via shard_map concatenation.
      # Each TP shard already holds its local q/k/v head slices → concatenate locally
      # to get [q_local | k_local | v_local] with no cross-device communication.
      q_flat = query.reshape(num_tokens, self.key_dim)  # (T, key_dim) sharded on ATTN_HEAD
      k_flat = key.reshape(num_tokens, self.key_dim)
      v_flat = value_raw.reshape(num_tokens, self.value_dim)  # (T, value_dim) sharded on ATTN_HEAD
      mixed_qkv = jax.shard_map(
          lambda q, k, v: jnp.concatenate([q, k, v], axis=-1),
          mesh=self.mesh,
          in_specs=(P_spec(attn_data, attn_head),) * 3,
          out_specs=P_spec(attn_data, attn_head),
          check_vma=False,
      )(q_flat, k_flat, v_flat)

      b_flat = b.reshape(num_tokens, self.num_v_heads)
      a_flat = a.reshape(num_tokens, self.num_v_heads)

      # Conv weight: transpose from (kernel_size, 1, conv_dim) → (conv_dim, 1, kernel_size),
      # then reorder so each TP shard gets its local [q_local | k_local | v_local] channels.
      conv_weight = jnp.transpose(self.conv1d.kernel.value, (2, 1, 0))
      conv_weight = reorder_concatenated_tensor_for_sharding(
          conv_weight, [self.key_dim, self.key_dim, self.value_dim], tp_size, 0
      )

      conv_state_paged, recurrent_state_paged = kv_cache

      # Use REF impl (pure JAX) to avoid Mosaic kernel compilation issues.
      gdn_config = GdnAttentionConfig(
          ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF
      )

      # Compile against the active request bucket rather than the runner's
      # maximum-size metadata buffers.
      dp_size = get_mesh_shape_product(self.mesh, attn_data)
      padded_num_reqs_per_dp = attention_metadata.padded_num_reqs // dp_size  # pyrefly: ignore[missing-attribute]
      state_indices = truncate_sharded_tensor(
          attention_metadata.mamba_state_indices.astype(jnp.int32),  # pyrefly: ignore[missing-attribute]
          padded_num_reqs_per_dp,
          dp_size,
      )
      query_start_loc = truncate_sharded_tensor(
          attention_metadata.query_start_loc,  # pyrefly: ignore[missing-attribute]
          padded_num_reqs_per_dp + 1,
          dp_size,
      )
      seq_lens = truncate_sharded_tensor(
          attention_metadata.seq_lens,  # pyrefly: ignore[missing-attribute]
          padded_num_reqs_per_dp,
          dp_size,
      )

      (new_conv_state_paged, new_recurrent_state_paged), gdn_output = (
          run_jax_gdn_attention(
              mixed_qkv,
              b_flat,
              a_flat,
              conv_state_paged,
              recurrent_state_paged,
              conv_weight,
              None,  # conv_bias: MaxText conv1d uses use_bias=False.
              jnp.asarray(self.A_log[...], dtype=cfg.dtype),
              jnp.asarray(self.dt_bias[...], dtype=cfg.dtype),
              state_indices,
              query_start_loc,
              attention_metadata.request_distribution,  # pyrefly: ignore[missing-attribute]
              seq_lens,
              self.num_k_heads,
              self.num_v_heads,
              self.head_k_dim,
              self.head_v_dim,
              cfg.gdn_conv_kernel_dim,
              mesh=self.mesh,
              config=gdn_config,
          )
      )

      # Reshape GDN output and apply gated norm + out projection.
      gdn_output = gdn_output.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
      gdn_output = checkpoint_name(gdn_output, "context")
      gated_output = self.norm(gdn_output, z)
      output = self.out_proj(gated_output)

      return output, (new_conv_state_paged, new_recurrent_state_paged)

    # Flatten head dimensions for concatenation before conv
    # q: (B, S, K_dim)
    q = query.reshape(batch, seq_len, -1)
    # k: (B, S, K_dim)
    k = key.reshape(batch, seq_len, -1)
    # v: (B, S, V_dim)
    v = value.reshape(batch, seq_len, -1)

    # =========================================================================
    # STEP B & C: 1D Convolution & Gated Delta Rule Recurrence
    # =========================================================================
    qkv = jnp.concatenate([q, k, v], axis=-1)
    batch, seq_len, _ = qkv.shape
    conv_kernel_size = self.config.gdn_conv_kernel_dim

    conv_state = None
    recurrent_state = None
    next_conv_state = None
    if model_mode != MODEL_MODE_TRAIN and active_cache is not None:
      recurrent_state, conv_state = active_cache.get_gdn_states()
      orig_cache_batch = conv_state.shape[0]

      if conv_state.shape[0] != batch:
        if conv_state.shape[0] == 1:
          conv_state = jnp.broadcast_to(conv_state, (batch,) + conv_state.shape[1:])
        elif conv_state.shape[0] < batch:
          pad_amt = batch - conv_state.shape[0]
          conv_state = jnp.pad(conv_state, ((0, pad_amt), (0, 0), (0, 0)))
        else:
          conv_state = conv_state[:batch]

      if recurrent_state.shape[0] != batch:
        if recurrent_state.shape[0] == 1:
          recurrent_state = jnp.broadcast_to(recurrent_state, (batch,) + recurrent_state.shape[1:])
        elif recurrent_state.shape[0] < batch:
          pad_amt = batch - recurrent_state.shape[0]
          recurrent_state = jnp.pad(recurrent_state, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
        else:
          recurrent_state = recurrent_state[:batch]

    if getattr(cfg, "use_gdn_kernel", False) and getattr(
        cfg, "use_hybrid_gdn", False
    ):
      from maxtext.models.hybrid_gdn import hybrid_fused_conv1d_gdn

      conv_state_arg = (
          conv_state
          if conv_state is not None
          else jnp.zeros(
              (batch, self.config.gdn_conv_kernel_dim - 1, qkv.shape[-1]),
              dtype=cfg.dtype,
          )
      )
      recurrent_state_arg = (
          recurrent_state
          if recurrent_state is not None
          else jnp.zeros(
              (batch, self.num_v_heads, self.head_k_dim, self.head_v_dim),
              dtype=cfg.dtype,
          )
      )
      conv_bias_arg = (
          self.conv1d.bias.value
          if hasattr(self.conv1d, "bias") and self.conv1d.bias is not None
          else jnp.zeros((qkv.shape[-1],), dtype=cfg.dtype)
      )
      if self.mesh is not None:
        logical_rules = get_logical_axis_rules()
        batch_pspec3 = logical_to_mesh_axes(
            (KV_BATCH, None, None), mesh=self.mesh, rules=logical_rules
        )
        batch_pspec4 = logical_to_mesh_axes(
            (KV_BATCH, None, None, None), mesh=self.mesh, rules=logical_rules
        )
        none_pspec3 = logical_to_mesh_axes(
            (None, None, None), mesh=self.mesh, rules=logical_rules
        )
        none_pspec1 = logical_to_mesh_axes(
            (None,), mesh=self.mesh, rules=logical_rules
        )

        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(
                batch_pspec3,  # qkv
                batch_pspec3,  # b
                batch_pspec3,  # a
                none_pspec3,  # conv_weight
                none_pspec1,  # conv_bias
                none_pspec1,  # a_log
                none_pspec1,  # dt_bias
                batch_pspec3,  # conv_state
                batch_pspec4,  # recurrent_state
            ),
            out_specs=(
                batch_pspec4,  # core_attn_out
                (
                    batch_pspec3,
                    batch_pspec4,
                ),  # (next_conv_state, next_recurrent_state)
            ),
            check_vma=False,
        )
        def shard_mapped_hybrid_gdn(
            qkv_val,
            b_val,
            a_val,
            cw_val,
            cb_val,
            alog_val,
            dt_val,
            cs_val,
            rs_val,
        ):
          return hybrid_fused_conv1d_gdn(
              qkv=qkv_val,
              b=b_val,
              a=a_val,
              conv_weight=cw_val,
              conv_bias=cb_val,
              a_log=alog_val,
              dt_bias=dt_val,
              conv_state=cs_val,
              recurrent_state=rs_val,
              num_k_heads=self.num_k_heads,
              num_v_heads=self.num_v_heads,
              head_k_dim=self.head_k_dim,
              head_v_dim=self.head_v_dim,
              conv_kernel_size=self.config.gdn_conv_kernel_dim,
              chunk_size=self.config.gdn_chunk_size,
              use_qk_norm_in_gdn=self.config.use_qk_norm_in_gdn,
              compute_dtype=self.config.dtype,
          )

        core_attn_out, (next_conv_state, next_recurrent_state) = (
            shard_mapped_hybrid_gdn(
                qkv,
                b,
                a,
                self.conv1d.kernel.value,
                conv_bias_arg,
                self.A_log[...],
                self.dt_bias[...],
                conv_state_arg,
                recurrent_state_arg,
            )
        )
      else:
        core_attn_out, (next_conv_state, next_recurrent_state) = (
            hybrid_fused_conv1d_gdn(
                qkv=qkv,
                b=b,
                a=a,
                conv_weight=self.conv1d.kernel.value,
                conv_bias=None,
                a_log=self.A_log[...],
                dt_bias=self.dt_bias[...],
                conv_state=conv_state_arg,
                recurrent_state=recurrent_state_arg,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                conv_kernel_size=self.config.gdn_conv_kernel_dim,
                chunk_size=self.config.gdn_chunk_size,
                use_qk_norm_in_gdn=self.config.use_qk_norm_in_gdn,
                compute_dtype=self.config.dtype,
            )
        )
    else:
      if conv_state is not None:
        conv_input = jnp.concatenate([conv_state, qkv], axis=1)
        if decoder_segment_ids is not None:
          valid_lens = jnp.sum(decoder_segment_ids != 0, axis=1)

          def extract_state(c_in, v_len):
            return jax.lax.dynamic_slice_in_dim(
                c_in, v_len, conv_kernel_size - 1, axis=0
            )

          next_conv_state = jax.vmap(extract_state)(conv_input, valid_lens)
        else:
          next_conv_state = conv_input[:, -(conv_kernel_size - 1) :, :]
      else:
        conv_input = jnp.pad(qkv, ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))

      conv_out = self.conv1d(conv_input)
      conv_out = conv_out[:, -seq_len:, :]
      qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(cfg.dtype)
      q_conv, k_conv, v_conv = jnp.split(
          qkv_conv, [self.key_dim, 2 * self.key_dim], axis=-1
      )

      query = q_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
      key = k_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
      value = v_conv.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

      A_log = jnp.asarray(self.A_log[...], dtype=cfg.dtype)
      dt_bias = jnp.asarray(self.dt_bias[...], dtype=cfg.dtype)
      beta = jax.nn.sigmoid(b)
      g = -jnp.exp(A_log) * jax.nn.softplus(a + dt_bias)

      if decoder_segment_ids is not None:
        mask = decoder_segment_ids != 0
        key = jnp.where(mask[..., None, None], key, 0.0)
        value = jnp.where(mask[..., None, None], value, 0.0)
        g = jnp.where(mask[..., None], g, 0.0)

      if cfg.use_qk_norm_in_gdn:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

      query = query.astype(cfg.dtype)
      key = key.astype(cfg.dtype)
      value = value.astype(cfg.dtype)
      beta = beta.astype(cfg.dtype)
      g = g.astype(jnp.float32)

      scale = (query.shape[-1] ** -0.5)
      query = query * scale

      # Pad before expansion if chunking
      pad_size = 0
      if getattr(cfg, "use_gdn_kernel", False) and seq_len > 1:
        pad_size = (cfg.gdn_chunk_size - (seq_len % cfg.gdn_chunk_size)) % cfg.gdn_chunk_size
        if pad_size > 0:
          query = jnp.pad(query, ((0, 0), (0, pad_size), (0, 0), (0, 0)))
          key = jnp.pad(key, ((0, 0), (0, pad_size), (0, 0), (0, 0)))
          value = jnp.pad(value, ((0, 0), (0, pad_size), (0, 0), (0, 0)))
          beta = jnp.pad(beta, ((0, 0), (0, pad_size), (0, 0)))
          g = jnp.pad(g, ((0, 0), (0, pad_size), (0, 0)))

      if (
          self.num_v_heads > self.num_k_heads
          and self.num_v_heads % self.num_k_heads == 0
      ):
        repeats = self.num_v_heads // self.num_k_heads
        query = query.reshape(batch, seq_len, self.num_k_heads, 1, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, 1, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_k_heads, repeats, self.head_v_dim)
        beta = beta.reshape(batch, seq_len, self.num_k_heads, repeats)
        g = g.reshape(batch, seq_len, self.num_k_heads, repeats)
      else:
        query = query[:, :, :, None, :]
        key = key[:, :, :, None, :]
        value = value[:, :, :, None, :]
        beta = beta[:, :, :, None]
        g = g[:, :, :, None]

      if seq_len == 1 and model_mode == MODEL_MODE_AUTOREGRESSIVE:
        core_attn_out, next_recurrent_state = jax_ar_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=recurrent_state,
        )
      elif getattr(cfg, "use_gdn_kernel", False):
        core_attn_out, next_recurrent_state = jax_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=cfg.gdn_chunk_size,
            initial_state=recurrent_state,
            pad_size=pad_size,
        )
      elif self.mesh is not None:
        logical_rules = self.config.logical_axis_rules
        recurrent_state_arg = (
            recurrent_state
            if recurrent_state is not None
            else jnp.zeros(
                (batch, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                dtype=cfg.dtype,
            )
        )
        qkv_pspec = logical_to_mesh_axes((KV_BATCH, None, KV_HEAD, None), mesh=self.mesh, rules=logical_rules)
        g_beta_pspec = logical_to_mesh_axes((KV_BATCH, None, KV_HEAD), mesh=self.mesh, rules=logical_rules)
        state_pspec = logical_to_mesh_axes((KV_BATCH, KV_HEAD, None, None), mesh=self.mesh, rules=logical_rules)
        # Keep every shard_map input/output batch spec consistent when replication is required.
        qkv_pspec = remove_incompatible_mesh_axes_from_partition_spec(
            qkv_pspec,
            query.shape,
            self.mesh,
            dims=(0,),
            allow_remove_axes=True,
        )
        g_beta_pspec = remove_incompatible_mesh_axes_from_partition_spec(
            g_beta_pspec,
            g.shape,
            self.mesh,
            dims=(0,),
            allow_remove_axes=True,
        )
        state_pspec = remove_incompatible_mesh_axes_from_partition_spec(
            state_pspec,
            recurrent_state_arg.shape,
            self.mesh,
            dims=(0,),
            allow_remove_axes=True,
        )

        @functools.partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(
                qkv_pspec,  # query
                qkv_pspec,  # key
                qkv_pspec,  # value
                g_beta_pspec,  # g
                g_beta_pspec,  # beta
                state_pspec,  # initial_state
            ),
            out_specs=(
                qkv_pspec,  # core_attn_out
                state_pspec,  # final_state
            ),
            check_vma=False,
        )
        def shard_mapped_delta_rule(q, k, v, g_val, beta_val, init_h):
          return jax_chunk_gated_delta_rule(
              query=q,
              key=k,
              value=v,
              g=g_val,
              beta=beta_val,
              chunk_size=cfg.gdn_chunk_size,
              initial_state=init_h,
              use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
              compute_dtype=cfg.dtype,
          )

        core_attn_out, next_recurrent_state = shard_mapped_delta_rule(
            query, key, value, g, beta, recurrent_state_arg
        )
      else:
        core_attn_out, next_recurrent_state = jax_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=cfg.gdn_chunk_size,
            initial_state=recurrent_state,
            use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
            compute_dtype=cfg.dtype,
        )

    if model_mode != MODEL_MODE_TRAIN and active_cache is not None:
      assert next_conv_state is not None
      assert next_recurrent_state is not None
      if next_conv_state.shape[0] != orig_cache_batch:
        if next_conv_state.shape[0] == 1:
          next_conv_state = jnp.broadcast_to(next_conv_state, (orig_cache_batch,) + next_conv_state.shape[1:])
          next_recurrent_state = jnp.broadcast_to(
              next_recurrent_state, (orig_cache_batch,) + next_recurrent_state.shape[1:]
          )
        elif next_conv_state.shape[0] < orig_cache_batch:
          pad_amt = orig_cache_batch - next_conv_state.shape[0]
          next_conv_state = jnp.pad(next_conv_state, ((0, pad_amt), (0, 0), (0, 0)))
          next_recurrent_state = jnp.pad(next_recurrent_state, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
        else:
          next_conv_state = next_conv_state[:orig_cache_batch]
          next_recurrent_state = next_recurrent_state[:orig_cache_batch]

    if model_mode != MODEL_MODE_TRAIN and active_cache is not None:
      active_cache.update_gdn_states(next_recurrent_state, next_conv_state)

    core_attn_out = checkpoint_name(core_attn_out, "context")

    # =========================================================================
    # STEP D: Final Output Stage
    # =========================================================================

    # The normalization and gating is applied per-head on the value dimension.

    # Apply the norm and gate. Output shape: (B, S, H_v, D_v)
    gated_output = self.norm(core_attn_out, z)

    # Final output shape: (B, S, E)
    output = self.out_proj(gated_output)

    return output, active_cache

  def init_kv_caches(self, batch_size: int):
    """Initializes KVCache dynamically using the traced runtime batch size."""
    cfg = self.config
    conv_dim = self.key_dim * 2 + self.value_dim
    conv_kernel_size = cfg.gdn_conv_kernel_dim

    return kvcache.KVCache(
        max_prefill_length=cfg.max_prefill_predict_length,
        max_target_length=cfg.max_target_length,
        batch=batch_size,
        key_seq_len=1,
        value_seq_len=1,
        key_heads=self.num_v_heads,
        value_heads=self.num_v_heads,
        key_head_size=self.head_k_dim,
        value_head_size=self.head_v_dim,
        dtype=self.dtype,
        is_gdn=True,
        conv_kernel_size=conv_kernel_size,
        conv_dim=conv_dim,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )


class Qwen3NextFullAttention(nnx.Module):
  """Qwen3-Next Full Attention Layer.

  This module implements the full self-attention mechanism as used in
  Qwen3-Next models for layers that do not use the Gated Delta Network.
  It wraps the main `attentions.Attention` class, which handles the core attention operation,
  including the query, key, value, and output projections.

  Qwen3 Next Attention differs from standard attention by the following features:
    - Query and Gate splitting from a single q projection.
    - Application of a sigmoid gate to the attention output.
    - Usage of `Qwen3NextRMSNorm` for query and key normalization.
    - Usage of `PartialRotaryEmbedding` for partial rotary position embeddings.
      - Partial ROPE is applied to the first 25% of head dimensions

  Attributes:
    config: MaxText configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer.
    quant: Optional quantization configuration.
    attention: An instance of `attentions.Attention` which contains the
      learnable parameters for query, key, value, and output projections
      (e.g., `attention.query`, `attention.key`, etc.), and performs
      the attention calculation.
  """

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config

    scaling_factor = self.config.head_dim**-0.5
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.attention = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        out_axis_names=(BATCH, LENGTH, EMBED),
        mesh=self.mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        use_qk_norm=cfg.use_qk_norm,
        query_pre_attn_scalar=scaling_factor,
        model_mode=model_mode,
        use_mrope=cfg.use_mrope,
        mrope_section=cfg.mrope_section,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    attention_output, kv_cache = self.attention(
        inputs_q=inputs,
        inputs_kv=inputs,
        inputs_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    return attention_output, kv_cache


class Qwen3NextSparseMoeBlock(nnx.Module):
  """
  This module encapsulates the unique MoE structure of Qwen3-Next, which includes:
  1. A set of routed experts, where each token is sent to a subset of experts.
  2. A single shared expert, which all tokens pass through.
  3. A learnable gate that determines the contribution of the shared expert.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    quant: Optional quantization configuration.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = self.config

    # 1. Instantiate and apply the routed experts block.
    self.routed_experts = moe.RoutedMoE(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=max_initializers.nd_dense_init(cfg.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.moe_mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

    # 2. Instantiate and apply the shared expert(s).
    shared_expert_mlp_dim = maxtext_utils.get_shared_expert_mlp_dim(cfg)
    self.shared_expert = MlpBlock(
        config=cfg,
        mesh=mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.shared_experts * shared_expert_mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
        model_mode=config.model_call_mode,
        rngs=rngs,
    )

    # 3. Instantiate the (optional) gate for the shared expert.
    if cfg.moe_shared_expert_gate:
      self.shared_expert_gate = DenseGeneral(
          in_features_shape=cfg.emb_dim,
          out_features_shape=1,
          use_bias=False,  # Qwen3-Next shared_expert_gate does not have a bias
          dtype=cfg.dtype,
          kernel_init=max_initializers.nd_dense_init(
              cfg.dense_init_scale, "fan_in", "truncated_normal"
          ),
          kernel_axes=("embed", None),
          matmul_precision=cfg.matmul_precision,
          rngs=rngs,
      )
    else:
      self.shared_expert_gate = None

  def __call__(
      self, hidden_states: Array, deterministic: bool
  ) -> tuple[Array, Array | None, Array | None]:
    """Applies the sparse MoE block to the input hidden states.

    Args:
      hidden_states: The input array from the previous layer. Shape: (batch,
        seq, embed_dim)
      deterministic: If True, disables dropout.

    Returns:
      A tuple containing:
        - The output array of the MoE block.
        - The load balancing loss from the routed experts, if applicable during
        training.
        - The aux-loss-free expert-bias updates from the routed experts, if
        applicable.
    """
    # 1. Apply the routed experts block.
    routed_output, load_balance_loss, moe_bias_updates = self.routed_experts(
        hidden_states
    )

    # 2. Apply the shared expert.
    shared_expert_output = self.shared_expert(hidden_states, deterministic=deterministic)

    # 3. Apply the (optional) gate for the shared expert.
    if self.shared_expert_gate is not None:
      shared_gate_output = self.shared_expert_gate(hidden_states)
      shared_expert_output = (
          jax.nn.sigmoid(shared_gate_output) * shared_expert_output
      )

    # 4. Combine the outputs.
    final_output = routed_output + shared_expert_output

    return final_output, load_balance_loss, moe_bias_updates


class Qwen3NextScannableBlock(nnx.Module):
  """A scannable block of Qwen3-Next decoder layers with hierarchical nested scans.

  Linear attention layers (local) are scanned via
  `nnx_scan.apply_scanned_layers`
  while full attention (global) is scanned via a length-1 `jax.lax.scan`.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      *,
      num_of_layers: int | None = None,
      layer_idx_offset: int = 0,
      remat_policy_fn: Any | None = None,
      apply_internal_remat: bool = False,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.remat_policy_fn = remat_policy_fn
    self.apply_internal_remat = apply_internal_remat
    cfg = self.config
    if num_of_layers is None:
      num_of_layers = cfg.inhomogeneous_layer_cycle_interval
    self.num_of_layers = num_of_layers
    self.layer_idx_offset = layer_idx_offset

    cycle_interval = cfg.inhomogeneous_layer_cycle_interval
    full_attention_offset = (
        getattr(cfg, "full_attention_layer_offset", 0) % cycle_interval
    )

    self.num_local = sum(
        1
        for i in range(num_of_layers)
        if (layer_idx_offset + i) % cycle_interval != full_attention_offset
    )
    self.num_global = sum(
        1
        for i in range(num_of_layers)
        if (layer_idx_offset + i) % cycle_interval == full_attention_offset
    )

    if self.num_local > 0:
      self.local_layers = nnx_scan.create_scanned_layers(
          lambda layer_rngs: Qwen3NextDecoderLayer(
              config=self.config,
              mesh=self.mesh,
              model_mode=self.model_mode,
              quant=self.quant,
              layer_idx=0,
              is_dense_layer=False,
              is_full_attention_layer=False,
              rngs=layer_rngs,
          ),
          length=self.num_local,
          param_scan_axis=self.config.param_scan_axis,
          metadata_axis_name="local_layers",
          rngs=self.rngs,
      )
    else:
      self.local_layers = None

    if self.num_global > 0:
      self.global_layer = Qwen3NextDecoderLayer(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          layer_idx=full_attention_offset,
          is_dense_layer=False,
          is_full_attention_layer=True,
          rngs=self.rngs,
      )
    else:
      self.global_layer = None

  def _run_layer(self, layer, y, layer_kwargs, kv_cache=None):
    """Invokes one Qwen3NextDecoderLayer, returning (output, updated_kv_cache)."""
    out = layer(y, **layer_kwargs, kv_cache=kv_cache)
    return out if isinstance(out, tuple) else (out, None)

  @property
  def _remat_enabled(self):
    """Whether the block rematerializes its own layers."""
    return self.apply_internal_remat and self.config.remat_policy != "none"

  def _scan_local_layers(self, y, layer_kwargs):
    """Runs the local (linear attention / GatedDeltaNet) layers via a per-layer rematerialized jax.lax.scan."""
    remat = self._remat_enabled
    return nnx_scan.apply_scanned_layers(
        self.local_layers,
        y,
        length=self.num_local,
        param_scan_axis=self.config.param_scan_axis,
        apply_fn=lambda layer, carry: self._run_layer(
            layer, carry, layer_kwargs
        )[0],
        remat=remat,
        remat_policy=self.remat_policy_fn if remat else None,
        prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config)
        if remat
        else True,
    )

  def _scan_global_layer(self, y, layer_kwargs):
    """Runs the single global-attention layer inside a length-1 jax.lax.scan."""
    cfg = self.config
    graphdef_g, intermediate_g, other_g = nnx.split(
        self.global_layer, nnx.Intermediate, ...
    )
    intermediate_xs = jax.tree.map(lambda x: x[None], intermediate_g)

    def run_global_layer(carry, intermediate_slice):
      hidden_states, other = carry
      layer = nnx.merge(graphdef_g, intermediate_slice, other)
      new_hidden_states = self._run_layer(layer, hidden_states, layer_kwargs)[0]
      _, new_intermediate, new_other = nnx.split(layer, nnx.Intermediate, ...)
      return (new_hidden_states, new_other), new_intermediate

    global_remat_policy = self.remat_policy_fn
    offload_names = maxtext_utils.get_save_and_offload_names(cfg)
    if offload_names[0] or offload_names[1]:
      save_names, offload_to_device = offload_names
      global_remat_policy = jax.checkpoint_policies.save_only_these_names(
          *(save_names + offload_to_device)
      )

    if self._remat_enabled:
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
      run_global_layer = jax.checkpoint(
          run_global_layer,
          policy=global_remat_policy,
          prevent_cse=prevent_cse,
      )

    with xla_metadata.set_xla_metadata(
        **{"skip-simplify-while-loops_trip-count-one": "true"}
    ):
      (y, final_other), stacked_intermediate = jax.lax.scan(
          run_global_layer,
          (y, other_g),
          intermediate_xs,
          length=1,
      )

    intermediate_state = jax.tree.map(lambda x: x[0], stacked_intermediate)
    nnx.update(self.global_layer, final_other, intermediate_state)
    return y

  def _forward_with_external_kv_cache(self, y, kv_cache, layer_kwargs):
    """Runs the block with externally-supplied per-layer kv caches."""
    updated_kvs = []
    if self.local_layers is not None:
      graphdef, params, state = nnx.split(self.local_layers, nnx.Param, ...)
      scan_axis = self.config.param_scan_axis
      if scan_axis != 0:
        params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)
      per_layer_states = []
      for i in range(self.num_local):
        current_params = jax.tree.map(lambda x, i=i: x[i], params)
        current_state = jax.tree.map(lambda x, i=i: x[i], state)
        layer = nnx.merge(graphdef, current_params, current_state)
        current_kv = (
            kv_cache[i]
            if (kv_cache is not None and i < len(kv_cache))
            else None
        )
        y, new_kv = self._run_layer(layer, y, layer_kwargs, current_kv)
        updated_kvs.append(new_kv)
        per_layer_states.append(nnx.state(layer))

      stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs), *per_layer_states)
      if scan_axis != 0:
        stacked_params, stacked_other = stacked_state.split(nnx.Param, ...)
        stacked_params = jax.tree.map(
            lambda x: jnp.moveaxis(x, 0, scan_axis), stacked_params
        )
        stacked_state = nnx.State.merge(stacked_params, stacked_other)
      nnx.update(self.local_layers, stacked_state)

    if self.global_layer is not None:
      global_kv = (
          kv_cache[self.num_local]
          if (kv_cache is not None and self.num_local < len(kv_cache))
          else None
      )
      y, new_kv = self._run_layer(self.global_layer, y, layer_kwargs, global_kv)
      updated_kvs.append(new_kv)

    return y, tuple(updated_kvs)

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray = None,
      decoder_positions: None | jnp.ndarray = None,
      deterministic: bool = False,
      model_mode: str = "train",
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
  ) -> tuple[Array, None]:
    cfg = self.config
    inputs = carry
    inputs = nn.with_logical_constraint(
        inputs,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    layer_kwargs = {
        "decoder_segment_ids": decoder_segment_ids,
        "decoder_positions": decoder_positions,
        "deterministic": deterministic,
        "model_mode": model_mode,
        "slot": slot,
        "previous_chunk": previous_chunk,
        "attention_metadata": attention_metadata,
    }

    if kv_cache is not None:
      return self._forward_with_external_kv_cache(
          inputs, kv_cache, layer_kwargs
      )

    y = inputs
    if self.local_layers is not None:
      y = self._scan_local_layers(y, layer_kwargs)
    if self.global_layer is not None:
      y = self._scan_global_layer(y, layer_kwargs)

    if cfg.scan_layers:
      return y, None
    return y


class Qwen3NextDecoderLayer(nnx.Module):
  """This layer is a hybrid, capable of functioning as either: 1.

  A standard attention + MoE layer. 2. A linear attention + MoE layer.

  The first `config.first_num_dense_layers` layers (by `layer_idx`) use a plain
  dense MLP instead of MoE, and always use full attention, mirroring DeepSeek
  V3's
  dense-prefix pattern (see `models/deepseek.py::DeepSeekDenseLayer`).

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      is_dense_layer: bool | None = None,
      is_full_attention_layer: bool | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.is_mhc_enabled = cfg.mhc_expansion_rate > 1

    if is_dense_layer is None:
      is_dense_layer = layer_idx < cfg.first_num_dense_layers
    self.is_dense_layer = is_dense_layer

    # First LayerNorm, applied before the attention block.
    self.input_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Determine the type of attention mechanism for the current layer. Dense layers
    # always use full attention (see class docstring). `full_attention_layer_offset`
    # picks which position in the cycle is full attention; -1 (Python's negative-modulo
    # wraps to cycle-1) reproduces the original "last position in the cycle" schedule.
    full_attention_offset = (
        cfg.full_attention_layer_offset % cfg.inhomogeneous_layer_cycle_interval
    )
    if is_full_attention_layer is None:
      is_full_attention_layer = (
          self.is_dense_layer
          or self.layer_idx % cfg.inhomogeneous_layer_cycle_interval
          == full_attention_offset
      )
    self.is_full_attention_layer = is_full_attention_layer

    # Conditionally instantiate either the Linear Attention or Full Attention block.
    if is_full_attention_layer:
      self.attention = Qwen3NextFullAttention(
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=model_mode,
          layer_idx=self.layer_idx,
          rngs=rngs,
      )
    else:
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
      self.attention = Qwen3NextGatedDeltaNet(
          config=cfg, inputs_shape=dummy_inputs_shape, mesh=self.mesh, dtype=cfg.dtype, model_mode=model_mode, rngs=rngs
      )

    # Second LayerNorm, applied before the MoE block.
    self.post_attention_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    # Dense layers use a plain MLP; all other layers use `Qwen3NextSparseMoeBlock`.
    if self.is_dense_layer:
      self.mlp = MlpBlock(
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      self.mlp = Qwen3NextSparseMoeBlock(
          config=cfg, mesh=self.mesh, quant=self.quant, rngs=rngs
      )

    # Manifold-Constrained Hyper Connections: replaces the plain residual add around the
    # attention and MoE branches with a learned multi-stream mixing. See maxtext/layers/mhc.py
    # and models/deepseek4.py::DeepSeek4DecoderLayer for the reference implementation.
    if self.is_mhc_enabled:
      self.mhc_attention = mhc.ManifoldConstrainedHyperConnections(
          cfg, cfg.emb_dim, self.mesh, rngs
      )
      self.mhc_mlp = mhc.ManifoldConstrainedHyperConnections(
          cfg, cfg.emb_dim, self.mesh, rngs
      )

  def pre_attention_norm_op(self, x):
    normed = self.input_layernorm(x)
    return nn.with_logical_constraint(normed, self.activation_axis_names)

  def post_attention_norm_op(self, x):
    normed = self.post_attention_layernorm(x)
    return nn.with_logical_constraint(normed, self.activation_axis_names)

  def attention_branch(
      self,
      inputs_q,
      inputs_kv=None,
      decoder_segment_ids=None,
      inputs_positions=None,
      deterministic=None,
      model_mode=None,
      kv_cache=None,
      attention_metadata=None,
      **kwargs,
  ):
    """Adapts Qwen3-Next's two attention variants to mHC's inputs_q/inputs_kv branch_fn convention."""
    del inputs_kv, kwargs
    if isinstance(self.attention, Qwen3NextFullAttention):
      return self.attention(
          inputs_q,
          decoder_segment_ids,
          inputs_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    return self.attention(
        inputs_q,
        model_mode=model_mode,
        kv_cache=kv_cache,
        decoder_segment_ids=decoder_segment_ids,
        attention_metadata=attention_metadata,
    )

  def mlp_op(self, inputs, deterministic, *args, **kwargs):
    """Adapts the dense/MoE MLP's return shape to mHC's MLP_MOE 3-tuple convention."""
    del args, kwargs
    if self.is_dense_layer:
      return self.mlp(inputs, deterministic=deterministic), None, None
    mlp_out, load_balance_loss, moe_bias_updates = self.mlp(
        inputs, deterministic=deterministic
    )
    return mlp_out, load_balance_loss, moe_bias_updates

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    if self.is_mhc_enabled:
      mhc_expand, mhc_reduce = mhc.get_functions(self.config.mhc_expansion_rate)
      inputs = mhc_expand(inputs)

      intermediate_inputs, _ = self.mhc_attention(
          self.pre_attention_norm_op,
          self.attention_branch,
          x=inputs,
          mhc_type=HyperConnectionType.ATTENTION,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )

      layer_output, metadata = self.mhc_mlp(
          self.post_attention_norm_op,
          self.mlp_op,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_MOE,
          deterministic=deterministic,
      )
      load_balance_loss = metadata.get("load_balance_loss", None)
      if (
          self.config.load_balance_loss_weight > 0.0
          and load_balance_loss is not None
      ):
        self.moe_lb_loss = nnx.Intermediate(load_balance_loss)
      moe_bias_updates = metadata.get("moe_bias_updates", None)
      if (
          self.config.routed_bias
          and self.config.routed_bias_update_rate > 0.0
          and moe_bias_updates is not None
      ):
        self.moe_bias_updates = nnx.Intermediate(moe_bias_updates)

      layer_output = mhc_reduce(layer_output)
      layer_output = nn.with_logical_constraint(
          layer_output, self.activation_axis_names
      )
      return layer_output, kv_cache

    residual = inputs

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Conditionally apply either the Linear Attention or Full Attention block.
    if isinstance(self.attention, Qwen3NextFullAttention):
      attention_output, new_kv_cache = cast(
          Qwen3NextFullAttention, self.attention
      )(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      attention_output, new_kv_cache = cast(Qwen3NextGatedDeltaNet, self.attention)(
          hidden_states,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
          attention_metadata=attention_metadata,
      )

    # First residual connection after attention
    hidden_states = residual + attention_output
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Prepare for the MoE block by capturing the new residual
    residual = hidden_states

    # Second LayerNorm, applied before the MoE block.
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)

    # Apply the dense MLP or `Qwen3NextSparseMoeBlock`.
    if self.is_dense_layer:
      mlp_output = self.mlp(hidden_states, deterministic=deterministic)
    else:
      mlp_output, load_balance_loss, moe_bias_updates = self.mlp(
          hidden_states, deterministic=deterministic
      )
      # We sow the load balancing loss so it can be collected and added to the total loss
      # during training.
      if (
          self.config.load_balance_loss_weight > 0.0
          and load_balance_loss is not None
      ):
        self.moe_lb_loss = nnx.Intermediate(load_balance_loss)
      if (
          self.config.routed_bias
          and self.config.routed_bias_update_rate > 0.0
          and moe_bias_updates is not None
      ):
        self.moe_bias_updates = nnx.Intermediate(moe_bias_updates)

    # Final residual connection (after the MoE block)
    layer_output = residual + mlp_output
    layer_output = nn.with_logical_constraint(
        layer_output,
        self.activation_axis_names,
    )
    return layer_output, new_kv_cache


# -----------------------------------------
# The Base Decoder Layer for Qwen3
# -----------------------------------------
class AttentionWithNorm(nnx.Module):
  """Base class with shared common components: self-attention block with normalization."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    # Corresponds to Qwen3's `input_layernorm`
    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Self-attention block
    query_pre_attn_scalar = config.head_dim**-0.5  # Qwen3 specific scaling
    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        use_qk_norm=config.use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        model_mode=model_mode,
        use_mrope=config.use_mrope,
        mrope_section=config.mrope_section,
        rngs=rngs,
    )

    # Post Attention LayerNorm (corresponds to Qwen3's `post_attention_layernorm`)
    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

  def apply_attention_with_norm(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    """Applies self-attention with pre and post-layer normalization."""
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # Pre attention norm
    lnx = self.pre_self_attention_layer_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)
    # Self attention
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)
    # Residual connection after attention
    intermediate_inputs = inputs + attention_lnx
    # Post attention norm
    hidden_states = self.post_self_attention_layer_norm(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, self.activation_axis_names)
    return hidden_states, intermediate_inputs, kv_cache


# -----------------------------------------
# The Dense Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3DecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (dense)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    mlp_lnx = self.mlp(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    return layer_output, kv_cache


# -----------------------------------------
# The MoE Decoder Layer for Qwen3
# -----------------------------------------
class Qwen3MoeDecoderLayer(AttentionWithNorm):
  """Qwen3 Transformer decoder layer (MoE)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, model_mode, quant, rngs)
    self.moe_block = RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=config.moe_mlp_dim,  # same as config.mlp_dim
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | jnp.ndarray = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    is_scan_carry = False
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      kv_cache = stacked_kv_cache[layer_idx]
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states, intermediate_inputs, kv_cache = self.apply_attention_with_norm(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    mlp_lnx, load_balance_loss, _ = self.moe_block(hidden_states)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.moe_lb_loss = nnx.Intermediate(load_balance_loss)

    layer_output = intermediate_inputs + mlp_lnx
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if is_scan_carry:

      def update_cache(cache, val):
        if jnp.size(val) > 0:
          return cache.at[layer_idx].set(val)
        return cache

      stacked_kv_cache = jax.tree_util.tree_map(
          update_cache, stacked_kv_cache, kv_cache
      )
      return (layer_output, stacked_kv_cache, layer_idx + 1), None
    else:
      return layer_output, kv_cache


class Qwen3OmniMoeVisionPatchMerger(nnx.Module):
  """Vision patch merger that spatially merges patches using an MLP.

  Attributes:
      config: Config containing model parameters
      hidden_size: Hidden dimension after spatial merging
      use_postshuffle_norm: Whether to apply normalization after spatial shuffle
      dtype: Data type for computation
      weight_dtype: Data type for weights
      kernel_init: Initializer for kernel weights
      rngs: RNG state for initialization
      ln_q: LayerNorm before MLP
      mlp_0: First MLP layer
      mlp_2: Second MLP layer
  """

  def __init__(
      self,
      config: Config,
      use_postshuffle_norm: bool = False,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: max_initializers.NdInitializer = max_initializers.nd_dense_init(
          1.0, "fan_in", "normal"
      ),
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3Omni vision patch merger.

    Args:
        config: Config containing model parameters
        use_postshuffle_norm: Whether to apply normalization after spatial shuffle
        dtype: Data type for computation
        weight_dtype: Data type for weights
        kernel_init: Initializer for kernel weights
        rngs: RNG state for initialization
    """
    self.config = config
    self.use_postshuffle_norm = use_postshuffle_norm
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init
    self.rngs = rngs

    # Calculate hidden_size after spatial merge
    spatial_merge_size = config.spatial_merge_size_for_vit
    base_hidden_size = config.hidden_size_for_vit
    out_hidden_size = config.out_hidden_size_for_vit

    self.hidden_size = base_hidden_size * (spatial_merge_size**2)

    # LayerNorm before MLP
    ln_features = self.hidden_size if use_postshuffle_norm else base_hidden_size
    self.ln_q = nnx.LayerNorm(
        num_features=ln_features,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        rngs=rngs,
    )

    # MLP layers: Linear -> GELU -> Linear
    self.mlp_0 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.mlp_2 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=out_hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden: Array) -> Array:
    """
    Args:
        hidden: Input tensor of shape (batch, seq_len, base_hidden_size) after spatial reordering

    Returns:
        Output tensor of shape (batch, seq_len//merge_size**2, out_hidden_size) - spatially merged
    """
    # Get dimensions
    spatial_merge_size = self.config.spatial_merge_size_for_vit
    base_hidden_size = self.config.hidden_size_for_vit
    tokens_per_block = spatial_merge_size**2

    batch_size = hidden.shape[0]
    seq_len = hidden.shape[1]
    num_blocks = seq_len // tokens_per_block

    hidden = hidden.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    # Apply layer norm
    if self.use_postshuffle_norm:
      hidden = self.ln_q(hidden)
    else:
      hidden_unmerged = hidden.reshape(batch_size, seq_len, base_hidden_size)
      hidden_unmerged = self.ln_q(hidden_unmerged)
      hidden = hidden_unmerged.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    # MLP: Linear -> GELU -> Linear
    hidden = self.mlp_0(hidden)
    hidden = jax.nn.gelu(hidden)
    hidden = self.mlp_2(hidden)

    return hidden


class Qwen3OmniMoeVisionMLP(nnx.Module):
  """Vision MLP block with GELU activation.

  Attributes:
      config: Config containing model parameters
      hidden_size: Hidden dimension size
      intermediate_size: Intermediate dimension size
      dtype: Data type for computation
      weight_dtype: Data type for weights
      kernel_init: Initializer for kernel weights
      rngs: RNG state for initialization
      linear_fc1: First linear layer
      linear_fc2: Second linear layer
  """

  def __init__(
      self,
      config: Config,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: max_initializers.NdInitializer = max_initializers.nd_dense_init(
          1.0, "fan_in", "normal"
      ),
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3Omni vision MLP.

    Args:
        config: Config containing model parameters
        dtype: Data type for computation
        weight_dtype: Data type for weights
        kernel_init: Initializer for kernel weights
        rngs: RNG state for initialization
    """
    self.config = config
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init
    self.rngs = rngs

    self.hidden_size = config.hidden_size_for_vit
    self.intermediate_size = config.intermediate_size_for_vit

    self.linear_fc1 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.intermediate_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.linear_fc2 = DenseGeneral(
        in_features_shape=self.intermediate_size,
        out_features_shape=self.hidden_size,
        use_bias=True,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_init=kernel_init,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_state: Array) -> Array:
    """
    Args:
        hidden_state: Input tensor of shape (..., hidden_size) - supports packed sequences

    Returns:
        Output tensor of shape (..., hidden_size)
    """
    hidden_state = self.linear_fc1(hidden_state)
    hidden_state = jax.nn.gelu(hidden_state)
    hidden_state = self.linear_fc2(hidden_state)
    return hidden_state


class Qwen3OmniMoeVisionPatchEmbed(nnx.Module):
  """3D convolution-based patch embedding for vision inputs.

  Attributes:
      config: Config containing model parameters
      patch_size: Spatial patch size
      temporal_patch_size: Temporal patch size
      in_channels: Number of input channels
      embed_dim: Embedding dimension
      dtype: Data type for computation
      weight_dtype: Data type for weights
      rngs: RNG state for initialization
      proj: Convolution projection layer
  """

  def __init__(
      self,
      config: Config,
      # Default to float32 for numerical stability in 3D convolutions on image/video inputs
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3Omni vision patch embedding.

    Args:
        config: Config containing model parameters
        dtype: Data type for computation (defaults to float32 for numerical stability)
        weight_dtype: Data type for weights (defaults to float32 for numerical stability)
        rngs: RNG state for initialization
    """
    self.config = config
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rngs = rngs

    self.patch_size = config.patch_size_for_vit
    self.temporal_patch_size = config.temporal_patch_size_for_vit
    self.in_channels = config.num_channels_for_vit
    self.embed_dim = config.hidden_size_for_vit

    kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)

    self.proj = nnx.Conv(
        in_features=self.in_channels,
        out_features=self.embed_dim,
        kernel_size=kernel_size,
        strides=kernel_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array, video_mask: Array | None = None) -> tuple[Array, Array | None]:
    """
    Args:
        hidden_states: Input tensor of shape (batch, in_channels, temporal*patch_size, height*patch_size, width*patch_size)
        video_mask: Optional pixel-level mask with shape
          (batch, 1, temporal*patch_size, height*patch_size, width*patch_size).
    Returns:
        Tuple of:
        - Output tensor of shape (batch, T*H*W, embed_dim) where T, H, W are the number of patches
        - Attention mask of shape (batch, T*H*W), or None when video_mask is not provided
    """
    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    hidden_states = self.proj(hidden_states)
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1] * hidden_states.shape[2] * hidden_states.shape[3]
    hidden_states = hidden_states.reshape(batch_size, seq_len, self.embed_dim)

    attention_mask = None
    if video_mask is not None:
      patch_mask = video_mask[
          :,
          0,
          :: self.temporal_patch_size,
          :: self.patch_size,
          :: self.patch_size,
      ]
      attention_mask = patch_mask.reshape(video_mask.shape[0], -1).astype(
          jnp.int32
      )

    return hidden_states, attention_mask


class Qwen3OmniMoeVisionAttention(nnx.Module):
  """Vision attention layer wrapper.

  Attributes:
      config: Config containing model parameters
      attn: Underlying attention module
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision attention layer.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    head_dim = self.config.hidden_size_for_vit // self.config.num_attention_heads_for_vit
    # Vision uses full SA, no kv cache
    self.attn = Attention(
        config=self.config,
        num_query_heads=self.config.num_attention_heads_for_vit,
        num_kv_heads=self.config.num_attention_heads_for_vit,
        head_dim=head_dim,
        max_target_length=self.config.num_position_embeddings_for_vit,
        attention_kernel=self.config.attention_for_vit,
        inputs_q_shape=(1, 1, self.config.hidden_size_for_vit),
        inputs_kv_shape=(1, 1, self.config.hidden_size_for_vit),
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=mesh,
        dropout_rate=0.0,
        attention_type=AttentionType.FULL,
        is_nope_layer=False,
        use_bias_in_projections=True,
        is_vision=True,
        use_qk_norm=False,
        query_pre_attn_scalar=head_dim ** (-0.5),
        model_mode="train",
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      num_frames: int,
      height: int,
      width: int,
      deterministic: bool = True,
  ) -> Array:
    """
    Args:
        hidden_states: Input tensor of shape (batch, T*H*W, hidden_size)
        num_frames: Number of temporal frames (static)
        height: Height in patches (static)
        width: Width in patches (static)
        deterministic: Whether to use deterministic mode (disable dropout)

    Returns:
        Output tensor of shape (batch, T*H*W, hidden_size)
    """
    # Pass through attention with static dimensions via rope_kwargs
    rope_kwargs = {
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }
    output, _ = self.attn(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
        rope_kwargs=rope_kwargs,
    )

    return output


class Qwen3OmniMoeVisionBlock(nnx.Module):
  """Vision transformer block with attention and MLP.

  Attributes:
      config: Config containing model parameters
      ln1: LayerNorm before attention
      ln2: LayerNorm before MLP
      attn: Attention module
      mlp: First MLP layer
      mlp_out: Second MLP layer
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision transformer block.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    hs = self.config.hidden_size_for_vit
    self.ln1 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.ln2 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.attn = Qwen3OmniMoeVisionAttention(config=config, mesh=mesh, rngs=rngs)
    self.mlp = DenseGeneral(
        in_features_shape=hs,
        out_features_shape=self.config.intermediate_size_for_vit,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )
    self.mlp_out = DenseGeneral(
        in_features_shape=self.config.intermediate_size_for_vit,
        out_features_shape=hs,
        use_bias=True,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      x: Array,
      num_frames: int,
      height: int,
      width: int,
  ) -> Array:
    """
    Args:
        x: Input tensor of shape (batch, T*H*W, hidden_size)
        num_frames: Number of temporal frames (static)
        height: Height in patches (static)i
        width: Width in patches (static)

    Returns:
        Output tensor of shape (batch, T*H*W, hidden_size)
    """
    x = x + self.attn(
        self.ln1(x), num_frames=num_frames, height=height, width=width
    )
    y = self.ln2(x)
    y = self.mlp(y)
    y = jax.nn.gelu(y)
    y = self.mlp_out(y)
    return x + y


class Qwen3OmniMoeVisionEncoder(nnx.Module):
  """Vision encoder with patch embedding, positional embedding, and transformer blocks.

  Attributes:
      config: Config containing model parameters
      patch_embed: Patch embedding module
      pos_embed_interpolate: Position embedding interpolation module
      blocks: List of transformer blocks
      merger_list: List of patch mergers for deep supervision
      spatial_merge_size: Size of spatial merging
      deep_idx: Indices of layers to extract deep features from
  """

  def __init__(self, config: Config, *, mesh=None, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision encoder.

    Args:
        config: Config containing model parameters
        mesh: JAX device mesh for sharding
        rngs: RNG state for initialization
    """
    self.config = config
    self.patch_embed = Qwen3OmniMoeVisionPatchEmbed(config=config, rngs=rngs)

    num_pos = config.num_position_embeddings_for_vit
    hs = config.hidden_size_for_vit
    self.spatial_merge_size = config.spatial_merge_size_for_vit

    self.pos_embed_interpolate = Qwen3OmniMoeVisionPosEmbedInterpolate(
        num_position_embeddings=num_pos,
        hidden_size=hs,
        spatial_merge_size=self.spatial_merge_size,
        rngs=rngs,
    )

    self.depth = config.num_hidden_layers_for_vit

    # Use setattr with string names instead of nnx.List to avoid Orbax integer key bug
    for i in range(self.depth):
      block_name = f"blocks_{i}"
      block = Qwen3OmniMoeVisionBlock(config=config, mesh=mesh, rngs=rngs)
      setattr(self, block_name, block)

    self.deep_idx = tuple(config.deepstack_visual_indexes_for_vit)
    # Use setattr with string names instead of nnx.List to avoid Orbax integer key bug
    for i, _ in enumerate(self.deep_idx):
      merger_name = f"merger_{i}"
      merger = Qwen3OmniMoeVisionPatchMerger(config=config, use_postshuffle_norm=True, rngs=rngs)
      setattr(self, merger_name, merger)

  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = True,
  ):
    """
    Args:
        hidden_states: Input visual tokens of shape (batch, in_channels, T*patch_size, H*patch_size, W*patch_size)
        deterministic: Whether to use deterministic mode

    Returns:
        Tuple of:
        - encoder_output: shape (batch, T*H*W, hidden_size_for_vit)
        - deep_features: List of intermediate features, each of shape (batch, T*H*W, out_hidden_size)
    """
    batch_size, _, num_frames, height, width = hidden_states.shape
    num_frames = num_frames // self.config.temporal_patch_size_for_vit
    height = height // self.config.patch_size_for_vit
    width = width // self.config.patch_size_for_vit
    hidden_states = hidden_states.reshape(
        -1,
        self.config.num_channels_for_vit,
        self.config.temporal_patch_size_for_vit,
        self.config.patch_size_for_vit,
        self.config.patch_size_for_vit,
    )

    x, _ = self.patch_embed(hidden_states)
    x = x.reshape(batch_size, -1, self.config.hidden_size_for_vit)
    pos = self.pos_embed_interpolate(num_frames, height, width)

    pos = pos[jnp.newaxis, :, :]
    x = x + pos

    h_traj = []
    for i in range(self.depth):
      block_name = f"blocks_{i}"
      blk = getattr(self, block_name)
      x = blk(x, num_frames=num_frames, height=height, width=width)
      h_traj.append(x)

    deep_feats = []
    for i, idx in enumerate(self.deep_idx):
      h = h_traj[idx]
      merger_name = f"merger_{i}"
      merger = getattr(self, merger_name)
      deep_feat = merger(h)
      deep_feats.append(deep_feat)

    return x, deep_feats


class Qwen3OmniMoeVisionProjector(nnx.Module):
  """Projection layer that converts vision encoder output to model embedding space.

  Attributes:
      config: Config containing model parameters
      merger: Patch merger for spatial reduction
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    """Initializes the Qwen3Omni vision projector.

    Args:
        config: Config containing model parameters
        rngs: RNG state for initialization
    """
    self.config = config
    self.merger = Qwen3OmniMoeVisionPatchMerger(config=config, use_postshuffle_norm=False, rngs=rngs)

  def __call__(self, hidden_states: Array) -> Array:
    """
    Args:
        hidden_states: Encoder output of shape (batch, T*H*W, hidden_size_for_vit)

    Returns:
        Projected output of shape (batch, T*H*W//merge_size**2, out_hidden_size_for_vit)
    """
    output = self.merger(hidden_states)
    return output


def qwen3omni_visionencoder_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  """Convert Qwen3OmniMoeVisionEncoder to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniMoeVisionEncoder,
      config=config,
      mesh=mesh,
      name="Qwen3OmniMoeVisionEncoder_0",
      abstract_init=False,
      metadata_fn=max_initializers.variable_to_logically_partitioned,
  )


def qwen3omni_visionprojector_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  """Convert Qwen3OmniMoeVisionProjector to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniMoeVisionProjector,
      config=config,
      name="Qwen3OmniMoeVisionProjector_0",
      abstract_init=False,
      metadata_fn=max_initializers.variable_to_logically_partitioned,
  )


class Qwen3OmniAudioEncoderLayer(nnx.Module):
  """Transformer encoder layer for audio model."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.hidden_states_shape = (
        self.config.per_device_batch_size,
        self.config.max_source_positions_for_audio,
        self.config.d_model_for_audio,
    )

    self.input_layer_norm = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    self.self_attention_audio = Attention(
        config=self.config,
        num_query_heads=self.config.encoder_attention_heads_for_audio,
        num_kv_heads=self.config.encoder_attention_heads_for_audio,
        head_dim=self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio,
        max_target_length=self.config.max_source_positions_for_audio,
        attention_kernel="dot_product",
        inputs_q_shape=self.hidden_states_shape,
        inputs_kv_shape=self.hidden_states_shape,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        mesh=self.mesh,
        dropout_rate=self.config.attention_dropout_for_audio,
        name="self_attention_audio",
        attention_type=AttentionType.FULL,
        is_nope_layer=True,  # No rotary position embeddings for audio
        use_bias_in_projections=True,
        use_qk_norm=False,
        query_pre_attn_scalar=1
        / math.sqrt(self.config.d_model_for_audio // self.config.encoder_attention_heads_for_audio),
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.rngs,
    )

    self.post_attention_layer_norm = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    self.AudioMLP = MlpBlock(
        config=self.config,
        mesh=self.mesh,
        in_features=self.config.d_model_for_audio,
        intermediate_dim=self.config.encoder_ffn_dim_for_audio,
        activations=("gelu",),  # Single GELU activation
        kernel_init=max_initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        intermediate_dropout_rate=0.0,  # No dropout to match AudioMLP
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        use_bias=True,  # AudioMLP uses bias
        use_pre_norm=False,  # Norm is handled outside
        quant=None,  # No quantization
        model_mode=None,  # Not needed for encoder
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      deterministic: bool = False,
  ):
    """Apply transformer encoder layer to audio hidden states.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, d_model_for_audio)
        deterministic: Whether to use deterministic mode (disable dropout)

    Returns:
        Output tensor of shape (batch, seq_len, d_model_for_audio)
    """
    residual = hidden_states
    hidden_states = self.input_layer_norm(hidden_states)
    hidden_states, _ = self.self_attention_audio(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        deterministic=deterministic,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layer_norm(hidden_states)
    hidden_states = self.AudioMLP(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class Qwen3OmniAudioEncoder(nnx.Module):
  """Full audio encoder with convs, positional embeddings, and transformer layers.

  Attributes:
      config: Config containing model parameters
      mesh: Mesh, JAX device mesh (used for sharding)
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs = None):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.positional_embedding = PositionalEmbedding(
        embedding_dims=self.config.d_model_for_audio,
        max_wavelength=self.config.max_timescale_for_audio,
        cast_as_fprop_dtype=True,
        fprop_dtype=self.config.dtype_mm,
    )

    self.layernorm_post = nnx.LayerNorm(
        num_features=self.config.d_model_for_audio,
        epsilon=1e-5,
        dtype=self.config.dtype_mm,
        rngs=self.rngs,
    )

    # Convolutional downsampling layers
    self.conv2d1 = nnx.Conv(
        in_features=1,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.conv2d2 = nnx.Conv(
        in_features=self.config.downsample_hidden_size_for_audio,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.conv2d3 = nnx.Conv(
        in_features=self.config.downsample_hidden_size_for_audio,
        out_features=self.config.downsample_hidden_size_for_audio,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=True,
        dtype=self.config.dtype_mm,
        param_dtype=self.config.weight_dtype,
        precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    conv_out_dim = self.config.downsample_hidden_size_for_audio * (
        (((self.config.num_mel_bins_for_audio + 1) // 2 + 1) // 2 + 1) // 2
    )
    self.conv_out = DenseGeneral(
        in_features_shape=conv_out_dim,
        out_features_shape=self.config.d_model_for_audio,
        use_bias=False,
        dtype=self.config.dtype_mm,
        weight_dtype=self.config.weight_dtype,
        kernel_init=nd_dense_init(self.config.dense_init_scale, "fan_in", "normal"),
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    # Transformer encoder layers
    for lyr in range(self.config.encoder_layers_for_audio):
      layer_name = f"layers_{lyr}"
      layer = Qwen3OmniAudioEncoderLayer(
          config=self.config,
          mesh=self.mesh,
          rngs=self.rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      audio_features: Array,
      deterministic: bool = False,
  ):
    """Process audio features through convs + transformer encoder.

    Args:
        audio_features: Input of shape (batch, num_mel_bins, audio_length)
        deterministic: Whether to use deterministic mode

    Returns:
        Encoded features of shape (batch, seq_len, d_model_for_audio)
    """
    batch_size, num_mel_bins, audio_length = audio_features.shape
    chunk_size = self.config.n_window_for_audio * 2

    # Reshape to chunks
    num_chunks = audio_length // chunk_size
    audio_chunks = audio_features.reshape(batch_size, num_mel_bins, num_chunks, chunk_size)
    audio_chunks = audio_chunks.transpose(0, 2, 1, 3)
    audio_chunks = audio_chunks.reshape(batch_size * num_chunks, num_mel_bins, chunk_size)

    # Add channel dimension
    hidden_states = audio_chunks[:, :, :, jnp.newaxis]

    # Apply convolutional layers
    hidden_states = self.conv2d1(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.conv2d2(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.conv2d3(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)

    # Reshape conv output
    bc, f, t, c = hidden_states.shape
    hidden_states = hidden_states.transpose(0, 2, 3, 1)
    hidden_states = hidden_states.reshape(bc, t, c * f)
    hidden_states = self.conv_out(hidden_states)

    # Add positional embeddings
    seq_len_per_chunk = hidden_states.shape[1]
    pos_emb = self.positional_embedding(seq_len_per_chunk)
    pos_emb = jnp.broadcast_to(
        pos_emb[None, :, :], (batch_size * num_chunks, seq_len_per_chunk, self.config.d_model_for_audio)
    )
    hidden_states = hidden_states + pos_emb

    # Apply transformer encoder layers
    for lyr in range(self.config.encoder_layers_for_audio):
      layer_name = f"layers_{lyr}"
      layer = getattr(self, layer_name)
      hidden_states = layer(
          hidden_states,
          deterministic=deterministic,
      )

    hidden_states = self.layernorm_post(hidden_states)

    # Reshape back: (batch*chunks, seq_len_per_chunk, d_model) -> (batch, chunks*seq_len_per_chunk, d_model)
    hidden_states = hidden_states.reshape(batch_size, num_chunks * seq_len_per_chunk, self.config.d_model_for_audio)

    return hidden_states


class Qwen3OmniAudioProjector(nnx.Module):
  """Projection layer that converts audio encoder output to model embedding space."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs = None):
    self.config = config
    self.proj1 = DenseGeneral(
        in_features_shape=config.d_model_for_audio,
        out_features_shape=config.d_model_for_audio,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.proj2 = DenseGeneral(
        in_features_shape=config.d_model_for_audio,
        out_features_shape=config.output_dim_for_audio,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(config.dense_init_scale, "fan_in", "normal"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    """
    Args:
        hidden_states: Encoder output of shape (num_chunks, seq_len, d_model_for_audio)

    Returns:
        Projected output of shape (num_chunks, seq_len, output_dim_for_audio)
    """
    hidden_states = self.proj1(hidden_states)
    hidden_states = jax.nn.gelu(hidden_states)
    hidden_states = self.proj2(hidden_states)
    return hidden_states


def qwen3omni_audioencoder_as_linen(config: Config, mesh: Mesh):
  """Convert AudioEncoder (convs + transformer layers, no projector) to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniAudioEncoder,
      config=config,
      mesh=mesh,
      name="Qwen3OmniAudioEncoder_0",
      abstract_init=False,
      metadata_fn=variable_to_logically_partitioned,
  )


def qwen3omni_audioprojector_as_linen(config: Config, mesh: Mesh):
  """Convert AudioProjector to Linen module."""
  return nnx_wrappers.to_linen(
      Qwen3OmniAudioProjector,
      config=config,
      name="Qwen3OmniAudioProjector_0",
      abstract_init=False,
      metadata_fn=variable_to_logically_partitioned,
  )


# Vision encoder Linen wrappers
Qwen3OmniMoeVisionPatchMergerToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionPatchMerger,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionMLPToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionMLP,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionPatchEmbedToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionPatchEmbed,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionAttentionToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionAttention,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionEncoderToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionEncoder,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniMoeVisionProjectorToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniMoeVisionProjector,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3MoeDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3MoeDecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3NextDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3NextDecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3NextScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3NextScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

# Audio encoder Linen wrappers
Qwen3OmniAudioEncoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniAudioEncoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniAudioEncoderToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniAudioEncoder,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3OmniAudioProjectorToLinen = nnx_wrappers.to_linen_class(
    Qwen3OmniAudioProjector,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
