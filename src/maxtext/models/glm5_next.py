# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.3-Flash model components based on the upstream GLM-5-Next architecture."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import (
    Array,
    Config,
    HyperConnectionType,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.inference import kvcache
from maxtext.layers import initializers, linears, mhc, moe, nnx_wrappers
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils


def l2norm(t: Array, eps: float = 1e-6) -> Array:
  return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps)


def scan_kimi_delta_attention(q, k, v, g, beta, mask=None, initial_state=None):
  """Scan Kimi Delta Attention across sequence dimension."""
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  v = v.astype(jnp.float32)
  g = g.astype(jnp.float32)
  beta = beta.astype(jnp.float32)

  q_norm = l2norm(q, eps=1e-6) * (q.shape[-1] ** -0.5)
  k_norm = l2norm(k, eps=1e-6)
  g_exp = jnp.exp(g)

  if mask is not None:
    mask_f = (mask != 0).astype(jnp.float32)
    k_norm = k_norm * mask_f[..., None, None]
    q_norm = q_norm * mask_f[..., None, None]
    beta = beta * mask_f[..., None]
    g_exp = jnp.where(mask_f[..., None, None] != 0, g_exp, 1.0)

  b, _, h, d = q.shape
  if initial_state is None:
    init_state = jnp.zeros((b, h, d, d), dtype=jnp.float32)
  else:
    init_state = initial_state.astype(jnp.float32)

  def step(state, inputs):
    q_i, k_i, v_i, g_i, b_i = inputs
    state = state * g_i[..., :, None]
    kv_mem = jnp.einsum("bhkd,bhk->bhd", state, k_i)
    delta = (v_i - kv_mem) * b_i[..., None]
    new_state = state + jnp.einsum("bhk,bhd->bhkd", k_i, delta)
    out = jnp.einsum("bhkd,bhk->bhd", new_state, q_i)
    return new_state, out

  xs = (
      jnp.swapaxes(q_norm, 0, 1),
      jnp.swapaxes(k_norm, 0, 1),
      jnp.swapaxes(v, 0, 1),
      jnp.swapaxes(g_exp, 0, 1),
      jnp.swapaxes(beta, 0, 1),
  )
  final_state, outs = jax.lax.scan(step, init_state, xs)
  outs = jnp.swapaxes(outs, 0, 1)
  return outs, final_state


class Glm5NextAttention(nnx.Module):
  """GLM-5.3-Flash KDA (Knowledge-Driven Attention / Gated Delta Attention) Layer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    self.emb_dim = config.emb_dim
    self.num_heads = config.linear_num_heads
    self.head_dim = config.linear_head_dim
    self.kda_conv_size = config.linear_conv_kernel_dim
    self.conv_dim = self.num_heads * self.head_dim
    self.linear_lower_bound = getattr(config, "linear_lower_bound", -5.0)

    # Projections
    self.q_proj = linears.DenseGeneral(
        self.emb_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "q_heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.k_proj = linears.DenseGeneral(
        self.emb_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "kv_heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.v_proj = linears.DenseGeneral(
        self.emb_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "kv_heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.b_proj = linears.DenseGeneral(
        self.emb_dim,
        self.num_heads,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.f_a_proj = linears.DenseGeneral(
        self.emb_dim,
        self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "head_dim"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.f_b_proj = linears.DenseGeneral(
        self.head_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("head_dim", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.g_a_proj = linears.DenseGeneral(
        self.emb_dim,
        self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "head_dim"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.g_b_proj = linears.DenseGeneral(
        self.head_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("head_dim", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # 1D Depthwise Convolution over concatenated (Q, K, V)
    self.conv1d = nnx.Conv(
        in_features=3 * self.conv_dim,
        out_features=3 * self.conv_dim,
        kernel_size=(self.kda_conv_size,),
        feature_group_count=3 * self.conv_dim,
        use_bias=False,
        padding="VALID",
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        rngs=self.rngs,
    )

    # Forget gate learnable parameters
    self.A_log = nnx.Param(
        jnp.zeros((self.num_heads,), dtype=self.weight_dtype),
        out_sharding=("heads",),
    )
    self.dt_bias = nnx.Param(
        jnp.zeros((self.conv_dim,), dtype=self.weight_dtype),
        out_sharding=("heads",),
    )

    # Output normalization and projection
    self.o_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("head_dim",),
        rngs=self.rngs,
    )
    self.o_proj = linears.DenseGeneral(
        self.conv_dim,
        self.emb_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("heads", "embed"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    if self.model_mode != MODEL_MODE_TRAIN:
      batch_size, _ = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      self.cache = kvcache.KVCache(
          max_prefill_length=config.max_prefill_predict_length,
          max_target_length=config.max_target_length,
          batch=batch_size,
          key_seq_len=1,
          value_seq_len=1,
          key_heads=self.num_heads,
          value_heads=self.num_heads,
          key_head_size=self.head_dim,
          value_head_size=self.head_dim,
          dtype=self.dtype,
          is_gdn=True,
          conv_kernel_size=self.kda_conv_size,
          conv_dim=3 * self.conv_dim,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )
    else:
      self.cache = None

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array | None = None,
      decoder_segment_ids: Array | None = None,
      decoder_positions: Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> tuple[Array, None]:
    x = inputs_q
    b, s, _ = x.shape

    if decoder_segment_ids is not None:
      mask = decoder_segment_ids != 0
      x_masked = jnp.where(mask[..., None], x, 0.0)
    else:
      x_masked = x

    # Projections
    q = self.q_proj(x_masked)
    k = self.k_proj(x_masked)
    v = self.v_proj(x_masked)

    # 1D Conv over concatenated [q, k, v]
    qkv = jnp.concatenate([q, k, v], axis=-1)
    conv_kernel_size = self.kda_conv_size

    conv_state = None
    recurrent_state = None
    next_conv_state = None

    if self.cache is not None and model_mode != MODEL_MODE_TRAIN:
      recurrent_state, conv_state = self.cache.get_gdn_states()
      conv_input = jnp.concatenate([conv_state, qkv], axis=1)

      if decoder_segment_ids is not None:
        valid_lens = jnp.sum(decoder_segment_ids != 0, axis=1)

        def extract_state(c_in, v_len):
          return jax.lax.dynamic_slice_in_dim(c_in, v_len, conv_kernel_size - 1, axis=0)

        next_conv_state = jax.vmap(extract_state)(conv_input, valid_lens)
      else:
        next_conv_state = conv_input[:, -(conv_kernel_size - 1) :, :]
    else:
      conv_input = jnp.pad(qkv, ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))

    conv_out = self.conv1d(conv_input)
    conv_out = conv_out[:, -s:, :]
    qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(self.dtype)

    q_conv, k_conv, v_conv = jnp.split(qkv_conv, 3, axis=-1)
    q = jnp.reshape(q_conv, (b, s, self.num_heads, self.head_dim))
    k = jnp.reshape(k_conv, (b, s, self.num_heads, self.head_dim))
    v = jnp.reshape(v_conv, (b, s, self.num_heads, self.head_dim))

    # Gates
    beta = jax.nn.sigmoid(self.b_proj(x_masked))
    f_a = self.f_a_proj(x_masked)
    f_b = self.f_b_proj(f_a)
    f_b = jnp.reshape(f_b, (b, s, self.num_heads, self.head_dim))
    dt_bias = jnp.reshape(self.dt_bias[...], (1, 1, self.num_heads, self.head_dim))
    decay_rate = jnp.exp(self.A_log[None, None, :, None])
    if self.linear_lower_bound is not None:
      g = self.linear_lower_bound * jax.nn.sigmoid(decay_rate * (f_b + dt_bias))
    else:
      g_softplus = jnp.where(f_b + dt_bias > 20.0, f_b + dt_bias, jax.nn.softplus(f_b + dt_bias))
      g = -decay_rate * g_softplus

    if s == 1 and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      q_step = q[:, 0].astype(jnp.float32)
      k_step = k[:, 0].astype(jnp.float32)
      v_step = v[:, 0].astype(jnp.float32)
      g_step = g[:, 0].astype(jnp.float32)
      beta_step = beta[:, 0].astype(jnp.float32)

      q_norm = l2norm(q_step, eps=1e-6) * (self.head_dim**-0.5)
      k_norm = l2norm(k_step, eps=1e-6)
      g_exp = jnp.exp(g_step)

      if recurrent_state is None:
        curr_state = jnp.zeros((b, self.num_heads, self.head_dim, self.head_dim), dtype=jnp.float32)
      else:
        curr_state = recurrent_state.astype(jnp.float32)

      curr_state = curr_state * g_exp[..., :, None]
      kv_mem = jnp.einsum("bhkd,bhk->bhd", curr_state, k_norm)
      delta = (v_step - kv_mem) * beta_step[..., None]
      next_recurrent_state = curr_state + jnp.einsum("bhk,bhd->bhkd", k_norm, delta)
      attn_out = jnp.einsum("bhkd,bhk->bhd", next_recurrent_state, q_norm)
      attn_out = jnp.expand_dims(attn_out, axis=1)
    else:
      attn_out, next_recurrent_state = scan_kimi_delta_attention(
          q, k, v, g, beta, mask=decoder_segment_ids, initial_state=recurrent_state
      )

    if self.cache is not None and model_mode != MODEL_MODE_TRAIN:
      self.cache.update_gdn_states(
          next_recurrent_state.astype(self.dtype),
          next_conv_state.astype(self.dtype),
      )

    # Output gating & norm
    g_a = self.g_a_proj(x_masked)
    g_b = self.g_b_proj(g_a)
    g_b = jnp.reshape(g_b, (b, s, self.num_heads, self.head_dim))

    normed = self.o_norm(attn_out.astype(self.dtype))
    gated = normed * jax.nn.sigmoid(g_b.astype(jnp.float32)).astype(self.dtype)
    gated_flat = jnp.reshape(gated, (b, s, self.conv_dim))
    out = self.o_proj(gated_flat)
    return out, None


class Glm5NextSparseAttention(nnx.Module):
  """GLM-5.3-Flash MLA (Multi-Head Latent Attention) / Sparse Attention Layer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    self.emb_dim = config.emb_dim
    self.num_heads = config.num_query_heads
    self.q_lora_rank = getattr(config, "q_lora_rank", 1536)
    self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)
    self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 256)
    self.v_head_dim = getattr(config, "v_head_dim", 256)
    self.head_dim = self.qk_nope_head_dim
    self.scaling = self.head_dim**-0.5

    # Query Projections
    self.q_a_proj = linears.DenseGeneral(
        self.emb_dim,
        self.q_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "q_lora"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.q_a_layernorm = RMSNorm(
        num_features=self.q_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.q_b_proj = linears.DenseGeneral(
        self.q_lora_rank,
        self.num_heads * self.qk_nope_head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("q_lora", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Key/Value Projections
    self.kv_a_proj_with_mqa = linears.DenseGeneral(
        self.emb_dim,
        self.kv_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "kv_lora"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.kv_a_layernorm = RMSNorm(
        num_features=self.kv_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.kv_b_proj = linears.DenseGeneral(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("kv_lora", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Output projection
    self.o_proj = linears.DenseGeneral(
        self.num_heads * self.v_head_dim,
        self.emb_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("heads", "embed"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    if self.model_mode != MODEL_MODE_TRAIN:
      batch_size, _ = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      self.cache = kvcache.KVCache(
          max_prefill_length=config.max_prefill_predict_length,
          max_target_length=config.max_target_length,
          batch=batch_size,
          key_seq_len=1,
          value_seq_len=1,
          key_heads=self.num_heads,
          value_heads=self.num_heads,
          key_head_size=self.qk_nope_head_dim,
          value_head_size=self.v_head_dim,
          dtype=self.dtype,
          is_gdn=False,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )
    else:
      self.cache = None

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array | None = None,
      decoder_segment_ids: Array | None = None,
      decoder_positions: Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> tuple[Array, None]:
    x = inputs_q
    b, s, _ = x.shape

    # Query: [B, S, Q_LORA] -> [B, S, Q_LORA] -> [B, S, H * D] -> [B, S, H, D]
    q_resid = self.q_a_layernorm(self.q_a_proj(x))
    q = self.q_b_proj(q_resid)
    q = jnp.reshape(q, (b, s, self.num_heads, self.qk_nope_head_dim))

    # Key / Value: [B, S, KV_LORA] -> [B, S, KV_LORA] -> [B, S, H * (D_K + D_V)]
    compressed_kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
    kv_b = self.kv_b_proj(compressed_kv)
    kv_b = jnp.reshape(kv_b, (b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim))
    k = kv_b[..., : self.qk_nope_head_dim]
    v = kv_b[..., self.qk_nope_head_dim :]

    if self.cache is not None and model_mode == MODEL_MODE_AUTOREGRESSIVE and s == 1:
      cached_prefill, cached_ar = self.cache(k, v, decoder_segment_ids, model_mode=model_mode)
      k_p, v_p, seg_p = cached_prefill
      k_a, v_a, seg_a, _ = cached_ar
      k_all = jnp.concatenate([k_p, k_a], axis=1)  # [B, max_target_length, H, D]
      v_all = jnp.concatenate([v_p, v_a], axis=1)  # [B, max_target_length, H, D]
      seg_all = jnp.concatenate([seg_p, seg_a], axis=1)  # [B, max_target_length]

      # [B, H, 1, D]
      q = jnp.swapaxes(q, 1, 2)
      # [B, H, max_target_length, D]
      k_all = jnp.swapaxes(k_all, 1, 2)
      v_all = jnp.swapaxes(v_all, 1, 2)

      scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_all) * self.scaling
      mask = seg_all[:, None, None, :] != 0
      scores = jnp.where(mask, scores, -1e9)

      attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(self.dtype)
      attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_all)
      attn_out = jnp.swapaxes(attn_out, 1, 2)
      attn_out = jnp.reshape(attn_out, (b, s, self.num_heads * self.v_head_dim))
    else:
      if self.cache is not None and model_mode == MODEL_MODE_PREFILL:
        self.cache(k, v, decoder_segment_ids, model_mode=model_mode)

      # [B, H, S, D]
      q = jnp.swapaxes(q, 1, 2)
      k = jnp.swapaxes(k, 1, 2)
      v = jnp.swapaxes(v, 1, 2)

      # Scaled dot-product causal attention
      scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scaling

      causal_mask = jnp.tril(jnp.ones((s, s), dtype=jnp.bool_))
      if decoder_segment_ids is not None:
        seg_mask = (decoder_segment_ids[:, None, :] != 0) & (decoder_segment_ids[:, :, None] != 0)
        full_mask = causal_mask[None, :, :] & seg_mask
        mask_value = -1e9
        scores = jnp.where(full_mask[:, None, :, :], scores, mask_value)
      else:
        mask_value = -1e9
        scores = jnp.where(causal_mask[None, None, :, :], scores, mask_value)

      attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(self.dtype)
      attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

      # [B, S, H * D]
      attn_out = jnp.swapaxes(attn_out, 1, 2)
      attn_out = jnp.reshape(attn_out, (b, s, self.num_heads * self.v_head_dim))

    out = self.o_proj(attn_out)
    return out, None


class Glm5NextDenseMLP(nnx.Module):
  """GLM-5.3-Flash Dense MLP with SwiGLU clamping."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      in_features: int,
      intermediate_dim: int,
      rngs: nnx.Rngs,
      quant=None,
      model_mode: str = MODEL_MODE_TRAIN,
  ):
    self.config = config
    self.mesh = mesh
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.swiglu_limit = getattr(config, "swiglu_limit", 10.0)

    self.wi_0 = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.intermediate_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        mesh=self.mesh,
        rngs=rngs,
    )
    self.wi_1 = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.intermediate_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        mesh=self.mesh,
        rngs=rngs,
    )
    self.wo = linears.DenseGeneral(
        in_features_shape=self.intermediate_dim,
        out_features_shape=in_features,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("mlp", "embed"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        mesh=self.mesh,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: Array,
      **kwargs,
  ) -> Array:
    gate = self.wi_0(inputs)
    up = self.wi_1(inputs)
    gate = jnp.minimum(gate, self.swiglu_limit)
    up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
    act = jax.nn.silu(gate) * up
    return self.wo(act)


class Glm5NextDecoderLayer(nnx.Module):
  """GLM-5.3-Flash Decoder Layer wrapping Attention and MLP/MoE in mHC blocks."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      rngs: nnx.Rngs,
      quant=None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    self.rngs = rngs

    self.input_layernorm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    is_sparse_attn = (layer_idx + 1) % config.inhomogeneous_layer_cycle_interval == 0
    if is_sparse_attn:
      self.attention = Glm5NextSparseAttention(
          config=config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )
    else:
      self.attention = Glm5NextAttention(
          config=config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )

    self.is_moe = layer_idx >= getattr(config, "first_num_dense_layers", 3)
    if self.is_moe:
      self.mlp = moe.RoutedAndSharedMoE(
          config=self.config,
          mesh=self.mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed_moe", None),
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = Glm5NextDenseMLP(
          config=config,
          mesh=self.mesh,
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )

    self.attn_hc = mhc.ManifoldConstrainedHyperConnections(
        config=config,
        dim=config.emb_dim,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.ffn_hc = mhc.ManifoldConstrainedHyperConnections(
        config=config,
        dim=config.emb_dim,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Array | None = None,
      decoder_positions: Array | None = None,
      deterministic: bool = True,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> tuple[Array, None]:
    """Forward pass for GLM-5.3-Flash Decoder Layer."""
    x = inputs

    x, _ = self.attn_hc(
        norm_fn=self.input_layernorm,
        branch_fn=self.attention,
        x=x,
        mhc_type=HyperConnectionType.ATTENTION,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        model_mode=model_mode,
        deterministic=deterministic,
    )

    ffn_type = HyperConnectionType.MLP_MOE if self.is_moe else HyperConnectionType.MLP_DENSE
    x, _ = self.ffn_hc(
        norm_fn=self.post_attention_layernorm,
        branch_fn=self.mlp,
        x=x,
        mhc_type=ffn_type,
    )

    return x, None


Glm5NextDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Glm5NextDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
