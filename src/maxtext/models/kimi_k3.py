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

"""Kimi-K3 model decoder layer and attention architecture implementation."""

from typing import Any, Optional
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.layers import attentions, attention_mla
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.linears import DenseGeneral as BaseDenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.inference.kvcache import KVCache
from maxtext.utils import max_utils


def round_bfloat16_logits(logits: jax.Array) -> jax.Array:
  """Preserve HF BF16 output rounding when XLA keeps extra accumulator bits."""
  return jax.lax.reduce_precision(logits.astype(jnp.float32), exponent_bits=8, mantissa_bits=7)


def situ_activation(x: jax.Array, beta: float = 4.0) -> jax.Array:
  """Situ activation function: beta * tanh(x / beta) * sigmoid(x)."""
  x_f = x.astype(jnp.float32)
  out = beta * jnp.tanh(x_f / beta) * jax.nn.sigmoid(x_f)
  return out.astype(x.dtype)


def linear_situ_activation(x: jax.Array, linear_beta: float = 25.0) -> jax.Array:
  """Linear situ transformation: linear_beta * tanh(x / linear_beta)."""
  x_f = x.astype(jnp.float32)
  out = linear_beta * jnp.tanh(x_f / linear_beta)
  return out.astype(x.dtype)


def jax_kda_recurrent_step(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Single-step Autoregressive KDA (Gated Delta Rule) calculation.

  Args:
    query: Query tensor [B, H, K_dim]
    key: Key tensor [B, H, K_dim]
    value: Value tensor [B, H, V_dim]
    g: Effective gate decay in log space [B, H, K_dim]
    beta: Sigmoid beta scaling [B, H]
    initial_state: Recurrent state [B, H, K_dim, V_dim]

  Returns:
    (core_attn_out, next_recurrent_state)
  """
  K = query.shape[-1]
  scale = K**-0.5
  q = (query * scale).astype(jnp.float32)
  k = key.astype(jnp.float32)
  v = value.astype(jnp.float32)
  g_exp = jnp.exp(g.astype(jnp.float32))  # [B, H, K_dim]
  b = beta.astype(jnp.float32)[..., None, None]  # [B, H, 1, 1]

  S = initial_state.astype(jnp.float32) * g_exp[..., None]
  k_exp = k[..., None]  # [B, H, K_dim, 1]
  v_prime = jnp.sum(k_exp * S, axis=-2)  # [B, H, V_dim]
  diff = v - v_prime
  delta = b * (k_exp * diff[..., None, :])
  S_new = S + delta

  q_exp = q[..., None]  # [B, H, K_dim, 1]
  core_attn_out = jnp.sum(q_exp * S_new, axis=-2).astype(query.dtype)
  return core_attn_out, S_new


def jax_kda_chunk_rule(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
  """Multi-token KDA recurrence across sequence dimension.

  Args:
    query: [B, T, H, K_dim]
    key: [B, T, H, K_dim]
    value: [B, T, H, V_dim]
    g: [B, T, H, K_dim]
    beta: [B, T, H]
    initial_state: [B, H, K_dim, V_dim] or None

  Returns:
    (output, final_state)
  """
  B, _, H, K = query.shape
  V = value.shape[-1]
  scale = K**-0.5
  q = (query * scale).astype(jnp.float32)
  k = key.astype(jnp.float32)
  v = value.astype(jnp.float32)
  g_f = g.astype(jnp.float32)
  b_f = beta.astype(jnp.float32)

  if initial_state is None:
    initial_state = jnp.zeros((B, H, K, V), dtype=jnp.float32)

  def scan_fn(S, xs):
    q_i, k_i, v_i, g_i, b_i = xs
    g_exp = jnp.exp(g_i)[..., None]
    S = S * g_exp
    k_exp = k_i[..., None]
    v_prime = jnp.sum(k_exp * S, axis=-2)
    diff = v_i - v_prime
    delta = b_i[..., None, None] * (k_exp * diff[..., None, :])
    S = S + delta
    q_exp = q_i[..., None]
    o_i = jnp.sum(q_exp * S, axis=-2)
    return S, o_i

  xs = (
      jnp.swapaxes(q, 0, 1),
      jnp.swapaxes(k, 0, 1),
      jnp.swapaxes(v, 0, 1),
      jnp.swapaxes(g_f, 0, 1),
      jnp.swapaxes(b_f, 0, 1),
  )
  final_state, o_seq = jax.lax.scan(scan_fn, initial_state, xs)
  output = jnp.swapaxes(o_seq, 0, 1).astype(query.dtype)
  return output, final_state


class DenseGeneral(BaseDenseGeneral):
  """Kimi projections round after FP32 accumulation across all shards."""

  def __init__(self, *args, **kwargs):
    kwargs["accumulate_in_float32"] = True
    super().__init__(*args, **kwargs)


class KimiGateProjection(DenseGeneral):
  """Low-rank KDA gate projection with the same accumulation boundary."""


class KimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention (KDA) implementing linear attention with 1D convolution and delta rule."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    cfg = self.config

    self.hidden_size = cfg.emb_dim
    self.head_dim = getattr(cfg, "kda_head_dim", 128)
    self.num_heads = getattr(cfg, "kda_num_heads", 96)
    self.projection_size = self.num_heads * self.head_dim
    self.conv_size = getattr(cfg, "gdn_conv_kernel_dim", 4)
    self.gate_lower_bound = getattr(cfg, "kda_gate_lower_bound", -5.0)
    if model_mode != MODEL_MODE_TRAIN:
      batch_size, _ = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
      self.cache = KVCache(
          max_prefill_length=cfg.max_prefill_predict_length,
          max_target_length=cfg.max_target_length,
          batch=batch_size,
          key_seq_len=1,
          value_seq_len=1,
          key_heads=self.num_heads,
          value_heads=self.num_heads,
          key_head_size=self.head_dim,
          value_head_size=self.head_dim,
          dtype=jnp.float32,
          is_gdn=True,
          conv_kernel_size=self.conv_size,
          conv_dim=3 * self.projection_size,
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      self.cache = None

    # Q, K, V linear projections
    self.q_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed_attn", "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.k_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed_attn", "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.v_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed_attn", "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # 1D Short Convolutions for Q, K, V
    def conv_kernel_init(key, shape, dtype=jnp.float32):
      return jax.nn.initializers.lecun_normal()(key, shape, dtype)

    self.q_conv1d = nnx.Conv(
        in_features=self.projection_size,
        out_features=self.projection_size,
        kernel_size=(self.conv_size,),
        feature_group_count=self.projection_size,
        padding="CAUSAL",
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=cfg.weight_dtype,
        kernel_init=conv_kernel_init,
        precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.k_conv1d = nnx.Conv(
        in_features=self.projection_size,
        out_features=self.projection_size,
        kernel_size=(self.conv_size,),
        feature_group_count=self.projection_size,
        padding="CAUSAL",
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=cfg.weight_dtype,
        kernel_init=conv_kernel_init,
        precision=cfg.matmul_precision,
        rngs=rngs,
    )
    self.v_conv1d = nnx.Conv(
        in_features=self.projection_size,
        out_features=self.projection_size,
        kernel_size=(self.conv_size,),
        feature_group_count=self.projection_size,
        padding="CAUSAL",
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=cfg.weight_dtype,
        kernel_init=conv_kernel_init,
        precision=cfg.matmul_precision,
        rngs=rngs,
    )

    # Gate and decay parameters
    self.A_log = nnx.Param(
        jnp.zeros((self.head_dim,), dtype=jnp.float32),
    )
    self.dt_bias = nnx.Param(
        jnp.zeros((self.projection_size,), dtype=jnp.float32),
    )
    self.f_a_proj = KimiGateProjection(
        in_features_shape=self.hidden_size,
        out_features_shape=self.head_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed_attn", None),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.f_b_proj = DenseGeneral(
        in_features_shape=self.head_dim,
        out_features_shape=self.projection_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=(None, "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.b_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.num_heads,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed_attn", "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.g_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.projection_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed_attn", "gdn_head"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # Gated RMSNorm and Out projection
    self.o_norm = RMSNorm(
        num_features=self.head_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.o_proj = DenseGeneral(
        in_features_shape=self.projection_size,
        out_features_shape=self.hidden_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("gdn_head", "embed_attn"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      kv_cache=None,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> tuple[jax.Array, Any]:
    batch, seq_len, _ = hidden_states.shape

    # 1. Linear Projections
    q_raw = self.q_proj(hidden_states)
    k_raw = self.k_proj(hidden_states)
    v_raw = self.v_proj(hidden_states)
    if hidden_states.dtype == jnp.bfloat16:
      q_raw, k_raw, v_raw = [round_bfloat16_logits(x).astype(jnp.bfloat16) for x in (q_raw, k_raw, v_raw)]

    # 2. Causal 1D Convolutions
    conv_state_q = None
    conv_state_k = None
    conv_state_v = None
    recurrent_state = None
    if kv_cache is not None and isinstance(kv_cache, dict):
      conv_state_q = kv_cache.get("conv_state_q")
      conv_state_k = kv_cache.get("conv_state_k")
      conv_state_v = kv_cache.get("conv_state_v")
      recurrent_state = kv_cache.get("recurrent_state")
    elif self.cache is not None and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      recurrent_state, conv_state = self.cache.get_gdn_states()
      conv_state_q, conv_state_k, conv_state_v = jnp.split(conv_state.astype(hidden_states.dtype), 3, axis=-1)

    # Keep convolution histories fixed-size even for short prefills.
    if conv_state_q is None:
      conv_state_q = jnp.zeros((batch, self.conv_size - 1, self.projection_size), dtype=q_raw.dtype)
      conv_state_k = jnp.zeros_like(conv_state_q)
      conv_state_v = jnp.zeros_like(conv_state_q)
    if self.conv_size > 1:
      q_in = jnp.concatenate([conv_state_q, q_raw], axis=1)
      k_in = jnp.concatenate([conv_state_k, k_raw], axis=1)
      v_in = jnp.concatenate([conv_state_v, v_raw], axis=1)
    else:
      q_in, k_in, v_in = q_raw, k_raw, v_raw

    segment_ids = kwargs.get("decoder_segment_ids")
    valid = jnp.ones((batch, seq_len), dtype=jnp.bool_)
    if model_mode == MODEL_MODE_PREFILL and segment_ids is not None:
      valid = segment_ids != 0
    lengths = jnp.sum(valid, axis=1)
    history_indices = lengths[:, None] + jnp.arange(self.conv_size - 1)[None, :]
    new_conv_q = jnp.take_along_axis(q_in, history_indices[..., None], axis=1)
    new_conv_k = jnp.take_along_axis(k_in, history_indices[..., None], axis=1)
    new_conv_v = jnp.take_along_axis(v_in, history_indices[..., None], axis=1)

    q_conv = jax.nn.silu(self.q_conv1d(q_in)[:, -seq_len:, :].astype(jnp.float32)).astype(hidden_states.dtype)
    k_conv = jax.nn.silu(self.k_conv1d(k_in)[:, -seq_len:, :].astype(jnp.float32)).astype(hidden_states.dtype)
    v_conv = jax.nn.silu(self.v_conv1d(v_in)[:, -seq_len:, :].astype(jnp.float32)).astype(hidden_states.dtype)

    # 3. Gate computations
    g_raw = self.f_b_proj(self.f_a_proj(hidden_states))  # [B, T, projection_size]
    g_raw = jnp.reshape(g_raw, (batch, seq_len, self.num_heads, self.head_dim))
    dt_bias = jnp.reshape(self.dt_bias[...], (self.num_heads, self.head_dim))
    A_log = self.A_log[...].reshape(1, self.head_dim)

    g_eff = self.gate_lower_bound * jax.nn.sigmoid(jnp.exp(A_log) * (g_raw.astype(jnp.float32) + dt_bias))
    beta_eff = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))  # [B, T, num_heads]
    g_eff = jnp.where(valid[..., None, None], g_eff, 0.0)
    beta_eff = jnp.where(valid[..., None], beta_eff, 0.0)

    # Reshape Q, K, V to [B, T, H, D]
    q = jnp.reshape(q_conv, (batch, seq_len, self.num_heads, self.head_dim))
    k = jnp.reshape(k_conv, (batch, seq_len, self.num_heads, self.head_dim))
    v = jnp.reshape(v_conv, (batch, seq_len, self.num_heads, self.head_dim))

    # L2 norm on Q and K
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
    k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)

    # 4. KDA Recurrence
    if seq_len == 1 and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      if recurrent_state is None:
        recurrent_state = jnp.zeros((batch, self.num_heads, self.head_dim, self.head_dim), dtype=jnp.float32)
      core_out, next_recurrent_state = jax_kda_recurrent_step(
          q[:, 0, :, :],
          k[:, 0, :, :],
          v[:, 0, :, :],
          g_eff[:, 0, :, :],
          beta_eff[:, 0, :],
          recurrent_state,
      )
      core_out = core_out[:, None, :, :]
    else:
      core_out, next_recurrent_state = jax_kda_chunk_rule(
          q,
          k,
          v,
          g_eff,
          beta_eff,
          initial_state=recurrent_state,
      )

    new_kv_cache = {
        "conv_state_q": new_conv_q,
        "conv_state_k": new_conv_k,
        "conv_state_v": new_conv_v,
        "recurrent_state": next_recurrent_state,
    }
    if self.cache is not None and kv_cache is None and model_mode != MODEL_MODE_TRAIN:
      self.cache.update_gdn_states(
          next_recurrent_state,
          jnp.concatenate([new_conv_q, new_conv_k, new_conv_v], axis=-1).astype(jnp.float32),
      )

    # 5. Gated Output Stage
    # Match the reference's bf16 recurrence output followed by a fused fp32
    # RMSNorm/scale/sigmoid product, rounded only once before o_proj.
    core_out = core_out.astype(hidden_states.dtype).astype(jnp.float32)
    gate_g = jax.nn.sigmoid(
        jnp.reshape(self.g_proj(hidden_states), (batch, seq_len, self.num_heads, self.head_dim)).astype(jnp.float32)
    )
    normed_out = core_out * jax.lax.rsqrt(jnp.mean(core_out**2, axis=-1, keepdims=True) + self.o_norm.epsilon)
    normed_out = (normed_out * self.o_norm.scale[...].astype(jnp.float32) * gate_g).astype(hidden_states.dtype)
    flat_out = jnp.reshape(normed_out, (batch, seq_len, self.projection_size))
    out = self.o_proj(flat_out)
    return out, new_kv_cache


class KimiEagerAttentionOp(nnx.Module):
  """HF eager MLA arithmetic over the combined expanded KV cache."""

  def __init__(self, base_op, scale):
    self.base_op = base_op
    self.scale = scale

  def generate_attention_mask(self, *args, **kwargs):
    return self.base_op.generate_attention_mask(*args, **kwargs)

  @property
  def max_logits(self):
    return self.base_op.max_logits

  def __call__(
      self,
      query,
      key,
      value,
      decoder_segment_ids,
      inputs_positions,
      model_mode,
      cached_values=None,
      previous_chunk=None,
      bidirectional_mask=None,
      **kwargs,
  ):
    if model_mode != MODEL_MODE_TRAIN:
      key, value, decoder_segment_ids = cached_values[0]
      if cached_values[1] is not None:
        ar_key, ar_value, ar_segments, _ = cached_values[1]
        key = jnp.concatenate([key, ar_key], axis=1)
        value = jnp.concatenate([value, ar_value], axis=1)
        decoder_segment_ids = jnp.concatenate([decoder_segment_ids, ar_segments], axis=1)

    dtype = query.dtype

    def round_output(array):
      if dtype == jnp.bfloat16:
        return round_bfloat16_logits(array).astype(dtype)
      return array.astype(dtype)

    # HF rounds the QK product, then its scaling, and normalized probabilities
    # before PV. Separate prefill/AR value sums cannot preserve that rounding.
    query, key, value = map(round_output, (query, key, value))
    scores = round_output(jnp.einsum("bthd,bshd->bhts", query, key, precision=self.base_op.config.matmul_precision))
    scores = round_output(scores * self.scale).astype(jnp.float32)
    mask = self.generate_attention_mask(
        query,
        key,
        decoder_segment_ids,
        model_mode,
        previous_chunk,
        bidirectional_mask,
        segment_positions=inputs_positions,
        decoder_segment_ids_kv=kwargs.get("decoder_segment_ids_kv"),
    )
    if mask is not None:
      scores = scores + mask[:, 0]
    if kwargs.get("record_max_logits", False):
      self.base_op.max_logits = nnx.Intermediate(jnp.max(scores, axis=(-2, -1)))
    probabilities = round_output(jax.nn.softmax(scores, axis=-1))
    return round_output(
        jnp.einsum("bhts,bshd->bthd", probabilities, value, precision=self.base_op.config.matmul_precision)
    )


class KimiMLAAttention(attention_mla.MLA):
  """Kimi MLA: unrotated positional channels and a learned output gate.

  The released KimiLinear reference concatenates the positional channels
  directly; it does not apply its configured rotary embedding in MLA.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    cfg = self.config
    for name in ("query", "wq_a", "wq_b", "wkv_a", "wkv_b", "out"):
      projection = getattr(self, name, None)
      if isinstance(projection, BaseDenseGeneral):
        projection.accumulate_in_float32 = True
    self.faithful_eager_attention = (
        cfg.attention == "dot_product"
        and not self.use_absorbed_mqa
        and self.kv_quant is None
        and self.quant is None
        and not self.use_indexer
        and not cfg.attention_sink
        and not cfg.attn_logits_soft_cap
        and not cfg.moba
        and not cfg.experimental_sa_quant_q_fp8
        and not cfg.experimental_sa_quant_k_fp8
        and cfg.dropout_rate == 0
    )
    if self.faithful_eager_attention:
      # Released HF Kimi eager MLA uses 1/sqrt(q_head_dim), independently of
      # the unused rotary/Yarn configuration.
      self.attention_op = KimiEagerAttentionOp(self.attention_op, self.qk_head_dim**-0.5)
    # The HF MLA low-rank norms use KimiRMSNorm's default epsilon.
    self.q_norm.epsilon = 1e-6
    self.kv_norm.epsilon = 1e-6
    self.g_proj = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=(self.num_query_heads, self.v_head_dim),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed_attn", "q_heads", "kv"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=self.rngs,
    )

  def apply_rotary_embedding(self, inputs, inputs_positions=None, rope_kwargs=None):
    return inputs

  def scale_mla_query(self, query):
    if self.faithful_eager_attention:
      return query
    return super().scale_mla_query(query)

  def __call__(self, inputs_q, inputs_kv, *args, **kwargs):
    gate = jax.nn.sigmoid(self.g_proj(inputs_q))
    return super().__call__(inputs_q, inputs_kv, *args, output_gate=gate, **kwargs)


class KimiMLP(nnx.Module):
  """Kimi Dense MLP block with SituAndMul activation."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      in_features: int,
      intermediate_dim: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    cfg = config
    self.gate_proj = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=intermediate_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.up_proj = DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=intermediate_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.down_proj = DenseGeneral(
        in_features_shape=intermediate_dim,
        out_features_shape=in_features,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("mlp", "embed"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.beta = getattr(cfg, "activation_situ_beta", 4.0)
    self.linear_beta = getattr(cfg, "activation_situ_linear_beta", 25.0)

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    situ_gate = situ_activation(gate.astype(jnp.float32), beta=self.beta)
    linear_up = (
        linear_situ_activation(up.astype(jnp.float32), linear_beta=self.linear_beta)
        if self.linear_beta is not None
        else up.astype(jnp.float32)
    )
    hidden = (situ_gate * linear_up).astype(x.dtype)
    return self.down_proj(hidden)


class KimiSparseMoeBlock(nnx.Module):
  """Kimi MoE block with Latent routed projection, shared experts, and Situ activation."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = config

    self.hidden_dim = cfg.emb_dim
    self.num_experts = cfg.num_experts
    self.num_experts_per_tok = cfg.num_experts_per_tok
    self.routed_expert_hidden_size = getattr(cfg, "routed_expert_hidden_size", 3584)
    self.moe_intermediate_size = cfg.moe_mlp_dim

    # Latent projection down and up
    self.routed_expert_down_proj = DenseGeneral(
        in_features_shape=self.hidden_dim,
        out_features_shape=self.routed_expert_hidden_size,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.routed_expert_norm = RMSNorm(
        num_features=self.routed_expert_hidden_size,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.routed_expert_up_proj = DenseGeneral(
        in_features_shape=self.routed_expert_hidden_size,
        out_features_shape=self.hidden_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("mlp", "embed"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # Core Routed MoE Block operating on latent dimension
    if hasattr(self.config, "_pydantic_config"):
      pydantic_cfg = object.__getattribute__(self.config, "_pydantic_config").model_copy(
          update={"moe_expert_input_dim": self.routed_expert_hidden_size}
      )
      from maxtext.configs.pyconfig import HyperParameters  # pylint: disable=import-outside-toplevel

      moe_config = HyperParameters(pydantic_cfg)
    else:
      moe_config = self.config.model_copy(update={"moe_expert_input_dim": self.routed_expert_hidden_size})
    self.MoeBlock_0 = moe.RoutedMoE(
        config=moe_config,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed_router", None),
        intermediate_dim=self.moe_intermediate_size,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        rngs=rngs,
    )

    # Shared Experts
    shared_intermediate_dim = getattr(cfg, "shared_expert_mlp_dim", self.moe_intermediate_size * cfg.shared_experts)
    self.shared_experts = KimiMLP(
        config=cfg,
        mesh=mesh,
        in_features=self.hidden_dim,
        intermediate_dim=shared_intermediate_dim,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      out_sharding=None,
      deterministic: bool = False,
  ) -> jax.Array:
    identity = hidden_states

    # Latent projection down
    latent_x = self.routed_expert_down_proj(hidden_states)

    # Routed experts forward pass
    routed_out, _, _ = self.MoeBlock_0(latent_x, gate_inputs=hidden_states, out_sharding=out_sharding)

    # Latent projection up
    routed_up = self.routed_expert_up_proj(self.routed_expert_norm(routed_out))

    # Shared experts forward pass
    shared_out = self.shared_experts(identity)
    return routed_up + shared_out


class KimiDecoderLayer(nnx.Module):
  """Unified Kimi-K3 decoder layer handling KDA / MLA attention and Attention Residual connections."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: Optional[Quant] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config

    self.is_kda = (layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval != 0
    self.is_dense = layer_idx < cfg.first_num_dense_layers

    # Norm layers
    self.input_layernorm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # Attention Residual connections
    self.self_attention_res_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.self_attention_res_proj = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=1,
        axis=-1,
        use_bias=False,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", None),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )
    self.mlp_res_norm = RMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        shard_mode=cfg.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.mlp_res_proj = DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=1,
        axis=-1,
        use_bias=False,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", None),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # Attention block
    if self.is_kda:
      self.self_attention = KimiDeltaAttention(
          config=cfg,
          mesh=mesh,
          model_mode=model_mode,
          quant=quant,
          rngs=rngs,
      )
    else:
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
      dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
      self.self_attention = KimiMLAAttention(
          config=cfg,
          num_query_heads=cfg.num_query_heads,
          num_kv_heads=cfg.num_kv_heads,
          head_dim=cfg.head_dim,
          max_target_length=cfg.max_target_length,
          max_prefill_predict_length=cfg.max_prefill_predict_length,
          attention_kernel=cfg.attention,
          attention_type=attentions.AttentionType.MLA,
          inputs_q_shape=dummy_inputs_shape,
          inputs_kv_shape=dummy_inputs_shape,
          mesh=mesh,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          dropout_rate=cfg.dropout_rate,
          quant=quant,
          kv_quant=quantizations.configure_kv_quant(cfg),
          q_lora_rank=cfg.q_lora_rank,
          kv_lora_rank=cfg.kv_lora_rank,
          qk_nope_head_dim=cfg.qk_nope_head_dim,
          qk_rope_head_dim=cfg.qk_rope_head_dim,
          v_head_dim=cfg.v_head_dim,
          max_position_embeddings=cfg.max_position_embeddings,
          original_max_position_embeddings=cfg.original_max_position_embeddings,
          mscale=cfg.mscale,
          rope_factor=cfg.rope_factor,
          model_mode=model_mode,
          rngs=rngs,
      )

    # MLP block
    if self.is_dense:
      self.mlp = KimiMLP(
          config=cfg,
          mesh=mesh,
          in_features=cfg.emb_dim,
          intermediate_dim=getattr(cfg, "intermediate_size", getattr(cfg, "mlp_dim", 33792)),
          rngs=rngs,
      )
    else:
      self.mlp = KimiSparseMoeBlock(
          config=cfg,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )

  def _apply_attn_res(self, prefix_sum: jax.Array, block_residual: Optional[jax.Array], proj, norm) -> jax.Array:
    """Applies attention residual connection across blocks."""
    if block_residual is None or block_residual.shape[1] == 0:
      return prefix_sum

    # v: [Tokens, NumBlocks + 1, D]
    v = jnp.concatenate([block_residual, prefix_sum[:, None, :]], axis=1)
    v_f = v.astype(jnp.float32)
    var = jnp.mean(v_f**2, axis=-1, keepdims=True)
    k = v_f * jax.lax.rsqrt(var + norm.epsilon)

    # score_weight: norm.scale * proj.kernel.squeeze(-1)
    w_norm = norm.scale[...].astype(jnp.float32)
    w_proj = proj.kernel[...].squeeze(-1).astype(jnp.float32)
    score_weight = w_norm * w_proj
    scores = jnp.sum(k * score_weight, axis=-1)  # [Tokens, NumBlocks + 1]
    probs = jax.nn.softmax(scores, axis=-1)[:, :, None]
    hidden_states = jnp.sum(probs * v_f, axis=1)
    return hidden_states.astype(prefix_sum.dtype)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      decoder_positions: Optional[jax.Array] = None,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      kv_cache=None,
      block_residual: Optional[jax.Array] = None,
      **kwargs,
  ) -> tuple[jax.Array, Any, jax.Array]:
    batch, seq_len, hidden_size = inputs.shape
    prefix_sum = inputs
    tokens = batch * seq_len
    flat_inputs = jnp.reshape(inputs, (tokens, hidden_size))

    # 1. Apply Attention Residual prior to attention
    if block_residual is not None and block_residual.shape[1] > 0:
      flat_inputs = self._apply_attn_res(
          flat_inputs,
          block_residual,
          self.self_attention_res_proj,
          self.self_attention_res_norm,
      )
      inputs = jnp.reshape(flat_inputs, (batch, seq_len, hidden_size))

    # Append to block_residual if at block boundary (layer 0)
    if self.layer_idx % self.config.attn_res_block_size == 0:
      flat_prefix = jnp.reshape(prefix_sum, (tokens, hidden_size))
      if block_residual is None:
        block_residual = flat_prefix[:, None, :]
      else:
        block_residual = jnp.concatenate([block_residual, flat_prefix[:, None, :]], axis=1)
      prefix_sum = None

    # 2. Self Attention
    lnx = self.input_layernorm(inputs)
    if self.is_kda:
      attn_out, new_kv_cache = self.self_attention(
          lnx,
          kv_cache=kv_cache,
          model_mode=model_mode,
          decoder_segment_ids=decoder_segment_ids,
          **kwargs,
      )
    else:
      attn_out, new_kv_cache = self.self_attention(
          inputs_q=lnx,
          inputs_kv=lnx,
          inputs_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          kv_cache=kv_cache,
          **kwargs,
      )

    prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

    # 3. Apply Attention Residual prior to MLP
    flat_prefix = jnp.reshape(prefix_sum, (tokens, hidden_size))
    flat_mlp_in = self._apply_attn_res(
        flat_prefix,
        block_residual,
        self.mlp_res_proj,
        self.mlp_res_norm,
    )
    mlp_in = jnp.reshape(flat_mlp_in, (batch, seq_len, hidden_size))

    # 4. MLP Forward Pass
    mlp_lnx = self.post_attention_layernorm(mlp_in)
    mlp_out = self.mlp(mlp_lnx)

    prefix_sum = prefix_sum + mlp_out
    return prefix_sum, new_kv_cache, block_residual
