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

"""inclusionAI Ming (BailingMoeV3 audio) model: hybrid KDA/MLA MoE decoder + whisper audio encoder.

Reference: https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny (`modeling_bailing_moe_v3.py`,
`modeling_bailing_moe_v3_audio.py`).

Text decoder layer `i` (cycle length `inhomogeneous_layer_cycle_interval`, 4 for Ming-3.0-audio-tiny):
  * attention: gated MLA when `(i + 1) % cycle == 0`, Kimi Delta Attention (KDA) otherwise.
  * MLP: dense SwiGLU for `i < first_num_dense_layers`, DeepSeek-style sigmoid-routed MoE otherwise.

Audio encoder: OpenAI-whisper encoder (no 30 s padding) followed by a 3x strided Conv1d + linear
projector whose outputs are L2-normalized and scattered into `<|audio_pad|>` token positions.
"""

# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any, Optional

from flax import nnx
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import (
    Array,
    AttentionType,
    BATCH_ATTN,
    Config,
    D_KV,
    HEAD,
    LENGTH,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_KV_BATCH,
    PREFILL_LENGTH,
)
from maxtext.inference import kvcache
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical


def is_mla_layer(layer_idx: int, config: Config) -> bool:
  """Mirrors `BailingMoeV3DecoderLayer.attention_layer_type`."""
  cycle = config.inhomogeneous_layer_cycle_interval
  num_layers = config.num_decoder_layers
  return (layer_idx + 1) % cycle == 0 or layer_idx >= num_layers // cycle * cycle


def is_moe_layer(layer_idx: int, config: Config) -> bool:
  return config.num_experts > 1 and layer_idx >= config.first_num_dense_layers


def scan_layout(config: Config) -> tuple[int, int, int]:
  """Static `scan_layers=True` layout: (num_prefix_layers, num_scanned_blocks, remainder_start).

  Layers [0, num_prefix_layers) are unrolled: the first attention cycle(s) containing the dense-MLP
  layers. The following `num_scanned_blocks` cycles are homogeneous ([KDA+MoE] x (cycle-1) + [MLA+MoE])
  and are scanned as `MingScannableBlock`s. Layers [remainder_start, num_decoder_layers) are unrolled.
  For Ming-3.0-audio-tiny (24 layers, cycle 4, 1 dense layer): (4, 5, 24).
  """
  cycle = config.inhomogeneous_layer_cycle_interval
  num_layers = config.num_decoder_layers
  num_prefix = min(num_layers, -(-config.first_num_dense_layers // cycle) * cycle)
  num_blocks = (num_layers - num_prefix) // cycle
  return num_prefix, num_blocks, num_prefix + num_blocks * cycle


# -----------------------------------------
# Kimi Delta Attention
# -----------------------------------------


def l2_normalize(x: Array, eps: float = 1e-6) -> Array:
  """fla `l2norm`: x / sqrt(sum(x^2) + eps), computed in float32."""
  x = x.astype(jnp.float32)
  return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def kda_gate(g: Array, a_log: Array, dt_bias: Array, lower_bound: float) -> Array:
  """KDA forget gate in log space (fla `safe_gate` / lower-bound variant), float32.

  Args:
    g: Raw gate projection `[B, S, H, K]`.
    a_log: Per-head decay parameter `[H]`.
    dt_bias: Per-channel bias `[H * K]`.
    lower_bound: Lower bound of the gate (e.g. -5).

  Returns:
    `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, shape `[B, S, H, K]`, in [lower_bound, 0).
  """
  num_heads, head_dim = g.shape[-2:]
  g = g.astype(jnp.float32) + dt_bias.astype(jnp.float32).reshape(num_heads, head_dim)
  return lower_bound * jax.nn.sigmoid(jnp.exp(a_log.astype(jnp.float32))[:, None] * g)


def kda_step(state: Array, q_t: Array, k_t: Array, v_t: Array, g_t: Array, beta_t: Array):
  """One step of the KDA delta rule.

  Args:
    state: `[B, H, K, V]` recurrent state.
    q_t: `[B, H, K]` (already l2-normalized and scaled), k_t: `[B, H, K]`, v_t: `[B, H, V]`.
    g_t: `[B, H, K]` per-channel log decay, beta_t: `[B, H]`.

  Returns:
    (new_state, o_t) with o_t `[B, H, V]`.
  """
  hp = lax.Precision.HIGHEST
  state = state * jnp.exp(g_t)[..., None]
  kv_mem = jnp.einsum("bhk,bhkv->bhv", k_t, state, precision=hp)
  delta = beta_t[..., None] * (v_t - kv_mem)
  state = state + jnp.einsum("bhk,bhv->bhkv", k_t, delta, precision=hp)
  o_t = jnp.einsum("bhk,bhkv->bhv", q_t, state, precision=hp)
  return state, o_t


def kda_recurrence(q: Array, k: Array, v: Array, g: Array, beta: Array, initial_state: Array):
  """Token-by-token KDA recurrence (fla `naive_recurrent_kda`), float32.

  Args:
    q, k: `[B, S, H, K]`, v: `[B, S, H, V]`, g: `[B, S, H, K]`, beta: `[B, S, H]`.
    initial_state: `[B, H, K, V]`.

  Returns:
    (o `[B, S, H, V]`, final_state `[B, H, K, V]`).
  """

  def body(state, xs):
    return kda_step(state, *xs)

  xs = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta))
  final_state, o = lax.scan(body, initial_state, xs)
  return jnp.moveaxis(o, 0, 1), final_state


class MingKimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention (KDA) as used by BailingMoeV3 (`no_kda_lora=True`, `kda_safe_gate=True`).

  y = o_proj(RMSNorm(KDA(q, k, v, g, beta)) * sigmoid(g_proj(x))), where
    q, k, v = SiLU(causal_depthwise_conv(q/k/v_proj(x)))
    g = lower_bound * sigmoid(exp(A_log) * (f_proj(x) + dt_bias))   (per channel, log space)
    beta = sigmoid(b_proj(x))
    KDA: S_t = diag(exp(g_t)) S_{t-1};  S_t += beta_t k_t (v_t - k_t^T S_t)^T;  o_t = (q_t / sqrt(K))^T S_t
  q, k are L2-normalized (eps 1e-6) before the recurrence.

  In inference modes the recurrent state and the last `kernel - 1` conv inputs are kept in a
  `KVCache(is_gdn=True)`, the same container used by the Qwen3-Next Gated DeltaNet.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | quantizations.AqtQuantization = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = config
    self.num_heads = cfg.num_query_heads
    self.head_dim = cfg.head_dim
    self.proj_dim = self.num_heads * self.head_dim
    self.conv_kernel_size = cfg.gdn_conv_kernel_dim
    self.lower_bound = cfg.kda_lower_bound

    def dense(out_features, name):
      del name
      return DenseGeneral(
          in_features_shape=cfg.emb_dim,
          out_features_shape=out_features,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("embed_attn", "gdn_head"),
          matmul_precision=cfg.matmul_precision,
          shard_mode=cfg.shard_mode,
          quant=quant,
          rngs=rngs,
      )

    self.q_proj = dense(self.proj_dim, "q_proj")
    self.k_proj = dense(self.proj_dim, "k_proj")
    self.v_proj = dense(self.proj_dim, "v_proj")
    self.f_proj = dense(self.proj_dim, "f_proj")
    self.b_proj = dense(self.num_heads, "b_proj")
    self.g_proj = dense(self.proj_dim, "g_proj")

    def depthwise_conv():
      return nnx.Conv(
          in_features=self.proj_dim,
          out_features=self.proj_dim,
          kernel_size=(self.conv_kernel_size,),
          feature_group_count=self.proj_dim,
          padding="VALID",
          use_bias=False,
          dtype=cfg.dtype,
          param_dtype=cfg.weight_dtype,
          precision=cfg.matmul_precision,
          rngs=rngs,
      )

    self.q_conv1d = depthwise_conv()
    self.k_conv1d = depthwise_conv()
    self.v_conv1d = depthwise_conv()

    def a_log_init(key, shape, dtype=jnp.float32):
      return jnp.log(jax.random.uniform(key, shape, dtype=jnp.float32, minval=1.0, maxval=16.0)).astype(dtype)

    # A_log and dt_bias are stored in float32 in the HF checkpoint.
    self.A_log = nnx.Param(a_log_init(rngs.params(), (self.num_heads,)))  # pylint: disable=invalid-name
    self.dt_bias = nnx.Param(jnp.zeros((self.proj_dim,), dtype=jnp.float32))

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
        in_features_shape=self.proj_dim,
        out_features_shape=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("gdn_head", "embed_attn"),
        matmul_precision=cfg.matmul_precision,
        shard_mode=cfg.shard_mode,
        quant=quant,
        rngs=rngs,
    )

    if model_mode != MODEL_MODE_TRAIN:
      batch_size, _ = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
      self.cache = kvcache.KVCache(
          max_prefill_length=cfg.max_prefill_predict_length,
          max_target_length=cfg.max_target_length,
          batch=batch_size,
          key_seq_len=1,
          value_seq_len=1,
          key_heads=self.num_heads,
          value_heads=self.num_heads,
          key_head_size=self.head_dim,
          value_head_size=self.head_dim,
          dtype=cfg.dtype,
          is_gdn=True,
          conv_kernel_size=self.conv_kernel_size,
          conv_dim=3 * self.proj_dim,
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      self.cache = None

  @staticmethod
  def _match_batch(x: Array, batch: int) -> Array:
    """Broadcasts / pads / slices the leading (batch) axis of a cached state to `batch`."""
    if x.shape[0] == batch:
      return x
    if x.shape[0] == 1:
      return jnp.broadcast_to(x, (batch,) + x.shape[1:])
    if x.shape[0] < batch:
      return jnp.pad(x, [(0, batch - x.shape[0])] + [(0, 0)] * (x.ndim - 1))
    return x[:batch]

  def __call__(
      self,
      hidden_states: Array,
      model_mode: str = MODEL_MODE_TRAIN,
      decoder_segment_ids: None | Array = None,
  ) -> Array:
    cfg = self.config
    batch, seq_len, _ = hidden_states.shape
    h, d = self.num_heads, self.head_dim

    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    qkv = jnp.concatenate([q, k, v], axis=-1)  # [B, S, 3C]

    # ---- causal depthwise short convolution (with cached history in inference) ----
    kernel = self.conv_kernel_size
    use_cache = model_mode != MODEL_MODE_TRAIN and self.cache is not None
    recurrent_state = None
    next_conv_state = None
    if use_cache:
      recurrent_state, conv_state = self.cache.get_gdn_states()
      cache_batch = conv_state.shape[0]
      conv_state = self._match_batch(conv_state, batch).astype(qkv.dtype)
      recurrent_state = self._match_batch(recurrent_state, batch)
      conv_input = jnp.concatenate([conv_state, qkv], axis=1)
      if decoder_segment_ids is not None and model_mode == MODEL_MODE_PREFILL:
        valid_lens = jnp.sum(decoder_segment_ids != 0, axis=1)
        next_conv_state = jax.vmap(lambda c, n: lax.dynamic_slice_in_dim(c, n, kernel - 1, axis=0))(
            conv_input, valid_lens
        )
      else:
        next_conv_state = conv_input[:, -(kernel - 1) :, :]
    else:
      conv_input = jnp.pad(qkv, ((0, 0), (kernel - 1, 0), (0, 0)))

    q_in, k_in, v_in = jnp.split(conv_input, [self.proj_dim, 2 * self.proj_dim], axis=-1)
    q = jax.nn.silu(self.q_conv1d(q_in).astype(jnp.float32))
    k = jax.nn.silu(self.k_conv1d(k_in).astype(jnp.float32))
    v = jax.nn.silu(self.v_conv1d(v_in).astype(jnp.float32))

    q = q.reshape(batch, seq_len, h, d)
    k = k.reshape(batch, seq_len, h, d)
    v = v.reshape(batch, seq_len, h, d)

    # ---- gates ----
    g = kda_gate(
        self.f_proj(hidden_states).reshape(batch, seq_len, h, d), self.A_log[...], self.dt_bias[...], self.lower_bound
    )
    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))  # [B, S, H]

    q = l2_normalize(q) * (d**-0.5)
    k = l2_normalize(k)

    if decoder_segment_ids is not None:
      # Padded positions must leave the recurrent state untouched: k = 0 -> no write, g = 0 -> no decay.
      valid = (decoder_segment_ids != 0)[..., None, None]
      k = jnp.where(valid, k, 0.0)
      v = jnp.where(valid, v, 0.0)
      g = jnp.where(valid, g, 0.0)

    if recurrent_state is None:
      recurrent_state = jnp.zeros((batch, h, d, d), dtype=jnp.float32)
    recurrent_state = recurrent_state.astype(jnp.float32)

    if seq_len == 1 and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      new_state, o = kda_step(recurrent_state, q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0])
      o = o[:, None]
    else:
      o, new_state = kda_recurrence(q, k, v, g, beta, recurrent_state)

    if use_cache:
      self.cache.update_gdn_states(
          self._match_batch(new_state, cache_batch).astype(cfg.dtype),  # pyrefly: ignore[unbound-name]
          self._match_batch(next_conv_state, cache_batch).astype(cfg.dtype),  # pyrefly: ignore[unbound-name]
      )

    # ---- gated RMSNorm + output projection ----
    gate = self.g_proj(hidden_states).reshape(batch, seq_len, h, d)
    o = self.o_norm(o.astype(cfg.dtype)).astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
    o = o.astype(cfg.dtype).reshape(batch, seq_len, self.proj_dim)
    return self.o_proj(o)


# -----------------------------------------
# Gated Multi-head Latent Attention
# -----------------------------------------


class MingMLA(attention_mla.MLA):
  """MaxText MLA with BailingMoeV3's head-wise sigmoid output gate (`gated_attention_proj_granularity_type`).

  out = dense((softmax(q k^T / sqrt(qk_head_dim)) v) * sigmoid(g_proj(x))[..., None])
  RoPE is applied to the interleaved rope dims through YaRN with `rope_factor=1` (plain RoPE).
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.g_proj = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.num_query_heads,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_attn", "q_heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding=None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      bidirectional_mask: Optional[Any] = None,
      rope_kwargs: dict | None = None,
      kv_cache: Optional[Array] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
  ):
    del out_sharding, deterministic, slot, bidirectional_mask, rope_kwargs, attention_metadata
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.prefill_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.prefill_input_axis_names)
      out_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV)
    else:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.input_axis_names)
      out_logical_name = (BATCH_ATTN, LENGTH, HEAD, D_KV)

    if model_mode != MODEL_MODE_TRAIN and decoder_segment_ids is None:
      decoder_segment_ids = jnp.ones(inputs_q.shape[:2], dtype=jnp.int32)

    query, _ = self.mla_query_projection(inputs_q, inputs_positions, model_mode)
    key, value, cached_values = self.mla_kv_projection(
        inputs_kv, inputs_positions, decoder_segment_ids, model_mode, previous_chunk
    )
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    out = self.attention_op(query, key, value, decoder_segment_ids, inputs_positions, model_mode, cached_values)

    gate = jax.nn.sigmoid(self.g_proj(inputs_q).astype(jnp.float32))  # [B, S, H]
    out = (out.astype(jnp.float32) * gate[..., None]).astype(out.dtype)

    out = self._maybe_shard_with_logical(out, self.out_axis_names)
    out = self.out_projection(out, out_sharding=create_sharding(self.mesh, out_logical_name))
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache


# -----------------------------------------
# Decoder layer
# -----------------------------------------


class MingDecoderLayer(nnx.Module):
  """BailingMoeV3 decoder layer: pre-norm (KDA | gated MLA) + pre-norm (dense MLP | MoE)."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: None | quantizations.AqtQuantization = None,
      layer_idx: int = -1,
      *,
      is_mla: bool | None = None,
      is_moe: bool | None = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.layer_idx = layer_idx
    cfg = config

    # Inside a scanned block the absolute layer index is not known, so the layer type is passed explicitly.
    self.is_mla = is_mla_layer(layer_idx, cfg) if is_mla is None else is_mla
    self.is_moe = is_moe_layer(layer_idx, cfg) if is_moe is None else is_moe

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
    length_name = "prefill_activation_norm_length" if model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    self.activation_axis_names = ("activation_batch", length_name, "activation_embed")
    self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
    self.mlp_intermediate_sharding = create_sharding(
        mesh, ("activation_batch", length_name, "activation_mlp"), rules=get_logical_axis_rules()
    )

    norm = functools.partial(
        RMSNorm,
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.pre_self_attention_layer_norm = norm()
    self.post_self_attention_layer_norm = norm()

    if self.is_mla:
      self.self_attention = MingMLA(
          config=cfg,
          num_query_heads=cfg.num_query_heads,
          num_kv_heads=cfg.num_kv_heads,
          head_dim=cfg.head_dim,
          max_target_length=cfg.max_target_length,
          max_prefill_predict_length=cfg.max_prefill_predict_length,
          attention_kernel=cfg.attention,
          attention_type=AttentionType(cfg.attention_type),
          inputs_q_shape=dummy_inputs_shape,
          inputs_kv_shape=dummy_inputs_shape,
          mesh=mesh,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          dropout_rate=cfg.dropout_rate,
          float32_qk_product=cfg.float32_qk_product,
          float32_logits=cfg.float32_logits,
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
    else:
      self.kda = MingKimiDeltaAttention(config=cfg, mesh=mesh, model_mode=model_mode, quant=quant, rngs=rngs)

    if self.is_moe:
      self.moe_block = moe.RoutedAndSharedMoE(
          config=cfg,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(cfg.dense_init_scale, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = linears.MlpBlock(
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          config=cfg,
          quant=quant,
          model_mode=model_mode,
          mesh=mesh,
          rngs=rngs,
      )

  def _shard(self, x):
    return maybe_shard_with_logical(
        x,
        logical_axes=self.activation_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        rules=get_logical_axis_rules(),
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    del attention_metadata, decoder_input_tokens
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self._shard(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    lnx = self._shard(self.pre_self_attention_layer_norm(x))
    if self.is_mla:
      attn, _ = self.self_attention(
          lnx,
          lnx,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
      )
    else:
      attn = self.kda(lnx, model_mode=model_mode, decoder_segment_ids=decoder_segment_ids)
    x = self._shard(x + self._shard(attn))

    hx = self._shard(self.post_self_attention_layer_norm(x))
    load_balance_loss = None
    moe_bias_updates = None
    if self.is_moe:
      mlp_out, load_balance_loss, moe_bias_updates = self.moe_block(
          hx, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding
      )
    else:
      mlp_out = self.mlp(
          hx,
          deterministic=deterministic,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
      )
    out = self._shard(x + self._shard(mlp_out))

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)
    return out, kv_cache


class MingScannableBlock(nnx.Module):
  """One homogeneous attention cycle: `layers_0..layers_{cycle-2}` are KDA + MoE, the last is MLA + MoE."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | quantizations.AqtQuantization = None,
  ):
    self.config = config
    self.num_layers = config.inhomogeneous_layer_cycle_interval
    for j in range(self.num_layers):
      setattr(
          self,
          f"layers_{j}",
          MingDecoderLayer(
              config=config,
              model_mode=model_mode,
              mesh=mesh,
              rngs=rngs,
              quant=quant,
              is_mla=j == self.num_layers - 1,
              is_moe=True,
          ),
      )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    del attention_metadata, decoder_input_tokens
    y = inputs[0] if isinstance(inputs, tuple) else inputs
    for j in range(self.num_layers):
      y, _ = getattr(self, f"layers_{j}")(
          y, decoder_segment_ids, decoder_positions, deterministic, model_mode, previous_chunk=previous_chunk, slot=slot
      )
    return y, kv_cache


# -----------------------------------------
# Whisper audio encoder + projector
# -----------------------------------------


def exact_gelu(x: Array) -> Array:
  """torch.nn.GELU() (erf form)."""
  return jax.nn.gelu(x, approximate=False)


class MingWhisperAttention(nnx.Module):
  """whisper.model.MultiHeadAttention (key projection has no bias), bidirectional, no mask."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    d = config.d_model_for_audio
    self.num_heads = config.encoder_attention_heads_for_audio
    self.head_dim = d // self.num_heads

    def proj(use_bias):
      return DenseGeneral(
          in_features_shape=d,
          out_features_shape=d,
          use_bias=use_bias,
          dtype=config.dtype_mm,
          weight_dtype=config.weight_dtype,
          matmul_precision=config.matmul_precision,
          rngs=rngs,
      )

    self.query = proj(True)
    self.key = proj(False)
    self.value = proj(True)
    self.out = proj(True)

  def __call__(self, x: Array) -> Array:
    b, t, d = x.shape
    q = self.query(x).reshape(b, t, self.num_heads, self.head_dim)
    k = self.key(x).reshape(b, t, self.num_heads, self.head_dim)
    v = self.value(x).reshape(b, t, self.num_heads, self.head_dim)
    hp = lax.Precision.HIGHEST
    logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32), precision=hp)
    probs = jax.nn.softmax(logits * (self.head_dim**-0.5), axis=-1)
    o = jnp.einsum("bhqk,bkhd->bqhd", probs, v.astype(jnp.float32), precision=hp)
    return self.out(o.reshape(b, t, d).astype(x.dtype))


class MingWhisperBlock(nnx.Module):
  """whisper.model.ResidualAttentionBlock (encoder, no cross-attention)."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    d = config.d_model_for_audio

    def ln():
      return nnx.LayerNorm(
          num_features=d, epsilon=1e-5, dtype=config.dtype_mm, param_dtype=config.weight_dtype, rngs=rngs
      )

    def dense(i, o):
      return DenseGeneral(
          in_features_shape=i,
          out_features_shape=o,
          use_bias=True,
          dtype=config.dtype_mm,
          weight_dtype=config.weight_dtype,
          matmul_precision=config.matmul_precision,
          rngs=rngs,
      )

    self.attn_ln = ln()
    self.attn = MingWhisperAttention(config, rngs=rngs)
    self.mlp_ln = ln()
    self.mlp_fc1 = dense(d, config.encoder_ffn_dim_for_audio)
    self.mlp_fc2 = dense(config.encoder_ffn_dim_for_audio, d)

  def __call__(self, x: Array) -> Array:
    x = x + self.attn(self.attn_ln(x))
    x = x + self.mlp_fc2(exact_gelu(self.mlp_fc1(self.mlp_ln(x))))
    return x


def whisper_conv_out_length(length: int) -> int:
  """Length after conv1 (k3, s1, p1) and conv2 (k3, s2, p1)."""
  length = (length + 2 * 1 - 3) // 1 + 1
  return (length + 2 * 1 - 3) // 2 + 1


def projector_out_length(length: int, num_convs: int) -> int:
  """Length after `num_convs` Conv1d(k3, s2, p1) downsamplers."""
  for _ in range(num_convs):
    length = (length + 2 * 1 - 3) // 2 + 1
  return length


def audio_chunk_lengths(mel_length: int, chunk_size: int) -> list[int]:
  """Static mel lengths of the independent whisper chunks (HF `_get_audio_features`)."""
  return [min(chunk_size, mel_length - start) for start in range(0, mel_length, chunk_size)]


class MingWhisperEncoder(nnx.Module):
  """WhisperEncoderForBailing: whisper encoder without 30 s padding, applied per <=30 s chunk.

  Input: log-mel features `[1, T_mel, n_mels]`. Output: per-chunk features `[num_chunks, L, d_model]`
  where chunks are zero-padded to the longest chunk like `torch.nn.utils.rnn.pad_sequence` (no mask).
  """

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    d = config.d_model_for_audio
    self.conv1 = nnx.Conv(
        in_features=config.num_mel_bins_for_audio,
        out_features=d,
        kernel_size=(3,),
        padding=((1, 1),),
        use_bias=True,
        dtype=config.dtype_mm,
        param_dtype=config.weight_dtype,
        precision=config.matmul_precision,
        rngs=rngs,
    )
    self.conv2 = nnx.Conv(
        in_features=d,
        out_features=d,
        kernel_size=(3,),
        strides=(2,),
        padding=((1, 1),),
        use_bias=True,
        dtype=config.dtype_mm,
        param_dtype=config.weight_dtype,
        precision=config.matmul_precision,
        rngs=rngs,
    )
    self.positional_embedding = nnx.Param(
        jnp.zeros((config.max_source_positions_for_audio, d), dtype=config.weight_dtype)
    )
    for i in range(config.encoder_layers_for_audio):
      setattr(self, f"blocks_{i}", MingWhisperBlock(config, rngs=rngs))
    self.ln_post = nnx.LayerNorm(
        num_features=d, epsilon=1e-5, dtype=config.dtype_mm, param_dtype=config.weight_dtype, rngs=rngs
    )

  def __call__(self, audio_features: Array, deterministic: bool = False) -> Array:
    del deterministic
    assert audio_features.ndim == 3 and audio_features.shape[0] == 1, "expected mel features [1, T_mel, n_mels]"
    mel_length = audio_features.shape[1]
    lengths = audio_chunk_lengths(mel_length, self.config.ming_audio_chunk_size)
    max_len = max(lengths)
    chunks = []
    for i, n in enumerate(lengths):
      start = i * self.config.ming_audio_chunk_size
      c = audio_features[0, start : start + n]
      chunks.append(jnp.pad(c, ((0, max_len - n), (0, 0))))
    x = jnp.stack(chunks).astype(self.config.dtype_mm)  # [num_chunks, max_len, n_mels]

    x = exact_gelu(self.conv1(x))
    x = exact_gelu(self.conv2(x))
    x = x + self.positional_embedding[...][: x.shape[1]].astype(x.dtype)
    for i in range(self.config.encoder_layers_for_audio):
      x = getattr(self, f"blocks_{i}")(x)
    return self.ln_post(x)


class MingAudioProjector(nnx.Module):
  """`linear_proj_audio`: [Conv1d(k3, s2, p1) + GELU] x num_convs -> Linear, then L2-normalize."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    in_dim = config.d_model_for_audio
    out_dim = config.output_dim_for_audio
    for i in range(config.num_conv_layers_for_audio):
      setattr(
          self,
          f"conv_{i}",
          nnx.Conv(
              in_features=in_dim if i == 0 else out_dim,
              out_features=out_dim,
              kernel_size=(3,),
              strides=(2,),
              padding=((1, 1),),
              use_bias=True,
              dtype=config.dtype_mm,
              param_dtype=config.weight_dtype,
              precision=config.matmul_precision,
              rngs=rngs,
          ),
      )
    self.linear = DenseGeneral(
        in_features_shape=out_dim,
        out_features_shape=out_dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

  def __call__(self, encoded: Array, mel_length: int | None = None) -> Array:
    """Projects `[num_chunks, L, d_audio]` encoder outputs to `[1, num_audio_tokens, emb_dim]`."""
    x = encoded
    for i in range(self.config.num_conv_layers_for_audio):
      x = exact_gelu(getattr(self, f"conv_{i}")(x))
    x = self.linear(x)

    if mel_length is None:
      valid = [x.shape[1]] * x.shape[0]
    else:
      valid = [
          projector_out_length(whisper_conv_out_length(n), self.config.num_conv_layers_for_audio)
          for n in audio_chunk_lengths(mel_length, self.config.ming_audio_chunk_size)
      ]
    x = jnp.concatenate([x[i, :n] for i, n in enumerate(valid)], axis=0)[None]
    if self.config.ming_norm_audio_embeds:
      xf = x.astype(jnp.float32)
      norm = jnp.sqrt(jnp.sum(xf * xf, axis=-1, keepdims=True))
      x = (xf / jnp.maximum(norm, 1e-12)).astype(x.dtype)
    return x
