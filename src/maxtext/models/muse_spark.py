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

"""Meta Muse-Spark (`MuseSparkForConditionalGeneration` text backbone) decoder layers.

Reference: https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B
(`transformers.models.muse_spark.modeling_muse_spark`).
"""

# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.common.common_types import (
    Array,
    AttentionType,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    DType,
    D_KV,
    HEAD,
    KV_HEAD,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    ShardMode,
    get_weight_dtype,
)
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils import sharding
from maxtext.utils.sharding import (
    create_sharding,
    get_logical_axis_rules,
    logical_to_mesh_axes,
    mesh_axes_for_dim,
    remove_incompatible_mesh_axes_from_partition_spec,
)


# ------------------------------------------------------------------------------
# Layer Schedule & Scan Layout Helpers
# ------------------------------------------------------------------------------


def is_full_attention_layer(layer_idx: int, config: Config) -> bool:
  """Returns True when `layer_idx` uses global NoPE attention instead of sliding-window RoPE attention.

  Matches `MuseSparkTextConfig`:
    `layer_types[i] = "full_attention" if (num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"`
    `layer_rope_theta[i] = 0 if (num_hidden_layers - 1 - i) % 4 == 0 else rope_theta`
  """
  return (config.num_decoder_layers - 1 - layer_idx) % config.inhomogeneous_layer_cycle_interval == 0


def is_moe_layer(layer_idx: int, config: Config) -> bool:
  """Returns True when `layer_idx` uses the bottlenecked Sparse MoE block."""
  return config.num_experts > 1 and layer_idx >= config.first_num_dense_layers


def scan_layout(config: Config) -> tuple[int, int, int]:
  """Returns `(num_prefix_layers, num_blocks, remainder_start)` for `scan_layers=True`.

  Because `(num_layers - 1 - i) % cycle == 0` places `full_attention` at the last sub-layer
  of every `cycle`-layer block aligned with the end of the stack (`i % cycle == num_layers % cycle`),
  unrolling the leading `num_prefix = num_layers % cycle` layers leaves an exact integer number
  of homogeneous `[sliding, ..., sliding, full]` MoE cycles of length `cycle`.
  If `first_num_dense_layers > num_layers % cycle`, whole cycles overlapping the dense prefix
  are also unrolled into `num_prefix`.
  """
  cycle = config.inhomogeneous_layer_cycle_interval
  n_layers = config.num_decoder_layers
  if config.num_experts <= 1 or cycle <= 0:
    return n_layers, 0, n_layers
  align = n_layers % cycle
  if config.first_num_dense_layers <= align:
    num_prefix = align
  else:
    extra = config.first_num_dense_layers - align
    num_prefix = align + ((extra + cycle - 1) // cycle) * cycle
  num_prefix = min(n_layers, num_prefix)
  num_blocks = (n_layers - num_prefix) // cycle
  remainder_start = num_prefix + num_blocks * cycle
  return num_prefix, num_blocks, remainder_start


# ------------------------------------------------------------------------------
# Normalization & Residual Gate Modules
# ------------------------------------------------------------------------------


def _rms_normalize(x_f32: jax.Array, epsilon: float) -> jax.Array:
  """Returns `x * rsqrt(mean(x^2) + epsilon)` over the last axis, in float32."""
  mean2 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
  return x_f32 * jax.lax.rsqrt(mean2 + epsilon)


def _centered_scale_fp32(scale: jax.Array) -> jax.Array:
  """Computes `(1.0 + scale).float()` in `scale.dtype` first, matching PyTorch `(1.0 + weight).float()`."""
  return jnp.asarray(jnp.asarray(1.0, dtype=scale.dtype) + scale, jnp.float32)


def _input_scaled_rms_norm(x: jax.Array, scale: jax.Array, epsilon: float) -> jax.Array:
  """Scales `x` by `(1.0 + scale).float()` in fp32, rounds to `x.dtype`, then RMS-normalizes in fp32."""
  scaled = (jnp.asarray(x, jnp.float32) * _centered_scale_fp32(scale)).astype(x.dtype).astype(jnp.float32)
  return _rms_normalize(scaled, epsilon).astype(x.dtype)


class MuseSparkRMSNorm(RMSNorm):
  """Muse-Spark RMSNorm (`MuseSparkRMSNorm`), using `pow(mean_sq + eps, -0.5)` in fp32 and casting to `x.dtype`."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: initializers.Initializer = jax.nn.initializers.ones,
      with_scale: bool = True,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=shard_mode,
        kernel_axes=kernel_axes,
        scale_init=scale_init,
        parameter_memory_host_offload=parameter_memory_host_offload,
        with_scale=with_scale,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    del out_sharding
    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
    # Matches `MuseSparkRMSNorm._norm`: `hidden_states * torch.pow(mean_squared, -0.5)`.
    normed = x_f32 * jnp.power(mean2 + self.epsilon, -0.5)
    if self.with_scale and self.scale is not None:
      scale = self.scale[...]
      if self.parameter_memory_host_offload:
        scale = jax.device_put(scale, max_utils.device_space())
      normed = normed * jnp.asarray(scale, jnp.float32)
    return jnp.asarray(normed, x.dtype)


class MuseSparkCenteredRMSNorm(RMSNorm):
  """Muse-Spark zero-centered RMSNorm (`MuseSparkTextCenteredRMSNorm`): `normed * (1.0 + weight).float()`."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=shard_mode,
        kernel_axes=kernel_axes,
        scale_init=jax.nn.initializers.zeros,
        parameter_memory_host_offload=parameter_memory_host_offload,
        with_scale=True,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    normed = _rms_normalize(jnp.asarray(x, jnp.float32), self.epsilon)
    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    output = normed * _centered_scale_fp32(scale)
    return jnp.asarray(output, x.dtype)


class MuseSparkInputScaledRMSNorm(RMSNorm):
  """Muse-Spark input-scaled RMSNorm (`MuseSparkTextInputScaledRMSNorm`).

  Scales the input by `(1.0 + weight).float()` in fp32, casts back to input dtype, then normalizes in fp32:
    `scaled = (x.float() * (1.0 + weight).float()).type_as(x).float()`
    `normed = (scaled * rsqrt(mean(scaled^2) + eps)).type_as(x)`
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-8,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=shard_mode,
        kernel_axes=kernel_axes,
        scale_init=jax.nn.initializers.zeros,
        parameter_memory_host_offload=parameter_memory_host_offload,
        with_scale=True,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    return _input_scaled_rms_norm(x, scale, self.epsilon)


class MuseSparkResidualGate(nnx.Module):
  """Learnable fp32 temperature-scaled residual gate (`MuseSparkResidualGate`)."""

  def __init__(
      self,
      num_features: int,
      temperature: float = 0.3,
      eps: float = 1e-3,
      weight_dtype: Any = jnp.float32,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      gate_init: initializers.Initializer = jax.nn.initializers.normal(stddev=0.02),
      *,
      rngs: nnx.Rngs,
  ):
    del weight_dtype  # Kept in float32 per `MuseSparkPreTrainedModel._keep_in_fp32_modules_strict`.
    self.num_features = num_features
    self.temperature = float(temperature)
    self.eps = float(eps)
    self.gate = nnx.Param(
        gate_init(rngs.params(), (num_features,), jnp.float32),
        sharding=kernel_axes,
    )

  def __call__(self, residual: jax.Array, hidden_states: jax.Array) -> jax.Array:
    w = jnp.asarray(self.gate[...], jnp.float32) / self.temperature
    beta = jax.nn.sigmoid(w)
    alpha = jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-w) * (1.0 + beta), self.eps))
    out = alpha * jnp.asarray(residual, jnp.float32) + beta * jnp.asarray(hidden_states, jnp.float32)
    return out.astype(residual.dtype)


# ------------------------------------------------------------------------------
# Gated GQA Attention
# ------------------------------------------------------------------------------


class MuseSparkAttention(attentions.Attention):
  """Muse-Spark Gated GQA Attention (`MuseSparkTextAttention`).

  Features:
  - Scaleless RMSNorm (`qk_norm`) on Q and K, followed by `query = (qk_norm(query) * qk_scale_factor).to(dtype)`.
  - Non-interleaved RoPE on sliding-window layers (`is_full_attention=False`), NoPE on full-attention layers.
  - Per-head scaleless RMSNorm (`o_norm`) on attention output before multiplying by `sigmoid(gate_proj(inputs_q))`.
  """

  def __init__(
      self,
      config: Config,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: tuple[int, ...],
      inputs_kv_shape: tuple[int, ...],
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      float32_qk_product: bool | None = None,
      float32_logits: bool | None = None,
      quant: quantizations.AqtQuantization | None = None,
      kv_quant: quantizations.KVQuant | None = None,
      is_full_attention: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    attention_type = AttentionType.GLOBAL if is_full_attention else AttentionType.LOCAL_SLIDING
    sliding_window_size = 0 if is_full_attention else int(config.sliding_window_size)
    if float32_qk_product is None:
      float32_qk_product = bool(config.float32_qk_product)
    if float32_logits is None:
      float32_logits = bool(config.float32_logits)
    super().__init__(
        config=config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        max_prefill_predict_length=max_prefill_predict_length,
        dropout_rate=dropout_rate,
        float32_qk_product=float32_qk_product,
        float32_logits=float32_logits,
        quant=quant,
        kv_quant=kv_quant,
        attention_type=attention_type,
        sliding_window_size=sliding_window_size,
        query_pre_attn_scalar=float(head_dim**-0.5),
        is_nope_layer=is_full_attention,
        use_bias_in_projections=config.attention_bias,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.is_full_attention = is_full_attention
    self.qk_scale_factor = float(config.muse_spark_qk_scale_factor)
    # Matches `MuseSparkTextAttention`: `gate_proj` never uses bias even when `attention_bias=True`.
    prev_use_bias = self.use_bias_in_projections
    self.use_bias_in_projections = False
    try:
      self.gate_proj = self.init_query_w(inputs_q_shape=inputs_q_shape)
    finally:
      self.use_bias_in_projections = prev_use_bias
    self.qk_norm = MuseSparkRMSNorm(
        num_features=head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )
    self.o_norm = MuseSparkRMSNorm(
        num_features=head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding: NamedSharding | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: int | None = None,
      bidirectional_mask: Any = None,
      rope_kwargs: dict | None = None,
      kv_cache: Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      **kwargs,
  ):
    del deterministic, slot, kwargs
    if model_mode == MODEL_MODE_PREFILL:
      input_axis_names = self.prefill_input_axis_names
    elif model_mode == MODEL_MODE_TRAIN:
      input_axis_names = self.input_axis_names
    else:
      input_axis_names = self.decode_input_axis_names

    inputs_q = self._maybe_shard_with_logical(inputs_q, input_axis_names)
    inputs_kv = self._maybe_shard_with_logical(inputs_kv, input_axis_names)
    qkv_sharding = create_sharding(self.mesh, input_axis_names)

    query = self.query_projection(inputs_q, out_sharding=qkv_sharding)
    key = self.kv_projection(inputs_kv, proj_name="key", out_sharding=qkv_sharding)
    value = self.kv_projection(inputs_kv, proj_name="value", out_sharding=qkv_sharding)

    query = (self.qk_norm(query) * self.qk_scale_factor).astype(query.dtype)
    key = self.qk_norm(key)

    if not self.is_nope_layer:
      if inputs_positions is None:
        inputs_positions = jnp.arange(inputs_q.shape[1], dtype=jnp.int32)[None, :]
      query = self.apply_rotary_embedding(query, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)
      key = self.apply_rotary_embedding(key, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)

    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      query = (query * self.query_pre_attn_scalar).astype(query.dtype)

    if model_mode == MODEL_MODE_PREFILL:
      query = self._maybe_shard_with_logical(query, self.prefill_query_axis_names)
      key = self._maybe_shard_with_logical(key, self.prefill_key_axis_names)
      value = self._maybe_shard_with_logical(value, self.prefill_value_axis_names)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      query = self._maybe_shard_with_logical(query, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      key = self._maybe_shard_with_logical(key, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
      value = self._maybe_shard_with_logical(value, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
    else:
      query = self._maybe_shard_with_logical(query, self.query_axis_names)
      key = self._maybe_shard_with_logical(key, self.key_axis_names)
      value = self._maybe_shard_with_logical(value, self.value_axis_names)

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa") and model_mode != MODEL_MODE_TRAIN:
      batch, seq_len, num_heads, head_dim = query.shape
      attn_out, updated_kv = self.forward_serve_vllm(
          query, key, value, rpa_kv_cache=kv_cache, rpa_metadata=attention_metadata
      )
      out = attn_out.reshape(batch, seq_len, num_heads, head_dim)
      kv_cache = updated_kv
    else:
      cached_values = [None, None]
      if model_mode != MODEL_MODE_TRAIN:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      out = self.attention_op(
          query,
          key,
          value,
          decoder_segment_ids,
          inputs_positions,
          model_mode,
          cached_values,
          previous_chunk,
          bidirectional_mask,
          self.sinks,
      )

    out = checkpoint_name(out, "attention_out")
    if model_mode == MODEL_MODE_PREFILL:
      out_axis_names = self.prefill_out_axis_names
    elif model_mode == MODEL_MODE_TRAIN:
      out_axis_names = self.out_axis_names
    else:
      out_axis_names = self.decode_out_axis_names
    out = self._maybe_shard_with_logical(out, out_axis_names)

    out = self.o_norm(out)
    gate = jax.nn.sigmoid(self.gate_proj(inputs_q, out_sharding=qkv_sharding).astype(jnp.float32)).astype(out.dtype)
    gate = self._maybe_shard_with_logical(gate, out_axis_names)
    out = out * gate

    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache


# ------------------------------------------------------------------------------
# Bottlenecked Sparse MoE (`MuseSparkTextSparseMoeBlock`)
# ------------------------------------------------------------------------------


class MuseSparkRouterGate(nnx.Module):
  """Router gate holding fp32 `kernel` and `MoEBiasVar` `bias`."""

  def __init__(
      self,
      emb_dim: int,
      num_experts: int,
      use_bias: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.use_bias = use_bias
    self.kernel = nnx.Param(
        initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")(
            rngs.params(), (emb_dim, num_experts), jnp.float32, 0, 1
        ),
        sharding=("embed_router", None),
    )
    if use_bias:
      self.bias = moe.MoEBiasVar(
          jnp.zeros((num_experts,), dtype=jnp.float32),
          sharding=(None,),
      )
    else:
      self.bias = None


class MuseSparkMoE(nnx.Module):
  """Muse-Spark bottlenecked Sparse MoE block (`MuseSparkTextSparseMoeBlock`).

  Structure:
  1. Router (`MuseSparkRouterGate`) operates on the full-width `emb_dim` (8192) hidden states in fp32,
     scales router logits by `muse_spark_output_multiplier`, applies sigmoid + `e_score_correction_bias`
     for top-k selection, and normalizes the un-biased sigmoid weights over the selected top-k experts.
  2. Shared width-bottleneck down-projection `pre_expert_proj` (`emb_dim -> moe_expert_input_dim`)
     followed by `pre_expert_norm` (`MuseSparkCenteredRMSNorm`).
  3. 256 routed SwiGLU experts (`wi_0`, `wi_1`, `wo` of shape `[num_experts, moe_expert_input_dim, moe_mlp_dim]`)
     followed by per-expert-token `post_expert_norm` (`MuseSparkInputScaledRMSNorm`) before weighting by
     `top_k_weights` and summing over the `k` selected experts.
  4. Shared width-bottleneck up-projection `post_expert_proj` (`moe_expert_input_dim -> emb_dim`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: quantizations.AqtQuantization | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.num_experts = int(config.num_experts)
    self.num_experts_per_tok = int(config.num_experts_per_tok)
    self.emb_dim = int(config.emb_dim)
    self.expert_input_dim = int(config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim)
    self.moe_mlp_dim = int(config.moe_mlp_dim)
    self.output_multiplier = float(config.muse_spark_output_multiplier)
    self.routed_scaling_factor = float(config.routed_scaling_factor)
    self.routed_score_func = str(config.routed_score_func or "sigmoid")
    self.routed_bias = bool(config.routed_bias)
    self.norm_topk_prob = bool(config.norm_topk_prob)
    self.float32_gate_logits = bool(config.float32_gate_logits)

    norm_weight_dtype = get_weight_dtype(config, "norm")
    expert_weight_dtype = get_weight_dtype(config, "moe_block")

    self.gate = MuseSparkRouterGate(
        emb_dim=self.emb_dim,
        num_experts=self.num_experts,
        use_bias=self.routed_bias,
        rngs=rngs,
    )
    self.pre_expert_proj = linears.DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.expert_input_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        use_bias=False,
        rngs=rngs,
    )
    self.pre_expert_norm = MuseSparkCenteredRMSNorm(
        num_features=self.expert_input_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    expert_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")
    self.wi_0 = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.expert_input_dim, self.moe_mlp_dim), expert_weight_dtype, 1, 2
        ),
        sharding=("exp", "embed_moe", "mlp_moe"),
    )
    self.wi_1 = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.expert_input_dim, self.moe_mlp_dim), expert_weight_dtype, 1, 2
        ),
        sharding=("exp", "embed_moe", "mlp_moe"),
    )
    self.wo = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.moe_mlp_dim, self.expert_input_dim), expert_weight_dtype, 1, 2
        ),
        sharding=("exp", "mlp_moe", "embed_moe"),
    )
    self.post_expert_norm = MuseSparkInputScaledRMSNorm(
        num_features=self.expert_input_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_expert_proj = linears.DenseGeneral(
        in_features_shape=self.expert_input_dim,
        out_features_shape=self.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        use_bias=False,
        rngs=rngs,
    )

  def _compute_load_balance_loss(self, top_k_index: jax.Array, routing_weights: jax.Array) -> jax.Array:
    """Computes sequence-wise auxiliary load-balance loss matching `RoutedMoE`."""
    if moe.uses_megatron_seq_aux_loss(self.config):
      routing_weights, top_k_index = moe.megatron_aux_loss_inputs(routing_weights, self.num_experts_per_tok)
    expert_mask = jax.nn.one_hot(top_k_index, num_classes=self.num_experts, dtype=jnp.int32)
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / self.num_experts_per_tok
    density_prob = jnp.mean(routing_weights.astype(jnp.float32), axis=1)
    return jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight

  def __call__(
      self,
      inputs: jax.Array,
      forced_routed_experts: jax.Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> tuple[jax.Array, jax.Array | None]:
    if self.mesh.shape.get("expert", 1) > 1:
      raise ValueError(
          "MuseSparkMoE does not support expert parallelism (`ici_expert_parallelism > 1`); "
          "shard experts with FSDP and tensor parallelism instead."
      )

    k = self.num_experts_per_tok
    d_in = self.expert_input_dim

    # 1. Router on full-width hidden_states
    gate_dtype = jnp.float32 if self.float32_gate_logits else inputs.dtype
    router_logits = jnp.einsum(
        "bsd,de->bse",
        inputs.astype(gate_dtype),
        self.gate.kernel[...].astype(gate_dtype),
        precision=jax.lax.Precision.HIGHEST if gate_dtype == jnp.float32 else self.config.matmul_precision,
    ).astype(jnp.float32)
    router_logits = router_logits * self.output_multiplier
    if self.routed_score_func == "sigmoid":
      routing_weights = jax.nn.sigmoid(router_logits)
    elif self.routed_score_func == "softmax":
      routing_weights = jax.nn.softmax(router_logits, axis=-1)
    else:
      routing_weights = linears._convert_to_activation_function(  # pylint: disable=protected-access
          self.routed_score_func
      )(router_logits)

    scores_for_choice = routing_weights
    if self.routed_bias and self.gate.bias is not None:
      scores_for_choice = scores_for_choice + self.gate.bias[...].astype(jnp.float32)

    _, top_k_index = jax.lax.top_k(scores_for_choice, k=k)
    valid_token_mask = None
    if forced_routed_experts is not None:
      top_k_index = forced_routed_experts.astype(top_k_index.dtype)
      valid_token_mask = moe.valid_expert_mask(top_k_index, self.num_experts)
    if self.config.return_routed_experts:
      self.selected_experts = nnx.Intermediate(jnp.reshape(top_k_index, (-1, k)))

    load_balance_loss = None
    if self.config.load_balance_loss_weight > 0.0:
      load_balance_loss = self._compute_load_balance_loss(top_k_index, routing_weights)

    gather_indices = jnp.where(valid_token_mask, top_k_index, 0) if valid_token_mask is not None else top_k_index
    top_k_weights = jnp.take_along_axis(routing_weights, gather_indices, axis=-1)
    if valid_token_mask is not None:
      top_k_weights = top_k_weights * valid_token_mask
    if self.norm_topk_prob:
      top_k_weights = top_k_weights / (jnp.sum(top_k_weights, axis=-1, keepdims=True) + 1e-15)
    top_k_weights = jnp.asarray(top_k_weights * self.routed_scaling_factor, jnp.float32)

    # 2. Width bottleneck before experts
    x = self.pre_expert_proj(inputs)
    x = self.pre_expert_norm(x)

    # 3. Resolve sharding specs from logical axis rules and evaluate routed experts via dropless ragged_dot
    if model_mode == MODEL_MODE_PREFILL:
      batch_axis, length_axis = "activation_batch", "prefill_activation_norm_length"
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      batch_axis, length_axis = DECODE_BATCH, DECODE_LENGTH
    else:
      batch_axis, length_axis = "activation_batch", "activation_norm_length"

    rules = get_logical_axis_rules() or self.config.logical_axis_rules
    x_raw_pspec = logical_to_mesh_axes((batch_axis, length_axis, None), mesh=self.mesh, rules=rules)
    x_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        P(x_raw_pspec[0], x_raw_pspec[1], None),
        (inputs.shape[0], inputs.shape[1], self.expert_input_dim),
        self.mesh,
        allow_remove_axes=True,
    )
    wi_raw_pspec = logical_to_mesh_axes(("exp", "embed_moe", "mlp_moe"), mesh=self.mesh, rules=rules)
    wi_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        wi_raw_pspec,
        (self.num_experts, self.expert_input_dim, self.moe_mlp_dim),
        self.mesh,
        allow_remove_axes=True,
    )
    wo_pspec = P(wi_pspec[0], wi_pspec[2], wi_pspec[1])
    fsdp_axes = mesh_axes_for_dim(wi_pspec[1])
    tp_axes = mesh_axes_for_dim(wi_pspec[2])
    fsdp_axis_arg = fsdp_axes[0] if len(fsdp_axes) == 1 else (fsdp_axes or None)
    tp_axis_arg = tp_axes[0] if len(tp_axes) == 1 else (tp_axes or None)
    scale_pspec = P(None)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(x_pspec, x_pspec, x_pspec, wi_pspec, wi_pspec, wo_pspec, scale_pspec),
        out_specs=x_pspec,
        check_vma=False,
    )
    def _local_moe_forward(x_local, idx_local, weights_local, wi_0_val, wi_1_val, wo_val, post_norm_scale):
      local_bsz, local_seq, _ = x_local.shape
      local_tokens = local_bsz * local_seq
      x_flat = jnp.reshape(x_local, (local_tokens, d_in))
      flat_expert_ids = jnp.reshape(idx_local, (local_tokens * k,))
      if valid_token_mask is not None:
        valid_flat = moe.valid_expert_mask(flat_expert_ids, self.num_experts)
        dummy_ids = jnp.arange(local_tokens * k, dtype=flat_expert_ids.dtype) % self.num_experts
        flat_expert_ids = jnp.where(valid_flat, flat_expert_ids, dummy_ids)
      flat_token_ids = jnp.repeat(jnp.arange(local_tokens, dtype=jnp.int32), k)
      flat_weights = jnp.reshape(weights_local, (local_tokens * k,))

      perm = jnp.argsort(flat_expert_ids, stable=True)
      sorted_expert_ids = flat_expert_ids[perm]
      sorted_token_ids = flat_token_ids[perm]
      sorted_weights = flat_weights[perm]
      x_sorted = x_flat[sorted_token_ids]
      group_sizes = jnp.bincount(sorted_expert_ids, length=self.num_experts).astype(jnp.int32)

      def _gather_expert_weights(rhs, lhs_dtype, gather_axis: int):
        """Casts the local expert-weight shard and all-gathers its FSDP-sharded embed axis."""
        rhs = rhs.astype(lhs_dtype)
        if fsdp_axis_arg is not None:
          rhs = jax.lax.all_gather(rhs, axis_name=fsdp_axis_arg, axis=gather_axis, tiled=True)
        return rhs

      def _ragged_dot(lhs, rhs_sharded, gather_axis: int):
        """ragged_dot whose backward re-gathers the expert weights instead of keeping the gathered copy alive."""

        @jax.custom_vjp
        def _dot_fn(lhs_in, rhs_in, grp_in):
          _, grp_in = jax.lax.optimization_barrier((lhs_in, grp_in))
          rhs_full = _gather_expert_weights(rhs_in, lhs_in.dtype, gather_axis)
          return jax.lax.ragged_dot(lhs_in, rhs_full, grp_in, precision=self.config.matmul_precision)

        def _dot_fwd(lhs_in, rhs_in, grp_in):
          return _dot_fn(lhs_in, rhs_in, grp_in), (lhs_in, rhs_in, grp_in)

        def _dot_bwd(res, g):
          lhs_in, rhs_in, grp_in = res
          _, grp_in = jax.lax.optimization_barrier((g, grp_in))
          _, vjp_fn = jax.vjp(
              lambda l, r: jax.lax.ragged_dot(
                  l,
                  _gather_expert_weights(r, l.dtype, gather_axis),
                  grp_in,
                  precision=self.config.matmul_precision,
              ),
              lhs_in,
              rhs_in,
          )
          d_lhs, d_rhs = vjp_fn(g)
          return d_lhs, d_rhs, None

        _dot_fn.defvjp(_dot_fwd, _dot_bwd)
        return _dot_fn(lhs, rhs_sharded, group_sizes)

      # Makes the wi_1 matmul (and, in the backward pass, its cotangents) wait for the wi_0 matmul, so XLA
      # cannot gather both expert weights up front; only one gathered expert weight is live at a time.
      @jax.custom_vjp
      def _gate_barrier(g_out, x_in):
        return jax.lax.optimization_barrier((g_out, x_in))

      def _gate_barrier_fwd(g_out, x_in):
        return _gate_barrier(g_out, x_in), None

      def _gate_barrier_bwd(_, grads):
        return jax.lax.optimization_barrier(grads)

      _gate_barrier.defvjp(_gate_barrier_fwd, _gate_barrier_bwd)

      gate_out = _ragged_dot(x_sorted, wi_0_val, 1)
      gate_out, x_for_wi1 = _gate_barrier(gate_out, x_sorted)
      up_out = _ragged_dot(x_for_wi1, wi_1_val, 1)
      act = jax.nn.silu(gate_out.astype(jnp.float32)).astype(x_local.dtype) * up_out
      expert_out = _ragged_dot(act, wo_val, 2)
      if tp_axis_arg is not None:
        expert_out = jax.lax.psum(expert_out, axis_name=tp_axis_arg)

      # Per-expert-output input-scaled RMSNorm before weighting by fp32 top_k_weights
      expert_out = _input_scaled_rms_norm(expert_out, post_norm_scale, self.post_expert_norm.epsilon)

      weighted_sorted = (expert_out.astype(jnp.float32) * sorted_weights[:, None]).astype(x_local.dtype)

      # Group back by token_id while preserving ascending expert_idx order (matches HF loop order)
      token_group_perm = jnp.argsort(sorted_token_ids, stable=True)
      by_token = jnp.reshape(weighted_sorted[token_group_perm], (local_bsz, local_seq, k, d_in))
      return jnp.sum(by_token, axis=2)

    moe_fn = _local_moe_forward
    if self.config.remat_policy == "full" and self.config.scan_layers:
      # A scanned block rematerializes all of its decoder layers as one unit; checkpointing the routed
      # experts separately keeps only one layer's [tokens * k, ...] expert intermediates live at a time.
      moe_fn = jax.checkpoint(moe_fn, policy=jax.checkpoint_policies.nothing_saveable)
    combined = moe_fn(
        x,
        top_k_index,
        top_k_weights,
        self.wi_0[...],
        self.wi_1[...],
        self.wo[...],
        self.post_expert_norm.scale[...],
    )

    # 4. Width bottleneck up-projection back to emb_dim
    return self.post_expert_proj(combined), load_balance_loss


# ------------------------------------------------------------------------------
# Decoder Layer & Scannable Block
# ------------------------------------------------------------------------------


class MuseSparkDecoderLayer(nnx.Module):
  """Single Muse-Spark transformer decoder layer (`MuseSparkTextDecoderLayer`)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: quantizations.AqtQuantization | None = None,
      layer_idx: int = 0,
      is_full_attention_override: bool | None = None,
      is_moe_override: bool | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.is_full_attention = (
        is_full_attention_layer(layer_idx, config)
        if is_full_attention_override is None
        else bool(is_full_attention_override)
    )
    self.is_moe = is_moe_layer(layer_idx, config) if is_moe_override is None else bool(is_moe_override)

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    norm_weight_dtype = get_weight_dtype(config, "norm")

    self.input_layernorm = MuseSparkCenteredRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.self_attention = MuseSparkAttention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        mesh=mesh,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        max_prefill_predict_length=config.max_prefill_predict_length,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        is_full_attention=self.is_full_attention,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.post_attention_layernorm = MuseSparkRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )
    self.post_attention_residual_gate = MuseSparkResidualGate(
        num_features=config.emb_dim,
        temperature=config.muse_spark_residual_gate_temperature,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    self.pre_feedforward_layernorm = MuseSparkCenteredRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    if self.is_moe:
      self.moe_block = MuseSparkMoE(
          config=config,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = linears.MlpBlock(
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
    self.post_feedforward_layernorm = MuseSparkInputScaledRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_feedforward_residual_gate = MuseSparkResidualGate(
        num_features=config.emb_dim,
        temperature=config.muse_spark_residual_gate_temperature,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    self.activation_axis_names = self._activation_axis_names_for_mode(model_mode)

  @staticmethod
  def _activation_axis_names_for_mode(model_mode: str) -> tuple[str, str, str]:
    if model_mode == MODEL_MODE_PREFILL:
      return ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    if model_mode == MODEL_MODE_AUTOREGRESSIVE:
      return (DECODE_BATCH, DECODE_LENGTH, "activation_embed")
    return ("activation_batch", "activation_norm_length", "activation_embed")

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: jax.Array | None,
      decoder_positions: jax.Array | None,
      deterministic: bool,
      model_mode: str,
      previous_chunk: Any = None,
      slot: int | None = None,
      kv_cache: jax.Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      forced_routed_experts: jax.Array | None = None,
      **kwargs,
  ):
    del kwargs
    activation_axis_names = self._activation_axis_names_for_mode(model_mode)
    inputs = sharding.maybe_shard_with_logical(
        inputs,
        activation_axis_names,
        self.mesh,
        self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # 1. Self-Attention sub-block
    residual = inputs
    hidden_states = self.input_layernorm(inputs)
    attn_out, kv_cache = self.self_attention(
        hidden_states,
        hidden_states,
        decoder_positions,
        decoder_segment_ids,
        model_mode=model_mode,
        deterministic=deterministic,
        previous_chunk=previous_chunk,
        slot=slot,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attn_out = self.post_attention_layernorm(attn_out)
    hidden_states = self.post_attention_residual_gate(residual, attn_out)

    # 2. Feedforward / MoE sub-block
    residual = hidden_states
    ffn_in = self.pre_feedforward_layernorm(hidden_states)
    load_balance_loss = None
    if self.is_moe:
      ffn_out, load_balance_loss = self.moe_block(
          ffn_in,
          forced_routed_experts=forced_routed_experts,
          model_mode=model_mode,
      )
    else:
      ffn_out = self.mlp(ffn_in, deterministic=deterministic)
    ffn_out = self.post_feedforward_layernorm(ffn_out)
    output = self.post_feedforward_residual_gate(residual, ffn_out)

    output = sharding.maybe_shard_with_logical(
        output,
        activation_axis_names,
        self.mesh,
        self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    return output, kv_cache


class MuseSparkScannableBlock(nnx.Module):
  """Scannable inhomogeneous cycle of `inhomogeneous_layer_cycle_interval` (4) Muse-Spark MoE layers.

  Within each cycle:
  - Sub-layers `0 .. cycle - 2` use sliding-window attention (`sliding_window_size=2048`) + RoPE
    (`is_full_attention=False`).
  - Sub-layer `cycle - 1` uses global causal attention + NoPE (`is_full_attention=True`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: quantizations.AqtQuantization | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.num_of_layers = int(config.inhomogeneous_layer_cycle_interval)
    for sub_id in range(self.num_of_layers):
      layer = MuseSparkDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          quant=quant,
          layer_idx=sub_id,
          is_full_attention_override=(sub_id == self.num_of_layers - 1),
          is_moe_override=True,
          rngs=rngs,
      )
      setattr(self, f"layers_{sub_id}", layer)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: jax.Array | None,
      decoder_positions: jax.Array | None,
      deterministic: bool,
      model_mode: str,
      previous_chunk: Any = None,
      slot: int | None = None,
      kv_cache: Any = None,
      attention_metadata: Any = None,
      forced_routed_experts: jax.Array | None = None,
      **kwargs,
  ):
    del kwargs
    y = inputs
    updated_kvs = []
    for sub_id in range(self.num_of_layers):
      sub_forced = forced_routed_experts[sub_id] if forced_routed_experts is not None else None
      sub_kv = kv_cache[sub_id] if kv_cache is not None else None
      sub_attn_meta = attention_metadata[sub_id] if isinstance(attention_metadata, tuple) else attention_metadata
      layer = getattr(self, f"layers_{sub_id}")
      y, new_kv = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=sub_kv,
          attention_metadata=sub_attn_meta,
          forced_routed_experts=sub_forced,
      )
      if kv_cache is not None:
        updated_kvs.append(new_kv)
    return y, (tuple(updated_kvs) if kv_cache is not None else None)
