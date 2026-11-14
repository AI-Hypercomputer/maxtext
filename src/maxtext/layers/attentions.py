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

"""Attentions Layers."""

import dataclasses
import functools
import math
import os
from typing import Any, Iterable, Optional, Tuple, Union, cast

from jax.ad_checkpoint import checkpoint_name
from jax.experimental.layout import Layout, with_layout_constraint
from jax.sharding import Mesh, NamedSharding
import jax
import jax.numpy as jnp

from flax import nnx
import qwix
from qwix import pallas as qpl

from maxtext.common.common_types import (
    DecoderBlockType,
    BATCH_ATTN,
    HEAD,
    PREFILL_LENGTH,
    D_KV,
    AxisNames,
    AxisIdxes,
    ATTN_LENGTH,
    ATTN_INPUT_LENGTH,
    DType,
    Config,
    Array,
    DECODE_LENGTH,
    DECODE_BATCH,
    PREFILL_KV_BATCH,
    KV_HEAD,
    KV_HEAD_DIM,
    KV_BATCH,
    ATTN_EMBED,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_TRAIN,
    MODEL_MODE_PREFILL,
    AttentionType,
    ShardMode,
)
from maxtext.kernels import fused_qkv
from maxtext.layers import quantizations
from maxtext.layers.attention_op import (
    AttentionOp,
    _resolve_attention_type,
    resolve_context_parallel_strategy,
)
from maxtext.layers.embeddings import (
    LLaMARotaryEmbedding,
    LlamaVisionRotaryEmbedding,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeVisionRotaryEmbedding,
    RotaryEmbedding,
    YarnRotaryEmbedding,
    PartialRotaryEmbedding,
    Gemma4PartialRotaryEmbedding,
)
from maxtext.layers.initializers import nd_dense_init, NdInitializer, default_bias_init
from maxtext.layers.linears import DenseGeneral, canonicalize_tuple, normalize_axes
from maxtext.layers.normalizations import (
    GlobalRMSNorm,
    Qwen3NextRMSNorm,
    RMSNorm,
    get_deferred_norm_psum_axes,
)
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.inference import kvcache
from maxtext.inference.kvcache import KVQuant
from maxtext.utils.sharding import maybe_shard_with_logical, create_sharding, logical_to_mesh_axes

# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


def _row_major(x):
  if not isinstance(x, jax.core.Tracer):
    return x
  return with_layout_constraint(x, Layout(major_to_minor=tuple(range(x.ndim))))


@jax.custom_vjp
def _pin_row_major_with_cotangent(x):
  """Identity that constrains `x` and its cotangent to the default (row-major) XLA layout.

  Under Ulysses the attention backward wants the hidden-state gradient tokens-minor, for the
  [B, S, H, D] -> [B, H, S, D] transpose ahead of the inverse all-to-all. Without a pin XLA carries that
  layout out through the residual add into the neighbouring MoE layers, where it transposes the
  expert-parallel-gathered token buffer (8x the local tokens) in front of the token reduce-scatter. Pinning
  the attention block's input and output keeps the relayout on the attention-local tensors.
  """
  return _row_major(x)


def _pin_row_major_fwd(x):
  return _row_major(x), None


def _pin_row_major_bwd(_, g):
  return (_row_major(g),)


_pin_row_major_with_cotangent.defvjp(_pin_row_major_fwd, _pin_row_major_bwd)


@dataclasses.dataclass(repr=False)
class L2Norm(nnx.Module):
  """
  Implementation of L2Norm in JAX.

  Args:
    eps: float, epsilon used for numerical stability (default value should be ok for most cases).
  """

  eps: float = 1e-6
  rngs: nnx.Rngs = None  # Not used in L2Norm but passed in by nnx.bridge.to_linen

  def __call__(self, x):
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


class Attention(nnx.Module):
  """Attention Module.

  This module implements multi-headed attention as described in the
  original Transformer paper. It projects the inputs into query, key, and
  value vectors, applies the attention mechanism, and projects the results to
  an output vector.

  Attributes:
    config: The model configuration.
    num_query_heads: Number of query attention heads.
    num_kv_heads: Number of key-value attention heads.
    head_dim: The dimension of each attention head.
    max_target_length: Maximum sequence length.
    mesh: The device mesh.
    attention_kernel: The attention kernel to use (e.g., 'dot_product', 'flash').
    inputs_q_shape: Query inputs shape for initialization, required by NNX.
    inputs_kv_shape: Key/value inputs shape for initialization, required by NNX.
    dtype: The data type for computation.
    weight_dtype: The data type for weights.
    max_prefill_predict_length: Maximum length for prefill.
    dropout_rate: The dropout rate.
    kernel_init: Initializer for the kernel of the dense layers.
    float32_qk_product: If True, compute query-key product in float32.
    float32_logits: If True, cast logits to float32 before softmax.
    quant: Quantization configuration.
    kv_quant: KV cache quantization configuration.
    attention_type: The type of attention (e.g., 'global', 'local_sliding').
    attn_logits_soft_cap: Soft cap for attention logits.
    ... and other configuration parameters.
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
      inputs_q_shape: Tuple,
      inputs_kv_shape: Tuple,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      float32_qk_product: bool = False,  # computes logits in float32 for stability.
      float32_logits: bool = False,  # cast logits in float32 for stability.
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.GLOBAL,
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
      share_kv_projections: bool = False,  # If true, Key and Value use the same projection
      # Temperature tuning parameters used for Llama4
      temperature_tuning: bool = False,
      temperature_tuning_scale: float = 0.1,
      temperature_tuning_floor_scale: float = 8192.0,
      # Shard the query activation as the same as the key and value.
      # TODO: Find a better sharding axis name.
      # TODO: Further break down the Training and Inference axes for the q, k, v.
      prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      query_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM),
      key_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM),
      value_axis_names: AxisNames = (KV_BATCH, ATTN_LENGTH, KV_HEAD, KV_HEAD_DIM),
      input_axis_names: AxisNames = (BATCH_ATTN, ATTN_INPUT_LENGTH, ATTN_EMBED),
      out_axis_names: AxisNames = (BATCH_ATTN, ATTN_LENGTH, HEAD, D_KV),
      prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, ATTN_EMBED),
      decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, ATTN_EMBED),
      prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
      decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
      prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      reshape_q: bool = False,
      is_nope_layer: bool = False,
      is_vision: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      base_kv_cache: bool = True,
      use_mrope: bool = False,
      mrope_section: tuple[int, int, int] | None = None,
      name: str | None = None,
      rope_type: str | None = None,
      use_v_norm: bool = False,
      rope_max_timescale: float | None = None,
      partial_rotary_factor: float | None = None,
      share_kv_layer: bool = False,
      defer_qk_norm_grad_allreduce: bool = False,
      rngs: nnx.Rngs | None = None,
  ):
    """Initializes the Attention module.

    Attributes:
      config: The model configuration.
      num_query_heads: Number of query attention heads.
      num_kv_heads: Number of key-value attention heads.
      head_dim: The dimension of each attention head.
      max_target_length: Maximum sequence length.
      mesh: The device mesh.
      attention_kernel: The attention kernel to use (e.g., 'dot_product', 'flash').
      inputs_q_shape: Query inputs shape for initialization, required by NNX.
      inputs_kv_shape: Key/value inputs shape for initialization, required by NNX.
      dtype: The data type for computation.
      weight_dtype: The data type for weights.
      max_prefill_predict_length: Maximum length for prefill.
      dropout_rate: The dropout rate.
      kernel_init: Initializer for the kernel of the dense layers.
      float32_qk_product: If True, compute query-key product in float32.
      float32_logits: If True, cast logits to float32 before softmax.
      quant: Quantization configuration.
      kv_quant: KV cache quantization configuration.
      attention_type: The type of attention (e.g., 'global', 'local_sliding').
      attn_logits_soft_cap: Soft cap for attention logits.
      sliding_window_size: The size of the sliding window for local attention.
      use_ragged_attention: Whether to use ragged attention for decoding.
      ragged_block_size: The block size for ragged attention.
      use_qk_norm: Whether to apply normalization to query and key.
      query_pre_attn_scalar: Scalar to apply to query before attention.
      use_bias_in_projections: Whether to use bias in Q, K, V, and output projections.
      share_kv_projections: If true, Key and Value use the same projection.
      temperature_tuning: Whether to use temperature tuning for attention.
      temperature_tuning_scale: The scale for temperature tuning.
      temperature_tuning_floor_scale: The floor scale for temperature tuning.
      ... other configuration parameters.
      is_nope_layer: Whether this is a "NoPE" (No Position-Embedding) layer.
      is_vision: Whether this is a vision attention layer.
      model_mode: The model's operational mode (e.g., 'train', 'prefill').
      base_kv_cache: Whether to use base (non-MLA) kv cache, if KVCache is used
      rope_type: Optional override for the RoPE type (e.g., 'default', 'yarn').
          If provided, this takes precedence over `config.rope_type`.
      use_v_norm: Whether to apply normalization to value.
      rope_max_timescale: The maximum timescale for RoPE.
      partial_rotary_factor: The factor for partial rotary embedding.
      share_kv_layer: If True, this layer reuses K / V from an earlier (donor) layer of the same
          attention type; k_proj / v_proj / k_norm / v_norm are not created and RoPE-on-K is
          skipped. The caller must pass `shared_key` / `shared_value` to `__call__`.
      rngs: RNG state for initialization, passed by the nnx.to_linen wrapper.
    """

    self.config = config
    self.num_query_heads = num_query_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.max_target_length = max_target_length
    self.mesh = mesh
    self.attention_kernel = attention_kernel
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.max_prefill_predict_length = max_prefill_predict_length
    self.dropout_rate = dropout_rate
    self.kernel_init = kernel_init
    self.float32_qk_product = float32_qk_product
    self.float32_logits = float32_logits
    self.quant = quant
    self.kv_quant = kv_quant
    self.attention_type = _resolve_attention_type(self.config, attention_type)
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.sliding_window_size = sliding_window_size
    self.use_ragged_attention = use_ragged_attention
    self.ragged_block_size = ragged_block_size
    self.use_qk_norm = use_qk_norm
    self.defer_qk_norm_grad_allreduce = defer_qk_norm_grad_allreduce
    self.query_pre_attn_scalar = query_pre_attn_scalar
    self.use_bias_in_projections = use_bias_in_projections
    self.share_kv_projections = share_kv_projections
    self.temperature_tuning = temperature_tuning
    self.temperature_tuning_scale = temperature_tuning_scale
    self.temperature_tuning_floor_scale = temperature_tuning_floor_scale
    self.prefill_query_axis_names = prefill_query_axis_names
    self.prefill_key_axis_names = prefill_key_axis_names
    self.prefill_value_axis_names = prefill_value_axis_names
    self.query_axis_names = query_axis_names
    self.key_axis_names = key_axis_names
    self.value_axis_names = value_axis_names
    self.input_axis_names = input_axis_names
    self.out_axis_names = out_axis_names
    self.prefill_input_axis_names = prefill_input_axis_names
    self.decode_input_axis_names = decode_input_axis_names
    self.prefill_out_axis_names = prefill_out_axis_names
    self.decode_out_axis_names = decode_out_axis_names
    self.prefill_cache_axis_order = prefill_cache_axis_order
    self.ar_cache_axis_order = ar_cache_axis_order
    self.compute_axis_order = compute_axis_order
    self.reshape_q = reshape_q
    self.is_nope_layer = is_nope_layer
    self.is_vision = is_vision
    self.model_mode = model_mode
    self.use_mrope = use_mrope
    self.mrope_section = mrope_section
    self.rngs = rngs
    # Use the rope type specified in the arguments if provided, otherwise fall back to the one in the config.
    self.rope_type = (rope_type or self.config.rope_type).lower()
    self.use_v_norm = use_v_norm
    self.rope_max_timescale = rope_max_timescale if rope_max_timescale is not None else self.config.rope_max_timescale
    self.partial_rotary_factor = partial_rotary_factor
    self.share_kv_layer = share_kv_layer

    self.is_qwen2 = self.config.decoder_block == DecoderBlockType.QWEN2
    self.is_qwen3_hybrid = (
        self.config.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5) and not self.is_vision
    )

    # Module attribute names must match names previously passed to Linen for checkpointing
    self.KVCache_0 = (
        self.init_kv_caches(inputs_kv_shape=inputs_kv_shape)
        if self.model_mode != MODEL_MODE_TRAIN
        and base_kv_cache
        and config.attention not in ("vllm_rpa", "vllm_batched_rpa")
        else None
    )

    self.rotary_embedding = self.init_rotary_embedding()

    self.attention_op = AttentionOp(
        config=self.config,
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        float32_qk_product=self.float32_qk_product,
        float32_logits=self.float32_logits,
        quant=self.quant,
        kv_quant=self.kv_quant,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self._attention_op_num_kv_heads(),
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        compute_axis_order=self.compute_axis_order,
        reshape_q=self.reshape_q,
        attention_type=self.attention_type,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        chunk_attn_window_size=self.config.chunk_attn_window_size,
        use_ragged_attention=self.use_ragged_attention,
        ragged_block_size=self.ragged_block_size,
        rngs=self.rngs,
    )

    self._init_projections(inputs_q_shape, inputs_kv_shape)

    if self.config.attention_sink:
      self.sinks = nnx.Param(
          default_bias_init(self.rngs.params(), (self.config.num_query_heads,), self.weight_dtype),
          out_sharding=(None,),
      )
    else:
      self.sinks = None

    is_llama4_decoder_block = self.config.decoder_block == DecoderBlockType.LLAMA4

    if self.use_qk_norm and not is_llama4_decoder_block:
      # Check if this is Olmo3, which uses a unique "Global" QK Norm strategy.
      # GlobalRMSNorm flattens (Heads, Dim) to normalize across the entire hidden state.
      use_global_qk_norm = self.config.model_name.startswith("olmo3")
      qk_norm_cls = GlobalRMSNorm if use_global_qk_norm else RMSNorm

      # For RMSNorm use `head_dim` (per-head normalization), while for GlobalRMSNorm use `num_heads * head_dim` (global normalization).
      q_features = (self.num_query_heads * self.head_dim) if use_global_qk_norm else self.head_dim
      k_features = (self.num_kv_heads * self.head_dim) if use_global_qk_norm else self.head_dim

      with_scale = getattr(self.config, "qk_norm_with_scale", True)

      self.query_norm = qk_norm_cls(
          num_features=q_features,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          shard_mode=self.config.shard_mode,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          with_scale=with_scale,
          rngs=self.rngs,
      )
      if self.share_kv_layer:
        self.key_norm = None
      else:
        self.key_norm = qk_norm_cls(
            num_features=k_features,
            dtype=self.config.dtype,
            weight_dtype=self.config.weight_dtype,
            shard_mode=self.config.shard_mode,
            epsilon=self.config.normalization_layer_epsilon,
            kernel_axes=("norm",),
            with_scale=with_scale,
            rngs=self.rngs,
        )
    elif self.is_qwen3_hybrid:
      self.query_norm = Qwen3NextRMSNorm(
          num_features=self.config.head_dim,
          epsilon=self.config.normalization_layer_epsilon,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
      self.key_norm = Qwen3NextRMSNorm(
          num_features=self.config.head_dim,
          epsilon=self.config.normalization_layer_epsilon,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
    else:
      self.query_norm = None
      self.key_norm = None

    if self.use_v_norm and not self.share_kv_layer:
      with_scale = self.config.v_norm_with_scale
      self.value_norm = RMSNorm(
          num_features=self.head_dim,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          shard_mode=self.config.shard_mode,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          with_scale=with_scale,
          rngs=self.rngs,
      )
    else:
      self.value_norm = None

    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
    )

  def _logical_to_mesh_axes(self, logical_name):
    # Pipeline parallelism uses context managers for logical rules instead of the config,
    # so pass None to ensure `logical_to_mesh_axes` defers to using the current Flax context manager
    logical_rules = None if self.config.using_pipeline_parallelism else self.config.logical_axis_rules
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=logical_rules)

  def _attention_op_num_kv_heads(self) -> int:
    return self.num_kv_heads

  def _validate_kv_heads(self) -> None:
    """Validates the number of key/value heads."""
    if self.num_kv_heads == -1:
      raise ValueError("num_kv_heads is not defined.")

    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError("Invalid num_kv_heads for GQA.")

  def _validate_kv_head_sharding(
      self,
      kernel_axes: Tuple[Optional[str], ...],
  ) -> None:
    """Validates that the key/value head dimension can be sharded evenly.

    Attention heads are atomic under tensor parallelism.  The `kv_heads`
    logical axis of the key/value projection is split across whichever mesh
    axes `logical_axis_rules` maps it to -- by default `tensor`,
    `tensor_sequence` and `autoregressive` -- so the KV head count has to be
    divisible by their combined size.  Note that `num_kv_heads` is the
    per-layer count, which for models such as Gemma 4 is `global_num_kv_heads`
    on global attention layers and `num_kv_heads` elsewhere.

    Ulysses context parallelism shards the KV heads a second time, over the
    context axis, using a runtime all-to-all rather than the weight's logical
    axes, so that axis is folded in explicitly.  `types.py` separately checks
    `num_kv_heads` against the context-parallel size on its own; checking the
    product here is what catches a head count that clears each factor
    individually but not both together.  The one exception is a context axis
    that is a multiple of the KV heads left on each device by the other axes:
    Ulysses then repeats each KV head across the context ranks whose query heads
    use it (`ulysses_attention.repeat_kv_heads_for_ulysses`) instead of splitting
    the KV heads, so the context axis places no constraint on them.

    Without this check an over-sharded mesh fails much later, either with an
    opaque XLA divisibility error or by silently leaving the projection
    unsharded until `assert_params_sufficiently_sharded` trips.

    Args:
      kernel_axes: Logical axis names of the key/value projection kernel.

    Raises:
      ValueError: If the KV heads cannot be split evenly across the mesh.
    """
    if "kv_heads" not in kernel_axes:
      # The projection is replicated, so the head count is unconstrained.
      return

    # Size-one mesh axes are already dropped, so this is the exact shard count.
    # An empty result means no logical rule shards the heads; that is not an
    # early exit, because Ulysses below can still shard them over the context
    # axis. An unsharded head dimension just leaves `kv_parallelism` at 1.
    kv_heads_index = kernel_axes.index("kv_heads")
    kv_head_axes = self._logical_to_mesh_axes(kernel_axes)[kv_heads_index]
    if kv_head_axes is None:
      kv_head_axes = ()
    elif isinstance(kv_head_axes, str):
      kv_head_axes = (kv_head_axes,)
    kv_head_axes = list(kv_head_axes)

    def shards(axes):
      return math.prod(self.mesh.shape.get(axis, 1) for axis in axes)

    # Ulysses exchanges sequence ownership for head ownership through an
    # all-to-all, so the context axis shards KV heads too even though no
    # logical rule says so -- unless it is a multiple of the KV heads left per
    # device, in which case those heads are repeated rather than split.
    if resolve_context_parallel_strategy(self.config, self.attention_type) == "ulysses":
      ulysses_axis = self.config.context_sharding
      if ulysses_axis not in kv_head_axes:
        local_kv_heads, remainder = divmod(self.num_kv_heads, shards(kv_head_axes))
        replicated = remainder == 0 and self.mesh.shape.get(ulysses_axis, 1) % local_kv_heads == 0
        if not replicated:
          kv_head_axes.append(ulysses_axis)

    kv_parallelism = shards(kv_head_axes)

    if self.num_kv_heads % kv_parallelism != 0:
      raise ValueError(
          f"num_kv_heads ({self.num_kv_heads}) for {self.attention_type}"
          f" attention layers must be divisible by {kv_parallelism}, the"
          f" combined size of the mesh axes {kv_head_axes} that shard KV heads."
          " Attention heads are atomic under tensor parallelism and cannot be"
          " split across more shards than there are heads. Either reduce the"
          " parallelism on those axes, raise the KV head count"
          " (`base_num_kv_heads`, or `global_num_kv_heads` for global attention"
          " layers), or move the parallelism onto an axis that does not shard"
          " KV heads (e.g. fsdp)."
      )

  def _init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple) -> None:
    """Initializes the query, key, value, and output projections."""
    if self.config.fused_qkv:
      self.qkv_proj = self.init_qkv_w(inputs_shape=inputs_q_shape)
    else:
      self.query = self.init_query_w(inputs_q_shape=inputs_q_shape)
      if not self.share_kv_layer:
        self.key = self.init_kv_w(inputs_kv_shape=inputs_kv_shape)
        if not self.share_kv_projections:
          self.value = self.init_kv_w(inputs_kv_shape=inputs_kv_shape)
    self.out = self.init_out_w(output_dim=inputs_q_shape[-1])

  def init_query_w(self, inputs_q_shape: Tuple) -> nnx.Module:
    """Query projection initialization."""

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    # We disable depth_scaling when using qk_norm or a query_pre_attn_scalar
    # to avoid applying scaling twice.
    if getattr(self.config, "use_qk_norm", False) or (
        self.query_pre_attn_scalar is not None and self.query_pre_attn_scalar != 1.0
    ):
      depth_scaling = 1.0
    else:
      depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

    def query_init(*args):
      # pylint: disable=no-value-for-parameter
      return self.kernel_init(*args) / depth_scaling

    kernel_axes = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("embed_attn", "q_heads", "kv")
    )
    in_features = self.convert_dense_general_inputs_shape(inputs_q_shape)
    out_features = (self.num_query_heads, self.head_dim)

    if self.is_qwen3_hybrid:
      out_features = (self.num_query_heads, self.head_dim * 2)

    block_size = getattr(self.config, "weight_block_size", None)
    return DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        axis=-1,
        kernel_init=query_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        shard_mode=self.config.shard_mode,
        weight_grad_in_kernel_order=self.config.dense_weight_grad_in_kernel_order,
        block_size=block_size,
        rngs=self.rngs,
    )

  def query_projection(self, inputs_q: Array, out_sharding: NamedSharding | None = None) -> Array:
    """Query projection."""

    return self.query(inputs_q, out_sharding=out_sharding)

  def init_kv_w(self, inputs_kv_shape: Tuple) -> nnx.Module:
    """Initializes the key or value projection.

    Args:
      inputs_kv_shape: Key/value inputs shape for initialization.

    Returns:
      A DenseGeneral module that performs the key or value projection.
    """
    self._validate_kv_heads()

    kernel_axes = (
        (None, None, None)
        if self.config.ici_context_autoregressive_parallelism > 1
        else ("embed_attn", "kv_heads", "kv_head_dim")
    )
    self._validate_kv_head_sharding(kernel_axes)

    block_size = getattr(self.config, "weight_block_size", None)
    return DenseGeneral(
        in_features_shape=self.convert_dense_general_inputs_shape(inputs_kv_shape),
        out_features_shape=(self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        shard_mode=self.config.shard_mode,
        weight_grad_in_kernel_order=self.config.dense_weight_grad_in_kernel_order,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        block_size=block_size,
        rngs=self.rngs,
    )

  def kv_projection(self, inputs_kv: Array, proj_name: str, out_sharding: NamedSharding | None = None) -> nnx.Module:
    """Applies the key or value projection.

    Args:
      inputs_kv: The input tensor to project.
      proj_name: The name of the projection ("key" or "value").

    Returns:
      The projected key or value tensor.

    Raises:
      ValueError: If `proj_name` is not one of the supported values
        ("key", "value").

    """
    if proj_name == "key":
      return self.key(inputs_kv, out_sharding=out_sharding)
    elif proj_name == "value":
      return self.value(inputs_kv, out_sharding=out_sharding)
    else:
      raise ValueError(f"proj_name must be 'key' or 'value', but got {proj_name}")

  def init_qkv_w(self, inputs_shape: Tuple) -> nnx.Module:
    """Initializes the a fused QKV projection using only one DenseGeneral module."""
    self._validate_kv_heads()

    return DenseGeneral(
        in_features_shape=self.convert_dense_general_inputs_shape(inputs_shape),
        out_features_shape=(self.num_query_heads + 2 * self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_attn", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        shard_mode=self.config.shard_mode,
        weight_grad_in_kernel_order=self.config.dense_weight_grad_in_kernel_order,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )

  def qkv_projection(self, inputs: Array, proj_name: str, out_sharding: NamedSharding | None = None):
    """Fused QKV projection"""

    qkv_proj = self.qkv_proj(inputs, out_sharding)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")

    # Since fused QKV projection places all heads along the same axis which could be tensor
    # parallel partitioned, we must use shard_map to split into equally partitioned Q, K, V arrays.
    q_bshd = self._logical_to_mesh_axes(self.query_axis_names)
    k_bshd = self._logical_to_mesh_axes(self.key_axis_names)
    v_bshd = self._logical_to_mesh_axes(self.value_axis_names)

    @jax.shard_map(mesh=self.mesh, in_specs=(q_bshd,), out_specs=(q_bshd, k_bshd, v_bshd))
    def split_qkv(qkv_proj: Array) -> tuple[Array, Array, Array]:
      num_local_heads = qkv_proj.shape[2]
      num_query_heads = (num_local_heads * self.num_query_heads) // (self.num_query_heads + 2 * self.num_kv_heads)
      num_kv_heads = (num_local_heads - num_query_heads) // 2

      return tuple(jnp.split(qkv_proj, [num_query_heads, num_query_heads + num_kv_heads], axis=2))

    return split_qkv(qkv_proj)

  @property
  def out_head_dim(self) -> int:
    return self.head_dim

  def init_out_w(self, output_dim: int) -> nnx.Module:
    """out projection"""
    in_features = (self.num_query_heads, self.out_head_dim)
    out_features = output_dim
    out_kernel_axis = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("heads", "kv", "embed_attn")
    )
    axis = (-2, -1)

    if self.is_qwen3_hybrid:
      in_features = self.num_query_heads * self.out_head_dim
      out_kernel_axis = ("mlp", "embed_attn")
      axis = (-1,)

    block_size = getattr(self.config, "weight_block_size", None)
    return DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=out_features,
        axis=axis,
        kernel_init=self.kernel_init,
        kernel_axes=out_kernel_axis,  # trade speed with memory
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        shard_mode=self.config.shard_mode,
        weight_grad_in_kernel_order=self.config.dense_weight_grad_in_kernel_order,
        matmul_precision=self.config.matmul_precision,
        use_bias=False if self.is_qwen2 else self.use_bias_in_projections,
        block_size=block_size,
        rngs=self.rngs,
    )

  @staticmethod
  def _check_fused_qt_rule(rule: qwix.QtRule, feature: str) -> None:
    """Raises for `additional_qt_config` settings that the fused QKV / out projections cannot honor.

    The fused QKV / out projections bypass Qwix (`without_qwix_interception`) and quantize one cotangent for both
    gradient arms, so a weight sparsity rule or a separate weight-gradient (drhs) cotangent calibration would be
    dropped silently. `get_fp8_full_qwix_rule_w_sparsity` always passes the `sparsity_rule` key (None when
    sparsity is off), so only an active rule is rejected.
    """
    extra = rule.additional_qt_config or {}
    sparsity_rule = extra.get("sparsity_rule")
    if sparsity_rule and sparsity_rule.weight_sparsity_n and sparsity_rule.weight_sparsity_m:
      raise NotImplementedError(f"{feature} does not support weight sparsity (weight_sparsity_n / weight_sparsity_m).")
    drhs_method = extra.get("drhs_grad_calibration_method")
    if drhs_method and drhs_method != rule.bwd_calibration_method:
      raise NotImplementedError(
          f"{feature} quantizes one cotangent for both gradient arms and does not support"
          f" drhs_grad_calibration_method={drhs_method!r} (bwd_calibration_method={rule.bwd_calibration_method!r})."
      )

  def _try_fused_out_kernel(self, out: Array, out_sharding: NamedSharding | None, model_mode: str) -> Array | None:
    """Runs FSDP custom-VJP Attention `out` projection with local absmax bwd and early `dw_o` RS."""
    if not getattr(self.config, "use_fused_out_kernel", False) or model_mode != MODEL_MODE_TRAIN:
      return None
    use_bias = False if self.is_qwen2 else self.use_bias_in_projections
    if self.is_qwen3_hybrid or use_bias or out.ndim != 4 or self.out.kernel_scale is not None:
      raise NotImplementedError("use_fused_out_kernel requires 4D [B, S, H, D] Attention out without bias or ServeFp8.")

    def _param_val(param):
      val = param.get_value() if hasattr(param, "get_value") else getattr(param, "value", param)
      return val.value if hasattr(val, "value") else val

    def _axes_tuple(entry) -> tuple[str, ...]:
      if entry is None:
        return ()
      raw = (entry,) if isinstance(entry, str) else tuple(entry)
      return tuple(ax for ax in raw if self.mesh.shape.get(ax, 1) > 1)

    def _pad_spec(spec, rank: int) -> jax.sharding.PartitionSpec:
      parts = tuple(spec) + (None,) * max(0, rank - len(spec))
      return jax.sharding.PartitionSpec(*parts[:rank])

    ctx_pspec = _pad_spec(self._logical_to_mesh_axes(self.out_axis_names), 4)
    wo_pspec = _pad_spec(self._logical_to_mesh_axes(self.out.kernel_axes), 3)
    y_pspec = (
        _pad_spec(out_sharding.spec, 3)
        if out_sharding is not None
        else _pad_spec(self._logical_to_mesh_axes(self.input_axis_names), 3)
    )
    if (
        ctx_pspec[2:] != (None, None)
        or wo_pspec[:2] != (None, None)
        or y_pspec[2] is not None
        or y_pspec[:2] != ctx_pspec[:2]
    ):
      raise NotImplementedError(
          f"use_fused_out_kernel requires TP=1 on heads/embed; got ctx={ctx_pspec}, wo={wo_pspec}, y={y_pspec}."
      )

    wo = _param_val(self.out.kernel)
    weight_gather_axes = _axes_tuple(wo_pspec[2])
    batch_seq_axes = tuple(dict.fromkeys(_axes_tuple(ctx_pspec[0]) + _axes_tuple(ctx_pspec[1])))
    if not set(weight_gather_axes).issubset(set(batch_seq_axes)):
      raise NotImplementedError(
          f"use_fused_out_kernel requires weight shard axes {weight_gather_axes} to shard batch/seq {batch_seq_axes}."
      )
    gather_ways = math.prod(self.mesh.shape[ax] for ax in weight_gather_axes) if weight_gather_axes else 1
    if wo.shape[2] % gather_ways != 0:
      raise NotImplementedError(f"out_features={wo.shape[2]} is not divisible by FSDP gather size {gather_ways}.")
    if self.quant is not None or (self.config.quantization and not self.config.use_qwix_quantization):
      raise NotImplementedError("use_fused_out_kernel only supports Qwix quantized training or unquantized mode.")

    rule = None
    if self.config.quantization and self.config.use_qwix_quantization:
      rule = qpl.get_current_rule("dot_general")
      if rule is not None and not isinstance(rule, qwix.QtRule):
        raise ValueError("Expect a QtRule for quantized training.")
    if rule is not None and (rule.act_qtype is not None or rule.weight_qtype is not None):
      if (
          rule.act_qtype is None
          or rule.weight_qtype is None
          or not jnp.issubdtype(rule.act_qtype, jnp.floating)
          or not jnp.issubdtype(rule.weight_qtype, jnp.floating)
      ):
        raise NotImplementedError(
            "use_fused_out_kernel requires floating-point act/weight qtypes,"
            f" got {rule.act_qtype} and {rule.weight_qtype}."
        )
      self._check_fused_qt_rule(rule, "use_fused_out_kernel")
      for name, method in (("act", rule.act_calibration_method), ("weight", rule.weight_calibration_method)):
        if not quantizations.is_static_calibration(method):
          raise NotImplementedError(f"use_fused_out_kernel needs fixed {name} calibration, got {method!r}.")
      act_qtype = rule.act_qtype
      weight_qtype = rule.weight_qtype
      bwd_qtype = rule.bwd_qtype
      act_scale = float(quantizations.get_static_scale(act_qtype, rule.act_calibration_method))
      weight_scale = float(quantizations.get_static_scale(weight_qtype, rule.weight_calibration_method))
      if bwd_qtype is not None:
        if not jnp.issubdtype(bwd_qtype, jnp.floating):
          raise NotImplementedError(f"use_fused_out_kernel requires floating-point bwd_qtype, got {bwd_qtype}.")
        if quantizations.is_static_calibration(rule.bwd_calibration_method):
          bwd_scale: float | None = float(quantizations.get_static_scale(bwd_qtype, rule.bwd_calibration_method))
        elif rule.bwd_calibration_method in (None, "absmax"):
          bwd_scale = None
        else:
          raise NotImplementedError(f"use_fused_out_kernel unsupported bwd calibration {rule.bwd_calibration_method!r}.")
      else:
        bwd_scale = None
    else:
      act_qtype = None
      weight_qtype = None
      bwd_qtype = None
      act_scale = 1.0
      weight_scale = 1.0
      bwd_scale = None

    # `_run_out` wraps `fused_out_local` in `jax.shard_map(..., check_vma=False)`, whose transpose
    # automatically `psum`s `wo` cotangents across unmentioned mesh axes (e.g. `data` / `context`).
    params = fused_qkv.FusedOutTrainParams(
        compute_dtype=jnp.dtype(self.dtype),
        act_qtype=act_qtype,
        weight_qtype=weight_qtype,
        bwd_qtype=bwd_qtype,
        act_scale=act_scale,
        weight_scale=weight_scale,
        bwd_scale=bwd_scale,
        weight_gather_axes=weight_gather_axes,
        extra_w_psum_axes=(),
        ctx_dtype=out.dtype,
        wo_dtype=wo.dtype,
    )

    @quantizations.without_qwix_interception
    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(ctx_pspec, wo_pspec),
        out_specs=y_pspec,
        check_vma=False,
    )
    def _run_out(ctx_l, wo_l):
      return fused_qkv.fused_out_local(params, ctx_l, wo_l)

    y = _run_out(out, wo)
    if out_sharding is not None:
      if self.config.shard_mode == ShardMode.EXPLICIT:
        y = jax.sharding.reshard(y, out_sharding)
      else:
        y = jax.lax.with_sharding_constraint(y, out_sharding)
    return y

  def out_projection(
      self,
      out: Array,
      out_sharding: NamedSharding | None = None,
      *,
      model_mode: str | None = None,
  ) -> Array:
    """out projection"""
    if model_mode is not None:
      fused_out = self._try_fused_out_kernel(out, out_sharding, model_mode)
      if fused_out is not None:
        return fused_out
    return self.out(out, out_sharding=out_sharding)

  def compute_shared_kv(
      self,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      rope_kwargs: dict | None = None,
  ) -> tuple[Array, Array]:
    """Computes the rotated, normed K / V for this layer.

    Used by KV-donor layers in models with cross-layer KV sharing (e.g. Gemma 4
    small): the donor calls this once, passes the result into its own
    ``__call__`` as ``shared_key`` / ``shared_value`` to avoid double-computing,
    and forwards the same tensors to downstream shared layers.
    """
    if self.share_kv_layer:
      raise ValueError("compute_shared_kv cannot be called on a share_kv_layer=True layer.")
    if self.config.fused_qkv:
      raise ValueError("compute_shared_kv is incompatible with fused_qkv.")
    qkv_sharding = create_sharding(self.mesh, self.input_axis_names)
    key = self.kv_projection(inputs_kv, proj_name="key", out_sharding=qkv_sharding)
    value = (
        key if self.share_kv_projections else self.kv_projection(inputs_kv, proj_name="value", out_sharding=qkv_sharding)
    )
    is_llama4_decoder_block = self.config.decoder_block == DecoderBlockType.LLAMA4
    if (self.use_qk_norm and not is_llama4_decoder_block) or self.is_qwen3_hybrid:
      key = self.key_norm(key)
    if self.use_v_norm:
      value = self.value_norm(value)
    if not self.is_nope_layer:
      key = self.apply_rotary_embedding(key, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)
    if self.use_qk_norm and is_llama4_decoder_block and not self.is_nope_layer:
      key = L2Norm(eps=self.config.normalization_layer_epsilon)(key)
    return key, value

  def convert_dense_general_inputs_shape(
      self,
      inputs_shape: tuple[int, ...] | None = None,
      axis: Union[Iterable[int], int] = -1,
  ) -> Union[Iterable[int], int]:
    axis = canonicalize_tuple(axis)
    return tuple(inputs_shape[ax] for ax in normalize_axes(axis, len(inputs_shape)))

  def init_rotary_embedding(self):
    """Initializes the rotary embeddings, handling different model types.

    Returns:
      The rotary embedding module that will be used in the model.
    """
    if self.config.attention_type == AttentionType.MLA.value:
      # For MLA attention RoPE is applied to only `self.qk_rope_head_dim` portion the heads.
      rope_embedding_dims = self.qk_rope_head_dim
    else:
      rope_embedding_dims = self.head_dim

    rope_type = self.rope_type
    rope_use_scale = self.config.rope_use_scale
    if self.is_vision:
      if self.config.model_name.startswith("qwen3") or self.config.model_name.startswith("weaver"):
        rotary_embedding = Qwen3OmniMoeVisionRotaryEmbedding(
            hidden_size=self.config.hidden_size_for_vit,
            num_attention_heads=self.config.num_attention_heads_for_vit,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
            fprop_dtype=self.dtype,
            rngs=self.rngs,
        )
      elif self.config.model_name.startswith("llama4"):
        rotary_embedding = LlamaVisionRotaryEmbedding(
            image_size=self.config.image_size_for_vit,
            patch_size=self.config.patch_size_for_vit,
            hidden_size=self.config.hidden_size_for_vit,
            num_attention_heads=self.config.num_attention_heads_for_vit,
            rope_theta=self.config.rope_theta_for_vit,
            cast_as_fprop_dtype=True,
            fprop_dtype=self.dtype,
            rngs=self.rngs,
        )
      else:
        raise ValueError(f"Unsupported model type for vision rotary embedding: {self.config.model_name}")

    elif self.use_mrope:
      rotary_embedding = Qwen3OmniMoeThinkerTextRotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.rope_max_timescale,
          embedding_dims=rope_embedding_dims,
          cast_as_fprop_dtype=True,
          fprop_dtype=self.dtype,
          mrope_section=self.mrope_section,
          partial_rotary_factor=(
              self.partial_rotary_factor if self.partial_rotary_factor is not None else self.config.partial_rotary_factor
          ),
          rngs=self.rngs,
      )

    elif self.config.model_name.startswith("llama3.1") or rope_type.startswith("llama3.1"):
      rotary_embedding = LLaMARotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.rope_max_timescale,
          mesh=self.mesh,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          use_scale=rope_use_scale,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
    elif rope_type.startswith("yarn"):
      rotary_embedding = YarnRotaryEmbedding(
          max_position_embeddings=self.config.max_position_embeddings,
          mesh=self.mesh,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          beta_fast=self.config.beta_fast,
          beta_slow=self.config.beta_slow,
          rope_theta=self.rope_max_timescale,
          rope_factor=self.config.rope_factor,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          interleave=self.config.rope_interleave,
          truncate=self.config.rope_truncate,
          attention_scaling=self.config.rope_attention_scaling,
          pairwise=self.config.rope_pairwise,
          direct_position_freqs=self.config.rope_direct_position_freqs,
          dot_pairwise=self.config.rope_dot_pairwise,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )

    elif self.is_qwen3_hybrid:
      rotary_embedding = PartialRotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.rope_max_timescale,
          mesh=self.mesh,
          embedding_dims=self.config.head_dim,
          partial_rotary_factor=self.config.partial_rotary_factor,
          cast_as_fprop_dtype=True,
          fprop_dtype=self.config.dtype,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
    elif self.partial_rotary_factor is not None and self.partial_rotary_factor < 1.0:
      if self.config.model_name.startswith("gemma4"):
        rotary_embedding = Gemma4PartialRotaryEmbedding(
            min_timescale=self.config.rope_min_timescale,
            max_timescale=self.rope_max_timescale,
            mesh=self.mesh,
            embedding_dims=rope_embedding_dims,
            partial_rotary_factor=self.partial_rotary_factor,
            cast_as_fprop_dtype=True,
            fprop_dtype=self.dtype,
            shard_mode=self.config.shard_mode,
            rngs=self.rngs,
        )
      else:
        rotary_embedding = PartialRotaryEmbedding(
            min_timescale=self.config.rope_min_timescale,
            max_timescale=self.rope_max_timescale,
            mesh=self.mesh,
            embedding_dims=rope_embedding_dims,
            partial_rotary_factor=self.partial_rotary_factor,
            cast_as_fprop_dtype=True,
            fprop_dtype=self.dtype,
            shard_mode=self.config.shard_mode,
            rngs=self.rngs,
        )
    else:
      max_timescale = self.rope_max_timescale
      # For local attention use local_rope_max_timescale if it is positive
      if self.attention_type == AttentionType.LOCAL_SLIDING and self.config.local_rope_max_timescale > 0:
        max_timescale = self.config.local_rope_max_timescale

      rope_linear_scaling_factor = self.config.rope_linear_scaling_factor
      # In gemma3, linear scaling factor does not apply to local sliding layers.
      if self.config.model_name.startswith("gemma3") and self.attention_type == AttentionType.LOCAL_SLIDING:
        rope_linear_scaling_factor = 1.0

      rotary_embedding = RotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=max_timescale,
          mesh=self.mesh,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          rope_linear_scaling_factor=rope_linear_scaling_factor,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
    return rotary_embedding

  def apply_rotary_embedding(
      self, inputs: Array, inputs_positions: Optional[Array | None] = None, rope_kwargs: dict | None = None
  ):
    """Applies rotary embeddings, handling different model types.

    Args:
      inputs: The input tensor to apply rotary embeddings to.
      inputs_positions: The positions of the inputs.
      rope_kwargs: A dictionary of keyword arguments for the rotary embedding.

    Returns:
      The input tensor with rotary embeddings applied.
    """
    if isinstance(self.rotary_embedding, Qwen3OmniMoeVisionRotaryEmbedding):
      # For Qwen3OmniMoe vision, pass static dimensions from kwargs.
      num_frames = rope_kwargs.get("num_frames")
      height = rope_kwargs.get("height")
      width = rope_kwargs.get("width")
      token_mask = rope_kwargs.get("token_mask")
      valid_grid = rope_kwargs.get("valid_grid")
      # Type cast required: Omni rotary embedding uses different __call__ parameters than other embeddings.
      return cast(Qwen3OmniMoeVisionRotaryEmbedding, self.rotary_embedding)(
          inputs, num_frames, height, width, token_mask=token_mask, valid_grid=valid_grid
      )
    else:
      return self.rotary_embedding(inputs, inputs_positions)

  def init_kv_caches(self, inputs_kv_shape: Tuple):
    """Initializes KVCache.

    Args:
      inputs_kv_shape: Key/value inputs shape for initialization.

    Returns:
      A KVCache module instance.

    """
    batch_size, _, _ = inputs_kv_shape
    # During initialization, seq_len of inputs_kv is max_target_length,
    # which is not always correct for some functions in KVCache.
    # However, KVCache internal cache shapes are based on max_prefill_length
    # and max_target_length, not the passed seq_len.
    # We can use a placeholder value. The correct fix might involve refactoring
    # KVCache.
    placeholder_seq_len = 1

    return kvcache.KVCache(
        max_prefill_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        batch=batch_size,
        key_seq_len=placeholder_seq_len,
        value_seq_len=placeholder_seq_len,
        key_heads=self.num_kv_heads,
        value_heads=self.num_kv_heads,
        key_head_size=self.head_dim,
        value_head_size=self.head_dim,
        dtype=self.dtype,
        kv_quant=self.kv_quant,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        use_chunked_prefill=self.config.use_chunked_prefill,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

  def update_kv_caches(self, key, value, decoder_segment_ids, model_mode, previous_chunk):
    """Updates the KV caches for prefill and autoregressive modes.

    This method uses a kvcache module to update and retrieve the key-value
    caches based on the current operational mode.

    Args:
      key: The key tensor for the current attention computation.
      value: The value tensor for the current attention computation.
      decoder_segment_ids: Segment IDs for the decoder, used for masking.
      model_mode: The operational mode ('train', 'prefill', 'autoregressive').
      previous_chunk: Information about previously processed chunks, used for
        chunked prefill.

    Returns:
      A list containing two elements:
      - The prefill key-value cache, or None.
      - The autoregressive key-value cache, or None.
    """
    prefill_kv_cache, ar_kv_cache = self.KVCache_0(
        key=key,
        value=value,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
    )
    return [prefill_kv_cache, ar_kv_cache]

  def forward_serve_vllm(
      self,
      query: Array,
      key: Array,
      value: Array,
      rpa_kv_cache: list[Array] | None = None,
      rpa_metadata: dict[str, Any] | None = None,
  ) -> tuple[Array, list[Array]]:
    """Forward function for vLLM serving with RPA attention."""
    if self.config.attention == "vllm_batched_rpa":
      os.environ["USE_BATCHED_RPA_KERNEL"] = "1"
    try:
      # pylint: disable=import-outside-toplevel
      # pytype: disable=import-error
      from tpu_inference.layers.common.attention_interface import sharded_ragged_paged_attention as rpa_ops
    except ImportError as e:
      raise ImportError(
          "vLLM RPA attention ops require the vllm-tpu package. Please install it with `pip install vllm-tpu`."
      ) from e

    query = query.reshape(-1, query.shape[2], query.shape[3]).astype(self.dtype)
    key = key.reshape(-1, key.shape[2], key.shape[3]).astype(self.dtype)
    value = value.reshape(-1, value.shape[2], value.shape[3]).astype(self.dtype)

    if rpa_kv_cache is None or rpa_metadata is None:
      # Return dummy values for dry runs (e.g. during model initialization or JIT tracing)
      return query, []

    # Sliding window applies only to LOCAL_SLIDING layers; global layers must run
    # full attention.
    if self.attention_type == AttentionType.LOCAL_SLIDING and self.config.sliding_window_size > 0:
      attention_chunk_size = self.config.sliding_window_size
    else:
      attention_chunk_size = None

    q_scale, k_scale, v_scale = None, None, None

    md = rpa_metadata

    # With cross-layer KV sharing (Gemma 4 E2B / E4B), a KV-shared layer has no
    # cache of its own: `rpa_kv_cache` here is the donor layer's cache, and
    # attention must run against the K/V the donor already wrote for this
    # position. Only the donor writes the cache; shared layers read it as-is.
    update_kv_cache = not self.share_kv_layer

    sinks = self.sinks.astype(jnp.float32) if self.sinks is not None else None
    if self.mesh is not None and "dcp" in self.mesh.shape and self.mesh.shape["dcp"] > 1:
      from tpu_inference.layers.common.cp_attention import dcp_forward  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

      kv_cache, output = dcp_forward(
          self.mesh,
          query,
          key,
          value,
          rpa_kv_cache,
          md,
          sm_scale=1.0,
          attention_chunk_size=attention_chunk_size,
          q_scale=q_scale,
          k_scale=k_scale,
          v_scale=v_scale,
      )
      return output, kv_cache

    output, kv_cache = rpa_ops(
        self.mesh,
        query,
        key,
        value,
        rpa_kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        sinks,
        1.0,
        attention_chunk_size,
        q_scale,
        k_scale,
        v_scale,
        update_kv_cache=update_kv_cache,
    )
    return output, kv_cache

  def _try_fused_qkv_kernel(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None,
      input_axis_names: AxisNames,
      model_mode: str,
      use_shared_kv: bool,
  ) -> tuple[Array, Array, Array] | None:
    """Runs fused FSDP QAG + Pallas QKV + QK-norm + RoPE + Splash layout when enabled."""
    if not getattr(self.config, "use_fused_qkv_kernel", False) or model_mode != MODEL_MODE_TRAIN:
      if self.defer_qk_norm_grad_allreduce and model_mode == MODEL_MODE_TRAIN:
        raise ValueError(
            "defer_qk_norm_grad_allreduce requires use_fused_qkv_kernel=True: only the fused QKV path defers the"
            " QK-norm scale gradients; the standard path reduces them itself, so the post-scan all-reduce would"
            " count them twice."
        )
      return None
    if (
        inputs_q is not inputs_kv
        or use_shared_kv
        or self.share_kv_layer
        or self.share_kv_projections
        or self.config.fused_qkv
        or self.use_bias_in_projections
        or not self.use_qk_norm
        or self.use_v_norm
        or self.is_qwen3_hybrid
        or self.is_nope_layer
        or self.is_vision
        or self.use_mrope
        or self.temperature_tuning
        or self.config.decoder_block == DecoderBlockType.LLAMA4
        or type(self.query_norm) is not RMSNorm  # pylint: disable=unidiomatic-typecheck
        or type(self.key_norm) is not RMSNorm  # pylint: disable=unidiomatic-typecheck
        or not self.query_norm.with_scale
        or not self.key_norm.with_scale
        or self.query_norm.scale_offset != 0.0
        or self.key_norm.scale_offset != 0.0
        or self.query_norm.epsilon != self.key_norm.epsilon
        or type(self.rotary_embedding) is not RotaryEmbedding  # pylint: disable=unidiomatic-typecheck
        or getattr(self.rotary_embedding, "rope_linear_scaling_factor", 1.0) != 1.0
        or inputs_positions is None
    ):
      raise NotImplementedError(
          "use_fused_qkv_kernel requires self-attention with per-head QK RMSNorm and standard RoPE."
      )

    def _param_val(param):
      val = param.get_value() if hasattr(param, "get_value") else getattr(param, "value", param)
      return val.value if hasattr(val, "value") else val

    def _axes_tuple(entry) -> tuple[str, ...]:
      if entry is None:
        return ()
      raw = (entry,) if isinstance(entry, str) else tuple(entry)
      return tuple(ax for ax in raw if self.mesh.shape.get(ax, 1) > 1)

    def _pad_spec(spec, rank: int) -> jax.sharding.PartitionSpec:
      parts = tuple(spec) + (None,) * max(0, rank - len(spec))
      return jax.sharding.PartitionSpec(*parts[:rank])

    x_pspec = _pad_spec(self._logical_to_mesh_axes(input_axis_names), 3)
    wq_pspec = _pad_spec(self._logical_to_mesh_axes(self.query.kernel_axes), 3)
    wk_pspec = _pad_spec(self._logical_to_mesh_axes(self.key.kernel_axes), 3)
    wv_pspec = _pad_spec(self._logical_to_mesh_axes(self.value.kernel_axes), 3)
    q_norm_pspec = _pad_spec(self._logical_to_mesh_axes(self.query_norm.kernel_axes), 1)
    k_norm_pspec = _pad_spec(self._logical_to_mesh_axes(self.key_norm.kernel_axes), 1)
    q_pspec = _pad_spec(self._logical_to_mesh_axes(self.query_axis_names), 4)
    k_pspec = _pad_spec(self._logical_to_mesh_axes(self.key_axis_names), 4)
    v_pspec = _pad_spec(self._logical_to_mesh_axes(self.value_axis_names), 4)

    if (
        x_pspec[2] is not None
        or wq_pspec[1:] != (None, None)
        or wq_pspec != wk_pspec
        or wq_pspec != wv_pspec
        or q_norm_pspec[0] is not None
        or k_norm_pspec[0] is not None
        or q_pspec[2:] != (None, None)
        or q_pspec[:2] != x_pspec[:2]
        or q_pspec != k_pspec
        or q_pspec != v_pspec
    ):
      raise NotImplementedError(
          f"use_fused_qkv_kernel requires TP=1 on QKV/norm/activations; got x={x_pspec}, wq={wq_pspec}, q={q_pspec}."
      )

    wq = _param_val(self.query.kernel)
    wk = _param_val(self.key.kernel)
    wv = _param_val(self.value.kernel)
    q_norm_scale = _param_val(self.query_norm.scale)
    k_norm_scale = _param_val(self.key_norm.scale)
    if self.query.kernel_scale is not None or self.key.kernel_scale is not None or self.value.kernel_scale is not None:
      raise NotImplementedError("use_fused_qkv_kernel does not support pre-quantized ServeFp8 weights.")

    weight_gather_axes = _axes_tuple(wq_pspec[0])
    batch_seq_axes = tuple(dict.fromkeys(_axes_tuple(x_pspec[0]) + _axes_tuple(x_pspec[1])))
    if not set(weight_gather_axes).issubset(set(batch_seq_axes)):
      raise NotImplementedError(
          f"use_fused_qkv_kernel requires weight shard axes {weight_gather_axes} to shard batch/seq {batch_seq_axes}."
      )
    gather_ways = math.prod(self.mesh.shape[ax] for ax in weight_gather_axes) if weight_gather_axes else 1
    if wq.shape[0] % gather_ways != 0:
      raise NotImplementedError(f"emb={wq.shape[0]} is not divisible by FSDP gather size {gather_ways}.")

    if self.quant is not None or (self.config.quantization and not self.config.use_qwix_quantization):
      raise NotImplementedError("use_fused_qkv_kernel only supports Qwix quantized training or unquantized mode.")
    rule = None
    if self.config.quantization and self.config.use_qwix_quantization:
      if getattr(self.config, "sa_qproj_bwd_bf16", False):
        raise NotImplementedError("use_fused_qkv_kernel does not support sa_qproj_bwd_bf16=True.")
      rule = qpl.get_current_rule("dot_general")
      if rule is not None and not isinstance(rule, qwix.QtRule):
        raise ValueError("Expect a QtRule for quantized training.")
    if rule is not None and (rule.act_qtype is not None or rule.weight_qtype is not None):
      if (
          rule.act_qtype is None
          or rule.weight_qtype is None
          or not jnp.issubdtype(rule.act_qtype, jnp.floating)
          or not jnp.issubdtype(rule.weight_qtype, jnp.floating)
      ):
        raise NotImplementedError(
            "use_fused_qkv_kernel requires floating-point act/weight qtypes,"
            f" got {rule.act_qtype} and {rule.weight_qtype}."
        )
      self._check_fused_qt_rule(rule, "use_fused_qkv_kernel")
      for name, method in (("act", rule.act_calibration_method), ("weight", rule.weight_calibration_method)):
        if not quantizations.is_static_calibration(method):
          raise NotImplementedError(f"use_fused_qkv_kernel needs fixed {name} calibration, got {method!r}.")
      act_qtype = rule.act_qtype
      weight_qtype = rule.weight_qtype
      bwd_qtype = rule.bwd_qtype
      act_scale = quantizations.get_static_scale(act_qtype, rule.act_calibration_method)
      weight_scale = quantizations.get_static_scale(weight_qtype, rule.weight_calibration_method)
      if bwd_qtype is not None:
        if not jnp.issubdtype(bwd_qtype, jnp.floating):
          raise NotImplementedError(f"use_fused_qkv_kernel requires floating-point bwd_qtype, got {bwd_qtype}.")
        if quantizations.is_static_calibration(rule.bwd_calibration_method):
          bwd_scale: float | None = quantizations.get_static_scale(bwd_qtype, rule.bwd_calibration_method)
        elif rule.bwd_calibration_method in (None, "absmax"):
          bwd_scale = None
        else:
          raise NotImplementedError(f"use_fused_qkv_kernel unsupported bwd calibration {rule.bwd_calibration_method!r}.")
      else:
        bwd_scale = None
    else:
      act_qtype = None
      weight_qtype = None
      bwd_qtype = None
      act_scale = 1.0
      weight_scale = 1.0
      bwd_scale = None

    has_q_scalar = bool(self.query_pre_attn_scalar) and self.query_pre_attn_scalar != 1.0
    is_splash_attn = self.attention_kernel in ("flash", "autoselected")
    quant_q_fp8 = is_splash_attn and bool(getattr(self.config, "experimental_sa_quant_q_fp8", False))
    quant_k_fp8 = is_splash_attn and bool(getattr(self.config, "experimental_sa_quant_k_fp8", False))
    q_scalar = float(self.query_pre_attn_scalar) if has_q_scalar else 1.0
    if getattr(self.config, "sa_q_prescaled_log2e", False) and is_splash_attn:
      q_scalar = q_scalar * math.log2(math.e)
    q_out_dtype = jnp.float8_e4m3fn if quant_q_fp8 else None
    k_out_dtype = jnp.float8_e4m3fn if quant_k_fp8 else None

    q_seq_minor = str(getattr(self.config, "sa_q_layout", "")).upper() == "SEQ_MINOR"
    k_seq_minor = str(getattr(self.config, "sa_k_layout", "")).upper() == "SEQ_MINOR"
    seq_axes = _axes_tuple(x_pspec[1])
    seq_ways = math.prod(self.mesh.shape[ax] for ax in seq_axes) if seq_axes else 1
    local_seq_len = inputs_q.shape[1] // seq_ways
    heads_per_tile = math.gcd(max(8, 2 * self.num_kv_heads), self.num_query_heads)
    on_tpu = self.mesh.devices.flat[0].platform == "tpu"
    shape_and_scale_supported = (
        local_seq_len % 128 == 0
        and self.head_dim % 128 == 0
        and self.num_query_heads % heads_per_tile == 0
        and self.num_query_heads % (2 * self.num_kv_heads) == 0
        and (act_qtype is None or (math.log2(act_scale).is_integer() and math.log2(weight_scale).is_integer()))
    )
    if on_tpu and not shape_and_scale_supported:
      raise NotImplementedError(
          f"use_fused_qkv_kernel Pallas kernel requires local_seq_len ({local_seq_len}) and head_dim ({self.head_dim})"
          f" divisible by 128, num_query_heads ({self.num_query_heads}) divisible by 2*num_kv_heads"
          f" ({2 * self.num_kv_heads}), and power-of-two static scales ({act_scale=}, {weight_scale=})."
      )
    kernel_supported = on_tpu and shape_and_scale_supported
    defer_axes = get_deferred_norm_psum_axes(self.config, self.mesh)
    if self.defer_qk_norm_grad_allreduce and model_mode == MODEL_MODE_TRAIN:
      if not defer_axes or set(batch_seq_axes) != set(defer_axes):
        raise ValueError(
            "defer_qk_norm_grad_allreduce requires active deferred norm axes matching attention batch_seq_axes;"
            f" got {defer_axes=} vs {batch_seq_axes=}."
        )
      defer_norm = True
    else:
      defer_norm = False
    # Only `fused_qkv_sharded` (`defer_norm=True`) places `custom_vjp` outside `shard_map` and needs
    # explicit `extra_w_psum_axes`; when `defer_norm=False`, `jax.shard_map(..., check_vma=False)`
    # transposes `wq`/`wk`/`wv` with an automatic `psum` over unmentioned mesh axes.
    extra_w_psum_axes = (
        tuple(ax for ax in batch_seq_axes if ax not in weight_gather_axes and self.mesh.shape.get(ax, 1) > 1)
        if defer_norm
        else ()
    )
    block_m_sel = (
        max(128, math.gcd(1024, local_seq_len)) if local_seq_len % 128 == 0 else min(1024, max(128, local_seq_len))
    )
    quant_rows_sel = max(32, math.gcd(256, block_m_sel)) if block_m_sel % 32 == 0 else min(256, block_m_sel)
    params = fused_qkv.FusedQkvTrainParams(
        num_q_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        eps=float(self.query_norm.epsilon),
        rope_max_timescale=float(self.rotary_embedding.max_timescale),
        rope_min_timescale=float(self.rotary_embedding.min_timescale),
        query_pre_attn_scalar=q_scalar,
        compute_dtype=jnp.dtype(self.dtype),
        act_qtype=act_qtype,
        weight_qtype=weight_qtype,
        bwd_qtype=bwd_qtype,
        act_scale=float(act_scale),
        weight_scale=float(weight_scale),
        bwd_scale=float(bwd_scale) if bwd_scale is not None else None,
        q_seq_minor=q_seq_minor,
        k_seq_minor=k_seq_minor,
        q_out_dtype=q_out_dtype,
        k_out_dtype=k_out_dtype,
        weight_gather_axes=weight_gather_axes,
        extra_w_psum_axes=extra_w_psum_axes,
        defer_norm_grad_psum=defer_norm,
        x_dtype=inputs_q.dtype,
        block_m=block_m_sel,
        heads_per_tile=heads_per_tile,
        quant_rows=quant_rows_sel,
        use_reference=not kernel_supported,
    )
    pos_2d = jnp.broadcast_to(inputs_positions, inputs_q.shape[:2])
    pos_2d = self._maybe_shard_with_logical(pos_2d, input_axis_names[:2])
    pos_pspec = jax.sharding.PartitionSpec(*x_pspec[:2])

    if defer_norm:
      sharded_cfg = fused_qkv.FusedQkvShardedConfig(
          params=params,
          mesh=self.mesh,
          x_pspec=x_pspec,
          wq_pspec=wq_pspec,
          wk_pspec=wk_pspec,
          wv_pspec=wv_pspec,
          q_norm_pspec=q_norm_pspec,
          k_norm_pspec=k_norm_pspec,
          pos_pspec=pos_pspec,
          q_pspec=q_pspec,
          k_pspec=k_pspec,
          v_pspec=v_pspec,
      )
      query, key, value = quantizations.without_qwix_interception(fused_qkv.fused_qkv_sharded)(
          sharded_cfg, inputs_q, wq, wk, wv, q_norm_scale, k_norm_scale, pos_2d
      )
    else:

      @quantizations.without_qwix_interception
      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(x_pspec, wq_pspec, wk_pspec, wv_pspec, q_norm_pspec, k_norm_pspec, pos_pspec),
          out_specs=(q_pspec, k_pspec, v_pspec),
          check_vma=False,
      )
      def _run(x_bsd, wq_l, wk_l, wv_l, q_ns, k_ns, pos_bs):
        return fused_qkv.fused_qkv_local(params, x_bsd, wq_l, wk_l, wv_l, q_ns, k_ns, pos_bs)

      query, key, value = _run(inputs_q, wq, wk, wv, q_norm_scale, k_norm_scale, pos_2d)
    query = self._maybe_shard_with_logical(query, self.query_axis_names)
    key = self._maybe_shard_with_logical(key, self.key_axis_names)
    value = self._maybe_shard_with_logical(value, self.value_axis_names)
    return query, key, value

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
      slot: Optional[int] = None,
      bidirectional_mask: Any = None,
      rope_kwargs: dict | None = None,
      kv_cache: Optional[Array] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
      shared_key: Array | None = None,
      shared_value: Array | None = None,
  ):
    """Applies Attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention, and project the results to an output vector.

    This method handles three modes:
    1.  **Training**: The KV cache is ignored.
    2.  **Prefill**: The KV cache is filled with the key-value pairs from the input sequence.
    3.  **Autoregressive Decoding**: The KV cache is used to provide context from previous steps.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: Input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: Key/values of shape `[batch, kv_length, kv_features]`.
      inputs_positions: Input positions for rotary embeddings.
      decoder_segment_ids: Segment IDs for masking.
      model_mode: The operational mode ('train', 'prefill', 'autoregressive').
      deterministic: If True, disables dropout.
      previous_chunk: Information about previously processed chunks for chunked prefill.
      slot: The batch slot index for paged attention.
      bidirectional_mask: A mask for bidirectional attention, used in multimodal models.
      kv_cache: Optional KV cache input, used when invoking from vLLM.
      attention_metadata: Optional mapping to store attention metadata, used when invoking from vLLM.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    if model_mode == MODEL_MODE_PREFILL:
      input_axis_names = self.prefill_input_axis_names
    elif model_mode == MODEL_MODE_TRAIN:
      input_axis_names = self.input_axis_names
    else:
      input_axis_names = self.decode_input_axis_names

    use_shared_kv = shared_key is not None and shared_value is not None
    if self.share_kv_layer and not use_shared_kv:
      raise ValueError("share_kv_layer=True requires both shared_key and shared_value to be provided.")
    if use_shared_kv and self.config.fused_qkv:
      raise ValueError("shared_key / shared_value are incompatible with fused_qkv.")

    pin_layout = (
        model_mode == MODEL_MODE_TRAIN
        and resolve_context_parallel_strategy(self.config, self.attention_type) == "ulysses"
        and not use_shared_kv
    )
    if pin_layout:
      same_kv = inputs_kv is inputs_q
      inputs_q = _pin_row_major_with_cotangent(inputs_q)
      inputs_kv = inputs_q if same_kv else _pin_row_major_with_cotangent(inputs_kv)
    is_self_attn = inputs_q is inputs_kv
    inputs_q = self._maybe_shard_with_logical(inputs_q, input_axis_names)
    inputs_kv = inputs_q if is_self_attn else self._maybe_shard_with_logical(inputs_kv, input_axis_names)
    qkv_sharding = create_sharding(self.mesh, input_axis_names)

    gate = None
    fused_qkv_out = self._try_fused_qkv_kernel(
        inputs_q, inputs_kv, inputs_positions, input_axis_names, model_mode, use_shared_kv
    )
    if fused_qkv_out is not None:
      query, key, value = fused_qkv_out
    else:
      # apply projection.
      if self.config.fused_qkv:
        query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
      elif use_shared_kv:
        # Donor layer already produced rotated, normed K/V — use them directly.
        query = self.query_projection(inputs_q, out_sharding=qkv_sharding)
        key, value = shared_key, shared_value
      else:
        query = self.query_projection(inputs_q, out_sharding=qkv_sharding)
        key = self.kv_projection(inputs_kv, proj_name="key", out_sharding=qkv_sharding)
        if self.share_kv_projections:
          value = key
        else:
          value = self.kv_projection(inputs_kv, proj_name="value", out_sharding=qkv_sharding)

      if self.is_qwen3_hybrid:
        # Split query into query & gate.
        query, gate = jnp.split(query, 2, axis=-1)
        batch_size, seq_len, _, _ = gate.shape
        gate = gate.reshape(batch_size, seq_len, self.config.num_query_heads * self.config.head_dim)

      is_llama4_decoder_block = self.config.decoder_block == DecoderBlockType.LLAMA4
      # NOTE: llama 4 does L2 normalization after RoPE
      # Apply Qwen3Next specific RMS Norm
      if (self.use_qk_norm and not is_llama4_decoder_block) or self.is_qwen3_hybrid:
        query = self.query_norm(query)
        if not use_shared_kv:
          key = self.key_norm(key)

      if self.use_v_norm and not use_shared_kv:
        value = self.value_norm(value)

      # NOTE: is_nope_layer should be used in attention mask and also used in attention tuning
      use_rope = not self.is_nope_layer
      use_qk_norm = self.use_qk_norm and use_rope

      if use_rope:
        query = self.apply_rotary_embedding(query, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)
        if not use_shared_kv:
          key = self.apply_rotary_embedding(key, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)

      if use_qk_norm and is_llama4_decoder_block:
        l2_norm = L2Norm(eps=self.config.normalization_layer_epsilon)
        query = l2_norm(query)
        if not use_shared_kv:
          key = l2_norm(key)

      has_q_scalar = bool(self.query_pre_attn_scalar) and self.query_pre_attn_scalar != 1.0
      # Optional fp8 (e4m3, static scale 1) Q/K for the Tokamax Splash kernel, train mode only (the
      # KV cache and decode paths keep bf16). Mirrors the MLA path in attention_mla.py: Q is cast once
      # after qk-norm, RoPE and the softmax scale (with LOG2E folded in f32 when sa_q_prescaled_log2e so
      # the kernel skips its own in-kernel rescale and q is rounded to fp8 exactly once).
      is_splash_attn = self.attention_kernel in ("flash", "autoselected")
      quant_q_fp8 = (
          model_mode == MODEL_MODE_TRAIN
          and is_splash_attn
          and bool(getattr(self.config, "experimental_sa_quant_q_fp8", False))
      )
      quant_k_fp8 = (
          model_mode == MODEL_MODE_TRAIN
          and is_splash_attn
          and bool(getattr(self.config, "experimental_sa_quant_k_fp8", False))
      )
      prescale_log2e = (
          bool(getattr(self.config, "sa_q_prescaled_log2e", False))
          and is_splash_attn
          and model_mode != MODEL_MODE_AUTOREGRESSIVE
      )
      if not quant_q_fp8 and not prescale_log2e and has_q_scalar:
        query = query * self.query_pre_attn_scalar

      if self.temperature_tuning and not use_rope:
        attn_scales = (
            jnp.log(jnp.floor((inputs_positions.astype(self.dtype) + 1.0) / self.temperature_tuning_floor_scale) + 1.0)
            * self.temperature_tuning_scale
            + 1.0
        )
        query = (query * attn_scales[:, :, jnp.newaxis, jnp.newaxis]).astype(self.dtype)

      if quant_q_fp8:
        q_scale = float(self.query_pre_attn_scalar) if has_q_scalar else 1.0
        if prescale_log2e:
          q_scale = q_scale * math.log2(math.e)
        query = (query.astype(jnp.float32) * q_scale).astype(jnp.float8_e4m3fn)
      elif prescale_log2e:
        q_scale = (float(self.query_pre_attn_scalar) if has_q_scalar else 1.0) * math.log2(math.e)
        query = (query.astype(jnp.float32) * q_scale).astype(self.dtype)
      if quant_k_fp8:
        key = key.astype(jnp.float8_e4m3fn)

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

    assert not self.config.quantize_kvcache or self.kv_quant

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
    out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")
    if model_mode == MODEL_MODE_PREFILL:
      out = self._maybe_shard_with_logical(out, self.prefill_out_axis_names)
    elif model_mode == MODEL_MODE_TRAIN:
      out = self._maybe_shard_with_logical(out, self.out_axis_names)
    else:
      out = self._maybe_shard_with_logical(out, self.decode_out_axis_names)
    if self.is_qwen3_hybrid:
      out = out.reshape(batch_size, seq_len, self.config.num_query_heads * self.config.head_dim)
      out = out * jax.nn.sigmoid(gate)
    out = self.out_projection(out, out_sharding=out_sharding, model_mode=model_mode)
    if getattr(self.config, "distill_beta", 0.0) > 0.0:
      self.sow(nnx.Intermediate, "out_projection_activations", out)
    if pin_layout:
      out = _pin_row_major_with_cotangent(out)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache
