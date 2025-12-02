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

"""Attentions Layers."""

import dataclasses
import functools
from typing import Any, Iterable, Optional, Tuple, Union, cast

from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding
import jax
import jax.numpy as jnp

from flax import nnx, linen as nn

from MaxText.common_types import (
    DecoderBlockType,
    BATCH,
    BATCH_NO_EXP,
    HEAD,
    PREFILL_LENGTH,
    D_KV,
    AxisNames,
    AxisIdxes,
    LENGTH,
    LENGTH_NO_EXP,
    DType,
    Config,
    Array,
    DECODE_LENGTH,
    DECODE_BATCH,
    PREFILL_KV_BATCH,
    KV_HEAD,
    KV_HEAD_DIM,
    KV_BATCH,
    KV_BATCH_NO_EXP,
    EMBED,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_TRAIN,
    MODEL_MODE_PREFILL,
    EP_AS_CONTEXT,
    AttentionType,
)
from MaxText.sharding import maybe_shard_with_logical
from MaxText.inference import kvcache
from MaxText.inference import page_manager
from MaxText.inference import paged_attention
from MaxText.inference.kvcache import KVQuant
from MaxText.layers import nnx_wrappers
from MaxText.layers.attention_op import AttentionOp
from MaxText.layers.embeddings import (
    LLaMARotaryEmbedding,
    LlamaVisionRotaryEmbedding,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeVisionRotaryEmbedding,
    RotaryEmbedding,
    YarnRotaryEmbedding,
    Qwen3NextRotaryEmbedding,
)
from MaxText.layers.initializers import nd_dense_init, NdInitializer, variable_to_logically_partitioned, default_bias_init
from MaxText.layers.linears import DenseGeneral, canonicalize_tuple, normalize_axes
from MaxText.layers.normalizations import RMSNorm, Qwen3NextRMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant

# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


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


def l2_norm_as_linen(self, eps: float = 1e-6):
  """
  Initializes the L2Norm module and returns it as a Linen module.

  Args:
    eps: float, epsilon used for numerical stability (default value should be ok for most cases).
  """
  return nnx_wrappers.to_linen(L2Norm, eps=eps, metadata_fn=variable_to_logically_partitioned)


def attention_as_linen(
    *,
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
    attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
    use_qk_norm: bool = False,
    query_pre_attn_scalar: float | None = None,
    use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
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
    query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
    ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
    out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
    ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
    prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
    decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
    prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
    decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
    prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    compute_axis_order: AxisIdxes = (0, 1, 2, 3),
    reshape_q: bool = False,
    is_nope_layer: bool = False,
    is_vision: bool = False,
    model_mode: str = MODEL_MODE_TRAIN,
    use_mrope: bool = False,
    mrope_section: tuple[int, int, int] | None = None,
    name: str | None = None,
):
  """A factory function to create an Attention as a Linen module.

  This function serves as a bridge to use the NNX-based `Attention` within a
  Linen model.
  """
  return nnx_wrappers.to_linen(
      Attention,
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
      kernel_init=kernel_init,
      float32_qk_product=float32_qk_product,
      float32_logits=float32_logits,
      quant=quant,
      kv_quant=kv_quant,
      attention_type=attention_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
      sliding_window_size=sliding_window_size,
      use_ragged_attention=use_ragged_attention,
      ragged_block_size=ragged_block_size,
      use_qk_norm=use_qk_norm,
      query_pre_attn_scalar=query_pre_attn_scalar,
      use_bias_in_projections=use_bias_in_projections,
      temperature_tuning=temperature_tuning,
      temperature_tuning_scale=temperature_tuning_scale,
      temperature_tuning_floor_scale=temperature_tuning_floor_scale,
      prefill_query_axis_names=prefill_query_axis_names,
      prefill_key_axis_names=prefill_key_axis_names,
      prefill_value_axis_names=prefill_value_axis_names,
      query_axis_names=query_axis_names,
      key_axis_names=key_axis_names,
      value_axis_names=value_axis_names,
      ep_query_axis_names=ep_query_axis_names,
      ep_key_axis_names=ep_key_axis_names,
      ep_value_axis_names=ep_value_axis_names,
      input_axis_names=input_axis_names,
      ep_input_axis_names=ep_input_axis_names,
      out_axis_names=out_axis_names,
      ep_out_axis_names=ep_out_axis_names,
      prefill_input_axis_names=prefill_input_axis_names,
      decode_input_axis_names=decode_input_axis_names,
      prefill_out_axis_names=prefill_out_axis_names,
      decode_out_axis_names=decode_out_axis_names,
      prefill_cache_axis_order=prefill_cache_axis_order,
      ar_cache_axis_order=ar_cache_axis_order,
      compute_axis_order=compute_axis_order,
      reshape_q=reshape_q,
      is_nope_layer=is_nope_layer,
      is_vision=is_vision,
      model_mode=model_mode,
      use_mrope=use_mrope,
      mrope_section=mrope_section,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )


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
      attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
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
      query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
      ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
      out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
      ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
      prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
      decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
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
      rngs: Optional[nnx.Rngs] = None,
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
      temperature_tuning: Whether to use temperature tuning for attention.
      temperature_tuning_scale: The scale for temperature tuning.
      temperature_tuning_floor_scale: The floor scale for temperature tuning.
      ... other configuration parameters.
      is_nope_layer: Whether this is a "NoPE" (No Position-Embedding) layer.
      is_vision: Whether this is a vision attention layer.
      model_mode: The model's operational mode (e.g., 'train', 'prefill').
      base_kv_cache: Whether to use base (non-MLA) kv cache, if KVCache is used
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
    self.attention_type = attention_type
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.sliding_window_size = sliding_window_size
    self.use_ragged_attention = use_ragged_attention
    self.ragged_block_size = ragged_block_size
    self.use_qk_norm = use_qk_norm
    self.query_pre_attn_scalar = query_pre_attn_scalar
    self.use_bias_in_projections = use_bias_in_projections
    self.temperature_tuning = temperature_tuning
    self.temperature_tuning_scale = temperature_tuning_scale
    self.temperature_tuning_floor_scale = temperature_tuning_floor_scale
    self.prefill_query_axis_names = prefill_query_axis_names
    self.prefill_key_axis_names = prefill_key_axis_names
    self.prefill_value_axis_names = prefill_value_axis_names
    self.query_axis_names = query_axis_names
    self.key_axis_names = key_axis_names
    self.value_axis_names = value_axis_names
    self.ep_query_axis_names = ep_query_axis_names
    self.ep_key_axis_names = ep_key_axis_names
    self.ep_value_axis_names = ep_value_axis_names
    self.input_axis_names = input_axis_names
    self.ep_input_axis_names = ep_input_axis_names
    self.out_axis_names = out_axis_names
    self.ep_out_axis_names = ep_out_axis_names
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

    self.is_qwen3_next = self.config.decoder_block == DecoderBlockType.QWEN3_NEXT

    # Module attribute names must match names previously passed to Linen for checkpointing
    self.KVCache_0 = (
        self.init_kv_caches(inputs_kv_shape=inputs_kv_shape)
        if self.model_mode != MODEL_MODE_TRAIN and base_kv_cache
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
        num_kv_heads=self.num_kv_heads,
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
    # When paged attention is enabled, paged attention op is used for all model modes except TRAIN,
    # which uses default attention op.
    if self.config.attention == "paged":
      self.paged_attention_op = paged_attention.PagedAttentionOp(
          mesh=self.mesh,
          num_pages=self.config.pagedattn_num_pages,
          tokens_per_page=self.config.pagedattn_tokens_per_page,
          max_pages_per_slot=(self.config.max_target_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          max_pages_per_prefill=(self.config.max_prefill_predict_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          pages_per_compute_block=self.config.pagedattn_pages_per_compute_block,
          num_kv_heads=self.num_kv_heads,
          kv_head_dim_size=self.head_dim,
          dtype=self.dtype,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
          rngs=self.rngs,
      )

    self._init_projections(inputs_q_shape, inputs_kv_shape)

    if self.config.attention_sink:
      self.sinks = nnx.Param(
          default_bias_init(self.rngs.params(), (self.config.num_query_heads,), self.weight_dtype),
          sharding=(None,),
      )
    else:
      self.sinks = None

    is_llama4_decoder_block = self.config.decoder_block == DecoderBlockType.LLAMA4
    if self.use_qk_norm and not is_llama4_decoder_block:
      self.query_norm = RMSNorm(
          num_features=self.head_dim,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          shard_mode=self.config.shard_mode,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
      self.key_norm = RMSNorm(
          num_features=self.head_dim,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          shard_mode=self.config.shard_mode,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    elif self.is_qwen3_next:
      self.query_norm = Qwen3NextRMSNorm(
          num_features=self.config.head_dim,
          eps=self.config.normalization_layer_epsilon,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          rngs=self.rngs,
      )
      self.key_norm = Qwen3NextRMSNorm(
          num_features=self.config.head_dim,
          eps=self.config.normalization_layer_epsilon,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          rngs=self.rngs,
      )
    else:
      self.query_norm = None
      self.key_norm = None

    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=config.shard_mode,
    )

  def _init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple) -> None:
    """Initializes the query, key, value, and output projections."""
    if self.config.fused_qkv:
      self.qkv_proj = self.init_qkv_w(inputs_shape=inputs_q_shape)
    else:
      self.query = self.init_query_w(inputs_q_shape=inputs_q_shape)
      self.key = self.init_kv_w(inputs_kv_shape=inputs_kv_shape)
      self.value = self.init_kv_w(inputs_kv_shape=inputs_kv_shape)
    self.out = self.init_out_w(output_dim=inputs_q_shape[-1])

  def init_query_w(self, inputs_q_shape: Tuple) -> nnx.Module:
    """Query projection initialization."""

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    # We disable depth_scaling when using qk_norm or a query_pre_attn_scalar
    # to avoid applying scaling twice.
    if self.config.use_qk_norm or (self.query_pre_attn_scalar is not None and self.query_pre_attn_scalar != 1.0):
      depth_scaling = 1.0
    else:
      depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

    def query_init(*args):
      # pylint: disable=no-value-for-parameter
      return self.kernel_init(*args) / depth_scaling

    kernel_axes = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("embed", "q_heads", "kv")
    )
    in_features = self.convert_dense_general_inputs_shape(inputs_q_shape)
    out_features = (self.num_query_heads, self.head_dim)

    if self.is_qwen3_next:
      out_features = (self.num_query_heads, self.head_dim * 2)

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
    if self.num_kv_heads == -1:
      raise ValueError("num_kv_heads is not defined.")

    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError("Invalid num_kv_heads for GQA.")

    kernel_axes = (
        (None, None, None)
        if self.config.ici_context_autoregressive_parallelism > 1
        else ("embed", "kv_heads", "kv_head_dim")
    )

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
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
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
    return DenseGeneral(
        in_features_shape=self.convert_dense_general_inputs_shape(inputs_shape),
        out_features_shape=(3, self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "qkv", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        shard_mode=self.config.shard_mode,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )

  def qkv_projection(self, inputs: Array, proj_name: str, out_sharding: NamedSharding | None = None):
    """Fused QKV projection"""

    qkv_proj = self.qkv_proj(inputs, out_sharding)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def init_out_w(self, output_dim: int) -> nnx.Module:
    """out projection"""
    in_features = (self.num_query_heads, self.head_dim)
    out_features = output_dim
    out_kernel_axis = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("heads", "kv", "embed")
    )
    axis = (-2, -1)

    if self.is_qwen3_next:
      in_features = self.num_query_heads * self.head_dim
      out_kernel_axis = ("mlp", "embed")
      axis = (-1,)

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
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )

  def out_projection(self, out: Array, out_sharding: NamedSharding | None = None) -> Array:
    """out projection"""
    return self.out(out, out_sharding=out_sharding)

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

    rope_type = self.config.rope_type.lower()
    rope_use_scale = self.config.rope_use_scale
    if self.is_vision:
      if self.config.model_name.startswith("qwen3-omni"):
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
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=rope_embedding_dims,
          cast_as_fprop_dtype=True,
          fprop_dtype=self.dtype,
          mrope_section=self.mrope_section,
          rngs=self.rngs,
      )

    elif self.config.model_name.startswith("llama3.1") or rope_type.startswith("llama3.1"):
      rotary_embedding = LLaMARotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          use_scale=rope_use_scale,
          rngs=self.rngs,
      )
    elif rope_type.startswith("yarn"):
      rotary_embedding = YarnRotaryEmbedding(
          max_position_embeddings=self.config.max_position_embeddings,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          beta_fast=self.config.beta_fast,
          beta_slow=self.config.beta_slow,
          rope_theta=self.config.rope_max_timescale,
          rope_factor=self.config.rope_factor,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          interleave=self.config.rope_interleave,
          truncate=self.config.rope_truncate,
          attention_scaling=self.config.rope_attention_scaling,
          rngs=self.rngs,
      )
    elif self.is_qwen3_next:
      rotary_embedding = Qwen3NextRotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=self.config.head_dim,
          partial_rotary_factor=self.config.partial_rotary_factor,
          cast_as_fprop_dtype=True,
          fprop_dtype=self.config.dtype,
          rngs=self.rngs,
      )
    else:
      max_timescale = self.config.rope_max_timescale
      # For local attention use local_rope_max_timescale if it's is positive
      if self.attention_type == AttentionType.LOCAL_SLIDING and self.config.local_rope_max_timescale > 0:
        max_timescale = self.config.local_rope_max_timescale

      rope_linear_scaling_factor = self.config.rope_linear_scaling_factor
      # In gemma3, linear scaling factor does not apply to local sliding layers.
      if self.config.model_name.startswith("gemma3") and self.attention_type == AttentionType.LOCAL_SLIDING:
        rope_linear_scaling_factor = 1.0

      rotary_embedding = RotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=max_timescale,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          rope_linear_scaling_factor=rope_linear_scaling_factor,
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
      # Type cast required: Omni rotary embedding uses different __call__ parameters than other embeddings.
      return cast(Qwen3OmniMoeVisionRotaryEmbedding, self.rotary_embedding)(inputs, num_frames, height, width)
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
  ) -> tuple[list[Array], Array]:
    """Forward function for vLLM serving with RPA attention."""
    try:
      # pylint: disable=import-outside-toplevel
      # pytype: disable=import-error
      from tpu_inference.layers.jax.attention_interface import sharded_ragged_paged_attention as rpa_ops
    except ImportError as e:
      raise ImportError(
          "vLLM RPA attention ops require the vllm-tpu package. Please install it with `pip install vllm-tpu`."
      ) from e

    if self.config.attention_sink:
      raise NotImplementedError("Attention sink is not supported in MaxText vLLM RPA attention.")

    if rpa_kv_cache is None or rpa_metadata is None:
      raise ValueError("kv_cache and attention_metadata must be provided when using vLLM.")

    query = query.reshape(-1, query.shape[2], query.shape[3])
    key = key.reshape(-1, key.shape[2], key.shape[3])
    value = value.reshape(-1, value.shape[2], value.shape[3])

    attention_chunk_size = self.config.chunk_attn_window_size if self.config.chunk_attn_window_size > 0 else None
    q_scale, k_scale, v_scale = None, None, None

    md = rpa_metadata

    output, kv_cache = rpa_ops(1.0, self.mesh, attention_chunk_size, q_scale, k_scale, v_scale)(
        query,
        key,
        value,
        rpa_kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
    )
    return kv_cache, output

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
      page_state: Optional[page_manager.PageState] = None,
      bidirectional_mask: Any = None,
      rope_kwargs: dict | None = None,
      kv_cache: Optional[Array] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
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
      page_state: The current state of the paged attention manager.
      bidirectional_mask: A mask for bidirectional attention, used in multimodal models.
      kv_cache: Optional KV cache input, used when invoking from vLLM.
      attention_metadata: Optional mapping to store attention metadata, used when invoking from vLLM.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.prefill_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.prefill_input_axis_names)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.ep_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.ep_input_axis_names)
    elif model_mode == MODEL_MODE_TRAIN:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.input_axis_names)
    else:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.decode_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.decode_input_axis_names)

    # apply projection.
    if self.config.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    else:
      query_sharding = NamedSharding(self.mesh, nn.logical_to_mesh_axes(self.query_axis_names))
      query = self.query_projection(inputs_q, out_sharding=query_sharding)
      key_sharding = NamedSharding(self.mesh, nn.logical_to_mesh_axes(self.key_axis_names))
      key = self.kv_projection(inputs_kv, proj_name="key", out_sharding=key_sharding)
      value_sharding = NamedSharding(self.mesh, nn.logical_to_mesh_axes(self.value_axis_names))
      value = self.kv_projection(inputs_kv, proj_name="value", out_sharding=value_sharding)

    gate = None
    if self.is_qwen3_next:
      # Split query into query & gate.
      query, gate = jnp.split(query, 2, axis=-1)
      batch_size, seq_len, _, _ = gate.shape
      gate = gate.reshape(batch_size, seq_len, self.config.num_query_heads * self.config.head_dim)

    is_llama4_decoder_block = self.config.decoder_block == DecoderBlockType.LLAMA4
    # NOTE: llama 4 does L2 normalization after RoPE
    # Apply Qwen3Next specific RMS Norm
    if (self.use_qk_norm and not is_llama4_decoder_block) or self.is_qwen3_next:
      query = self.query_norm(query)
      key = self.key_norm(key)

    # NOTE: is_nope_layer should be used in attention mask and also used in attention tuning
    use_rope = not self.is_nope_layer
    use_qk_norm = self.use_qk_norm and use_rope

    if use_rope:
      query = self.apply_rotary_embedding(query, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)
      key = self.apply_rotary_embedding(key, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)

    if use_qk_norm and is_llama4_decoder_block:
      l2_norm = L2Norm(eps=self.config.normalization_layer_epsilon)
      query = l2_norm(query)
      key = l2_norm(key)

    # apply query_pre_attn_scalar if it's present.
    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      query = query * self.query_pre_attn_scalar

    if self.temperature_tuning and not use_rope:
      attn_scales = (
          jnp.log(jnp.floor((inputs_positions.astype(self.dtype) + 1.0) / self.temperature_tuning_floor_scale) + 1.0)
          * self.temperature_tuning_scale
          + 1.0
      )
      query = (query * attn_scales[:, :, jnp.newaxis, jnp.newaxis]).astype(self.dtype)

    if model_mode == MODEL_MODE_PREFILL:
      query = self._maybe_shard_with_logical(query, self.prefill_query_axis_names)
      key = self._maybe_shard_with_logical(key, self.prefill_key_axis_names)
      value = self._maybe_shard_with_logical(value, self.prefill_value_axis_names)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      query = self._maybe_shard_with_logical(query, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      key = self._maybe_shard_with_logical(key, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
      value = self._maybe_shard_with_logical(value, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      query = self._maybe_shard_with_logical(query, self.ep_query_axis_names)
      key = self._maybe_shard_with_logical(key, self.ep_key_axis_names)
      value = self._maybe_shard_with_logical(value, self.ep_value_axis_names)
    else:
      query = self._maybe_shard_with_logical(query, self.query_axis_names)
      key = self._maybe_shard_with_logical(key, self.key_axis_names)
      value = self._maybe_shard_with_logical(value, self.value_axis_names)

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    assert not self.config.quantize_kvcache or self.kv_quant

    if self.config.attention == "paged" and model_mode != MODEL_MODE_TRAIN:
      unnormalized_out, _, exp_sum = self.paged_attention_op(
          query, key, value, decoder_segment_ids, model_mode, previous_chunk, slot=slot, page_state=page_state
      )
      out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out

    elif self.config.attention == "vllm_rpa" and model_mode != MODEL_MODE_TRAIN:
      batch, seq_len, num_heads, head_dim = query.shape
      updated_kv, attn_out = self.forward_serve_vllm(
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
          model_mode,
          cached_values,
          previous_chunk,
          bidirectional_mask,
          self.sinks,
      )
    if self.is_qwen3_next:
      out = out.reshape(batch_size, seq_len, self.config.num_query_heads * self.config.head_dim)
      out = out * jax.nn.sigmoid(gate)
    if model_mode == MODEL_MODE_PREFILL:
      out = self._maybe_shard_with_logical(out, self.prefill_out_axis_names)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      out = self._maybe_shard_with_logical(out, self.ep_out_axis_names)
    elif model_mode == MODEL_MODE_TRAIN:
      out = self._maybe_shard_with_logical(out, self.out_axis_names)
    else:
      out = self._maybe_shard_with_logical(out, self.decode_out_axis_names)
    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache
