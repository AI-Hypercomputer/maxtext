# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-agnostic Key-Value (KV) cache for modern MaxText models (m3).

This module provides a clean, pure-NNX KV cache for standard Transformer
architectures (MHA, GQA, MQA). It manages prefill and autoregressive decoding
state buffers without model-specific branches, flags, or legacy Linen wrappers.

Architecture:
  * Manages two-phase caching:
      - Prefill Cache: stores full prompt KV representations computed during prefill.
      - Autoregressive (AR) Cache: ring buffer updated token-by-token during decoding.
  * Axis Transposition: caches tensors in an optimized hardware axis order
    (default: (seq, heads, batch, dim)) for TPU memory layout and fusion.
  * Integration: exposes standard cache variables (e.g. ,
    , ) expected by MaxEngine / continuous batching.

Non-standard or specialized caching schemes (e.g. DeepSeek MLA latent cache,
DeepSeek-V4 compressed memory, Qwen3-Next GDN recurrent states) belong in their
respective model family directories rather than adding branches to this module.
"""

from typing import Any, Optional
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import (
    CACHE_BATCH,
    CACHE_BATCH_PREFILL,
    CACHE_HEADS,
    CACHE_KV,
    CACHE_SEQUENCE,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    DType,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
)


def transpose_tuple(t: tuple, order: tuple[int, ...]) -> tuple:
  """Permutes a tuple according to order."""
  return tuple(t[i] for i in order)


def reverse_transpose(arr: jax.Array, order: tuple[int, ...]) -> jax.Array:
  """Inverts an axis permutation on a JAX array."""
  inv_order = tuple(order.index(i) for i in range(len(order)))
  return jnp.transpose(arr, inv_order)


class KVCache(nnx.Module):
  """Clean, model-agnostic Key-Value Cache for standard Transformer attention."""

  def __init__(
      self,
      max_prefill_length: int,
      max_target_length: int,
      batch: int,
      key_head_size: int,
      value_head_size: int,
      dtype: DType,
      key_heads: Optional[int] = None,
      value_heads: Optional[int] = None,
      key_num_heads: Optional[int] = None,
      value_num_heads: Optional[int] = None,
      weight_dtype: Optional[DType] = None,
      prefill_cache_axis_order: tuple[int, ...] = (1, 2, 0, 3),
      ar_cache_axis_order: tuple[int, ...] = (1, 2, 0, 3),
      model_mode: str = MODEL_MODE_PREFILL,
      use_chunked_prefill: bool = False,
      mesh: Optional[Mesh] = None,
      **kwargs,
  ):
    """Initializes standard Key-Value Cache.

    Args:
      max_prefill_length: Maximum sequence length for prompt prefill.
      max_target_length: Maximum total target length (prefill + decode tokens).
      batch: Batch size for cache allocations.
      key_head_size: Dimensionality of each key head.
      value_head_size: Dimensionality of each value head.
      dtype: Data type for cached key and value tensors.
      key_heads: Number of key heads.
      value_heads: Number of value heads.
      key_num_heads: Alias for key_heads.
      value_num_heads: Alias for value_heads.
      weight_dtype: Storage data type.
      prefill_cache_axis_order: Permutation order for prefill cache layout.
      ar_cache_axis_order: Permutation order for autoregressive cache layout.
      model_mode: Initial operational mode ('prefill' or 'autoregressive').
      use_chunked_prefill: Whether chunked prefill is enabled.
      mesh: Optional JAX device mesh.
      **kwargs: Ignored compatibility kwargs.
    """
    super().__init__()
    self.max_prefill_length = max_prefill_length
    self.max_target_length = max_target_length
    self.batch = batch
    self.key_heads = key_heads if key_heads is not None else key_num_heads
    self.value_heads = value_heads if value_heads is not None else value_num_heads
    self.key_head_size = key_head_size
    self.value_head_size = value_head_size
    self.dtype = dtype
    self.weight_dtype = weight_dtype or dtype
    self.prefill_cache_axis_order = prefill_cache_axis_order
    self.ar_cache_axis_order = ar_cache_axis_order
    self.model_mode = model_mode
    self.use_chunked_prefill = use_chunked_prefill

    # Logical axis names for mesh SPMD partitioning
    self.cache_logical_axis_names = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
    self.prefill_cache_logical_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)

    if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
      self._initialize_prefill_caches(model_mode)
      self._initialize_ar_caches(model_mode)

  def _initialize_prefill_caches(self, model_mode: str) -> None:
    """Allocates prefill key, value, and segment ID cache state buffers."""
    cache_length = self.max_prefill_length
    cache_logical_axis_names = (
        self.prefill_cache_logical_axis_names
        if model_mode == MODEL_MODE_PREFILL
        else self.cache_logical_axis_names
    )
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.prefill_cache_axis_order)

    cache_logical_shape_k = (self.batch, cache_length, self.key_heads, self.key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape_k, self.prefill_cache_axis_order)

    cache_logical_shape_v = (self.batch, cache_length, self.value_heads, self.value_head_size)
    cache_shape_value = transpose_tuple(cache_logical_shape_v, self.prefill_cache_axis_order)

    self.cached_prefill_key = nnx.Cache(
        jnp.zeros(cache_shape_key, dtype=self.dtype),
        out_sharding=cache_axis_names,
    )
    self.cached_prefill_value = nnx.Cache(
        jnp.zeros(cache_shape_value, dtype=self.dtype),
        out_sharding=cache_axis_names,
    )

    segment_id_axis_names = (
        (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
        if model_mode == MODEL_MODE_PREFILL
        else (CACHE_BATCH, CACHE_SEQUENCE)
    )
    self.cache_prefill_segment_id = nnx.Cache(
        jnp.zeros((self.batch, cache_length), dtype=jnp.int32),
        out_sharding=segment_id_axis_names,
    )

    # Placeholders for scales to maintain signature compatibility
    self.cached_prefill_key_scale = None
    self.cached_prefill_value_scale = None

  def _initialize_ar_caches(self, model_mode: str) -> None:
    """Allocates autoregressive ring-buffer cache state buffers."""
    if self.max_target_length <= self.max_prefill_length:
      raise ValueError(
          f"max_target_length ({self.max_target_length}) must be greater than "
          f"max_prefill_length ({self.max_prefill_length})"
      )
    cache_length = self.max_target_length - self.max_prefill_length

    cache_logical_axis_names = (
        self.prefill_cache_logical_axis_names
        if model_mode == MODEL_MODE_PREFILL
        else self.cache_logical_axis_names
    )
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.ar_cache_axis_order)

    cache_logical_shape_k = (self.batch, cache_length, self.key_heads, self.key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape_k, self.ar_cache_axis_order)

    cache_logical_shape_v = (self.batch, cache_length, self.value_heads, self.value_head_size)
    cache_shape_value = transpose_tuple(cache_logical_shape_v, self.ar_cache_axis_order)

    self.cached_ar_key = nnx.Cache(
        jnp.zeros(cache_shape_key, dtype=self.dtype),
        out_sharding=cache_axis_names,
    )
    self.cached_ar_value = nnx.Cache(
        jnp.zeros(cache_shape_value, dtype=self.dtype),
        out_sharding=cache_axis_names,
    )

    segment_id_axis_names = (
        (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
        if model_mode == MODEL_MODE_PREFILL
        else (CACHE_BATCH, CACHE_SEQUENCE)
    )
    self.cache_ar_segment_id = nnx.Cache(
        jnp.zeros((self.batch, cache_length), dtype=jnp.int32),
        out_sharding=segment_id_axis_names,
    )

    self.cached_ar_lengths = nnx.Cache(
        jnp.zeros((self.batch,), dtype=jnp.int32),
        out_sharding=(CACHE_BATCH,),
    )
    self.cache_ar_index = nnx.Cache(
        jnp.zeros((1,), dtype=jnp.int32),
        out_sharding=(),
    )

    self.cached_ar_key_scale = None
    self.cached_ar_value_scale = None

  def kv_cache_prefill(
      self,
      key: jax.Array,
      value: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      previous_chunk: Any = None,
  ) -> tuple[jax.Array, jax.Array, Optional[jax.Array]]:
    """Caches prefill KV states."""
    key_shaped = jnp.transpose(key, self.prefill_cache_axis_order)
    value_shaped = jnp.transpose(value, self.prefill_cache_axis_order)

    self.cached_prefill_key.set_value(key_shaped)
    self.cached_prefill_value.set_value(value_shaped)

    if decoder_segment_ids is not None and getattr(self, "cache_prefill_segment_id", None) is not None:
      self.cache_prefill_segment_id.set_value(decoder_segment_ids)

    return key, value, decoder_segment_ids

  def kv_cache_autoregressive(
      self,
      key: jax.Array,
      value: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
  ) -> tuple[tuple[jax.Array, jax.Array, Optional[jax.Array]], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Updates autoregressive cache with a single token and returns full caches."""
    _, sequence, _, _ = value.shape
    if sequence != 1:
      raise ValueError(f"Sequence length must be 1 during autoregression, got {sequence=}")

    one_token_key_shaped = jnp.transpose(key, self.ar_cache_axis_order)
    one_token_value_shaped = jnp.transpose(value, self.ar_cache_axis_order)

    ar_axis_names = transpose_tuple(self.cache_logical_axis_names, self.ar_cache_axis_order)
    update_axis = ar_axis_names.index(CACHE_SEQUENCE)
    ar_idx = jnp.squeeze(self.cache_ar_index.get_value())

    # Update key and value ring buffers
    new_k = jax.lax.dynamic_update_index_in_dim(self.cached_ar_key.get_value(), one_token_key_shaped, ar_idx, update_axis)
    new_v = jax.lax.dynamic_update_index_in_dim(self.cached_ar_value.get_value(), one_token_value_shaped, ar_idx, update_axis)
    self.cached_ar_key.set_value(new_k)
    self.cached_ar_value.set_value(new_v)

    # Update active sequence indicator
    active_indicator = jnp.zeros((self.batch, 1), dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    new_seg = jax.lax.dynamic_update_index_in_dim(self.cache_ar_segment_id.get_value(), active_indicator, ar_idx, 1)
    self.cache_ar_segment_id.set_value(new_seg)

    # Advance index and length counters
    max_ar_len = self.max_target_length - self.max_prefill_length
    self.cache_ar_index.set_value(jnp.mod(self.cache_ar_index.get_value() + 1, max_ar_len))
    self.cached_ar_lengths.set_value(self.cached_ar_lengths.get_value().at[:].add(1))

    # Retrieve unpacked prefill and AR cache states in standard (b, s, heads, dim) layout
    cached_prefill = (
        reverse_transpose(self.cached_prefill_key.get_value(), self.prefill_cache_axis_order),
        reverse_transpose(self.cached_prefill_value.get_value(), self.prefill_cache_axis_order),
        self.cache_prefill_segment_id.get_value(),
    )
    cached_ar = (
        reverse_transpose(self.cached_ar_key.get_value(), self.ar_cache_axis_order),
        reverse_transpose(self.cached_ar_value.get_value(), self.ar_cache_axis_order),
        self.cache_ar_segment_id.get_value(),
        self.cached_ar_lengths.get_value(),
    )
    return cached_prefill, cached_ar

  def __call__(
      self,
      key: jax.Array,
      value: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      model_mode: str = MODEL_MODE_PREFILL,
      previous_chunk: Any = None,
      **kwargs,
  ) -> tuple:
    """Updates KV cache states according to model_mode and returns cached states.

    Args:
      key: Input key tensor of shape [batch, seq, heads, dim].
      value: Input value tensor of shape [batch, seq, heads, dim].
      decoder_segment_ids: Optional tensor of segment IDs.
      model_mode: Operational mode ('prefill' or 'autoregressive').
      previous_chunk: Optional previous chunk cache for chunked prefill.
      **kwargs: Extra parameters.

    Returns:
      Tuple of (prefill_cache, ar_cache), where each cache is a tuple of cached states or None.
    """
    if model_mode == MODEL_MODE_PREFILL:
      return self.kv_cache_prefill(key, value, decoder_segment_ids, previous_chunk), None
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value, decoder_segment_ids)
    else:
      raise ValueError(f"Unsupported model_mode in KVCache: {model_mode}")
