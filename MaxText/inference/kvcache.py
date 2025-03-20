import enum
import functools
import math
from typing import Any, Optional, Tuple
from flax import linen as nn
from flax.linen import partitioning
import jax
import jax.numpy as jnp
import common_types
from layers import quantizations

Array = common_types.Array
AxisNames = common_types.AxisNames
AxisIdxes = common_types.AxisIdxes
KVQuant = quantizations.KVQuant
KVTensor = quantizations.KVTensor

CACHE_BATCH_PREFILL = common_types.CACHE_BATCH_PREFILL
CACHE_BATCH = common_types.CACHE_BATCH
CACHE_SEQUENCE = common_types.CACHE_SEQUENCE
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV
CACHE_SCALE_BATCH = common_types.CACHE_SCALE_BATCH
CACHE_SCALE_SEQUENCE = common_types.CACHE_SCALE_SEQUENCE
CACHE_SCALE_HEADS = common_types.CACHE_SCALE_HEADS
CACHE_SCALE_KV = common_types.CACHE_SCALE_KV


def reverse_transpose(transposed_array, transpose_axis_order):
  return jax.numpy.moveaxis(transposed_array, (0, 1, 2, 3), transpose_axis_order)


def transpose_tuple(items: tuple[Any, Any, Any, Any], axis_order: AxisIdxes) -> tuple[Any, Any, Any, Any]:
  return tuple([items[i] for i in axis_order])


class KVCache(nn.Module):
  max_prefill_length: int
  max_target_length: int
  dtype: common_types.DType
  kv_quant: Optional[KVQuant] = None
  prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV)
  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  use_chunked_prefill: bool = False

  def _get_cached_kv_dtype(self):
    return self.kv_quant.dtype if self.kv_quant else self.dtype

  def _get_cache_scale_logical_shape(self, batch, heads, cache_length):
    assert self.kv_quant
    if self.kv_quant.axis_cfg == "dkv":
      return (batch, cache_length, heads, 1)
    if self.kv_quant.axis_cfg == "heads_and_dkv":
      return (batch, cache_length, 1, 1)
    raise f"Invalid config for kv_quant_axis:{self.kv_quant.axis_cfg}"

  def _get_prefill_cache_vars(self, batch, heads, key_head_size, value_head_size, model_mode):

    cache_length = self.max_prefill_length
    dtype = self._get_cached_kv_dtype()

    if model_mode == common_types.MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.prefill_cache_axis_order)

    cache_logical_shape = (batch, cache_length, heads, key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape, self.prefill_cache_axis_order)

    cache_logical_shape = (batch, cache_length, heads, value_head_size)
    cache_shape_value = transpose_tuple(cache_logical_shape, self.prefill_cache_axis_order)

    cached_key_var = self.variable(
        "cache",
        "cached_prefill_key",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape_key,
        dtype,
    )
    cached_value_var = self.variable(
        "cache",
        "cached_prefill_value",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape_value,
        dtype,
    )
    if model_mode == common_types.MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)

    cached_segment_id_var = self.variable(
        "cache",
        "cache_prefill_segment_id",
        nn.with_logical_partitioning(jnp.zeros, segment_id_axis_names),
        (cache_logical_shape[0], cache_length),
        jnp.int32,
    )

    if self.kv_quant:
      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, heads, cache_length)
      cache_scale_axis_names = transpose_tuple(self.cache_scale_logical_axis_names, self.prefill_cache_axis_order)
      cache_scale_shape = transpose_tuple(cache_scale_logical_shape, self.prefill_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_prefill_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_prefill_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var

  def _get_ar_cache_vars(self, batch, heads, key_head_size, value_head_size, model_mode):

    dtype = self._get_cached_kv_dtype()
    cache_length = self.max_target_length - self.max_prefill_length

    if model_mode == common_types.MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.ar_cache_axis_order)

    cache_logical_shape = (batch, cache_length, heads, key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape, self.ar_cache_axis_order)

    cache_logical_shape = (batch, cache_length, heads, value_head_size)
    cache_shape_value = transpose_tuple(cache_logical_shape, self.ar_cache_axis_order)

    # TODO(b/339703100): investigate the issue why with_logical_partitioning doesn't enforce sharding
    cached_key_var = self.variable(
        "cache",
        "cached_ar_key",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape_key,
        dtype,
    )
    cached_key_var.value = nn.with_logical_constraint(
        cached_key_var.value,
        cache_axis_names,
    )

    cached_value_var = self.variable(
        "cache",
        "cached_ar_value",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape_value,
        dtype,
    )
    cached_value_var.value = nn.with_logical_constraint(
        cached_value_var.value,
        cache_axis_names,
    )

    if model_mode == common_types.MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)
    cached_segment_id_var = self.variable(
        "cache",
        "cache_ar_segment_id",
        nn.with_logical_partitioning(jnp.zeros, segment_id_axis_names),
        (cache_logical_shape[0], cache_length),
        jnp.int32,
    )

    cached_lengths_var = self.variable(
        "cache",
        "cached_ar_lengths",
        nn.with_logical_partitioning(jnp.zeros, (CACHE_BATCH,)),
        (cache_logical_shape[0],),
        jnp.int32,
    )

    if self.kv_quant:
      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, heads, cache_length)
      cache_scale_axis_names = transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)
      cache_scale_shape = transpose_tuple(cache_scale_logical_shape, self.ar_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_ar_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_ar_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    cache_index_var = self.variable("cache", "cache_ar_index", nn.with_logical_partitioning(jnp.zeros, ()), (1,), jnp.int32)
    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var, cache_index_var, cached_lengths_var

  def chunked_prefill_kv_cache(self, key: Array, value: Array, decoder_segment_ids: Array, previous_chunk: Any = None):
    """
    function returns appropriate prefill_cache if there is previous_chunk already processed
    if no pervious chunk is processed,
    function returns a cache that has non zero first chunk part of key and value

    else
    function updates current key and value at the desired location and returns entire key and value

    """
    batch, _, heads, key_head_size = key.shape
    batch, _, heads, value_head_size = value.shape

    assert key.dtype == value.dtype, "Key and Value Dtypes should match."

    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, heads, key_head_size, value_head_size, common_types.MODEL_MODE_PREFILL
    )
    # TODO: Find a way to not enable the ar cache for prefill mode.
    _ = self._get_ar_cache_vars(
        batch, heads, key_head_size, value_head_size, common_types.MODEL_MODE_PREFILL
    )  # initialize it now

    key_shaped_for_cache = jnp.transpose(key, self.prefill_cache_axis_order)
    value_shaped_for_cache = jnp.transpose(value, self.prefill_cache_axis_order)

    next_pos = 0
    if previous_chunk != None:
      """
      if there is previous chunk information present,
        1. Fetch the cached key, value
        2. Update current key value at the desired position.
        3. take transpose before returning as that is how the attention op expects the key and value
      """
      next_pos = previous_chunk["true_length_array"].shape[1]
      cached_key = self.get_cached_values(cached_prefill_key_vars, key.dtype, self.prefill_cache_axis_order)
      cached_value = self.get_cached_values(cached_prefill_value_vars, value.dtype, self.prefill_cache_axis_order)
      cached_key_value = jnp.transpose(cached_key, self.prefill_cache_axis_order)
      cached_value_value = jnp.transpose(cached_value, self.prefill_cache_axis_order)

      cached_prefill_key_vars[0].value = jax.lax.dynamic_update_slice(
          cached_key_value, key_shaped_for_cache, (next_pos, 0, 0, 0)
      )

      cached_prefill_value_vars[0].value = jax.lax.dynamic_update_slice(
          cached_value_value, value_shaped_for_cache, (next_pos, 0, 0, 0)
      )
      cached_prefill_segment_id_var.value = decoder_segment_ids
      return (
          jnp.transpose(cached_prefill_key_vars[0].value, self.key_axis_order),
          jnp.transpose(cached_prefill_value_vars[0].value, self.key_axis_order),
          cached_prefill_segment_id_var.value,
      )
    else:
      """
      if there is previous chunk information present,
        1. Fetch the cached key, value
        2. Update current key value at the desired position. (beginning - (0,0,0,0))
        3. take transpose before returning as that is how the attention op expects the key and value

      """
      cached_prefill_key_vars[0].value = jax.lax.dynamic_update_slice(
          cached_prefill_key_vars[0].value, key_shaped_for_cache, (next_pos, 0, 0, 0)
      )
      cached_prefill_value_vars[0].value = jax.lax.dynamic_update_slice(
          cached_prefill_value_vars[0].value, value_shaped_for_cache, (next_pos, 0, 0, 0)
      )
      cached_prefill_segment_id_var.value = decoder_segment_ids
      return (
          jnp.transpose(cached_prefill_key_vars[0].value, self.key_axis_order),
          jnp.transpose(cached_prefill_value_vars[0].value, self.key_axis_order),
          cached_prefill_segment_id_var.value,
      )

  def kv_cache_prefill(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      previous_chunk: Any = None,
  ):
    """In prefill mode, we zero out the existing cache, run the computation and
    prepare the cache as necessary.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      decoder_segment_ids: [b, s] -- marking segment ids for tokens

    Returns:
      key, value, decoder_segment_id.

    """
    if self.use_chunked_prefill:
      return self.chunked_prefill_kv_cache(key, value, decoder_segment_ids, previous_chunk)

    batch, _, heads, key_head_size = key.shape
    batch, _, heads, value_head_size = value.shape
    assert key.dtype == value.dtype, "Key and Value Dtypes should match."

    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, heads, key_head_size, value_head_size, common_types.MODEL_MODE_PREFILL
    )
    # TODO: Find a way to not enable the ar cache for prefill mode.
    _ = self._get_ar_cache_vars(
        batch, heads, key_head_size, value_head_size, common_types.MODEL_MODE_PREFILL
    )  # initialize it now

    key_shaped_for_cache = jnp.transpose(key, self.prefill_cache_axis_order)
    value_shaped_for_cache = jnp.transpose(value, self.prefill_cache_axis_order)

    if self.kv_quant:
      prefill_key_axis_names = transpose_tuple(self.cache_logical_axis_names, self.prefill_cache_axis_order)
      key_shaped_for_cache, key_scale_shaped_for_cache = self.kv_quant.quantize(key_shaped_for_cache, prefill_key_axis_names)
      value_shaped_for_cache, value_scale_shaped_for_cache = self.kv_quant.quantize(
          value_shaped_for_cache, prefill_key_axis_names
      )
      cached_prefill_key_vars[1].value = key_scale_shaped_for_cache
      cached_prefill_value_vars[1].value = value_scale_shaped_for_cache

    cached_prefill_key_vars[0].value = key_shaped_for_cache
    cached_prefill_value_vars[0].value = value_shaped_for_cache

    if decoder_segment_ids is not None:
      cached_prefill_segment_id_var.value = decoder_segment_ids
    return key, value, decoder_segment_ids

  def update_ar_key_value(
      self,
      one_token_key: Array,
      one_token_value: Array,
      cached_key_vars: tuple[nn.Variable, nn.Variable | None],
      cached_value_vars: tuple[nn.Variable, nn.Variable | None],
      one_hot_indices: Array,
      lengths: Array,
      use_ragged_attention: bool,
  ) -> None:
    """Adds a single token's results to the ar kv cache

    Args:
        one_token_key (Array): Key of one token to add to the cache
        one_token_value (Array): Value of one token to add to the cache
        cached_ar_key (tuple[nn.Variable, nn.Variable|None],): Cached keys to add new token key to, possibly with scale
        cached_ar_value (tuple[nn.Variable, nn.Variable|None],: Cached values to add new token value to, possible with scale
        one_hot_indices (Array): Location of the new token within the cache

    Returns:
        tuple[Array, Array]: Updated caches for key and value with new token info added
    """

    cached_key_var, cached_key_scale_var = cached_key_vars
    cached_value_var, cached_value_scale_var = cached_value_vars

    # In order to update the key, value caches with the current key and
    # value, we reshape the one_token_key and one_token_value
    one_token_key_shaped_for_cache = jnp.transpose(one_token_key, self.ar_cache_axis_order)
    one_token_value_shaped_for_cache = jnp.transpose(one_token_value, self.ar_cache_axis_order)

    ar_cache_axis_names = transpose_tuple(self.cache_logical_axis_names, self.ar_cache_axis_order)
    if self.kv_quant:
      one_token_key_shaped_for_cache, one_token_key_scale_shaped_for_cache = self.kv_quant.quantize(
          one_token_key_shaped_for_cache, ar_cache_axis_names
      )
      one_token_value_shaped_for_cache, one_token_value_scale_shaped_for_cache = self.kv_quant.quantize(
          one_token_value_shaped_for_cache, ar_cache_axis_names
      )

    ar_cache_update_idx = jnp.squeeze(one_hot_indices)
    ar_cache_sequence_axis = ar_cache_update_axis = ar_cache_axis_names.index(CACHE_SEQUENCE)
    ar_cache_batch_axis = ar_cache_axis_names.index(CACHE_BATCH)

    if use_ragged_attention:
      cache_locations = [slice(None)] * 4
      new_token_locations = [slice(None)] * 4
      new_token_locations[ar_cache_sequence_axis] = 0

      def key_body(i, val):
        cache_locations[ar_cache_batch_axis] = i
        cache_locations[ar_cache_sequence_axis] = lengths[i]
        new_token_locations[ar_cache_batch_axis] = i
        return val.at[tuple(cache_locations)].set(one_token_key_shaped_for_cache[tuple(new_token_locations)])

      def value_body(i, val):
        cache_locations[ar_cache_batch_axis] = i
        cache_locations[ar_cache_sequence_axis] = lengths[i]
        new_token_locations[ar_cache_batch_axis] = i
        return val.at[tuple(cache_locations)].set(one_token_value_shaped_for_cache[tuple(new_token_locations)])

      cached_key_var.value = jax.lax.fori_loop(
          0, one_token_key_shaped_for_cache.shape[0], key_body, cached_key_var.value, unroll=8
      )
      cached_value_var.value = jax.lax.fori_loop(
          0, one_token_value_shaped_for_cache.shape[0], value_body, cached_value_var.value, unroll=8
      )

    else:
      one_hot_indices = one_hot_indices.astype(int)
      cached_key_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_var.value, one_token_key_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis
      )
      cached_value_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_var.value, one_token_value_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis
      )
    cached_key_var.value = nn.with_logical_constraint(cached_key_var.value, ar_cache_axis_names)
    cached_value_var.value = nn.with_logical_constraint(cached_value_var.value, ar_cache_axis_names)

    if self.kv_quant:
      ar_cache_scale_axis_names = transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)
      ar_cache_scale_update_axis = ar_cache_scale_axis_names.index(CACHE_SCALE_SEQUENCE)
      cached_key_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_scale_var.value, one_token_key_scale_shaped_for_cache, ar_cache_update_idx, ar_cache_scale_update_axis
      )
      cached_value_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_scale_var.value,
          one_token_value_scale_shaped_for_cache,
          ar_cache_update_idx,
          ar_cache_scale_update_axis,
      )
    return

  def get_cached_values(self, cache_vars, target_dtype, cache_axis_order) -> jax.Array | KVTensor:
    cache_var, cache_scale_var = cache_vars
    cache_value = cache_var.value
    if cache_scale_var is not None:
      scale_value = cache_scale_var.value
      dtype = cache_value.dtype
      if dtype == jnp.int8:
        scale_value /= quantizations.MAX_INT8
      elif dtype == jnp.int4:
        scale_value /= quantizations.MAX_INT4
      elif dtype == jnp.float8_e4m3fn:
        scale_value /= quantizations.E4M3_MAX

      cache_value = KVTensor(qvalue=cache_value, scale=[scale_value], scale_t=None, dequant_dtype=target_dtype, bias=[])
    cache_value_in_logical_shape = jax.tree.map(lambda x: reverse_transpose(x, cache_axis_order), cache_value)
    return cache_value_in_logical_shape

  def kv_cache_autoregressive(
      self,
      key: Array,
      value: Array,
      use_ragged_attention: bool = False,
  ):
    """In autoregressive mode, we update the cache for this entry and
       then return the full cache.

    Args:
      key: in shape [b, 1, n, d].
      value: in shape [b, 1, n, d].
      decoder_segment_ids: [b, 1] -- marking segment ids for tokens

    Returns:
      tuple of (key, value, segment_id) for both prefill and ar cache,
    Raises:
      ValueError: when key/value shape is not [batch, 1, num_heads, heads_dim].
    """
    batch, sequence, heads, key_head_size = key.shape
    batch, sequence, heads, value_head_size = value.shape

    if sequence != 1:
      raise ValueError(f"Sequence length should be 1 during autoregression, got {sequence=}")

    cached_ar_key_vars, cached_ar_value_vars, cached_ar_segment_id_var, cache_ar_index_var, cache_ar_lengths_var = (
        self._get_ar_cache_vars(batch, heads, key_head_size, value_head_size, common_types.MODEL_MODE_AUTOREGRESSIVE)
    )

    self.update_ar_key_value(
        key,
        value,
        cached_ar_key_vars,
        cached_ar_value_vars,
        cache_ar_index_var.value,
        cache_ar_lengths_var.value,
        use_ragged_attention,
    )
    active_indicator = jnp.zeros((batch, 1), dtype=jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    cached_ar_segment_id_var.value = jax.lax.dynamic_update_index_in_dim(
        cached_ar_segment_id_var.value, active_indicator, jnp.squeeze(cache_ar_index_var.value), 1
    )
    cache_ar_index_var.value = jnp.mod(cache_ar_index_var.value + 1, self.max_target_length - self.max_prefill_length)
    cache_ar_lengths_var.value = cache_ar_lengths_var.value.at[:].add(1)

    # The below retrieves the existing prefill cache variables, not creating new ones
    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, heads, key_head_size, value_head_size, common_types.MODEL_MODE_AUTOREGRESSIVE
    )

    cached_prefill = (
        self.get_cached_values(cached_prefill_key_vars, key.dtype, self.prefill_cache_axis_order),
        self.get_cached_values(cached_prefill_value_vars, value.dtype, self.prefill_cache_axis_order),
        cached_prefill_segment_id_var.value,
    )

    cached_ar = (
        self.get_cached_values(cached_ar_key_vars, key.dtype, self.ar_cache_axis_order),
        self.get_cached_values(cached_ar_value_vars, value.dtype, self.ar_cache_axis_order),
        cached_ar_segment_id_var.value,
        cache_ar_lengths_var.value,
    )
    return cached_prefill, cached_ar

  @nn.compact
  def __call__(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      use_ragged_attention: bool = False,
      previous_chunk: Any = None,
  ) -> tuple:
    """KV cache takes the current state and updates the state accordingly.

    The key and value have dimension [b, s, n_kv, d],
    but we cache them with a reshape as defined in *_axis_order config as a TPU
    fusion optimization. This also enables the "scatter via one-hot
    broadcast" trick, which means we do a one-hot broadcast instead of a
    scatter/gather operations, resulting in a 3-4x speedup in practice.

    Args:
      key: in shape [b, s, n_kv, d].
      value: in shape [b, s, n_kv, d].
      model_mode: model mode controlling model

    Returns:
      two tuples of (k, v, decoder_segments) -- either can be Nones

    """
    if model_mode == common_types.MODEL_MODE_PREFILL:
      return self.kv_cache_prefill(key, value, decoder_segment_ids, previous_chunk), None
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value, use_ragged_attention)
    else:
      raise ValueError(f"Model Mode isn't supported! {model_mode=}")
