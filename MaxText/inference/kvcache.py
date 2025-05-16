#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Implementation of the kvcache."""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from flax import linen as nn

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.aqt_tensor import QTensor as KVTensor
from aqt.jax.v2.flax import aqt_flax

from MaxText.common_types import Array, AxisNames, AxisIdxes, Config, CACHE_BATCH_PREFILL, DType, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE, CACHE_HEADS_NONE, DECODING_ACTIVE_SEQUENCE_INDICATOR
from MaxText.common_types import CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV, CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV


MAX_INT8 = 127.5
MAX_INT4 = 7.5
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)


def reverse_transpose(transposed_array, transpose_axis_order):
  return jax.numpy.moveaxis(transposed_array, (0, 1, 2, 3), transpose_axis_order)


def transpose_tuple(items: tuple[Any, ...], axis_order: AxisIdxes) -> tuple[Any, ...]:
  return tuple((items[i] for i in axis_order))


class KVQuant:
  """Class to configure quantization for KV cache."""

  axis_cfg = ""
  dtype = None

  def __init__(self, config: Config):
    assert config.quantize_kvcache
    self.axis_cfg = config.kv_quant_axis
    self.dtype = self._get_dtype(config.kv_quant_dtype)

  def _get_dtype(self, dtype_cfg: str):
    if dtype_cfg == "int4":
      return jnp.int4
    if dtype_cfg == "int8":
      return jnp.int8
    if dtype_cfg == "fp8":
      return jnp.float8_e4m3fn
    raise ValueError(f"Invalid kv_quant_dtype: {dtype_cfg}")

  def _get_max_axis(self, axis_names: AxisNames):
    if self.axis_cfg == "dkv":
      return axis_names.index(CACHE_KV)
    if self.axis_cfg == "heads_and_dkv":
      return (axis_names.index(CACHE_HEADS), axis_names.index(CACHE_KV))
    raise ValueError(f"Invalid KV quant axis cfg: {self.axis_cfg}")

  def quantize(self, kv: Array, axis_names: AxisNames):
    """Quantize key/values stored in kvcache."""
    assert self.axis_cfg, "KV quant axis cannot be None"
    max_axis = self._get_max_axis(axis_names)
    scale = jnp.max(jnp.abs(kv), axis=max_axis, keepdims=True)
    if self.dtype == jnp.int8:
      value = jnp.int8(jnp.rint(kv * (MAX_INT8 / scale)))
      return value, scale
    if self.dtype == jnp.int4:
      value = jnp.int4(jnp.rint(kv * (MAX_INT4 / scale)))
      return value, scale
    if self.dtype == jnp.float8_e4m3fn:
      value = jnp.float8_e4m3fn(kv * (E4M3_MAX / scale))
      return value, scale
    raise ValueError(f"Invalid KV quant dtype:{self.dtype}.")

  def einsum_fn_with_rhs_qtensor(
      self,
      kv: Array | aqt_tensor.QTensor,
      rhs_dequant_mode=None,
      rhs_calibration_mode=None,
      lhs_dequant_mode=None,
      lhs_calibration_mode=None,
  ):
    """einsum function where QTensor is the right-hand-side"""
    # Assumes kv is already quantized.
    einsum = jnp.einsum
    if isinstance(kv, aqt_tensor.QTensor):
      if kv.qvalue.dtype != jnp.float8_e4m3fn:
        num_bits = 4 if kv.qvalue.dtype == jnp.int4 else 8
        kv_cfg = aqt_config.dot_general_make(
            lhs_bits=None,
            rhs_bits=num_bits,
            bwd_bits=None,
            use_fwd_quant=False,
        )
      else:
        kv_cfg = aqt_config.config_fwd_fp8()

      if rhs_dequant_mode:
        aqt_config.set_fwd_dequant_mode(kv_cfg, rhs_dequant_mode=rhs_dequant_mode)
      if rhs_calibration_mode:
        aqt_config.set_fwd_calibration_mode(
            kv_cfg,
            rhs_calibration_mode=rhs_calibration_mode,
        )
      if lhs_dequant_mode:
        aqt_config.set_fwd_dequant_mode(kv_cfg, lhs_dequant_mode=lhs_dequant_mode)
      if lhs_calibration_mode:
        aqt_config.set_fwd_calibration_mode(
            kv_cfg,
            lhs_calibration_mode=lhs_calibration_mode,
        )
      einsum = aqt_flax.AqtEinsum(
          rhs_quant_mode=aqt_flax.QuantMode.TRAIN,
          lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
          rhs_freeze_mode=aqt_flax.FreezerMode.NONE,
          cfg=kv_cfg,
      )
    return einsum

  def einsum_fn_with_rhs_qtensor_and_dequant(self, value):
    """Get einstein summation for different dequant modes."""
    if self.dtype == jnp.float8_e4m3fn:
      return self.einsum_fn_with_rhs_qtensor(
          value,
          lhs_dequant_mode=aqt_config.DequantMode.THIS_INPUT,
          lhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
          rhs_dequant_mode=aqt_config.DequantMode.OTHER_INPUT,
          rhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
      )
    else:
      return self.einsum_fn_with_rhs_qtensor(
          value,
          rhs_dequant_mode=aqt_config.DequantMode.OTHER_INPUT,
          rhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
      )


class KVCache(nn.Module):
  """Implementation of the KVCache."""

  max_prefill_length: int
  max_target_length: int
  dtype: DType
  kv_quant: Optional[KVQuant] = None
  prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV)
  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  key_axis_order: AxisIdxes = (2, 0, 1, 3)
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

  def _get_prefill_cache_vars(self, batch, key_heads, value_heads, key_head_size, value_head_size, model_mode):
    """Get a shaped abstraction of the state"""

    cache_length = self.max_prefill_length
    dtype = self._get_cached_kv_dtype()

    if model_mode == MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.prefill_cache_axis_order)

    cache_logical_shape = (batch, cache_length, key_heads, key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape, self.prefill_cache_axis_order)

    cache_logical_shape = (batch, cache_length, value_heads, value_head_size)
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
    if model_mode == MODEL_MODE_PREFILL:
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
      cache_scale_axis_names = transpose_tuple(self.cache_scale_logical_axis_names, self.prefill_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, key_heads, cache_length)
      cache_key_scale_shape = transpose_tuple(cache_scale_logical_shape, self.prefill_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, value_heads, cache_length)
      cache_value_scale_shape = transpose_tuple(cache_scale_logical_shape, self.prefill_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_prefill_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_key_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_prefill_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_value_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var

  def _get_ar_cache_vars(self, batch, key_heads, value_heads, key_head_size, value_head_size, model_mode):
    """get ar cache vars"""

    dtype = self._get_cached_kv_dtype()
    if self.max_target_length <= self.max_prefill_length:
      raise ValueError(
          f"max_target_length: {self.max_target_length} should be greater than max_prefill_length:"
          f" {self.max_prefill_length}!"
      )
    cache_length = self.max_target_length - self.max_prefill_length

    if model_mode == MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.ar_cache_axis_order)

    cache_logical_shape = (batch, cache_length, key_heads, key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape, self.ar_cache_axis_order)

    cache_logical_shape = (batch, cache_length, value_heads, value_head_size)
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

    if model_mode == MODEL_MODE_PREFILL:
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
      cache_scale_axis_names = transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, key_heads, cache_length)
      cache_key_scale_shape = transpose_tuple(cache_scale_logical_shape, self.ar_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, value_heads, cache_length)
      cache_value_scale_shape = transpose_tuple(cache_scale_logical_shape, self.ar_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_ar_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_key_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_ar_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_value_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    cache_index_var = self.variable("cache", "cache_ar_index", nn.with_logical_partitioning(jnp.zeros, ()), (1,), jnp.int32)
    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var, cache_index_var, cached_lengths_var

  def kv_cache_chunked_prefill(
      self, key: Array, value: Array, decoder_segment_ids: Array, previous_chunk: Optional[Array] = None
  ):
    """Update the current kv cache into previous chunk and return needed length.

    The previous chunk kv cache should be in the model's param.

    Prefill cache need to be max prefill length to prevent different shape of kv cache.
    Different shape of kv cache in previous chunk could produce different compiled graph.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      decoder_segment_ids: [b, s] -- marking segment ids for tokens
      previous_chunk:
        In shape [b, s]. The tokens without padding in previous chunk.
        Use to preserve the previous kv cache.

    Returns:
      key, value, decoder_segment_id.
    """

    assert not self.kv_quant, "Not support kv_quant now."
    batch, key_seq_len, key_heads, key_head_size = key.shape
    batch, value_seq_len, value_heads, value_head_size = value.shape
    if decoder_segment_ids is not None:
      batch, segment_id_seq_len = decoder_segment_ids.shape
      assert key_seq_len == segment_id_seq_len, f"{key_seq_len=}, {segment_id_seq_len=} should match."

    assert key.dtype == value.dtype, "Key and Value Dtypes should match."
    assert key_seq_len == value_seq_len, f"{key_seq_len=}, {value_seq_len=} should match."

    next_pos = 0
    if previous_chunk is not None:
      # We only have 1 prompt in prefill mode.
      next_pos = previous_chunk.shape[1]

    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, key_heads, value_heads, key_head_size, value_head_size, MODEL_MODE_PREFILL
    )
    # TODO: Find a way to not enable the ar cache for prefill mode.
    _ = self._get_ar_cache_vars(
        batch, key_heads, value_heads, key_head_size, value_head_size, MODEL_MODE_PREFILL
    )  # initialize it now

    key_shaped_for_cache = jnp.transpose(key, self.prefill_cache_axis_order)
    value_shaped_for_cache = jnp.transpose(value, self.prefill_cache_axis_order)

    # For quantized kv cached. Could be get without transpose twice.
    cached_key = self.get_cached_values(cached_prefill_key_vars, key.dtype, self.prefill_cache_axis_order)
    cached_value = self.get_cached_values(cached_prefill_value_vars, value.dtype, self.prefill_cache_axis_order)
    cached_key_value = jnp.transpose(cached_key, self.prefill_cache_axis_order)
    cached_value_value = jnp.transpose(cached_value, self.prefill_cache_axis_order)

    seq_axis = self.prefill_cache_logical_axis_names.index(CACHE_SEQUENCE)
    cache_seq_axis = self.prefill_cache_axis_order.index(seq_axis)

    assert next_pos + key_shaped_for_cache.shape[cache_seq_axis] <= self.max_prefill_length, (
        f"Previous kv cache[{next_pos}] + "
        f"current kv cache[{key_shaped_for_cache.shape[cache_seq_axis]}] "
        f"> max length[{self.max_prefill_length}]"
    )

    # We don't zero out remain values. Use segment id to mask out.
    cached_prefill_key_vars[0].value = jax.lax.dynamic_update_slice_in_dim(
        cached_key_value, key_shaped_for_cache, next_pos, cache_seq_axis
    )
    cached_prefill_value_vars[0].value = jax.lax.dynamic_update_slice_in_dim(
        cached_value_value, value_shaped_for_cache, next_pos, cache_seq_axis
    )

    if decoder_segment_ids is not None:
      # Need zero out the remain values to prevent wrong mask in autoregressive.
      previous_segment_id = cached_prefill_segment_id_var.value[:, :next_pos]
      cached_prefill_segment_id_var.value = jnp.zeros_like(cached_prefill_segment_id_var.value, dtype=jnp.int32)
      cached_prefill_segment_id_var.value = jax.lax.dynamic_update_slice_in_dim(
          cached_prefill_segment_id_var.value, previous_segment_id, start_index=0, axis=1
      )
      cached_prefill_segment_id_var.value = jax.lax.dynamic_update_slice_in_dim(
          cached_prefill_segment_id_var.value, decoder_segment_ids, next_pos, axis=1
      )

    # Return needed kv cache to reduce computation of attention.
    needed_prefill_key_value = jax.lax.dynamic_slice_in_dim(
        cached_prefill_key_vars[0].value, start_index=0, slice_size=(next_pos + key_seq_len), axis=cache_seq_axis
    )
    needed_prefill_value_value = jax.lax.dynamic_slice_in_dim(
        cached_prefill_value_vars[0].value, start_index=0, slice_size=(next_pos + value_seq_len), axis=cache_seq_axis
    )
    needed_segment_id = None
    if decoder_segment_ids is not None:
      needed_segment_id = jax.lax.dynamic_slice_in_dim(
          cached_prefill_segment_id_var.value, start_index=0, slice_size=(next_pos + segment_id_seq_len), axis=1
      )

    return (
        jnp.transpose(needed_prefill_key_value, self.key_axis_order),
        jnp.transpose(needed_prefill_value_value, self.key_axis_order),
        needed_segment_id,
    )

  def kv_cache_prefill(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
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

    batch, _, key_heads, key_head_size = key.shape
    batch, _, value_heads, value_head_size = value.shape
    assert key.dtype == value.dtype, "Key and Value Dtypes should match."

    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, key_heads, value_heads, key_head_size, value_head_size, MODEL_MODE_PREFILL
    )
    # TODO: Find a way to not enable the ar cache for prefill mode.
    _ = self._get_ar_cache_vars(
        batch, key_heads, value_heads, key_head_size, value_head_size, MODEL_MODE_PREFILL
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
      assert cached_key_scale_var is not None, "cached_key_scale_var cannot be None"
      assert cached_value_scale_var is not None, "cached_value_scale_var cannot be None"
      cached_key_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_scale_var.value, one_token_key_scale_shaped_for_cache, ar_cache_update_idx, ar_cache_scale_update_axis
      )
      cached_value_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_scale_var.value,
          one_token_value_scale_shaped_for_cache,
          ar_cache_update_idx,
          ar_cache_scale_update_axis,
      )

  def get_cached_values(self, cache_vars, target_dtype, cache_axis_order) -> jax.Array | KVTensor:
    """get cached values"""
    cache_var, cache_scale_var = cache_vars
    cache_value = cache_var.value
    if cache_scale_var is not None:
      scale_value = cache_scale_var.value
      dtype = cache_value.dtype
      if dtype == jnp.int8:
        scale_value /= MAX_INT8
      elif dtype == jnp.int4:
        scale_value /= MAX_INT4
      elif dtype == jnp.float8_e4m3fn:
        scale_value /= E4M3_MAX

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
    batch, sequence, key_heads, key_head_size = key.shape
    batch, sequence, value_heads, value_head_size = value.shape

    if sequence != 1:
      raise ValueError(f"Sequence length should be 1 during autoregression, got {sequence=}")

    cached_ar_key_vars, cached_ar_value_vars, cached_ar_segment_id_var, cache_ar_index_var, cache_ar_lengths_var = (
        self._get_ar_cache_vars(batch, key_heads, value_heads, key_head_size, value_head_size, MODEL_MODE_AUTOREGRESSIVE)
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
    active_indicator = jnp.zeros((batch, 1), dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    cached_ar_segment_id_var.value = jax.lax.dynamic_update_index_in_dim(
        cached_ar_segment_id_var.value, active_indicator, jnp.squeeze(cache_ar_index_var.value), 1
    )
    cache_ar_index_var.value = jnp.mod(cache_ar_index_var.value + 1, self.max_target_length - self.max_prefill_length)
    cache_ar_lengths_var.value = cache_ar_lengths_var.value.at[:].add(1)

    # The below retrieves the existing prefill cache variables, not creating new ones
    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, key_heads, value_heads, key_head_size, value_head_size, MODEL_MODE_AUTOREGRESSIVE
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
    if model_mode == MODEL_MODE_PREFILL:
      if self.use_chunked_prefill:
        return self.kv_cache_chunked_prefill(key, value, decoder_segment_ids, previous_chunk), None
      else:
        return self.kv_cache_prefill(key, value, decoder_segment_ids), None
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value, use_ragged_attention)
    else:
      raise ValueError(f"Model Mode isn't supported! {model_mode=}")


class MlaKVCache(KVCache):
  """Implementation of the KVCache for MLA."""

  prefill_cache_logical_axis_names: AxisNames = (
      CACHE_BATCH_PREFILL,
      CACHE_SEQUENCE,
      CACHE_HEADS_NONE,
      CACHE_KV,
  )
  cache_logical_axis_names: AxisNames = (
      CACHE_BATCH,
      CACHE_SEQUENCE,
      CACHE_HEADS_NONE,
      CACHE_KV,
  )

  def key_latent_add_head_dim(self, key_latent: Array):
    b, l, hz = key_latent.shape
    return key_latent.reshape(b, l, 1, hz)

  def key_latent_remove_head_dim(self, key_latent: Array):
    b, l, _, hz = key_latent.shape
    return key_latent.reshape(b, l, hz)

  @nn.compact
  def __call__(
      self,
      key_latent: Array,
      key_rope: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      use_ragged_attention: bool = False,
      previous_chunk: Any = None,
  ) -> Tuple[
      Optional[Tuple[Array, Array, Array]],
      Optional[Tuple[Array, Array, Array, Array]],
  ]:
    assert model_mode != MODEL_MODE_TRAIN, "incorrectly updating kvcache in train mode."
    assert self.kv_quant is None, "kvcache quantization not supported with mla."
    key_latent = self.key_latent_add_head_dim(key_latent)
    prefill_cache, ar_cache = super().__call__(key_latent, key_rope, decoder_segment_ids, model_mode)
    if prefill_cache:
      key_latent, key_rope, decoder_segments_ids = prefill_cache
      prefill_cache = (
          self.key_latent_remove_head_dim(key_latent),
          key_rope,
          decoder_segments_ids,
      )
    if ar_cache:
      key_latent, key_rope, decoder_segments_ids, lengths = ar_cache
      ar_cache = (
          self.key_latent_remove_head_dim(key_latent),
          key_rope,
          decoder_segments_ids,
          lengths,
      )
    return prefill_cache, ar_cache
