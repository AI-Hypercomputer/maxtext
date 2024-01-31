#  Copyright 2023 Google LLC
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

"""Attentions Layers."""

import functools
import math
from typing import Optional, Sequence

from flax import linen as nn
import jax
from jax import lax
from jax import random
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.experimental.pallas.ops import attention as pallas_attention
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention
import jax.numpy as jnp

import common_types
from layers import embeddings
from layers import initializers
from layers import linears
from layers import quantizations

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
PRNGKey = common_types.PRNGKey

DenseGeneral = linears.DenseGeneral
RotaryEmbedding = embeddings.RotaryEmbedding
NdInitializer = initializers.NdInitializer

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


nd_dense_init = initializers.nd_dense_init
shard_map = shard_map.shard_map

dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))

def exp2(x):
  two = jnp.float32(2.71828)

  return lax.pow(two.astype(x.dtype), x)

def _maybe_aqt_einsum(int8_training, aqt_rng):
  """Maybe overwrite dot general with aqt_dot_general."""
  if not int8_training:
    return jnp.einsum
  else:
    aqt_dot_general = quantizations.int8_dot_general(aqt_rng)
    return functools.partial(jnp.einsum, _dot_general=aqt_dot_general)

class AttentionOp(nn.Module):
  mesh: Mesh
  attention_kernel: str
  max_target_length: int
  use_int8: bool
  num_query_heads: int
  num_kv_heads: int
  max_prefill_predict_length: int = -1 
  float32_logits: bool = False
  flash_axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  dtype: DType = jnp.float32

  def check_attention_inputs(
    self,
    query: Array,
    key: Array,
    value: Array) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, 'k, v must have same rank.'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        'q, k, v batch dims must match.')
    assert key.shape[-2] == value.shape[-2], ('k, v num_kv_heads must match.')
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Following Pallas MHA Flash Attention Reference.
  # https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
  # This mask models (1) separate sequences (decoder_segment_ids) and (2) causality
  def generate_attention_mask(
      self,
      query,
      key,
      decoder_segment_ids: Array | None,
      model_mode: str
  ) -> Array | None:
    mask = None
    if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      mask = decoder_segment_ids[:, None, None, None, :] == common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    elif decoder_segment_ids is not None:
      mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      mask = mask[:, None, None,:, :]

    causal_mask = None
    # We enforce causality except for AUTOREGRESSION
    if model_mode != common_types.MODEL_MODE_AUTOREGRESSIVE:
      _, q_seq_len, _, _ = query.shape
      _, kv_seq_len, _, _ = key.shape
      mask_shape = (q_seq_len, kv_seq_len)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      causal_mask = (col_ids <= row_ids)[None, None, None, :, :]

    if (mask is not None) and (causal_mask is not None):
      output_mask = jnp.logical_and(mask, causal_mask)
    elif mask is not None:
      output_mask = mask
    elif causal_mask is not None:
      output_mask = causal_mask
    else:
      output_mask = None

    return jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE) if output_mask is not None else None
  
  def apply_attention(self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str) -> Array:
    self.check_attention_inputs(query, key, value)
    if self.attention_kernel == "dot_product":
      return self.apply_attention_dot(query, key, value, decoder_segment_ids, model_mode)
    elif self.attention_kernel == 'flash':
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError("""Decode not supported with flash attention.
                            Use `dot_product` instead.""")
      return self.tpu_flash_attention(query, key, value, decoder_segment_ids), None, None
    elif self.attention_kernel == 'gpu_flash_xla' or self.attention_kernel == 'gpu_flash_triton':
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError("""Decode not supported with flash attention.
                            Use `dot_product` instead.""")
      return self.gpu_flash_attention(query, key, value), None, None
    else:
      raise ValueError(f'Unexpected attention kernel {self.attention_kernel=}.')

  def tpu_flash_attention(
    self,
    query: Array,
    key: Array,
    value: Array,
    decoder_segment_ids: Array | None) -> Array:
    """TPU Flash Attention."""
    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))
    if not(query.shape[1] == key.shape[1] == value.shape[1]):
      raise ValueError(f"The flash attention kernel requires Q, K and V to have the same number of heads"
                       "{query.shape=} {key.shape=}, {value.shape=}")

    if decoder_segment_ids is not None:
      decoder_segment_ids = tpu_flash_attention.SegmentIds(
          decoder_segment_ids, decoder_segment_ids
      )
    axis_names = nn.logical_to_mesh_axes(self.flash_axis_names)
    segment_axis_names = nn.logical_to_mesh_axes(
        (BATCH, 'activation_length_no_heads')
    )

    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(
            axis_names,
            axis_names,
            axis_names,
            segment_axis_names,
        ),
        out_specs=axis_names,
        check_rep=False,
    )
    def wrap_flash_attention(query, key, value, decoder_segment_ids):
      if decoder_segment_ids is not None:
        assert (
            query.shape[2]
            == decoder_segment_ids.q.shape[1]
        ), 'Sharding along sequence dimension not allowed in flash attention'
      assert query.shape[2] >= 128, "Flash only supports seq_len >= 128"
      return tpu_flash_attention.flash_attention(
          query,
          key,
          value,
          causal=True,
          segment_ids=decoder_segment_ids,
          block_sizes=tpu_flash_attention.BlockSizes(
              block_q=min(512, query.shape[2]),
              block_k_major=min(512, key.shape[2]),
              block_k=min(512, key.shape[2]),
              block_b=min(2, query.shape[0]),
              block_q_major_dkv=min(512, query.shape[2]),
              block_k_major_dkv=min(512, key.shape[2]),
              block_q_dkv=min(512, query.shape[2]),
              block_k_dkv=min(512, key.shape[2]),
              block_q_dq=min(1024, query.shape[2]),
              block_k_dq=min(256, key.shape[2]),
              block_k_major_dq=min(512, key.shape[2]),
          ),
      )

    devices_in_data_fsdp = self.mesh.shape['data'] * self.mesh.shape['fsdp']
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        'Batch dimension should be shardable among the devices in data and fsdp'
        ' axis'
    )
    x = wrap_flash_attention(query, key, value, decoder_segment_ids)
    x = jnp.transpose(x, axes=(0, 2, 1, 3))
    return x

  def gpu_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
  ) -> Array:
    """GPU Flash Attention."""
    b, n, s, h = key.shape # pylint: disable=unused-variable
    if self.attention_kernel == "gpu_flash_xla":
      bwd_pass_impl = "xla"
    elif self.attention_kernel == "gpu_flash_triton":
      bwd_pass_impl = "triton"
    else:
      raise ValueError(f"Can't convert {self.attention_kernel } to a bwd_pass_impl")
  
    axis_names = nn.logical_to_mesh_axes(self.flash_axis_names)
    segment_axis_names = nn.logical_to_mesh_axes((BATCH, LENGTH))

    @functools.partial(
        shard_map,
        mesh = self.mesh,
        in_specs = (
          axis_names,
          axis_names,
          axis_names,
          segment_axis_names),
        out_specs = axis_names,
        check_rep=False)
    def wrap_gpu_flash_attention(query, key, value):
      return pallas_attention.mha(
        query, key, value, sm_scale=1.0 / math.sqrt(h), backward_pass_impl=bwd_pass_impl,
        num_stages = 1, causal = True, segment_ids = None
      )
    return wrap_gpu_flash_attention(query, key, value)

  def apply_attention_dot(
      self,
      query: Array, 
      key: Array,   
      value: Array, 
      decoder_segment_ids: Array | None,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
  ) -> Array:
    """Apply Attention."""
    aqt_rng = self.make_rng('aqt')

    if self.float32_logits:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    attn_weights = self.qk_product(query, key)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
    if attn_mask is not None:
      attn_weights += attn_mask
    
    # TODO(Pate): This mini-flash method does not currently work as expected
    # Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py
    local_max = jnp.max(attn_weights, axis=-1, keepdims=True)
    local_exps = exp2(attn_weights - local_max)
    local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)

    local_sum = jnp.moveaxis(local_sum, -2, 1)
    local_max = jnp.moveaxis(local_max, -2, 1)

    local_max = jnp.reshape(local_max, (local_max.shape[0], local_max.shape[1], local_max.shape[2] * local_max.shape[3], 1)) 
    local_sum = jnp.reshape(local_sum, (local_sum.shape[0], local_sum.shape[1], local_sum.shape[2] * local_sum.shape[3], 1))

    local_out = self.wv_product(local_exps, value, aqt_rng)
    return local_out, local_max, local_sum


  def qk_product(self, query: Array, key: Array) -> Array:
    """Query-Key product.
    
    Args:
      query: Query projection, in shape of [b, t, n, d], where b: batch size, t:
        query length, n: number of heads, d: project dimension. 
      key: Key projection in shape of [b, s, n_kv, d] for where s: key length, n_kv is
        kv heads (sometimes k). The number of group for query is n // n_kv (sometimes g).

    Returns:
      results in shape [b, n_kv, n // n_kv,  t, s].
    """
    b, t, n, d = query.shape  
    n_kv = key.shape[-2]      # (4, 6, 8, 256
    assert n_kv == self.num_kv_heads  # 8 == 8
    query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d)) # (4, 1, 8, 1, 256)
    result = jnp.einsum('btkgd,bskd->bkgts', query, key) # (4, 8, 1, 1, 6)
    return result # (4, 8, 1, 1, 6)


  def wv_product(
      self,
      attn_weights: Array,  # (4, 8, 1, 1, 6) -> (batch, n_kv, groups, q_len, k_len)
      value: Array,         # (4, 6, 8, 256) -> (batch, v_len, n_kv, kv_dim)
      aqt_rng: PRNGKey | None) -> Array:
    """weighted value product.
    
    Args:
      attn_weights: Computed results of qk_einsum, in shape [batch_size, num_kv_heads, group_size, q_len, k_len]. 
      value: Value projection, in shape of [batch_size, v_len, num_kv_heads, kv_dim].
      aqt_rng: A PRNGKey for aqt ops.

    Returns:
      result in shape [batch_size, q_len, num_kv_heads * group_size, kv_dim]
    """
    einsum = _maybe_aqt_einsum(self.use_int8, aqt_rng)
    out = einsum('bkgts,bskd->btkgd', attn_weights, value)  # (4, 1, 8, 1, 256)
    b, t, n_kv, g, d = out.shape  # (4, 1, 8, 1, 256)
    result = jnp.reshape(out, (b, t, n_kv * g, d))  # (4, 1, 8 * 1, 256)
    return result   # (4, 1, 8, 256)

  def revert_kvlen_axis(self, kv):
    """Revert key/value length axis.

    Args:
      kv: in shape [b, ..., n, d, s].
    
    Returns:
      reshaped kv as [b, ..., s, n, d]
    """
    return jnp.moveaxis(kv, -1, -3)

  def move_kvlen_axis(self, kv):
    """Move key/value length axis to the end.

    Args:
      kv: in shape [b, ..., s, n, d].
    
    Returns:
      reshaped kv as [b, ..., n, d, s]
    """
    return jnp.moveaxis(kv, -3, -1)

  def cached_kv_shape(self, kv_shape):
    """Cached KV shape.

    The key and value have dimension [batch, length, num_heads, head_dim], but
    we cache them as [batch, num_heads, head_dim, length] as a TPU fusion
    optimization. This also enables the "scatter via one-hot broadcast" trick,
    which means we do a one-hot broadcast instead of a scatter/gather
    operations, resulting in a 3-4x speedup in practice.

    Args:
      kv_shape: shape of key or value for caching, as [b, ..., s, n, d].
    
    Returns:
      Swapped kv_shape as [b, ..., n, d, s] for cache.
    """
    return kv_shape[:-3] + tuple(kv_shape[i] for i in [-2, -1, -3])
  
  def _get_prefill_cache(self, batch, heads, kv_head_size, dtype):
    kv_cache_layout = ('cache_batch', 'cache_heads', 'cache_kv', 'cache_sequence')
    cache_logical_shape = (batch, self.max_prefill_predict_length, heads, kv_head_size)
    cached_key = self.variable('cache', 'cached_prefill_key',
                               nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                               self.cached_kv_shape(cache_logical_shape), dtype)
    cached_value = self.variable('cache', 'cached_prefill_value',
                                 nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                 self.cached_kv_shape(cache_logical_shape), dtype)
    cached_segment_id = self.variable('cache', 'cache_prefill_segment_id',
                  nn.with_logical_partitioning(jnp.zeros, ('cache_batch', 'cache_sequence')),
                  (cache_logical_shape[0], self.max_prefill_predict_length), jnp.int32)
    return cached_key, cached_value, cached_segment_id

  def _get_ar_cache(self, batch, heads, kv_head_size, dtype):
    kv_cache_layout = ('cache_batch', 'cache_heads', 'cache_kv', 'cache_sequence')
    cache_logical_shape = (batch, self.max_target_length - self.max_prefill_predict_length, heads, kv_head_size)
    cached_key = self.variable('cache', 'cached_ar_key',
                               nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                               self.cached_kv_shape(cache_logical_shape), dtype)
    cached_value = self.variable('cache', 'cached_ar_value',
                                 nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                 self.cached_kv_shape(cache_logical_shape), dtype)
    cached_segment_id = self.variable('cache', 'cache_ar_segment_id',
                  nn.with_logical_partitioning(jnp.zeros, ('cache_batch', 'cache_sequence')),
                  (cache_logical_shape[0], self.max_target_length - self.max_prefill_predict_length), jnp.int32)
    cache_index = self.variable('cache', 'cache_ar_index',
                          nn.with_logical_partitioning(jnp.zeros, ()),
                          (1,), jnp.int32)
    return cached_key, cached_value, cached_segment_id, cache_index

  def kv_cache_prefill(self,
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
      batch, sequence, heads, kv_head_size = key.shape
      assert key.dtype == value.dtype, "Key and Value Dtypes should match."
      assert self.max_prefill_predict_length == sequence, "Set prefill length must match prefill sequence"
      
      cached_prefill_key, cached_prefill_value, cached_prefill_segment_id = self._get_prefill_cache(batch, heads, kv_head_size, key.dtype)
      self._get_ar_cache(batch, heads, kv_head_size, key.dtype) # initialize it now

      key_shaped_for_cache = self.move_kvlen_axis(key)
      value_shaped_for_cache = self.move_kvlen_axis(value)
        
      cached_prefill_key.value = key_shaped_for_cache
      cached_prefill_value.value = value_shaped_for_cache
      if decoder_segment_ids is not None:
        cached_prefill_segment_id.value = decoder_segment_ids

      return key, value, decoder_segment_ids
  
  
  # TODO(Pate): Docstring
  def get_one_hot_indices(self, 
                          length: int, 
                          cur_index: int, 
                          dtype: type,
                         ):
    one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=dtype)
    one_hot_indices_int32 = jax.nn.one_hot(cur_index, length, dtype=jnp.int32)
    return one_hot_indices, one_hot_indices_int32
  

  # TODO(Pate): Docstring and types
  def update_ar_key_value(self, 
                                 key,
                                 value,
                                 cached_ar_key, 
                                 cached_ar_value, 
                                 one_hot_indices,
                                ):
    # In order to update the key, value caches with the current key and
    # value, we move the length axis to the back
    one_token_key = self.move_kvlen_axis(key)
    one_token_value = self.move_kvlen_axis(value)

    # We implement an efficient scatter into the cache via one-hot broadcast and addition.
    ar_key = cached_ar_key.value + one_token_key * one_hot_indices
    ar_value = cached_ar_value.value + one_token_value * one_hot_indices
    cached_ar_key.value = ar_key
    cached_ar_value.value = ar_value

    # Move the keys and values back to their original shapes.
    return self.revert_kvlen_axis(ar_key), self.revert_kvlen_axis(ar_value)
    

  # TODO(Pate): Docstring and types
  def kv_cache_autoregressive(self,
                              key: Array,
                              value: Array,
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
      batch, sequence, heads, kv_head_size = key.shape
      if sequence != 1:
        raise ValueError(f"Sequence length should be 1 during autoregression, got {sequence=}")
      is_initialized = self.has_variable('cache', 'cache_ar_index')
      if not is_initialized:
        raise ValueError("Error, we can't do autoregression if we haven't seeded the KV Cache.")

      cached_ar_key, cached_ar_value, cached_ar_segment_id, cache_ar_index = self._get_ar_cache(batch, heads, kv_head_size, key.dtype)
      _, _, _, length = cached_ar_key.value.shape

      # Create a OHE of the current index. NOTE: the index is increased below.
      one_hot_indices, one_hot_indices_int32 = self.get_one_hot_indices(length, cache_ar_index.value, key.dtype)

      # Update key, value caches with our new 1d spatial slices.
      ar_key, ar_value = self.update_ar_key_value(key, value, cached_ar_key, cached_ar_value, one_hot_indices)
      cached_ar_segment_id.value = cached_ar_segment_id.value + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR * one_hot_indices_int32
      cache_ar_index.value = jnp.mod(cache_ar_index.value + 1, self.max_target_length)

      # Prep are return both prefill and ar caches
      cached_prefill_key, cached_prefill_value, cached_prefill_segment_id = self._get_prefill_cache(self.max_target_length, heads, kv_head_size, key.dtype)
      cached_prefill =  self.revert_kvlen_axis(cached_prefill_key.value), self.revert_kvlen_axis(cached_prefill_value.value), cached_prefill_segment_id.value
      return cached_prefill, (ar_key, ar_value, cached_ar_segment_id.value)


  def kv_cache(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str
  ) -> tuple[Array, Array, Array]:
    """KV cache takes the current state and updates the state accordingly.

    The key and value have dimension [batch, length, num_heads, head_dim],
    but we cache them as [batch, num_heads, head_dim, length] as a TPU
    fusion optimization. This also enables the "scatter via one-hot
    broadcast" trick, which means we do a one-hot broadcast instead of a
    scatter/gather operations, resulting in a 3-4x speedup in practice.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      model_mode: model mode controlling model

    Returns:
      two tuples of (k, v, decoder_segments) -- either can be Nones

    """
    if key.shape != value.shape:
      raise ValueError(f"Can't KV cache with mismatched shapes {key.shape=}, {value.shape=}")
    

    if model_mode == common_types.MODEL_MODE_TRAIN:
      return (key, value, decoder_segment_ids), None
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      return self.kv_cache_prefill(key, value, decoder_segment_ids), None
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value)
    else:
      raise ValueError(f"Model Mode isn't supported! {model_mode=}")

  @nn.compact
  def __call__(self, query, key, value, decoder_segment_ids, model_mode):
    prefill_kv_cache, ar_kv_cache = self.kv_cache(key, value, decoder_segment_ids, model_mode)

    # TODO(Pate): This mini-flash method does not currently work as expected
    # Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py
    local_out1, local_max1, local_sum1 = self.apply_attention(
      query,
      prefill_kv_cache[0],
      prefill_kv_cache[1],
      prefill_kv_cache[2],
      model_mode=model_mode,
    )
    if ar_kv_cache is None:
      if local_sum1 is not None:
        return local_out1 / local_sum1
      return local_out1

    local_out2, local_max2, local_sum2  = self.apply_attention(
      query,
      ar_kv_cache[0],
      ar_kv_cache[1],
      ar_kv_cache[2],
      model_mode=model_mode,
    )

    local_outs = [local_out1, local_out2]
    local_maxes = [local_max1, local_max2]
    local_sums = [local_sum1, local_sum2]

    global_max = functools.reduce(jnp.maximum, local_maxes)
    global_sum = sum([
      exp2(local_max - global_max) * local_sum
      for (local_sum, local_max) in zip(local_sums, local_maxes)
    ])

    attn_out = 0
    for local_max, local_out in zip(local_maxes, local_outs):
      local_normalizer = exp2(local_max - global_max) / global_sum
      attn_out += local_normalizer * local_out
    return attn_out


class Attention(nn.Module):
  """ Generic Attention.

    Attributes:
      num_query_heads: number of query attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      num_kv_heads: number of kv attention heads.
      head_dim: dimension of each head.
      mesh: Mesh, device mesh
      attention_kernel: str, guidance on if we should use an attention kernel
      dtype: the dtype of the computation.
      max_target_length: maximum target length
      max_prefill_predict_length: size of the maximum prefill
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
      use_int8: bool, if true accelerate in int8
  """
    
  config: Config
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  max_target_length: int
  mesh: Mesh
  attention_kernel: str
  dtype: DType = jnp.float32
  max_prefill_predict_length: int = -1
  dropout_rate: float = 0.
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal')
  float32_logits: bool = False  # computes logits in float32 for stability.
  use_int8: bool = False
  

  query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  key_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  value_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)

  def query_projection(self, inputs_q: Array) -> Array:
    """Query projection."""

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    def query_init(*args):
      #pylint: disable=no-value-for-parameter
      return self.kernel_init(*args) / depth_scaling

    query_proj = DenseGeneral(
      features=(self.num_query_heads, self.head_dim),
      axis=-1,
      kernel_init=query_init,
      kernel_axes=('embed', 'heads', 'kv'),
      dtype=self.dtype,
      name='query',
      use_int8=self.use_int8)(inputs_q)
    return query_proj

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> Array:
    """Projection for Key and Value.

    Args:
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length,
        num_kv_heads, kv_dim]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, head_dim]`.
    """
    if self.num_kv_heads == -1:
      raise ValueError('num_kv_heads is not defined.')

    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError('Invaid num_kv_heads for GQA.')

    kv_proj = DenseGeneral(
        features=(self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=('embed', 'heads', 'kv'),
        dtype=self.dtype,
        name=proj_name,
        use_int8=self.use_int8)(inputs_kv)
    return kv_proj

  def qkv_projection(self, inputs: Array, proj_name: str):
    """ Fused QKV projection"""

    qkv_proj = DenseGeneral(
      features=(3, self.num_query_heads, self.head_dim),
      axis = -1,
      kernel_init=self.kernel_init,
        kernel_axes=('embed','qkv', 'heads', 'kv'),
        dtype=self.dtype,
        name=proj_name,
        use_int8=self.use_int8)(inputs)
    query, key, value = qkv_proj[:,:,0,...], qkv_proj[:,:,1,...], qkv_proj[:,:,2,...] 
    return query, key, value

  def out_projection(self, output_dim: int, out: Array) -> Array:
    out_proj = DenseGeneral(
      features=output_dim,
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      kernel_axes=('heads', 'kv', 'embed'),
      dtype=self.dtype,
      name='out',
      use_int8=self.use_int8)(out)
    return out_proj

  def key_rotary(self, key: Array, inputs_positions: Array):
    """Apply Rotary Embedding to key."""
    key = RotaryEmbedding(
      embedding_dims=self.head_dim,
      name='key_rotary')(inputs=key, position=inputs_positions)
    return key

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               inputs_positions: Array,
               decoder_segment_ids: Array | None = None,
               *,
               model_mode: str = common_types.MODEL_MODE_TRAIN,
               deterministic: bool = False):
    """Applies Attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are three modes: training, prefill and autoregression. During training, the KV cahce
    is ignored. During prefill, the cache is filled. During autoregression the cache is used.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      model_mode: corresponding to train, prefill and decode.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    # apply projection.
    if self.config.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name='qkv_proj')
    else:
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_kv, proj_name='key')
      value = self.kv_projection(inputs_kv, proj_name='value')

    # apply ROPE
    query = RotaryEmbedding(
        embedding_dims=self.head_dim, name='query_rotary'
    )(inputs=query, position=inputs_positions)
    key = self.key_rotary(key, inputs_positions)

    # annotate with sharding constraint.
    query = nn.with_logical_constraint(query, self.query_axis_names)
    query = checkpoint_name(query, 'query_proj')
    key = nn.with_logical_constraint(key, self.key_axis_names)
    key = checkpoint_name(key, 'key_proj')
    value = nn.with_logical_constraint(value, self.value_axis_names)
    value = checkpoint_name(value, 'value_proj')

    attention_op = AttentionOp(mesh=self.mesh,
                               attention_kernel=self.attention_kernel,
                               max_target_length=self.max_target_length,
                               max_prefill_predict_length=self.max_prefill_predict_length,
                               float32_logits=self.float32_logits,
                               use_int8=self.use_int8,
                               num_query_heads=self.num_query_heads,
                               num_kv_heads=self.num_kv_heads,
                               dtype=self.dtype)
    
    out = attention_op(query, key, value, decoder_segment_ids, model_mode)

    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)
    return out