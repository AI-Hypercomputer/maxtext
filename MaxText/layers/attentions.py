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
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
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
Quant = quantizations.AqtQuantization

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

# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


def apply_mask_to_logits(logits: Array, mask: Array):
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a tensor with some dtype where 0 represents true and values
  below a large negative number (here set to
  get_large_negative_number(logits.dtype) / 2) represent false. Applying the mask
  leaves the logits alone in the true case and replaces them by
  get_large_negative_number(logits.dtype) in the false case. Previously, this was
  done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the values in memory rather than
  just the predicate. This implementation avoids that problem.

  from https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L706

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor of mask values with the encoding described in the
      function documentation.

  Returns:
    Masked logits.
  """
  return jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), logits, DEFAULT_MASK_VALUE)

def _maybe_aqt_einsum(quant: Quant):
  """Maybe overwrite dot general with aqt_dot_general."""
  return jnp.einsum if quant is None else quant.einsum()

class AttentionOp(nn.Module):
  mesh: Mesh
  attention_kernel: str
  max_target_length: int
  num_query_heads: int
  num_kv_heads: int
  float32_qk_product: bool = False
  max_prefill_predict_length: int = -1 
  float32_logits: bool = False
  flash_axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  dropout_rate: float = 0.
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None
  quantize_kvcache: bool = False

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
      model_mode: str):
    self.check_attention_inputs(query, key, value)
    length = query.shape[-3]
    if self.attention_kernel == 'dot_product' or\
          (self.attention_kernel == 'autoselected' and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE) or\
          (self.attention_kernel == 'autoselected' and length < 128):
      return self.apply_attention_dot(query, key, value, decoder_segment_ids, model_mode)
    elif self.attention_kernel == 'flash' or\
          self.attention_kernel == 'autoselected':
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError("""Decode not supported with flash attention.
                            Use `dot_product` instead.""")
      return self.tpu_flash_attention(query, key, value, decoder_segment_ids), None, None
    elif self.attention_kernel == 'cudnn_flash_te':
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError("""Decode not supported with flash attention.
                           Use `dot_product` instead.""")
      return self.cudnn_flash_attention(query, key, value, decoder_segment_ids, model_mode), None, None
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


    if decoder_segment_ids is not None:
      decoder_segment_ids = splash_attention_kernel.SegmentIds(
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
        ), 'Sharding along sequence dimension not allowed in tpu kernel attention'
      block_sizes = splash_attention_kernel.BlockSizes(
                                                  block_q=min(512, query.shape[2]),
                                                  block_kv_compute=min(512, key.shape[2]),
                                                  block_kv=min(512, key.shape[2]),
                                                  block_q_dkv=min(512, query.shape[2]),
                                                  block_kv_dkv=min(512, key.shape[2]),
                                                  block_kv_dkv_compute=min(512, query.shape[2]),
                                                  block_q_dq=min(512, query.shape[2]),
                                                  block_kv_dq=min(512, query.shape[2]),
      )

      masks = [splash_attention_mask.CausalMask( shape=(query.shape[2],query.shape[2])) for i in range(query.shape[1])]
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=masks)
      splash_kernel = splash_attention_kernel.make_splash_mha(mask = multi_head_mask,
                                                              head_shards = 1,
                                                              q_seq_shards = 1,
                                                              block_sizes = block_sizes)
      
      return jax.vmap(splash_kernel)(query,key,value, segment_ids = decoder_segment_ids)

    devices_in_data_fsdp = self.mesh.shape['data'] * self.mesh.shape['fsdp']
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        'Batch dimension should be shardable among the devices in data and fsdp'
        ' axis'
    )
    x = wrap_flash_attention(query, key, value, decoder_segment_ids)
    x = jnp.transpose(x, axes=(0, 2, 1, 3))
    return x
  
  def cudnn_flash_attention(
    self,
    query: Array,
    key: Array,
    value: Array,
    decoder_segment_ids: Array | None,
    model_mode: str = common_types.MODEL_MODE_TRAIN,
    ) -> Array:
    """CUDNN Flash Attention with Transformer Engine.
      1. Stable API, supports GQA
      2. Supports head_dim till 128; head_dim=256 support will be added soon 
    """
    # These imports are only meant to work in a GPU build.
    from transformer_engine.jax.flax.transformer import DotProductAttention # pytype: disable=import-error

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    #generate attn_mask
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)

    dpa_layer = DotProductAttention(head_dim=head_dim,
                  num_attention_heads=self.num_query_heads,
                  num_gqa_groups=self.num_kv_heads,
                  attn_mask_type='causal', # 'causal' or 'padding'
                  attn_bias_type='NO_BIAS', # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
                  attention_dropout=self.dropout_rate,
                  dropout_rng_name='aqt',
                  dtype=self.dtype,
                  float32_logits=self.float32_logits,
                  qkv_layout='BSHD_BSHD_BSHD', # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
                  scale_factor=1.0/math.sqrt(head_dim),
                  transpose_batch_sequence=False)
    return dpa_layer(query, key, value, mask=attn_mask)

  def compute_local_attention(self,
                              attn_weights: Array, 
                              value: Array) -> tuple[Array, Array, Array]:
    """Computes the attention of a local subset of the kv cache. 
    Local attention results will need to be combined with any other local attentions and normalized
    Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py

    Args:
        attn_weights (Array): Product of query and key
        value (Array): Current value
        aqt_rng (PRNGKey | None): Optional rng

    Returns:
        (local_out, local_max,): where
          local_out is local unnormalized output
          local_max is the local max of exponentials
          local_sum is the sum of exponentials for this chunk, divided by exp(local_max).
    """
    local_max = jnp.max(attn_weights, axis=-1, keepdims=True)
    local_exps = jnp.exp(attn_weights - local_max)
    local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)

    local_sum = jnp.moveaxis(local_sum, -2, 1)
    local_max = jnp.moveaxis(local_max, -2, 1)

    local_max = jnp.reshape(local_max, 
                            (local_max.shape[0], 
                             local_max.shape[1], 
                             local_max.shape[2] * local_max.shape[3], 
                             1)) 
    local_sum = jnp.reshape(local_sum, 
                            (local_sum.shape[0], 
                             local_sum.shape[1], 
                             local_sum.shape[2] * local_sum.shape[3], 
                             1)) 

    local_out = self.wv_product(local_exps, value)
    return local_out, local_max, local_sum

  def apply_attention_dot(
      self,
      query: Array, 
      key: Array,   
      value: Array, 
      decoder_segment_ids: Array | None,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
  ):
    """Apply Attention."""
    # Casting qk_product and softmaxt computation for float32 for model stability.
    if self.float32_qk_product:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    attn_weights = self.qk_product(query, key)

    # Casting softmaxt computation for float32 for model stability.
    if self.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    return self.compute_local_attention(attn_weights, value)

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
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
    query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
    result = jnp.einsum('btkgd,bskd->bkgts', query, key)
    return result # (4, 8, 1, 1, 6)


  def wv_product(
      self,
      attn_weights: Array,
      value: Array) -> Array:
    """weighted value product.

    Args:
      attn_weights: Computed results of qk_einsum, in shape [batch_size, num_kv_heads, group_size, q_len, k_len]. 
      value: Value projection, in shape of [batch_size, v_len, num_kv_heads, kv_dim].

    Returns:
      result in shape [batch_size, q_len, num_kv_heads * group_size, kv_dim]
    """
    out = jnp.einsum('bkgts,bskd->btkgd', attn_weights, value)
    b, t, n_kv, g, d = out.shape
    result = jnp.reshape(out, (b, t, n_kv * g, d))
    return result

  def revert_kvlen_axis(self, kv):
    """Revert key/value length axis.

    Args:
      kv: in shape [b, ..., n, d, s].

    Returns:
      reshaped kv as [b, ..., s, n, d]
    """
    return jax.numpy.moveaxis(kv, (0,1,2,3), (1,2,0,3))


  def move_kvlen_axis(self, kv):
    """Move key/value length axis to the end.

    Args:
      kv: in shape [b, ..., s, n, d].

    Returns:
      reshaped kv as [b, ..., n, d, s]
    """
    return jax.numpy.moveaxis(kv, (0,1,2,3), (2,0,1,3))

  def cached_kv_shape(self, kv_shape):
    """Cached KV shape.

    The key and value have dimension [batch, length, num_heads, head_dim], but
    we cache them as [length, num_heads, batch, head_dim, ] for optimized read/write performance.

    Args:
      kv_shape: shape of key or value for caching, as [b, ..., s, n, d].

    Returns:
      Swapped kv_shape as [b, ..., n, d, s] for cache.
    """
    return (kv_shape[1], kv_shape[2], kv_shape[0], kv_shape[3])

  def _get_prefill_cache(self, batch, heads, kv_head_size, quantize_kvcache):
    dtype = jnp.int8 if quantize_kvcache else jnp.bfloat16

    kv_cache_layout = ('cache_sequence', 'cache_heads', 'cache_batch', 'cache_kv', )
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

    if self.quantize_kvcache:
      cache_logical_shape_scale = (batch, self.max_prefill_predict_length, heads, 1)
      cached_key_scale_var = self.variable('cache', 'cached_prefill_key_scale',
                                    nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                    self.cached_kv_shape(cache_logical_shape_scale), jnp.bfloat16)
      cached_value_scale_var = self.variable('cache', 'cached_prefill_value_scale',
                                      nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                      self.cached_kv_shape(cache_logical_shape_scale), jnp.bfloat16)
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    key_vars = (cached_key, cached_key_scale_var)
    value_vars = (cached_value, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id

  def _get_ar_cache(self, batch, heads, kv_head_size, quantize_kvcache):
    dtype = jnp.int8 if quantize_kvcache else jnp.bfloat16
    cache_length = self.max_target_length - self.max_prefill_predict_length
    kv_cache_layout = ('cache_sequence', 'cache_heads', 'cache_batch', 'cache_kv', )
    cache_logical_shape = (batch, cache_length, heads, kv_head_size)
    cached_key = self.variable('cache', 'cached_ar_key',
                               nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                               self.cached_kv_shape(cache_logical_shape), dtype)
    cached_value = self.variable('cache', 'cached_ar_value',
                                 nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                 self.cached_kv_shape(cache_logical_shape), dtype)
    cached_segment_id = self.variable('cache', 'cache_ar_segment_id',
                  nn.with_logical_partitioning(jnp.zeros, ('cache_batch', 'cache_sequence')),
                  (cache_logical_shape[0], cache_length), jnp.int32)

    if self.quantize_kvcache:
      cache_logical_shape_scale = (batch, cache_length, heads, 1)
      cached_key_scale_var = self.variable('cache', 'cached_ar_key_scale',
                                    nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                    self.cached_kv_shape(cache_logical_shape_scale), jnp.bfloat16)
      cached_value_scale_var = self.variable('cache', 'cached_ar_value_scale',
                                      nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                      self.cached_kv_shape(cache_logical_shape_scale), jnp.bfloat16)
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    cache_index = self.variable('cache', 'cache_ar_index',
                          nn.with_logical_partitioning(jnp.zeros, ()),
                          (1,), jnp.int32)
    key_vars = (cached_key, cached_key_scale_var)
    value_vars = (cached_value, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id, cache_index

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
      
      cached_prefill_key_var, cached_prefill_value_var, cached_prefill_segment_id = self._get_prefill_cache(batch, heads, kv_head_size, self.quantize_kvcache)
      self._get_ar_cache(batch, heads, kv_head_size, self.quantize_kvcache) # initialize it now

      key_shaped_for_cache = self.move_kvlen_axis(key)
      value_shaped_for_cache = self.move_kvlen_axis(value)

      if self.quantize_kvcache:
        key_shaped_for_cache, key_scale = quantizations.quantize_kv(key_shaped_for_cache)
        value_shaped_for_cache, value_scale  = quantizations.quantize_kv(value_shaped_for_cache)
        cached_prefill_key_var[1].value = key_scale
        cached_prefill_value_var[1].value = value_scale

      cached_prefill_key_var[0].value = key_shaped_for_cache
      cached_prefill_value_var[0].value = value_shaped_for_cache

      if decoder_segment_ids is not None:
        cached_prefill_segment_id.value = decoder_segment_ids

      return key, value, decoder_segment_ids
  

  def update_ar_key_value(self, 
                          one_token_key: Array,
                          one_token_value: Array,
                          cached_key_vars: tuple[nn.Variable, nn.Variable|None], 
                          cached_value_vars: tuple[nn.Variable, nn.Variable|None], 
                          one_hot_indices: Array) -> tuple[Array, Array]:
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
    # value, we move the length axis to the back
    one_token_key = self.move_kvlen_axis(one_token_key)
    one_token_value = self.move_kvlen_axis(one_token_value)

    if self.quantize_kvcache:
      one_token_key, one_token_key_scale = quantizations.quantize_kv(one_token_key)
      one_token_value, one_token_value_scale = quantizations.quantize_kv(one_token_value)

    one_hot_indices = one_hot_indices.astype(int)


    ar_key = cached_key_var.value
    ar_key = jax.lax.dynamic_update_index_in_dim(ar_key, one_token_key, jnp.squeeze(one_hot_indices), 0)
    ar_key = nn.with_logical_constraint(ar_key, ('cache_sequence', 'cache_heads', 'cache_batch', 'cache_kv',))
    cached_key_var.value = ar_key

    ar_value = cached_value_var.value
    ar_value = jax.lax.dynamic_update_index_in_dim(ar_value, one_token_value, jnp.squeeze(one_hot_indices), 0)
    ar_value = nn.with_logical_constraint(ar_value, ('cache_sequence', 'cache_heads', 'cache_batch', 'cache_kv',))
    cached_value_var.value = ar_value

    if self.quantize_kvcache:
      ar_key_scale = jax.lax.dynamic_update_index_in_dim(cached_key_scale_var.value, one_token_key_scale, jnp.squeeze(one_hot_indices), 0)
      ar_value_scale = jax.lax.dynamic_update_index_in_dim(cached_value_scale_var.value, one_token_value_scale, jnp.squeeze(one_hot_indices), 0)
      cached_key_scale_var.value = ar_key_scale
      cached_value_scale_var.value = ar_value_scale

      ar_key = quantizations.unquantize_kv(cached_key_var.value, cached_key_scale_var.value, one_token_key.dtype)
      ar_value = quantizations.unquantize_kv(cached_value_var.value, cached_value_scale_var.value, one_token_value.dtype)

    # Move the keys and values back to their original shapes.
    return self.revert_kvlen_axis(ar_key), self.revert_kvlen_axis(ar_value)

  def prefill_cache_var_model_var(self, cache_var, target_dtype):
    if not self.quantize_kvcache:
      return self.revert_kvlen_axis(cache_var[0].value)
    else:
      raw_cache, quant_scale = cache_var
      raw_cache_unquantized = quantizations.unquantize_kv(raw_cache.value, quant_scale.value, target_dtype) 
      return self.revert_kvlen_axis(raw_cache_unquantized)

    

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

      cached_ar_key_var, cached_ar_value_var, cached_ar_segment_id, cache_ar_index = self._get_ar_cache(batch, heads, kv_head_size, self.quantize_kvcache)

      key = nn.with_logical_constraint(key, (BATCH, LENGTH, HEAD, D_KV))
      value = nn.with_logical_constraint(value, (BATCH, LENGTH, HEAD, D_KV))

      ar_key, ar_value = self.update_ar_key_value(key, value, cached_ar_key_var, cached_ar_value_var, cache_ar_index.value)
      active_indicator = jnp.zeros((batch, 1), dtype = jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
      cached_ar_segment_id.value = jax.lax.dynamic_update_index_in_dim(cached_ar_segment_id.value, active_indicator, jnp.squeeze(cache_ar_index.value), 1)
      cache_ar_index.value = jnp.mod(cache_ar_index.value + 1, self.max_target_length - self.max_prefill_predict_length)

      # Prep and return both prefill and ar caches
      cached_prefill_key_var, cached_prefill_value_var, cached_prefill_segment_id = self._get_prefill_cache(self.max_target_length, heads, kv_head_size, self.quantize_kvcache)

      cached_prefill = self.prefill_cache_var_model_var(cached_prefill_key_var, key.dtype), self.prefill_cache_var_model_var(cached_prefill_value_var, value.dtype), cached_prefill_segment_id.value
      return cached_prefill, (ar_key, ar_value, cached_ar_segment_id.value)

  def kv_cache(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str
  ) -> tuple:
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
  
  
  def normalize_attention(self, 
                          local_outs,
                          local_maxes,
                          local_sums):
    """Normalize across multiple localized attentions

    Args:
        local_outs (list): List of unnormalized outputs entries for each local attention
        local_maxes (list): List of max exponentials entries for each local attention
        local_sums (list): List of exponential sum entries for each local attention

    Returns:
        Array: Combined attention that has been normalized 
    """
    # Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py
    global_max = functools.reduce(jnp.maximum, local_maxes)
    global_sum = sum([
      jnp.exp(local_max - global_max) * local_sum
      for (local_sum, local_max) in zip(local_sums, local_maxes)
    ])

    attn_out = 0
    for local_max, local_out in zip(local_maxes, local_outs):
      local_normalizer = jnp.exp(local_max - global_max) / global_sum
      attn_out += local_normalizer * local_out
    return attn_out


  @nn.compact
  def __call__(self, query, key, value, decoder_segment_ids, model_mode):
    prefill_kv_cache, ar_kv_cache = self.kv_cache(key, value, decoder_segment_ids, model_mode)

    prefill_unnormalized_output, prefill_exponentials_max, prefill_exponentials_sum = self.apply_attention(
      query=query,
      key=prefill_kv_cache[0],
      value=prefill_kv_cache[1],
      decoder_segment_ids=prefill_kv_cache[2],
      model_mode=model_mode,
    )

    # Return the "prefill" cache if it actually the combined prefill+ar kv cache
    if ar_kv_cache is None:
      if prefill_exponentials_sum is not None:
        return prefill_unnormalized_output / prefill_exponentials_sum
      return prefill_unnormalized_output

    ar_unnormalized_output, ar_exponentials_max, ar_exponentials_sum  = self.apply_attention(
      query=query,
      key=ar_kv_cache[0],
      value=ar_kv_cache[1],
      decoder_segment_ids=ar_kv_cache[2],
      model_mode=model_mode,
    )

    unnormalized_outputs = [prefill_unnormalized_output, ar_unnormalized_output]
    exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
    exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
    return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)


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
      weight_dtype: the dtype of the weights.
      max_target_length: maximum target length
      max_prefill_predict_length: size of the maximum prefill
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_qk_product: bool, if True then compute logits via float32 qk_product to avoid
        numerical issues with bfloat16.
      float32_logits: bool, if True then cast logits to float32 before softmax to avoid
        numerical issues with bfloat16.
      quant: Quant, stores quantization parameters, defaults to None implying no quantization.
      quantize_kvcache: bool, quantize the kv cache.
  """

  config: Config
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  max_target_length: int
  mesh: Mesh
  attention_kernel: str
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  max_prefill_predict_length: int = -1
  dropout_rate: float = 0.
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal')
  float32_qk_product: bool = False  # computes logits in float32 for stability.
  float32_logits: bool = False  # cast logits in float32 for stability.
  quant: Optional[Quant] = None
  quantize_kvcache: bool = False


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
      weight_dtype=self.weight_dtype,
      name='query',
      quant=self.quant)(inputs_q)
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
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant)(inputs_kv)
    return kv_proj

  def qkv_projection(self, inputs: Array, proj_name: str):
    """ Fused QKV projection"""

    qkv_proj = DenseGeneral(
      features=(3, self.num_query_heads, self.head_dim),
      axis = -1,
      kernel_init=self.kernel_init,
        kernel_axes=('embed', 'qkv', 'heads', 'kv'),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant)(inputs)
    qkv_proj = checkpoint_name(qkv_proj, 'qkv_proj')
    query, key, value = qkv_proj[:,:,0,...], qkv_proj[:,:,1,...], qkv_proj[:,:,2,...]
    return query, key, value

  def out_projection(self, output_dim: int, out: Array) -> Array:
    out_proj = DenseGeneral(
      features=output_dim,
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      kernel_axes=('heads', 'kv', 'embed'),
      dtype=self.dtype,
      weight_dtype=self.weight_dtype,
      name='out',
      quant=self.quant)(out)
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
                               float32_qk_product=self.float32_qk_product,
                               float32_logits=self.float32_logits,
                               quant=self.quant,
                               quantize_kvcache=self.quantize_kvcache,
                               num_query_heads=self.num_query_heads,
                               num_kv_heads=self.num_kv_heads,
                               dropout_rate = self.dropout_rate,
                               dtype=self.dtype)

    out = attention_op(query, key, value, decoder_segment_ids, model_mode)

    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)
    out = checkpoint_name(out, 'out_proj')
    return out
