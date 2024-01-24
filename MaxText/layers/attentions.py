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
  float32_qk_product: bool = False
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
      return self.tpu_flash_attention(query, key, value, decoder_segment_ids)
    elif self.attention_kernel == 'gpu_flash_xla' or self.attention_kernel == 'gpu_flash_triton':
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError("""Decode not supported with flash attention.
                            Use `dot_product` instead.""")
      return self.gpu_flash_attention(query, key, value)
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

    # Casting qk_product and softmaxt computation for float32 for model stability.
    if self.float32_qk_product:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    # QK Product, a.k.a `attn_weights`: [batch, num_kv_heads, num_query_heads_per_kv_head, q_length, kv_length]
    attn_weights = self.qk_product(query, key)

    if self.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = jax.nn.softmax(attn_weights).astype(self.dtype)

    # Take the linear combination of `value`.
    return self.wv_product(attn_weights, value, aqt_rng)

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
    return jnp.einsum('btkgd,bskd->bkgts', query, key)


  def wv_product(
      self,
      attn_weights: Array,
      value: Array,
      aqt_rng: PRNGKey | None) -> Array:
    """weighted value product.
    
    Args:
      attn_weights: Computed results of qk_einsum, in shape of [b, n, t, s].
      value: Value projection, in shape of [b, s, d] for multi-head attention.
      aqt_rng: A PRNGKey for aqt ops.

    Returns:
      result in shape [b, t, n, d]
    """
    einsum = _maybe_aqt_einsum(self.use_int8, aqt_rng)
    out = einsum('bkgts,bskd->btkgd', attn_weights, value)
    b, t, n_kv, g, d = out.shape
    return jnp.reshape(out, (b, t, n_kv * g, d))

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
  
  def _get_cache(self, cache_logical_shape, dtype):
    kv_cache_layout = ('cache_batch', 'cache_heads', 'cache_kv', 'cache_sequence')
    cached_key = self.variable('cache', 'cached_key',
                               nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                               self.cached_kv_shape(cache_logical_shape), dtype)
    cached_value = self.variable('cache', 'cached_value',
                                 nn.with_logical_partitioning(jnp.zeros, kv_cache_layout),
                                 self.cached_kv_shape(cache_logical_shape), dtype)
    cached_segment_id = self.variable('cache', 'cache_segment_id',
                  nn.with_logical_partitioning(jnp.zeros, ('cache_batch', 'cache_sequence')),
                  (cache_logical_shape[0], self.max_target_length), jnp.int32)
    cache_index = self.variable('cache', 'cache_index',
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
        tuple of key, value, decoder_segment_id.

      """
      assert key.dtype == value.dtype, "Key and Value Dtypes should match."
      cache_logical_shape = (key.shape[0], self.max_target_length, key.shape[2], key.shape[3])
      _, sequence, _, _ = key.shape

      cached_key, cached_value, cached_segment_id, cache_index = self._get_cache(cache_logical_shape, key.dtype)
      cached_key.value *= 0 # we might be prefilling something that already exists!
      cached_value.value *= 0 # we might be prefilling something that already exists!
      cached_segment_id.value *= 0
      cache_index.value = jnp.array(sequence)

      key_shaped_for_cache = self.move_kvlen_axis(key)
      value_shaped_for_cache = self.move_kvlen_axis(value)
        
      cached_key.value = cached_key.value.at[:, :, :, 0:sequence].set(key_shaped_for_cache)
      cached_value.value = cached_value.value.at[:, :, :, 0:sequence].set(value_shaped_for_cache)
      cached_segment_id.value = cached_segment_id.value.at[:, 0:sequence].set(decoder_segment_ids)
      return key, value, decoder_segment_ids

  def kv_cache_autoregressive(self,
                              key: Array,
                              value: Array,
                              decoder_segment_ids: Array,
                             ):
      """In autoregressive mode, we update the cache for this entry and 
         then return the full cache.

      Args:
        key: in shape [b, 1, n, d].
        value: in shape [b, 1, n, d].
        decoder_segment_ids: [b, 1] -- marking segment ids for tokens

      Returns:
        tuple of key, value with cached,
      Raises:
        ValueError: when key/value shape is not [batch, 1, num_heads, heads_dim].
      """
      cache_logical_shape = (key.shape[0], self.max_target_length, key.shape[2], key.shape[3])
      _, sequence, _, _ = key.shape
      if sequence != 1:
        raise ValueError(f"Sequence length should be 1 during autoregression, got {sequence=}")
      is_initialized = self.has_variable('cache', 'cached_key')
      if not is_initialized:
        raise ValueError("Error, we can't do autoregression if we haven't seeded the KV Cache.")

      cached_key, cached_value, cached_segment_id, cache_index = self._get_cache(cache_logical_shape, key.dtype)
      _, _, _, length = cached_key.value.shape

      # Create a OHE of the current index. NOTE: the index is increased below.
      cur_index = cache_index.value
      one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
      one_hot_indices_int32 = jax.nn.one_hot(cur_index, length, dtype=jnp.int32)

      # In order to update the key, value caches with the current key and
      # value, we move the length axis to the back, similar to what we did for
      # the cached ones above.
      # Note these are currently the key and value of a single position, since
      # we feed one position at a time.
      one_token_key = self.move_kvlen_axis(key)
      one_token_value = self.move_kvlen_axis(value)

      # Update key, value caches with our new 1d spatial slices.
      # We implement an efficient scatter into the cache via one-hot
      # broadcast and addition.
      key = cached_key.value + one_token_key * one_hot_indices
      value = cached_value.value + one_token_value * one_hot_indices
      cached_key.value = key
      cached_value.value = value
      cached_segment_id.value = cached_segment_id.value + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR * one_hot_indices_int32
      cache_index.value = cache_index.value + 1

      # Move the keys and values back to their original shapes.
      key = self.revert_kvlen_axis(key)
      value = self.revert_kvlen_axis(value)

      return key, value, cached_segment_id.value

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
      tuple of key, value with cached,

    """
    if key.shape != value.shape:
      raise ValueError(f"Can't KV cache with mismatched shapes {key.shape=}, {value.shape=}")

    if model_mode == common_types.MODEL_MODE_TRAIN:
      return key, value, decoder_segment_ids
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      return self.kv_cache_prefill(key, value, decoder_segment_ids)
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value, decoder_segment_ids)
    else:
      raise ValueError(f"Model Mode isn't supported! {model_mode=}")

  @nn.compact
  def __call__(self, query, key, value, decoder_segment_ids, model_mode):
    key_to_use, value_to_use, decoder_segment_ids_to_use = self.kv_cache(key, value, decoder_segment_ids, model_mode)

    # Apply attention.
    out = self.apply_attention(
        query,
        key_to_use,
        value_to_use,
        decoder_segment_ids_to_use,
        model_mode=model_mode,
    )
    return out

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
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_qk_product: bool, if True then compute logits via float32 qk_product to avoid
        numerical issues with bfloat16.
      float32_logits: bool, if True then cast logits to float32 before softmax to avoid
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
  dropout_rate: float = 0.
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal')
  float32_qk_product: bool = False  # computes logits in float32 for stability.
  float32_logits: bool = False  # cast logits in float32 for stability.
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
        kernel_axes=('embed', 'qkv', 'heads', 'kv'),
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
                               float32_qk_product=self.float32_qk_product,
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