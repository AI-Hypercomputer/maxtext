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
LLaMARotaryEmbedding = embeddings.LLaMARotaryEmbedding
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

def _maybe_aqt_einsum(int8_training, aqt_rng):
  """Maybe overwrite dot general with aqt_dot_general."""
  if not int8_training:
    return jnp.einsum
  else:
    aqt_dot_general = quantizations.int8_dot_general(aqt_rng)
    return functools.partial(jnp.einsum, _dot_general=aqt_dot_general)


class MultiHeadDotProductAttention(nn.Module):

  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """
  num_heads: int
  head_dim: int
  mesh: Mesh
  dtype: DType = jnp.float32
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
      features=(self.num_heads, self.head_dim),
      axis=-1,
      kernel_init=query_init,
      kernel_axes=('embed', 'heads', 'kv'),
      dtype=self.dtype,
      name='query',
      use_int8=self.use_int8)(inputs_q)
    return query_proj

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> Array:
    """Projection for Key and Value.

    Args;
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, num_heads,
      head_dim]`.
    """
    kv_proj = DenseGeneral(
      features=(self.num_heads, self.head_dim),
      axis=-1,
      kernel_init=self.kernel_init,
      kernel_axes=('embed', 'heads', 'kv'),
      dtype=self.dtype,
      name=proj_name,
      use_int8=self.use_int8)(inputs_kv)
    return kv_proj

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

  def attention_dropout(
      self,
      attn_weights: Array,
      dropout_rng: PRNGKey | None) -> Array:
    """Apply attention dropout."""
    keep_prob = 1.0 - self.dropout_rate
    # Broadcast dropout along the query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
        keep_prob, dtype=self.dtype
    )
    attn_weights = attn_weights * multiplier
    return attn_weights

  def check_attention_inputs(
      self,
      query: Array,
      key: Array,
      value: Array) -> None:
    """Check attention inputs."""

    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        'q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        'q, k, v num_heads must match.')
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  def qk_product(self, query: Array, key: Array) -> Array:
    """Query-Key product.

    Args:
      query: Query projection, in shape of [b, t, n, d], where b: batch size, t:
        query length, n: number of heads, d: project dimension.
      key: Key projection in shape of [b, s, n, d] for multihead attention.

    Returns:
      results in shape [b, n, t, s].
    """
    return jnp.einsum('btnd,bsnd->bnts', query, key)

  def wv_product(
      self,
      attn_weights: Array,
      value: Array,
      aqt_rng: PRNGKey | None) -> Array:
    """weighted value product.

    Args:
      attn_weights: Computed results of qk_einsum, in shape of [b, n, t, s] for
        multi-head attention
      value: Value projection, in shape of [b, s, n, d] for multi-head attention
      aqt_rng: A PRNGKey for aqt ops.

    Returns:
      result in shape [b, t, n, d]
    """

    einsum = _maybe_aqt_einsum(self.use_int8, aqt_rng)
    return einsum('bnts,bsnd->btnd', attn_weights, value)

  def key_rotary(self, key: Array, inputs_positions: Array):
    """Apply Rotary Embedding to key."""
    key = LLaMARotaryEmbedding(
      embedding_dims=self.head_dim,
      name='key_rotary')(inputs=key, position=inputs_positions)
    return key


  # Following Pallas MHA Flash Attention Reference.
  # https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
  def generate_attention_mask(
      self,
      query,
      key,
      decoder_segment_ids: Array | None,
      decode: bool = False,
  ) -> Array | None:
    mask = None
    if decoder_segment_ids is not None:
      mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      mask = mask[:, None, :, :]

    _, q_seq_len, _, _ = query.shape
    _, kv_seq_len, _, _ = key.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]

    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    return jnp.where(mask, 0.0, DEFAULT_MASK_VALUE) if mask is not None else None

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      dropout_rng: PRNGKey | None,
      deterministic: bool,
      decode: bool = False,
  ) -> Array:
    """Apply Attention."""

    aqt_rng = self.make_rng('aqt')
    self.check_attention_inputs(query, key, value)

    # Casting logits and softmax computation for float32 for model stability.
    if self.float32_logits:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    # QK Product, a.k.a `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = self.qk_product(query, key)

    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, decode)
    if attn_mask is not None:
      attn_weights += attn_mask

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = jax.nn.softmax(attn_weights).astype(self.dtype)

    # Apply attention dropout.
    if not deterministic and self.dropout_rate > 0.:
      attn_weights = self.attention_dropout(attn_weights, dropout_rng)



    # Take the linear combination of `value`.
    return self.wv_product(attn_weights, value, aqt_rng)

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

  def decode(
      self,
      key: Array,
      value: Array,
      query_shape: Sequence[int]
  ) -> tuple[Array, Array]:
    """Decoding method.

    The key and value have dimension [batch, length, num_heads, head_dim],
    but we cache them as [batch, num_heads, head_dim, length] as a TPU
    fusion optimization. This also enables the "scatter via one-hot
    broadcast" trick, which means we do a one-hot broadcast instead of a
    scatter/gather operations, resulting in a 3-4x speedup in practice.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      mask: 
      query_shape: expected to be [b, 1, n, d].

    Returns:
      tuple of key, value with cached, 
    Raises:
      ValueError: when query shape is not [batch, 1, num_heads, heads_dim].
    """
    # Detect if we're initializing by absence of existing cache data.
    is_initialized = self.has_variable('cache', 'cached_key')

    cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                               self.cached_kv_shape(key.shape), key.dtype)
    cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                 self.cached_kv_shape(value.shape), value.dtype)
    cache_index = self.variable('cache', 'cache_index',
                                lambda: jnp.array(0, dtype=jnp.int32))

    if not is_initialized:
      return key, value

    batch, num_heads, head_dim, length = cached_key.value.shape
    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    # Sanity shape check of cached key against input query.
    expected_shape = (batch, 1, num_heads, head_dim)
    if expected_shape != query_shape:
      raise ValueError(f"""Autoregressive cache shape error,
                        expected query shape %s instead got
                        {(expected_shape, query_shape)}""")
    # Create a OHE of the current index. NOTE: the index is increased below.
    cur_index = cache_index.value
    one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)

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
    cache_index.value = cache_index.value + 1

    # Move the keys and values back to their original shapes.
    key = self.revert_kvlen_axis(key)
    value = self.revert_kvlen_axis(value)

    return key, value

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               inputs_positions: Array,
               decoder_segment_ids: Array | None = None,
               *,
               decode: bool = False,
               deterministic: bool = False):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    # apply projection.
    query = self.query_projection(inputs_q)
    key = self.kv_projection(inputs_kv, proj_name='key')
    value = self.kv_projection(inputs_kv, proj_name='value')

    # apply ROPE
    query = LLaMARotaryEmbedding(
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

    if decode:
      key, value = self.decode(key, value, query.shape)


    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    out = self.apply_attention(
        query,
        key,
        value,
        decoder_segment_ids,
        dropout_rng,
        deterministic,
        decode=decode
    )
    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)
    return out


class FlashMultiHeadDotProductAttention(MultiHeadDotProductAttention):
  """Multi-head flash attention."""

  max_target_length: int = -1
  flash_axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  device_type: str = 'tpu'

  def tpu_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None) -> Array:
    """TPU Flash Attention."""

    if self.max_target_length == -1:
      raise ValueError('max_target_length must be defined for flash MHA.')

    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jax.numpy.transpose(query, axes=(0, 2, 1, 3))
    key = jax.numpy.transpose(key, axes=(0, 2, 1, 3))
    value = jax.numpy.transpose(value, axes=(0, 2, 1, 3))

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
            == self.max_target_length
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
    x = jax.numpy.transpose(x, axes=(0, 2, 1, 3))
    return x

  def gpu_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
  ) -> Array:
    """GPU Flash Attention."""
    b, n, s, h = key.shape # pylint: disable=unused-variable
    bwd_pass_impl = self.config.gpu_flash_attention_backward_pass_impl
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

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      dropout_rng: PRNGKey | None,
      deterministic: bool,
      decode: bool = False) -> Array:
    """"Applies flash attention."""

    del dropout_rng, deterministic

    if decode:
      raise ValueError("""Decode not supported with flash attention.
                            Use MHA instead.""")
    if self.device_type.lower() == 'tpu':
      return self.tpu_flash_attention(query, key, value, decoder_segment_ids)
    elif self.device_type.lower() == 'gpu':
      return self.gpu_flash_attention(query, key, value)
    else:
      raise ValueError('Unexpected deivce type.')


class MultiQueryDotProductAttention(MultiHeadDotProductAttention):
  """Multi-Query Attention https://arxiv.org/abs/1911.02150."""

  key_axis_names: AxisNames = (BATCH, LENGTH, D_KV)
  value_axis_names: AxisNames = (BATCH, LENGTH, D_KV)

  def check_attention_inputs(
      self,
      query: Array,
      key: Array,
      value: Array) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, 'k, v must have same rank.'
    assert query.shape[:-3] == key.shape[:-2] == value.shape[:-2], (
        'q, k, v batch dims must match.')
    assert query.shape[-2] == self.num_heads, 'q num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> Array:
    """Projection for Key and Value.

    Args:
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length, kv_dim]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, head_dim]`.
    """

    kv_proj = DenseGeneral(
        features=self.head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=('embed', 'kv'),
        dtype=self.dtype,
        name=proj_name,
        use_int8=self.use_int8)(inputs_kv)
    return kv_proj

  def qk_product(self, query: Array, key: Array) -> Array:
    """Query-Key product.

    Args:
      query: Query projection, in shape of [b, t, n, d], where b: batch size, t:
        query length, n: number of heads, d: project dimension.
      key: Key projection in shape of [b, s, d] for multi-query attention.

    Returns:
      results in shape [b, n, t, s].
    """
    return jnp.einsum('btnd,bsd->bnts', query, key)

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
    return einsum('bnts,bsd->btnd', attn_weights, value)

  def key_rotary(self, key: Array, inputs_positions: Array):
    """Apply Rotary Embedding to Key."""
    key_input_shape = key.shape
    key = jnp.expand_dims(key, axis=-2)  # [b, s, d] -> [b, s, 1, d]
    key = super().key_rotary(key, inputs_positions)
    return jnp.reshape(key, key_input_shape)

  def revert_kvlen_axis(self, kv):
    """Revert key/value length axis.

    Args:
      kv: in shape [b, ..., d, s].
    
    Returns:
      reshaped kv as [b, ..., s, d]
    """
    return jnp.moveaxis(kv, -1, -2)

  def move_kvlen_axis(self, kv):
    """Move key/value length axis to the end.

    Args:
      kv: in shape [b, ..., s, d].
    
    Returns:
      reshaped kv as [b, ..., d, s]
    """
    return jnp.moveaxis(kv, -2, -1)

  def cached_kv_shape(self, kv_shape):
    """Cached KV shape.

    The key and value have dimension [batch, length, num_heads, head_dim], but
    we cache them as [batch, num_heads, head_dim, length] as a TPU fusion
    optimization. This also enables the "scatter via one-hot broadcast" trick,
    which means we do a one-hot broadcast instead of a scatter/gather
    operations, resulting in a 3-4x speedup in practice.

    Args:
      kv_shape: shape of key or value for caching, as [b, ..., s, d].
    
    Returns:
      Swapped kv_shape as [b, ..., d, s] for cache.
    """
    return kv_shape[:-2] + tuple(kv_shape[i] for i in [-1, -2])


class GroupedQueryDotProductAttention(MultiHeadDotProductAttention):
  """Grouped-Query Attention https://arxiv.org/abs/2305.13245."""

  num_kv_heads: int = -1

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

    if self.num_heads % self.num_kv_heads != 0:
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

  def qk_product(self, query: Array, key: Array) -> Array:
    """Query-Key product.
    
    Args:
      query: Query projection, in shape of [b, t, n, d], where b: batch size, t:
        query length, n: number of heads, d: project dimension. 
      key: Key projection in shape of [b, s, n_kv, d] for where n_kv is 
        kv heads. The number of group for query is n // n_kv.

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
