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
from typing import Optional

from flax import linen as nn
import jax
from jax import lax
from jax import random
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
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

nd_dense_init = initializers.nd_dense_init
shard_map = shard_map.shard_map

dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


def combine_biases(*masks: Array | None):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def combine_masks(*masks: Array | None, dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


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
      kernel_init=query_init
      kernel_axes=('embed', 'heads', 'kv'),
      dtype=self.dtype,
      name='query'
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
    # Broadcast dropout mask along the query dim.
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
    query = LLaMARotaryEmbedding(
      embedding_dims=self.head_dim, 
      name='key_rotary')(inputs=key, position=inputs_positions)
    return query

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      attention_bias: Array | None,
      dropout_rng: PRNGKey | None,
      deterministic: bool,
      decode: bool = False,
  ) -> Array:
    """Apply Attention."""

    del decoder_segment_ids
    del decode

    aqt_rng = self.make_rng('aqt')
    self.check_attention_inputs(query, key, value)

    # Casting logits and softmax computation for float32 for model stability.
    if self.float32_logits:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    # QK Product, a.k.a `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = self.qk_product(query, key)

    # Apply attention bias: masking, dropout, proximity bias, etc.
    if attention_bias is not None:
      attn_weights = attn_weights + attention_bias.astype(attn_weights.dtype)

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = jax.nn.softmax(attn_weights).astype(self.dtype)

    # Apply attention dropout.
    if not deterministic and self.dropout_rate > 0.:
      attn_weights = self.attention_dropout(attn_weights, dropout_rng)

    # Take the linear combination of `value`.
    return self.wv_product(attn_weights, value, aqt_rng)

  def decode(
      self,
      key: Array,
      value: Array,
      mask: Array,
      bias: Array,
      query_shape: Sequence[int]
  ) -> tuple[Array, Array, Optional[Array], Optional[Array]]:
    """Decoding method.
    
    The key and value have dimension [batch, length, num_heads, head_dim],
    but we cache them as [batch, num_heads, head_dim, length] as a TPU
    fusion optimization. This also enables the "scatter via one-hot
    broadcast" trick, which means we do a one-hot broadcast instead of a
    scatter/gather operations, resulting in a 3-4x speedup in practice.

    Args:
      key: have diem
      value:
      mask:
      bias:
      query_shape:

    Returns:

    Raises:
      ValueError: when query shape is not [batch, 1, num_heads, heads_dim].
    """
    def swap_dims(x):
      return x[:-3] + tuple(x[i] for i in [-2, -1, -3])

    cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                               swap_dims(key.shape), key.dtype)
    cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                 swap_dims(value.shape), value.dtype)
    cache_index = self.variable('cache', 'cache_index',
                                lambda: jnp.array(0, dtype=jnp.int32))

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
    one_token_key = jnp.moveaxis(key, -3, -1)
    one_token_value = jnp.moveaxis(value, -3, -1)
    # Update key, value caches with our new 1d spatial slices.
    # We implement an efficient scatter into the cache via one-hot
    # broadcast and addition.
    key = cached_key.value + one_token_key * one_hot_indices
    value = cached_value.value + one_token_value * one_hot_indices
    cached_key.value = key
    cached_value.value = value
    cache_index.value = cache_index.value + 1
    # Move the keys and values back to their original shapes.
    key = jnp.moveaxis(key, -1, -3)
    value = jnp.moveaxis(value, -1, -3)

    # Causal mask for cached decoder self-attention: our single query
    # position should only attend to those key positions that have already
    # been generated and cached, not the remaining zero elements.
    mask = combine_masks(
        mask,
        jnp.broadcast_to(
            jnp.arange(length) <= cur_index,
            # (1, 1, length) represent (head dim, query length, key length)
            # query length is 1 because during decoding we deal with one
            # index.
            # The same mask is applied to all batch elements and heads.
            (batch, 1, 1, length)))

    # Grab the correct relative attention bias during decoding. This is
    # only required during single step decoding.
    if bias is not None:
      # The bias is a full attention matrix, but during decoding we only
      # have to take a slice of it.
      # This is equivalent to bias[..., cur_index:cur_index+1, :].
      bias = dynamic_vector_slice_in_dim(
          jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)
    return key, value, bias, mask

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               attention_type,
               decoder_segment_ids = None,
               inputs_positions: Optional[Array] = None,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
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
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
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
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      if is_initialized:
        key, value, bias, mask = self.decode(
            key, value, bias, mask, query.shape
        )

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.0).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype),
      )
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    out = self.apply_attention(
        query,
        key,
        value,
        decoder_segment_ids,
        attention_bias,
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

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      attention_bias: Array | None,
      dropout_rng: PRNGKey | None,
      deterministic: bool,
      decode: bool = False) -> Array:
    """"Applies flash attention."""

    if decode:
      raise ValueError("""Decode not supported with flash attention.
                            Use MHA instead.""")

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


class MultiQueryDotProductAttention(MultiHeadDotProductAttention):
  """Multi-Query Attention https://arxiv.org/abs/1911.02150."""

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> Array:
    """Projection for Key and Value.

    Args;
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, head_dim]`.
    """

    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=self.head_dim,
        kernel_axes=('embed', 'kv'),
        dtype=self.dtype,
        use_int8=self.use_int8)
    proj = projection(kernel_init=self.kernel_init, name=proj_name)(inputs_kv)
    return proj
  
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
    key = jnp.expand_dims(key, axis=-2) # [b, s, d] -> [b, s, 1, d]
    key = super().key_rotary(key, inputs_positions)
    return jnp.reshape(key, key_input_shape)
