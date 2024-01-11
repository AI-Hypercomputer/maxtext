#  Copyright 2023 Google LLC

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#       https://www.apache.org/licenses/LICENSE-2.0

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
from jax.experimental.pallas.ops.tpu import flash_attention
import jax.numpy as jnp
import common_types
from layers import embeddings
from layers import initializers
from layers import linears
from layers import normalizations
from layers import quantizations

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
PRNGKey = common_types.PRNGKey

DenseGeneral = linears.DenseGeneral
LLaMARotaryEmbedding = embeddings.LLaMARotaryEmbedding
NdInitializer = initializers.NdInitializer
RMSNorm = normalizations.RMSNorm

nd_dense_init = initializers.nd_dense_init
shard_map = shard_map.shard_map

dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


def get_large_negative_number(dtype: jnp.dtype):
  """Returns a large negative value for the given dtype."""
  # from https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L694
  # -0.7 is a float64 in Jax. Explicit cast output to target dtype.
  if jnp.issubdtype(dtype, jnp.inexact):
    dtype_max = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtype_max = jnp.iinfo(dtype).max
  else:
    raise ValueError('Unsupported dtype for inputs.')
  return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


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

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor of mask values with the encoding described in the
      function documentation.

  Returns:
    Masked logits.
  """
  # from https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L706
  # handle -inf values in mask biases possible after combine_biases step
  #    and return masked logits with either logit value or min_value
  min_value = get_large_negative_number(logits.dtype)
  return jnp.where((mask >= min_value * 0.5), logits, min_value)


def combine_biases(*masks: Optional[Array]):
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


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
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


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          aqt_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False,
                          cfg: Config = None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout.
    aqt_rng: Jax PRNGKey to be used for aqt library.
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
    cfg: Configuration.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  def compute_qk_attn_weights(query, key):
    """Computes all query-key dot product pairs."""
    return jnp.einsum('bqhd,bkhd->bhqk', query, key)

  def compute_weighted_values(attn_weights, value, cfg, aqt_rng):
    """Computes attn_weights * values."""
    if not cfg.int8_training:
      weighted_values = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
    else:
      aqt_dot_general = quantizations.int8_dot_general(aqt_rng)
      weighted_values = jnp.einsum(
          'bhqk,bkhd->bqhd', attn_weights, value, _dot_general=aqt_dot_general)
    return weighted_values

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # QK Product, a.k.a `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = compute_qk_attn_weights(query, key)

  # fp32 append
  attn_weights = attn_weights.astype(jnp.float32)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = apply_mask_to_logits(attn_weights, bias)

  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    # Broadcast dropout mask along the query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return compute_weighted_values(attn_weights, value, cfg, aqt_rng)


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
      use_rotary_position_emb: apply rotary_position_emb to query and key if True.
      use_qk_norm: apply normalizations on query and key if True.
      query_scale_style: the style to apply query scale one of ('init', 'post')
        and defaults to 'init'. 'init' means initializing query projection layer weights
        with depth scaling in T5. 'post' means applying post depth scaling to query while
        initializing query projection layer without depth scaling used in GPT3.
  """

  num_heads: int
  head_dim: int
  config: Config
  mesh: Mesh
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal')
  float32_logits: bool = False  # computes logits in float32 for stability.
  use_rotary_position_emb: bool = True
  use_qk_norm: bool = True
  query_scale_style: str = 'init'
  combined_qkv: bool = False

  def apply_attention(
      self,
      query,
      key,
      value,
      attention_type,
      decoder_segment_ids,
      attention_bias,
      dropout_rng,
      deterministic,
      decode,
  ):
    """Apply Attention."""
    if attention_type == 'flash':
      if decode:
        raise ValueError("""Decode not supported with flash attention.
                             Use MHA instead.""")
      # reshaped to ('batch', 'heads', 'length', 'kv')
      query = jax.numpy.transpose(query, axes=(0, 2, 1, 3))
      key = jax.numpy.transpose(key, axes=(0, 2, 1, 3))
      value = jax.numpy.transpose(value, axes=(0, 2, 1, 3))
      if decoder_segment_ids is not None:
        decoder_segment_ids = flash_attention.SegmentIds(
            decoder_segment_ids, decoder_segment_ids
        )
      axis_names = nn.logical_to_mesh_axes((
          'activation_batch',
          'activation_heads',
          'activation_length',
          'activation_kv',
      ))
      segment_axis_names = nn.logical_to_mesh_axes(
          ('activation_batch', 'activation_length_no_heads')
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
              == self.config.max_target_length
              == decoder_segment_ids.q.shape[1]
          ), 'Sharding along sequence dimension not allowed in flash attention'
        return flash_attention.flash_attention(
            query,
            key,
            value,
            causal=True,
            segment_ids=decoder_segment_ids,
            block_sizes=flash_attention.BlockSizes(
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
          'Batch dimension should be shardable among the devices in data and'
          ' fsdp axis'
      )
      x = wrap_flash_attention(query, key, value, decoder_segment_ids)
      x = jax.numpy.transpose(x, axes=(0, 2, 1, 3))
    else:
      aqt_rng = self.make_rng('aqt')
      x = dot_product_attention(
          query,
          key,
          value,
          bias=attention_bias,
          dropout_rng=dropout_rng,
          dropout_rate=self.dropout_rate,
          aqt_rng=aqt_rng,
          deterministic=deterministic,
          dtype=self.dtype,
          float32_logits=self.float32_logits,
          cfg=self.config)
    return x

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               attention_type: str,
               decoder_segment_ids: Optional[Array] = None,
               inputs_positions: Optional[Array] = None,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> Array:
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
    cfg = self.config


    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    if self.combined_qkv:
      combined_projection = functools.partial(
          DenseGeneral,
          axis=-1,
          features=(self.num_heads, self.head_dim, 3),
          kernel_axes=('embed', 'heads', 'kv', 'qkv'),
          dtype=self.dtype,
          use_bias=cfg.use_bias_linear,
          config=cfg)

      # batch, length, heads, kv, 3 -> 3, batch, length, heads, kv
      query, key, value = jnp.moveaxis(combined_projection(kernel_init=self.kernel_init, name='combined_qkv')(inputs_q), 0, -1)
    else:
      projection = functools.partial(
          DenseGeneral,
          axis=-1,
          features=(self.num_heads, self.head_dim),
          kernel_axes=('embed', 'heads', 'kv'),
          dtype=self.dtype,
          use_bias=cfg.use_bias_linear,
          config=cfg)
      if self.query_scale_style == 'init':
        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor.
        def query_init(*args):
          # pylint: disable=no-value-for-parameter
          return self.kernel_init(*args) / depth_scaling

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]
        query = projection(kernel_init=query_init, name='query')(inputs_q)
      else:
        query = projection(kernel_init=self.kernel_init, name='query')(inputs_q)
      key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
      value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

    # Apply RoPE if True
    if self.use_rotary_position_emb:
      query = LLaMARotaryEmbedding(
          embedding_dims=self.head_dim, name='query_rotary'
      )(inputs=query, position=inputs_positions)
      key = LLaMARotaryEmbedding(embedding_dims=self.head_dim, name='key_rotary')(
          inputs=key, position=inputs_positions
      )

    if self.use_qk_norm:
      # Layer norms here prevent (near) one-hot softmaxes, which can lead to
      # unstable training loss and nans, see the "QK Normalization" subsection in
      # https://arxiv.org/pdf/2302.05442.pdf.
      query = RMSNorm(
          dtype=self.dtype,
          name='query_norm',
          kernel_axes=('heads',),
          use_bias=cfg.use_bias_layer_norm,
          use_mean_center=cfg.use_mean_center_layer_norm,
          reductions_in_fp32=cfg.reductions_in_fp32_layer_norm,
          epsilon=cfg.epsilon_layer_norm,
          )(query)
      key = RMSNorm(
          dtype=self.dtype,
          name='key_norm',
          kernel_axes=('heads',),
          use_bias=cfg.use_bias_layer_norm,
          use_mean_center=cfg.use_mean_center_layer_norm,
          reductions_in_fp32=cfg.reductions_in_fp32_layer_norm,
          epsilon=cfg.epsilon_layer_norm,
          )(key)
      value = RMSNorm(
          dtype=self.dtype,
          name='value_norm',
          kernel_axes=('heads',),
          use_bias=cfg.use_bias_layer_norm,
          use_mean_center=cfg.use_mean_center_layer_norm,
          reductions_in_fp32=cfg.reductions_in_fp32_layer_norm,
          epsilon=cfg.epsilon_layer_norm,
          )(value)

    if self.query_scale_style == 'post':
      # NOTE: Different from initializing query projection layer weights with
      #       depth scaling in T5, GPT3 applies rescaling on query logits with
      #       1/sqrt(depth_kq).
      query /= depth_scaling
    query = nn.with_logical_constraint(
        query,
        (
            'activation_batch',
            'activation_length',
            'activation_heads',
            'activation_kv',
        ),
    )
    query = checkpoint_name(query, 'query_proj')
    key = nn.with_logical_constraint(
        key,
        (
            'activation_batch',
            'activation_length',
            'activation_heads',
            'activation_kv',
        ),
    )
    key = checkpoint_name(key, 'key_proj')
    value = nn.with_logical_constraint(
        value,
        (
            'activation_batch',
            'activation_length',
            'activation_heads',
            'activation_kv',
        ),
    )
    value = checkpoint_name(value, 'value_proj')

    if decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')

      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.
      def swap_dims(x):
        return x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        batch, num_heads, head_dim, length = cached_key.value.shape
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # Sanity shape check of cached key against input query.
        expected_shape = (batch, 1, num_heads, head_dim)
        if expected_shape != query.shape:
          raise ValueError(f"""Autoregressive cache shape error,
                           expected query shape %s instead got
                           {(expected_shape, query.shape)}""")
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

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, get_large_negative_number(self.dtype)))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = self.apply_attention(
        query,
        key,
        value,
        attention_type,
        decoder_segment_ids,
        attention_bias,
        dropout_rng,
        deterministic,
        decode=decode,
    )
    x = nn.with_logical_constraint(
        x,
        (
            'activation_batch',
            'activation_length',
            'activation_heads',
            'activation_kv',
        ),
    )

    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=('heads', 'kv', 'embed'),
        dtype=self.dtype,
        name='out',
        use_bias=cfg.use_bias_linear,
        config=cfg)(
            x)
    return out
