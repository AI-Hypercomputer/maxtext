"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from aqt.jax.v2 import aqt_dot_general as aqt
from aqt.jax.v2.config import config_v3
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh

import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union


from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import numpy as np

import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention




withLP = nn.with_logical_partitioning
ScanIn = nn_partitioning.ScanIn

Config = Any

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[
    [PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)

def get_aqt_cfg():
  return config_v3(
    fwd_bits=8,
    dlhs_bits=8,
    drhs_bits=None,
    rng_type='jax.uniform',
    dlhs_local_aqt = None,
    drhs_local_aqt = None,
    fwd_accumulator_dtype = jnp.int32,
    dlhs_accumulator_dtype = jnp.int32,
    drhs_accumulator_dtype = jnp.int32,
  )

#------------------------------------------------------------------------------
# DenseGeneral for attention layers.
#------------------------------------------------------------------------------


def nd_dense_init(scale, mode, distribution):
  """Initializer with in_axis, out_axis set at call time."""
  def init_fn(key, shape, dtype, in_axis, out_axis):
    fn = jax.nn.initializers.variance_scaling(
        scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)
  return init_fn


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
  """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  features: Union[Iterable[int], int]
  config: Config
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    def compute_dot_general(inputs, kernel, axis, contract_ind, cfg):
      """Computes a dot_general operation that may be quantized as determined by cfg options"""
      if not cfg.int8_training:
        return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))
      else:
        aqt_key = self.make_rng('aqt')
        aqt_cfg = get_aqt_cfg()
        aqt_dot_general = aqt.make_dot_general(aqt_cfg)
        context = aqt.Context(key=aqt_key, train_step=None)
        return aqt_dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), context=context)

    cfg = self.config
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    kernel = self.param(
        'kernel',
        withLP(self.kernel_init, self.kernel_axes),
        kernel_shape,
        jnp.float32,
        kernel_in_axis,
        kernel_out_axis)
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    return compute_dot_general(inputs, kernel, axis, contract_ind, cfg)

def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(f"""Don't know how to convert {fn_or_string}
                         to an activation function""")


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
  """

  num_heads: int
  head_dim: int
  config: Config
  mesh: Mesh
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal')

  def apply_attention(self, query, key, value, attention_type,
                      decoder_segment_ids, dropout_rng, deterministic, model_mode):
    """ Apply Attention
    """
    if decoder_segment_ids is not None:
        decoder_segment_ids = tpu_flash_attention.SegmentIds(decoder_segment_ids, decoder_segment_ids)
  
    if attention_type == 'flash':
      if model_mode != "train":
        raise ValueError("""Decode not supported with flash attention.
                             Use MHA instead.""")
      # reshaped to ('batch', 'heads', 'length', 'kv')
      query = jax.numpy.transpose(query, axes = (0,2,1,3))
      key = jax.numpy.transpose(key, axes = (0,2,1,3))
      value = jax.numpy.transpose(value, axes = (0,2,1,3))
      axis_names = nn.logical_to_mesh_axes(('activation_batch', 'activation_heads', 'activation_length', 'activation_kv'))
      segment_axis_names = nn.logical_to_mesh_axes(('activation_batch', 'activation_length_no_heads'))
      @functools.partial(shard_map, mesh = self.mesh, in_specs = (
          axis_names,
          axis_names,
          axis_names,
          segment_axis_names,
      ), out_specs = axis_names, check_rep=False)
      def wrap_flash_attention(query, key, value, decoder_segment_ids):
        if decoder_segment_ids is not None:
          assert query.shape[2] == self.config.max_target_length == decoder_segment_ids.q.shape[1], \
            "Sharding along sequence dimension not allowed in flash attention"
        return tpu_flash_attention.flash_attention(
              query,
              key,
              value,
              causal = True,
              segment_ids = decoder_segment_ids,
              block_sizes = tpu_flash_attention.BlockSizes(
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

              )
            )
      devices_in_data_fsdp = self.mesh.shape['data'] * self.mesh.shape['fsdp']
      assert (query.shape[0]/devices_in_data_fsdp).is_integer(), \
              'Batch dimension should be shardable among the devices in data and fsdp axis'
      x = wrap_flash_attention(query, key, value, decoder_segment_ids)
      x = jax.numpy.transpose(x, axes = (0,2,1,3))
    else:
      query = jax.numpy.transpose(query, axes = (0,2,1,3))
      key = jax.numpy.transpose(key, axes = (0,2,1,3))
      value = jax.numpy.transpose(value, axes = (0,2,1,3))
      x = tpu_flash_attention.mha_reference(
          query,
          key,
          value,
          None,
          causal = True,
          segment_ids = decoder_segment_ids
      )
      x = jax.numpy.transpose(x, axes = (0,2,1,3))
    return x

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               attention_type,
               decoder_segment_ids = None,
               inputs_positions:Optional[Array] = None,
               *,
               model_mode: [str] = "train",
               deterministic: bool = False) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are three modes: 'train', 'prefill' and 'autoregressive'. The mode is
    determined by `model_mode` argument. For decoding, this method is called twice,
    first to initialize the cache ('prefill') and then for an actual decoding process
    ('autoregressive').

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      model_mode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    cfg = self.config

    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_axes=('embed', 'heads', 'kv'),
        dtype=self.dtype,
        config=cfg)

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    def query_init(*args):
      #pylint: disable=no-value-for-parameter
      return self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_init, name='query')(inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

    #Apply RoPE
    query = LLaMARotaryEmbedding(embedding_dims=self.head_dim,
                                 name='query_rotary'
                                 )(inputs=query, position=inputs_positions)
    key = LLaMARotaryEmbedding(embedding_dims=self.head_dim,
                               name='key_rotary'
                               )(inputs=key, position=inputs_positions)

    # Layer norms here prevent (near) one-hot softmaxes, which can lead to
    # unstable training loss and nans, see the "QK Normalization" subsection in
    # https://arxiv.org/pdf/2302.05442.pdf.
    query = LayerNorm(dtype=self.dtype, name='query_layer_norm', kernel_axes = ('heads',))(query)
    key = LayerNorm(dtype=self.dtype, name='key_layer_norm', kernel_axes = ('heads',))(key)
    value = LayerNorm(dtype=self.dtype, name='value_layer_norm', kernel_axes = ('heads',))(value)

    query = nn.with_logical_constraint(
        query, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv')
    )
    query = checkpoint_name(query, 'query_proj')
    key = nn.with_logical_constraint(key, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv'))
    key = checkpoint_name(key, 'key_proj')
    value = nn.with_logical_constraint(
        value, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv')
    )
    value = checkpoint_name(value, 'value_proj')

    if model_mode == "autoregressive":
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

        # Assign an index for decoding.
        decoder_segment_ids = jnp.ones((key.shape[0], 1) ) * cur_index


    if model_mode == "prefill":
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')



    # Apply attention.
    x = self.apply_attention(query, key, value, attention_type,
                              decoder_segment_ids, dropout_rng, deterministic, model_mode=model_mode)
    x = nn.with_logical_constraint(
        x, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv')
    )

    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=('heads', 'kv', 'embed'),
        dtype=self.dtype,
        name='out',
        config=cfg)(
            x)
    return out





class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  config: Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseGeneral(
          self.intermediate_dim,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          kernel_axes=('embed', 'mlp'),
          name=dense_name,
          config=cfg)(
              inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)  # Broadcast along length.
    x = nn.with_logical_constraint(x, ('activation_batch', 'activation_length', 'activation_mlp'))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('mlp', 'embed'),
        name='wo',
        config=cfg)(
            x)
    return output



#------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
#------------------------------------------------------------------------------

class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  kernel_axes: Tuple[str, ...] = ()
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.param(
        'scale', withLP(self.scale_init, self.kernel_axes), (features,), jnp.float32)

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
  """
  # pylint: disable=attribute-defined-outside-init
  config: Config
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init

  def setup(self):
    self.embedding = self.param(
        'embedding',
        withLP(self.embedding_init, ('vocab', 'embed')),
        (self.num_embeddings, self.features),
        jnp.float32)

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    cfg = self.config
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')

    if cfg.use_iota_embed:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
    output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'activation_embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)

class LLaMARotaryEmbedding(nn.Module):
  """LLaMA variant of ROPE where inputs are split in a different way.

  Attributes:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
  """
  min_timescale: int = 1
  max_timescale: int = 10_000
  embedding_dims: int = 0
  cast_as_fprop_dtype: bool = True
  fprop_dtype: DType = jnp.bfloat16

  def setup(self) -> None:
    if self.embedding_dims % 2:
      raise ValueError(
          'Embedding dim for rotary position embedding must be a multiple of 2.'
      )

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: jax.Array,
      position: Optional[jax.Array] = None,
  ) -> jax.Array:
    """Generates a jax.Array of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position jax.Array which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a jax.Array of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          'Input is assumed to be a rank 4 tensor of shape'
          '[batch, sequence, heads, dims].'
      )
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          'The embedding dims of the rotary position embedding'
          'must match the hidden dimension of the inputs.'
      )
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = (
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** fraction
    )
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    reshape_tensor = inputs.astype(jnp.float32).reshape(
        *inputs.shape[:-1], -1, 2
    )
    first_half = reshape_tensor[..., 0]
    second_half = reshape_tensor[..., 1]
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    x_out = jnp.stack((first_part, second_part), axis=-1).reshape(
        *first_part.shape[:-1], -1
    )
    return x_out

#------------------------------------------------------------------------------
# The network: Decoder & Transformer Definitions
#------------------------------------------------------------------------------


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: Config
  mesh: Mesh

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
               max_decode_length):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(inputs, ('activation_batch', 'activation_length', 'activation_embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm', kernel_axes=('embed',))(
            inputs)
    lnx = nn.with_logical_constraint(lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    # Self-attention block
    attention_lnx = MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='self_attention',
        config=cfg,
        mesh = mesh)(
            lnx,
            lnx,
            attention_type=cfg.attention,
            decoder_segment_ids=decoder_segment_ids,
            inputs_positions=decoder_positions,
            deterministic=deterministic,
            model_mode=model_mode)
    attention_lnx = nn.with_logical_constraint(attention_lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
        config=cfg,
    )(lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            next_layer_addition, deterministic=deterministic)

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = nn.with_logical_constraint(layer_output, ('activation_batch', 'activation_length', 'activation_embed'))

    if cfg.record_internal_nn_metrics:
      self.sow('intermediates', 'activation_mean', jnp.mean(layer_output))
      self.sow('intermediates', 'activation_stdev', jnp.std(layer_output))
      self.sow('intermediates', 'activation_fraction_zero', jnp.sum(layer_output==0) / jnp.size(layer_output))

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output



class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: Config
  shared_embedding: nn.Module
  mesh: Mesh

  @nn.compact
  def __call__(self,
               decoder_input_tokens,
               decoder_segment_ids=None,
               decoder_positions=None,
               deterministic=False,
               model_mode="train",
               max_decode_length=None):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    BlockLayer = DecoderLayer

    if cfg.remat_policy != 'none':
      if cfg.remat_policy == 'minimal':
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      elif cfg.remat_policy == 'proj':
        policy = jax.checkpoint_policies.save_only_these_names(
            'query_proj', 'value_proj', 'key_proj'
        )
      else:
        assert cfg.remat_policy == 'full', "Remat policy needs to be on list of remat policies"
        policy = None
      BlockLayer = nn.remat(  # pylint: disable=invalid-name
          BlockLayer,
          prevent_cse=not cfg.scan_layers,
          policy=policy,
          static_argnums=(-1, -2, -3, -4, -5))
    if cfg.scan_layers:
      initializing = self.is_mutable_collection('params')
      params_spec = (
          cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis))
      cache_spec = 0
      y, _ = nn.scan(
          BlockLayer,
          variable_axes={
              'params': params_spec,
              'cache': cache_spec,
              'intermediates': 0
          },
          split_rngs={
              'params': True,
              'dropout': cfg.enable_dropout,
              'aqt': cfg.int8_training
          },
          in_axes=(nn.broadcast, nn.broadcast,
                   nn.broadcast, nn.broadcast, nn.broadcast),
          length=cfg.num_decoder_layers,
          metadata_params={nn.PARTITION_NAME: 'layers'})(
              config=cfg, mesh=mesh,
              name='decoder')(y, decoder_segment_ids, decoder_positions,
                              deterministic, model_mode, max_decode_length)
    else:
      for lyr in range(cfg.num_decoder_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = BlockLayer(
            config=cfg, mesh = mesh, name=f'layers_{lyr}')(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                max_decode_length)

    y = LayerNorm(dtype=cfg.dtype, name='decoder_norm', kernel_axes = ('embed',))(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense',
          config=cfg)(
              y)
    logits = nn.with_logical_constraint(logits, ('activation_batch', 'activation_length', 'activation_vocab'))
    return logits


class Transformer(nn.Module):
  """An decoder-only Transformer model."""
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh

  def setup(self):
    """Initialize shared_embedding, decoder"""
    cfg = self.config
    mesh = self.mesh
    self.shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='token_embedder',
        config=cfg)

    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding, mesh = mesh)

  def __call__(
      self,
      decoder_input_tokens,
      decoder_target_tokens,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      model_mode="train",
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    assert model_mode in ["train", "autoregressive", "prefill"]
    cfg = self.config

    if decoder_segment_ids is not None and model_mode=="autoregressive":
      raise ValueError(
          'During decoding, packing should not be used but '
          '`decoder_segment_ids` was passed to `Transformer.decode`.')

    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        model_mode=model_mode,
        max_decode_length=max_decode_length)
    return logits
