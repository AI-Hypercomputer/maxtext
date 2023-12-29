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
from jax import random
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
# from jax.experimental.pallas.ops.tpu import flash_attention
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations

from layers import models



class MultiHeadDotProductAttention(models.MultiHeadDotProductAttention):
  """Multi-head dot-product attention for Llama2.

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

  @nn.compact
  def __call__(self,
               inputs_q: models.Array,
               inputs_kv: models.Array,
               attention_type,
               decoder_segment_ids = None,
               inputs_positions:Optional[models.Array] = None,
               mask: Optional[models.Array] = None,
               bias: Optional[models.Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> models.Array:
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
    # if decode:

    #   # Detect if we're initializing by absence of existing cache data.
    #   is_initialized = self.has_variable('cache', 'cached_key')

    #   cache_index = self.variable('cache', 'cache_index',
    #                               lambda: jnp.array(0, dtype=jnp.int32))
    #   # jax.debug.print("Anisha: is_initialized = {is_initialized}", is_initialized=is_initialized)
    #   if is_initialized and inputs_positions is None:
    #     cur_index = cache_index.value.item().astype(int)
    #     inputs_positions = jnp.arange(cur_index, cur_index+1, dtype=jnp.float32)[jnp.newaxis, :]
    #     jax.debug.print("Anisha is_initialized cur_index {cc}", cc=cur_index)
    
    cfg = self.config

    projection = functools.partial(
        linears.DenseGeneral,
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

    # query = projection( name='query')(inputs_q)
    # key = projection(name='key')(inputs_kv)
    # value = projection(name='value')(inputs_kv)
    # jax.debug.print("Anisha: shape query, key, value {qkv}", qkv=[query.shape, key.shape, value.shape])
    # jax.debug.print("Anisha: query, key, value {qkv}", qkv=[query, key, value])
    #Apply RoPE
    query = models.LLaMARotaryEmbedding(embedding_dims=self.head_dim,
                                 name='query_rotary', fprop_dtype=jnp.float32
                                 )(inputs=query, position=inputs_positions)
    key = models.LLaMARotaryEmbedding(embedding_dims=self.head_dim,
                               name='key_rotary', fprop_dtype=jnp.float32
                               )(inputs=key, position=inputs_positions)
    # jax.debug.print("Anisha: after rotary shapes of query, key, {qk}", qk=[query.shape, key.shape])
    # jax.debug.print("Anisha: after rotary query, key, {qk}", qk=[query, key])

    # Layer norms here prevent (near) one-hot softmaxes, which can lead to
    # unstable training loss and nans, see the "QK Normalization" subsection in
    # https://arxiv.org/pdf/2302.05442.pdf.
    #Llama architecture doesn't have these layernorms
    # query = LayerNorm(dtype=self.dtype, name='query_layer_norm', kernel_axes = ('heads',))(query)
    # key = LayerNorm(dtype=self.dtype, name='key_layer_norm', kernel_axes = ('heads',))(key)
    # value = LayerNorm(dtype=self.dtype, name='value_layer_norm', kernel_axes = ('heads',))(value)

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
      # jax.debug.print("Anisha: is_initialized = {is_initialized}", is_initialized=is_initialized)
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
        mask = attentions.combine_masks(
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
          bias = attentions.dynamic_vector_slice_in_dim(
              jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = attentions.combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')
    # jax.debug.print("Anisha: attention_bias = {attention_bias}, mask = {mask}", attention_bias=attention_bias, mask=mask)
    # Apply attention.
    x = self.apply_attention(query, key, value, attention_type,
                              decoder_segment_ids, attention_bias, dropout_rng, deterministic, decode=decode)
    x = nn.with_logical_constraint(
        x, ('activation_batch', 'activation_length', 'activation_heads', 'activation_kv')
    )

    # Back to the original inputs dimensions.
    out = linears.DenseGeneral(
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
  config: models.Config
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: linears.NdInitializer = linears.nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    # activations = []
    # for idx, act_fn in enumerate(self.activations):
    #   dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
  
    #   activations.append(x)
    gate_mlp = linears.DenseGeneral(
      self.intermediate_dim,
      dtype=self.dtype,
      kernel_init=self.kernel_init,
      kernel_axes=('embed', 'mlp'),
      name="wi",
      config=cfg)
    
    x = gate_mlp(
          inputs)
    x = linears._convert_to_activation_function(self.activations[0])(x)
    # jax.debug.print("Anisha: inp_of_ln_lnx =  {inputs} ", inputs = inputs)
    # Take elementwise product of above intermediate activations.
    #x = functools.reduce(operator.mul, activations)

    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)  # Broadcast along length.
    x = nn.with_logical_constraint(x, ('activation_batch', 'activation_length', 'activation_mlp'))

    up_proj_x = linears.DenseGeneral(
          self.intermediate_dim,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          kernel_axes=('embed', 'mlp'),
          name="ffn_layer1",
          config=cfg)(
              inputs)
    x = jnp.multiply(x, up_proj_x)

    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)  # Broadcast along length.
    x = nn.with_logical_constraint(x, ('activation_batch', 'activation_length', 'activation_mlp'))
    output = linears.DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('mlp', 'embed'),
        name='wo',
        config=cfg)(
            x)
    return output



#------------------------------------------------------------------------------
# The network: Decoder & Transformer Definitions
#------------------------------------------------------------------------------


class DecoderLayer(models.DecoderLayer):
  """Transformer decoder layer that attends to the encoder."""
  config: models.Config
  mesh: Mesh
  name: str

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               decoder_mask,
               deterministic,
               decode,
               max_decode_length):
    cfg = self.config
    mesh = self.mesh
    name = self.name

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    #input_layernorm aka pre_self_attention_layer_norm
    #jax.debug.print(name+"Anisha: inp_of_ln_lnx =  {inputs} ", inputs = inputs)
    
    lnx_rms = models.RMSNorm(
        dtype=cfg.dtype, 
        name='pre_self_attention_layer_norm', 
        kernel_axes=('embed',),
        epsilon=1e-05
        )
    lnx = lnx_rms(inputs)
    #jax.debug.print("Anisha: ln eps =  {eps} ", eps = lnx_rms.epsilon)
    # jax.debug.print("Anisha: inp_of_attention_lnx =  {lnx} ", lnx = lnx)
    
    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

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
            mask=decoder_mask,
            bias = None,
            deterministic=deterministic,
            decode=decode)
    attention_lnx = nn.with_logical_constraint(
        attention_lnx,
        ('activation_batch', 'activation_length', 'activation_embed'))
    #jax.debug.print("Anisha: attention_lnx =  {attention_lnx} ", attention_lnx = attention_lnx)
    #jax.debug.print("Anisha: inputs again =  {inputs} ", inputs = inputs)
    intermediate_inputs = inputs + attention_lnx
    #jax.debug.print("Anisha: attention + residual =  {intermediate_inputs} ", intermediate_inputs = intermediate_inputs)
    
    # Fully Connected
    hidden_states = models.RMSNorm(
        dtype=cfg.dtype, name='post_self_attention_layer_norm', kernel_axes=('embed',),
        epsilon=1e-05
        )(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, ('activation_batch', 'activation_length', 'activation_embed'))
    
    #jax.debug.print("Anisha: input of mlp_lnx =  {hidden_states} ", hidden_states = hidden_states)
    # MLP block.
    mlp_lnx = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
        config=cfg,
    )(hidden_states, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(
        mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
    )

    # jax.debug.print("Anisha: inp, ln(inp), attn, resi, ln(resi), mlp_lnx =  {mlp_lnx} ", 
    #                 mlp_lnx = [inputs, lnx, attention_lnx, intermediate_inputs, hidden_states,mlp_lnx ])


    layer_output = mlp_lnx + intermediate_inputs

    layer_output = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ('activation_batch', 'activation_length', 'activation_embed'),
    )

    if cfg.record_internal_nn_metrics:
      self.sow('intermediates', 'activation_mean', jnp.mean(layer_output))
      self.sow('intermediates', 'activation_stdev', jnp.std(layer_output))
      self.sow(
          'intermediates',
          'activation_fraction_zero',
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: models.Config
  shared_embedding: nn.Module
  mesh: Mesh

  @nn.compact
  def __call__(self,
               decoder_input_tokens,
               decoder_segment_ids=None,
               decoder_positions=None,
               decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    # jax.debug.print("Anisha: decoder_input_tokens = {decoder_input_tokens} ", decoder_input_tokens=decoder_input_tokens)
    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    # jax.debug.print("Anisha: y embedding size = {shape}", shape=y.shape)
    # jax.debug.print("Anisha: y embedding =  {y} ", y=y)
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
        assert (
            cfg.remat_policy == 'full'
        ), 'Remat policy needs to be on list of remat policies'
        policy = None
      BlockLayer = nn.remat(  # pylint: disable=invalid-name
          BlockLayer,
          prevent_cse=not cfg.scan_layers,
          policy=policy,
          static_argnums=(-1, -2, -3, -4, -5),
      )
    if cfg.scan_layers:
      initializing = self.is_mutable_collection('params')
      params_spec = (
          cfg.param_scan_axis if initializing else models.ScanIn(cfg.param_scan_axis)
      )
      cache_spec = 0
      y, _ = nn.scan(
          BlockLayer,
          variable_axes={
              'params': params_spec,
              'cache': cache_spec,
              'intermediates': 0,
          },
          split_rngs={
              'params': True,
              'dropout': cfg.enable_dropout,
              'aqt': cfg.int8_training,
          },
          in_axes=(
              nn.broadcast,
              nn.broadcast,
              nn.broadcast,
              nn.broadcast,
              nn.broadcast,
              nn.broadcast,
          ),
          length=cfg.num_decoder_layers,
          metadata_params={nn.PARTITION_NAME: 'layers'},
      )(config=cfg, mesh=mesh, name='decoder')(
          y,
          decoder_segment_ids,
          decoder_positions,
          decoder_mask,
          deterministic,
          decode,
          max_decode_length,
      )
    else:
      for lyr in range(cfg.num_decoder_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = BlockLayer(config=cfg, mesh=mesh, name=f'layers_{lyr}')(
            y,
            decoder_segment_ids,
            decoder_positions,
            decoder_mask,
            deterministic,
            decode,
            max_decode_length,
        )

    # jax.debug.print("Anisha after all decoder layers = {y}",y = y)
    y = models.RMSNorm(dtype=cfg.dtype, name='decoder_norm', epsilon=1e-05, 
                kernel_axes=('embed',))(y)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        y, deterministic=deterministic
    )
    jax.debug.print("Anisha after ln of last decoder layer = {y}",y = y)
    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    logits = linears.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense',
          config=cfg)(y)
    logits = nn.with_logical_constraint(
        logits, ('activation_batch', 'activation_length', 'activation_vocab'))
    jax.debug.print("Anisha logits = {logits}",logits = logits)

    return logits



class Transformer(models.Transformer):
  """An decoder-only Transformer model."""
  # pylint: disable=attribute-defined-outside-init
  config: models.Config
  mesh: Mesh

  def setup(self, decoder = None):
    """Initialize shared_embedding & decoder layers."""

    cfg = self.config
    mesh = self.mesh
    self.shared_embedding = models.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='token_embedder',
        config=cfg,
    )

    self.decoder = Decoder(
        config=cfg, shared_embedding=self.shared_embedding, mesh=mesh)
    