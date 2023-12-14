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

"""Transformer models."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Callable, Optional


from flax import linen as nn
import jax
import jax.numpy as jnp
import common_types
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
LLaMARotaryEmbedding = embeddings.LLaMARotaryEmbedding
MultiHeadDotProductAttention = attentions.MultiHeadDotProductAttention
RMSNorm = normalizations.RMSNorm


#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: DType = jnp.float32) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
      # [batch, len_q] -> [batch, len_q, 1]
      jnp.expand_dims(query_input, axis=-1),
      # [batch, len_q] -> [batch, 1, len_kv]
      jnp.expand_dims(key_input, axis=-2))

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
  will be `[batch, heads, len, len]` and this function will produce a
  causal mask of shape `[batch, 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch, len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: DType,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None
                      ) -> Optional[Array]:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]
    decoder_segment_ids: decoder segmentation info for packed examples. [batch,
      length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attneition` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, length, length]
    inputs_mask = make_attention_mask(
        decoder_causal_attention,
        decoder_causal_attention,
        jnp.logical_and,
        dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      make_attention_mask(
          decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
        make_attention_mask(
            decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

  return attentions.combine_masks(*masks, dtype=dtype)



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
               decoder_mask,
               deterministic,
               decode,
               max_decode_length):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = RMSNorm(
        dtype=cfg.dtype, 
        name='pre_self_attention_norm', 
        kernel_axes=('embed',)
        use_bias=cfg.use_bias_layer_norm,
        use_mean_center=cfg.use_mean_center_layer_norm,
        reductions_in_fp32=cfg.reductions_in_fp32_layer_norm,
        )(inputs)
    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    # Self-attention block
    attention_lnx = MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        use_rotary_position_emb=cfg.use_rotary_position_emb,
        use_qk_norm=cfg.use_qk_norm,
        name='self_attention',
        config=cfg,
        mesh=mesh)(
            lnx,
            lnx,
            attention_type=cfg.attention,
            decoder_segment_ids=decoder_segment_ids,
            inputs_positions=decoder_positions,
            mask=decoder_mask,
            bias=None,
            deterministic=deterministic,
            decode=decode)
    attention_lnx = nn.with_logical_constraint(
        attention_lnx,
        ('activation_batch', 'activation_length', 'activation_embed'))

    # MLP block.
    mlp_lnx = linears.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
        use_pre_norm=cfg.use_pre_norm_mlp,
        config=cfg,
    )(lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(
        mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
    )

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,)
    )(next_layer_addition, deterministic=deterministic)

    layer_output = next_layer_addition_dropped_out + inputs
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
               decoder_mask=None,
               deterministic=False,
               decode=False,
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
          cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis)
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

    y = RMSNorm(
        dtype=cfg.dtype, 
        name='decoder_norm', 
        kernel_axes=('embed',)
        use_bias=cfg.use_bias_layer_norm,
        use_mean_center=cfg.use_mean_center_layer_norm,
        reductions_in_fp32=cfg.reductions_in_fp32_layer_norm,
        )(y)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        y, deterministic=deterministic
    )

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = linears.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense',
          config=cfg)(y)
    logits = nn.with_logical_constraint(
        logits, ('activation_batch', 'activation_length', 'activation_vocab'))
    return logits


class Transformer(nn.Module):
  """An decoder-only Transformer model."""
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh

  def setup(self):
    """Initialize shared_embedding & decoder layers."""

    cfg = self.config
    mesh = self.mesh
    self.shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='token_embedder',
        embed_lookup_style=cfg.embed_lookup_style,
        config=cfg,
    )

    self.decoder = Decoder(
        config=cfg, shared_embedding=self.shared_embedding, mesh=mesh
    )

  def __call__(
      self,
      decoder_input_tokens,
      decoder_target_tokens,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
    else:
      decoder_mask = make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=cfg.dtype,
          decoder_segment_ids=decoder_segment_ids)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if decoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        decoder_mask=decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)
    return logits