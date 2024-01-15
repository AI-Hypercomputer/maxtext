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
import functools
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
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm
LayerNorm = normalizations.LayerNorm
PositionalEmbedding = embeddings.PositionalEmbedding

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
               padding_mask,
               deterministic,
               model_mode,
              ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = RMSNorm(
        dtype=cfg.dtype,
        name='pre_self_attention_norm',
        epsilon=cfg.norm_epsilon,
        kernel_axes=('embed',))(inputs)
    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    attention_layer = Attention(
      config = self.config,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      attention_kernel=cfg.attention,
      mesh=mesh,
      dtype=cfg.dtype,
      dropout_rate=cfg.dropout_rate,
      name='self_attention',
      use_int8=cfg.int8_training)


    attention_lnx = attention_layer(
      lnx,
      lnx,
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=deterministic,
      model_mode=model_mode)

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

  def get_decoder_layer(self):
    if self.config.model_name == "default":
      return DecoderLayer
    elif self.config.model_name.startswith("llama2"):
      from layers import llama2
      return llama2.LlamaDecoderLayer
    elif self.config.model_name.startswith("mistral"):
      # TODO(ranran): update to Mistral with sliding window attention
      from layers import llama2
      return llama2.LlamaDecoderLayer
    elif self.config.model_name.startswith("gamma"):
      from layers import gamma
      return gamma.GammaDecoderLayer
    elif self.config.model_name.startswith("gpt3"):
      from layers import gpt3
      return gpt3.Gpt3DecoderLayer
    else:
      raise ValueError(f"Incorrect model name {self.config.model_name=}")

  def get_norm_layer(self):
    if self.config.model_name == "default" or \
        self.config.model_name.startswith(("llama2", "mistral", "gamma")):
      return RMSNorm
    elif self.config.model_name.startswith("gpt3"):
      return functools.partial(LayerNorm, reductions_in_fp32=False, use_bias=True)
    else:
      raise ValueError(f"Incorrect model name {self.config.model_name=}")


  @nn.compact
  def __call__(self,
               decoder_input_tokens,
               decoder_positions,
               decoder_segment_ids=None,
               padding_mask=None,
               deterministic=False,
               model_mode=common_types.MODEL_MODE_TRAIN,
              ):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_positional_embedding:
      if cfg.trainable_position_size:
        y += Embed(
          num_embeddings=cfg.trainable_position_size,
          features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name='position_embedder',
          config=cfg)(decoder_positions)
      else:
        y = PositionalEmbedding(cfg.base_emb_dim)(y, decoder_positions)

    BlockLayer = self.get_decoder_layer()

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
          static_argnums=(-1, -2, -3, -4, -5, -6),
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
          ),
          length=cfg.num_decoder_layers,
          metadata_params={nn.PARTITION_NAME: 'layers'},
      )(config=cfg, mesh=mesh, name='layers')(
          y,
          decoder_segment_ids,
          decoder_positions,
          padding_mask,
          deterministic,
          model_mode,
      )
    else:
      for lyr in range(cfg.num_decoder_layers):
        y = BlockLayer(config=cfg, mesh=mesh, name=f'layers_{lyr}')(
            y,
            decoder_segment_ids,
            decoder_positions,
            padding_mask,
            deterministic,
            model_mode,
        )

    norm_layer = self.get_norm_layer()
    y = norm_layer(dtype=cfg.dtype, name='decoder_norm', epsilon=cfg.norm_epsilon, kernel_axes=('embed',))(y)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        y, deterministic=deterministic
    )

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      if self.config.norm_logits_via_embedding:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = linears.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
          kernel_axes=('embed', 'vocab'),
          name='logits_dense',
          use_int8=cfg.int8_training)(y)
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
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name='token_embedder',
        config=cfg,
    )

    self.decoder = Decoder(
        config=cfg, shared_embedding=self.shared_embedding, mesh=mesh
    )

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      padding_mask=None,
      enable_dropout=True,
      model_mode=common_types.MODEL_MODE_TRAIN
  ):
    """Applies Transformer decoder-branch on encoded-input and target."""

    if decoder_segment_ids is not None and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
        f'During autoregressive decoding we assume the tokens are in the active sequence'
        f' which is always {common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR}.')

    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        padding_mask=padding_mask,
        deterministic=not enable_dropout,
        model_mode=model_mode,
    )
    return logits
