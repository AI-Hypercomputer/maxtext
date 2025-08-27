# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer models."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx
from MaxText.layers import initializers

from MaxText.common_types import MODEL_MODE_PREFILL, DecoderBlockType, Config, MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE, DECODING_ACTIVE_SEQUENCE_INDICATOR
from MaxText.inference import page_manager
from MaxText import multimodal_utils
from MaxText.layers import nnx_wrappers
from MaxText.layers.decoders import Decoder
from MaxText.layers.embeddings import Embed, embed_as_linen
from MaxText.layers.encoders import VisionEncoder
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.multi_token_prediction import MultiTokenPredictionBlock
from MaxText.maxtext_utils import all_gather_over_fsdp

# ------------------------------------------------------------------------------
# The network: Transformer Definitions
# ------------------------------------------------------------------------------

class TransformerLinenPure(nn.Module):
  """An autoregressive transformer model."""

  # Make new attributes required, so that all Transformer dependencies (train, decode,
  # compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh
  quant: Quant
  # Possible model_mode values can be found in MaxText.common_types.
  # We generally use MaxText.common_types.MODEL_MODE_TRAIN or
  # MaxText.common_types.MODEL_MODE_PREFILL for initializations here.
  # TODO: Make model_mode required after confirming no users are affected.
  model_mode: str = MODEL_MODE_TRAIN  # May be different than the model_mode passed to __call__
  # pylint: enable=attribute-defined-outside-init

  def init(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    """Initializes the model."""
    module = self.clone(model_mode=model_mode)
    return nn.Module.init(module, *args, **kwargs)

  def apply(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    """Applies the model."""
    module = self.clone(model_mode=model_mode)
    return nn.Module.apply(module, *args, **kwargs)

  def setup(self):
    """Initialize shared_embedding & decoder layers."""

    cfg = self.config
    mesh = self.mesh
    self.shared_embedding = embed_as_linen(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name="token_embedder",
        config=cfg,
    )
    self.vision_encoder = VisionEncoder(config=cfg, mesh=mesh) if cfg.use_multimodal else None
    self.decoder = Decoder(
        config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode
    )
    # If MTP is enabled via config, set up the MTP block.
    if self.config.mtp_num_layers > 0:
      # Get the list of layer blueprints for the current model.
      layer_types = self.decoder.get_decoder_layers()
      # For MTP, we use the DecoderLayer blueprint to ensure architectural consistency.
      # By convention, this is the last layer in the list.
      mtp_layer = layer_types[-1]
      self.mtp_block = MultiTokenPredictionBlock(
          config=self.config, mesh=self.mesh, name="mtp_block", transformer_layer_module=mtp_layer, decoder=self.decoder
      )

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids=None,
      encoder_images: None | jnp.ndarray = None,
      enable_dropout=True,
      previous_chunk=None,
      true_length: None | int = None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      decoder_target_tokens: None | jnp.ndarray = None,
      decoder_target_mask: None | jnp.ndarray = None,
      nnx_method=None,
  ):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      true_length: (Optional) Prompt length before padding
      slot: (Optional) An integer representing the decode batch index selected
        for this request.
    """

    if decoder_segment_ids is not None and self.model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    bidirectional_mask = None
    image_embeddings = None
    if self.config.use_multimodal and encoder_images is not None:
      image_embeddings = self.vision_encoder(input_images=encoder_images, deterministic=not enable_dropout)

      if self.config.decoder_block == DecoderBlockType.GEMMA3:
        bidirectional_mask = decoder_input_tokens == multimodal_utils.GEMMA_TOKEN_PLACEHOLDER
      elif self.config.decoder_block == DecoderBlockType.LLAMA4:
        bidirectional_mask = decoder_input_tokens == multimodal_utils.LLAMA4_PATCH_TOKEN

    logits, hidden_state = self.decoder(
        shared_embedding=self.shared_embedding,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        previous_chunk=previous_chunk,
        slot=slot,
        page_state=page_state,
        bidirectional_mask=bidirectional_mask,
        image_embeddings=image_embeddings,
    )

    # If we are initializing the model AND MTP is enabled, we must create
    # dummy target tensors. This allows Flax to trace the MTPBlock and create
    # all its necessary parameters, without requiring the main training pipeline
    # to be aware of this initialization detail.
    if self.is_initializing() and self.config.mtp_num_layers > 0:
      if decoder_target_tokens is None:
        dummy_shape = decoder_input_tokens.shape
        decoder_target_tokens = jnp.ones(dummy_shape, dtype=jnp.int32)
        decoder_target_mask = jnp.ones(dummy_shape, dtype=jnp.int32)
        decoder_segment_ids = jnp.ones(dummy_shape, dtype=jnp.int32)

    # The Multi-Token Prediction (MTP) block functions as a "side-car" to the main
    # model, active only during training. It computes an auxiliary loss based on
    # predicting multiple future tokens, as described in the DeepSeek-V3 paper.
    # To ensure architectural consistency, it uses two key components from the parent Transformer:
    #   1. The same `DecoderLayer` blueprint for its internal transformer blocks.
    #   2. The `shared_embedding` for both embedding future tokens and for its final
    #      logit projection.
    # Its only effect is to "sow" these losses; it does not alter the primary logits output.
    if self.config.mtp_num_layers > 0:
      self.mtp_block(
          shared_embedding=self.shared_embedding,
          main_hidden_state=hidden_state,
          input_ids=decoder_input_tokens,
          target_ids=decoder_target_tokens,
          target_mask=decoder_target_mask,
          position_ids=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=not enable_dropout,
      )

    return logits


def transformer_as_linen(
    config: Config,
    mesh: Mesh,
    quant: Quant,
    model_mode: str = MODEL_MODE_TRAIN,
    *,
    name: str | None = None,
) -> nnx_wrappers.ToLinen | TransformerLinenPure:
  if config.enable_nnx:
    return TransformerLinen(
        Transformer,
        args=(),
        kwargs=nn.FrozenDict({
          "mesh": mesh,
        "config": config,
        "quant": quant,
        "model_mode": model_mode,
      }),
      metadata_fn=initializers.variable_to_logically_partitioned,
      name=name,
  )
  else:
    return TransformerLinenPure(
      config, mesh, quant, model_mode=model_mode, name=name
    )

class TransformerLinen(nnx_wrappers.ToLinen):
  """Transformer model as a linen module."""

  def init(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    """Initializes the model."""
    model_kwargs = self.kwargs.copy({"model_mode": model_mode})  # type: ignore[wrong-arg-types]
    module = self.clone(kwargs=model_kwargs)
    return nnx_wrappers.ToLinen.init(module, *args, **kwargs)

  def apply(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    """Applies the model."""
    model_kwargs = self.kwargs.copy({"model_mode": model_mode})  # type: ignore[wrong-arg-types]
    module = self.clone(kwargs=model_kwargs)
    return nnx_wrappers.ToLinen.apply(module, *args, **kwargs)

class Transformer(nnx.Module):
  """An autoregressive transformer model."""

  # Make new attributes required, so that all Transformer dependencies (train, decode,
  # compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  def __init__(self, config: Config, mesh: Mesh, quant: Quant, *, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs):
    """Initialize shared_embedding & decoder layers."""
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode

    cfg = self.config
    mesh = self.mesh
    self.token_embedder = Embed(
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        rngs=rngs
    )
    self.vision_encoder = VisionEncoder(config=cfg, mesh=mesh) if cfg.use_multimodal else None

    decoder_linen = Decoder(config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode)
    self.decoder = nnx_wrappers.ToNNX(decoder_linen, rngs=rngs)

    if self.model_mode == MODEL_MODE_PREFILL:
      seq_len = cfg.max_prefill_predict_length
    elif self.model_mode == MODEL_MODE_AUTOREGRESSIVE:
      seq_len = 1
    else:
      seq_len = cfg.max_target_length

    batch_size = cfg.micro_batch_size_to_train_on
    dummy_decoder_input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    dummy_decoder_positions = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    self.decoder.lazy_init(
      shared_embedding=self.token_embedder,
      decoder_input_tokens=dummy_decoder_input_tokens,
      decoder_positions=dummy_decoder_positions,
    )

    # If MTP is enabled via config, set up the MTP block.
    if self.config.mtp_num_layers > 0:
      # Get the list of layer blueprints for the current model.
      layer_types = self.decoder.get_decoder_layers()
      # For MTP, we use the DecoderLayer blueprint to ensure architectural consistency.
      # By convention, this is the last layer in the list.
      mtp_layer = layer_types[-1]
      mtp_block_linen = MultiTokenPredictionBlock(
          config=self.config, mesh=self.mesh, name="mtp_block", transformer_layer_module=mtp_layer, decoder=self.decoder
      )
      self.mtp_block = nnx_wrappers.ToNNX(mtp_block_linen, rngs=rngs)

      self.mtp_block.lazy_init(
        shared_embedding=self.token_embedder,
        main_hidden_state=jnp.ones((1, 1, self.config.emb_dim), dtype=self.config.dtype),
        input_ids=jnp.ones((1, 1), dtype=jnp.int32),
        target_ids=jnp.ones((1, 1), dtype=jnp.int32),
        target_mask=jnp.ones((1, 1), dtype=jnp.int32),
        position_ids=jnp.ones((1, 1), dtype=jnp.int32),
        decoder_segment_ids=jnp.ones((1, 1), dtype=jnp.int32),
        deterministic=True,
      )

  def no_op(self, *args, **kwargs):
    """A no-op method to allow the model to be used in a lazy context."""
    return

  def init_cache(self, cache_size: int, batch_size: int, dtype=jnp.float32):
    return True

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids=None,
      cache=None,
      encoder_images: jax.Array | None = None,
      enable_dropout=True,
      previous_chunk=None,
      true_length: int | None = None,
      slot: int | None = None,
      page_state: page_manager.PageState | None = None,
      decoder_target_tokens: jax.Array | None = None,
      decoder_target_mask: jax.Array | None = None,
  ):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      true_length: (Optional) Prompt length before padding
      slot: (Optional) An integer representing the decode batch index selected
        for this request.
    """

    if decoder_segment_ids is not None and self.model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    bidirectional_mask = None
    image_embeddings = None
    if self.config.use_multimodal and encoder_images is not None:
      image_embeddings = self.vision_encoder(input_images=encoder_images, deterministic=not enable_dropout)

      if self.config.decoder_block == DecoderBlockType.GEMMA3:
        bidirectional_mask = decoder_input_tokens == multimodal_utils.GEMMA_TOKEN_PLACEHOLDER
      elif self.config.decoder_block == DecoderBlockType.LLAMA4:
        bidirectional_mask = decoder_input_tokens == multimodal_utils.LLAMA4_PATCH_TOKEN

    logits, hidden_state = self.decoder(
        shared_embedding=self.token_embedder,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        previous_chunk=previous_chunk,
        slot=slot,
        page_state=page_state,
        bidirectional_mask=bidirectional_mask,
        image_embeddings=image_embeddings,
    )

    # If we are initializing the model AND MTP is enabled, we must create
    # dummy target tensors. This allows Flax to trace the MTPBlock and create
    # all its necessary parameters, without requiring the main training pipeline
    # to be aware of this initialization detail.
    # if self.is_initializing() and self.config.mtp_num_layers > 0:
    #   if decoder_target_tokens is None:
    #     dummy_shape = decoder_input_tokens.shape
    #     decoder_target_tokens = jnp.ones(dummy_shape, dtype=jnp.int32)
    #     decoder_target_mask = jnp.ones(dummy_shape, dtype=jnp.int32)
    #     decoder_segment_ids = jnp.ones(dummy_shape, dtype=jnp.int32)

    # The Multi-Token Prediction (MTP) block functions as a "side-car" to the main
    # model, active only during training. It computes an auxiliary loss based on
    # predicting multiple future tokens, as described in the DeepSeek-V3 paper.
    # To ensure architectural consistency, it uses two key components from the parent Transformer:
    #   1. The same `DecoderLayer` blueprint for its internal transformer blocks.
    #   2. The `shared_embedding` for both embedding future tokens and for its final
    #      logit projection.
    # Its only effect is to "sow" these losses; it does not alter the primary logits output.
    if self.config.mtp_num_layers > 0:
      self.mtp_block(
          shared_embedding=self.token_embedder,
          main_hidden_state=hidden_state,
          input_ids=decoder_input_tokens,
          target_ids=decoder_target_tokens,
          target_mask=decoder_target_mask,
          position_ids=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=not enable_dropout,
      )

    return logits

class ZeroOneTransformer(nn.Module):
  """
  A wrapper for the base Transformer model designed to implement the Zero-1
  FSDP optimization.

  The goal of this optimization is to reduce communication overhead. In the standard
  FSDP implementation, an all-gather operation on the model weights is performed twice
  for each gradient accumulation microbatch (once for the forward pass, once for the backward pass).
  This class changes that behavior. When enabled, it performs the all-gather operation
  only *once* per full gradient accumulation step. It gathers the full weights into
  memory, runs all the microbatch forward and backward passes, and then releases the
  full weights. This trades higher peak memory usage for significantly reduced
  network communication, which can improve training speed if sufficient memory is
  available.
  """

  config: Config
  mesh: Mesh
  quant: Quant
  # Possible model_mode values can be found in MaxText.common_types.
  # We generally use MaxText.common_types.MODEL_MODE_TRAIN or
  # MaxText.common_types.MODEL_MODE_PREFILL for initializations here.
  # TODO: Make model_mode required after confirming no users are affected.
  model_mode: str = MODEL_MODE_TRAIN  # May be different than the model_mode passed to __call__

  def setup(self):
    self.model = transformer_as_linen(self.config, self.mesh, self.quant, self.model_mode)

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids=None,
      encoder_images: None | jnp.ndarray = None,
      enable_dropout=True,
      previous_chunk=None,
      true_length: None | int = None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      partition_spec=None,
      decoder_target_tokens: None | jnp.ndarray = None,
      decoder_target_mask: None | jnp.ndarray = None,
      nnx_method: str | None = None,
  ):
    if self.is_initializing():
      return self.model(
          decoder_input_tokens=decoder_input_tokens,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          encoder_images=encoder_images,
          enable_dropout=enable_dropout,
          previous_chunk=previous_chunk,
          true_length=true_length,
          slot=slot,
          page_state=page_state,
      )
    all_model_weights = all_gather_over_fsdp(
        self.model.variables, partition_spec, mesh=self.mesh, logical_axis_rules=self.config.logical_axis_rules
    )

    return self.model.apply(
        all_model_weights,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        encoder_images=encoder_images,
        enable_dropout=enable_dropout,
        model_mode=self.model_mode,
        previous_chunk=previous_chunk,
        true_length=true_length,
        slot=slot,
        page_state=page_state,
        mutable=False,
        decoder_target_tokens=decoder_target_tokens,
        decoder_target_mask=decoder_target_mask,
        nnx_method=nnx_method,
    )
