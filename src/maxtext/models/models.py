# Copyright 2023–2026 Google LLC
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

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import Config, DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN, MultimodalInput
from maxtext.layers.nnx_decoders import NNXDecoder
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers.embeddings import Embed
from maxtext.layers.encoders import AudioEncoder, VisionEncoder
from maxtext.layers.multi_token_prediction import MultiTokenPredictionBlock
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.multimodal import processor as mm_processor

# ------------------------------------------------------------------------------
# The network: Transformer Definitions
# ------------------------------------------------------------------------------


def transformer_as_linen(
    config: Config,
    mesh: Mesh,
    quant: Quant,
    model_mode: str = MODEL_MODE_TRAIN,
    *,
    name: str | None = None,
) -> nnx_wrappers.ToLinen:
  """Constructs an NNX Transformer wrapped as a Linen module.

  Returns a `TransformerLinen` that wraps the NNX-style Transformer so it can be
  driven through the Linen init/apply API (checkpoint conversion, AOT compile,
  the inference engine). Pure-NNX call sites build `Transformer` directly via
  `model_creation_utils.from_config`.

  Args:
    config (Config): The configuration object specifying model hyperparameters and options.
    mesh (Mesh): The JAX sharding mesh for device partitioning.
    quant (Quant): The quantization module or configuration to use.
    model_mode (str, optional): The operational mode for the model, e.g.
      training, prefill, or autoregressive. Defaults to `MODEL_MODE_TRAIN`.
    name (str, optional): Optional module name for construction.

  Returns:
    nnx_wrappers.ToLinen: An NNX Transformer wrapped as a Linen module.
  """
  return TransformerLinen(
      Transformer,
      args=(),
      kwargs=nn.FrozenDict(
          {
              "mesh": mesh,
              "config": config,
              "quant": quant,
              "model_mode": model_mode,
          }
      ),
      metadata_fn=initializers.variable_to_logically_partitioned,
      name=name,
  )


class TransformerLinen(nnx_wrappers.ToLinen):
  """Transformer model as a linen module."""

  def init(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    """Initializes the model."""
    model_kwargs = self.kwargs.copy({"model_mode": model_mode})  # type: ignore[wrong-arg-types]
    module = self.clone(kwargs=model_kwargs)
    kwargs["model_mode"] = model_mode
    return nnx_wrappers.ToLinen.init(module, *args, **kwargs)

  def apply(self, *args, model_mode: str = MODEL_MODE_TRAIN, **kwargs):
    """Applies the model."""
    model_kwargs = self.kwargs.copy({"model_mode": model_mode})  # type: ignore[wrong-arg-types]
    module = self.clone(kwargs=model_kwargs)
    kwargs["model_mode"] = model_mode
    return nnx_wrappers.ToLinen.apply(module, *args, **kwargs)


class Transformer(nnx.Module):
  """An autoregressive transformer model."""

  # Make new attributes required, so that all Transformer dependencies (train, decode,
  # compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Quant,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: nnx.Rngs,
  ):
    """Initialize shared_embedding & decoder layers."""
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode

    cfg = self.config
    mesh = self.mesh
    self.token_embedder = Embed(
        mesh=self.mesh,
        num_embeddings=cfg.vocab_size,
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=cfg,
        rngs=rngs,
    )
    self.vision_encoder = VisionEncoder(config=cfg, mesh=mesh, rngs=rngs) if cfg.use_multimodal else None
    self.audio_encoder = AudioEncoder(config=cfg, mesh=mesh, rngs=rngs) if cfg.use_audio else None
    self.decoder = NNXDecoder(config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode, rngs=rngs)

    # If MTP is enabled via config, set up the MTP block.
    if self.config.mtp_num_layers > 0:
      # Get the list of layer blueprints for the current model.
      layer_types = self.decoder.get_decoder_layers()
      # For MTP, we use the DecoderLayer blueprint to ensure architectural consistency.
      # By convention, this is the last layer in the list.
      mtp_layer = layer_types[-1]
      self.mtp_block = MultiTokenPredictionBlock(
          config=self.config,
          mesh=self.mesh,
          transformer_layer_module=mtp_layer,
          decoder=self.decoder,
          rngs=rngs,
      )

  def no_op(self, *args, **kwargs):
    """A no-op method to allow the model to be used in a lazy context."""
    return

  def logits_from_hidden_states_for_vocab_tiling(self, hidden_states, deterministic, model_mode):
    """Computes logits from hidden states; used by vocabulary tiling."""
    return self.decoder.apply_output_head(
        shared_embedding=self.token_embedder,
        y=hidden_states,
        deterministic=deterministic,
        model_mode=model_mode,
    )

  def init_cache(self, cache_size: int, batch_size: int, dtype=jnp.float32):
    """Initializes the KV cache for the Transformer.

    Args:
      cache_size: The maximum size of the KV cache.
      batch_size: The batch size for which the cache is initialized.
      dtype: Data type for the cache. Defaults to `jnp.float32`.

    Returns:
      True if the cache is successfully initialized.
    """
    return True

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids=None,
      cache=None,
      encoder_images: jax.Array | None = None,
      encoder_image_masks: jax.Array | None = None,
      encoder_videos: jax.Array | None = None,
      encoder_video_masks: jax.Array | None = None,
      encoder_audios: jax.Array | None = None,
      enable_dropout=True,
      model_mode=MODEL_MODE_TRAIN,
      previous_chunk=None,
      true_length: int | None = None,
      slot: int | None = None,
      decoder_target_tokens: jax.Array | None = None,
      decoder_target_mask: jax.Array | None = None,
      kv_caches: list[jax.Array] | None = None,
      attention_metadata: dict[str, Any] | None = None,
  ):
    """Applies the Zero-1 FSDP wrapped Transformer model.

    This method handles the all-gather operation for model weights before
    applying the underlying Transformer model, and then releases them.

    Args:
      decoder_input_tokens: Input tokens for the decoder.
      decoder_positions: Positional encodings for the decoder inputs.
      decoder_segment_ids: Segment IDs for the decoder inputs (optional).
      encoder_images: Encoder images for multimodal models (optional).
      enable_dropout: Whether to enable dropout. Defaults to True.
      previous_chunk: Previous chunk for incremental decoding (optional).
      true_length: True length of the prompt before padding (optional).
      slot: An integer representing the decode batch index selected for this request (optional).
      partition_spec: Partition specification for FSDP all-gather.
      decoder_target_tokens: Target tokens for the decoder (optional, used in MTP).
      decoder_target_mask: Target mask for the decoder (optional, used in MTP).
      nnx_method: Method to call on the NNX module (optional).
      kv_caches: List of KV caches for each attention layer, used when invoking from vLLM (optional).
      attention_metadata: Mapping to store attention metadata, used when invoking from vLLM (optional).

    Returns:
      Logits from the Transformer model. Logits, hidden_state, kv_caches if called by vLLM.
    """
    if decoder_segment_ids is not None and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    bidirectional_mask_image = None
    bidirectional_mask_video = None
    image_embeddings = None
    video_embeddings = None
    audio_embeddings = None
    deepstack_visual_embeds = None
    if self.config.use_multimodal and encoder_images is not None:
      image_embeddings, deepstack_visual_embeds = self.vision_encoder(
          input_images=encoder_images, deterministic=not enable_dropout
      )
      bidirectional_mask_image = mm_processor.get_bidirectional_mask_vision(
          self.config, decoder_input_tokens, is_video=False
      )

    if self.config.use_multimodal and encoder_videos is not None:
      video_embeddings, deepstack_visual_embeds = self.vision_encoder(
          input_images=encoder_videos, deterministic=not enable_dropout
      )
      bidirectional_mask_video = mm_processor.get_bidirectional_mask_vision(
          self.config, decoder_input_tokens, is_video=True
      )

    if self.config.use_multimodal and encoder_audios is not None and self.audio_encoder is not None:
      audio_embeddings = self.audio_encoder(input_audio=encoder_audios, deterministic=not enable_dropout)

    # Create audio mask for placeholder tokens (qwen3-omni models)
    audio_masks = None
    if audio_embeddings is not None:
      audio_masks = mm_processor.get_bidirectional_mask_audio(self.config, decoder_input_tokens)

    multimodal_input = None
    if image_embeddings is not None or video_embeddings is not None or audio_embeddings is not None:
      multimodal_input = MultimodalInput(
          image_embeddings=image_embeddings,
          image_masks=encoder_image_masks,
          video_embeddings=video_embeddings,
          video_masks=encoder_video_masks,
          audio_embeddings=audio_embeddings,
          audio_masks=audio_masks,
          bidirectional_mask=bidirectional_mask_image,
          bidirectional_mask_video=bidirectional_mask_video,
      )

    logits, hidden_state, kv_caches = self.decoder(
        shared_embedding=self.token_embedder,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        slot=slot,
        multimodal_input=multimodal_input,
        kv_caches=kv_caches,
        attention_metadata=attention_metadata,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )  # pytype: disable=wrong-keyword-args

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
          model_mode=model_mode,
      )

    if self.config.attention == "vllm_rpa":
      # In vLLM, logits are computed separately after updating the KV cache.
      return hidden_state, kv_caches

    return logits
