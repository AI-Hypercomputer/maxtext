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

from typing import Optional

import jax.numpy as jnp
from jax.sharding import Mesh

from flax import linen as nn

from MaxText.common_types import DecoderBlockType, Config, MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE, DECODING_ACTIVE_SEQUENCE_INDICATOR
from MaxText.inference import page_manager
from MaxText import multimodal_utils
from MaxText.layers.blocks import Decoder, VisionEncoder
from MaxText.layers.embeddings import Embed
from MaxText.layers.quantizations import AqtQuantization as Quant

# ------------------------------------------------------------------------------
# The network: Transformer Definitions
# ------------------------------------------------------------------------------


class Transformer(nn.Module):
  """An decoder-only Transformer model."""

  # Make new attributes required, so that all Transformer dependencies (train, decode, compile, etc) will error instead
  #   of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh
  quant: Quant

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
        name="token_embedder",
        config=cfg,
    )

    self.vision_encoder = VisionEncoder(config=cfg, mesh=mesh) if cfg.use_multimodal else None
    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding, mesh=mesh, quant=self.quant)

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids=None,
      encoder_images: Optional[jnp.ndarray] = None,
      enable_dropout=True,
      model_mode=MODEL_MODE_TRAIN,
      previous_chunk=None,
      true_length: Optional[int] = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      true_length: (Optional) Prompt length before padding
      slot: (Optional) An integer representing the decode batch index selected
        for this request.
    """

    if decoder_segment_ids is not None and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    bidirectional_mask = None
    image_embeddings = None
    # TODO(hengtaoguo): Here we temporarily skip multimodal support for Llama4 models because of WIP
    if self.config.use_multimodal and encoder_images is not None and not self.config.model_name.startswith("llama4"):
      image_embeddings = self.vision_encoder(input_images=encoder_images, deterministic=not enable_dropout)

      if self.config.decoder_block == DecoderBlockType.GEMMA3:
        bidirectional_mask = decoder_input_tokens == multimodal_utils.GEMMA_TOKEN_PLACEHOLDER

    logits, _ = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        slot=slot,
        page_state=page_state,
        bidirectional_mask=bidirectional_mask,
        image_embeddings=image_embeddings,
    )
    return logits
