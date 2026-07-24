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

"""Module for encoder layers."""

import jax
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config, VisionEncoderBlockType


class VisionEncoder(nnx.Module):
  """Vision encoder to encode images into soft tokens."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.encoder_name, self.projector_name = self._setup_vision_encoder_layers()

  def _setup_vision_encoder_layers(self):
    """Setup vision encoder layers specific to the model, instantiate NNX modules."""
    self.vision_encoder_block = self.config.vision_encoder_block

    if self.vision_encoder_block == VisionEncoderBlockType.GEMMA3:
      from maxtext.models import gemma3  # pylint: disable=import-outside-toplevel

      encoder_name = "Gemma3VisionEncoderLayer_0"
      projector_name = "VisionEmbedder_0"
      setattr(self, encoder_name, gemma3.Gemma3VisionEncoderLayer(config=self.config, mesh=self.mesh, rngs=self.rngs))
      setattr(self, projector_name, gemma3.VisionEmbedder(config=self.config, mesh=self.mesh, rngs=self.rngs))
      return encoder_name, projector_name
    elif self.vision_encoder_block == VisionEncoderBlockType.LLAMA4:
      from maxtext.models import llama4  # pylint: disable=import-outside-toplevel

      encoder_name = "Llama4VisionModel_0"
      projector_name = "Llama4MultiModalProjector_0"
      setattr(self, encoder_name, llama4.Llama4VisionModel(config=self.config, mesh=self.mesh, rngs=self.rngs))
      setattr(self, projector_name, llama4.Llama4MultiModalProjector(config=self.config, mesh=self.mesh, rngs=self.rngs))
      return encoder_name, projector_name
    elif self.vision_encoder_block == VisionEncoderBlockType.QWEN3_OMNI:
      from maxtext.models import qwen3  # pylint: disable=import-outside-toplevel

      encoder_name = "Qwen3OmniMoeVisionEncoder_0"
      projector_name = "Qwen3OmniMoeVisionProjector_0"
      setattr(self, encoder_name, qwen3.Qwen3OmniMoeVisionEncoder(config=self.config, mesh=self.mesh, rngs=self.rngs))
      setattr(self, projector_name, qwen3.Qwen3OmniMoeVisionProjector(config=self.config, rngs=self.rngs))
      return encoder_name, projector_name
    elif self.vision_encoder_block == VisionEncoderBlockType.GEMMA4:
      from maxtext.models import gemma4_vision  # pylint: disable=import-outside-toplevel

      encoder_name = "Gemma4VisionEncoderLayer_0"
      projector_name = "Gemma4VisionProjector_0"
      setattr(
          self, encoder_name, gemma4_vision.Gemma4VisionEncoderLayer(config=self.config, mesh=self.mesh, rngs=self.rngs)
      )
      setattr(
          self, projector_name, gemma4_vision.Gemma4VisionProjector(config=self.config, mesh=self.mesh, rngs=self.rngs)
      )
      return encoder_name, projector_name
    elif self.vision_encoder_block == VisionEncoderBlockType.QWEN3_5:
      from maxtext.models import qwen3_5_vision  # pylint: disable=import-outside-toplevel

      encoder_name = "Qwen3_5MoeVisionEncoder_0"
      projector_name = "Qwen3_5MoeVisionProjector_0"
      setattr(
          self, encoder_name, qwen3_5_vision.Qwen3_5MoeVisionEncoder(config=self.config, mesh=self.mesh, rngs=self.rngs)
      )
      setattr(self, projector_name, qwen3_5_vision.Qwen3_5MoeVisionProjector(config=self.config, rngs=self.rngs))
      return encoder_name, projector_name
    elif self.vision_encoder_block == VisionEncoderBlockType.QWEN3_VL:
      from maxtext.models import qwen3_vl_vision  # pylint: disable=import-outside-toplevel

      encoder_name = "Qwen3VLVisionEncoder_0"
      projector_name = "Qwen3VLVisionProjector_0"
      setattr(
          self, encoder_name, qwen3_vl_vision.Qwen3VLVisionEncoder(config=self.config, mesh=self.mesh, rngs=self.rngs)
      )
      setattr(self, projector_name, qwen3_vl_vision.Qwen3VLVisionProjector(config=self.config, rngs=self.rngs))
      return encoder_name, projector_name
    else:
      supported_blocks = [block.value for block in VisionEncoderBlockType if block != VisionEncoderBlockType.NONE]
      raise ValueError(
          f"Unsupported vision_encoder_block={self.vision_encoder_block.value!r} "
          f"for model_name={self.config.model_name!r}. Supported values are: {supported_blocks}."
      )

  def __call__(self, input_images, input_masks=None, video_grid_thw=None, deterministic=False):
    # vision encoder output, frozen params in many cases
    encoder = getattr(self, self.encoder_name)
    if self.vision_encoder_block.value.startswith("qwen3") and input_masks is not None:
      encoder_output = encoder(
          input_images, video_mask=input_masks, video_grid_thw=video_grid_thw, deterministic=deterministic
      )
    else:
      encoder_output = encoder(input_images, deterministic=deterministic)
    deep_feats = None
    if isinstance(encoder_output, tuple):
      embeddings = encoder_output[0]
      deep_feats = encoder_output[1]
    else:
      embeddings = encoder_output

    if self.config.freeze_vision_encoder_params:
      embeddings = jax.lax.stop_gradient(embeddings)
      if deep_feats is not None:
        deep_feats = [jax.lax.stop_gradient(feat) for feat in deep_feats]

    # vision embedder / projection layer, not frozen in most cases, trained / finetuned together with main model
    projector = getattr(self, self.projector_name)
    embeddings = projector(embeddings)

    return embeddings, deep_feats


class AudioEncoder(nnx.Module):
  """Audio encoder to encode audio features into soft tokens."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.encoder_name, self.projector_name = self._setup_audio_encoder_layers()

  def _setup_audio_encoder_layers(self):
    """Setup audio encoder layers specific to the model, instantiate NNX modules."""
    if self.config.model_name in ["qwen3-omni-30b-a3b"]:
      from maxtext.models import qwen3  # pylint: disable=import-outside-toplevel

      encoder_name = "Qwen3OmniAudioEncoder_0"
      projector_name = "Qwen3OmniAudioProjector_0"
      setattr(self, encoder_name, qwen3.Qwen3OmniAudioEncoder(config=self.config, mesh=self.mesh, rngs=self.rngs))
      setattr(self, projector_name, qwen3.Qwen3OmniAudioProjector(config=self.config, rngs=self.rngs))
      return encoder_name, projector_name
    else:
      raise ValueError(f"No AudioEncoder implemented for {self.config.model_name} yet")

  def __call__(self, input_audio, deterministic=False):
    # audio encoder output (includes convs + encoder, outputs before projector)
    encoder = getattr(self, self.encoder_name)
    embeddings = encoder(input_audio, deterministic=deterministic)

    if self.config.freeze_audio_encoder_params:
      embeddings = jax.lax.stop_gradient(embeddings)

    # audio projector layer
    projector = getattr(self, self.projector_name)
    embeddings = projector(embeddings)

    return embeddings
