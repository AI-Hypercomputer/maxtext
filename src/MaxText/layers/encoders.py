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

""""Module for encoder layers."""

import jax
import jax.numpy as jnp  # DEBUG (remove later) - needed for debug prints
from flax import linen as nn
from jax.sharding import Mesh

from MaxText.common_types import Config
from MaxText.layers import quantizations

# Type alias for cleaner type hints
Quant = quantizations.AqtQuantization


class VisionEncoder(nn.Module):
  """Vision encoder to encode images into soft tokens."""

  config: Config
  mesh: Mesh

  def setup(self):
    self.vision_encoder_layer = self.get_vision_encoder_layers()

  def get_vision_encoder_layers(self):
    """Get vision encoder layers specific to the model, classes of nn.Module type."""
    if self.config.model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
      from MaxText.layers import gemma3  # pylint: disable=import-outside-toplevel

      return [gemma3.gemma3visionencoder_as_linen, gemma3.visionembedder_as_linen]
    elif self.config.model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
      from MaxText.layers import llama4  # pylint: disable=import-outside-toplevel

      return [llama4.llama4visionmodel_as_linen, llama4.llama4multimodalprojector_as_linen]
    elif self.config.model_name in ["qwen3-omni-30b-a3b"]:
      from MaxText.layers import qwen3  # pylint: disable=import-outside-toplevel

      return [qwen3.qwen3omni_visionencoder_as_linen, qwen3.qwen3omni_visionprojector_as_linen]
    else:
      raise ValueError(f"No VisionEncoder implemented for {self.config.model_name} yet")

  @nn.compact
  def __call__(self, input_images, deterministic=False):
    cfg = self.config
    mesh = self.mesh
    # vision encoder output, frozen params in many cases
    embeddings = self.vision_encoder_layer[0](config=cfg, mesh=mesh)(input_images, deterministic=deterministic)
    deep_feats = None
    if cfg.deepstack_visual_indexes_for_vit:
      deep_feats = embeddings[1]
      embeddings = embeddings[0]
 
    if cfg.freeze_vision_encoder_params:
      embeddings = jax.lax.stop_gradient(embeddings)
      if deep_feats is not None:
        deep_feats = [jax.lax.stop_gradient(feat) for feat in deep_feats]

    if len(self.vision_encoder_layer) > 1:
      # vision embedder / projection layer, not frozen in most cases, trained / finetuned together with main model
      embeddings = self.vision_encoder_layer[1](config=cfg, mesh=mesh)(embeddings)

    if cfg.deepstack_visual_indexes_for_vit:
      return embeddings, deep_feats
    else:
      return embeddings


class AudioEncoder(nn.Module):
  """Audio encoder to encode audio features into soft tokens."""

  config: Config
  mesh: Mesh

  def setup(self):
    self.audio_encoder_layer = self.get_audio_encoder_layers()

  def get_audio_encoder_layers(self):
    """Get audio encoder layers specific to the model, classes of nn.Module type."""
    if self.config.deepstack_visual_indexes_for_vit:
      from MaxText.layers import qwen3  # pylint: disable=import-outside-toplevel

      return [qwen3.qwen3omni_audioencoder_as_linen, qwen3.qwen3omni_audioprojector_as_linen]
    else:
      raise ValueError(f"No AudioEncoder implemented for {self.config.model_name} yet")

  @nn.compact
  def __call__(self, input_audio, deterministic=False):
    cfg = self.config
    mesh = self.mesh
    # audio encoder output (includes convs + encoder, outputs before projector)
    embeddings = self.audio_encoder_layer[0](config=cfg, mesh=mesh)(input_audio, deterministic=deterministic)

    if cfg.freeze_audio_encoder_params:
      embeddings = jax.lax.stop_gradient(embeddings)

    if len(self.audio_encoder_layer) > 1:
      # audio projector layer
      embeddings = self.audio_encoder_layer[1](config=cfg, mesh=mesh)(embeddings)

    return embeddings
