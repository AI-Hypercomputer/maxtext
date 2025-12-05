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
from flax import nnx
from jax.sharding import Mesh

from MaxText.common_types import Config
from MaxText.layers import nnx_wrappers
from MaxText.layers import initializers


class VisionEncoder(nnx.Module):
  """Vision encoder to encode images into soft tokens."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.encoder_name, self.projector_name = self._setup_vision_encoder_layers()

  def _setup_vision_encoder_layers(self):
    """Setup vision encoder layers specific to the model, instantiate NNX modules."""
    if self.config.model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
      from MaxText.layers import gemma3  # pylint: disable=import-outside-toplevel

      encoder_name = "Gemma3VisionEncoderLayer_0"
      projector_name = "VisionEmbedder_0"
      setattr(self, encoder_name, gemma3.Gemma3VisionEncoderLayer(config=self.config, mesh=self.mesh, rngs=self.rngs))
      setattr(self, projector_name, gemma3.VisionEmbedder(config=self.config, mesh=self.mesh, rngs=self.rngs))
      return encoder_name, projector_name
    elif self.config.model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
      from MaxText.layers import llama4  # pylint: disable=import-outside-toplevel

      encoder_name = "Llama4VisionModel_0"
      projector_name = "Llama4MultiModalProjector_0"
      setattr(self, encoder_name, llama4.Llama4VisionModel(config=self.config, mesh=self.mesh, rngs=self.rngs))
      setattr(self, projector_name, llama4.Llama4MultiModalProjector(config=self.config, mesh=self.mesh, rngs=self.rngs))
      return encoder_name, projector_name
    elif self.config.model_name in ["qwen3-omni-30b-a3b"]:
      from MaxText.layers import qwen3  # pylint: disable=import-outside-toplevel

      encoder_name = "Qwen3OmniMoeVisionEncoder_0"
      projector_name = "Qwen3OmniMoeVisionProjector_0"
      setattr(self, encoder_name, qwen3.Qwen3OmniMoeVisionEncoder(config=self.config, mesh=self.mesh, rngs=self.rngs))
      setattr(self, projector_name, qwen3.Qwen3OmniMoeVisionProjector(config=self.config, rngs=self.rngs))
      return encoder_name, projector_name
    else:
      raise ValueError(f"No VisionEncoder implemented for {self.config.model_name} yet")

  def __call__(self, input_images, deterministic=False):
    # vision encoder output, frozen params in many cases
    encoder = getattr(self, self.encoder_name)
    embeddings = encoder(input_images, deterministic=deterministic)

    if self.config.freeze_vision_encoder_params:
      embeddings = jax.lax.stop_gradient(embeddings)

    # vision embedder / projection layer, not frozen in most cases, trained / finetuned together with main model
    projector = getattr(self, self.projector_name)
    embeddings = projector(embeddings)

    return embeddings


def vision_encoder_as_linen(
    config: Config,
    mesh: Mesh,
):
  """Creates a VisionEncoder module."""
  module = nnx_wrappers.to_linen(
      VisionEncoder,
      config=config,
      mesh=mesh,
      name="vision_encoder",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
  return module
