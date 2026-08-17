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

"""Stable Diffusion v1.5 Model definition in Flax Linen for MaxText."""

from typing import Any
from flax import linen as nn
import jax
import jax.numpy as jnp

from maxtext.common.common_types import Config
from maxtext.diffusion.clip_text_encoder import FlaxCLIPTextModel
from diffusers.models.unets.unet_2d_condition_flax import FlaxUNet2DConditionModel
from diffusers.models.vae_flax import FlaxAutoencoderKL


class StableDiffusionModel(nn.Module):
  """Unified Flax Linen model for Stable Diffusion v1.5."""

  config: Any = None
  vocab_size: int = 49408
  text_embed_dim: int = 768
  max_position_embeddings: int = 77
  text_num_layers: int = 12
  text_num_heads: int = 12
  text_intermediate_size: int = 3072

  unet_sample_size: int = 64
  unet_in_channels: int = 4
  unet_out_channels: int = 4
  unet_block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
  unet_layers_per_block: int = 2
  unet_attention_head_dim: int = 8
  cross_attention_dim: int = 768

  vae_latent_channels: int = 4
  vae_out_channels: int = 3
  vae_scaling_factor: float = 0.18215
  dtype: Any = jnp.float32

  def setup(self):
    self.text_encoder = FlaxCLIPTextModel(
        vocab_size=self.vocab_size,
        embed_dim=self.text_embed_dim,
        max_position_embeddings=self.max_position_embeddings,
        num_layers=self.text_num_layers,
        num_heads=self.text_num_heads,
        intermediate_size=self.text_intermediate_size,
        dtype=self.dtype,
        name="text_encoder",
    )
    self.unet = FlaxUNet2DConditionModel(
        sample_size=self.unet_sample_size,
        in_channels=self.unet_in_channels,
        out_channels=self.unet_out_channels,
        block_out_channels=self.unet_block_out_channels,
        layers_per_block=self.unet_layers_per_block,
        attention_head_dim=self.unet_attention_head_dim,
        cross_attention_dim=self.cross_attention_dim,
        dtype=self.dtype,
        name="unet",
    )
    self.vae = FlaxAutoencoderKL(
        in_channels=self.vae_out_channels,
        out_channels=self.vae_out_channels,
        latent_channels=self.vae_latent_channels,
        scaling_factor=self.vae_scaling_factor,
        dtype=self.dtype,
        name="vae",
    )

  def __call__(
      self,
      sample: jnp.ndarray,
      timesteps: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      train: bool = False,
  ) -> jnp.ndarray:
    """Runs forward pass of the UNet on noisy latents."""
    return self.unet(sample, timesteps, encoder_hidden_states, train=train).sample

  def encode_text(self, input_ids: jnp.ndarray) -> jnp.ndarray:
    """Encodes text input IDs into prompt embeddings."""
    return self.text_encoder(input_ids)

  def decode_latents(self, latents: jnp.ndarray) -> jnp.ndarray:
    """Decodes latent representation into RGB image."""
    scaled_latents = (1.0 / self.vae_scaling_factor) * latents
    return self.vae(scaled_latents, method=self.vae.decode).sample

  def encode_image(self, image: jnp.ndarray) -> jnp.ndarray:
    """Encodes RGB image into latent representation."""
    return self.vae(image, method=self.vae.encode).latent_dist.sample()
