# Copyright 2026 Google LLC
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

"""Unit tests for Stable Diffusion layers comparing MaxText implementation against PyTorch reference."""

import os
import unittest
import pytest

try:
  import torch
  from transformers import CLIPTextModel as PtCLIPTextModel, CLIPTokenizer
  from diffusers import UNet2DConditionModel, FlaxUNet2DConditionModel, AutoencoderKL, FlaxAutoencoderKL
  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or diffusers not available"),
]

import jax
import jax.numpy as jnp
import numpy as np
from maxtext.diffusion.clip_text_encoder import FlaxCLIPTextModel


class StableDiffusionLayersTest(unittest.TestCase):
  """Unit tests verifying numerical equivalence of MaxText diffusion layers with PyTorch."""

  def setUp(self):
    super().setUp()
    np.random.seed(42)
    torch.manual_seed(42)
    self.cache_dir = os.environ.get("HF_HOME", "/dev/shm/hengtaoguo")
    self.model_id = "runwayml/stable-diffusion-v1-5"

  def test_clip_text_encoder_numerical_match(self):
    """Verifies FlaxCLIPTextModel produces exact logits matching PyTorch CLIPTextModel."""
    tokenizer = CLIPTokenizer.from_pretrained(
        self.model_id, subfolder="text_encoder", cache_dir=self.cache_dir
    )
    pt_model = PtCLIPTextModel.from_pretrained(
        self.model_id, subfolder="text_encoder", cache_dir=self.cache_dir
    )
    pt_model.eval()

    prompt = "a photo of an astronaut riding a horse on mars"
    inputs = tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")

    with torch.no_grad():
      pt_output = pt_model(inputs.input_ids)[0].numpy()

    flax_model = FlaxCLIPTextModel()
    input_ids = jnp.array(inputs.input_ids.numpy())
    vars_init = flax_model.init(jax.random.PRNGKey(0), input_ids)
    params = vars_init["params"]

    # Weight copy
    pt_sd = pt_model.state_dict()
    params["embeddings"]["token_embedding"]["embedding"] = jnp.array(
        pt_sd["embeddings.token_embedding.weight"].numpy()
    )
    params["embeddings"]["position_embedding"]["embedding"] = jnp.array(
        pt_sd["embeddings.position_embedding.weight"].numpy()
    )
    params["final_layer_norm"]["scale"] = jnp.array(
        pt_sd["final_layer_norm.weight"].numpy()
    )
    params["final_layer_norm"]["bias"] = jnp.array(
        pt_sd["final_layer_norm.bias"].numpy()
    )

    for i in range(12):
      l_name = f"layers_{i}"
      params["encoder"][l_name]["layer_norm1"]["scale"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm1.weight"].numpy()
      )
      params["encoder"][l_name]["layer_norm1"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm1.bias"].numpy()
      )
      params["encoder"][l_name]["self_attn"]["q_proj"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.q_proj.weight"].numpy().T
      )
      params["encoder"][l_name]["self_attn"]["q_proj"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.q_proj.bias"].numpy()
      )
      params["encoder"][l_name]["self_attn"]["k_proj"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.k_proj.weight"].numpy().T
      )
      params["encoder"][l_name]["self_attn"]["k_proj"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.k_proj.bias"].numpy()
      )
      params["encoder"][l_name]["self_attn"]["v_proj"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.v_proj.weight"].numpy().T
      )
      params["encoder"][l_name]["self_attn"]["v_proj"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.v_proj.bias"].numpy()
      )
      params["encoder"][l_name]["self_attn"]["out_proj"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.out_proj.weight"].numpy().T
      )
      params["encoder"][l_name]["self_attn"]["out_proj"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.self_attn.out_proj.bias"].numpy()
      )
      params["encoder"][l_name]["layer_norm2"]["scale"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm2.weight"].numpy()
      )
      params["encoder"][l_name]["layer_norm2"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm2.bias"].numpy()
      )
      params["encoder"][l_name]["mlp"]["fc1"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc1.weight"].numpy().T
      )
      params["encoder"][l_name]["mlp"]["fc1"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc1.bias"].numpy()
      )
      params["encoder"][l_name]["mlp"]["fc2"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc2.weight"].numpy().T
      )
      params["encoder"][l_name]["mlp"]["fc2"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc2.bias"].numpy()
      )

    flax_output = flax_model.apply({"params": params}, input_ids)
    np.testing.assert_allclose(
        pt_output,
        np.array(flax_output),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Flax CLIP Text Model output deviates from PyTorch reference!",
    )

  def test_vae_decode_numerical_match(self):
    """Verifies Flax VAE decoding produces exact outputs matching PyTorch AutoencoderKL."""
    pt_vae = AutoencoderKL.from_pretrained(
        self.model_id, subfolder="vae", cache_dir=self.cache_dir
    )
    pt_vae.eval()

    flax_vae, flax_vae_params = FlaxAutoencoderKL.from_pretrained(
        self.model_id, subfolder="vae", cache_dir=self.cache_dir, from_pt=True
    )

    latents_np = np.random.randn(1, 4, 64, 64).astype(np.float32)

    with torch.no_grad():
      pt_latents = torch.tensor(latents_np)
      pt_decoded = pt_vae.decode(pt_latents).sample.numpy()

    flax_latents = jnp.array(latents_np)
    flax_decoded = flax_vae.apply(
        {"params": flax_vae_params}, flax_latents, method=flax_vae.decode
    ).sample

    np.testing.assert_allclose(
        pt_decoded,
        np.array(flax_decoded),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Flax VAE decode output deviates from PyTorch reference!",
    )

  def test_unet_forward_numerical_match(self):
    """Verifies Flax UNet forward pass matches PyTorch UNet2DConditionModel."""
    pt_unet = UNet2DConditionModel.from_pretrained(
        self.model_id, subfolder="unet", cache_dir=self.cache_dir
    )
    pt_unet.eval()

    flax_unet, flax_unet_params = FlaxUNet2DConditionModel.from_pretrained(
        self.model_id, subfolder="unet", cache_dir=self.cache_dir, from_pt=True
    )

    sample_np = np.random.randn(1, 4, 64, 64).astype(np.float32)
    timestep_np = np.array([50], dtype=np.int32)
    encoder_hidden_states_np = np.random.randn(1, 77, 768).astype(np.float32)

    with torch.no_grad():
      pt_sample = torch.tensor(sample_np)
      pt_timestep = torch.tensor([50])
      pt_context = torch.tensor(encoder_hidden_states_np)
      pt_output = pt_unet(pt_sample, pt_timestep, encoder_hidden_states=pt_context).sample.numpy()

    sample_flax = jnp.array(sample_np)
    timesteps_flax = jnp.array([50], dtype=jnp.int32)
    flax_out = flax_unet.apply(
        {"params": flax_unet_params},
        sample_flax,
        timesteps_flax,
        jnp.array(encoder_hidden_states_np),
    ).sample

    np.testing.assert_allclose(
        pt_output,
        np.array(flax_out),
        rtol=1e-3,
        atol=1e-3,
        err_msg="Flax UNet forward output deviates from PyTorch reference!",
    )


if __name__ == "__main__":
  unittest.main()
