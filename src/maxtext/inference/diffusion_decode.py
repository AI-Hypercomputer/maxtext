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

"""CLI utility for running text-to-image diffusion inference in MaxText."""

import os
import sys
from typing import Sequence
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image
import torch
from transformers import CLIPTokenizer

from diffusers import (
    FlaxAutoencoderKL,
    FlaxDPMSolverMultistepScheduler,
    FlaxPNDMScheduler,
    FlaxUNet2DConditionModel,
)
from maxtext.configs import pyconfig
from maxtext.diffusion.clip_text_encoder import FlaxCLIPTextModel
from maxtext.utils import max_logging, max_utils
from maxtext.utils.globals import HF_IDS


def load_model_weights(config):
  """Loads model weights for CLIP Text Encoder, UNet, and VAE."""
  model_id = HF_IDS.get(config.model_name, "runwayml/stable-diffusion-v1-5")
  cache_dir = getattr(config, "hf_cache_dir", "/dev/shm/hengtaoguo")

  max_logging.log(f"Loading tokenizer and text encoder for {model_id}...")
  tokenizer = CLIPTokenizer.from_pretrained(
      model_id, subfolder="text_encoder", cache_dir=cache_dir
  )

  # Check if loading from converted MaxText Orbax checkpoint
  load_path = getattr(config, "load_parameters_path", "")
  if load_path and os.path.exists(load_path):
    try:
      max_logging.log(f"Loading parameters from MaxText checkpoint: {load_path}")
      checkpointer = ocp.PyTreeCheckpointer()
      ckpt_params = checkpointer.restore(load_path)
      text_params = ckpt_params.get("text_encoder", None)
      unet_params = ckpt_params.get("unet", None)
      vae_params = ckpt_params.get("vae", None)
    except Exception as e:
      max_logging.log(
          f"Warning: Failed to restore checkpoint from {load_path} ({e}), falling back to direct weights initialization."
      )
      text_params, unet_params, vae_params = None, None, None
  else:
    text_params, unet_params, vae_params = None, None, None

  # Initialize / load Text Encoder
  text_encoder = FlaxCLIPTextModel()
  if text_params is None:
    from transformers import CLIPTextModel as PtCLIPTextModel

    pt_text = PtCLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", cache_dir=cache_dir
    )
    pt_text.eval()
    pt_sd = pt_text.state_dict()

    dummy_input = jnp.zeros((1, 77), dtype=jnp.int32)
    vars_init = text_encoder.init(jax.random.PRNGKey(0), dummy_input)
    text_params = vars_init["params"]

    text_params["embeddings"]["token_embedding"]["embedding"] = jnp.array(
        pt_sd["embeddings.token_embedding.weight"].numpy()
    )
    text_params["embeddings"]["position_embedding"]["embedding"] = jnp.array(
        pt_sd["embeddings.position_embedding.weight"].numpy()
    )
    text_params["final_layer_norm"]["scale"] = jnp.array(
        pt_sd["final_layer_norm.weight"].numpy()
    )
    text_params["final_layer_norm"]["bias"] = jnp.array(
        pt_sd["final_layer_norm.bias"].numpy()
    )

    for i in range(12):
      l_name = f"layers_{i}"
      text_params["encoder"][l_name]["layer_norm1"]["scale"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm1.weight"].numpy()
      )
      text_params["encoder"][l_name]["layer_norm1"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm1.bias"].numpy()
      )
      text_params["encoder"][l_name]["self_attn"]["q_proj"]["kernel"] = (
          jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.q_proj.weight"].numpy().T)
      )
      text_params["encoder"][l_name]["self_attn"]["q_proj"]["bias"] = (
          jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.q_proj.bias"].numpy())
      )
      text_params["encoder"][l_name]["self_attn"]["k_proj"]["kernel"] = (
          jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.k_proj.weight"].numpy().T)
      )
      text_params["encoder"][l_name]["self_attn"]["k_proj"]["bias"] = (
          jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.k_proj.bias"].numpy())
      )
      text_params["encoder"][l_name]["self_attn"]["v_proj"]["kernel"] = (
          jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.v_proj.weight"].numpy().T)
      )
      text_params["encoder"][l_name]["self_attn"]["v_proj"]["bias"] = (
          jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.v_proj.bias"].numpy())
      )
      text_params["encoder"][l_name]["self_attn"]["out_proj"]["kernel"] = (
          jnp.array(
              pt_sd[f"encoder.layers.{i}.self_attn.out_proj.weight"].numpy().T
          )
      )
      text_params["encoder"][l_name]["self_attn"]["out_proj"]["bias"] = (
          jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.out_proj.bias"].numpy())
      )
      text_params["encoder"][l_name]["layer_norm2"]["scale"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm2.weight"].numpy()
      )
      text_params["encoder"][l_name]["layer_norm2"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.layer_norm2.bias"].numpy()
      )
      text_params["encoder"][l_name]["mlp"]["fc1"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc1.weight"].numpy().T
      )
      text_params["encoder"][l_name]["mlp"]["fc1"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc1.bias"].numpy()
      )
      text_params["encoder"][l_name]["mlp"]["fc2"]["kernel"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc2.weight"].numpy().T
      )
      text_params["encoder"][l_name]["mlp"]["fc2"]["bias"] = jnp.array(
          pt_sd[f"encoder.layers.{i}.mlp.fc2.bias"].numpy()
      )

  # Initialize / load UNet
  max_logging.log(f"Loading UNet for {model_id}...")
  if unet_params is None:
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", cache_dir=cache_dir, from_pt=True
    )
  else:
    unet = FlaxUNet2DConditionModel(
        sample_size=getattr(config, "sample_size", 64),
        in_channels=getattr(config, "in_channels", 4),
        out_channels=getattr(config, "out_channels", 4),
        block_out_channels=tuple(
            getattr(config, "block_out_channels", [320, 640, 1280, 1280])
        ),
        layers_per_block=getattr(config, "layers_per_block", 2),
        attention_head_dim=getattr(config, "attention_head_dim", 8),
        cross_attention_dim=getattr(config, "cross_attention_dim", 768),
    )

  # Initialize / load VAE
  max_logging.log(f"Loading VAE for {model_id}...")
  if vae_params is None:
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        model_id, subfolder="vae", cache_dir=cache_dir, from_pt=True
    )
  else:
    vae = FlaxAutoencoderKL(
        in_channels=getattr(config, "vae_in_channels", 3),
        out_channels=getattr(config, "vae_out_channels", 3),
        latent_channels=getattr(config, "vae_latent_channels", 4),
        scaling_factor=getattr(config, "vae_scaling_factor", 0.18215),
    )

  # Initialize Scheduler
  scheduler_type = getattr(config, "scheduler_type", "pndm").lower()
  if scheduler_type == "dpmsolver":
    scheduler, scheduler_state = (
        FlaxDPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler", cache_dir=cache_dir
        )
    )
  else:
    scheduler, scheduler_state = FlaxPNDMScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=cache_dir
    )

  return (
      tokenizer,
      text_encoder,
      text_params,
      unet,
      unet_params,
      vae,
      vae_params,
      scheduler,
      scheduler_state,
  )


def generate_image(config):
  """Executes full diffusion pipeline to generate image from text prompt."""
  (
      tokenizer,
      text_encoder,
      text_params,
      unet,
      unet_params,
      vae,
      vae_params,
      scheduler,
      scheduler_state,
  ) = load_model_weights(config)

  prompt = config.prompt
  negative_prompt = getattr(config, "negative_prompt", "")
  max_logging.log(f"Prompt: '{prompt}'")
  max_logging.log(f"Negative Prompt: '{negative_prompt}'")

  # 1. Encode text prompts
  text_inputs = tokenizer(
      [prompt], padding="max_length", max_length=77, return_tensors="pt"
  )
  uncond_inputs = tokenizer(
      [negative_prompt], padding="max_length", max_length=77, return_tensors="pt"
  )

  text_embeddings = text_encoder.apply(
      {"params": text_params}, jnp.array(text_inputs.input_ids.numpy())
  )
  uncond_embeddings = text_encoder.apply(
      {"params": text_params}, jnp.array(uncond_inputs.input_ids.numpy())
  )

  context = jnp.concatenate([uncond_embeddings, text_embeddings], axis=0)

  # 2. Initialize latent noise
  seed = getattr(config, "seed", 42)
  prng_key = jax.random.PRNGKey(seed)
  prng_key, subkey = jax.random.split(prng_key)

  height = int(getattr(config, "image_height", 512))
  width = int(getattr(config, "image_width", 512))
  latent_h, latent_w = height // 8, width // 8
  batch_size = int(getattr(config, "batch_size", 1))

  latents = jax.random.normal(
      subkey, shape=(batch_size, 4, latent_h, latent_w), dtype=jnp.float32
  )
  latents = latents * scheduler_state.init_noise_sigma

  # 3. Diffusion sampling loop
  num_inference_steps = getattr(config, "num_inference_steps", 50)
  guidance_scale = getattr(config, "guidance_scale", 7.5)

  scheduler_state = scheduler.set_timesteps(
      scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape
  )
  timesteps = scheduler_state.timesteps

  max_logging.log(
      f"Starting diffusion sampling loop on {len(jax.devices())} JAX device(s) ({num_inference_steps} steps)..."
  )

  for i, t in enumerate(timesteps):
    latent_input = jnp.concatenate([latents, latents], axis=0)
    latent_input = scheduler.scale_model_input(scheduler_state, latent_input, t)

    t_vec = jnp.broadcast_to(t, (latent_input.shape[0],))
    noise_pred = unet.apply(
        {"params": unet_params}, latent_input, t_vec, context
    ).sample

    noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2, axis=0)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    latents, scheduler_state = scheduler.step(
        scheduler_state, noise_pred, t, latents, return_dict=False
    )
    if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
      max_logging.log(f"Sampling Step {i + 1}/{len(timesteps)} completed.")

  # 4. Decode latents with VAE
  max_logging.log("Decoding latents with VAE...")
  vae_scaling_factor = getattr(config, "vae_scaling_factor", 0.18215)
  scaled_latents = (1.0 / vae_scaling_factor) * latents
  image = vae.apply(
      {"params": vae_params}, scaled_latents, method=vae.decode
  ).sample

  # 5. Post-process and save image
  image = (image / 2 + 0.5)
  image = jnp.clip(image, 0, 1)
  image = np.array(image[0])
  image = np.transpose(image, (1, 2, 0))  # (H, W, C)
  image = (image * 255).round().astype(np.uint8)

  pil_img = Image.fromarray(image)
  output_path = getattr(
      config,
      "output_image_path",
      "/home/hengtaoguo_google_com/projects/astronaut_rides_horse.png",
  )
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  pil_img.save(output_path)
  max_logging.log(f"Image successfully generated and saved to: {output_path}")
  return output_path


def main(argv: Sequence[str]) -> None:
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if len(argv) < 2:
    argv = [sys.argv[0], "src/maxtext/configs/base.yml", "model_name=stable-diffusion-v1.5"]
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  generate_image(config)


if __name__ == "__main__":
  main(sys.argv)
