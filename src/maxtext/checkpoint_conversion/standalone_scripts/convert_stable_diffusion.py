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

"""Converts Stable Diffusion v1.5 from Hugging Face into MaxText Orbax checkpoint format."""

import os
import sys
from typing import Sequence
import absl
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from maxtext.configs import pyconfig
from maxtext.diffusion.clip_text_encoder import FlaxCLIPTextModel
from maxtext.utils import max_logging, max_utils
from maxtext.utils.globals import HF_IDS

absl.logging.set_verbosity(absl.logging.INFO)


def convert_stable_diffusion_checkpoint(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    output_dir: str = "/home/hengtaoguo_google_com/projects/checkpoints/stable-diffusion-v1-5",
    cache_dir: str = "/dev/shm/hengtaoguo",
):
  """Converts SD 1.5 PyTorch weights to MaxText Orbax checkpoint format."""
  max_logging.log(f"Starting SD 1.5 checkpoint conversion from {model_id}...")
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(cache_dir, exist_ok=True)

  # 1. Convert Text Encoder
  max_logging.log("1/3: Converting Text Encoder weights...")
  pt_text = CLIPTextModel.from_pretrained(
      model_id, subfolder="text_encoder", cache_dir=cache_dir
  )
  pt_text.eval()
  pt_sd = pt_text.state_dict()

  text_model = FlaxCLIPTextModel()
  dummy_input = jnp.zeros((1, 77), dtype=jnp.int32)
  vars_init = text_model.init(jax.random.PRNGKey(0), dummy_input)
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
    text_params["encoder"][l_name]["self_attn"]["q_proj"]["kernel"] = jnp.array(
        pt_sd[f"encoder.layers.{i}.self_attn.q_proj.weight"].numpy().T
    )
    text_params["encoder"][l_name]["self_attn"]["q_proj"]["bias"] = jnp.array(
        pt_sd[f"encoder.layers.{i}.self_attn.q_proj.bias"].numpy()
    )
    text_params["encoder"][l_name]["self_attn"]["k_proj"]["kernel"] = jnp.array(
        pt_sd[f"encoder.layers.{i}.self_attn.k_proj.weight"].numpy().T
    )
    text_params["encoder"][l_name]["self_attn"]["k_proj"]["bias"] = jnp.array(
        pt_sd[f"encoder.layers.{i}.self_attn.k_proj.bias"].numpy()
    )
    text_params["encoder"][l_name]["self_attn"]["v_proj"]["kernel"] = jnp.array(
        pt_sd[f"encoder.layers.{i}.self_attn.v_proj.weight"].numpy().T
    )
    text_params["encoder"][l_name]["self_attn"]["v_proj"]["bias"] = jnp.array(
        pt_sd[f"encoder.layers.{i}.self_attn.v_proj.bias"].numpy()
    )
    text_params["encoder"][l_name]["self_attn"]["out_proj"]["kernel"] = (
        jnp.array(pt_sd[f"encoder.layers.{i}.self_attn.out_proj.weight"].numpy().T)
    )
    text_params["encoder"][l_name]["self_attn"]["out_proj"]["bias"] = jnp.array(
        pt_sd[f"encoder.layers.{i}.self_attn.out_proj.bias"].numpy()
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

  # 2. Convert UNet
  max_logging.log("2/3: Converting UNet weights...")
  _, unet_params = FlaxUNet2DConditionModel.from_pretrained(
      model_id, subfolder="unet", cache_dir=cache_dir, from_pt=True
  )

  # 3. Convert VAE
  max_logging.log("3/3: Converting VAE weights...")
  _, vae_params = FlaxAutoencoderKL.from_pretrained(
      model_id, subfolder="vae", cache_dir=cache_dir, from_pt=True
  )

  checkpoint_data = {
      "text_encoder": text_params,
      "unet": unet_params,
      "vae": vae_params,
  }

  max_logging.log(f"Saving MaxText Orbax checkpoint to {output_dir}...")
  import shutil
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
  checkpointer = ocp.PyTreeCheckpointer()
  checkpointer.save(os.path.abspath(output_dir), checkpoint_data)
  max_logging.log(f"SUCCESSFULLY saved converted checkpoint to {output_dir}!")


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    argv = [sys.argv[0], "src/maxtext/configs/base.yml", "model_name=stable-diffusion-v1.5"]
  config = pyconfig.initialize(argv)
  model_id = HF_IDS.get(config.model_name, "runwayml/stable-diffusion-v1-5")
  output_dir = getattr(
      config,
      "base_output_directory",
      "/home/hengtaoguo_google_com/projects/checkpoints/stable-diffusion-v1.5",
  )
  cache_dir = getattr(config, "hf_cache_dir", "/dev/shm/hengtaoguo")
  convert_stable_diffusion_checkpoint(
      model_id=model_id, output_dir=output_dir, cache_dir=cache_dir
  )


if __name__ == "__main__":
  main(sys.argv)
