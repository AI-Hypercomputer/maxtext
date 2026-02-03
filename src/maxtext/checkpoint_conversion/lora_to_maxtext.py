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

"""
This script converts a HuggingFace LoRA adapter to MaxText LoRA adapter format.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model (e.g., "llama3.1-8b").
  base_output_directory: (Required) The directory where the MaxText LoRA adapter
                         will be saved. Can be set in config file or as command-line override.
  hf_lora_adapter_path: (Required) Path to the HF LoRA adapter directory or HuggingFace repo ID.
  scan_layers: (bool) Whether the MaxText model uses scanned layers.
               This must match the training configuration.

Environment Variables:
  HF_AUTH_TOKEN: (Optional) HuggingFace authentication token if needed for adapter.

Example Usage:
  To convert HF LoRA to MaxText adapter:

    python src/MaxText/utils/ckpt_conversion/apply_lora.py \
    MaxText/configs/sft.yml model_name="llama3.1-8b" \
    hf_lora_adapter_path="username/lora-adapter-repo" \
    base_output_directory="/path/to/output/directory" \
    scan_layers=False
"""

import argparse
import os
import sys
import json
from typing import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from etils import epath
from flax import nnx

from orbax import checkpoint as ocp
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models, quantizations
from MaxText.utils.ckpt_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from MaxText.utils.ckpt_conversion.utils.utils import apply_hook_fns, HF_IDS
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import max_utils
from absl import logging


def load_hf_lora_adapter(adapter_path: str, hf_model_id: str) -> dict:
  """Load HF LoRA adapter weights directly from safetensors files."""
  max_logging.log(f"Loading HF LoRA adapter from {adapter_path}")
  
  # Check adapter compatibility
  adapter_config = None
  if os.path.isdir(adapter_path):
    # Local directory
    adapter_dir = epath.Path(adapter_path)
    config_file = adapter_dir / "adapter_config.json"
    if config_file.exists():
      with open(config_file, 'r') as f:
        adapter_config = json.load(f)
  else:
    # HF Hub repo
    try:
      config_file = hf_hub_download(
        adapter_path, 
        "adapter_config.json", 
        token=os.environ.get("HF_AUTH_TOKEN")
      )
      with open(config_file, 'r') as f:
        adapter_config = json.load(f)
    except Exception:
      max_logging.log("Warning: Could not load adapter_config.json from HF Hub")
  
  if adapter_config:
    base_model = adapter_config.get("base_model_name_or_path")
    # if base_model and base_model.replace("-Instruct", "") != hf_model_id.replace("-Instruct", ""):
    #   raise ValueError(f"Adapter base model '{base_model}' does not match expected model '{hf_model_id}'")
    max_logging.log(f"Adapter compatible with model {hf_model_id}")
  
  # Handle both local paths and HF Hub paths
  if os.path.isdir(adapter_path):
    # Local directory
    adapter_dir = epath.Path(adapter_path)
    adapter_files = list(adapter_dir.glob("*.safetensors"))
    if not adapter_files:
      adapter_files = list(adapter_dir.glob("*.bin"))
    if not adapter_files:
      raise ValueError(f"No LoRA adapter files found in {adapter_path}")
    adapter_file = adapter_files[0]
  else:
    # Assume it's a HF Hub repo ID
    try:
      # Try to download the adapter config to get the file list
      from huggingface_hub import list_repo_files
      files = list_repo_files(adapter_path, token=os.environ.get("HF_AUTH_TOKEN"))
      safetensor_files = [f for f in files if f.endswith('.safetensors')]
      if not safetensor_files:
        bin_files = [f for f in files if f.endswith('.bin')]
        if not bin_files:
          raise ValueError(f"No LoRA adapter files found in {adapter_path}")
        adapter_file = bin_files[0]
      else:
        adapter_file = safetensor_files[0]
      
      # Download the adapter file
      adapter_file = hf_hub_download(
        adapter_path, 
        adapter_file, 
        token=os.environ.get("HF_AUTH_TOKEN")
      )
    except Exception as e:
      raise ValueError(f"Failed to load LoRA adapter from {adapter_path}: {e}")
  
  # Load the adapter weights
  if adapter_file.endswith('.safetensors'):
    with safe_open(adapter_file, framework="numpy") as f:
      lora_weights = {k: f.get_tensor(k) for k in f.keys()}
  else:
    # For .bin files, we'd need torch.load, but safetensors is preferred
    raise ValueError(f"Unsupported adapter file format: {adapter_file}")
  
  max_logging.log(f"Loaded {len(lora_weights)} LoRA parameters from adapter")
  return lora_weights


def convert_hf_lora_key_to_maxtext(hf_key: str, param_mapping: dict, config) -> str:
  """Convert HF LoRA key to MaxText parameter path using the mapping from to_maxtext.py."""
  # HF LoRA keys: base_model.model.layers.{layer}.{module}.lora_A/B.weight
  
  # 1. Clean up LoRA suffixes to get the base module path
  # e.g. ...q_proj.lora_A.weight -> ...q_proj
  hf_param_key = hf_key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
  hf_param_key = hf_param_key.replace(".lora_A", "").replace(".lora_B", "")
  
  # 2. Handle prefix. Expected target is usually "model.layers..."
  # Input could be "base_model.model.model.layers..." or "base_model.model.layers..."
  if hf_param_key.startswith("base_model.model."):
    hf_param_key = hf_param_key[len("base_model.model."):]
  
  # 3. Search for the corresponding MaxText key
  for mt_key, hf_keys in param_mapping.items():
    if isinstance(hf_keys, list):
      for hf_k in hf_keys:
        # Match disregarding .weight suffix on the base model param
        if hf_k.replace(".weight", "") == hf_param_key:
          return mt_key
    elif isinstance(hf_keys, str):
      if hf_keys.replace(".weight", "") == hf_param_key:
        return mt_key
  
  return None


def convert_lora_to_maxtext_adapter(config, lora_weights: dict, output_path: str, hf_model_id: str):
    """Converts HF LoRA weights to MaxText adapter format without merging."""
    
    # 1. Setup Mesh and Model Structure (Abstractly)
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, axis_names=config.mesh_axes)
    quant = quantizations.configure_quantization(config)
    
    # Initialize rngs for model creation
    rngs = nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1))
    
    # Use the model definition to understand the target parameter paths
    model = models.Transformer(config=config, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN, rngs=rngs)

    hf_token = config.hf_access_token
    
    # Get the parameter mapping (MT -> HF)
    model_key = config.model_name
    if "-Instruct" in model_key:
        max_logging.log("Warning: You want an Instruct version, so we are using the base model architecture instead.")
        model_key = model_key.replace("-Instruct", "")
    hf_config_obj = AutoConfig.from_pretrained(hf_model_id, token=hf_token)
    param_map_mt_to_hf = PARAM_MAPPING[model_key](hf_config_obj.to_dict(), config, config.scan_layers)

    # 2. Initialize an empty dictionary for the MaxText Adapter
    mt_adapter_tree = {}
    mapped_count = 0

    # 3. Map HF LoRA weights to MaxText keys
    for hf_key, weight in lora_weights.items():
        # Identify the MaxText path for this specific HF weight
        mt_key = convert_hf_lora_key_to_maxtext(hf_key, param_map_mt_to_hf, config)
        
        if mt_key:
            # Determine if this is the 'A' or 'B' matrix
            suffix = "lora_A" if "lora_A" in hf_key else "lora_B"
            
            # Construct a nested dictionary path in mt_adapter_tree
            # MaxText expects: { 'decoder': { 'layers': { '0': { 'query': { 'lora_A': ... } } } } }
            parts = mt_key.split("/")
            current = mt_adapter_tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Convert weight to JAX array and store
            current[suffix] = jnp.array(weight)
            mapped_count += 1
        else:
            max_logging.log(f"Warning: Could not map HF LoRA key {hf_key} to MaxText key")

    max_logging.log(f"Successfully mapped {mapped_count} out of {len(lora_weights)} LoRA parameters")

    # 4. Save as a standalone adapter checkpoint
    max_logging.log(f"Saving MaxText LoRA adapter to {output_path}")
    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    ckptr.save(epath.Path(output_path), mt_adapter_tree)
    
    max_logging.log("LoRA adapter conversion completed successfully")


def main(args: Sequence[str]) -> None:
  # Set logging to INFO level to see max_logging.log messages
  logging.set_verbosity(logging.INFO)
  
  # Check if the user is using an Instruct version. If so, use the base model architecture
  original_model_name = None
  for i, arg in enumerate(args):
    if arg.startswith("model_name="):
      model_name_arg = args[i].split("=")[1]
      # Remove quotes if present
      model_name_arg = model_name_arg.strip("'").strip('"')
      original_model_name = model_name_arg
      
      if "-Instruct" in model_name_arg:
        max_logging.log("Warning: You want an Instruct version, so we are using the base model architecture instead.")
        model_name_arg = model_name_arg.replace("-Instruct", "")
        args[i] = f"model_name={model_name_arg}"
      break

  # Initialize maxtext config
  config = pyconfig.initialize(args)
  
  if not hasattr(config, 'hf_lora_adapter_path') or not config.hf_lora_adapter_path:
    raise ValueError("hf_lora_adapter_path must be specified")
  
  # Determine HF model ID and check if supported
  hf_model_id = HF_IDS.get(config.model_name)
  if hf_model_id is None:
    raise ValueError(f"Model '{config.model_name}' is not supported. Use a supported model_name from HF_IDS.")
  
  if not hasattr(config, 'base_output_directory') or not config.base_output_directory:
    raise ValueError("base_output_directory must be specified (in config file or as command-line argument)")
  
  output_dir = config.base_output_directory
  
  # Use original model name for output path
  model_name_for_path = original_model_name or config.model_name
  adapter_name = os.path.basename(config.hf_lora_adapter_path)
  full_output_path = os.path.join(output_dir, model_name_for_path, adapter_name)
  
  os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
  
  if os.path.exists(full_output_path):
    import shutil
    max_logging.log(f"Output directory {full_output_path} exists. Removing it to allow Orbax to save.")
    shutil.rmtree(full_output_path)
  
  # Load LoRA adapter and check compatibility
  lora_weights = load_hf_lora_adapter(config.hf_lora_adapter_path, hf_model_id)
  
  # Convert LoRA to MaxText adapter format and save
  convert_lora_to_maxtext_adapter(config, lora_weights, full_output_path, hf_model_id)
  
  # Verify output was created
  if not os.path.exists(full_output_path):
    raise RuntimeError(f"Failed to create output directory {full_output_path}")


if __name__ == "__main__":
  # Argument parsing similar to to_maxtext.py
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--simulated_cpu_devices_count", type=int, required=False, default=16
  )
  
  # Parse local arguments
  local_args, remaining_args = parser.parse_known_args()
  
  # Reconstruct model_args (script name + the args MaxText needs)
  model_args = [sys.argv[0]] + remaining_args

  # Set jax environment
  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={local_args.simulated_cpu_devices_count}"
  
  main(model_args)