# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script converts a MaxText LoRA adapter (checkpoint) back to HuggingFace PEFT format.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model (e.g., "llama3.1-8b").
  load_parameters_path: (Required) Path to the MaxText checkpoint directory.
  base_output_directory: (Required) The directory where the HuggingFace adapter will be saved.
  lora_rank: The rank of the LoRA adapter.
  lora_alpha: The alpha parameter for LoRA.

Example Usage:
  python src/maxtext/checkpoint_conversion/maxtext_to_hf_lora.py \
    src/maxtext/configs/base.yml \
    model_name="llama3.1-8b" \
    load_parameters_path="maxtext/lora/ckpt_path/" \
    base_output_directory="output/path/" \
    lora_rank=16 \
    lora_alpha=32
"""

import argparse
import jax
import os
import json
import numpy as np
import sys
from safetensors.numpy import save_file
from orbax import checkpoint as ocp
from etils import epath
from transformers import AutoConfig
from maxtext.configs import pyconfig
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
from maxtext.utils.globals import HF_IDS


def convert(argv):
  """Converts a MaxText LoRA adapter checkpoint to HuggingFace PEFT format."""
  config = pyconfig.initialize(argv)

  if not hasattr(config, "load_parameters_path") or not config.load_parameters_path:
    raise ValueError("load_parameters_path must be specified")

  if not hasattr(config, "base_output_directory") or not config.base_output_directory:
    raise ValueError("base_output_directory must be specified")

  maxtext_ckpt_path = config.load_parameters_path
  output_dir = config.base_output_directory
  model_name = config.model_name
  lora_r = config.lora_rank
  lora_alpha = config.lora_alpha

  hf_model_id = HF_IDS.get(config.model_name)
  if hf_model_id is None:
    raise ValueError(f"Model '{config.model_name}' is not supported. Use a supported model_name from HF_IDS.")

  print(f"[*] Starting conversion for model: {model_name}")
  print(f"[*] Path: {maxtext_ckpt_path}")

  mapping_model_name = config.model_name.replace("-Instruct", "")

  # Initialize Orbax Checkpointer
  mngr = ocp.PyTreeCheckpointer()
  mt_params = mngr.restore(epath.Path(maxtext_ckpt_path))

  # Load HF Config for mapping
  hf_config = AutoConfig.from_pretrained(hf_model_id).to_dict()

  # Get the parameter mapping for the specific model
  if mapping_model_name not in PARAM_MAPPING:
    raise ValueError(f"Model {mapping_model_name} not found in PARAM_MAPPING")

  mapping = PARAM_MAPPING[mapping_model_name](hf_config, config, config.scan_layers)

  final_hf_weights = {}
  found_hf_modules = set()

  def process_data(current_dict, parent_path="decoder/layers"):
    """Recursive function to traverse MaxText params and map to HF."""
    for module_name, content in current_dict.items():
      path = f"{parent_path}/{module_name}"

      # Identify LoRA layers
      if isinstance(content, dict) and "kernel_lora_a" in content:
        lookup_key = "params-" + path.replace("/", "-") + "-kernel"

        if lookup_key in mapping:
          # Get the JAX values (as numpy)
          data_a = np.array(content["kernel_lora_a"]["value"])
          data_b = np.array(content["kernel_lora_b"]["value"])
          hf_paths = mapping[lookup_key]

          if not isinstance(hf_paths, list):
            hf_paths = [hf_paths]

          # MaxText stacks multiple heads/projections, iterate through them
          for i in range(min(data_a.shape[1], len(hf_paths))):
            full_hf_path = hf_paths[i]

            module_type = full_hf_path.split(".")[-2]
            found_hf_modules.add(module_type)

            name = hf_paths[i].replace(".weight", "")
            # Apply Transpose (.T) to match PyTorch dimension logic
            final_hf_weights[f"base_model.model.{name}.lora_A.weight"] = data_a[:, i, :].T
            final_hf_weights[f"base_model.model.{name}.lora_B.weight"] = data_b[:, i, :].T

          print(f"[DEBUG] Mapped {lookup_key} to {len(hf_paths)} HF layers")

      elif isinstance(content, dict):
        process_data(content, path)

  # Start recursion
  start_node = mt_params.get("decoder", {}).get("layers", mt_params)
  process_data(start_node)

  # Save Safetensors
  os.makedirs(output_dir, exist_ok=True)
  adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
  save_file(final_hf_weights, adapter_file)

  # Create PEFT adapter_config.json
  config_json = {
      "base_model_name_or_path": hf_model_id,
      "peft_type": "LORA",
      "task_type": "CAUSAL_LM",
      "r": int(lora_r),
      "lora_alpha": int(lora_alpha),
      "target_modules": list(found_hf_modules),
      "lora_dropout": 0.0,
      "bias": "none",
      "inference_mode": True,
  }

  config_file = os.path.join(output_dir, "adapter_config.json")
  with open(config_file, "w", encoding="utf-8") as f:
    json.dump(config_json, f, indent=4)

  print("\n[!] Conversion Complete!")
  print(f"    Saved weights to: {adapter_file}")
  print(f"    Saved config to: {config_file}")
  print(f"[!] Target modules detected: {list(found_hf_modules)}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--simulated_cpu_devices_count", type=int, required=False, default=16)

  # Parse local arguments
  local_args, remaining_args = parser.parse_known_args()

  # Reconstruct model_args (script name + the args MaxText needs)
  model_args = [sys.argv[0]] + remaining_args

  # Set jax environment
  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={local_args.simulated_cpu_devices_count}"

  convert(model_args)
