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

Key Parameters:
  model_name: The name of the model (e.g., "llama3.1-8b").
  maxtext_ckpt_path: Path to the MaxText checkpoint directory (e.g., .../checkpoints/100/model_params).
  hf_model_id: The base HuggingFace model ID for config mapping.
  output_dir: The directory where the HuggingFace adapter will be saved.
  lora_r: The rank of the LoRA adapter.
  lora_alpha: The alpha parameter for LoRA.

Example Usage:
  python src/maxtext/checkpoint_conversion/maxtext_to_hf_lora.py \
    model_name="llama3.1-8b" \
    maxtext_ckpt_path="/path/to/maxtext_lora/ckpt" \
    hf_model_id="meta-llama/Llama-3.1-8B" \
    output_dir="/path/to/hf/adapter/output"
"""

import os
import json
import numpy as np
import sys
from safetensors.numpy import save_file
from orbax import checkpoint as ocp
from etils import epath
from transformers import AutoConfig
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING

def parse_args(args):
    """Parses command line arguments in the format key=value."""
    parsed_args = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed_args[key] = value
    return parsed_args

def convert(model_name, maxtext_ckpt_path, hf_model_id, output_dir, lora_r=16, lora_alpha=32):
    print(f"[*] Starting conversion from {maxtext_ckpt_path}")
    
    # Initialize Orbax Checkpointer
    mngr = ocp.PyTreeCheckpointer()
    mt_params = mngr.restore(epath.Path(maxtext_ckpt_path))
    
    # Load HF Config for mapping
    hf_config = AutoConfig.from_pretrained(hf_model_id).to_dict()
    
    class MockConfig:
        scan_layers = True
        model_name = "llama3.1-8b"

    # Get the parameter mapping for the specific model
    mapping = PARAM_MAPPING[model_name](hf_config, MockConfig(), scan_layers=True)
    final_hf_weights = {}

    def process_data(current_dict, parent_path="decoder/layers"):
        """Recursive function to traverse MaxText params and map to HF."""
        for module_name, content in current_dict.items():
            path = f"{parent_path}/{module_name}"
            
            # Identify LoRA layers
            if isinstance(content, dict) and 'kernel_lora_a' in content:
                lookup_key = "params-" + path.replace("/", "-") + "-kernel"
                
                if lookup_key in mapping:
                    # Get the JAX values (as numpy)
                    data_a = np.array(content['kernel_lora_a']['value'])
                    data_b = np.array(content['kernel_lora_b']['value'])
                    hf_paths = mapping[lookup_key]
                    
                    # MaxText stacks multiple heads/projections, iterate through them
                    for i in range(data_a.shape[1]):
                        name = hf_paths[i].replace(".weight", "")
                        # Apply Transpose (.T) to match PyTorch dimension logic
                        final_hf_weights[f"base_model.model.{name}.lora_A.weight"] = data_a[:, i, :].T
                        final_hf_weights[f"base_model.model.{name}.lora_B.weight"] = data_b[:, i, :].T
                    
                    print(f"[DEBUG] Processed: {path}")
            
            elif isinstance(content, dict):
                process_data(content, path)

    # Start recursion
    process_data(mt_params['decoder']['layers'])

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
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "inference_mode": True
    }
    
    config_file = os.path.join(output_dir, "adapter_config.json")
    with open(config_file, "w") as f:
        json.dump(config_json, f, indent=4)

    print(f"\n[!] Conversion Complete!")
    print(f"    Saved weights to: {adapter_file}")
    print(f"    Saved config to: {config_file}")

if __name__ == "__main__":
    cli_args = parse_args(sys.argv[1:])
    
    # Required parameters check
    required = ["model_name", "maxtext_ckpt_path", "hf_model_id", "output_dir"]
    if not all(k in cli_args for k in required):
        print(__doc__)
        sys.exit(1)

    convert(
        model_name=cli_args["model_name"],
        maxtext_ckpt_path=cli_args["maxtext_ckpt_path"],
        hf_model_id=cli_args["hf_model_id"],
        output_dir=cli_args["output_dir"],
        lora_r=cli_args.get("lora_r", 16),
        lora_alpha=cli_args.get("lora_alpha", 32)
    )