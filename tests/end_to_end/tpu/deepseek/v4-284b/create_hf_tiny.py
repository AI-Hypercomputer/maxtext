import argparse
import os
import re
import sys

# Optional: if transformers is not in standard path or needs a custom checkout
transformers_repo_path = os.environ.get("TRANSFORMERS_REPO_PATH", "")
if transformers_repo_path:
  sys.path.insert(0, os.path.join(transformers_repo_path, "src"))

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

def convert_to_original_layout(state_dict):
  new_state_dict = {}
  for key, tensor in state_dict.items():
    # 1. Rename lm_head.weight to head.weight
    if key == "lm_head.weight":
      new_state_dict["head.weight"] = tensor
      continue
    
    # 2. Check if it's an expert weight that needs to be split
    # Format: model.layers.{i}.mlp.experts.gate_up_proj
    # Format: model.layers.{i}.mlp.experts.down_proj
    match_gate_up = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj", key)
    match_down = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.down_proj", key)
    
    if match_gate_up:
      layer_idx = int(match_gate_up.group(1))
      num_experts = tensor.shape[0]
      for e in range(num_experts):
        expert_tensor = tensor[e] # [2 * intermediate, hidden]
        # Split along axis 0 into w1 and w3
        w1, w3 = torch.chunk(expert_tensor, 2, dim=0) # each is [intermediate, hidden]
        new_state_dict[f"model.layers.{layer_idx}.mlp.experts.{e}.w1.weight"] = w1
        new_state_dict[f"model.layers.{layer_idx}.mlp.experts.{e}.w3.weight"] = w3
      continue
      
    if match_down:
      layer_idx = int(match_down.group(1))
      num_experts = tensor.shape[0]
      for e in range(num_experts):
        w2 = tensor[e] # [hidden, intermediate]
        new_state_dict[f"model.layers.{layer_idx}.mlp.experts.{e}.w2.weight"] = w2
      continue
      
    # 3. Otherwise, keep the key unchanged
    new_state_dict[key] = tensor
    
  return new_state_dict

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the HF checkpoint and config")
  args = parser.parse_args()

  config = DeepseekV4Config(
      num_hidden_layers=7,
      num_experts_per_tok=3,
      n_routed_experts=8,
      compress_ratios=[0, 0, 4, 128, 4, 128, 4],
      num_nextn_predict_layers=0,
  )

  print("Instantiating HF DeepseekV4 model with tiny config...")
  model = AutoModelForCausalLM.from_config(config)

  # Convert model parameters to bfloat16
  model = model.to(torch.bfloat16)

  print("Converting state dict to original format layout...")
  state_dict = model.state_dict()
  converted_state_dict = convert_to_original_layout(state_dict)

  print(f"Saving config and converted safetensors checkpoint to {args.save_dir}...")
  os.makedirs(args.save_dir, exist_ok=True)
  config.save_pretrained(args.save_dir)
  save_file(converted_state_dict, os.path.join(args.save_dir, "model.safetensors"))
  print("Done!")

if __name__ == "__main__":
  main()
