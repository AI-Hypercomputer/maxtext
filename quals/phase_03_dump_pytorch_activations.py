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
Phase 3 Modular Block 1: PyTorch Logit/Activation Dumper.
Loads original PyTorch SFT model, hooks layer 0, and dumps chosen/rejected states to separate files.
"""

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

pytorch_activations = {}

def main():
  model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  print("=== Step 3.1: Extracting PyTorch Logit Baselines & Layer 0 Activations ===")
  print(f"Loading model: {model_id}")

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      device_map="cpu",
      trust_remote_code=True
  )
  model.eval()

  # Register hooks on Layer 0 projections
  layer_0 = model.model.layers[0]
  def make_hook(name):
    def hook(module, input_tensor, output_tensor):
      if isinstance(output_tensor, tuple):
        pytorch_activations[name] = output_tensor[0].float().detach().cpu().numpy()[0]
      else:
        pytorch_activations[name] = output_tensor.float().detach().cpu().numpy()[0]
    return hook

  layer_0.self_attn.q_proj.register_forward_hook(make_hook('q_proj'))
  layer_0.self_attn.k_proj.register_forward_hook(make_hook('k_proj'))
  layer_0.self_attn.v_proj.register_forward_hook(make_hook('v_proj'))
  layer_0.self_attn.o_proj.register_forward_hook(make_hook('o_proj'))
  layer_0.mlp.down_proj.register_forward_hook(make_hook('mlp_out'))

  prompt = "What is DPO?"
  chosen = "DPO stands for Direct Preference Optimization, an algorithm for aligning LLMs."
  rejected = "DPO is a marketing strategy used to target customers' preferences."

  chosen_ids = tokenizer.encode(prompt + chosen)
  rejected_ids = tokenizer.encode(prompt + rejected)

  os.makedirs("quals/logs", exist_ok=True)

  with torch.no_grad():
    # 1. Execute Chosen Pass
    chosen_tensor = torch.tensor([chosen_ids], dtype=torch.long)
    py_embeds = model.model.embed_tokens(chosen_tensor).float().numpy()[0]
    chosen_logits = model(chosen_tensor).logits.float().numpy()[0]
    
    # Dump Chosen intermediate activations immediately to prevent overwrite
    np.save("quals/logs/pytorch_chosen_embeds.npy", py_embeds)
    np.save("quals/logs/pytorch_chosen_logits.npy", chosen_logits)
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp_out']:
      np.save(f"quals/logs/pytorch_chosen_{name}.npy", pytorch_activations[name])
      
    # Clear activations cache
    pytorch_activations.clear()

    # 2. Execute Rejected Pass
    rejected_tensor = torch.tensor([rejected_ids], dtype=torch.long)
    rejected_logits = model(rejected_tensor).logits.float().numpy()[0]
    
    np.save("quals/logs/pytorch_rejected_logits.npy", rejected_logits)
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp_out']:
      np.save(f"quals/logs/pytorch_rejected_{name}.npy", pytorch_activations[name])
      
  print("SUCCESS: PyTorch SFT baseline activations successfully dumped to quals/logs/")

if __name__ == "__main__":
  main()
