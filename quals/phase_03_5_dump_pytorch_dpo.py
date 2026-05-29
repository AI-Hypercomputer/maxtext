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
Phase 3.5 Modular Block 1: PyTorch TRL DPO Loss Stack Dumper.
Executes original PyTorch tokenizer, next-token log-probabilities extraction,
and standard TRL preference Loss math on identical strings.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_batch_logps(logits, labels, loss_mask):
  """Computes the log probabilities of response tokens only, matching TRL DPOTrainer."""
  # Shift logits and labels by 1 step for next-token prediction
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous()
  shift_mask = loss_mask[..., 1:].contiguous()
  
  # Gather log-softmax values
  log_probs = F.log_softmax(shift_logits, dim=-1)
  
  # Gather target token log-probabilities
  per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
  
  # Mask out prompt and padding tokens
  per_token_logps = per_token_logps * shift_mask
  return per_token_logps.sum(-1)


def main():
  model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  print("=== Step 3.5.1: Extracting PyTorch TRL DPO Loss Reference ===")
  print(f"Loading model: {model_id}")

  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      device_map="cpu",
      trust_remote_code=True
  )
  model.eval()

  prompt_str = "What is DPO?"
  chosen_str = "DPO stands for Direct Preference Optimization, an algorithm for aligning LLMs."
  rejected_str = "DPO is a marketing strategy used to target customers' preferences."

  # Tokenize prompt, chosen, and rejected
  prompt_tokens = tokenizer.encode(prompt_str)
  chosen_tokens = tokenizer.encode(chosen_str)
  rejected_tokens = tokenizer.encode(rejected_str)

  # Prepare exact sizes matching JAX padding: max_prompt_length = 512, max_response_length = 512
  max_prompt_len = 512
  max_response_len = 512

  # 1. DPO Left-Padding for Prompt
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643 # standard pad for Qwen
  
  def left_pad(x, length):
    pad_width = max(length - len(x), 0)
    return [pad_id] * pad_width + x[-length:]

  def right_pad(x, length):
    pad_width = max(length - len(x), 0)
    return x[:length] + [pad_id] * pad_width

  padded_prompt = left_pad(prompt_tokens, max_prompt_len)
  padded_chosen = right_pad(chosen_tokens, max_response_len)
  padded_rejected = right_pad(rejected_tokens, max_response_len)

  # Concatenate prompt + chosen and prompt + rejected
  chosen_ids = padded_prompt + padded_chosen
  rejected_ids = padded_prompt + padded_rejected

  # Create loss masks (1 for response tokens, 0 for prompt/padding tokens)
  chosen_labels_mask = [0] * max_prompt_len + [1 if t != pad_id else 0 for t in padded_chosen]
  rejected_labels_mask = [0] * max_prompt_len + [1 if t != pad_id else 0 for t in padded_rejected]

  # Convert to tensors
  chosen_tensor = torch.tensor([chosen_ids], dtype=torch.long)
  rejected_tensor = torch.tensor([rejected_ids], dtype=torch.long)
  chosen_mask = torch.tensor([chosen_labels_mask], dtype=torch.float32)
  rejected_mask = torch.tensor([rejected_labels_mask], dtype=torch.float32)

  beta = 0.1 # standard DPO beta

  # Create attention masks matching JAX inputs_mask (1 for non-pad tokens)
  chosen_attention_mask = [0 if t == pad_id else 1 for t in chosen_ids]
  rejected_attention_mask = [0 if t == pad_id else 1 for t in rejected_ids]

  chosen_attn_tensor = torch.tensor([chosen_attention_mask], dtype=torch.long)
  rejected_attn_tensor = torch.tensor([rejected_attention_mask], dtype=torch.long)

  def get_jax_style_positions(attn_mask):
    positions = torch.cumsum(attn_mask, dim=-1)
    return positions - (positions >= 1).long()

  chosen_positions = get_jax_style_positions(chosen_attn_tensor)
  rejected_positions = get_jax_style_positions(rejected_attn_tensor)

  with torch.no_grad():
    # We run both Policy Model and Reference Model forward passes using JAX-style positions AND attention masks
    # 1. Policy logits
    policy_chosen_logits = model(
        chosen_tensor,
        attention_mask=chosen_attn_tensor,
        position_ids=chosen_positions
    ).logits.float()
    
    policy_rejected_logits = model(
        rejected_tensor,
        attention_mask=rejected_attn_tensor,
        position_ids=rejected_positions
    ).logits.float()

    # 2. Reference logits (identical since SFT baseline weights match reference model)
    ref_chosen_logits = policy_chosen_logits.clone()
    ref_rejected_logits = policy_rejected_logits.clone()

    # Compute Log-Probabilities on response tokens
    policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_tensor, chosen_mask)
    policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_tensor, rejected_mask)

    ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_tensor, chosen_mask)
    ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_tensor, rejected_mask)

    # Compute DPO Loss (Hugging Face TRL formulation)
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps
    
    delta = chosen_log_ratio - rejected_log_ratio
    loss = -F.logsigmoid(beta * delta).mean()

    # Compute rewards margin and rewards accuracy
    chosen_rewards = beta * chosen_log_ratio
    rejected_rewards = beta * rejected_log_ratio
    margin = chosen_rewards - rejected_rewards
    accuracy = (chosen_rewards > rejected_rewards).float().mean()

  print(f"PyTorch Chosen logps: {policy_chosen_logps.item():.6f}")
  print(f"PyTorch Rejected logps: {policy_rejected_logps.item():.6f}")
  print(f"PyTorch DPO Loss: {loss.item():.6f}")
  print(f"PyTorch Rewards Accuracy: {accuracy.item():.6f}")

  # Save outputs for multi-layered parity checks
  os.makedirs("quals/logs", exist_ok=True)
  np.save("quals/logs/pytorch_dpo_chosen_ids.npy", np.array(chosen_ids))
  np.save("quals/logs/pytorch_dpo_rejected_ids.npy", np.array(rejected_ids))
  np.save("quals/logs/pytorch_dpo_chosen_mask.npy", np.array(chosen_labels_mask))
  np.save("quals/logs/pytorch_dpo_rejected_mask.npy", np.array(rejected_labels_mask))
  
  np.save("quals/logs/pytorch_dpo_chosen_logps.npy", policy_chosen_logps.numpy())
  np.save("quals/logs/pytorch_dpo_rejected_logps.npy", policy_rejected_logps.numpy())
  np.save("quals/logs/pytorch_dpo_loss.npy", loss.numpy())
  np.save("quals/logs/pytorch_dpo_margin.npy", margin.numpy())
  np.save("quals/logs/pytorch_dpo_accuracy.npy", accuracy.numpy())

  print("SUCCESS: PyTorch DPO stack baseline parameters dumped successfully to quals/logs/")

if __name__ == "__main__":
  main()
