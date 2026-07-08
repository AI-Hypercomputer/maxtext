import sys
import os
import torch
from transformers import AutoModelForCausalLM

def main():
  hf_model_path = "tests/end_to_end/tpu/deepseek/v4-284b/hf_tiny_model"
  print("Loading model using dtype=torch.bfloat16...")
  model = AutoModelForCausalLM.from_pretrained(
      hf_model_path,
      dtype=torch.bfloat16,
      trust_remote_code=True,
  )
  print(f"model.model.layers[0].input_layernorm.weight dtype: {model.model.layers[0].input_layernorm.weight.dtype}")
  print(f"model.model.layers[0].self_attn.q_a_proj.weight dtype: {model.model.layers[0].self_attn.q_a_proj.weight.dtype}")

  print("\nLoading model using torch_dtype=torch.bfloat16...")
  model2 = AutoModelForCausalLM.from_pretrained(
      hf_model_path,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
  )
  print(f"model2.model.layers[0].input_layernorm.weight dtype: {model2.model.layers[0].input_layernorm.weight.dtype}")
  print(f"model2.model.layers[0].self_attn.q_a_proj.weight dtype: {model2.model.layers[0].self_attn.q_a_proj.weight.dtype}")

if __name__ == "__main__":
  main()
