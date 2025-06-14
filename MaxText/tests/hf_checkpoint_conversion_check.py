"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Sequence
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from absl import app  # Removed flags

from MaxText.utils.ckpt_conversion.utils.hf_utils import (
    check_predicted_tokens_match,
    check_arrays_match,
)
# Read Hugging Face token from environment variable
hf_token = os.environ.get("HF_AUTH_TOKEN")

"""
This script is to verify HuggingFace (HF) checkpoints that have been converted from MaxText. 

It loads the converted HF model and a "golden" (reference) HF model, and:
    1. runs a foward pass of the converted ckpt model
    2. compares their weights, output logits for a given input
    3. Compare the predicted token sequences
    
Extra Requirements:
    torch
    huggingface_hub
    transformers
    accelerate
"""


def get_all_modules(model):
  """Get all weights names from a HF model."""
  modules = []
  for name, _ in model.named_modules():
    if name and hasattr(model.get_submodule(name), "weight"):
      modules.append(name)
  return modules


def check_weights_match(model, golden_model, tol=0.1):
  """Compare weights between two HF models."""
  modules = get_all_modules(golden_model)

  for module in modules:
    golden_weights = golden_model.get_submodule(module).state_dict()["weight"]
    model_weight = model.get_submodule(module).state_dict()["weight"]
    check_arrays_match(golden_weights, model_weight, tol)


def get_logits(inputs, model, golden_model):
  """Get logits from two HF models for comparison."""
  logits = model(**inputs, output_hidden_states=True).logits
  golden_logits = golden_model(**inputs, output_hidden_states=True).logits

  return logits, golden_logits


def main(argv: Sequence[str]) -> None:
  # Parse arguments from argv
  # Default values
  parsed_args = {"golden_model_id": "google/gemma-2-2b-it", "hf_checkpoint_path": os.path.expanduser("~/.hf_output/")}
  for arg in argv[1:]:
    if "=" in arg:
      key, value = arg.split("=", 1)
      if key in parsed_args:
        parsed_args[key] = value
      else:
        print(f"Warning: Unknown argument '{key}' found in argv. Ignoring.")

  golden_model = AutoModelForCausalLM.from_pretrained(parsed_args["golden_model_id"], torch_dtype=torch.float32)

  tokenizer = AutoTokenizer.from_pretrained(parsed_args["hf_checkpoint_path"])
  model = AutoModelForCausalLM.from_pretrained(parsed_args["hf_checkpoint_path"], torch_dtype=torch.float32)

  # TODO: (@yixuannwang) use 3 prompts to verify
  input_text = "I love to"
  inputs = tokenizer(input_text, return_tensors="pt")
  # --- Generate Output ---
  with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=8)
  # --- Decode and Print ---
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))

  # Check weights match
  print("########### check weights match ############### ")
  check_weights_match(model, golden_model)

  # Run forward pass to get logits
  logits, golden_logits = get_logits(inputs, model, golden_model)

  # Check logits from the first 5 tokens match
  print("########### check logits match ############### ")
  check_arrays_match(logits[0, :5, :], golden_logits[0, :5, :], atol=0.2)

  print("########### check predicted token match ############### ")
  check_predicted_tokens_match(logits, golden_logits)


if __name__ == "__main__":
  app.run(main)
