import argparse
import os
import sys

# Optional: if transformers is not in standard path or needs a custom checkout
transformers_repo_path = os.environ.get("TRANSFORMERS_REPO_PATH", "")
if transformers_repo_path:
  sys.path.insert(0, os.path.join(transformers_repo_path, "src"))

import torch
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

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

  print(f"Saving model to {args.save_dir}...")
  # We save with save_original_format=False to keep the Hugging Face format,
  # which matches MaxText's to_maxtext.py conversion expectations.
  model.save_pretrained(args.save_dir, save_original_format=False)
  print("Done!")

if __name__ == "__main__":
  main()
