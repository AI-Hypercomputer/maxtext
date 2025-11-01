#!/usr/bin/env python3
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
End-to-end SFT (Supervised Fine-Tuning) test script.

This script performs supervised fine-tuning on Llama3.1-8B-Instruct using the
HuggingFaceH4/ultrachat_200k dataset, based on the sft_llama3_demo.ipynb notebook.

Usage:
  HF_TOKEN=<your_hf_token> python3 end_to_end/tpu/test_sft.py

Environment Variables:
  HF_TOKEN: Hugging Face access token (required)
  MODEL_NAME: Model name (default: llama3.1-8b)
  STEPS: Number of training steps (default: 100)
  PER_DEVICE_BATCH_SIZE: Batch size per device (default: 1)
  MAX_TARGET_LENGTH: Maximum target length (default: 1024)
  LEARNING_RATE: Learning rate (default: 2.0e-5)
  BASE_OUTPUT_DIRECTORY: Output directory (default: /tmp/out/maxtext_llama3_8b)
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
from huggingface_hub import login

# Add MaxText to path
MAXTEXT_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(MAXTEXT_REPO_ROOT, "src"))

from MaxText import pyconfig
from MaxText.sft.sft_trainer import train as sft_train
from MaxText.vllm_decode import decode
from MaxText.globals import MAXTEXT_PKG_DIR


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(description="Run SFT training and decoding")
  parser.add_argument(
      "--hf_token",
      type=str,
      default=os.environ.get("HF_TOKEN"),
      help="Hugging Face access token",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default=os.environ.get("MODEL_NAME", "llama3.1-8b"),
      help="Model name",
  )
  parser.add_argument(
      "--model_checkpoint_path",
      type=str,
      default=os.environ.get("MODEL_CHECKPOINT_PATH", "/tmp/llama_checkpoint"),
      help="Path to model checkpoint",
  )
  parser.add_argument(
      "--base_output_directory",
      type=str,
      default=os.environ.get("BASE_OUTPUT_DIRECTORY", "/tmp/out/maxtext_llama3_8b"),
      help="Base output directory",
  )
  parser.add_argument(
      "--steps",
      type=int,
      default=int(os.environ.get("STEPS", "100")),
      help="Number of training steps",
  )
  parser.add_argument(
      "--per_device_batch_size",
      type=int,
      default=int(os.environ.get("PER_DEVICE_BATCH_SIZE", "1")),
      help="Per device batch size",
  )
  parser.add_argument(
      "--max_target_length",
      type=int,
      default=int(os.environ.get("MAX_TARGET_LENGTH", "1024")),
      help="Maximum target length",
  )
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=float(os.environ.get("LEARNING_RATE", "2.0e-5")),
      help="Learning rate",
  )
  parser.add_argument(
      "--skip_checkpoint_download",
      action="store_true",
      help="Skip downloading checkpoint from HuggingFace",
  )
  parser.add_argument(
      "--skip_training",
      action="store_true",
      help="Skip training (useful for testing decode only)",
  )
  parser.add_argument(
      "--skip_decode",
      action="store_true",
      help="Skip decode step",
  )
  parser.add_argument(
      "--prompt",
      type=str,
      default="Suggest some famous landmarks in London.",
      help="Prompt for decoding",
  )
  return parser.parse_args()


def download_checkpoint(args):
  """Download checkpoint from HuggingFace if needed."""
  if args.skip_checkpoint_download or os.path.exists(args.model_checkpoint_path):
    print(f"Using existing checkpoint at {args.model_checkpoint_path}")
    return

  print(f"Downloading checkpoint to {args.model_checkpoint_path}")
  import subprocess

  cmd = [
      "python3",
      "-m",
      "MaxText.utils.ckpt_conversion.to_maxtext",
      f"{MAXTEXT_PKG_DIR}/configs/base.yml",
      f"model_name={args.model_name}",
      f"base_output_directory={args.model_checkpoint_path}",
      f"hf_access_token={args.hf_token}",
      "use_multimodal=false",
      "scan_layers=false",
  ]

  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"Error downloading checkpoint: {result.stderr}")
    sys.exit(1)
  print("Checkpoint downloaded successfully")


def run_training(args):
  """Run SFT training."""
  if args.skip_training:
    print("Skipping training")
    return None, None

  print("=" * 60)
  print("STARTING SFT TRAINING")
  print("=" * 60)

  # Initialize JAX distributed if not already initialized
  if not jax.distributed.is_initialized():
    jax.distributed.initialize()

  print(f"JAX version: {jax.__version__}")
  print(f"JAX devices: {jax.devices()}")

  # Configure SFT training
  config_argv = [
      "",
      f"{MAXTEXT_PKG_DIR}/configs/sft.yml",
      f"load_parameters_path={args.model_checkpoint_path}",
      f"model_name={args.model_name}",
      f"steps={args.steps}",
      f"per_device_batch_size={args.per_device_batch_size}",
      f"max_target_length={args.max_target_length}",
      f"learning_rate={args.learning_rate}",
      "eval_steps=5",
      "weight_dtype=bfloat16",
      "dtype=bfloat16",
      "hf_path=HuggingFaceH4/ultrachat_200k",
      f"hf_access_token={args.hf_token}",
      f"base_output_directory={args.base_output_directory}",
      "run_name=sft_llama3_8b_test",
      "tokenizer_path=meta-llama/Llama-3.1-8B-Instruct",
      "eval_interval=10",
      "profiler=xplane",
  ]

  config = pyconfig.initialize(config_argv)

  print("Configuration loaded:")
  print(f"  - Model: {config.model_name}")
  print(f"  - Dataset: {config.hf_path}")
  print(f"  - Steps: {config.steps}")
  print(f"  - Use SFT: {config.use_sft}")
  print(f"  - Learning Rate: {config.learning_rate}")

  # Execute training
  trainer, mesh = sft_train(config)

  print("Training complete!")
  print(f"Model saved at: {args.base_output_directory}")

  return trainer, mesh


def run_decode(args, trainer, mesh):
  """Run decoding on trained model."""
  if args.skip_decode:
    print("Skipping decode")
    return

  print("=" * 60)
  print("STARTING DECODING")
  print("=" * 60)

  decode_config_argv = [
      "",
      f"{MAXTEXT_PKG_DIR}/configs/sft.yml",
      f"model_name={args.model_name}",
      f"per_device_batch_size={args.per_device_batch_size}",
      "max_target_length=128",
      "max_prefill_predict_length=64",
      "weight_dtype=bfloat16",
      "dtype=bfloat16",
      f"hf_access_token={args.hf_token}",
      "base_output_directory=/tmp/maxtext_output",
      "run_name=sft_llama3_decode",
      "tokenizer_path=meta-llama/Llama-3.1-8B-Instruct",
      f"prompt={args.prompt}",
      "use_chat_template=true",
      "decode_sampling_temperature=0.0",
      "decode_sampling_nucleus_p=1.0",
      "decode_sampling_top_k=0.0",
  ]

  decode_config = pyconfig.initialize(decode_config_argv)
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  decode(decode_config, trainer.model, mesh)

  print("Decoding complete!")


def main():
  """Main function."""
  args = parse_args()

  # Validate HF token
  if not args.hf_token:
    print("Error: HF_TOKEN is required. Set it as an environment variable or pass --hf_token")
    sys.exit(1)

  # Login to HuggingFace
  print("Logging in to HuggingFace...")
  login(token=args.hf_token)

  # Download checkpoint if needed
  download_checkpoint(args)

  # Run training
  trainer, mesh = run_training(args)

  # Run decode
  if trainer is not None and mesh is not None:
    run_decode(args, trainer, mesh)

  print("=" * 60)
  print("SFT TEST COMPLETED SUCCESSFULLY")
  print("=" * 60)


if __name__ == "__main__":
  main()
