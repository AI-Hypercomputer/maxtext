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
Unified GRPO Script

This script provides a unified CLI interface for running GRPO training
across different model sizes and configurations. It uses the grpo_train function
from grpo_tunix_trainer.py which consolidates all the GRPO-specific logic.

Usage Examples:

# Llama3.1-8B (single host)
python3 src/MaxText/examples/grpo_runner.py \\
  --model_name=llama3.1-8b \\
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \\
  --load_parameters_path=gs://path/to/checkpoint \\
  --base_output_directory=/tmp/grpo_output \\
  --hf_access_token=$HF_TOKEN \\
  --steps=100

# Llama3.1-70B with Pathways (multi-host)
python3 src/MaxText/examples/grpo_runner.py \\
  --model_name=llama3.1-70b \\
  --tokenizer_path=meta-llama/Llama-3.1-70B-Instruct \\
  --load_parameters_path=gs://path/to/checkpoint \\
  --base_output_directory=gs://path/to/output \\
  --hf_access_token=$HF_TOKEN \\
  --use_pathways=true \\
  --steps=100

# Custom dataset
python3 src/MaxText/examples/grpo_runner.py \\
  --model_name=llama3.1-8b \\
  --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \\
  --load_parameters_path=gs://path/to/checkpoint \\
  --base_output_directory=/tmp/grpo_output \\
  --hf_access_token=$HF_TOKEN \\
  --hf_path=custom/dataset \\
  --steps=100
"""

import argparse
import os
import sys

# Add MaxText to path
script_dir = os.path.dirname(os.path.abspath(__file__))
maxtext_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, maxtext_root)

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.experimental.rl.grpo_tunix_trainer import grpo_train


def create_parser():
  """Create argument parser for GRPO demo."""
  parser = argparse.ArgumentParser(
      description="Unified GRPO Script",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )

  # Model Configuration
  parser.add_argument(
      "--model_name",
      type=str,
      required=True,
      help="Model name (e.g., llama3.1-8b, llama3.1-70b)",
  )
  parser.add_argument(
      "--tokenizer_path",
      type=str,
      required=True,
      help="HuggingFace tokenizer path (e.g., meta-llama/Llama-3.1-8B-Instruct)",
  )
  parser.add_argument(
      "--load_parameters_path",
      type=str,
      required=True,
      help="Path to model checkpoint (local or gs://)",
  )

  # Output Configuration
  parser.add_argument(
      "--base_output_directory",
      type=str,
      required=True,
      help="Base output directory for logs and checkpoints",
  )
  parser.add_argument(
      "--run_name",
      type=str,
      default=None,
      help="Run name for this experiment",
  )

  # Dataset Configuration
  parser.add_argument(
      "--hf_access_token",
      type=str,
      default=os.environ.get("HF_TOKEN"),
      help="HuggingFace access token (default: $HF_TOKEN env var)",
  )
  parser.add_argument(
      "--hf_path",
      type=str,
      default="gsm8k",
      help="HuggingFace dataset path",
  )
  parser.add_argument(
      "--hf_data_split",
      type=str,
      default="main",
      help="HuggingFace dataset split",
  )
  parser.add_argument(
      "--hf_data_files",
      type=str,
      default="train",
      help="HuggingFace dataset files",
  )

  # Training Configuration
  parser.add_argument(
      "--steps",
      type=int,
      default=100,
      help="Number of training steps",
  )
  parser.add_argument(
      "--per_device_batch_size",
      type=int,
      default=1,
      help="Per device batch size",
  )
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=3e-6,
      help="Learning rate",
  )

  # GRPO-Specific Parameters
  parser.add_argument(
      "--num_generations",
      type=int,
      default=2,
      help="Number of generations per prompt (group size)",
  )
  parser.add_argument(
      "--grpo_beta",
      type=float,
      default=0.08,
      help="KL divergence penalty coefficient",
  )
  parser.add_argument(
      "--grpo_epsilon",
      type=float,
      default=0.2,
      help="Clipping value for stable updates",
  )

  # Sequence Lengths
  parser.add_argument(
      "--max_prefill_predict_length",
      type=int,
      default=256,
      help="Maximum prompt length",
  )
  parser.add_argument(
      "--max_target_length",
      type=int,
      default=768,
      help="Maximum total sequence length",
  )

  # Pathways/Multi-Host Configuration
  parser.add_argument(
      "--use_pathways",
      action="store_true",
      help="Use Pathways for multi-host training",
  )
  parser.add_argument(
      "--inference_devices_per_replica",
      type=int,
      default=4,
      help="Number of devices per inference replica",
  )
  parser.add_argument(
      "--inference_replicas",
      type=int,
      default=1,
      help="Number of inference replicas",
  )

  # Parallelism Configuration
  parser.add_argument(
      "--ici_fsdp_parallelism",
      type=int,
      default=-1,
      help="FSDP parallelism (-1 for auto)",
  )
  parser.add_argument(
      "--ici_tensor_parallelism",
      type=int,
      default=-1,
      help="Tensor parallelism (-1 for auto)",
  )

  # Device Allocation Configuration
  parser.add_argument(
      "--trainer_devices_fraction",
      type=float,
      default=0.5,
      help="Fraction of devices for training (0.0-1.0)",
  )
  parser.add_argument(
      "--sampler_devices_fraction",
      type=float,
      default=0.5,
      help="Fraction of devices for sampling (0.0-1.0)",
  )
  parser.add_argument(
      "--chips_per_vm",
      type=int,
      default=4,
      help="Number of chips per VM (hardware dependent)",
  )

  # Other Configuration
  parser.add_argument(
      "--profiler",
      type=str,
      default="xplane",
      help="Profiler to use (xplane, none)",
  )
  parser.add_argument(
      "--checkpoint_period",
      type=int,
      default=50,
      help="Checkpoint saving period",
  )
  parser.add_argument(
      "--config_file",
      type=str,
      default=None,
      help="Optional custom config file (overrides grpo.yml)",
  )

  return parser


def build_config_argv(args):
  """Build configuration arguments for MaxText pyconfig."""
  # Use custom config or default grpo.yml
  if args.config_file:
    base_config = args.config_file
  else:
    base_config = os.path.join(MAXTEXT_PKG_DIR, "configs", "grpo.yml")

  # Build training config argv
  config_argv = [
      "",  # Placeholder for argv[0]
      base_config,
      f"model_name={args.model_name}",
      f"tokenizer_path={args.tokenizer_path}",
      f"load_parameters_path={args.load_parameters_path}",
      f"base_output_directory={args.base_output_directory}",
      f"steps={args.steps}",
      f"per_device_batch_size={args.per_device_batch_size}",
      f"learning_rate={args.learning_rate}",
      f"num_generations={args.num_generations}",
      f"grpo_beta={args.grpo_beta}",
      f"grpo_epsilon={args.grpo_epsilon}",
      f"max_prefill_predict_length={args.max_prefill_predict_length}",
      f"max_target_length={args.max_target_length}",
      f"profiler={args.profiler}",
      f"checkpoint_period={args.checkpoint_period}",
      f"hf_path={args.hf_path}",
      f"hf_data_split={args.hf_data_split}",
      f"hf_data_files={args.hf_data_files}",
  ]

  # Add optional parameters
  if args.run_name:
    config_argv.append(f"run_name={args.run_name}")

  if args.hf_access_token:
    config_argv.append(f"hf_access_token={args.hf_access_token}")

  if args.use_pathways:
    config_argv.append("use_pathways_reshard=True")
    config_argv.append(f"inference_devices_per_replica={args.inference_devices_per_replica}")
    config_argv.append(f"inference_replicas={args.inference_replicas}")

  if args.ici_fsdp_parallelism > 0:
    config_argv.append(f"ici_fsdp_parallelism={args.ici_fsdp_parallelism}")

  if args.ici_tensor_parallelism > 0:
    config_argv.append(f"ici_tensor_parallelism={args.ici_tensor_parallelism}")

  # Add device allocation parameters
  config_argv.append(f"trainer_devices_fraction={args.trainer_devices_fraction}")
  config_argv.append(f"sampler_devices_fraction={args.sampler_devices_fraction}")
  config_argv.append(f"chips_per_vm={args.chips_per_vm}")

  return config_argv


def main():
  """Main entry point for GRPO demo."""
  parser = create_parser()
  args = parser.parse_args()

  # Validate required environment/arguments
  if not args.hf_access_token:
    print("Error: HF_TOKEN is required. Set it as an environment variable or pass --hf_access_token")
    sys.exit(1)

  print("=" * 80)
  print("GRPO - Unified Training Script")
  print("=" * 80)
  print(f"Model: {args.model_name}")
  print(f"Tokenizer: {args.tokenizer_path}")
  print(f"Checkpoint: {args.load_parameters_path}")
  print(f"Dataset: {args.hf_path}")
  print(f"Output: {args.base_output_directory}")
  print(f"Steps: {args.steps}")
  print(f"GRPO Beta: {args.grpo_beta}")
  print(f"Num Generations: {args.num_generations}")
  print(f"Use Pathways: {args.use_pathways}")
  print("=" * 80)

  # Build config arguments
  config_argv = build_config_argv(args)

  # Initialize configuration
  config = pyconfig.initialize(config_argv)

  # Run GRPO training using the unified trainer
  grpo_train(config)

  print("=" * 80)
  print("GRPO Training Completed Successfully!")
  print("=" * 80)


if __name__ == "__main__":
  main()
