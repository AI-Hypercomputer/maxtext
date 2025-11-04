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

# pylint: disable=bare-except, consider-using-generator
"""
This tutorial demonstrates training the Llama3.1 8B-IT model on
the GSM8K math reasoning benchmark using Group Relative Policy Optimization (GRPO).
GRPO can enhance your model's problem-solving skills on mathematical word problems,
coding problems, etc.

This script uses the unified `rl_train` function from MaxText.rl.train_rl.

Usage:
  # Minimal usage (only required arguments):
  python3 src/MaxText/examples/grpo_llama3_1_8b_demo.py \\
    load_parameters_path=/path/to/scanned/model/ckpt_load_dir/ \\
    base_output_directory=/tmp/grpo_output \\
    hf_access_token=$HF_TOKEN

  # With optional overrides:
  python3 src/MaxText/examples/grpo_llama3_1_8b_demo.py \\
    load_parameters_path=/path/to/scanned/model/ckpt_load_dir/ \\
    base_output_directory=/tmp/grpo_output \\
    hf_access_token=$HF_TOKEN \\
    steps=100 \\
    num_batches=10

  Note: model_name, tokenizer_path, and chat_template_path are hardcoded for llama3.1-8b
        but can be overridden via command line if needed.

GRPO is an RL algorithm designed to enhance the reasoning abilities of LLMs. It
is a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
eliminating the need for a separate value function model. GRPO works by
generating multiple responses for a given prompt, evaluating these responses
using a reward model, and then calculating a relative advantage based on the
group's performance to update the policy.

We use Tunix as the library for GRPO.
And we use vLLM as the library for efficient model inference and generation.

In this tutorial we use a single host TPUVM such as `v6e-8/v5p-8`. Let's get started!
"""

import os
import sys
from absl import app
from typing import Sequence

import jax
import pathwaysutils

# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

# Add project root to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from MaxText import max_utils, pyconfig
from MaxText.rl.train_rl import rl_train


def main(argv: Sequence[str]) -> None:
  """Main function to run RL training using the unified rl_train function."""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  # Hardcoded configuration for Llama3.1-8B
  # These can be overridden via command line arguments
  config_file = os.path.join(script_dir, "..", "configs", "rl.yml")
  default_args = [
      config_file,
      "model_name=llama3.1-8b",
      "tokenizer_path=meta-llama/Llama-3.1-8B-Instruct",
      "hf_model_name=meta-llama/Llama-3.1-8B-Instruct",
      "chat_template_path=" + os.path.join(script_dir, "chat_templates", "gsm8k_rl.json"),
  ]
  
  # Merge command line arguments with defaults (command line takes precedence)
  # Filter out empty strings and ensure config file is first
  cli_args = [arg for arg in argv[1:] if arg]  # Skip script name
  merged_args = [config_file] + cli_args
  
  # Add defaults only if not already specified
  default_dict = {}
  for arg in default_args[1:]:  # Skip config file
    if "=" in arg:
      key, value = arg.split("=", 1)
      default_dict[key] = value
  
  # Check if any defaults are missing from CLI args
  cli_dict = {}
  for arg in cli_args:
    if "=" in arg:
      key, _ = arg.split("=", 1)
      cli_dict[key] = True
  
  # Add missing defaults
  for key, value in default_dict.items():
    if key not in cli_dict:
      merged_args.append(f"{key}={value}")

  # Initialize configuration from merged arguments
  tmvp_config = pyconfig.initialize(merged_args)
  max_utils.print_system_information()

  # Run RL training
  rl_train(tmvp_config)


if __name__ == "__main__":
  app.run(main)
