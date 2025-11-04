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
  # Note: Arguments can use --key=value OR key=value format (both work)
  python3 src/MaxText/examples/grpo_llama3_1_8b_demo.py \\
    load_parameters_path=gs://bucket/path/to/checkpoint \\
    base_output_directory=/tmp/grpo_output \\
    hf_access_token=$HF_TOKEN

  # With optional overrides:
  python3 src/MaxText/examples/grpo_llama3_1_8b_demo.py \\
    load_parameters_path=gs://bucket/path/to/checkpoint \\
    base_output_directory=/tmp/grpo_output \\
    hf_access_token=$HF_TOKEN \\
    steps=100 \\
    num_batches=10

  # Alternative format with -- prefix (also supported):
  python3 src/MaxText/examples/grpo_llama3_1_8b_demo.py \\
    --load_parameters_path=gs://bucket/path/to/checkpoint \\
    --base_output_directory=/tmp/grpo_output \\
    --hf_access_token=$HF_TOKEN

  Note: model_name, tokenizer_path, and chat_template_path are hardcoded for llama3.1-8b
        but can be overridden via command line if needed (e.g., model_name=llama3.1-70b).

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
  # These match the requirements in src/MaxText/configs/rl.yml
  config_file = os.path.join(script_dir, "..", "configs", "rl.yml")
  chat_template_path = os.path.join(script_dir, "chat_templates", "gsm8k_rl.json")
  
  # Required hardcoded values for llama3.1-8b as specified in rl.yml
  hardcoded_config = {
      "model_name": "llama3.1-8b",  # Required: loads model dimensions from configs/models/llama3.1-8b.yml
      "tokenizer_path": "meta-llama/Llama-3.1-8B-Instruct",  # Required: HF tokenizer path
      "hf_model_name": "meta-llama/Llama-3.1-8B-Instruct",  # Required: for vLLM rollout (see rl.yml line 121)
      "chat_template_path": chat_template_path,  # Required: for GSM8K template (used in train_rl.py line 170)
  }
  
  # Process command line arguments: strip -- prefix and convert to key=value format
  # argv[0] is script name, argv[1:] are the arguments
  # The shell already splits arguments, so paths with spaces should be quoted by user
  cli_args = []
  cli_dict = {}
  
  print(f"[Debug] Received {len(argv)} arguments: {argv[:min(5, len(argv))]}")
  
  for arg in argv[1:]:
    if not arg:
      continue
    
    # Strip -- prefix if present (convert --key=value to key=value)
    if arg.startswith("--"):
      arg_without_dash = arg[2:]
      # Ensure it has '=' for key=value format
      if "=" not in arg_without_dash:
        raise ValueError(
            f"Invalid argument format: '{arg}'. Expected --key=value format.\n"
            f"Example: --load_parameters_path=gs://bucket/path"
        )
      cli_args.append(arg_without_dash)
      # Split on first '=' only to handle paths with '=' characters
      key, value = arg_without_dash.split("=", 1)
      cli_dict[key] = True
      print(f"[Debug] Parsed: {key}={value[:50]}..." if len(value) > 50 else f"[Debug] Parsed: {key}={value}")
    elif "=" in arg:
      # Already in key=value format without --
      cli_args.append(arg)
      key, value = arg.split("=", 1)
      cli_dict[key] = True
      print(f"[Debug] Parsed: {key}={value[:50]}..." if len(value) > 50 else f"[Debug] Parsed: {key}={value}")
    else:
      # Not a key=value argument - might be a positional argument
      print(f"[Warning] Ignoring non-key=value argument: '{arg}'")
  
  # Build merged arguments: config file first, then CLI args, then hardcoded defaults
  merged_args = [config_file] + cli_args
  
  # Add hardcoded defaults only if not specified in CLI
  for key, value in hardcoded_config.items():
    if key not in cli_dict:
      merged_args.append(f"{key}={value}")
      print(f"[Config] Using hardcoded {key}={value}")
    else:
      print(f"[Config] Using CLI override {key} (from command line)")

  print(f"[Config] Final arguments count: {len(merged_args)}")
  print(f"[Config] Config file: {config_file}")
  
  # Verify the format is correct for pyconfig.initialize
  # Expected: ["config_file", "key1=value1", "key2=value2", ...]
  assert merged_args[0] == config_file, "First argument must be config file"
  for arg in merged_args[1:]:
    assert "=" in arg, f"All config arguments must be in key=value format, got: {arg}"
  
  # Show the complete config that will be sent to pyconfig.initialize
  print("\n" + "="*80)
  print("COMPLETE CONFIG BEING PASSED TO pyconfig.initialize():")
  print("="*80)
  print(f"Config file: {merged_args[0]}")
  print("\nConfig parameters:")
  config_dict = {}
  for arg in merged_args[1:]:
    if "=" in arg:
      key, value = arg.split("=", 1)
      config_dict[key] = value
      # Truncate long values for display
      display_value = value if len(value) <= 100 else value[:100] + "..."
      print(f"  {key}={display_value}")
  print("="*80 + "\n")
  
  # Verify chat template file exists
  if not os.path.exists(chat_template_path):
    raise FileNotFoundError(
        f"Chat template not found: {chat_template_path}\n"
        f"Expected at: {chat_template_path}"
    )

  # Initialize configuration from merged arguments
  print("[Config] Initializing configuration with pyconfig.initialize()...")
  tmvp_config = pyconfig.initialize(merged_args)
  
  # Verify critical config values are set correctly
  assert tmvp_config.model_name == "llama3.1-8b", f"model_name mismatch: {tmvp_config.model_name}"
  assert tmvp_config.hf_model_name == "meta-llama/Llama-3.1-8B-Instruct", f"hf_model_name mismatch: {tmvp_config.hf_model_name}"
  assert tmvp_config.tokenizer_path == "meta-llama/Llama-3.1-8B-Instruct", f"tokenizer_path mismatch: {tmvp_config.tokenizer_path}"
  assert os.path.exists(tmvp_config.chat_template_path), f"chat_template_path not found: {tmvp_config.chat_template_path}"
  
  print(f"[Config] Verified: model_name={tmvp_config.model_name}")
  print(f"[Config] Verified: hf_model_name={tmvp_config.hf_model_name}")
  print(f"[Config] Verified: tokenizer_path={tmvp_config.tokenizer_path}")
  print(f"[Config] Verified: chat_template_path={tmvp_config.chat_template_path}")
  
  max_utils.print_system_information()

  # Run RL training
  rl_train(tmvp_config)


if __name__ == "__main__":
  # Use sys.argv directly to avoid absl flag parsing issues
  # MaxText uses key=value format which absl doesn't recognize
  main(sys.argv)
