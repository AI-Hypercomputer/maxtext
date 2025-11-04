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
  python3 src/MaxText/examples/grpo_llama3_1_8b_demo.py \\
    --model_name=llama3.1-8b \\
    --tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \\
    --load_parameters_path=/path/to/scanned/model/ckpt_load_dir/ \\
    --base_output_directory=/tmp/grpo_output \\
    --hf_access_token=$HF_TOKEN \\
    --steps=100 \\
    --config_path=src/MaxText/configs/rl.yml \\
    --chat_template_path=src/MaxText/examples/chat_templates/gsm8k_rl.json

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

  # Initialize configuration from command line arguments and config file
  tmvp_config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  # Run RL training
  rl_train(tmvp_config)


if __name__ == "__main__":
  app.run(main)
