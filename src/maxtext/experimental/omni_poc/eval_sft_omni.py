# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation runner for Omni (Gemma 3 Vision + Qwen 3 LLM).

Reuses core evaluation routines from `benchmarks.multimodal.multimodal_eval`,
providing Omni-specific defaults and CLI.

Usage:
  export HF_TOKEN="<YOUR_HF_TOKEN>"

  python3 -m maxtext.experimental.omni_poc.eval_sft_omni \\
    src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \\
    load_parameters_path=gs://YOUR_BUCKET/path/to/checkpoint/items \\
    base_output_directory=gs://YOUR_BUCKET/eval_output \\
    run_name=eval_run \\
    --num_examples=100 \\
    --hf_eval_split=test

Note:
  - Defaults to full dataset evaluation (--num_examples=-1).
  - Defaults to SFT prompt formatting (--ckpt_type=sft).
"""

import maxtext
# Eagerly initialize core MaxText C++ and model dependencies
_ = (maxtext.Mesh, maxtext.pyconfig, maxtext.models, maxtext.model_creation_utils)

import argparse
import os
import sys
from typing import Sequence

import jax

from benchmarks.multimodal import multimodal_eval
from maxtext.configs import pyconfig
from maxtext.inference.inference_utils import str2bool
from maxtext.utils.globals import MAXTEXT_PKG_DIR


def main(argv: Sequence[str]) -> None:
  """CLI entry point for Omni multimodal evaluation."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  # Add default arguments to mirror multimodal_eval.py
  parser = argparse.ArgumentParser(description="Omni Multimodal Evaluation")
  parser.add_argument("--num_examples", type=int, default=-1)
  parser.add_argument("--tmp_results_file", type=str, default="omni_eval_results.csv")
  parser.add_argument("--remove_tmp_results", type=str2bool, default=True)
  parser.add_argument("--ckpt_type", type=str, default="sft", choices=["base", "sft"])
  parser.add_argument("--image_resize", type=int, default=-1)
  parser.add_argument("--hf_eval_split", "--split", type=str, default="test")

  local_args, remaining_args = parser.parse_known_args(list(argv[1:]))

  # Add default config if not provided
  if not any(a.endswith((".yml", ".yaml")) for a in remaining_args):
    remaining_args.insert(
        0,
        os.path.join(
            MAXTEXT_PKG_DIR,
            "experimental",
            "omni_poc",
            "configs",
            "sft-maxtext-omni-gemma3-qwen3.yml",
        ),
    )

  # Apply evaluation defaults and forward CLI overrides to pyconfig
  defaults = [
      ("override_model_config", "True"),
      ("per_device_batch_size", "1"),
      ("async_checkpointing", "False"),
      ("hf_eval_files", ""),
      ("hf_train_files", ""),
      ("hf_path", "HuggingFaceM4/ChartQA"),
      ("hf_eval_split", local_args.hf_eval_split or "test"),
  ]
  for key, val in defaults:
    if val is not None and not any(a.startswith(f"{key}=") for a in remaining_args):
      remaining_args.append(f"{key}={val}")

  # Initialize MaxText configuration
  cfg = pyconfig.initialize([argv[0]] + remaining_args)

  # Validate checkpoint parameter
  if not cfg.load_parameters_path:
    raise ValueError("Please specify a checkpoint path using load_parameters_path=<path>.")

  # Forward HF token to environment if provided in config
  if getattr(cfg, "hf_access_token", ""):
    os.environ["HF_TOKEN"] = cfg.hf_access_token

  multimodal_eval.validate_config(cfg)

  # Evaluation
  multimodal_eval.main(cfg, local_args)


if __name__ == "__main__":
  main(sys.argv)
