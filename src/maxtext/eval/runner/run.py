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

"""Unified CLI entry point for MaxText model evaluation.

Dispatches to the appropriate runner based on --runner:

  eval       Custom dataset runner. Requires --config.
  lm_eval    lm-evaluation-harness runner.
  evalchemy  evalchemy runner.

Both lm_eval and evalchemy dispatch to harness_runner.py.

Usage::

  # lm-eval
  python -m maxtext.eval.runner.run \
      --runner lm_eval \
      --checkpoint_path gs://<bucket>/checkpoints/0/items \
      --model_name llama3.1-8b \
      --hf_path meta-llama/Llama-3.1-8B-Instruct \
      --tasks mmlu gpqa \
      --base_output_directory gs://<bucket>/ \
      --run_name my_run \
      --max_model_len 8192 \
      --tensor_parallel_size 4 \
      --hf_token $HF_TOKEN

  # evalchemy
  python -m maxtext.eval.runner.run \
      --runner evalchemy \
      --checkpoint_path gs://<bucket>/checkpoints/0/items \
      --model_name qwen3-30b-a3b \
      --hf_path Qwen/Qwen3-30B-A3B \
      --tasks ifeval math500 \
      --base_output_directory gs://<bucket>/ \
      --run_name my_run \
      --max_model_len 8192 \
      --tensor_parallel_size 8 \
      --enable_expert_parallel \
      --hf_token $HF_TOKEN

  # custom eval_runner
  python -m maxtext.eval.runner.run \
      --runner eval \
      --config src/maxtext/eval/configs/mlperf.yml \
      --checkpoint_path gs://<bucket>/checkpoints/0/items \
      --model_name llama3.1-8b \
      --hf_path meta-llama/Llama-3.1-8B-Instruct \
      --base_output_directory gs://<bucket>/ \
      --run_name my_run \
      --hf_token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
  pre_parser = argparse.ArgumentParser(
      description="MaxText eval unified entry point.",
      add_help=False,
  )
  pre_parser.add_argument(
      "--runner",
      required=True,
      choices=["eval", "lm_eval", "evalchemy"],
      help="Which evaluation runner to use.",
  )
  pre_args, remaining = pre_parser.parse_known_args()

  sys.argv = [sys.argv[0]] + remaining

  if pre_args.runner == "eval":
    from maxtext.eval.runner.eval_runner import main as _main  # pylint: disable=import-outside-toplevel

    _main()
  elif pre_args.runner == "lm_eval":
    from maxtext.eval.runner.harness_runner import main as _main  # pylint: disable=import-outside-toplevel

    _main()
  else:  # evalchemy
    if "--backend" not in remaining:
      sys.argv += ["--backend", "evalchemy"]
    from maxtext.eval.runner.harness_runner import main as _main  # pylint: disable=import-outside-toplevel

    _main()


if __name__ == "__main__":
  main()
