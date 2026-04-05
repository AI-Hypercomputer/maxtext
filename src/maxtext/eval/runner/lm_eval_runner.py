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

"""lm-evaluation-harness integration for MaxText eval.

Runs lm-evaluation-harness benchmarks against a vLLM server
started via VllmServerManager. Uses the local-completions backend.

Usage::

  python -m maxtext.eval.runner.lm_eval_runner \
      --checkpoint_path gs://<bucket>/run/checkpoints/0/items \
      --model_name llama3.1-8b \
      --hf_path meta-llama/Llama-3.1-8B-Instruct \
      --tasks mmlu gpqa \
      --base_output_directory gs://<bucket>/ \
      --run_name my_run \
      --max_model_len 8192 \
      --tensor_parallel_size 4 \
      --hf_token $HF_TOKEN

Requires: pip install "lm_eval[api]"
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

# Maps MaxText benchmark names to lm-eval task names.
_TASK_MAP: dict[str, str] = {
    "mmlu": "mmlu",           # loglikelihood, 14042 questions
    "gpqa": "gpqa_diamond",   # loglikelihood, 198 questions
    "math": "hendrycks_math", # generation, 12500 problems (5 subjects)
    "gsm8k": "gsm8k",         # generation, 8500 grade-school math problems
}


def _build_model_args(base_url: str, tokenizer_path: str, model_name: str, hf_token: str | None) -> str:
  """Build the lm-eval model_args string for local-completions backend."""
  args = [
      f"model={model_name}",
      f"base_url={base_url}/v1/completions",
      f"tokenizer_backend=huggingface",
      f"tokenizer={tokenizer_path}",
  ]
  if hf_token:
    args.append(f"token={hf_token}")
  return ",".join(args)


def _map_lm_eval_results(lm_eval_results: dict, tasks: list[str]) -> dict:
  """Extract per-task accuracy metrics from lm-eval output into a flat dict."""
  scores = {}
  results_section = lm_eval_results.get("results", {})
  for task in tasks:
    lm_task = _TASK_MAP.get(task, task)
    task_results = results_section.get(lm_task, {})
    acc = task_results.get("acc,none")
    if acc is None:
      acc = task_results.get("acc")
    acc_norm = task_results.get("acc_norm,none")
    if acc_norm is None:
      acc_norm = task_results.get("acc_norm")
    if acc is not None:
      scores[f"{task}_accuracy"] = round(acc * 100, 2)
    if acc_norm is not None:
      scores[f"{task}_accuracy_norm"] = round(acc_norm * 100, 2)
  return scores


def run_lm_eval(cfg: dict, hf_token: str | None = None) -> dict:
  """Run lm-evaluation-harness benchmarks against a MaxText vLLM server.

  Args:
    cfg: Configuration dict. Required keys: checkpoint_path (or hf_mode),
      model_name, hf_path, tasks, max_model_len, results_path.
    hf_token: HuggingFace token for gated tokenizers.

  Returns:
    Dict with keys: results (lm-eval output), scores (flat metric dict),
    local_path (written JSON file).
  """
  # pylint: disable=import-outside-toplevel
  try:
    import lm_eval as lm_eval_lib
  except ImportError as e:
    raise ImportError(
        "lm-evaluation-harness is required. Install with: pip install 'lm_eval[api]'"
    ) from e

  from maxtext.eval.reporting.json_reporter import write_results
  from maxtext.eval.runner.server_manager import VllmServerManager
  from maxtext.eval.runner.warmup import warmup_server

  model_name = cfg["model_name"]
  hf_path = cfg["hf_path"]
  tasks = cfg["tasks"]
  results_path = cfg["results_path"]
  checkpoint_path = cfg.get("checkpoint_path")
  hf_mode = cfg.get("hf_mode", False)
  use_maxtext_adapter = bool(checkpoint_path) and not hf_mode
  token = hf_token or os.environ.get("HF_TOKEN") or None

  max_model_len = int(cfg["max_model_len"])
  tensor_parallel_size = int(cfg.get("tensor_parallel_size", 4))
  max_num_batched_tokens = cfg.get("max_num_batched_tokens")
  if max_num_batched_tokens is not None:
    max_num_batched_tokens = int(max_num_batched_tokens)
  max_num_seqs = cfg.get("max_num_seqs")
  if max_num_seqs is not None:
    max_num_seqs = int(max_num_seqs)
  server_host = cfg.get("server_host", "localhost")
  server_port = int(cfg.get("server_port", 8000))
  num_fewshot = cfg.get("num_fewshot", 0)
  num_samples = cfg.get("num_samples")  # None = use full dataset
  gcs_results_path = cfg.get("gcs_results_path")

  lm_tasks = []
  for t in tasks:
    lm_task = _TASK_MAP.get(t)
    if lm_task is None:
      raise ValueError(
          f"No lm-eval task mapping for benchmark '{t}'. "
          f"Known tasks: {list(_TASK_MAP.keys())}"
      )
    lm_tasks.append(lm_task)

  server_env = {"HF_TOKEN": token} if token else None
  additional_vllm_kwargs = {}
  if cfg.get("enable_expert_parallel"):
    additional_vllm_kwargs["enable_expert_parallel"] = True

  with VllmServerManager(
      model_path=hf_path,
      checkpoint_path=checkpoint_path if use_maxtext_adapter else None,
      maxtext_model_name=model_name if use_maxtext_adapter else None,
      host=server_host,
      port=server_port,
      tensor_parallel_size=tensor_parallel_size,
      max_model_len=max_model_len,
      max_num_batched_tokens=max_num_batched_tokens,
      max_num_seqs=max_num_seqs,
      env=server_env,
      additional_vllm_kwargs=additional_vllm_kwargs or None,
  ) as server:
    warmup_server(base_url=server.base_url, model=model_name)

    model_args = _build_model_args(
        base_url=server.base_url,
        tokenizer_path=hf_path,
        model_name=model_name,
        hf_token=token,
    )
    logger.info("Running lm-eval tasks %s via local-completions at %s", lm_tasks, server.base_url)
    lm_results = lm_eval_lib.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=lm_tasks,
        num_fewshot=num_fewshot,
        limit=num_samples,
        log_samples=False,
    )

  scores = _map_lm_eval_results(lm_results, tasks)
  logger.info("lm-eval scores: %s", scores)

  output = write_results(
      benchmark="+".join(tasks),
      model_name=model_name,
      scores=scores,
      generation_stats={"lm_eval_config": lm_results.get("config", {})},
      config=cfg,
      results_path=results_path,
  )

  if gcs_results_path:
    from maxtext.eval.reporting.gcs_reporter import upload_results  # pylint: disable=import-outside-toplevel
    upload_results(output["local_path"], gcs_results_path)

  return output


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="MaxText lm-evaluation-harness runner.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--checkpoint_path", help="MaxText orbax checkpoint path (/0/items).")
  parser.add_argument("--model_name", required=True, help="MaxText model name (e.g. llama3.1-8b).")
  parser.add_argument("--hf_path", required=True, help="HF model ID for tokenizer.")
  parser.add_argument("--tasks", nargs="+", default=["mmlu"], help="Benchmark names to evaluate.")
  parser.add_argument("--base_output_directory", required=True, help="Base output directory.")
  parser.add_argument("--run_name", required=True, help="Run name.")
  parser.add_argument("--max_model_len", type=int, required=True, help="vLLM max context length.")
  parser.add_argument("--tensor_parallel_size", type=int, default=4, help="vLLM tensor parallelism.")
  parser.add_argument("--server_host", default="localhost", help="vLLM server bind host.")
  parser.add_argument("--server_port", type=int, default=8000, help="vLLM server port.")
  parser.add_argument("--max_num_batched_tokens", type=int, help="vLLM tokens per scheduler step.")
  parser.add_argument("--max_num_seqs", type=int, help="vLLM max concurrent sequences.")
  parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot examples per task.")
  parser.add_argument("--num_samples", type=int, help="Limit samples per task (None = full dataset).")
  parser.add_argument("--hf_token", help="HuggingFace token for gated tokenizers.")
  parser.add_argument("--hf_mode", action="store_true", help="HF safetensors mode.")
  parser.add_argument(
      "--enable_expert_parallel",
      action="store_true",
      help=(
          "Enable expert parallelism in vLLM. Required for MoE models such as "
          "qwen3-30b-a3b, qwen3-235b-a22b, deepseek-v3, etc. Without this flag "
          "tpu-inference omits the 'expert' mesh axis and MaxText's MoE sharding "
          "raises KeyError."
      ),
  )
  parser.add_argument("--gcs_results_path", help="Optional GCS path to upload results.")
  parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
  return parser


def main() -> None:
  import logging as _logging  # pylint: disable=import-outside-toplevel
  parser = _build_arg_parser()
  args = parser.parse_args()

  _logging.basicConfig(
      level=getattr(_logging, args.log_level),
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  results_path = f"{args.base_output_directory.rstrip('/')}/{args.run_name}/eval_results"
  cfg = {k: v for k, v in vars(args).items() if k not in ("log_level", "hf_token")}
  cfg["results_path"] = results_path

  run_lm_eval(cfg, hf_token=args.hf_token)


if __name__ == "__main__":
  main()
