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

"""Evalchemy integration for MaxText eval.

Runs evalchemy benchmarks against a VllmServerManager in-process vLLM server.
Evalchemy (https://github.com/mlfoundations/evalchemy) extends lm-evaluation-
harness with additional benchmarks: MATH-500, GPQA Diamond, AIME, HumanEval,
LiveCodeBench, IFEval, AlpacaEval 2, Arena-Hard, MT-Bench, WildBench, MixEval,
ZeroEval, LiveBench, and all standard lm-eval tasks.

Usage:

  python -m maxtext.eval.runner.evalchemy_runner \\
      --checkpoint_path gs://<bucket>/run/checkpoints/0/items \\
      --model_name llama3.1-8b \\
      --hf_path meta-llama/Llama-3.1-8B-Instruct \\
      --tasks ifeval math500 gpqa_diamond \\
      --base_output_directory gs://<bucket>/ \\
      --run_name eval_run \\
      --max_model_len 8192 \\
      --tensor_parallel_size 4 \\
      --hf_token $HF_TOKEN

Requires: pip install evalchemy
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)

# Maps MaxText benchmark names to evalchemy/lm-eval task names.
_TASK_MAP: dict[str, str] = {
    "ifeval": "ifeval",
    "alpacaeval": "alpaca_eval_v2",
    "arena_hard": "arena_hard",
    "mtbench": "mt_bench",
    "wildbench": "wildbench",
    "mixeval": "mixeval",
    "zeroeval": "zeroeval",
    "math500": "math_500",
    "aime24": "aime2024",
    "aime25": "aime2025",
    "amc23": "amc2023",
    "gpqa_diamond": "gpqa_diamond",
    "humaneval": "humaneval",
    "livecodebench": "livecodebench",
    "gsm8k": "gsm8k",
}


def _build_model_args(
    base_url: str,
    tokenizer_path: str,
    model_name: str,
    hf_token: str | None,
) -> str:
  """Build the lm-eval model_args string for the local-chat-completions backend.

  Args:
    base_url: Base URL of the vLLM HTTP server.
    tokenizer_path: HF model ID or local path used for the tokenizer.
    model_name: Model name sent in the model field of API requests.
    hf_token: Optional HuggingFace token for gated tokenizers.

  Returns:
    Comma-separated model_args for lm-eval local-chat-completions.
  """
  endpoint = f"{base_url}/v1/chat/completions"
  args = [
      f"model={model_name}",
      f"base_url={endpoint}",
      "tokenizer_backend=huggingface",
      f"tokenizer={tokenizer_path}",
  ]
  if hf_token:
    args.append(f"token={hf_token}")
  return ",".join(args)


def _map_evalchemy_results(results: dict, tasks: list[str]) -> dict:
  """Extract per-task accuracy metrics from evalchemy output into a flat dict.

  Args:
    results: Raw lm-eval/evalchemy output dict.
    tasks: List of MaxText benchmark names.

  Returns:
    Flat dict mapping {task}_accuracy to a percentage value (0-100).
  """
  scores: dict[str, float] = {}
  results_section = results.get("results", {})
  for task in tasks:
    lm_eval_task = _TASK_MAP.get(task, task)
    task_results = results_section.get(lm_eval_task, {})
    acc = task_results.get("acc,none")
    if acc is None:
      acc = task_results.get("exact_match,none")
    if acc is None:
      acc = task_results.get("acc")
    if acc is None:
      acc = task_results.get("score")
    if acc is not None:
      scores[f"{task}_accuracy"] = round(float(acc) * 100, 2)
  return scores


def run_evalchemy(cfg: dict, hf_token: str | None = None) -> dict:
  """Run evalchemy benchmarks against a MaxText vLLM server.

  Starts an in-process vLLM server via VllmServerManager, warms it up,
  then calls lm_eval.simple_evaluate with evalchemy's task registry.

  Args:
    cfg: Configuration dict:
      - checkpoint_path: MaxText orbax checkpoint path (None for HF mode).
      - model_name: MaxText / vLLM model name.
      - hf_path: HF model ID used for the tokenizer.
      - tasks: List of MaxText benchmark names.
      - max_model_len: vLLM maximum context length.
      - results_path: Local directory for JSON result files.
      - tensor_parallel_size: Default 4.
      - max_num_batched_tokens: vLLM tokens-per-step (optional).
      - max_num_seqs: vLLM max concurrent sequences (optional).
      - server_host: Hostname for the HTTP server (default "localhost").
      - server_port: Port for the HTTP server (default 8000).
      - num_fewshot: Few-shot examples per task (default 0).
      - num_samples: Limit samples per task (None = full dataset).
      - gcs_results_path: Optional GCS path to upload results JSON.
      - hf_mode: When True, loads HF safetensors instead of MaxText checkpoint.
    hf_token: HuggingFace token for gated tokenizers.

  Returns:
    Results (lm-eval output), scores (flat metric dict),
    local_path (written JSON file path).

  Raises:
    ImportError: If evalchemy or lm_eval is not installed.
    ValueError: If a requested task name is not in _TASK_MAP.
  """
  # pylint: disable=import-outside-toplevel
  try:
    import evalchemy as _evalchemy  # noqa: F401 registers custom tasks with lm_eval
    import lm_eval
  except ImportError as e:
    raise ImportError(
        "Ensure evalchemy and lm_eval are installed."
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
  num_samples = cfg.get("num_samples")
  gcs_results_path = cfg.get("gcs_results_path")

  lm_eval_tasks: list[str] = []
  for t in tasks:
    lm_eval_task = _TASK_MAP.get(t)
    if lm_eval_task is None:
      raise ValueError(
          f"No evalchemy task mapping for benchmark '{t}'. "
          f"Known tasks: {list(_TASK_MAP.keys())}"
      )
    lm_eval_tasks.append(lm_eval_task)

  server_env = {"HF_TOKEN": token} if token else None

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
  ) as server:
    warmup_server(base_url=server.base_url, model=model_name)

    model_args = _build_model_args(
        base_url=server.base_url,
        tokenizer_path=hf_path,
        model_name=model_name,
        hf_token=token,
    )
    logger.info(
        "Running evalchemy tasks %s via local-chat-completions at %s",
        lm_eval_tasks,
        server.base_url,
    )
    evalchemy_results = lm_eval.simple_evaluate(
        model="local-chat-completions",
        model_args=model_args,
        tasks=lm_eval_tasks,
        num_fewshot=num_fewshot,
        limit=num_samples,
        log_samples=False,
    )

  scores = _map_evalchemy_results(evalchemy_results, tasks)
  logger.info("evalchemy scores: %s", scores)

  output = write_results(
      benchmark="+".join(tasks),
      model_name=model_name,
      scores=scores,
      generation_stats={"evalchemy_config": evalchemy_results.get("config", {})},
      config=cfg,
      results_path=results_path,
  )

  if gcs_results_path:
    from maxtext.eval.reporting.gcs_reporter import upload_results  # pylint: disable=import-outside-toplevel
    upload_results(output["local_path"], gcs_results_path)

  return output


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="MaxText evalchemy runner.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--checkpoint_path",
      help="MaxText orbax checkpoint path.",
  )
  parser.add_argument(
      "--model_name",
      required=True,
      help="MaxText model name (e.g. llama3.1-8b).",
  )
  parser.add_argument(
      "--hf_path",
      required=True,
      help="HF model ID for tokenizer.",
  )
  parser.add_argument(
      "--tasks",
      nargs="+",
      default=["ifeval"],
      help="Benchmark names to evaluate. Choices: " + ", ".join(_TASK_MAP.keys()),
  )
  parser.add_argument(
      "--base_output_directory",
      required=True,
      help="Base output directory.",
  )
  parser.add_argument(
      "--run_name",
      required=True,
      help="Run name/identifier.",
  )
  parser.add_argument(
      "--max_model_len",
      type=int,
      required=True,
      help="vLLM max context length.",
  )
  parser.add_argument(
      "--tensor_parallel_size",
      type=int,
      default=4,
      help="vLLM tensor parallelism.",
  )
  parser.add_argument("--server_host", default="localhost", help="vLLM server bind host.")
  parser.add_argument("--server_port", type=int, default=8000, help="vLLM server port.")
  parser.add_argument(
      "--max_num_batched_tokens",
      type=int,
      help="vLLM tokens per scheduler step.",
  )
  parser.add_argument(
      "--max_num_seqs",
      type=int,
      help="vLLM max concurrent sequences.",
  )
  parser.add_argument(
      "--num_fewshot",
      type=int,
      default=0,
      help="Few-shot examples per task.",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      help="Limit samples per task (None = full dataset).",
  )
  parser.add_argument(
      "--hf_mode",
      action="store_true",
      help="HF safetensors mode.",
  )
  parser.add_argument(
      "--hf_token",
      help="HuggingFace token for gated tokenizers.",
  )
  parser.add_argument(
      "--gcs_results_path",
      help="Optional GCS path to upload results.",
  )
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
  )
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

  run_evalchemy(cfg, hf_token=args.hf_token)


if __name__ == "__main__":
  main()
