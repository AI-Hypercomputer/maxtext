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

"""lm-evaluation-harness and evalchemy runner for MaxText eval.

Supports two backends selected via --backend:

  lm_eval (default)
    Uses the /v1/completions endpoint (local-completions lm-eval backend).

  evalchemy
    Uses the /v1/chat/completions endpoint (local-chat-completions backend)
    and imports evalchemy to register its extended task registry.

Unified entry point:
  python -m maxtext.eval.runner.run --runner lm_eval [flags]
  python -m maxtext.eval.runner.run --runner evalchemy [flags]
"""

from __future__ import annotations

import argparse
import logging

from maxtext.eval.runner.common import (
    add_server_args,
    build_server_manager,
    maybe_upload_to_gcs,
    resolve_token,
)

logger = logging.getLogger(__name__)


def _map_results(raw_results: dict, tasks: list[str]) -> dict:
  """Extract per-task accuracy metrics from lm-eval / evalchemy output."""
  scores: dict[str, float] = {}
  results_section = raw_results.get("results", {})
  for task in tasks:
    task_r = results_section.get(task, {})

    acc = None
    for key in (
        "acc,none",
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc",
        "score",
    ):
      if task_r.get(key) is not None:
        acc = task_r[key]
        break

    acc_norm = None
    for key in ("acc_norm,none", "acc_norm"):
      if task_r.get(key) is not None:
        acc_norm = task_r[key]
        break

    if acc is not None:
      scores[f"{task}_accuracy"] = round(float(acc) * 100, 2)
    if acc_norm is not None:
      scores[f"{task}_accuracy_norm"] = round(float(acc_norm) * 100, 2)

    if acc is None and task_r:
      logger.warning(
          "No known accuracy keys found for task '%s'. Available: %s",
          task,
          list(task_r.keys()),
      )

  return scores


def run_harness(cfg: dict, hf_token: str | None = None) -> dict:
  """Run lm-eval or evalchemy benchmarks against a MaxText vLLM server.

  Args:
    cfg: Configuration dict. Required keys: model_name, hf_path, tasks,
      max_model_len, results_path. Optional: backend (default "lm_eval"),
      num_fewshot, num_samples, gcs_results_path, and all server keys handled
      by build_server_manager.
    hf_token: HuggingFace token for gated tokenizers.

  Returns:
    Dict with keys: results, scores, JSON file path.

  Raises:
    ImportError: If lm_eval (or evalchemy for that backend) is not installed.
  """
  # pylint: disable=import-outside-toplevel
  try:
    import lm_eval as lm_eval_lib
  except ImportError as exc:
    raise ImportError("Install lm-eval.") from exc

  from maxtext.eval.reporting.json_reporter import write_results
  from maxtext.eval.runner.warmup import warmup_server

  backend = cfg.get("backend", "lm_eval")
  if backend == "evalchemy":
    try:
      import evalchemy as _evalchemy  # pylint: disable=unused-import  # registers custom tasks with lm_eval
    except ImportError as exc:
      raise ImportError("Install evalchemy.") from exc

  model_name = cfg["model_name"]
  hf_path = cfg["hf_path"]
  tasks = cfg["tasks"]
  results_path = cfg["results_path"]
  num_fewshot = cfg.get("num_fewshot", 0)
  num_samples = cfg.get("num_samples")
  gcs_results_path = cfg.get("gcs_results_path")
  token = resolve_token(cfg, hf_token)

  lm_model_type = "local-chat-completions" if backend == "evalchemy" else "local-completions"
  raw_results: dict = {}

  with build_server_manager(cfg, token) as server:
    import jax as _jax
    from jax.experimental import multihost_utils as _multihost_utils

    is_rank0 = _jax.process_index() == 0

    if is_rank0:
      warmup_server(base_url=server.base_url, model=model_name)

      completions_path = "/v1/chat/completions" if backend == "evalchemy" else "/v1/completions"
      model_args_parts = [
          f"model={model_name}",
          f"base_url={server.base_url}{completions_path}",
          "tokenizer_backend=huggingface",
          f"tokenizer={hf_path}",
      ]
      if token:
        model_args_parts.append(f"token={token}")
      model_args = ",".join(model_args_parts)

      logger.info(
          "Running %s tasks %s via %s at %s",
          backend,
          tasks,
          lm_model_type,
          server.base_url,
      )
      raw_results = lm_eval_lib.simple_evaluate(
          model=lm_model_type,
          model_args=model_args,
          tasks=tasks,
          num_fewshot=num_fewshot,
          limit=num_samples,
          log_samples=False,
      )

    # All ranks block here until rank-0 finishes evaluation. Non-rank-0 hosts
    # keep their in-process LLM alive so rank-0's llm.generate() calls can
    # complete their tensor-parallel collectives across all hosts.
    _multihost_utils.sync_global_devices(f"harness_{backend}_complete")

  # All ranks exit the context manager together (LLM stopped on all).
  # Only rank-0 has raw_results defined; non-rank-0 return early.
  if not is_rank0:
    return {}

  scores = _map_results(raw_results, tasks)
  logger.info("%s scores: %s", backend, scores)

  output = write_results(
      benchmark="+".join(tasks),
      model_name=model_name,
      scores=scores,
      generation_stats={
          f"{backend}_config": raw_results.get("config", {}),
          f"{backend}_results": raw_results.get("results", {}),
      },
      config=cfg,
      results_path=results_path,
  )
  maybe_upload_to_gcs(output, gcs_results_path)
  return output


def _build_arg_parser() -> argparse.ArgumentParser:
  """Build argument parser."""
  parser = argparse.ArgumentParser(
      description="MaxText lm-eval / evalchemy runner.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  add_server_args(parser)
  parser.add_argument(
      "--backend",
      choices=["lm_eval", "evalchemy"],
      default="lm_eval",
      help=(
          "Evaluation backend. 'lm_eval' uses /v1/completions (local-completions); "
          "'evalchemy' uses /v1/chat/completions (local-chat-completions) and "
          "registers evalchemy's extended task library."
      ),
  )
  parser.add_argument(
      "--tasks",
      nargs="+",
      default=["mmlu"],
      help=(
          "lm-eval task names passed directly to simple_evaluate. "
          "Any task registered in lm-eval or evalchemy is accepted (e.g. gsm8k, mmlu, gpqa_diamond, ifeval, math_500)."
      ),
  )
  parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot examples per task.")
  parser.add_argument("--num_samples", type=int, help="Limit samples per task (None = full dataset).")
  return parser


def main() -> None:
  parser = _build_arg_parser()
  args = parser.parse_args()

  logging.basicConfig(
      level=getattr(logging, args.log_level),
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  results_path = f"{args.base_output_directory.rstrip('/')}/{args.run_name}/eval_results"
  cfg = {k: v for k, v in vars(args).items() if k not in ("log_level", "hf_token")}
  cfg["results_path"] = results_path

  run_harness(cfg, hf_token=args.hf_token)


if __name__ == "__main__":
  main()
