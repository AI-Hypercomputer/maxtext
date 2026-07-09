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

"""OpenAI simple-evals runner for MaxText eval (Phase 1: grader-free evals).

Runs OpenAI's simple-evals benchmarks against a MaxText vLLM server using the
server's OpenAI-compatible /v1/chat/completions endpoint.

Phase 1 supports only the grader-free evals (mmlu, gpqa, gsm8k, drop, mgsm,
aime2024, aime2025). Grader-dependent evals (math, simpleqa, browsecomp,
healthbench) require an LLM grader endpoint and are not yet supported.

Unified entry point:
  python -m maxtext.eval.runner.run --runner simple_evals [flags]
"""

from __future__ import annotations

import argparse
import logging
import random
import threading
import time
from typing import Any

# Matches OpenAI simple-evals' default system message (OPENAI_SYSTEM_MESSAGE_API)
# so scores stay comparable to their published baselines.
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
_MAX_RETRIES = 6

from maxtext.eval.runner.common import (
    add_server_args,
    build_server_manager,
    maybe_upload_to_gcs,
    resolve_token,
)

logger = logging.getLogger(__name__)

# Grader-free evals only (Phase 1). Adding a grader-dependent eval here without
# wiring a grader endpoint produces silently wrong scores, so keep this list in
# sync with what the runner can actually evaluate.
SUPPORTED_TASKS = ("mmlu", "gpqa", "gsm8k", "drop", "mgsm", "aime2024", "aime2025")


def _describe_api_error(exc: Exception | None) -> str:
  """Render an OpenAI SDK exception with the detail its default str() hides.

  APITimeoutError/APIConnectionError never got a response back, so the only
  useful signal is the request URL and the wrapped httpx exception in
  __cause__ (e.g. ReadTimeout vs ConnectTimeout -- "server accepted the
  connection but never answered" vs "couldn't even connect" are very
  different failures). APIStatusError (4xx/5xx) does carry a response, so
  surface its status code and body instead.
  """
  if exc is None:
    return "n/a"
  parts = [f"{type(exc).__name__}: {exc}"]
  request = getattr(exc, "request", None)
  if request is not None:
    parts.append(f"request={getattr(request, 'method', '?')} {getattr(request, 'url', '?')}")
  response = getattr(exc, "response", None)
  if response is not None:
    body = getattr(exc, "body", None)
    text = str(body) if body is not None else getattr(response, "text", "")
    parts.append(f"status={getattr(response, 'status_code', '?')} body={text[:500]}")
  if exc.__cause__ is not None:
    parts.append(f"cause={type(exc.__cause__).__name__}: {exc.__cause__}")
  return " | ".join(parts)


def _build_vllm_sampler(
    base_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
    reasoning_effort: str | None = None,
):
  """Build a simple-evals sampler backed by a vLLM OpenAI-compatible server.

  Mirrors upstream ChatCompletionSampler behavior: prepends a system message
  and retries on transient errors so a single flaky request does not crash the
  whole eval (simple-evals runs requests in a ThreadPool that propagates
  exceptions).
  """
  # pylint: disable=import-outside-toplevel
  try:
    import openai
    from openai import OpenAI
  except ImportError as exc:
    raise ImportError("Install openai (pip install openai).") from exc

  from maxtext.eval.third_party.simple_evals.types import SamplerBase, SamplerResponse

  class _VllmChatSampler(SamplerBase):
    """Sample from a vLLM server via the OpenAI chat completions API."""

    def __init__(self) -> None:
      # vLLM ignores the API key but the OpenAI client requires a non-empty one.
      # max_retries=0: we already retry with our own backoff below. Leaving the
      # SDK's default (2) stacks a *second*, uncoordinated retry loop underneath
      # ours -- each of our trials silently becomes up to 3 real HTTP attempts,
      # which is pure amplification against an already-overloaded server.
      self.client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY", max_retries=0)
      self.model = model_name
      self.max_tokens = max_tokens
      self.temperature = temperature
      self.system_message = system_message
      self.reasoning_effort = reasoning_effort

    def _pack_message(self, role: str, content: Any) -> dict:
      return {"role": str(role), "content": content}

    def __call__(self, message_list):
      if self.system_message:
        message_list = [self._pack_message("system", self.system_message)] + message_list
      call_start = time.monotonic()
      last_exc: Exception | None = None
      for trial in range(_MAX_RETRIES):
        attempt_start = time.monotonic()
        try:
          extra_body = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
          response = self.client.chat.completions.create(
              model=self.model,
              messages=message_list,
              temperature=self.temperature,
              max_tokens=self.max_tokens,
              extra_body=extra_body or None,
          )
          content = response.choices[0].message.content
          if content is None:
            raise ValueError("vLLM server returned empty response; retrying.")
          return SamplerResponse(
              response_text=content,
              response_metadata={"usage": response.usage},
              actual_queried_message_list=message_list,
          )
        except openai.BadRequestError as exc:
          logger.warning("Bad request, returning empty response: %s", _describe_api_error(exc))
          return SamplerResponse(
              response_text="No response (bad request).",
              response_metadata={"usage": None},
              actual_queried_message_list=message_list,
          )
        except Exception as exc:  # pylint: disable=broad-except
          # `exc` is unbound outside this block (Python clears "as exc" targets
          # at end of scope), so stash it to reference after the loop exhausts.
          last_exc = exc
          backoff = 2**trial + random.uniform(0, 1)  # jitter avoids synchronized retry bursts
          logger.warning(
              "vLLM request failed (trial %d/%d) [%s] cause=%s attempt_elapsed=%.1fs "
              "total_elapsed=%.1fs thread=%s; retrying in %.1fs: %s",
              trial + 1,
              _MAX_RETRIES,
              type(exc).__name__,
              type(exc.__cause__).__name__ if exc.__cause__ else "n/a",
              time.monotonic() - attempt_start,
              time.monotonic() - call_start,
              threading.current_thread().name,
              backoff,
              _describe_api_error(exc),
          )
          time.sleep(backoff)
      logger.error(
          "vLLM request exhausted all %d retries after %.1fs total (thread=%s): %s",
          _MAX_RETRIES,
          time.monotonic() - call_start,
          threading.current_thread().name,
          _describe_api_error(last_exc),
          exc_info=last_exc,
      )
      raise RuntimeError(f"vLLM request failed after {_MAX_RETRIES} retries.") from last_exc

  return _VllmChatSampler()


def _build_eval(task: str, num_examples: int | None, n_repeats: int):
  """Instantiate a simple-evals Eval object for a supported task.

  Note: instantiating an eval downloads its dataset, so this is not called in
  unit tests. Task validation happens in run_simple_evals before this is called.
  """
  # pylint: disable=import-outside-toplevel
  if task == "mmlu":
    from maxtext.eval.third_party.simple_evals.mmlu_eval import MMLUEval

    return MMLUEval(num_examples=num_examples)
  if task == "gpqa":
    from maxtext.eval.third_party.simple_evals.gpqa_eval import GPQAEval

    # GPQA forbids n_repeats > 1 when subsetting examples.
    repeats = 1 if num_examples is not None else n_repeats
    return GPQAEval(num_examples=num_examples, n_repeats=repeats)
  if task == "drop":
    from maxtext.eval.third_party.simple_evals.drop_eval import DropEval

    return DropEval(num_examples=num_examples)
  if task == "mgsm":
    from maxtext.eval.third_party.simple_evals.mgsm_eval import MGSMEval

    # MGSMEval counts num_examples_per_lang *per language*, not a total, and
    # defaults to all 11 languages (2750 examples). Pin to English so
    # --num_samples means the same thing here as for every other task; 250 is
    # MGSM-en's full test set, matching upstream's own default.
    return MGSMEval(num_examples_per_lang=num_examples or 250, languages=["en"])
  if task == "gsm8k":
    from maxtext.eval.native_evals.gsm8k_eval import GSM8KEval

    return GSM8KEval(num_examples=num_examples)
  if task in ("aime2024", "aime2025"):
    from maxtext.eval.native_evals.aime_eval import AIMEEval

    # AIME forbids n_repeats > 1 when subsetting examples (mirrors GPQA).
    repeats = 1 if num_examples is not None else n_repeats
    year = 2024 if task == "aime2024" else 2025
    return AIMEEval(year=year, num_examples=num_examples, n_repeats=repeats)
  raise ValueError(f"Unsupported simple_evals task: {task}. Supported: {list(SUPPORTED_TASKS)}.")


def _map_scores(task: str, eval_result) -> dict:
  """Flatten a simple-evals EvalResult into a flat scores dict.

  The top-line score is reported as a percentage under '{task}_accuracy';
  remaining numeric metrics are passed through under '{task}_{metric}'.
  """
  scores: dict[str, float] = {}
  if eval_result.score is not None:
    scores[f"{task}_accuracy"] = round(float(eval_result.score) * 100, 2)
  for name, value in (eval_result.metrics or {}).items():
    if isinstance(value, (int, float)):
      scores[f"{task}_{name}"] = round(float(value), 4)
  return scores


def _validate_tasks(tasks: list[str]) -> None:
  """Raise ValueError if any task is unsupported in Phase 1."""
  unsupported = [t for t in tasks if t not in SUPPORTED_TASKS]
  if unsupported:
    raise ValueError(
        f"Unsupported simple_evals task(s): {unsupported}. "
        f"Phase 1 supports grader-free evals only: {list(SUPPORTED_TASKS)}."
    )


def run_simple_evals(cfg: dict, hf_token: str | None = None) -> dict:
  """Run OpenAI simple-evals benchmarks against a MaxText vLLM server.

  Args:
    cfg: Configuration dict. Required keys: model_name, hf_path, tasks,
      max_model_len, results_path. Optional: num_samples, n_repeats, max_tokens,
      temperature, reasoning_effort, gcs_results_path, and all server keys
      handled by build_server_manager.
    hf_token: HuggingFace token for gated tokenizers.

  Returns:
    Dict with keys: results, local_path. Empty dict on non-rank-0 hosts.
  """
  # pylint: disable=import-outside-toplevel
  from maxtext.eval.reporting.json_reporter import write_results
  from maxtext.eval.runner.warmup import warmup_server

  model_name = cfg["model_name"]
  tasks = cfg["tasks"]
  results_path = cfg["results_path"]
  num_samples = cfg.get("num_samples")
  n_repeats = int(cfg.get("n_repeats") or 1)
  max_tokens = int(cfg.get("max_tokens") or 2048)
  temperature = float(cfg.get("temperature") or 0.0)
  system_message = cfg.get("system_message", DEFAULT_SYSTEM_MESSAGE)
  reasoning_effort = cfg.get("reasoning_effort")
  gcs_results_path = cfg.get("gcs_results_path")
  token = resolve_token(cfg, hf_token)

  _validate_tasks(tasks)

  scores: dict[str, float] = {}
  is_rank0 = False  # set inside with block; initialized here so post-block ref is safe
  with build_server_manager(cfg, token) as server:
    import jax as _jax
    from jax.experimental import multihost_utils as _multihost_utils

    is_rank0 = _jax.process_index() == 0

    if is_rank0:
      if not cfg.get("skip_warmup"):
        warmup_server(base_url=server.base_url, model=model_name)

      sampler = _build_vllm_sampler(
          base_url=server.base_url,
          model_name=model_name,
          max_tokens=max_tokens,
          temperature=temperature,
          system_message=system_message,
          reasoning_effort=reasoning_effort,
      )
      for task in tasks:
        logger.info("Running simple_evals task '%s' at %s", task, server.base_url)
        eval_obj = _build_eval(task, num_samples, n_repeats)
        eval_result = eval_obj(sampler)
        task_scores = _map_scores(task, eval_result)
        logger.info("simple_evals task '%s' scores: %s", task, task_scores)
        scores.update(task_scores)

    # All ranks block here until rank-0 finishes evaluation. Non-rank-0 hosts
    # keep their in-process LLM alive so rank-0's llm.generate() calls can
    # complete their tensor-parallel collectives across all hosts.
    _multihost_utils.sync_global_devices("simple_evals_complete")

  # All ranks exit the context manager together (LLM stopped on all).
  # Only rank-0 has scores populated; non-rank-0 return early.
  if not is_rank0:
    return {}

  output = write_results(
      benchmark="+".join(tasks),
      model_name=model_name,
      scores=scores,
      generation_stats={"num_samples": num_samples, "n_repeats": n_repeats},
      config=cfg,
      results_path=results_path,
  )
  maybe_upload_to_gcs(output, gcs_results_path)
  return output


def _build_arg_parser() -> argparse.ArgumentParser:
  """Build argument parser."""
  parser = argparse.ArgumentParser(
      description="MaxText OpenAI simple-evals runner (Phase 1: grader-free evals).",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  add_server_args(parser)
  parser.add_argument(
      "--tasks",
      nargs="+",
      default=["mmlu"],
      help=f"simple-evals task names. Phase 1 supports grader-free evals only: {list(SUPPORTED_TASKS)}.",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      help="Limit examples per task (None = full dataset).",
  )
  parser.add_argument(
      "--n_repeats",
      type=int,
      default=1,
      help=(
          "Number of repeats per example (gpqa and aime2024/aime2025 only; "
          "forced to 1 when --num_samples is set)."
      ),
  )
  parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation.")
  parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
  parser.add_argument(
      "--system_message",
      type=str,
      default=DEFAULT_SYSTEM_MESSAGE,
      help="System message prepended to each prompt (matches OpenAI simple-evals default).",
  )
  parser.add_argument(
      "--reasoning_effort",
      type=str,
      choices=["low", "medium", "high"],
      default=None,
      help="Reasoning effort level passed as extra_body to /v1/chat/completions (reasoning models only).",
  )
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

  run_simple_evals(cfg, hf_token=args.hf_token)


if __name__ == "__main__":
  main()
