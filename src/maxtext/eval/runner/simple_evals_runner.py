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
mgsm_en, aime2024, aime2025). Grader-dependent evals (math, simpleqa, browsecomp,
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
_MAX_ATTEMPTS = 3
_REQUEST_TIMEOUT_S = 600.0

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
SUPPORTED_TASKS = ("mmlu", "gpqa", "gsm8k", "drop", "mgsm", "mgsm_en", "aime2024", "aime2025")


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
    debug_collector=None,
    concurrency: int = 1,
    continue_on_request_error: bool = False,
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
      self.client = OpenAI(
          base_url=f"{base_url}/v1",
          api_key="EMPTY",
          max_retries=0,
          timeout=_REQUEST_TIMEOUT_S,
      )
      self.model = model_name
      self.max_tokens = max_tokens
      self.temperature = temperature
      self.system_message = system_message
      self.reasoning_effort = reasoning_effort
      self.request_semaphore = threading.BoundedSemaphore(concurrency)
      self.continue_on_request_error = continue_on_request_error

    def _pack_message(self, role: str, content: Any) -> dict:
      return {"role": str(role), "content": content}

    def __call__(self, message_list):
      with self.request_semaphore:
        return self._sample(message_list)

    def _sample(self, message_list):
      if self.system_message:
        message_list = [self._pack_message("system", self.system_message)] + message_list
      debug_record = debug_collector.begin(message_list) if debug_collector else None
      call_start = time.monotonic()
      last_exc: Exception | None = None
      attempts_made = 0
      for trial in range(_MAX_ATTEMPTS):
        attempts_made = trial + 1
        attempt_start = time.monotonic()
        if debug_record is not None:
          debug_record["attempt_count"] += 1
        try:
          extra_body = {
              # Evaluators grade only message.content. Reasoning and raw output
              # are opt-in diagnostic data and are never copied into it.
              "include_reasoning": debug_record is not None,
              "include_raw_output": debug_record is not None,
          }
          if self.reasoning_effort:
            extra_body["reasoning_effort"] = self.reasoning_effort
          response = self.client.chat.completions.create(
              model=self.model,
              messages=message_list,
              temperature=self.temperature,
              max_tokens=self.max_tokens,
              extra_body=extra_body,
          )
          choice = response.choices[0]
          message = choice.message
          # An analysis-only GPT-OSS truncation is a valid model result:
          # Harmony returns reasoning plus final_content=None with
          # finish_reason="length". Preserve that as an empty scored answer
          # instead of misclassifying it as an inference failure and retrying.
          content = message.content or ""
          usage = response.usage
          response_metadata = {"usage": usage}
          if debug_record is not None:
            response_extra = getattr(response, "model_extra", None) or {}
            diagnostics = response_extra.get("diagnostics") or getattr(response, "diagnostics", None) or {}
            debug_record.update(
                status="success",
                response_text=content,
                reasoning_text=getattr(message, "reasoning", None),
                raw_response_text=diagnostics.get("raw_output"),
                successful_attempt_latency_s=time.monotonic() - attempt_start,
                finish_reason=getattr(choice, "finish_reason", None),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
            debug_collector.finish(debug_record)
            response_metadata.update(debug_record)
          return SamplerResponse(
              response_text=content,
              response_metadata=response_metadata,
              actual_queried_message_list=message_list,
          )
        except openai.BadRequestError as exc:
          logger.warning("vLLM rejected an invalid request: %s", _describe_api_error(exc))
          response_metadata = {"usage": None, "status": "bad_request"}
          if debug_record is not None:
            debug_record.update(
                status="bad_request",
                response_text="No response (bad request).",
                finish_reason=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                error_type=type(exc).__name__,
                error_status=getattr(getattr(exc, "response", None), "status_code", None),
                error_detail=_describe_api_error(exc),
            )
            debug_record["attempt_errors"].append(
                {"type": type(exc).__name__, "detail": _describe_api_error(exc)}
            )
            debug_collector.finish(debug_record)
            response_metadata.update(debug_record)
          if self.continue_on_request_error:
            return SamplerResponse(
                response_text="No response (bad request).",
                response_metadata=response_metadata,
                actual_queried_message_list=message_list,
            )
          raise RuntimeError("vLLM rejected the evaluation request as invalid.") from exc
        except Exception as exc:  # pylint: disable=broad-except
          # `exc` is unbound outside this block (Python clears "as exc" targets
          # at end of scope), so stash it to reference after the loop exhausts.
          last_exc = exc
          if debug_record is not None:
            debug_record["attempt_errors"].append(
                {
                    "type": type(exc).__name__,
                    "status": getattr(getattr(exc, "response", None), "status_code", None),
                    "detail": _describe_api_error(exc),
                    "latency_s": time.monotonic() - attempt_start,
                }
            )
          status = getattr(getattr(exc, "response", None), "status_code", None)
          retryable = isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)) or (
              status in (408, 409, 429) or (status is not None and status >= 500)
          )
          if not retryable:
            logger.error("Non-retryable vLLM response failure: %s", _describe_api_error(exc))
            break
          if trial + 1 < _MAX_ATTEMPTS:
            backoff = random.uniform(0, min(30.0, float(2**trial)))
            logger.warning(
                "vLLM request failed (trial %d/%d) [%s] cause=%s attempt_elapsed=%.1fs "
                "total_elapsed=%.1fs thread=%s; retrying in %.1fs: %s",
                trial + 1,
                _MAX_ATTEMPTS,
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
          "vLLM request failed after %d attempt(s) and %.1fs total (thread=%s): %s",
          attempts_made,
          time.monotonic() - call_start,
          threading.current_thread().name,
          _describe_api_error(last_exc),
          exc_info=last_exc,
      )
      response_metadata = {"usage": None, "status": "error"}
      if debug_record is not None:
        debug_record.update(
            status="error",
            response_text="No response (request failed).",
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            error_type=type(last_exc).__name__ if last_exc else "unknown",
            error_status=getattr(getattr(last_exc, "response", None), "status_code", None),
            error_detail=_describe_api_error(last_exc),
        )
        debug_collector.finish(debug_record)
        response_metadata = debug_record
      if self.continue_on_request_error:
        return SamplerResponse(
            response_text="No response (request failed).",
            response_metadata=response_metadata,
            actual_queried_message_list=message_list,
        )
      raise RuntimeError(f"vLLM request failed after {attempts_made} attempt(s).") from last_exc

  return _VllmChatSampler()


def _effective_repeats(task: str, num_examples: int | None, n_repeats: int | None) -> int:
  """Resolve upstream-compatible repeat defaults for tasks that support them."""
  if task not in ("gpqa", "aime2024", "aime2025"):
    return 1
  if num_examples is not None:
    return 1
  if n_repeats is not None:
    return n_repeats
  return 4 if task == "gpqa" else 1


def _build_eval(task: str, num_examples: int | None, n_repeats: int | None):
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
    repeats = _effective_repeats(task, num_examples, n_repeats)
    return GPQAEval(num_examples=num_examples, n_repeats=repeats)
  if task == "drop":
    from maxtext.eval.third_party.simple_evals.drop_eval import DropEval

    return DropEval(num_examples=num_examples)
  if task == "mgsm":
    from maxtext.eval.third_party.simple_evals.mgsm_eval import MGSMEval

    # Match upstream: evaluate every language. num_samples is per language.
    return MGSMEval(num_examples_per_lang=num_examples or 250)
  if task == "mgsm_en":
    from maxtext.eval.third_party.simple_evals.mgsm_eval import MGSMEval

    return MGSMEval(num_examples_per_lang=num_examples or 250, languages=["en"])
  if task == "gsm8k":
    from maxtext.eval.native_evals.gsm8k_eval import GSM8KEval

    return GSM8KEval(num_examples=num_examples)
  if task in ("aime2024", "aime2025"):
    from maxtext.eval.native_evals.aime_eval import AIMEEval

    # AIME forbids n_repeats > 1 when subsetting examples (mirrors GPQA).
    repeats = _effective_repeats(task, num_examples, n_repeats)
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
      temperature, reasoning_effort, concurrency, log_debug_info,
      continue_on_request_error, gcs_results_path, and all server keys handled
      by build_server_manager.
    hf_token: HuggingFace token for gated tokenizers.

  Returns:
    Dict with keys: results, local_path. Empty dict on non-rank-0 hosts.
  """
  # pylint: disable=import-outside-toplevel
  from maxtext.eval.reporting.json_reporter import write_results
  from maxtext.eval.runner.warmup import warmup_chat_server

  model_name = cfg["model_name"]
  tasks = cfg["tasks"]
  results_path = cfg["results_path"]
  num_samples = cfg.get("num_samples")
  n_repeats = cfg.get("n_repeats")
  if n_repeats is not None:
    n_repeats = int(n_repeats)
  max_tokens = int(cfg.get("max_tokens") or 2048)
  temperature_value = cfg.get("temperature")
  temperature = float(0 if temperature_value is None else temperature_value)
  system_message = cfg.get("system_message", DEFAULT_SYSTEM_MESSAGE)
  reasoning_effort = cfg.get("reasoning_effort")
  log_debug_info = bool(cfg.get("log_debug_info", False))
  continue_on_request_error = bool(cfg.get("continue_on_request_error", False))
  gcs_results_path = cfg.get("gcs_results_path")
  token = resolve_token(cfg, hf_token)

  _validate_tasks(tasks)

  scores: dict[str, float] = {}
  eval_results: dict[str, Any] = {}
  task_durations: dict[str, float] = {}
  debug_collector = None
  if log_debug_info:
    from maxtext.eval.reporting.simple_evals_debug_reporter import SimpleEvalsDebugCollector

    debug_collector = SimpleEvalsDebugCollector()
  is_rank0 = False  # set inside with block; initialized here so post-block ref is safe
  with build_server_manager(cfg, token) as server:
    import jax as _jax
    from jax.experimental import multihost_utils as _multihost_utils

    is_rank0 = _jax.process_index() == 0

    if is_rank0:
      from maxtext.eval.third_party.simple_evals import common as simple_evals_common

      simple_evals_common.set_default_num_threads(server.concurrency)
      if not cfg.get("skip_warmup"):
        warmup_chat_server(
            base_url=server.base_url,
            model=model_name,
            concurrency=server.concurrency,
            max_model_len=int(cfg["max_model_len"]),
            system_message=system_message,
            reasoning_effort=reasoning_effort,
            max_tokens=min(16, max_tokens),
        )

      sampler = _build_vllm_sampler(
          base_url=server.base_url,
          model_name=model_name,
          max_tokens=max_tokens,
          temperature=temperature,
          system_message=system_message,
          reasoning_effort=reasoning_effort,
          debug_collector=debug_collector,
          concurrency=server.concurrency,
          continue_on_request_error=continue_on_request_error,
      )
      for task in tasks:
        logger.info("Running simple_evals task '%s' at %s", task, server.base_url)
        eval_obj = _build_eval(task, num_samples, n_repeats)
        if debug_collector is not None:
          debug_collector.set_task(task)
        task_start = time.monotonic()
        eval_result = eval_obj(sampler)
        task_durations[task] = time.monotonic() - task_start
        eval_results[task] = eval_result
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
      generation_stats={
          "num_samples": num_samples,
          "n_repeats": n_repeats,
          "effective_repeats": {
              task: _effective_repeats(task, num_samples, n_repeats) for task in tasks
          },
          "task_variants": {
              task: ("all_11_languages" if task == "mgsm" else "english_only" if task == "mgsm_en" else "default")
              for task in tasks
          },
          "effective_concurrency": server.concurrency,
          "protocol_version": "openai-simple-evals-652c89d-maxtext-v2",
      },
      config=cfg,
      results_path=results_path,
  )
  maybe_upload_to_gcs(output, gcs_results_path)
  if debug_collector is not None:
    from maxtext.eval.reporting.simple_evals_debug_reporter import build_debug_report, write_debug_report

    report = build_debug_report(
        collector=debug_collector,
        eval_results=eval_results,
        task_durations=task_durations,
        model_name=model_name,
        max_tokens=max_tokens,
        max_model_len=int(cfg["max_model_len"]),
    )
    debug_local_path = write_debug_report(report, results_path, output["local_path"])
    output["debug_local_path"] = debug_local_path
    if gcs_results_path:
      secondary_debug_path = (
          gcs_results_path
          if gcs_results_path.endswith("/")
          else f"{gcs_results_path.rsplit('.', 1)[0]}.debug.txt"
      )
      maybe_upload_to_gcs({"local_path": debug_local_path}, secondary_debug_path)
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
      help="Limit examples per task (for mgsm/mgsm_en, this is per language; None = full dataset).",
  )
  parser.add_argument(
      "--n_repeats",
      type=int,
      default=None,
      help=(
          "Number of repeats per example (gpqa and aime2024/aime2025 only; defaults to upstream GPQA=4 "
          "and AIME=1, and is forced to 1 when --num_samples is set)."
      ),
  )
  parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens per generation.")
  parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (upstream default: 0.5).")
  parser.add_argument(
      "--concurrency",
      type=int,
      default=None,
      help=(
          "Maximum in-flight inference requests. By default, choose automatically from CPU count, "
          "accelerator count, max_num_seqs, and chat batch capacity."
      ),
  )
  parser.add_argument(
      "--log-debug-info",
      action="store_true",
      help="Write inference reliability, speed, truncation, and error-case diagnostics to a text file.",
  )
  parser.add_argument(
      "--continue-on-request-error",
      action="store_true",
      help="Score terminal inference failures as zero instead of aborting the benchmark.",
  )
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
