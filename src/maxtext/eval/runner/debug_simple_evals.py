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

"""One-shot TPU smoke test for the simple-evals chat path.

This launches the same in-process vLLM server used by simple-evals and checks
prompt rendering, Harmony output separation, truncation handling, diagnostic
isolation, and small-batch response ordering. It performs no benchmark grading.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import logging
import os
import time
from typing import Any

import requests

from maxtext.eval.runner.common import add_server_args, build_server_manager, resolve_token

logger = logging.getLogger(__name__)

_CONTROL_MARKERS = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>")


class DebugCheckError(RuntimeError):
  """Raised when the live server violates a prompt/output invariant."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise DebugCheckError(message)


def _post_chat(
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    reasoning_effort: str,
    include_reasoning: bool,
    include_raw_output: bool,
    timeout_s: float,
) -> dict[str, Any]:
  payload = {
      "model": model_name,
      "messages": messages,
      "temperature": 0.0,
      "max_tokens": max_tokens,
      "reasoning_effort": reasoning_effort,
      "include_reasoning": include_reasoning,
      "include_raw_output": include_raw_output,
  }
  started = time.monotonic()
  response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
  elapsed_s = time.monotonic() - started
  try:
    body = response.json()
  except ValueError as exc:
    raise DebugCheckError(
        f"Chat endpoint returned non-JSON HTTP {response.status_code}: {response.text[:1000]}"
    ) from exc
  if response.status_code != 200:
    raise DebugCheckError(f"Chat endpoint returned HTTP {response.status_code}: {body}")
  return {"elapsed_s": elapsed_s, "request": payload, "response": body}


def _choice(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  response = result["response"]
  choices = response.get("choices") or []
  _require(len(choices) == 1, f"Expected exactly one response choice, got {len(choices)}.")
  message = choices[0].get("message")
  _require(isinstance(message, dict), "Response choice has no message object.")
  return choices[0], message


def _validate_usage(result: dict[str, Any]) -> None:
  usage = result["response"].get("usage") or {}
  prompt_tokens = usage.get("prompt_tokens")
  completion_tokens = usage.get("completion_tokens")
  total_tokens = usage.get("total_tokens")
  _require(isinstance(prompt_tokens, int) and prompt_tokens > 0, f"Invalid prompt token count: {prompt_tokens!r}.")
  _require(
      isinstance(completion_tokens, int) and completion_tokens >= 0,
      f"Invalid completion token count: {completion_tokens!r}.",
  )
  _require(
      total_tokens == prompt_tokens + completion_tokens,
      f"Inconsistent token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}.",
  )


def _validate_primary_response(result: dict[str, Any], sentinel: str, expect_harmony: bool) -> None:
  choice, message = _choice(result)
  content = message.get("content")
  _require(isinstance(content, str) and sentinel in content, f"Final content did not contain {sentinel!r}: {content!r}")
  _require(
      choice.get("finish_reason") == "stop",
      f"Primary request did not stop normally: {choice.get('finish_reason')!r}.",
  )
  _require(
      not any(marker in content for marker in _CONTROL_MARKERS),
      "Harmony control markers leaked into final content.",
  )
  diagnostics = result["response"].get("diagnostics")
  _require(isinstance(diagnostics, dict), "Opt-in diagnostics object is missing.")
  _require(isinstance(diagnostics.get("raw_output"), str), "Diagnostic raw_output is missing or not text.")
  _require("raw_output" not in message, "Raw output leaked into the assistant message object.")
  if expect_harmony:
    _require("reasoning" in message, "GPT-OSS response did not expose the requested reasoning field.")
  _validate_usage(result)


def _validate_nondiagnostic_response(result: dict[str, Any], sentinel: str) -> None:
  _, message = _choice(result)
  content = message.get("content")
  _require(isinstance(content, str) and sentinel in content, f"Final content did not contain {sentinel!r}: {content!r}")
  _require("reasoning" not in message, "Reasoning was returned despite include_reasoning=false.")
  _require("diagnostics" not in result["response"], "Raw diagnostics were returned without opting in.")
  _validate_usage(result)


def _validate_truncation_response(result: dict[str, Any]) -> None:
  choice, message = _choice(result)
  _require(choice.get("finish_reason") == "length", f"One-token request was not reported as truncated: {choice}.")
  _require(message.get("content") in (None, ""), "Analysis-only truncation unexpectedly produced final content.")
  diagnostics = result["response"].get("diagnostics") or {}
  _require("raw_output" in diagnostics, "Truncated response did not preserve raw diagnostic output.")
  _validate_usage(result)


def _validate_bad_reasoning_effort(base_url: str, model_name: str, timeout_s: float) -> dict[str, Any]:
  payload = {
      "model": model_name,
      "messages": [{"role": "user", "content": "Return OK."}],
      "temperature": 0.0,
      "max_tokens": 8,
      "reasoning_effort": "none",
  }
  response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
  _require(
      response.status_code == 400,
      f"Invalid Harmony reasoning effort returned HTTP {response.status_code}, not 400.",
  )
  return {"request": payload, "status_code": response.status_code, "response": response.json()}


def _run_live_checks(server: Any, cfg: dict[str, Any], report: dict[str, Any] | None = None) -> dict[str, Any]:
  base_url = server.base_url
  model_name = cfg["model_name"]
  timeout_s = float(cfg["request_timeout_s"])
  max_tokens = int(cfg["debug_max_tokens"])
  reasoning_effort = cfg["reasoning_effort"]
  llm = getattr(server, "_llm", None)
  hf_config = getattr(getattr(llm, "model_config", None), "hf_config", None)
  model_type = getattr(hf_config, "model_type", None)
  expect_harmony = model_type == "gpt_oss"

  if report is None:
    report = {}
  report.update(
      {
          "model": model_name,
          "hf_path": cfg["hf_path"],
          "model_type": model_type,
          "expect_harmony": expect_harmony,
          "resolved_concurrency": server.concurrency,
          "checks": {},
          "warnings": [],
      }
  )

  primary_sentinel = "FINAL_SMOKE_OK"
  primary = _post_chat(
      base_url,
      model_name,
      [
          {"role": "system", "content": "Follow the user's requested final-answer format exactly."},
          {
              "role": "user",
              "content": (
                  "Privately reason through 17 multiplied by 23. Ignore the numeric result in the final response; "
                  f"the final response must be exactly {primary_sentinel}."
              ),
          },
      ],
      max_tokens=max_tokens,
      reasoning_effort=reasoning_effort,
      include_reasoning=True,
      include_raw_output=True,
      timeout_s=timeout_s,
  )
  _validate_primary_response(primary, primary_sentinel, expect_harmony)
  report["checks"]["harmony_final_reasoning_raw"] = primary

  nondiagnostic_sentinel = "NO_DIAGNOSTICS_OK"
  nondiagnostic = _post_chat(
      base_url,
      model_name,
      [{"role": "user", "content": f"Respond exactly with {nondiagnostic_sentinel}."}],
      max_tokens=max_tokens,
      reasoning_effort="low",
      include_reasoning=False,
      include_raw_output=False,
      timeout_s=timeout_s,
  )
  _validate_nondiagnostic_response(nondiagnostic, nondiagnostic_sentinel)
  report["checks"]["diagnostic_opt_in"] = nondiagnostic

  if expect_harmony:
    truncated = _post_chat(
        base_url,
        model_name,
        [{"role": "user", "content": "Carefully derive the first 100 prime numbers."}],
        max_tokens=1,
        reasoning_effort="high",
        include_reasoning=True,
        include_raw_output=True,
        timeout_s=timeout_s,
    )
    _validate_truncation_response(truncated)
    report["checks"]["analysis_only_truncation"] = truncated
    report["checks"]["invalid_reasoning_effort"] = _validate_bad_reasoning_effort(
        base_url, model_name, timeout_s
    )
  else:
    report["warnings"].append("Model is not hf_config.model_type='gpt_oss'; Harmony-only checks were skipped.")

  batch_size = min(2, int(cfg["batch_check_requests"]), server.concurrency)
  if batch_size >= 2:
    sentinels = [f"BATCH_ORDER_OK_{index}" for index in range(batch_size)]

    def send(sentinel: str) -> dict[str, Any]:
      return _post_chat(
          base_url,
          model_name,
          [{"role": "user", "content": f"Respond exactly with {sentinel}."}],
          max_tokens=max_tokens,
          reasoning_effort="low",
          include_reasoning=False,
          include_raw_output=False,
          timeout_s=timeout_s,
      )

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
      batch_results = list(pool.map(send, sentinels))
    for sentinel, result in zip(sentinels, batch_results):
      _validate_nondiagnostic_response(result, sentinel)
    report["checks"]["batch_response_order"] = batch_results
  else:
    report["warnings"].append("Resolved/requested concurrency is below 2; batch-order check was skipped.")

  return report


def _write_report(report: dict[str, Any], output_path: str) -> str:
  absolute_path = os.path.abspath(output_path)
  os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
  with open(absolute_path, "w", encoding="utf-8") as output_file:
    json.dump(report, output_file, indent=2, sort_keys=True, default=str)
    output_file.write("\n")
  return absolute_path


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="One-shot TPU debug check for the simple-evals chat path.")
  add_server_args(parser)
  parser.add_argument(
      "--debug_max_tokens",
      type=int,
      default=1024,
      help="Generation budget for sentinel checks. Use at least 128; default: 1024.",
  )
  parser.add_argument(
      "--reasoning_effort",
      choices=["low", "medium", "high"],
      default="high",
      help="Reasoning effort for the primary Harmony check.",
  )
  parser.add_argument("--request_timeout_s", type=float, default=600.0, help="Per-request timeout in seconds.")
  parser.add_argument(
      "--batch_check_requests",
      type=int,
      default=2,
      help="Concurrent sentinel requests for the batch-order check; capped at 2.",
  )
  parser.add_argument(
      "--debug_output",
      help="Local JSON report path. Defaults to ./simple_evals_tpu_debug_<UTC timestamp>.json.",
  )
  parser.add_argument(
      "--concurrency",
      type=int,
      default=None,
      help="Maximum in-flight requests. Omit to use the normal automatic TPU/CPU-based selection.",
  )
  return parser


def main() -> None:
  parser = _build_arg_parser()
  args = parser.parse_args()
  if args.debug_max_tokens < 128:
    parser.error("--debug_max_tokens must be at least 128 so the final sentinel is unlikely to be truncated.")
  if args.request_timeout_s <= 0 or args.batch_check_requests <= 0:
    parser.error("--request_timeout_s and --batch_check_requests must be positive.")

  logging.basicConfig(
      level=getattr(logging, args.log_level),
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  cfg = vars(args).copy()
  token = resolve_token(cfg, args.hf_token)
  timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  output_path = args.debug_output or f"simple_evals_tpu_debug_{timestamp}.json"
  report: dict[str, Any] = {
      "status": "FAIL",
      "started_at_utc": timestamp,
  }
  error: BaseException | None = None
  is_rank0 = True
  try:
    with build_server_manager(cfg, token) as server:
      import jax as _jax  # pylint: disable=import-outside-toplevel
      from jax.experimental import multihost_utils as _multihost_utils  # pylint: disable=import-outside-toplevel

      is_rank0 = _jax.process_index() == 0
      try:
        if is_rank0:
          _run_live_checks(server, cfg, report)
          report["status"] = "PASS"
      finally:
        # Keep every host's in-process LLM alive while rank 0 drives the HTTP
        # checks, then let all hosts tear down together even if a check fails.
        _multihost_utils.sync_global_devices("simple_evals_debug_complete")
  except BaseException as exc:  # noqa: B902  pylint: disable=broad-except
    error = exc
    report["error_type"] = type(exc).__name__
    report["error"] = str(exc)
    logger.exception("TPU simple-evals debug check failed: %s", exc)
  finally:
    if is_rank0:
      report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
      written_path = _write_report(report, output_path)

  if not is_rank0:
    return

  print(f"simple-evals TPU debug: {report['status']}")
  print(f"report: {written_path}")
  for warning in report.get("warnings", []):
    print(f"warning: {warning}")
  if error is not None:
    raise SystemExit(1) from error


if __name__ == "__main__":
  main()
