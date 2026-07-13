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

"""Human-readable diagnostics for the OpenAI simple-evals runner."""

from __future__ import annotations

import logging
import math
import os
import statistics
import threading
import time
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class SimpleEvalsDebugCollector:
  """Thread-safe collection of logical request diagnostics."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._next_id = 1
    self._active = 0
    self.peak_in_flight = 0
    self.current_task = "unknown"
    self.records: list[dict[str, Any]] = []

  def set_task(self, task: str) -> None:
    with self._lock:
      self.current_task = task

  def begin(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
    with self._lock:
      request_id = self._next_id
      self._next_id += 1
      self._active += 1
      self.peak_in_flight = max(self.peak_in_flight, self._active)
      task = self.current_task
    return {
        "request_id": request_id,
        "task": task,
        "prompt_messages": messages,
        "started_monotonic": time.monotonic(),
        "attempt_count": 0,
        "attempt_errors": [],
    }

  def finish(self, record: dict[str, Any]) -> None:
    record["latency_s"] = time.monotonic() - record.pop("started_monotonic")
    with self._lock:
      self.records.append(record)
      self._active -= 1


def _percentile(values: list[float], percentile: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  position = (len(ordered) - 1) * percentile
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _fmt_number(value: float | int | None, digits: int = 2) -> str:
  if value is None:
    return "n/a"
  if isinstance(value, int):
    return str(value)
  return f"{value:.{digits}f}"


def _rate(numerator: int, denominator: int) -> str:
  return "n/a" if denominator == 0 else f"{numerator / denominator:.2%} ({numerator}/{denominator})"


def _latency_line(records: list[dict[str, Any]]) -> str:
  values = [float(r["latency_s"]) for r in records if r.get("latency_s") is not None]
  if not values:
    return "n/a"
  return (
      f"mean={statistics.fmean(values):.3f}s p50={_percentile(values, 0.50):.3f}s "
      f"p90={_percentile(values, 0.90):.3f}s p95={_percentile(values, 0.95):.3f}s "
      f"p99={_percentile(values, 0.99):.3f}s max={max(values):.3f}s"
  )


def _token_line(records: list[dict[str, Any]], key: str) -> str:
  values = [int(r[key]) for r in records if r.get(key) is not None]
  if not values:
    return "n/a"
  return (
      f"mean={statistics.fmean(values):.1f} p50={_percentile(values, 0.50):.1f} "
      f"p95={_percentile(values, 0.95):.1f} max={max(values)}"
  )


def _attempt_latency_line(records: list[dict[str, Any]]) -> str:
  values = [
      float(error["latency_s"])
      for record in records
      for error in record.get("attempt_errors", [])
      if error.get("latency_s") is not None
  ]
  values.extend(
      float(record["successful_attempt_latency_s"])
      for record in records
      if record.get("successful_attempt_latency_s") is not None
  )
  if not values:
    return "n/a"
  return (
      f"mean={statistics.fmean(values):.3f}s p50={_percentile(values, 0.50):.3f}s "
      f"p95={_percentile(values, 0.95):.3f}s max={max(values):.3f}s"
  )


def _clip(value: Any, limit: int) -> str:
  text = str(value)
  if len(text) <= limit:
    return text
  return text[:limit] + f"\n... [report-clipped {len(text) - limit} characters]"


def _prompt_text(record: dict[str, Any]) -> str:
  return "\n\n".join(
      f"[{message.get('role', 'unknown')}] {message.get('content', '')}"
      for message in record.get("prompt_messages", [])
  )


def _example_map(eval_results: dict[str, Any]) -> dict[int, dict[str, Any]]:
  examples: dict[int, dict[str, Any]] = {}
  for result in eval_results.values():
    metadata = (result.metadata or {}).get("example_level_metadata", [])
    for item in metadata:
      if item and item.get("request_id") is not None:
        examples[int(item["request_id"])] = item
  return examples


def _wilson_interval(correct: int, total: int) -> tuple[float, float] | None:
  if total == 0:
    return None
  z = 1.96
  p = correct / total
  denominator = 1 + z * z / total
  center = (p + z * z / (2 * total)) / denominator
  margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
  return center - margin, center + margin


def build_debug_report(
    collector: SimpleEvalsDebugCollector,
    eval_results: dict[str, Any],
    task_durations: dict[str, float],
    model_name: str,
    max_tokens: int,
    max_model_len: int,
) -> str:
  """Build a bounded plain-text report from request and grading records."""
  records = sorted(collector.records, key=lambda record: int(record["request_id"]))
  examples = _example_map(eval_results)
  lines = [
      "SIMPLE-EVALS DEBUG REPORT",
      "=" * 80,
      f"model: {model_name}",
      f"tasks: {', '.join(eval_results)}",
      f"max_tokens: {max_tokens}",
      f"max_model_len: {max_model_len}",
      f"logical_requests: {len(records)}",
      f"peak_in_flight: {collector.peak_in_flight}",
      "",
  ]

  for task in list(eval_results) + ["OVERALL"]:
    task_records = records if task == "OVERALL" else [r for r in records if r.get("task") == task]
    successful = [r for r in task_records if r.get("status") == "success"]
    terminal = [r for r in task_records if r.get("status") in ("error", "bad_request")]
    retried = [r for r in task_records if int(r.get("attempt_count", 0)) > 1]
    recovered = [r for r in successful if int(r.get("attempt_count", 0)) > 1]
    truncated = [r for r in successful if r.get("finish_reason") == "length"]
    empty = [r for r in successful if not str(r.get("response_text") or "").strip()]
    missing_usage = [r for r in successful if r.get("prompt_tokens") is None or r.get("completion_tokens") is None]
    saturated = [r for r in successful if int(r.get("completion_tokens") or 0) >= max_tokens]
    attempts = sum(int(r.get("attempt_count", 0)) for r in task_records)
    failed_attempts = sum(len(r.get("attempt_errors", [])) for r in task_records)
    duration = sum(task_durations.values()) if task == "OVERALL" else task_durations.get(task, 0.0)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in successful)
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in successful)
    task_examples = [examples[int(r["request_id"])] for r in task_records if int(r["request_id"]) in examples]
    incorrect = [e for e in task_examples if float(e.get("score") or 0.0) == 0.0]
    extraction_failures = [e for e in task_examples if e.get("extracted_answer") is None]
    scorable_noninfra = [
        e for e in task_examples if e.get("request_status") not in ("error", "bad_request")
    ]
    adjusted_correct = sum(float(e.get("score") or 0.0) > 0.0 for e in scorable_noninfra)
    correct = sum(float(e.get("score") or 0.0) > 0.0 for e in task_examples)
    truncated_ids = {int(r["request_id"]) for r in truncated}
    truncated_examples = [examples[request_id] for request_id in truncated_ids if request_id in examples]
    nontruncated_examples = [
        examples[int(r["request_id"])]
        for r in successful
        if int(r["request_id"]) not in truncated_ids and int(r["request_id"]) in examples
    ]
    truncated_correct = sum(float(e.get("score") or 0.0) > 0.0 for e in truncated_examples)
    nontruncated_correct = sum(float(e.get("score") or 0.0) > 0.0 for e in nontruncated_examples)
    interval = _wilson_interval(correct, len(task_examples))
    context_pressure = [
        r for r in successful if int(r.get("prompt_tokens") or 0) + max_tokens >= 0.9 * max_model_len
    ]

    lines.extend(
        [
            f"[{task}]",
            f"wall_time_s: {_fmt_number(duration)}",
            f"throughput_requests_per_s: {_fmt_number(len(task_records) / duration if duration else None)}",
            f"throughput_completion_tokens_per_s: {_fmt_number(completion_tokens / duration if duration else None)}",
            "throughput_total_tokens_per_s: "
            + _fmt_number((prompt_tokens + completion_tokens) / duration if duration else None),
            f"latency: {_latency_line(task_records)}",
            f"physical_attempt_latency: {_attempt_latency_line(task_records)}",
            f"prompt_tokens: {_token_line(successful, 'prompt_tokens')}",
            f"completion_tokens: {_token_line(successful, 'completion_tokens')}",
            f"terminal_error_rate: {_rate(len(terminal), len(task_records))}",
            f"physical_attempt_failure_rate: {_rate(failed_attempts, attempts)}",
            f"retry_rate: {_rate(len(retried), len(task_records))}",
            f"recovered_after_retry_rate: {_rate(len(recovered), len(task_records))}",
            f"empty_response_rate: {_rate(len(empty), len(successful))}",
            f"missing_usage_metadata_rate: {_rate(len(missing_usage), len(successful))}",
            f"output_truncation_rate: {_rate(len(truncated), len(successful))}",
            f"completion_token_saturation_rate: {_rate(len(saturated), len(successful))}",
            f"context_pressure_at_least_90pct: {_rate(len(context_pressure), len(successful))}",
            f"incorrect_answer_rate: {_rate(len(incorrect), len(task_examples))}",
            f"answer_extraction_failure_rate: {_rate(len(extraction_failures), len(task_examples))}",
            f"grading_coverage: {_rate(len(task_examples), len(task_records))}",
            f"official_accuracy: {_rate(correct, len(task_examples))}",
            f"diagnostic_accuracy_excluding_infrastructure_errors: {_rate(adjusted_correct, len(scorable_noninfra))}",
            f"truncated_response_accuracy: {_rate(truncated_correct, len(truncated_examples))}",
            f"nontruncated_response_accuracy: {_rate(nontruncated_correct, len(nontruncated_examples))}",
            "accuracy_95pct_wilson_interval: "
            + (f"[{interval[0]:.2%}, {interval[1]:.2%}]" if interval else "n/a"),
            f"finish_reasons: {dict(Counter(str(r.get('finish_reason')) for r in successful))}",
            "error_types: "
            + str(dict(Counter(str(r.get("error_type")) for r in terminal if r.get("error_type")))),
            "",
        ]
    )

  categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for record in records:
    example = examples.get(int(record["request_id"]), {})
    status = record.get("status")
    if status in ("error", "bad_request"):
      categories["terminal inference failures"].append(record)
    if record.get("finish_reason") == "length":
      categories["truncated responses"].append(record)
    if example and example.get("extracted_answer") is None:
      categories["answer extraction failures"].append(record)
    elif float(example.get("score") or 0.0) == 0.0 and status == "success":
      categories["incorrect answers"].append(record)
  categories["slowest successful requests"] = sorted(
      (r for r in records if r.get("status") == "success"),
      key=lambda r: float(r.get("latency_s") or 0.0),
      reverse=True,
  )

  lines.extend(["DIAGNOSTIC SAMPLES", "=" * 80, "Each category is capped at five examples.", ""])
  for category, samples in categories.items():
    lines.append(f"-- {category} --")
    for record in samples[:5]:
      example = examples.get(int(record["request_id"]), {})
      sample_lines = [
          f"request_id={record['request_id']} task={record.get('task')} status={record.get('status')} "
          f"latency_s={float(record.get('latency_s') or 0.0):.3f}",
          f"attempts={record.get('attempt_count')} finish_reason={record.get('finish_reason')} "
          f"prompt_tokens={record.get('prompt_tokens')} completion_tokens={record.get('completion_tokens')}",
          f"score={example.get('score')} correct_answer={example.get('correct_answer')!r} "
          f"extracted_answer={example.get('extracted_answer')!r}",
          f"error={_clip(record.get('error_detail') or '', 1000)}",
          "prompt:\n" + _clip(_prompt_text(record), 2000),
          "final_output:\n" + _clip(record.get("response_text") or "", 4000),
      ]
      if record.get("reasoning_text") is not None:
        sample_lines.append("reasoning (diagnostic-only):\n" + _clip(record["reasoning_text"], 4000))
      if record.get("raw_response_text") is not None:
        sample_lines.append("raw_output (diagnostic-only):\n" + _clip(record["raw_response_text"], 4000))
      sample_lines.append("")
      lines.extend(sample_lines)
    if not samples:
      lines.append("none\n")
  return "\n".join(lines).rstrip() + "\n"


def write_debug_report(report: str, results_path: str, json_local_path: str) -> str:
  """Write a debug report beside its JSON artifact and return the local path."""
  debug_filename = os.path.splitext(os.path.basename(json_local_path))[0] + ".debug.txt"
  local_path = os.path.join(os.path.dirname(json_local_path), debug_filename)
  with open(local_path, "w", encoding="utf-8") as report_file:
    report_file.write(report)
  if results_path.startswith("gs://"):
    from maxtext.utils.gcs_utils import upload_blob  # pylint: disable=import-outside-toplevel

    upload_blob(f"{results_path.rstrip('/')}/{debug_filename}", local_path)
    logger.info("Debug report written to %s/%s", results_path.rstrip("/"), debug_filename)
  else:
    logger.info("Debug report written to %s", local_path)
  return os.path.abspath(local_path)
