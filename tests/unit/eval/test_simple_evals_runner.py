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

"""Unit tests for simple_evals_runner pure helpers.

These tests avoid instantiating real Eval objects (which download datasets) and
avoid booting a vLLM server; they cover the score mapping and task validation
logic only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from maxtext.eval.native_evals.aime_eval import extract_answer
from maxtext.eval.native_evals.gsm8k_eval import (
    _load_gsm8k_examples,
    extract_answer as extract_gsm8k_answer,
)
from maxtext.eval.reporting.simple_evals_debug_reporter import (
    SimpleEvalsDebugCollector,
    build_debug_report,
    write_debug_report,
)
from maxtext.eval.runner.simple_evals_runner import (
    SUPPORTED_TASKS,
    _build_arg_parser,
    _build_eval,
    _build_vllm_sampler,
    _effective_repeats,
    _map_scores,
    _validate_tasks,
)
from maxtext.eval.third_party.simple_evals import common as simple_evals_common
from maxtext.eval.third_party.simple_evals.types import EvalResult


def _make_result(score, metrics=None) -> EvalResult:
  return EvalResult(
      score=score,
      metrics=metrics or {},
      htmls=[],
      convos=[],
      metadata=None,
  )


class TestMapScores(unittest.TestCase):
  """Tests for _map_scores."""

  def test_score_to_percentage(self):
    scores = _map_scores("mmlu", _make_result(0.725))
    self.assertIn("mmlu_accuracy", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy"], 72.5, places=1)

  def test_zero_score_not_dropped(self):
    scores = _map_scores("gpqa", _make_result(0.0))
    self.assertIn("gpqa_accuracy", scores)
    self.assertAlmostEqual(scores["gpqa_accuracy"], 0.0, places=1)

  def test_none_score_omits_accuracy(self):
    scores = _map_scores("mmlu", _make_result(None))
    self.assertNotIn("mmlu_accuracy", scores)

  def test_numeric_metrics_passed_through(self):
    scores = _map_scores("gpqa", _make_result(0.5, {"chars": 1234.0}))
    self.assertIn("gpqa_accuracy", scores)
    self.assertIn("gpqa_chars", scores)
    self.assertAlmostEqual(scores["gpqa_chars"], 1234.0, places=1)

  def test_non_numeric_metrics_skipped(self):
    scores = _map_scores("mmlu", _make_result(0.5, {"note": "skip-me"}))
    self.assertNotIn("mmlu_note", scores)


class TestValidateTasks(unittest.TestCase):
  """Tests for _validate_tasks."""

  def test_supported_tasks_pass(self):
    _validate_tasks(list(SUPPORTED_TASKS))  # should not raise

  def test_new_tasks_are_registered(self):
    for task in ("gsm8k", "drop", "mgsm", "mgsm_en", "aime2024", "aime2025"):
      self.assertIn(task, SUPPORTED_TASKS)

  def test_unsupported_task_raises(self):
    with self.assertRaises(ValueError):
      _validate_tasks(["mmlu", "healthbench"])

  def test_empty_list_passes(self):
    _validate_tasks([])  # should not raise


class TestBuildEval(unittest.TestCase):
  """Tests for _build_eval task guard (does not download datasets)."""

  def test_unsupported_task_raises(self):
    with self.assertRaises(ValueError):
      _build_eval("math", num_examples=5, n_repeats=1)

  def test_repeat_defaults_match_task_protocols(self):
    self.assertEqual(_effective_repeats("gpqa", None, None), 4)
    self.assertEqual(_effective_repeats("aime2024", None, None), 1)
    self.assertEqual(_effective_repeats("mmlu", None, 9), 1)

  def test_subsetting_forces_one_repeat(self):
    self.assertEqual(_effective_repeats("gpqa", 10, 4), 1)


class TestDebugInfo(unittest.TestCase):
  """Tests for debug flag and plain-text diagnostics."""

  def test_debug_flag_defaults_off_and_can_be_enabled(self):
    parser = _build_arg_parser()
    required = [
        "--model_name", "model", "--hf_path", "hf", "--base_output_directory", "/tmp",
        "--run_name", "run", "--max_model_len", "4096",
    ]
    self.assertFalse(parser.parse_args(required).log_debug_info)
    defaults = parser.parse_args(required)
    self.assertIsNone(defaults.concurrency)
    self.assertIsNone(defaults.n_repeats)
    self.assertAlmostEqual(defaults.temperature, 0.5)
    self.assertFalse(defaults.continue_on_request_error)
    self.assertTrue(parser.parse_args(required + ["--log-debug-info"]).log_debug_info)

  def test_report_contains_rates_and_error_sample(self):
    collector = SimpleEvalsDebugCollector()
    collector.records = [
        {
            "request_id": 1,
            "task": "mmlu",
            "status": "success",
            "attempt_count": 1,
            "attempt_errors": [],
            "latency_s": 2.0,
            "successful_attempt_latency_s": 2.0,
            "prompt_tokens": 100,
            "completion_tokens": 32,
            "finish_reason": "length",
            "response_text": "Answer: B",
            "reasoning_text": "private reasoning",
            "raw_response_text": "<raw harmony output>",
            "prompt_messages": [{"role": "user", "content": "Question"}],
        }
    ]
    collector.peak_in_flight = 1
    result = _make_result(0.0)
    result.metadata = {
        "example_level_metadata": [
            {
                "request_id": 1,
                "request_status": "success",
                "score": 0.0,
                "correct_answer": "A",
                "extracted_answer": "B",
            }
        ]
    }
    report = build_debug_report(collector, {"mmlu": result}, {"mmlu": 2.0}, "model", 32, 4096)
    self.assertIn("output_truncation_rate: 100.00% (1/1)", report)
    self.assertIn("incorrect_answer_rate: 100.00% (1/1)", report)
    self.assertIn("-- incorrect answers --", report)
    self.assertIn("correct_answer='A' extracted_answer='B'", report)
    self.assertIn("final_output:\nAnswer: B", report)
    self.assertIn("reasoning (diagnostic-only):\nprivate reasoning", report)
    self.assertIn("raw_output (diagnostic-only):\n<raw harmony output>", report)

  def test_report_filename_matches_json_stem(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      json_path = Path(tmpdir) / "mmlu_model_20260101T000000Z.json"
      json_path.write_text("{}", encoding="utf-8")
      debug_path = write_debug_report("report\n", "/local/results", str(json_path))
      self.assertEqual(Path(debug_path).name, "mmlu_model_20260101T000000Z.debug.txt")
      self.assertEqual(Path(debug_path).read_text(encoding="utf-8"), "report\n")


class TestSamplerConcurrency(unittest.TestCase):
  def test_task_worker_pool_uses_resolved_concurrency(self):
    lock = threading.Lock()
    active = 0
    peak = 0

    def work(value):
      nonlocal active, peak
      with lock:
        active += 1
        peak = max(peak, active)
      time.sleep(0.02)
      with lock:
        active -= 1
      return value

    simple_evals_common.set_default_num_threads(2)
    self.assertEqual(simple_evals_common.map_with_progress(work, list(range(8)), pbar=False), list(range(8)))
    self.assertEqual(peak, 2)

  def test_inflight_requests_are_bounded(self):
    lock = threading.Lock()
    active = 0
    peak = 0

    def create(**_kwargs):
      nonlocal active, peak
      with lock:
        active += 1
        peak = max(peak, active)
      time.sleep(0.02)
      with lock:
        active -= 1
      return SimpleNamespace(
          choices=[SimpleNamespace(message=SimpleNamespace(content="Answer: A"), finish_reason="stop")],
          usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
      )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with mock.patch("openai.OpenAI", return_value=client):
      sampler = _build_vllm_sampler("http://server", "model", 32, 0.5, concurrency=2)
      with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: sampler([{"role": "user", "content": "q"}]), range(8)))
    self.assertEqual(len(results), 8)
    self.assertEqual(peak, 2)

  def test_debug_collection_does_not_change_terminal_error_behavior(self):
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[], usage=None)))
    )
    collector = SimpleEvalsDebugCollector()
    with mock.patch("openai.OpenAI", return_value=client):
      sampler = _build_vllm_sampler(
          "http://server",
          "model",
          32,
          0.5,
          debug_collector=collector,
          continue_on_request_error=False,
      )
      with self.assertRaises(RuntimeError):
        sampler([{"role": "user", "content": "q"}])
    self.assertEqual(collector.records[0]["status"], "error")

  def test_analysis_only_truncation_is_success_and_raw_stays_diagnostic(self):
    captured_kwargs = {}

    def create(**kwargs):
      captured_kwargs.update(kwargs)
      return SimpleNamespace(
          choices=[
              SimpleNamespace(
                  message=SimpleNamespace(content=None, reasoning="unfinished reasoning"),
                  finish_reason="length",
              )
          ],
          usage=SimpleNamespace(prompt_tokens=10, completion_tokens=32, total_tokens=42),
          model_extra={"diagnostics": {"raw_output": "<raw analysis-only output>"}},
      )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    collector = SimpleEvalsDebugCollector()
    with mock.patch("openai.OpenAI", return_value=client):
      sampler = _build_vllm_sampler(
          "http://server",
          "model",
          32,
          0.5,
          reasoning_effort="high",
          debug_collector=collector,
      )
      response = sampler([{"role": "user", "content": "q"}])

    self.assertEqual(response.response_text, "")
    self.assertNotIn("<raw analysis-only output>", response.response_text)
    self.assertEqual(collector.records[0]["status"], "success")
    self.assertEqual(collector.records[0]["finish_reason"], "length")
    self.assertEqual(collector.records[0]["reasoning_text"], "unfinished reasoning")
    self.assertEqual(collector.records[0]["raw_response_text"], "<raw analysis-only output>")
    self.assertEqual(collector.records[0]["attempt_count"], 1)
    self.assertEqual(
        captured_kwargs["extra_body"],
        {
            "include_reasoning": True,
            "include_raw_output": True,
            "reasoning_effort": "high",
        },
    )

  def test_continue_on_request_error_is_explicit(self):
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[], usage=None)))
    )
    with mock.patch("openai.OpenAI", return_value=client):
      sampler = _build_vllm_sampler(
          "http://server",
          "model",
          32,
          0.5,
          continue_on_request_error=True,
      )
      response = sampler([{"role": "user", "content": "q"}])
    self.assertEqual(response.response_metadata["status"], "error")


class TestAimeExtractAnswer(unittest.TestCase):
  """Tests for aime_eval.extract_answer (pure function, no dataset download)."""

  def test_answer_line(self):
    self.assertEqual(extract_answer("Steps...\nAnswer: 42"), "42")

  def test_answer_line_case_insensitive_with_dollar(self):
    self.assertEqual(extract_answer("steps...\nanswer: $007"), "007")

  def test_boxed_without_required_answer_line_is_rejected(self):
    self.assertIsNone(extract_answer("Steps...\nSo the result is \\boxed{123}."))

  def test_answer_line_preferred_over_boxed(self):
    self.assertEqual(extract_answer("\\boxed{1} is wrong.\nAnswer: 2"), "2")

  def test_last_integer_without_required_answer_line_is_rejected(self):
    self.assertIsNone(extract_answer("There are 3 cases, then 4 more, total 7."))

  def test_last_valid_answer_line_wins(self):
    self.assertEqual(extract_answer("Answer: 41\nCorrection follows.\nAnswer: 42"), "42")

  def test_answer_line_must_be_final(self):
    self.assertIsNone(extract_answer("Answer: 42\nBut I am still reasoning."))

  def test_no_integer_returns_none(self):
    self.assertIsNone(extract_answer("I cannot solve this."))


class TestGsm8kExtractAnswer(unittest.TestCase):
  def test_final_answer_line(self):
    self.assertEqual(extract_gsm8k_answer("Reasoning\nAnswer: 1,234"), "1234")

  def test_last_answer_line_wins(self):
    self.assertEqual(extract_gsm8k_answer("Answer: 10\nCorrection\nAnswer: 12.5"), "12.5")

  def test_unstructured_number_is_rejected(self):
    self.assertIsNone(extract_gsm8k_answer("The result might be 42."))

  def test_answer_line_must_be_final(self):
    self.assertIsNone(extract_gsm8k_answer("Answer: 42\nAdditional text"))

  def test_dataset_uses_hub_cache_instead_of_direct_parquet_url(self):
    rows = [SimpleNamespace(to_dict=lambda: {"question": "q", "answer": "#### 1"}) for _ in range(1319)]
    dataframe = SimpleNamespace(iterrows=lambda: enumerate(rows))
    with (
        mock.patch("huggingface_hub.hf_hub_download", return_value="/cache/gsm8k.parquet") as download,
        mock.patch("maxtext.eval.native_evals.gsm8k_eval.pandas.read_parquet", return_value=dataframe) as read_parquet,
    ):
      examples = _load_gsm8k_examples()

    self.assertEqual(len(examples), 1319)
    self.assertEqual(examples[0]["answer"], "#### 1")
    download.assert_called_once_with(
        repo_id="openai/gsm8k",
        repo_type="dataset",
        filename="main/test-00000-of-00001.parquet",
    )
    read_parquet.assert_called_once_with("/cache/gsm8k.parquet")


if __name__ == "__main__":
  unittest.main()
