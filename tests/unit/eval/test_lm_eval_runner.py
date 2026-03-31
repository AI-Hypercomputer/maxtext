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

"""Unit tests for lm_eval_runner helper functions (no server required)."""

from __future__ import annotations

import unittest

from maxtext.eval.runner.lm_eval_runner import _build_model_args, _map_lm_eval_results


class TestBuildModelArgs(unittest.TestCase):
  """Tests for _build_model_args."""

  def test_basic_args(self):
    result = _build_model_args(
        base_url="http://localhost:8000",
        tokenizer_path="meta-llama/Llama-3.1-8B",
        model_name="llama3.1-8b",
        hf_token=None,
    )
    self.assertIn("model=llama3.1-8b", result)
    self.assertIn("http://localhost:8000/v1/completions", result)
    self.assertIn("tokenizer=meta-llama/Llama-3.1-8B", result)
    self.assertIn("tokenizer_backend=huggingface", result)

  def test_with_hf_token(self):
    result = _build_model_args(
        base_url="http://localhost:8000",
        tokenizer_path="meta-llama/Llama-3.1-8B",
        model_name="llama3.1-8b",
        hf_token="hf_abc123",
    )
    self.assertIn("token=hf_abc123", result)

  def test_no_hf_token(self):
    result = _build_model_args(
        base_url="http://localhost:8000",
        tokenizer_path="meta-llama/Llama-3.1-8B",
        model_name="llama3.1-8b",
        hf_token=None,
    )
    self.assertNotIn("token=", result)

  def test_empty_token_string_not_included(self):
    # An empty string should not be treated as a valid token.
    result = _build_model_args(
        base_url="http://localhost:8000",
        tokenizer_path="hf/model",
        model_name="my-model",
        hf_token="",
    )
    self.assertNotIn("token=", result)

  def test_args_are_comma_separated(self):
    result = _build_model_args(
        base_url="http://host:9000",
        tokenizer_path="tok/path",
        model_name="my-model",
        hf_token=None,
    )
    # Should be a single string with no newlines.
    self.assertNotIn("\n", result)
    # All key=value pairs must be comma-joined.
    parts = result.split(",")
    self.assertGreaterEqual(len(parts), 4)


class TestMapLmEvalResults(unittest.TestCase):
  """Tests for _map_lm_eval_results."""

  def _make_lm_results(self, task_key: str, acc: float | None = None, acc_norm: float | None = None):
    """Build a minimal lm-eval results dict for a single task."""
    task_dict: dict = {}
    if acc is not None:
      task_dict["acc,none"] = acc
    if acc_norm is not None:
      task_dict["acc_norm,none"] = acc_norm
    return {"results": {task_key: task_dict}}

  def test_mmlu_accuracy(self):
    lm_results = self._make_lm_results("mmlu", acc=0.725)
    scores = _map_lm_eval_results(lm_results, ["mmlu"])
    self.assertIn("mmlu_accuracy", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy"], 72.5, places=1)

  def test_gpqa_accuracy(self):
    lm_results = self._make_lm_results("gpqa_diamond", acc=0.4040)
    scores = _map_lm_eval_results(lm_results, ["gpqa"])
    self.assertIn("gpqa_accuracy", scores)
    self.assertAlmostEqual(scores["gpqa_accuracy"], 40.4, places=1)

  def test_missing_task_returns_nothing(self):
    lm_results = {"results": {}}
    scores = _map_lm_eval_results(lm_results, ["mmlu"])
    self.assertNotIn("mmlu_accuracy", scores)

  def test_acc_norm_extracted(self):
    lm_results = self._make_lm_results("mmlu", acc=0.5, acc_norm=0.6)
    scores = _map_lm_eval_results(lm_results, ["mmlu"])
    self.assertIn("mmlu_accuracy", scores)
    self.assertIn("mmlu_accuracy_norm", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy_norm"], 60.0, places=1)

  def test_multiple_tasks(self):
    lm_results = {
        "results": {
            "mmlu": {"acc,none": 0.80},
            "gpqa_diamond": {"acc,none": 0.35},
        }
    }
    scores = _map_lm_eval_results(lm_results, ["mmlu", "gpqa"])
    self.assertIn("mmlu_accuracy", scores)
    self.assertIn("gpqa_accuracy", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy"], 80.0, places=1)
    self.assertAlmostEqual(scores["gpqa_accuracy"], 35.0, places=1)

  def test_unknown_task_name_mapped_to_itself(self):
    # If a task is not in _TASK_MAP, _map_lm_eval_results looks it up by the
    # raw name as a fallback.  Verify it doesn't crash.
    lm_results = {"results": {"my_custom_task": {"acc,none": 0.9}}}
    scores = _map_lm_eval_results(lm_results, ["my_custom_task"])
    # The raw task name is used as the results key when not in _TASK_MAP.
    self.assertIn("my_custom_task_accuracy", scores)

  def test_acc_without_none_suffix(self):
    # Older lm-eval versions may omit the ",none" suffix.
    lm_results = {"results": {"mmlu": {"acc": 0.55}}}
    scores = _map_lm_eval_results(lm_results, ["mmlu"])
    self.assertIn("mmlu_accuracy", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy"], 55.0, places=1)

  def test_empty_tasks_list(self):
    lm_results = {"results": {"mmlu": {"acc,none": 0.9}}}
    scores = _map_lm_eval_results(lm_results, [])
    self.assertEqual(scores, {})

  def test_scores_multiplied_by_100(self):
    lm_results = self._make_lm_results("mmlu", acc=1.0)
    scores = _map_lm_eval_results(lm_results, ["mmlu"])
    # raw value is 1.0 (fraction); expect 100.0 in output.
    self.assertAlmostEqual(scores["mmlu_accuracy"], 100.0, places=1)


if __name__ == "__main__":
  unittest.main()
