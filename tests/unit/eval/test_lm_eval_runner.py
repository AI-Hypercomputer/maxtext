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

"""Unit tests for harness_runner._map_results."""

from __future__ import annotations

import unittest

from maxtext.eval.runner.harness_runner import _map_results


class TestMapResults(unittest.TestCase):
  """Tests for _map_results."""

  def _make_raw(self, task_key: str, **metric_values) -> dict:
    """Build a lm-eval results dict for a single task."""
    return {"results": {task_key: metric_values}}

  def test_mmlu_accuracy(self):
    raw = self._make_raw("mmlu", **{"acc,none": 0.725})
    scores = _map_results(raw, ["mmlu"])
    self.assertIn("mmlu_accuracy", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy"], 72.5, places=1)

  def test_gpqa_diamond(self):
    raw = self._make_raw("gpqa_diamond", **{"acc,none": 0.4040})
    scores = _map_results(raw, ["gpqa_diamond"])
    self.assertIn("gpqa_diamond_accuracy", scores)
    self.assertAlmostEqual(scores["gpqa_diamond_accuracy"], 40.4, places=1)

  def test_gsm8k_strict_match_key(self):
    raw = self._make_raw("gsm8k", **{"exact_match,strict-match": 0.80})
    scores = _map_results(raw, ["gsm8k"])
    self.assertIn("gsm8k_accuracy", scores)
    self.assertAlmostEqual(scores["gsm8k_accuracy"], 80.0, places=1)

  def test_gsm8k_flexible_extract_key(self):
    raw = self._make_raw("gsm8k", **{"exact_match,flexible-extract": 0.75})
    scores = _map_results(raw, ["gsm8k"])
    self.assertIn("gsm8k_accuracy", scores)
    self.assertAlmostEqual(scores["gsm8k_accuracy"], 75.0, places=1)

  def test_zero_score_not_dropped(self):
    raw = self._make_raw("gsm8k", **{"exact_match,strict-match": 0.0})
    scores = _map_results(raw, ["gsm8k"])
    self.assertIn("gsm8k_accuracy", scores)
    self.assertAlmostEqual(scores["gsm8k_accuracy"], 0.0, places=1)

  def test_acc_norm_extracted(self):
    raw = self._make_raw("mmlu", **{"acc,none": 0.5, "acc_norm,none": 0.6})
    scores = _map_results(raw, ["mmlu"])
    self.assertIn("mmlu_accuracy", scores)
    self.assertIn("mmlu_accuracy_norm", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy_norm"], 60.0, places=1)

  def test_multiple_tasks(self):
    raw = {
        "results": {
            "mmlu": {"acc,none": 0.80},
            "gpqa_diamond": {"acc,none": 0.35},
        }
    }
    scores = _map_results(raw, ["mmlu", "gpqa_diamond"])
    self.assertIn("mmlu_accuracy", scores)
    self.assertIn("gpqa_diamond_accuracy", scores)
    self.assertAlmostEqual(scores["mmlu_accuracy"], 80.0, places=1)
    self.assertAlmostEqual(scores["gpqa_diamond_accuracy"], 35.0, places=1)

  def test_missing_task_returns_nothing(self):
    raw = {"results": {}}
    scores = _map_results(raw, ["mmlu"])
    self.assertNotIn("mmlu_accuracy", scores)

  def test_custom_task_name_used_directly(self):
    raw = self._make_raw("my_custom_task", **{"acc,none": 0.9})
    scores = _map_results(raw, ["my_custom_task"])
    self.assertIn("my_custom_task_accuracy", scores)

  def test_empty_tasks_list(self):
    raw = self._make_raw("mmlu", **{"acc,none": 0.9})
    scores = _map_results(raw, [])
    self.assertEqual(scores, {})


_REQUIRED_ARGS = [
    "--model_name",
    "llama3.1-8b",
    "--hf_path",
    "meta-llama/Llama-3.1-8B",
    "--base_output_directory",
    "gs://bucket/",
    "--run_name",
    "test_run",
    "--max_model_len",
    "4096",
]


class TestArgParser(unittest.TestCase):
  """Tests for _build_arg_parser — new flags added in the vllm_eval commits."""

  def setUp(self):
    from maxtext.eval.runner.harness_runner import _build_arg_parser  # pylint: disable=import-outside-toplevel

    self.parser = _build_arg_parser()

  def test_apply_chat_template_default_false(self):
    args = self.parser.parse_args(_REQUIRED_ARGS)
    self.assertFalse(args.apply_chat_template)

  def test_apply_chat_template_flag_sets_true(self):
    args = self.parser.parse_args(_REQUIRED_ARGS + ["--apply_chat_template"])
    self.assertTrue(args.apply_chat_template)

  def test_fewshot_as_multiturn_default_false(self):
    args = self.parser.parse_args(_REQUIRED_ARGS)
    self.assertFalse(args.fewshot_as_multiturn)

  def test_fewshot_as_multiturn_flag_sets_true(self):
    args = self.parser.parse_args(_REQUIRED_ARGS + ["--fewshot_as_multiturn"])
    self.assertTrue(args.fewshot_as_multiturn)

  def test_gen_kwargs_default_none(self):
    args = self.parser.parse_args(_REQUIRED_ARGS)
    self.assertIsNone(args.gen_kwargs)

  def test_gen_kwargs_passthrough(self):
    args = self.parser.parse_args(_REQUIRED_ARGS + ["--gen_kwargs", "until=[],max_gen_toks=1024"])
    self.assertEqual(args.gen_kwargs, "until=[],max_gen_toks=1024")


if __name__ == "__main__":
  unittest.main()
