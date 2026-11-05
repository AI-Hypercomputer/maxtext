# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MaxText coding RL dataset processor and metrics (`code_rl`)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pytest

from maxtext.trainers.post_train.rl import code_rl
from maxtext.trainers.post_train.rl import utils_rl

pytestmark = [pytest.mark.post_training]


class CodeRlDatasetProcessorTest(unittest.TestCase):
  """Tests for maxtext.trainers.post_train.rl.code_rl.dataset_processor."""

  def test_process_asserts_record(self) -> None:
    raw = {
        "prompt_id": "mbpp_42",
        "problem": "Write a function `cube(n)` that returns `n ** 3`.",
        "tests": ["assert cube(2) == 8", "assert cube(-3) == -27"],
        "format": "asserts",
        "entry_point": "cube",
    }
    rec = code_rl.process_code_rl_record(raw)
    self.assertEqual(rec["prompt_id"], "mbpp_42")
    self.assertEqual(rec["format"], "asserts")
    self.assertIn("cube(n)", rec["prompts"])
    self.assertEqual(rec["question"], raw["problem"])
    self.assertIn("```python", rec["format_instructions"])
    self.assertEqual(rec["entry_point"], "cube")
    self.assertEqual(len(rec["tests"]), 2)

  def test_process_data_entrypoint_loadable_by_utils_rl(self) -> None:
    fn = utils_rl.load_custom_callable("maxtext.trainers.post_train.rl.code_rl.dataset_processor", "process_data")
    rec = fn(
        "json",
        None,
        None,
        None,
        {"problem": "Write add(a, b).", "tests": ["assert add(1, 2) == 3"], "prompt_id": "p1"},
    )
    self.assertIn("Write add(a, b).", rec["prompts"])
    self.assertEqual(rec["question"], "Write add(a, b).")
    self.assertEqual(rec["prompt_id"], "p1")

  def test_process_stdio_record_and_auto_detect_format(self) -> None:
    raw = {
        "task_id": 108,
        "question": "Read integer x from stdin and print x + 1.",
        "tests": {"inputs": ["10\n", "0\n"], "outputs": ["11\n", "1\n"]},
    }
    rec = code_rl.process_code_rl_record(raw)
    self.assertEqual(rec["prompt_id"], "108")
    self.assertEqual(rec["format"], "stdio")
    self.assertEqual(rec["tests"]["inputs"], ["10\n", "0\n"])
    self.assertEqual(rec["tests"]["outputs"], ["11\n", "1\n"])

  def test_partition_visible_and_hidden_tests_disjoint_and_deterministic(self) -> None:
    assert_tests = ["assert f(1) == 1", "assert f(2) == 4", "assert f(3) == 9", "assert f(4) == 16"]
    vis_a, hid_a = code_rl.partition_visible_and_hidden_tests(assert_tests, "item_1", fmt="asserts")
    vis_b, hid_b = code_rl.partition_visible_and_hidden_tests(assert_tests, "item_1", fmt="asserts")
    self.assertEqual(vis_a, vis_b)
    self.assertEqual(hid_a, hid_b)
    self.assertTrue(set(vis_a).isdisjoint(set(hid_a)))
    self.assertEqual(len(vis_a) + len(hid_a), len(assert_tests))

    stdio_tests = {"format": "stdio", "inputs": ["1\n", "2\n", "3\n"], "outputs": ["2\n", "4\n", "6\n"]}
    vis_io, hid_io = code_rl.partition_visible_and_hidden_tests(stdio_tests, "item_2", fmt="stdio")
    self.assertEqual(vis_io["inputs"], ["1\n", "2\n"])
    self.assertEqual(hid_io["inputs"], ["3\n"])

  def test_process_record_rejects_missing_problem_or_tests(self) -> None:
    with self.assertRaises(ValueError):
      code_rl.process_code_rl_record({"problem": "   ", "tests": ["assert True"]})
    with self.assertRaises(ValueError):
      code_rl.process_code_rl_record({"problem": "Valid problem", "tests": []})
    with self.assertRaises(ValueError):
      code_rl.process_code_rl_record(
          {
              "problem": "Valid problem",
              "tests": ["assert True"],
              "format": "invalid_fmt",
          }
      )

  def test_format_code_rl_dataset_and_load_jsonl(self) -> None:
    raw_items = [
        {
            "prompt_id": f"p_{i}",
            "problem": f"Compute f_{i}(x).",
            "tests": [f"assert f_{i}(1) == 1"],
        }
        for i in range(4)
    ]
    processed = code_rl.format_code_rl_dataset(raw_items)
    self.assertEqual(len(processed), 4)
    self.assertEqual(processed[2]["prompt_id"], "p_2")

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
      for item in raw_items:
        tmp.write(json.dumps(item) + "\n")
      tmp_path = tmp.name

    try:
      loaded = code_rl.load_jsonl_records(tmp_path)
      self.assertEqual(len(loaded), 4)
      self.assertEqual(loaded[0]["prompt_id"], "p_0")
    finally:
      os.unlink(tmp_path)


class CodeRlMetricsTest(unittest.TestCase):
  """Tests for maxtext.trainers.post_train.rl.code_rl.metrics."""

  def test_compute_code_rl_metrics_solve_and_repair_gain(self) -> None:
    turn1_passed = [True, False, False, True]
    final_passed = [True, True, False, True]
    assistant_lengths = [400, 1200, 8192, 600]
    m = code_rl.compute_code_rl_metrics(
        turn1_passed=turn1_passed,
        final_passed=final_passed,
        assistant_lengths=assistant_lengths,
        max_response_length=8192,
    )
    self.assertAlmostEqual(m["code_rl/turn1_solve"], 0.50)
    self.assertAlmostEqual(m["code_rl/final_solve"], 0.75)
    self.assertAlmostEqual(m["code_rl/repair_gain"], 0.25)
    self.assertAlmostEqual(m["code_rl/mean_assistant_tokens"], 2598.0)
    self.assertAlmostEqual(m["code_rl/truncation_rate"], 0.25)

  def test_compute_code_rl_metrics_empty_batch(self) -> None:
    m = code_rl.compute_code_rl_metrics([], [], [])
    self.assertEqual(m["code_rl/turn1_solve"], 0.0)
    self.assertEqual(m["code_rl/final_solve"], 0.0)
    self.assertEqual(m["code_rl/repair_gain"], 0.0)
    self.assertEqual(m["code_rl/truncation_rate"], 0.0)

  def test_extract_trajectory_code_rl_metrics(self) -> None:
    trajectories = [
        {
            "conversation_masks": [0, 1, 1, 1, 0, 1, 1],
            "trajectory_reward": 1.0,
            "env_extras": {
                "passed": True,
                "turn_records": [{"test_result": {"all_passed": False}}],
            },
            "judge": {"passed": True, "assistant_len": 500},
        },
        {
            "conversation_masks": [1] * 8192,
            "trajectory_reward": 0.0,
            "env_extras": {
                "passed": False,
                "turn_records": [{"test_result": {"all_passed": False}}],
            },
            "judge": {"passed": False, "assistant_len": 8192},
        },
    ]
    m = code_rl.extract_trajectory_code_rl_metrics(trajectories, max_response_length=8192)
    self.assertAlmostEqual(m["code_rl/turn1_solve"], 0.0)
    self.assertAlmostEqual(m["code_rl/final_solve"], 0.5)
    self.assertAlmostEqual(m["code_rl/repair_gain"], 0.5)
    self.assertAlmostEqual(m["code_rl/truncation_rate"], 0.5)

    tunix_metrics = code_rl.get_code_rl_metrics(
        prompts=["p1", "p2"],
        completions=["c1", "c2"],
        rewards=np.array([1.0, 0.0]),
        advantages=np.array([0.5, -0.5]),
        trajectories=trajectories,
    )
    for required_key in (
        "reward/turn1_solve",
        "reward/final_solve",
        "reward/repair_gain",
        "length/assistant_tokens_mean",
        "length/assistant_tokens_p90",
        "length/truncation_rate_at_8192",
    ):
      self.assertIn(required_key, tunix_metrics)
      val, agg_fn = tunix_metrics[required_key]
      self.assertIsInstance(val, (int, float))
      self.assertTrue(callable(agg_fn))
    self.assertAlmostEqual(tunix_metrics["reward/repair_gain"][0], 0.5)


if __name__ == "__main__":
  unittest.main()
