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

"""Unit tests for maxtext.eval.scoring scorer modules."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestRougeScorer(unittest.TestCase):
  """Tests for maxtext.eval.scoring.rouge_scorer.score_batch."""

  def _make_mock_rouge_metric(self, rouge1=0.9, rouge2=0.85, rougeL=0.88, rougeLsum=0.88):
    """Return a mock evaluate metric whose compute() returns fixed ROUGE scores."""
    mock_metric = MagicMock()
    mock_metric.compute.return_value = {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "rougeLsum": rougeLsum,
    }
    return mock_metric

  def _run_score_batch(self, responses, references, mock_metric):
    """Import score_batch with evaluate and nltk patched, return result dict."""
    with patch("evaluate.load", return_value=mock_metric), \
         patch("nltk.download"), \
         patch("nltk.sent_tokenize", side_effect=lambda s: [s]):
      from maxtext.eval.scoring.rouge_scorer import score_batch
      return score_batch(responses, references)

  def test_perfect_match(self):
    texts = ["The quick brown fox", "Hello world"]
    mock_metric = self._make_mock_rouge_metric(rouge1=1.0, rouge2=1.0, rougeL=1.0, rougeLsum=1.0)
    result = self._run_score_batch(texts, texts, mock_metric)
    # High ROUGE scores expected for identical strings.
    self.assertGreater(result["rouge1"], 50.0)
    self.assertGreater(result["rouge2"], 50.0)
    self.assertGreater(result["rougeL"], 50.0)
    # gen_num should match the batch size.
    self.assertEqual(result["gen_num"], 2)

  def test_empty_inputs(self):
    mock_metric = self._make_mock_rouge_metric(rouge1=0.0, rouge2=0.0, rougeL=0.0, rougeLsum=0.0)
    result = self._run_score_batch([], [], mock_metric)
    self.assertEqual(result["gen_num"], 0)

  def test_length_mismatch_raises(self):
    mock_metric = self._make_mock_rouge_metric()
    with self.assertRaises(ValueError):
      self._run_score_batch(["a", "b"], ["only_one"], mock_metric)

  def test_returns_rouge_keys(self):
    mock_metric = self._make_mock_rouge_metric()
    result = self._run_score_batch(["hello"], ["hello"], mock_metric)
    # Must contain at least the four standard ROUGE keys plus gen_num.
    self.assertIn("gen_num", result)
    # At least one of rougeL / rougeLsum must be present.
    has_rouge_keys = (
        "rouge1" in result
        and "rouge2" in result
        and ("rougeL" in result or "rougeLsum" in result)
    )
    self.assertTrue(has_rouge_keys)

  def test_partial_overlap(self):
    mock_metric = self._make_mock_rouge_metric(rouge1=0.5, rouge2=0.2, rougeL=0.45, rougeLsum=0.45)
    # Multiply ×100 in eval_accuracy_mlperf, so 0.5 → 50.0.
    result = self._run_score_batch(["fox brown quick"], ["quick brown fox"], mock_metric)
    self.assertGreater(result["rouge1"], 0.0)
    self.assertLess(result["rouge1"], 100.0)


if __name__ == "__main__":
  unittest.main()
