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

"""Unit tests for maxtext.eval.scoring.math_scorer."""

import re
import unittest

from maxtext.eval.scoring.math_scorer import (
    MathScoreResult,
    build_format_regexes,
    score_batch,
    score_response,
)

_THINK_START = "<think>"
_THINK_END = "</think>"
_ANS_START = "<answer>"
_ANS_END = "</answer>"


def _fmt(reasoning: str, answer: str) -> str:
  return f"{_THINK_START}{reasoning}{_THINK_END}{_ANS_START}{answer}{_ANS_END}"


def _regexes():
  return build_format_regexes(
      reasoning_start_token=_THINK_START,
      reasoning_end_token=_THINK_END,
      solution_start_token=_ANS_START,
      solution_end_token=_ANS_END,
  )


class TestBuildFormatRegexes(unittest.TestCase):

  def test_returns_two_compiled_patterns(self):
    match_fmt, fallback = _regexes()
    self.assertIsInstance(match_fmt, re.Pattern)
    self.assertIsInstance(fallback, re.Pattern)

  def test_full_match_captures_answer(self):
    match_fmt, _ = _regexes()
    response = _fmt("some reasoning", "42")
    m = match_fmt.search(response)
    self.assertIsNotNone(m)
    self.assertEqual(m.group(1), "42")

  def test_full_match_requires_reasoning_tags(self):
    match_fmt, _ = _regexes()
    # Answer tags only, no reasoning tags — should NOT match full pattern.
    self.assertIsNone(match_fmt.search(f"{_ANS_START}42{_ANS_END}"))

  def test_fallback_matches_answer_tag_only(self):
    _, fallback = _regexes()
    response = f"some text {_ANS_START}42{_ANS_END}"
    matches = fallback.findall(response)
    self.assertEqual(matches, ["42"])

  def test_custom_tokens(self):
    match_fmt, fallback = build_format_regexes(
        reasoning_start_token="<reasoning>",
        reasoning_end_token="</reasoning>",
        solution_start_token="<solution>",
        solution_end_token="</solution>",
    )
    response = "<reasoning>work</reasoning><solution>7</solution>"
    self.assertIsNotNone(match_fmt.search(response))
    self.assertEqual(fallback.findall(response), ["7"])


class TestScoreResponse(unittest.TestCase):

  def setUp(self):
    self.match_fmt, self.fallback = _regexes()

  def _score(self, response, reference, normalize=False):
    return score_response(response, reference, self.match_fmt, self.fallback, normalize=normalize)

  def test_correct_answer_full_format(self):
    result = self._score(_fmt("step by step", "42"), "42")
    self.assertTrue(result.is_correct)
    self.assertTrue(result.has_correct_format)

  def test_wrong_answer_correct_format(self):
    result = self._score(_fmt("step by step", "99"), "42")
    self.assertFalse(result.is_correct)
    self.assertTrue(result.has_correct_format)

  def test_no_tags_no_format(self):
    result = self._score("the answer is 42", "42")
    self.assertFalse(result.has_correct_format)

  def test_fallback_answer_extraction(self):
    # No reasoning tags, but answer tag present — extracts via fallback.
    response = f"blah {_ANS_START}42{_ANS_END}"
    result = self._score(response, "42")
    self.assertFalse(result.has_correct_format)
    self.assertTrue(result.is_correct)

  def test_partial_correctness_within_tolerance(self):
    # 101 / 100 = 1.01, within [0.9, 1.1]
    result = self._score(_fmt("work", "101"), "100")
    self.assertFalse(result.is_correct)
    self.assertTrue(result.is_partially_correct)

  def test_partial_correctness_outside_tolerance(self):
    # 200 / 100 = 2.0, outside [0.9, 1.1]
    result = self._score(_fmt("work", "200"), "100")
    self.assertFalse(result.is_partially_correct)

  def test_returns_named_tuple(self):
    result = self._score(_fmt("work", "1"), "1")
    self.assertIsInstance(result, MathScoreResult)

  def test_no_crash_on_empty_response(self):
    result = self._score("", "42")
    self.assertFalse(result.is_correct)
    self.assertFalse(result.has_correct_format)

  def test_normalize_flag_does_not_crash(self):
    result = self._score(_fmt("work", "42"), "42", normalize=True)
    self.assertTrue(result.is_correct)


class TestScoreBatch(unittest.TestCase):

  def setUp(self):
    self.match_fmt, self.fallback = _regexes()

  def test_all_correct(self):
    responses = [_fmt("r", "1"), _fmt("r", "2"), _fmt("r", "3")]
    refs = ["1", "2", "3"]
    out = score_batch(responses, refs)
    self.assertAlmostEqual(out["accuracy"], 1.0)
    self.assertEqual(out["num_correct"], 3)
    self.assertEqual(out["num_total"], 3)

  def test_partial_correct(self):
    responses = [_fmt("r", "1"), _fmt("r", "99")]
    refs = ["1", "2"]
    out = score_batch(responses, refs)
    self.assertAlmostEqual(out["accuracy"], 0.5)
    self.assertEqual(out["num_correct"], 1)

  def test_all_wrong(self):
    responses = [_fmt("r", "99")]
    refs = ["1"]
    out = score_batch(responses, refs)
    self.assertAlmostEqual(out["accuracy"], 0.0)

  def test_empty_batch(self):
    out = score_batch([], [])
    self.assertAlmostEqual(out["accuracy"], 0.0)
    self.assertEqual(out["num_total"], 0)

  def test_length_mismatch_raises(self):
    with self.assertRaises(ValueError):
      score_batch(["a", "b"], ["1"])

  def test_returns_all_expected_keys(self):
    out = score_batch([_fmt("r", "1")], ["1"])
    for key in ("accuracy", "partial_accuracy", "format_accuracy",
                "num_correct", "num_partial_correct", "num_correct_format", "num_total"):
      self.assertIn(key, out)


if __name__ == "__main__":
  unittest.main()
