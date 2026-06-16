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

"""Tests for math_verify_pool grading and score-assignment logic."""
import pytest
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sympy

from maxtext.trainers.post_train.rl import math_verify_pool as mvp
from maxtext.trainers.post_train.rl.math_verify_pool import (
    are_equal_under_sympy,
    math_verify_pool,
    verify_math_worker,
)

pytestmark = [pytest.mark.post_training]


def _make_config(reward=1.0):
  return SimpleNamespace(reward_exact_answer=reward)


class _FakeAsyncResult:
  """Stand-in for multiprocessing.pool.AsyncResult.

  Runs the target synchronously at construction and caches the outcome so
  `ready()`/`get()` match the real `AsyncResult` contract without spawning
  a worker. Lets us drive `math_verify_pool`'s busy-poll in-process.
  """

  def __init__(self, fn, args):
    try:
      self._value = fn(*args)
      self._exc = None
    except Exception as exc:  # pylint: disable=broad-except
      self._value = None
      self._exc = exc

  def ready(self):
    return True

  def get(self, timeout=None):  # pylint: disable=unused-argument
    if self._exc is not None:
      raise self._exc
    return self._value


class _FakePool:
  """Minimal pool stub: runs `apply_async` synchronously in-process."""

  def apply_async(self, fn, args):
    return _FakeAsyncResult(fn, args)


def _fake_get_pool(num_procs):  # pylint: disable=unused-argument
  return _FakePool()


class VerifyMathWorkerTest(unittest.TestCase):
  """Unit tests for the in-process grader (no pool, no spawned workers)."""

  def test_exact_numeric_match(self):
    score = verify_math_worker(["\\boxed{42}"], ["\\boxed{42}"])
    self.assertEqual(score, 1.0)

  def test_numeric_mismatch(self):
    score = verify_math_worker(["\\boxed{42}"], ["\\boxed{99}"])
    self.assertEqual(score, 0.0)

  def test_multiple_golds_one_matches(self):
    score = verify_math_worker(["\\boxed{1}", "\\boxed{42}"], ["\\boxed{42}"])
    self.assertEqual(score, 1.0)

  def test_multiple_golds_none_matches(self):
    score = verify_math_worker(["\\boxed{1}", "\\boxed{2}"], ["\\boxed{99}"])
    self.assertEqual(score, 0.0)

  def test_empty_prediction_returns_zero(self):
    score = verify_math_worker(["\\boxed{42}"], [""])
    self.assertEqual(score, 0.0)

  def test_fraction_equivalent_to_decimal(self):
    # 1/2 and 0.5 are numerically equal — verify() should catch this even
    # if are_equal_under_sympy's structural match does not.
    score = verify_math_worker(["\\boxed{\\frac{1}{2}}"], ["\\boxed{0.5}"])
    self.assertEqual(score, 1.0)


class AreEqualUnderSympyTest(unittest.TestCase):
  """Tests for the structural sympy AST equality helper.

  `are_equal_under_sympy` is invoked first inside the worker and short-circuits
  `verify()` when it returns True. Its job is cheap structural equality on
  unevaluated expressions.
  """

  def test_same_integer(self):
    self.assertTrue(are_equal_under_sympy(sympy.Integer(42), sympy.Integer(42)))

  def test_different_integer(self):
    self.assertFalse(are_equal_under_sympy(sympy.Integer(42), sympy.Integer(99)))

  def test_same_symbol(self):
    x = sympy.Symbol("x")
    self.assertTrue(are_equal_under_sympy(x, x))

  def test_malformed_input_does_not_raise(self):
    # Unparsable strings must not propagate an exception; they return False.
    self.assertFalse(are_equal_under_sympy("$$$", "%%%"))


@patch.object(mvp, "_get_pool", _fake_get_pool)
class MathVerifyPoolScoreAssignmentTest(unittest.TestCase):
  """Regression tests for the score-assignment bug.

  Prior version granted `reward_exact_answer` on every completed job, ignoring
  the grader's score. These tests exist to keep that bug from coming back.

  `_get_pool` is patched with an in-process fake so the busy-poll drains on
  the first iteration — no spawn, no 300s global_timeout.
  """

  def test_correct_answer_gets_reward(self):
    items = [(0, ["\\boxed{42}"], ["\\boxed{42}"])]
    scores = [0.0]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], 1.0)

  def test_wrong_answer_does_not_get_reward(self):
    items = [(0, ["\\boxed{42}"], ["\\boxed{99}"])]
    scores = [0.0]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], 0.0)

  def test_wrong_answer_preserves_prior_penalty(self):
    # `check_numbers` seeds scores[idx] with `penalty_incorrect_format`; a
    # wrong grader verdict must not overwrite that with the reward.
    items = [(0, ["\\boxed{42}"], ["\\boxed{99}"])]
    scores = [-0.5]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], -0.5)

  def test_mixed_batch_scores_each_item_independently(self):
    items = [
        (0, ["\\boxed{1}"], ["\\boxed{1}"]),  # correct
        (1, ["\\boxed{2}"], ["\\boxed{99}"]),  # wrong
        (2, ["\\boxed{3}"], ["\\boxed{3}"]),  # correct
    ]
    scores = [0.0, 0.0, 0.0]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], 1.0)
    self.assertEqual(result[1], 0.0)
    self.assertEqual(result[2], 1.0)

  def test_reward_uses_max_not_overwrite(self):
    # A correct answer must not lower an already-higher pre-existing score.
    items = [(0, ["\\boxed{42}"], ["\\boxed{42}"])]
    scores = [0.7]
    result = math_verify_pool(_make_config(0.3), items, scores)
    self.assertEqual(result[0], 0.7)

  def test_empty_items_returns_scores_unchanged(self):
    scores = [0.1, 0.2, 0.3]
    result = math_verify_pool(_make_config(1.0), [], scores)
    self.assertEqual(result, [0.1, 0.2, 0.3])


if __name__ == "__main__":
  unittest.main()
