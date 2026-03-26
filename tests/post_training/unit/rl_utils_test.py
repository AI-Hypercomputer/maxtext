# Copyright 2026 Google LLC
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

"""Unit tests for RL result parsing and reward scoring (CPU-only)."""

import unittest
import pytest
from types import SimpleNamespace

pytestmark = [pytest.mark.post_training]

evaluate_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.evaluate_rl",
    reason="tunix (required by evaluate_rl) is not installed GPU",
)

utils_rl = pytest.importorskip(
    "maxtext.trainers.post_train.rl.utils_rl",
    reason="tunix (required by utils_rl) is not installed GPU",
)


def _make_config():
  """Create a minimal config object with the parameters required by score_responses."""
  return SimpleNamespace(
      reasoning_start_token="<reasoning>",
      reasoning_end_token="</reasoning>",
      solution_start_token="<answer>",
      solution_end_token="</answer>",
      reward_exact_answer=3.0,
      reward_exact_format_match=2.0,
      reward_partial_format_match=0.5,
      reward_white_space_format_match=1.5,
      reward_ratio_guess_to_answer_high=1.0,
      reward_ratio_guess_to_answer_low=0.5,
      penalty_incorrect_format=-0.5,
      penalty_incorrect_answer=-0.5,
      dataset_name="test",
      debug=SimpleNamespace(rl=False),
  )


class TestScoreResponses(unittest.TestCase):
  """Tests for evaluate_rl.score_responses parsing and correctness logic."""

  def setUp(self):
    self.config = _make_config()

  @pytest.mark.cpu_only
  def test_nested_tags(self):
    """Response with nested reasoning tags still extracts the correct answer."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question="What is 11/3?",
        responses=[
            "<reasoning>Need to use <reasoning> and </reasoning>, "
            "<answer> and </answer></reasoning><answer>11/3</answer>"
        ],
        answers=["11/3"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  @pytest.mark.cpu_only
  def test_with_extra_ending_tags(self):
    """Answer with extra ending tags such as <end_of_turn>."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question=(
            "James buys a new wardrobe.  He buys 10 suits and 10 dress pants.  "
            "He also buys 3 dress shirts per suit.  The suits cost $750 each and "
            "the dress pants cost 1/5 that cost.  The dress shirts were $60 each.  "
            "How much did everything cost?"
        ),
        responses=[
            "<reasoning>This is the sum of the cost of the suits, the pants, and the "
            "shirts: $7500 + $1500 + $1800 = $10800.\n\n</reasoning>\n"
            "<answer>10800</answer><end_of_turn>"
        ],
        answers=["10,800"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertTrue(has_correct_format)

  @pytest.mark.cpu_only
  def test_with_incomplete_reasoning_tags(self):
    """(1) Incomplete reasoning tags still extracts the correct answer."""
    """(2) Currency symbols works with math_verify."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question="What is the price of the item?",
        responses=["<reasoning>The item costs $16.<answer>$16</answer>"],
        answers=["16"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertFalse(has_correct_format)

  @pytest.mark.cpu_only
  def test_for_mcq_value(self):
    """Test for MCQ, where model responds with a math value."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question=(
            r"What is the quantity of the item? "
            r"(A) $2 \frac{1}{3}$ (B) $3 \frac{1}{3}$ "
            r"(C) $1 \frac{2}{3}$ (D) $1 \frac{1}{3}$ (E) 2"
        ),
        responses=["<reasoning>The answer is 3\frac{1}{3}.<answer>3\frac{1}{3}</answer>"],
        answers=["3\frac{1}{3}", "B"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertFalse(has_correct_format)

  @pytest.mark.cpu_only
  def test_for_mcq_option(self):
    """Test for MCQ, where model responds with an option."""
    is_correct, is_partially_correct, has_correct_format = evaluate_rl.score_responses(
        tmvp_config=self.config,
        question=(
            r"What is the quantity of the item? "
            r"(A) $2 \frac{1}{3}$ (B) $3 \frac{1}{3}$ "
            r"(C) $1 \frac{2}{3}$ (D) $1 \frac{1}{3}$ (E) 2"
        ),
        responses=["<reasoning>The answer is B.<answer>B</answer>"],
        answers=["3\frac{1}{3}", "B"],
    )
    self.assertTrue(is_correct)
    self.assertTrue(is_partially_correct)
    self.assertFalse(has_correct_format)


class TestProcessAnswer(unittest.TestCase):
  """Tests for utils_rl.process_answer."""

  @pytest.mark.cpu_only
  def test_for_mcq(self):
    self.assertEqual(len(utils_rl.process_answer("(A) 1\n(B) 2\n(C) 3\n", "B", "MCQ")), 2)
    self.assertEqual(len(utils_rl.process_answer("A. 1\nB. 2\n(C) 3\n", "B", "MCQ")), 2)
    self.assertEqual(
        len(utils_rl.process_answer("$\\textbf{(A)}~\\frac{1}{24}\\qquad\\textbf{(B)}~\\frac{1}{12}$", "B", "MCQ")), 2
    )
    self.assertEqual(len(utils_rl.process_answer("$(\\mathrm {A}) \\ 1 \\qquad (\\mathrm {B}) \\ 2$", "B", "MCQ")), 2)


class TestMathVerifyFunc(unittest.TestCase):
  """Tests for utils_rl.verify_math."""

  @pytest.mark.cpu_only
  def test_support_for_relational(self):
    self.assertEqual(utils_rl.verify_math(["$1 < x < 2$"], ["$(1,2)$"]), 1.0)

  @pytest.mark.cpu_only
  def test_ordered_tuples(self):
    self.assertEqual(utils_rl.verify_math(["${1,2,3}$"], ["${3,2,1}$"]), 1.0)
    self.assertEqual(utils_rl.verify_math(["$1,2,3$"], ["${3,2,1}$"]), 1.0)
    self.assertEqual(utils_rl.verify_math(["$(1,2,3)$"], ["${3,2,1}$"]), 0.0)

  @pytest.mark.cpu_only
  def test_multiple_boxed(self):
    self.assertEqual(utils_rl.verify_math(["$\\boxed{1},\\boxed{2}$"], ["${1,2}$"]), 1.0)

  @pytest.mark.cpu_only
  def test_list_of_numbers(self):
    self.assertEqual(utils_rl.verify_math(["$1$ and $2$ and $3$"], ["$1,2,3$"]), 1.0)

  @pytest.mark.cpu_only
  def test_string_extraction(self):
    self.assertEqual(utils_rl.verify_math(["$B$"], ["$B$"]), 1.0)


class TestNormalizeFinalAnswer(unittest.TestCase):
  """Tests for utils_rl.normalize_final_answer."""

  @pytest.mark.cpu_only
  def test_comma_boxed_and_currency(self):
    # Comma-separated numbers, \\boxed{}, and leading $ are all normalized to plain integers
    self.assertEqual(utils_rl.normalize_final_answer("1,000"), "1000")
    self.assertEqual(utils_rl.normalize_final_answer("$1,000"), "1000")
    self.assertEqual(utils_rl.normalize_final_answer("\\boxed{1,000}"), "1000")

  @pytest.mark.cpu_only
  def test_equation_splitting_and_unit_removal(self):
    # Expressions with '=' are split on '='; trailing unit words are stripped
    self.assertEqual(utils_rl.normalize_final_answer("x = 10"), "10")
    self.assertEqual(utils_rl.normalize_final_answer("total = 100 meters"), "100")
    self.assertEqual(utils_rl.normalize_final_answer("42 mph"), "42")

  @pytest.mark.cpu_only
  def test_latex_wrappers(self):
    # \\text{}, \\textbf{}, and \\overline{} wrappers are removed, leaving inner content
    self.assertEqual(utils_rl.normalize_final_answer("\\text{hello}"), "hello")
    self.assertEqual(utils_rl.normalize_final_answer("\\textbf{42}"), "42")
    self.assertEqual(utils_rl.normalize_final_answer("\\overline{AB}"), "AB")

  @pytest.mark.cpu_only
  def test_dollar_math_extraction(self):
    # Content inside $...$ is extracted
    self.assertEqual(utils_rl.normalize_final_answer("The answer is $\\frac{1}{2}$"), "\\frac{1}{2}")
    self.assertEqual(utils_rl.normalize_final_answer("The answer is 3 $\\frac{1}{2}$"), "3+\\frac{1}{2}")

  @pytest.mark.cpu_only
  def test_shorthand_frac_and_sqrt(self):
    # Shorthand \\fracab and \\sqrta are expanded to their full LaTeX forms
    self.assertEqual(utils_rl.normalize_final_answer("\\fracab"), "\\frac{a}{b}")
    self.assertEqual(utils_rl.normalize_final_answer("\\sqrta"), "\\sqrt{a}")


class TestMatchFormatApproximatelyScores(unittest.TestCase):
  """Tests for utils_rl.match_format_approximately.

  Each tag that appears exactly once contributes reward_partial_format_match (0.5).
  Each tag that is absent or appears more than once contributes penalty_incorrect_format (-0.5).
  With 4 tags the score ranges from -2.0 (all wrong) to 2.0 (all correct).
  """

  def setUp(self):
    self.config = _make_config()

  def _score(self, completion):
    return utils_rl.match_format_approximately(None, completion, self.config)

  @pytest.mark.cpu_only
  def test_score_all_tags_present_exactly_once(self):
    # All four tags present exactly once -> 4 * 0.5 = 2.0
    self.assertEqual(self._score(["<reasoning>think</reasoning><answer>42</answer>"])[0], 2.0)

  @pytest.mark.cpu_only
  def test_score_no_tags_present(self):
    # No tags at all -> 4 * -0.5 = -2.0
    self.assertEqual(self._score(["The answer is 42."])[0], -2.0)

  @pytest.mark.cpu_only
  def test_score_only_answer_tags_present(self):
    # Only <answer>...</answer> present -> 2 * 0.5 + 2 * -0.5 = 0.0
    self.assertEqual(self._score(["<answer>42</answer>"])[0], 0.0)

  @pytest.mark.cpu_only
  def test_score_duplicate_reasoning_start_tag(self):
    # Duplicate <reasoning> tag -> 3 * 0.5 + 1 * -0.5 = 1.0
    self.assertEqual(self._score(["<reasoning><reasoning>think</reasoning><answer>42</answer>"])[0], 1.0)

  @pytest.mark.cpu_only
  def test_score_multiple_completions(self):
    # Multiple completions at once -> one score per entry
    multi_completions = [
        "<reasoning>think</reasoning><answer>42</answer>",  # 2.0
        "no tags here",  # -2.0
    ]
    scores = self._score(multi_completions)
    self.assertEqual(len(scores), 2)
    self.assertEqual(scores[0], 2.0)
    self.assertEqual(scores[1], -2.0)


class TestCheckNumbers(unittest.TestCase):
  """Tests for utils_rl.check_numbers.

  Covers two scenarios:
    1. Whether the regex can extract an answer from the completion.
    2. Whether the extracted value matches (or does not match) the reference answer.
  """

  def setUp(self):
    self.config = _make_config()

  def _check(self, completions, answer):
    return utils_rl.check_numbers(
        prompts=None,
        completions=completions,
        answer=answer,
        tmvp_config=self.config,
        question=["test question"] * len(completions),
    )

  # ---------------------------------------------------------------
  # Scenario 1: regex extraction succeeds / fails
  # ---------------------------------------------------------------

  @pytest.mark.cpu_only
  def test_extraction_succeeds_full_format(self):
    """Full <reasoning>…</reasoning><answer>…</answer> format allows extraction."""
    scores = self._check(
        completions=["<reasoning>40 + 2 = 42</reasoning><answer>42</answer>"],
        answer=[["42"]],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  @pytest.mark.cpu_only
  def test_extraction_fails_no_tags(self):
    """Plain-text completion without any tags yields score 0 (cannot extract)."""
    scores = self._check(
        completions=["The answer is 42."],
        answer=[["42"]],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_format)

  @pytest.mark.cpu_only
  def test_extraction_fails_answer_tags_only(self):
    """<answer> tag alone (no <reasoning> block) is matched by the regex as a fallback, score 1.5."""
    scores = self._check(
        completions=["<answer>42</answer>"],
        answer=[["42"]],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  @pytest.mark.cpu_only
  def test_extraction_fails_reasoning_tags_only(self):
    """<reasoning> block with no <answer> tag cannot be extracted, score 0."""
    scores = self._check(
        completions=["<reasoning>The answer is 42.</reasoning>"],
        answer=[["42"]],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_format)

  @pytest.mark.cpu_only
  def test_extraction_batch_mixed(self):
    """Batch with one extractable and one non-extractable completion."""
    scores = self._check(
        completions=[
            "<reasoning>thinking</reasoning><answer>7</answer>",  # extractable
            "just 7",  # not extractable
        ],
        answer=[["7"], ["7"]],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)
    self.assertEqual(scores[1], self.config.penalty_incorrect_format)

  @pytest.mark.cpu_only
  def test_extraction_for_mcq(self):
    """Batch with two multiple-choice questions and one single-answer question."""
    scores = self._check(
        completions=[
            "<reasoning>thinking</reasoning><answer>7</answer>",
            "<reasoning>thinking</reasoning><answer>A</answer>",
            "<reasoning>thinking</reasoning><answer>A</answer>",
        ],
        answer=[["7", "B"], ["7", "A"], ["7", "7"]],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)
    self.assertEqual(scores[1], self.config.reward_exact_answer)
    self.assertEqual(scores[2], self.config.penalty_incorrect_answer)  # extracted "A" does not match "7"

  # ---------------------------------------------------------------
  # Scenario 2: extraction succeeds, value matches/mismatches the answer
  # ---------------------------------------------------------------

  @pytest.mark.cpu_only
  def test_extracted_matches_integer_answer(self):
    """Extracted integer equal to reference answer earns 1.5."""
    scores = self._check(
        completions=["<reasoning>simple</reasoning><answer>100</answer>"],
        answer=[["100"]],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  @pytest.mark.cpu_only
  def test_extracted_does_not_match_answer(self):
    """Extracted number that differs from the reference answer earns 0.0."""
    scores = self._check(
        completions=["<reasoning>wrong path</reasoning><answer>99</answer>"],
        answer=[["42"]],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_answer)

  @pytest.mark.cpu_only
  def test_extracted_matches_comma_formatted_number(self):
    """Comma-formatted guess (e.g. '1,000') normalizes to match integer answer '1000'."""
    scores = self._check(
        completions=["<reasoning>cost calculation</reasoning><answer>1,000</answer>"],
        answer=[["1000"]],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  @pytest.mark.cpu_only
  def test_extracted_matches_with_currency_prefix(self):
    """Leading '$' in extracted answer is normalized away before comparison."""
    scores = self._check(
        completions=["<reasoning>price is $16</reasoning><answer>$16</answer>"],
        answer=[["16"]],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  @pytest.mark.cpu_only
  def test_extracted_non_numeric_no_match(self):
    """Non-numeric extraction that cannot be float-converted and does not math-verify returns 0."""
    scores = self._check(
        completions=["<reasoning>thinking</reasoning><answer>blue</answer>"],
        answer=[["red"]],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_format)


class TestExtractHashAnswer(unittest.TestCase):
  """Tests for utils_rl.extract_hash_answer."""

  @pytest.mark.cpu_only
  def test_with_hash(self):
    """Test extraction when #### is present."""
    self.assertEqual(utils_rl.extract_hash_answer("The answer is #### 42"), "42")
    self.assertEqual(utils_rl.extract_hash_answer("Some reasoning ####   123.45  "), "123.45")
    self.assertEqual(utils_rl.extract_hash_answer("####"), "")

  @pytest.mark.cpu_only
  def test_without_hash(self):
    """Test extraction when #### is not present."""
    self.assertIsNone(utils_rl.extract_hash_answer("The answer is 42"))
    self.assertIsNone(utils_rl.extract_hash_answer(""))


class TestGetOptimizer(unittest.TestCase):
  """Tests for utils_rl.get_optimizer."""

  def _make_optimizer_config(self, gradient_clipping_threshold=0.0):
    return SimpleNamespace(
        learning_rate=1e-4,
        warmup_steps_fraction=0.1,
        gradient_clipping_threshold=gradient_clipping_threshold,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_weight_decay=0.01,
    )

  @pytest.mark.cpu_only
  def test_returns_optimizer_without_clipping(self):
    """get_optimizer returns an optax optimizer when gradient clipping is disabled."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    config = self._make_optimizer_config(gradient_clipping_threshold=0.0)
    opt = utils_rl.get_optimizer(config, max_train_steps=100)
    # Should be usable: init on a simple param tree
    params = {"w": jnp.ones(3)}
    state = opt.init(params)
    self.assertIn("learning_rate", state.hyperparams)

  @pytest.mark.cpu_only
  def test_returns_optimizer_with_clipping(self):
    """get_optimizer includes gradient clipping when threshold > 0."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    config = self._make_optimizer_config(gradient_clipping_threshold=1.0)
    opt = utils_rl.get_optimizer(config, max_train_steps=100)
    params = {"w": jnp.ones(3)}
    state = opt.init(params)
    self.assertIn("learning_rate", state.hyperparams)


if __name__ == "__main__":
  unittest.main()
