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

"""Math reasoning scorer (Logic from utils_rl).

Evaluates model responses on math reasoning tasks (GSM8K, DAPO, MATH,
OpenMathInstruct, AIME, etc.) using symbolic equality via math_verify.

Metrics:
  accuracy         - fraction of responses with an exactly correct answer
  partial_accuracy - fraction within 10% of the numeric ground truth
  format_accuracy  - fraction that follow the full <reasoning>/<answer> format

Note: this module is intentionally standalone with no dependency on
maxtext.trainers.post_train.rl.  The normalization utilities below mirror those
in utils_rl.py by design. They should not import from the RL module so that the
scoring library can be used without a full RL training environment.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from math_verify import parse
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

EPSILON = 1e-6

_SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

_REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]

_math_verify_func = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def _boxed(x: str) -> str:
  return "\\boxed{" + x + "}" if not x.startswith("\\boxed{") else x


def _fix_latex_escaping(text: str) -> str:
  """Recover LaTeX commands mangled by Python string escape processing."""
  escape_fixes = [
      ("\f", "rac", r"\frac"),
      ("\n", "ewline", r"\newline"),
      ("\n", "e", r"\ne"),
      ("\t", "heta", r"\theta"),
      ("\t", "an", r"\tan"),
      ("\t", "o", r"\to"),
      ("\t", "imes", r"\times"),
  ]
  for escape_char, suffix, replacement in escape_fixes:
    text = text.replace(escape_char + suffix, replacement)
  return text


def _normalize_final_answer(final_answer: str) -> str:
  """Normalize a final answer string for dataset-specific comparison."""
  final_answer = final_answer.split("=")[-1]
  for before, after in _SUBSTITUTIONS:
    final_answer = final_answer.replace(before, after)
  for expr in _REMOVED_EXPRESSIONS:
    final_answer = final_answer.replace(expr, "")
  final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
  final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
  final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
  final_answer = final_answer.replace("$", "")
  if final_answer.replace(",", "").isdigit():
    final_answer = final_answer.replace(",", "")
  return final_answer


def build_format_regexes(
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    solution_start_token: str = "<answer>",
    solution_end_token: str = "</answer>",
) -> tuple[re.Pattern, re.Pattern]:
  """Return (match_format_regex, answer_fallback_regex) for the given tokens.

  match_format_regex  - full structure: <reasoning> </reasoning><answer> </answer>
  answer_fallback_regex - just the answer tag: <answer> </answer>

  These defaults match the token conventions used in MaxText RL training.
  """
  match_format = re.compile(
      rf"{re.escape(reasoning_start_token)}.+{re.escape(reasoning_end_token)}.*?"
      rf"{re.escape(solution_start_token)}(.+?){re.escape(solution_end_token)}",
      flags=re.MULTILINE | re.DOTALL,
  )
  answer_fallback = re.compile(
      rf"{re.escape(solution_start_token)}(.+?){re.escape(solution_end_token)}",
      flags=re.MULTILINE | re.DOTALL,
  )
  return match_format, answer_fallback



class MathScoreResult(NamedTuple):
  is_correct: bool
  is_partially_correct: bool
  has_correct_format: bool


def score_response(
    response: str,
    reference: str,
    match_format_regex: re.Pattern,
    answer_fallback_regex: re.Pattern,
    normalize: bool = False,
) -> MathScoreResult:
  """Score a single generated response against a reference answer.

  Args:
    response: Model-generated text.
    reference: Ground-truth answer string.
    match_format_regex: Compiled regex for full <reasoning><answer> structure.
    answer_fallback_regex: Compiled regex for <answer> tag only (fallback).
    normalize: If True, apply dataset-specific normalization before comparison
      (use for DAPO / OpenMathInstruct-2 style datasets).

  Returns:
    MathScoreResult(is_correct, is_partially_correct, has_correct_format)
  """
  full_match = match_format_regex.search(response)
  if full_match is not None:
    extracted = full_match.group(1)
  else:
    fallback_matches = answer_fallback_regex.findall(response)
    extracted = fallback_matches[-1].strip() if fallback_matches else "-1000000"

  is_correct = False
  is_partially_correct = False
  has_correct_format = full_match is not None

  try:
    norm_ref = _fix_latex_escaping(reference)
    norm_ext = _fix_latex_escaping(extracted)
    if normalize:
      norm_ext = _normalize_final_answer(norm_ext).strip()
      norm_ref = _normalize_final_answer(reference).strip()

    is_correct = _math_verify_func([_boxed(norm_ref)], [_boxed(norm_ext)])[0] > 0.1

    val_ext = parse(_boxed(norm_ext))
    val_ref = parse(_boxed(norm_ref))
    if val_ext and val_ref:
      ratio = (val_ext[0] + EPSILON) / (val_ref[0] + EPSILON)
      is_partially_correct = 0.9 <= ratio <= 1.1

  except (TimeoutException, Exception):  # pylint: disable=broad-exception-caught
    pass

  return MathScoreResult(is_correct, is_partially_correct, has_correct_format)


def score_batch(
    responses: list[str],
    references: list[str],
    *,
    reasoning_start_token: str = "<think>",
    reasoning_end_token: str = "</think>",
    solution_start_token: str = "<answer>",
    solution_end_token: str = "</answer>",
    normalize: bool = False,
) -> dict[str, float]:
  """Score a batch of responses against references.

  Args:
    responses: List of model-generated texts.
    references: List of ground-truth answer strings (same length).
    reasoning_start_token: Opening tag for the reasoning section.
    reasoning_end_token: Closing tag for the reasoning section.
    solution_start_token: Opening tag for the answer section.
    solution_end_token: Closing tag for the answer section.
    normalize: Apply dataset-specific answer normalization (DAPO/OpenMathInstruct).

  Returns:
    Dict with keys: accuracy, partial_accuracy, format_accuracy,
                    num_correct, num_partial_correct, num_correct_format, num_total.
  """
  if len(responses) != len(references):
    raise ValueError(f"responses and references must have the same length, got {len(responses)} vs {len(references)}")

  match_fmt, fallback = build_format_regexes(
      reasoning_start_token, reasoning_end_token, solution_start_token, solution_end_token
  )

  num_correct = num_partial = num_fmt = 0
  for response, reference in zip(responses, references):
    result = score_response(response, reference, match_fmt, fallback, normalize=normalize)
    if result.is_correct:
      num_correct += 1
    if result.is_partially_correct:
      num_partial += 1
    if result.has_correct_format:
      num_fmt += 1

  total = len(responses)
  return {
      "accuracy": num_correct / total if total else 0.0,
      "partial_accuracy": num_partial / total if total else 0.0,
      "format_accuracy": num_fmt / total if total else 0.0,
      "num_correct": num_correct,
      "num_partial_correct": num_partial,
      "num_correct_format": num_fmt,
      "num_total": total,
  }
