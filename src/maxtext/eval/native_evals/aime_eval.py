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

"""AIME: American Invitational Mathematics Examination.

Competition math problems (AIME I + II) whose answers are always integers in
[0, 999], used here as a high-difficulty math reasoning benchmark for both the
2024 and 2025 contests.

Not part of upstream openai/simple-evals; authored for MaxText on top of the
vendored simple_evals Eval/SamplerBase conventions in
maxtext.eval.third_party.simple_evals. Answers are plain integers, so unlike
MATH this needs no LLM equality grader.
"""

from __future__ import annotations

import random
import re

import pandas

from maxtext.eval.third_party.simple_evals import common
from maxtext.eval.third_party.simple_evals.common import HTML_JINJA
from maxtext.eval.third_party.simple_evals.types import Eval, EvalResult, SamplerBase, SingleEvalResult

# (parquet_url, problem_column, answer_column) per contest year. Both are the
# full 30-problem set (AIME I + II) with a plain integer answer column, so no
# boxed-solution parsing is needed to recover gold labels.
_AIME_2024_URL = (
    "https://huggingface.co/datasets/Maxwell-Jia/AIME_2024/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
_AIME_2025_URL = (
    "https://huggingface.co/datasets/math-ai/aime25/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)

_DATASET_BY_YEAR = {
    2024: (_AIME_2024_URL, "Problem", "Answer"),
    2025: (_AIME_2025_URL, "problem", "answer"),
}

QUERY_TEMPLATE = (
    "Solve this competition math problem. The final answer is always an integer between 0 and 999. "
    'Think step by step, then write the final answer by itself on the last line in the format '
    '"Answer: $N" (without quotes), where $N is the integer answer.\n\n{problem}'
)

_ANSWER_LINE_RE = re.compile(r"(?i)(?:^|\n)\s*Answer\s*:\s*\$?([0-9]{1,3})\$?\s*\Z")


def extract_answer(response_text: str) -> str | None:
  """Pull the integer answer out of a response.

  Requires the explicit final-line format requested by QUERY_TEMPLATE.
  """
  match = _ANSWER_LINE_RE.search(response_text)
  return match.group(1) if match else None


class AIMEEval(Eval):
  """AIME eval for a single contest year (2024 or 2025)."""

  def __init__(self, year: int, num_examples: int | None = None, n_repeats: int = 1):
    if year not in _DATASET_BY_YEAR:
      raise ValueError(f"Unsupported AIME year: {year}. Supported: {sorted(_DATASET_BY_YEAR)}.")
    url, problem_col, answer_col = _DATASET_BY_YEAR[year]
    df = pandas.read_parquet(url)
    examples = [{"problem": row[problem_col], "answer": str(int(row[answer_col]))} for _, row in df.iterrows()]
    rng = random.Random(0)
    if num_examples:
      assert n_repeats == 1, "n_repeats only supported for num_examples = None"
      examples = rng.sample(examples, num_examples)
    self.examples = examples * n_repeats
    self.year = year

  def __call__(self, sampler: SamplerBase) -> EvalResult:
    def fn(row: dict):
      prompt_messages = [
          sampler._pack_message(content=QUERY_TEMPLATE.format(problem=row["problem"]), role="user")
      ]
      sampler_response = sampler(prompt_messages)
      response_text = sampler_response.response_text
      actual_queried_prompt_messages = sampler_response.actual_queried_message_list
      extracted_answer = extract_answer(response_text)
      score = 1.0 if extracted_answer is not None and int(extracted_answer) == int(row["answer"]) else 0.0
      html = common.jinja_env.from_string(HTML_JINJA).render(
          prompt_messages=actual_queried_prompt_messages,
          next_message=dict(content=response_text, role="assistant"),
          score=score,
          correct_answer=row["answer"],
          extracted_answer=extracted_answer,
      )
      convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
      return SingleEvalResult(
          html=html,
          score=score,
          convo=convo,
          metrics={"chars": len(response_text)},
          example_level_metadata={
              "request_id": sampler_response.response_metadata.get("request_id"),
              "request_status": sampler_response.response_metadata.get("status", "success"),
              "score": score,
              "correct_answer": row["answer"],
              "extracted_answer": extracted_answer,
          },
      )

    results = common.map_with_progress(fn, self.examples)
    return common.aggregate_results(results)
