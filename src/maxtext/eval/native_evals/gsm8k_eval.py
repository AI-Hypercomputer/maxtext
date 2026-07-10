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

"""GSM8K: Grade School Math 8K.

Cobbe et al., "Training Verifiers to Solve Math Word Problems"
https://arxiv.org/abs/2110.14168

Not part of upstream openai/simple-evals; authored for MaxText on top of the
vendored simple_evals Eval/SamplerBase conventions in
maxtext.eval.third_party.simple_evals, reusing MGSM's English answer
extraction and scoring (same "reason then 'Answer: N'" format, same grading).
"""

from __future__ import annotations

import random
import re

import pandas

from maxtext.eval.third_party.simple_evals import common
from maxtext.eval.third_party.simple_evals.common import HTML_JINJA
from maxtext.eval.third_party.simple_evals.mgsm_eval import score_mgsm
from maxtext.eval.third_party.simple_evals.types import Eval, EvalResult, SamplerBase, SingleEvalResult

# HF-hosted auto-converted parquet for the canonical GSM8K test split (1319 examples).
DATASET_URL = (
    "https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet"
)

QUERY_TEMPLATE = (
    "Solve this math problem. Give the reasoning steps before giving the final answer on the last line "
    'by itself in the format of "Answer:". Do not add anything other than the integer answer after '
    '"Answer:".\n\n{question}'
)

_ANSWER_LINE_RE = re.compile(r"(?i)(?:^|\n)\s*Answer\s*:\s*(-?[\d,]+(?:\.\d+)?)\s*\Z")


def extract_answer(response_text: str) -> str | None:
  """Extract the last answer that obeys GSM8K's requested final-line format."""
  match = _ANSWER_LINE_RE.search(response_text)
  return match.group(1).replace(",", "") if match else None


class GSM8KEval(Eval):
  """GSM8K eval: zero-shot CoT prompt, gold answer parsed from the '#### N' suffix."""

  def __init__(self, num_examples: int | None = None):
    df = pandas.read_parquet(DATASET_URL)
    examples = [row.to_dict() for _, row in df.iterrows()]
    for example in examples:
      example["target"] = example["answer"].split("####")[-1].strip().replace(",", "")
    if num_examples:
      examples = random.Random(0).sample(examples, num_examples)
    self.examples = examples

  def __call__(self, sampler: SamplerBase) -> EvalResult:
    def fn(row: dict):
      prompt_messages = [
          sampler._pack_message(content=QUERY_TEMPLATE.format(question=row["question"]), role="user")
      ]
      sampler_response = sampler(prompt_messages)
      response_text = sampler_response.response_text
      actual_queried_prompt_messages = sampler_response.actual_queried_message_list
      extracted_answer = extract_answer(response_text)
      score = score_mgsm(row["target"], extracted_answer or "")
      html = common.jinja_env.from_string(HTML_JINJA).render(
          prompt_messages=actual_queried_prompt_messages,
          next_message=dict(content=response_text, role="assistant"),
          score=score,
          correct_answer=row["target"],
          extracted_answer=extracted_answer or None,
      )
      convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
      return SingleEvalResult(
          html=html,
          score=score,
          convo=convo,
          example_level_metadata={
              "request_id": sampler_response.response_metadata.get("request_id"),
              "request_status": sampler_response.response_metadata.get("status", "success"),
              "score": score,
              "correct_answer": row["target"],
              "extracted_answer": extracted_answer or None,
          },
      )

    results = common.map_with_progress(fn, self.examples)
    return common.aggregate_results(results)
