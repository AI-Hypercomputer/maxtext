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

# Canonical GSM8K main/test split (1,319 examples). Do not read the Hub's
# implementation-detail `refs/convert/parquet` URL directly: it can redirect
# to restricted object storage and fails with HTTP 403 on TPU hosts. The Hub
# client handles authentication, redirects, Xet storage, and local caching.
_DATASET_REPO_ID = "openai/gsm8k"
_TEST_PARQUET_FILENAME = "main/test-00000-of-00001.parquet"

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


def _load_gsm8k_examples() -> list[dict]:
  """Download/cache the canonical GSM8K test parquet through HF Hub."""
  try:
    from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError(
        "GSM8K evaluation requires huggingface_hub. Install it with `pip install huggingface_hub`."
    ) from exc

  try:
    parquet_path = hf_hub_download(
        repo_id=_DATASET_REPO_ID,
        repo_type="dataset",
        filename=_TEST_PARQUET_FILENAME,
    )
  except Exception as exc:  # pylint: disable=broad-except
    raise RuntimeError(
        "Could not download the canonical GSM8K main/test split from Hugging Face. "
        "Check outbound Hub access and HF_TOKEN for any proxy policy."
    ) from exc

  df = pandas.read_parquet(parquet_path)
  examples = [row.to_dict() for _, row in df.iterrows()]
  if len(examples) != 1319:
    raise ValueError(f"Expected 1319 GSM8K main/test examples, found {len(examples)}.")
  return examples


class GSM8KEval(Eval):
  """GSM8K eval: zero-shot CoT prompt, gold answer parsed from the '#### N' suffix."""

  def __init__(self, num_examples: int | None = None):
    examples = _load_gsm8k_examples()
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
