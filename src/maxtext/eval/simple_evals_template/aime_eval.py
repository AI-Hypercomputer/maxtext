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

"""AIME 2024 and 2025 simple-evals tasks.

Run via the simple_evals runner:
  python -m maxtext.eval.runner.run \
      --runner simple_evals \
      --tasks aime2024|aime2025 \
      --checkpoint_path <checkpoint_path> \
      --model_name <model_name> \
      --hf_path <hf_path> \
      --base_output_directory <output_dir> \
      --run_name <run_name> \
      --max_model_len <max_model_len> \
      --tensor_parallel_size <tensor_parallel_size> \
      --hf_token "$HF_TOKEN"
"""

from __future__ import annotations

import random
import re

import pandas

from maxtext.eval.simple_evals_template import common
from maxtext.eval.simple_evals_template.common import HTML_JINJA
from maxtext.eval.simple_evals_template.types import Eval, EvalResult, SamplerBase, SingleEvalResult

# SamplerBase intentionally preserves the vendored simple-evals API.
# pylint: disable=protected-access

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
    "Think step by step, then write the final answer by itself on the last line in the format "
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
    try:
      from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel
      # Map raw URL to hf_hub_download parameters
      if "AIME_2024" in url:
        parquet_path = hf_hub_download(
            repo_id="Maxwell-Jia/AIME_2024", repo_type="dataset", filename="default/train/0000.parquet"
        )
      else:
        parquet_path = hf_hub_download(
            repo_id="math-ai/aime25", repo_type="dataset", filename="default/test/0000.parquet"
        )
    except Exception as exc:
      raise RuntimeError(f"Could not download AIME dataset: {exc}") from exc
    df = pandas.read_parquet(parquet_path)
    examples = [{"problem": row[problem_col], "answer": str(int(row[answer_col]))} for _, row in df.iterrows()]
    rng = random.Random(0)
    if num_examples:
      assert n_repeats == 1, "n_repeats only supported for num_examples = None"
      examples = rng.sample(examples, num_examples)
    self.examples = examples * n_repeats
    self.year = year

  def __call__(self, sampler: SamplerBase) -> EvalResult:
    def fn(row: dict):
      prompt_messages = [sampler._pack_message(content=QUERY_TEMPLATE.format(problem=row["problem"]), role="user")]
      sampler_response = sampler(prompt_messages)
      response_text = sampler_response.response_text
      actual_queried_prompt_messages = sampler_response.actual_queried_message_list
      extracted_answer = extract_answer(response_text) if common.request_succeeded(sampler_response) else None
      score = 1.0 if extracted_answer is not None and int(extracted_answer) == int(row["answer"]) else 0.0
      html = common.jinja_env.from_string(HTML_JINJA).render(
          prompt_messages=actual_queried_prompt_messages,
          next_message={"content": response_text, "role": "assistant"},
          score=score,
          correct_answer=row["answer"],
          extracted_answer=extracted_answer,
      )
      convo = actual_queried_prompt_messages + [{"content": response_text, "role": "assistant"}]
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
