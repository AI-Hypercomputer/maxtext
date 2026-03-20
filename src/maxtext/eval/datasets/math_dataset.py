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

"""Math reasoning benchmark datasets (MATH-500, GSM8K)."""

from __future__ import annotations

from maxtext.eval.datasets.base import BenchmarkDataset, SampleRequest

_MATH_SYSTEM_PROMPT = (
    "You are a math expert. Solve the following problem step by step. "
    "Wrap your reasoning in <think>...</think> tags and your final answer "
    "in <answer>...</answer> tags."
)

_GSM8K_SYSTEM_PROMPT = (
    "You are a math expert. Solve the following problem step by step. "
    "Wrap your reasoning in <think>...</think> tags and your final answer "
    "in <answer>...</answer> tags."
)


class MathDataset(BenchmarkDataset):
  """MATH-500 — 500-problem subset of the MATH benchmark.

  Uses lighteval/MATH-Hard or hendrycks/competition_math filtered
  to the standard 500-problem test set via the lighteval/MATH HF dataset.
  """
  name = "math"

  def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
    # pylint: disable=import-outside-toplevel
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("lighteval/MATH", "all", split="test")
    if num_samples is not None:
      ds = ds.select(range(min(num_samples, len(ds))))

    requests = []
    for row in ds:
      problem = row["problem"]
      solution = row["solution"]

      if tokenizer is not None:
        messages = [
            {"role": "system", "content": _MATH_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        prompt = f"{_MATH_SYSTEM_PROMPT}\n\n{problem}"

      requests.append(
          SampleRequest(
              prompt=prompt,
              reference=solution,
              metadata={"level": row.get("level", ""), "type": row.get("type", "")},
          )
      )

    return requests


class Gsm8kDataset(BenchmarkDataset):
  """GSM8K — Grade School Math 8K.

  Uses openai/gsm8k HuggingFace dataset, main config, test split.
  """
  name = "gsm8k"

  def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
    # pylint: disable=import-outside-toplevel
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("openai/gsm8k", "main", split="test")
    if num_samples is not None:
      ds = ds.select(range(min(num_samples, len(ds))))

    requests = []
    for row in ds:
      question = row["question"]
      answer_raw = row["answer"]
      # Extract the numeric answer after "####" (GSM8K answer format: "... #### <number>")
      reference = answer_raw.split("####")[-1].strip() if "####" in answer_raw else answer_raw

      if tokenizer is not None:
        messages = [
            {"role": "system", "content": _GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        prompt = f"{_GSM8K_SYSTEM_PROMPT}\n\n{question}"

      requests.append(SampleRequest(prompt=prompt, reference=reference))

    return requests
