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

"""MMLU benchmark dataset (cais/mmlu, all subjects)."""

from __future__ import annotations

from maxtext.eval.datasets.base import BenchmarkDataset, SampleRequest

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following multiple-choice "
    "question by outputting only the letter of the correct answer (A, B, C, or D)."
)

_OPTION_LABELS = ["A", "B", "C", "D"]


def _build_prompt(question: str, choices: list[str]) -> str:
  options = "\n".join(f"{label}. {text}" for label, text in zip(_OPTION_LABELS, choices))
  return f"{question}\n\n{options}\n\nAnswer:"


class MmluDataset(BenchmarkDataset):
  """MMLU — Massive Multitask Language Understanding.

  Uses the cais/mmlu HuggingFace dataset.
  """
  name = "mmlu"

  def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
    # pylint: disable=import-outside-toplevel
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("cais/mmlu", "all", split="test")
    if num_samples is not None:
      ds = ds.select(range(min(num_samples, len(ds))))

    requests = []
    for row in ds:
      question = row["question"]
      choices = row["choices"]
      answer_idx = int(row["answer"])
      reference = _OPTION_LABELS[answer_idx]
      subject = row.get("subject", "")

      if tokenizer is not None:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(question, choices)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        prompt = f"{_SYSTEM_PROMPT}\n\n{_build_prompt(question, choices)}"

      requests.append(SampleRequest(prompt=prompt, reference=reference, metadata={"subject": subject}))

    return requests
