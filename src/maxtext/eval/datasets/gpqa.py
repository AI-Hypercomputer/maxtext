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

"""GPQA Diamond benchmark dataset (Idavidrein/gpqa)."""

from __future__ import annotations

import random

from maxtext.eval.datasets.base import BenchmarkDataset, SampleRequest

_SYSTEM_PROMPT = (
    "You are an expert researcher. Answer the following multiple-choice question "
    "by outputting only the letter of the correct answer (A, B, C, or D)."
)

_OPTION_LABELS = ["A", "B", "C", "D"]


def _build_prompt(question: str, choices: list[str]) -> str:
  options = "\n".join(f"{label}. {text}" for label, text in zip(_OPTION_LABELS, choices))
  return f"{question}\n\n{options}\n\nAnswer:"


class GpqaDataset(BenchmarkDataset):
  """GPQA Diamond"""

  name = "gpqa"

  def __init__(self, config: str = "gpqa_diamond", shuffle_seed: int = 42):
    self._config = config
    self._shuffle_seed = shuffle_seed

  def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
    # pylint: disable=import-outside-toplevel
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("Idavidrein/gpqa", self._config, split="train")
    if num_samples is not None:
      ds = ds.select(range(min(num_samples, len(ds))))

    rng = random.Random(self._shuffle_seed)
    requests = []
    for row in ds:
      correct = row["Correct Answer"]
      distractors = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
      choices = [correct] + distractors
      rng.shuffle(choices)
      correct_label = _OPTION_LABELS[choices.index(correct)]

      question = row["Question"]
      if tokenizer is not None:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(question, choices)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        prompt = f"{_SYSTEM_PROMPT}\n\n{_build_prompt(question, choices)}"

      requests.append(SampleRequest(prompt=prompt, reference=correct_label))

    return requests
