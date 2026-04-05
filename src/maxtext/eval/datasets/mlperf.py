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

"""MLPerf OpenOrca summarisation dataset."""

from __future__ import annotations

from maxtext.eval.datasets.base import BenchmarkDataset, SampleRequest

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Summarize the following conversation."
)


class MlperfOpenOrcaDataset(BenchmarkDataset):
  """MLPerf OpenOrca — summarisation benchmark used in MLPerf Inference.

  Uses Open-Orca/OpenOrca HuggingFace dataset.
  """

  name = "mlperf_openorca"

  def sample_requests(self, num_samples, tokenizer) -> list[SampleRequest]:
    # pylint: disable=import-outside-toplevel
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

    requests = []
    for row in ds:
      if not row.get("response", "").strip():
        continue

      system_prompt = row.get("system_prompt", _SYSTEM_PROMPT) or _SYSTEM_PROMPT
      question = row["question"]
      reference = row["response"]

      if tokenizer is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        prompt = f"{system_prompt}\n\nUser: {question}\nAssistant:"

      requests.append(SampleRequest(prompt=prompt, reference=reference))

      if num_samples is not None and len(requests) >= num_samples:
        break

    return requests
