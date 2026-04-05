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

"""ROUGE scorer for MLPerf OpenOrca benchmark."""

from __future__ import annotations

import numpy as np


def score_batch(
    responses: list[str],
    references: list[str],
    use_stemmer: bool = True,  # noqa: ARG001 (API consistency)
) -> dict:
  """Compute ROUGE scores for a batch of generated responses.

  Args:
    responses: List of model-generated summaries.
    references: List of reference summaries.
    use_stemmer: Accepted for API consistency (handled by the evaluate library).

  Returns:
    Dict with keys: rouge1, rouge2, rougeL, rougeLsum, gen_num.

  Raises:
    ValueError: If responses and references have different lengths.
  """
  if len(responses) != len(references):
    raise ValueError(
        f"Length mismatch: {len(responses)} responses vs {len(references)} references."
    )

  import evaluate  # pylint: disable=import-outside-toplevel
  import nltk  # pylint: disable=import-outside-toplevel

  nltk.download("punkt", quiet=True)
  nltk.download("punkt_tab", quiet=True)
  metric = evaluate.load("rouge")

  preds = []
  targets = []
  for resp, ref in zip(responses, references):
    pred = "\n".join(nltk.sent_tokenize(resp.strip()))
    target = "\n".join(nltk.sent_tokenize(ref.strip()))
    preds.append(pred)
    targets.append(target)

  result = metric.compute(predictions=preds, references=targets)
  result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
  result["gen_num"] = len(preds)
  return result
