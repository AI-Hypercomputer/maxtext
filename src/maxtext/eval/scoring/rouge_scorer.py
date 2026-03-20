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

"""ROUGE scorer — thin wrapper around the tpu-inference benchmark_utils implementation.

Delegates to eval.vllm.benchmark_utils.eval_accuracy_mlperf.
"""

from __future__ import annotations

from maxtext.eval.vllm.backend_request_func import RequestFuncInput, RequestFuncOutput
from maxtext.eval.vllm.benchmark_utils import eval_accuracy_mlperf


def _make_outputs(responses: list[str], references: list[str]) -> list[RequestFuncOutput]:
  """Wrap plain string pairs into RequestFuncOutput objects for the upstream scorer."""
  outputs = []
  for resp, ref in zip(responses, references):
    inp = RequestFuncInput(
        prompt="", api_url="", prompt_len=0, output_len=0, model="",
        completion=ref,
    )
    outputs.append(RequestFuncOutput(generated_text=resp, success=True, input_request=inp))
  return outputs


def score_batch(
    responses: list[str],
    references: list[str],
    use_stemmer: bool = True,  # noqa: ARG001
) -> dict:
  """Compute ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.

  Args:
    responses: List of model-generated summaries.
    references: List of reference summaries (same length).
    use_stemmer: Accepted for API consistency, stemming is handled internally
      by the upstream library.

  Returns:
    Dict with keys: rouge1, rouge2, rougeL, rougeLsum, gen_num.
  """
  if len(responses) != len(references):
    raise ValueError(f"Length mismatch: {len(responses)} vs {len(references)}")
  outputs = _make_outputs(responses, references)
  return eval_accuracy_mlperf(outputs)
