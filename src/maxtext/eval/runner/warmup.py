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

"""Server warmup — delegates bucket-sampled warmup to the tpu-inference benchmark_utils.

Uses eval.vllm.benchmark_utils.sample_warmup_requests.
"""

from __future__ import annotations

import asyncio
import logging

from maxtext.eval.vllm.backend_request_func import (
    RequestFuncInput,
    async_request_openai_completions,
)
from maxtext.eval.vllm.benchmark_dataset import SampleRequest
from maxtext.eval.vllm.benchmark_utils import sample_warmup_requests

logger = logging.getLogger(__name__)

_COMPLETIONS_PATH = "/v1/completions"
_SIMPLE_WARMUP_PROMPT = "What is 1 + 1?"


async def _send_warmup_request(api_url: str, model: str, prompt: str, max_tokens: int) -> bool:
  req = RequestFuncInput(
      prompt=prompt,
      api_url=api_url,
      prompt_len=0,
      output_len=max_tokens,
      model=model,
  )
  output = await async_request_openai_completions(req)
  return output.success


async def _run_warmup(api_url: str, model: str, requests: list[tuple[str, int]]) -> None:
  tasks = [_send_warmup_request(api_url, model, p, n) for p, n in requests]
  results = await asyncio.gather(*tasks, return_exceptions=True)
  errors = [r for r in results if isinstance(r, Exception) or r is False]
  if errors:
    logger.warning("Warmup: %d/%d requests failed.", len(errors), len(results))
  else:
    logger.info("Warmup complete (%d requests).", len(results))


def warmup_server(
    base_url: str,
    model: str,
    sample_requests: list[SampleRequest] | None = None,
    max_tokens: int = 16,
) -> None:
  """Send warmup requests to trigger XLA compilation before eval.

  Args:
    base_url: Base URL of the vLLM server.
    model: Model name as registered with the vLLM server.
    sample_requests: Optional list of SampleRequest from the dataset.
    max_tokens: Max tokens per warmup response.
  """
  api_url = f"{base_url}{_COMPLETIONS_PATH}"

  if sample_requests:
    bucketed = list(sample_warmup_requests(sample_requests))
    logger.info("Running warmup: %d prompt-length buckets.", len(bucketed))
    warmup_pairs = [(r.prompt, max_tokens) for r in bucketed]
  else:
    logger.info("Running simple warmup.")
    warmup_pairs = [(_SIMPLE_WARMUP_PROMPT, max_tokens)]

  asyncio.run(_run_warmup(api_url, model, warmup_pairs))
