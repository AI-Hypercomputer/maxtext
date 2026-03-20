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

"""Async HTTP client — thin wrapper around the upstream tpu-inference backend_request_func.

Delegates individual requests to
eval.vllm.backend_request_func.async_request_openai_completions
and fans out concurrently via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from maxtext.eval.vllm.backend_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_completions,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 64
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.0
_COMPLETIONS_PATH = "/v1/completions"


@dataclass
class GenerationResult:
  """Result of a single generation request.

  Attributes:
    text: The generated text (empty string on error).
    prompt_tokens: Number of prompt tokens consumed.
    completion_tokens: Number of completion tokens generated.
    error: Non-empty error message if the request failed.
    latency_s: E2E wall-clock latency in seconds.
  """

  text: str = ""
  prompt_tokens: int = 0
  completion_tokens: int = 0
  error: str = ""
  latency_s: float = 0.0


def _to_generation_result(output: RequestFuncOutput) -> GenerationResult:
  return GenerationResult(
      text=output.generated_text,
      prompt_tokens=output.prompt_len,
      completion_tokens=output.output_tokens or 0,
      error=output.error,
      latency_s=output.latency,
  )


async def generate_batch_async(
    prompts: list[str],
    base_url: str,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[GenerationResult]:
  """Send prompts concurrently via the tpu-inference completions request function.

  Args:
    prompts: List of fully-formatted prompt strings.
    base_url: Base URL of the vLLM server.
    model: Model name as registered with the vLLM server.
    max_tokens: Maximum tokens to generate per response.
    temperature: Sampling temperature.
    concurrency: Maximum number of in-flight requests.

  Returns:
    List of GenerationResult in the order of prompts.
  """
  api_url = f"{base_url}{_COMPLETIONS_PATH}"
  semaphore = asyncio.Semaphore(concurrency)

  async def _generate_single(prompt: str) -> GenerationResult:
    req = RequestFuncInput(
        prompt=prompt,
        api_url=api_url,
        prompt_len=0,
        output_len=max_tokens,
        model=model,
        extra_body={"temperature": temperature},
    )
    async with semaphore:
      output = await async_request_openai_completions(req)
    return _to_generation_result(output)

  return await asyncio.gather(*[_generate_single(p) for p in prompts])


def generate_batch(
    prompts: list[str],
    base_url: str,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    concurrency: int = _DEFAULT_CONCURRENCY,
    request_timeout: int = 600,  # noqa: ARG001 (timeout handled by AIOHTTP_TIMEOUT in tpu-inference)
) -> list[GenerationResult]:
  """Synchronous wrapper around generate_batch_async."""
  return asyncio.run(
      generate_batch_async(
          prompts=prompts,
          base_url=base_url,
          model=model,
          max_tokens=max_tokens,
          temperature=temperature,
          concurrency=concurrency,
      )
  )
