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

"""Async HTTP client for the /v1/completions endpoint.

Fans out requests concurrently with a semaphore-bounded asyncio pool and
returns results in prompt order.  Uses aiohttp for non-blocking I/O.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 64
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.0
_COMPLETIONS_PATH = "/v1/completions"
_REQUEST_TIMEOUT_S = 600 # (TODO): Check if this is reasoanable.


@dataclass
class GenerationResult:
  """Result of a single /v1/completions request.

  Attributes:
    text: Generated text (empty string on error).
    prompt_tokens: Tokens consumed by the prompt.
    completion_tokens: Tokens in the generated completion.
    error: Non-empty error message if the request failed.
    latency_s: End-to-end wall-clock latency in seconds.
  """

  text: str = ""
  prompt_tokens: int = 0
  completion_tokens: int = 0
  error: str = ""
  latency_s: float = field(default=0.0)


async def generate_batch_async(
    prompts: list[str],
    base_url: str,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    concurrency: int = _DEFAULT_CONCURRENCY,
    request_timeout: int = _REQUEST_TIMEOUT_S,
) -> list[GenerationResult]:
  """Send all prompts concurrently and return results in prompt order.

  Args:
    prompts: Formatted prompt strings.
    base_url: Base URL of the server.
    model: Model name to send in each request.
    max_tokens: Maximum tokens to generate per response.
    temperature: Sampling temperature.
    concurrency: Maximum number of in-flight requests at once.
    request_timeout: Per-request wall-clock timeout in seconds.

  Returns:
    List of GenerationResult in the same order as prompts.
  """
  import aiohttp  # pylint: disable=import-outside-toplevel

  api_url = f"{base_url}{_COMPLETIONS_PATH}"
  semaphore = asyncio.Semaphore(concurrency)
  timeout = aiohttp.ClientTimeout(total=request_timeout)

  async def _generate_one(session: aiohttp.ClientSession, prompt: str) -> GenerationResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    async with semaphore:
      try:
        async with session.post(api_url, json=payload) as resp:
          if resp.status != 200:
            body = await resp.text()
            return GenerationResult(error=f"HTTP {resp.status}: {body[:200]}")
          data = await resp.json()
      except aiohttp.ClientError as exc:
        return GenerationResult(error=str(exc))
    latency = time.monotonic() - t0

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return GenerationResult(
        text=choice.get("text", ""),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_s=latency,
    )

  async with aiohttp.ClientSession(timeout=timeout) as session:
    return list(await asyncio.gather(*[_generate_one(session, p) for p in prompts]))


def generate_batch(
    prompts: list[str],
    base_url: str,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
    concurrency: int = _DEFAULT_CONCURRENCY,
    request_timeout: int = _REQUEST_TIMEOUT_S,
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
          request_timeout=request_timeout,
      )
  )
