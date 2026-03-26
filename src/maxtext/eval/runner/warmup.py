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

"""Server warmup — triggers XLA compilation before evaluation begins.

Sends one request per prompt-length bucket (16, 32, 64, 128, 256, 512, 1024
tokens) so that vLLM compiles all the kernel shapes that will be seen during
eval.  Falls back to a single short prompt when no dataset requests are
provided.
"""

from __future__ import annotations

import asyncio
import logging

from maxtext.eval.datasets.base import SampleRequest

logger = logging.getLogger(__name__)

_COMPLETIONS_PATH = "/v1/completions"
_SIMPLE_WARMUP_PROMPT = "What is 1 + 1?"
_WARMUP_BUCKETS = [0, 16, 32, 64, 128, 256, 512, 1024]


def _sample_by_buckets(requests: list[SampleRequest]) -> list[SampleRequest]:
  """Return one request per prompt-length bucket."""
  sampled = []
  for start, end in zip(_WARMUP_BUCKETS[:-1], _WARMUP_BUCKETS[1:]):
    for req in requests:
      approx_len = len(req.prompt.split())
      if start < approx_len <= end:
        sampled.append(req)
        break
  return sampled


async def _send_warmup_requests(
    api_url: str,
    model: str,
    pairs: list[tuple[str, int]],
) -> None:
  """Send warmup requests concurrently and log any failures."""
  import aiohttp  # pylint: disable=import-outside-toplevel

  async def _post(session: aiohttp.ClientSession, prompt: str, max_tokens: int) -> bool:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    try:
      async with session.post(api_url, json=payload) as resp:
        return resp.status == 200
    except aiohttp.ClientError:
      return False

  async with aiohttp.ClientSession() as session:
    results = await asyncio.gather(*[_post(session, p, n) for p, n in pairs])

  failures = sum(1 for ok in results if not ok)
  if failures:
    logger.warning("Warmup: %d/%d requests failed.", failures, len(results))
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
    model: Model name.
    sample_requests: Optional dataset sample requests used to derive
      prompt-length buckets.  When omitted, a single short prompt is sent.
    max_tokens: Maximum tokens per warmup response.
  """
  api_url = f"{base_url}{_COMPLETIONS_PATH}"

  if sample_requests:
    bucketed = _sample_by_buckets(sample_requests)
    logger.info("Running bucketed warmup: %d prompt-length buckets.", len(bucketed))
    pairs = [(r.prompt, max_tokens) for r in bucketed]
  else:
    logger.info("Running simple warmup (no dataset samples provided).")
    pairs = [(_SIMPLE_WARMUP_PROMPT, max_tokens)]

  asyncio.run(_send_warmup_requests(api_url, model, pairs))
