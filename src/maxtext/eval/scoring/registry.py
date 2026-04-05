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

"""Registry mapping dataset/benchmark names to their scorer functions."""

from __future__ import annotations

from typing import Callable

from maxtext.eval.scoring import rouge_scorer

# Maps benchmark name to score_batch callable.
SCORER_REGISTRY: dict[str, Callable[..., dict]] = {
    "mlperf_openorca": rouge_scorer.score_batch,
    "openorca": rouge_scorer.score_batch,
}


def get_scorer(benchmark_name: str) -> Callable[..., dict]:
  """Return the scorer for benchmark_name.

  Args:
    benchmark_name: Benchmark identifier (e.g. "mlperf_openorca").

  Returns:
    The scorer callable.

  Raises:
    KeyError: If no scorer is registered for the given name.
  """
  key = benchmark_name.lower()
  if key not in SCORER_REGISTRY:
    raise KeyError(
        f"No scorer registered for benchmark '{benchmark_name}'. "
        f"Available: {sorted(SCORER_REGISTRY)}. "
    )
  return SCORER_REGISTRY[key]


def register_scorer(benchmark_name: str, scorer_fn: Callable[..., dict]) -> None:
  """Register a custom scorer for benchmark_name.

  Args:
    benchmark_name: Benchmark identifier.
    scorer_fn: Scorer callable.
  """
  SCORER_REGISTRY[benchmark_name.lower()] = scorer_fn
