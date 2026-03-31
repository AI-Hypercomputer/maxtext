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

"""Registry mapping benchmark names to BenchmarkDataset classes.

This can be used to define custom dataset loaders for benchmarks not covered by lm_eval and evalchemy.
"""

from __future__ import annotations

from maxtext.eval.datasets.base import BenchmarkDataset
from maxtext.eval.datasets.mlperf import MlperfOpenOrcaDataset

DATASET_REGISTRY: dict[str, type[BenchmarkDataset]] = {
    "mlperf_openorca": MlperfOpenOrcaDataset,
    "openorca": MlperfOpenOrcaDataset,
}


def get_dataset(benchmark_name: str) -> BenchmarkDataset:
  """Instantiate and return the mapping for benchmark_name.

  Args:
    benchmark_name: Benchmark identifier (e.g. "mlperf_openorca").

  Returns:
    An instance of the corresponding BenchmarkDataset subclass.

  Raises:
    KeyError: If no dataset is registered for the given name.
  """
  key = benchmark_name.lower()
  if key not in DATASET_REGISTRY:
    raise KeyError(
        f"No dataset registered for benchmark '{benchmark_name}'. "
        f"Available: {sorted(DATASET_REGISTRY)}. "
        f"For MMLU/GPQA/MATH use lm_eval_runner or evalchemy_runner instead."
    )
  return DATASET_REGISTRY[key]()


def register_dataset(benchmark_name: str, dataset_cls: type[BenchmarkDataset]) -> None:
  """Register a custom dataset class for benchmark_name.

  Args:
    benchmark_name: Lowercase benchmark identifier.
    dataset_cls: A BenchmarkDataset subclass.
  """
  DATASET_REGISTRY[benchmark_name.lower()] = dataset_cls
