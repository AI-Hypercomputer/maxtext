# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mixture configuration generator for multi-domain PyGrain datasets in MaxText."""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np


DEFAULT_LOCAL_DATA_PATTERN = (
    "tests/assets/local_datasets/c4_en_dataset_minimal/c4/en/3.0.1/c4-train.array_record-*"
)


def generate_mixture_config(
    data_path_pattern: Union[str, List[str]] = DEFAULT_LOCAL_DATA_PATTERN,
    num_domains: int = 35,
    weights: Optional[List[float]] = None,
    distribution: str = "power_law",
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
  """Generates a multi-domain dataset mixture configuration dict.

  Args:
    data_path_pattern: File glob pattern or list of patterns. If a single pattern
      is given, it is reused across all domains.
    num_domains: Total number of mixture domains (default: 35).
    weights: Optional explicit list of weights for each domain.
    distribution: Distribution type if weights are not provided ('power_law',
      'uniform', 'dirichlet').
    output_path: Optional file path to write JSON configuration.

  Returns:
    Dictionary structured as:
    {
      "domain_00": {"path": "...", "weight": 0.08},
      ...
    }
  """
  if weights is not None:
    if len(weights) != num_domains:
      raise ValueError(
          f"Length of weights ({len(weights)}) must match num_domains ({num_domains})"
      )
    raw_weights = np.array(weights, dtype=np.float64)
  elif distribution == "uniform":
    raw_weights = np.ones(num_domains, dtype=np.float64)
  elif distribution == "power_law":
    # Zipfian / power-law distribution typical of realistic multi-domain mixtures
    ranks = np.arange(1, num_domains + 1, dtype=np.float64)
    raw_weights = 1.0 / (ranks ** 0.8)
  elif distribution == "dirichlet":
    rng = np.random.default_rng(seed=42)
    raw_weights = rng.dirichlet(np.ones(num_domains))
  else:
    raise ValueError(f"Unknown distribution: {distribution}")

  normalized_weights = raw_weights / np.sum(raw_weights)

  if isinstance(data_path_pattern, str):
    patterns = [data_path_pattern] * num_domains
  elif len(data_path_pattern) == num_domains:
    patterns = data_path_pattern
  else:
    patterns = [
        data_path_pattern[i % len(data_path_pattern)]
        for i in range(num_domains)
    ]

  config: Dict[str, Dict[str, Any]] = {}
  for i in range(num_domains):
    domain_key = f"domain_{i:02d}"
    config[domain_key] = {
        "path": patterns[i],
        "weight": round(float(normalized_weights[i]), 6),
    }

  if output_path:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(config, f, indent=2)
    print(f"Mixture config written to {output_path} ({num_domains} domains)")

  return config


def main():
  parser = argparse.ArgumentParser(
      description="Generate MaxText multi-domain PyGrain mixture configuration JSON"
  )
  parser.add_argument(
      "--num-domains",
      type=int,
      default=35,
      help="Number of mixture domains (default: 35)",
  )
  parser.add_argument(
      "--data-pattern",
      type=str,
      default=DEFAULT_LOCAL_DATA_PATTERN,
      help="Dataset file path glob pattern",
  )
  parser.add_argument(
      "--distribution",
      type=str,
      choices=["power_law", "uniform", "dirichlet"],
      default="power_law",
      help="Domain weight distribution",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="tests/assets/test_35_domain_mixture.json",
      help="Output JSON file path",
  )
  args = parser.parse_args()

  generate_mixture_config(
      data_path_pattern=args.data_pattern,
      num_domains=args.num_domains,
      distribution=args.distribution,
      output_path=args.output,
  )


if __name__ == "__main__":
  main()
