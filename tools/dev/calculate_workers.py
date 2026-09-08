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

"""Calculates the optimal number of workers for a given test flavor."""

import argparse
import json
import math
import os
import sys


# Maximum worker limits by hardware flavor
MAX_CPU_WORKERS = 4
MAX_TPU_WORKERS = 3
MAX_GPU_WORKERS = 3
DEFAULT_WORKERS = 1

# Right-sizing parameters for fast test suites
LIGHTWEIGHT_SUITE_MINUTES_THRESHOLD = 8.0
TARGET_MINUTES_PER_WORKER = 4.0


def get_max_workers_for_flavor(flavor: str) -> int:
  """Returns the maximum worker safety cap for a given test flavor."""
  if flavor.startswith("cpu-"):
    return MAX_CPU_WORKERS
  if flavor.startswith("tpu-") or flavor.startswith("tpu7x-"):
    return MAX_TPU_WORKERS
  if flavor.startswith("gpu-"):
    return MAX_GPU_WORKERS
  return DEFAULT_WORKERS


def calculate_workers(flavor: str, baseline_data: dict[str, float] | None = None) -> tuple[int, list[int]]:
  """Calculates total workers and worker groups based on baseline data.

  Args:
    flavor: The test flavor name (e.g. 'cpu-unit', 'tpu-unit').
    baseline_data: Optional dictionary mapping baseline keys to durations (sec).

  Returns:
    A tuple of (total_workers, worker_groups).
  """
  max_workers = get_max_workers_for_flavor(flavor)

  if not baseline_data:
    return max_workers, list(range(1, max_workers + 1))

  prefix = f"{flavor}::"
  matching = [float(dur) for k, dur in baseline_data.items() if k.startswith(prefix) and isinstance(dur, (int, float))]
  test_count = len(matching)
  total_seconds = sum(matching)
  total_minutes = total_seconds / 60.0

  if test_count == 0:
    return DEFAULT_WORKERS, [DEFAULT_WORKERS]

  # Never allocate more workers than available tests
  workers = min(max_workers, test_count)

  # For lightweight test suites, right-size worker count
  if total_minutes < LIGHTWEIGHT_SUITE_MINUTES_THRESHOLD:
    workers = min(workers, max(1, math.ceil(total_minutes / TARGET_MINUTES_PER_WORKER)))

  workers = max(1, workers)
  return workers, list(range(1, workers + 1))


def main() -> int:
  """CLI entry point for calculating worker parameters."""
  parser = argparse.ArgumentParser(description="Calculate total workers and worker groups for a test flavor.")
  parser.add_argument(
      "--flavor",
      type=str,
      required=True,
      help="Test flavor name (e.g. cpu-unit, tpu-unit)",
  )
  parser.add_argument(
      "--baseline",
      type=str,
      default=None,
      help="Path to per_test_baseline.json",
  )

  args = parser.parse_args()

  baseline_data = None
  if args.baseline and os.path.isfile(args.baseline):
    try:
      with open(args.baseline, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"Warning: Failed to read baseline file '{args.baseline}': {e}",
          file=sys.stderr,
      )

  total_workers, worker_groups = calculate_workers(args.flavor, baseline_data)
  worker_groups_json = json.dumps(worker_groups)

  print(f"total_workers={total_workers}")
  print(f"worker_groups={worker_groups_json}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
