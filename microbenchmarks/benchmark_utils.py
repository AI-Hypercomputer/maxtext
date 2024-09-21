"""Utility functions for microbenchmarking."""

import datetime
import random
import string

import jax
import jsonlines


def simple_timeit(f, *args, tries=10, task=None):
  """Simple utility to time a function for multiple runs."""
  assert task is not None

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  for _ in range(tries):
    jax.devices()  # Force synchronization across devices
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())

  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"{task}: average time milliseconds: {average_time_ms:.2f}")
  return average_time_ms


def maybe_write_metrics_file(
    metrics_dir, metrics, test_name, test_start_time, test_end_time
):
  """Writes all_gather metrics to a JSONL file."""

  # Only write metrics from one host.
  if jax.process_index() != 0:
    return

  jsonl_name = "metrics_report.jsonl"
  jsonl_path = metrics_dir + "/" + jsonl_name

  metrics_data = {
      "metrics": metrics,
      "dimensions": {
          "testsuite": "microbenchmark",
          "test_name": f"{test_name}",
          "test_start_timestamp": f"{test_start_time}",
          "test_end_timestamp": f"{test_end_time}",
      },
  }

  print(f"Writing metrics to JSONL file: {jsonl_path}")
  with jsonlines.open(jsonl_path, mode="a") as writer:
    writer.write(metrics_data)
