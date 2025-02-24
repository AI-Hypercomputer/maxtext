# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test jax.device_put, device_get latency of prefix caching."""

import argparse
import datetime
import jax
import pickle
from typing import Any, Tuple


# Same in maxtext max_utils
def print_mem_stats(label: str):
  print(f"\nMemstats: {label}:")
  try:
    for d in jax.local_devices():
      stats = d.memory_stats()
      used = round(stats["bytes_in_use"] / 2**30, 2)
      limit = round(stats["bytes_limit"] / 2**30, 2)
      print(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
  except (RuntimeError, KeyError, TypeError) as ex:
    print(f"\tMemstats unavailable, error: {ex}")


def tree_copy(tree: Any) -> Any:
  """Copy a PyTree with all jax.array.
  Args:
    tree: PyTree with all jax.array to copy.
  Return:
    Copy of tree.
  """
  return jax.tree.map(lambda x: x.copy(), tree)


def run_with_ms_time(f, *args) -> Tuple[Any, float]:
  """Run the function with timing in ms.
  Args:
    f: function to run and measure
    *args: arguments to pass to f
  Return:
    output_of_f, time_in_ms
  """
  s = datetime.datetime.now()
  output = f(*args)
  jax.block_until_ready(output)
  e = datetime.datetime.now()
  return output, (1000 * (e - s).total_seconds())


def benchmark(prefill_result, tries=10) -> None:
  """Run benchmark for move prefill result between host and devices."""

  prefill_result_nbytes = jax.tree.reduce(
      lambda acc, array: acc + array.nbytes,
      prefill_result,
      0,
  )

  # warm up
  copied_result, ms = run_with_ms_time(tree_copy, prefill_result)
  host_result, ms = run_with_ms_time(jax.device_get, copied_result)
  copied_result, ms = run_with_ms_time(tree_copy, host_result)
  _, ms = run_with_ms_time(jax.device_put, copied_result)

  device_copy_ms_list = []
  device_get_ms_list = []
  host_copy_ms_list = []
  device_put_ms_list = []
  # Copy between ops to prevent caching between devices and host of the same array.
  for _ in range(tries):
    copied_result, ms = run_with_ms_time(tree_copy, prefill_result)
    device_copy_ms_list.append(ms)
    host_result, ms = run_with_ms_time(jax.device_get, copied_result)
    device_get_ms_list.append(ms)
    copied_result, ms = run_with_ms_time(tree_copy, host_result)
    host_copy_ms_list.append(ms)
    _, ms = run_with_ms_time(jax.device_put, copied_result)
    device_put_ms_list.append(ms)

  prefix_cache_size_gb = prefill_result_nbytes / 1_000_000_000
  copy_latency_ms_on_device = sum(device_copy_ms_list) / len(device_copy_ms_list)
  device_get_latency_ms = sum(device_get_ms_list) / len(device_get_ms_list)
  device_get_bandwidth_gb_per_sec = prefix_cache_size_gb / device_get_latency_ms * 1000
  copy_latency_ms_on_host = sum(host_copy_ms_list) / len(host_copy_ms_list)
  device_put_latency_ms = sum(device_put_ms_list) / len(device_put_ms_list)
  device_put_bandwidth_gb_per_sec = prefix_cache_size_gb / device_put_latency_ms * 1000

  print("Per Prefix Caching")
  print(f"  Size: {prefix_cache_size_gb} GB.")
  print(f"  Copy Latency on Device (ms): {copy_latency_ms_on_device}")
  print(f"  Device Get Latency (ms): {device_get_latency_ms}")
  print(f"  Device Get Bandwidth (GB/s): {device_get_bandwidth_gb_per_sec}")
  print(f"  Copy Latency on Host (ms): {copy_latency_ms_on_host}")
  print(f"  Device Put Latency (ms): {device_put_latency_ms}")
  print(f"  Device Put Bandwidth (GB/s): {device_put_bandwidth_gb_per_sec}")


def main(args: argparse.Namespace) -> None:
  prefill_result_path = args.prefill_result_path

  with open(prefill_result_path, "rb") as f:
    prefill_result = pickle.load(f)
  print_mem_stats("After loading prefill result")

  benchmark(prefill_result)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--prefill_result_path",
      required=True,
      type=str,
      help="Path to saved prefix caching in pickle. Need to be PyTree containing all jax.array.\n"
      "The cache will used to test jax.device_put and jax.device_get.\n "
      "For example: \n"
      '  with open("/tmp/prefill_result.pkl", "wb") as f:\n'
      "      pickle.dump([jnp.ones((8096, 8, 64), dtype=jnp.bfloat16) for _ in range(48)], f)",
  )
  parsed_args = parser.parse_args()
  main(parsed_args)
