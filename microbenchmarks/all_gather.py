"""All-gather benchmark."""
import argparse
import datetime
from functools import partial
from pathlib import Path
import random
import string

import jax
import numpy as np
from benchmark_utils import maybe_write_metrics_file
from benchmark_utils import simple_timeit


TRACE_BASE_DIR = None
METRICS_JSONL_DIR = None

matrix_size_gbyte_to_bandwidth = {}


def all_gather(matrix_dim):
  """Performs an all_gather operation and calculates the achieved bandwidth."""
  dtype = jax.numpy.bfloat16
  matrix = jax.numpy.ones((matrix_dim, matrix_dim), dtype=dtype)

  selected_devices = jax.devices()
  mesh = jax.sharding.Mesh(selected_devices, "axis")
  sharded_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("axis")
  )
  unsharded_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(None)
  )

  # matrix = jax.device_put(matrix, sharded_sharding)
  arrays = [
    jax.device_put(matrix[index], d)
        for d, index in sharded_sharding.addressable_devices_indices_map(matrix.shape).items()]

  matrix = jax.make_array_from_single_device_arrays(matrix.shape, sharded_sharding, arrays)
  
  @partial(jax.jit, out_shardings=unsharded_sharding)
  def unshard_array(input_matrix):
    return input_matrix

  average_time_ms = simple_timeit(unshard_array, matrix, task="unshard_array")

  matrix_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9
  number_of_devices = len(jax.devices())
  sharded_matrix_size_gbyte = matrix_size_gbyte / number_of_devices

  # Calculate achieved bandwidth
  achieved_bandwidth_gbyte_s = (
      sharded_matrix_size_gbyte
      * (number_of_devices - 1)
      / (average_time_ms / 1e3)
  )
  matrix_size_gbyte_to_bandwidth[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
  print(
      f"Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
      f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
  )


def run_benchmark():
  """Runs the all_gather benchmark and saves traces."""

  trace_dir = None
  if TRACE_BASE_DIR:
    trace_name = (
        "t_all_gather_"
        + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    )
    trace_dir = f"{TRACE_BASE_DIR}/{trace_name}"
    jax.profiler.start_trace(str(trace_dir))

  test_start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  matrix_size = 1024
  try:
    while matrix_size <= 30000:
      all_gather(matrix_size)
      matrix_size += 1024
  except MemoryError:
    print(
        "MemoryError: Failed to create or process matrix of size "
        f"{matrix_size} x {matrix_size}.\n"
    )
  except Exception as e:
    print(
        f"Exception: {e} occurred at size {matrix_size} x {matrix_size}.\n"
    )

  if TRACE_BASE_DIR:
    jax.profiler.stop_trace()
    print(f"Trace saved to {trace_dir}")

  test_end_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

  # Calculate and write metrics
  max_achieved_bandwidth_gbyte_s = max(matrix_size_gbyte_to_bandwidth.values())
  median_achieved_bandwidth_gbyte_s = np.percentile(
      list(matrix_size_gbyte_to_bandwidth.values()), 50
  )
  p90_achieved_bandwidth_gbyte_s = np.percentile(
      list(matrix_size_gbyte_to_bandwidth.values()), 90
  )

  metrics = {
      "max_achieved_bandwidth_gbyte_s": max_achieved_bandwidth_gbyte_s,
      "median_achieved_bandwidth_gbyte_s": median_achieved_bandwidth_gbyte_s,
      "p90_achieved_bandwidth_gbyte_s": p90_achieved_bandwidth_gbyte_s,
  }
  if METRICS_JSONL_DIR:
    maybe_write_metrics_file(METRICS_JSONL_DIR, metrics, "all_gather", test_start_time, test_end_time)


def main():
  """Parses arguments and runs the benchmark."""
  parser = argparse.ArgumentParser(
      description=(
          "A script to analyze the benchmark results and dump the result"
          " to a JSONL file."
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )

  parser.add_argument(
      "--trace_dir",
      type=str,
      help=(
          "Set the output directory, such as"
          " `--trace_dir=/tmp/microbenchmark/outputs`"
      ),
  )
  parser.add_argument(
      "--metrics_jsonl_dir",
      type=str,
      help=(
          "The directory to generate the metrics JSONL file, such as"
          " `--metrics_jsonl_dir=/tmp/microbenchmark/outputs/`"
      ),
  )

  args = parser.parse_args()

  global TRACE_BASE_DIR, METRICS_JSONL_DIR
  if args.trace_dir:
    TRACE_BASE_DIR = args.trace_dir
  if args.metrics_jsonl_dir:
    METRICS_JSONL_DIR = args.metrics_jsonl_dir

  run_benchmark()


if __name__ == "__main__":
  main()
