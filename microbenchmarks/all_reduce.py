import argparse
import datetime
from functools import partial
import random
import string

import jax
from jax import pmap
import jax.numpy as jnp
import numpy as np
from benchmark_utils import maybe_write_metrics_file
from benchmark_utils import simple_timeit


TRACE_BASE_DIR = None
METRICS_JSONL_DIR = None

matrix_size_gbyte_to_bandwidth = {}

def all_reduce_sum(matrix_dim):
  dtype = jax.numpy.bfloat16
  matrix = jnp.ones(
      (jax.local_device_count(), matrix_dim, matrix_dim), dtype=dtype
  )
  
  @partial(pmap, axis_name="devices")
  def parallel_sum(x):
    return jax.lax.psum(x, axis_name="devices")
    
  # Preload the sharded data to devices. This is to avoid the data transfer
  # time in the all_reduce operation.
  matrix_split = jnp.array_split(matrix, jax.local_device_count(), axis=0)
  matrix_distributed = jax.device_put_sharded(matrix_split, jax.local_devices())

  average_time_ms = simple_timeit(parallel_sum, matrix_distributed, task="parallel_sum")

  print(f"Average time milliseconds: {average_time_ms:.2f}")

  matrix_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9
  shard_size_gbyte = matrix.size * dtype.dtype.itemsize / 1e9 / jax.local_device_count()
  number_of_devices = len(jax.devices())
  # Send the data to all other (N-1) devices.
  achieved_bandwidth_gbyte_s = (
      shard_size_gbyte * (number_of_devices - 1) / number_of_devices / (average_time_ms / 1e3)
  )
  matrix_size_gbyte_to_bandwidth[matrix_size_gbyte] = achieved_bandwidth_gbyte_s
  print(
      f"Matrix shape: {matrix.shape}, {dtype=}, {matrix_size_gbyte=},"
      f" {achieved_bandwidth_gbyte_s=}"
  )


def run_benchmark():
  test_start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  trace_name = f"t_all_reduce_sum_" + "".join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
  )
  trace_dir = None
  if TRACE_BASE_DIR:
    trace_dir = f"{TRACE_BASE_DIR}/{trace_name}"
    jax.profiler.start_trace(trace_dir)

  # Sweep the data size to saturate the bandwidth.
  matrix_size = 1024
  while True:
    try:
      all_reduce_sum(matrix_size)
      matrix_size += 1024
      if matrix_size > 10000:
        break
    except MemoryError:
      print(
          "MemoryError: Failed to create or process matrix of size"
          f" {matrix_size} x {matrix_size}.\n"
      )
      break
    except Exception as e:
      print(f"Exception: {e} occurred at size {matrix_size} x {matrix_size}.\n")
      break
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
    maybe_write_metrics_file(
        METRICS_JSONL_DIR, metrics, "all_reduce", test_start_time, test_end_time
    )


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
          " `--metrics_jsonl_dir=/tmp/microbenchmark/outputs/metrics.jsonl`"
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
  
