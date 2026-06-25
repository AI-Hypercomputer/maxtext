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

# NOTE: This test/benchmark requires a multislice TPU setup to run correctly.
# It is excluded from standard pytest discovery.

"""A microbenchmark to test DCN network bandwidth using shard map.

This script should be run on a multi-slice TPU cluster (specifically across 2 slices
with any ICI/slice dimensions
"""

import datetime
import functools
import subprocess
from types import SimpleNamespace

import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from maxtext.utils import train_utils


def get_default_interface():
  try:
    route_output = subprocess.check_output("ip route show", shell=True, text=True)
    for line in route_output.splitlines():
      if "default" in line:
        return line.split("dev")[1].strip().split()[0]
  except (subprocess.SubprocessError, IndexError):
    pass
  return "eth0"


def simple_timeit(f, *args, tries=10, task=None):
  """Simple utility to time a function for multiple runs."""
  assert task is not None
  outcomes_ms = []

  # Warm up
  jax.block_until_ready(f(*args))

  for _ in range(tries):
    jax.devices()  # Force synchronization
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())

  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  return average_time_ms


def create_mesh(dcn_size: int, ici_size: int):
  """Creates a hybrid mesh with DCN and ICI axes."""
  dcn_parallelism = [dcn_size, 1]
  ici_parallelism = [1, ici_size]

  total_devices = jax.device_count()
  if total_devices != (dcn_size * ici_size):
    raise ValueError(f"Need {dcn_size * ici_size} devices, but found {total_devices}")
  mesh_devices = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism, devices=jax.devices())
  mesh = Mesh(mesh_devices, ("dcn", "ici"))
  return mesh


def run_dcn_benchmark(case=None, limit=None, burst=None, latency="50ms"):
  """Runs the DCN bandwidth benchmark for specified throttling cases."""
  print(f"JAX process index: {jax.process_index()} / {jax.process_count()}")
  print(f"Total devices: {jax.device_count()}, local devices: {jax.local_device_count()}")

  dcn_size = 2
  total_devices = jax.device_count()
  if total_devices % dcn_size != 0:
    raise ValueError(f"Total devices ({total_devices}) must be divisible by dcn_size ({dcn_size}) for 2-slice setup.")
  ici_size = total_devices // dcn_size
  mesh = create_mesh(dcn_size, ici_size)

  # Predefined cases
  all_cases = [
      ("none", "NO THROTTLING (Baseline)", False, None, None),
      ("100g", "100G Throttling", True, "100gbit", "600mb"),
      ("50g", "50G Throttling", True, "50gbit", "300mb"),
  ]

  # Filter based on arguments if requested
  if case:
    cases_to_run = [c for c in all_cases if c[0] == case]
    if not cases_to_run:
      # If custom limit/burst are provided
      if limit and burst:
        cases_to_run = [(case, f"CUSTOM Throttling ({case})", True, limit, burst)]
      else:
        raise ValueError(f"Unknown case: {case}")
  else:
    cases_to_run = all_cases

  # Qwen3-30B MoE layer weight shape: (128, 2048, 768)
  shape = (128, 2048, 768)
  dtype = jnp.bfloat16

  # Calculate size
  num_elements = 1
  for d in shape:
    num_elements *= d
  matrix_size_gbyte = num_elements * dtype.dtype.itemsize / 1e9

  # We define shard map collective psum along the DCN axis.
  # Input x is sharded across 'dcn' axis.
  @functools.partial(shard_map, mesh=mesh, in_specs=P("dcn", None, None), out_specs=P(None, None, None))
  def psum_dcn_op(x):
    return jax.lax.psum(x, "dcn")

  # Initialize matrix
  matrix = jnp.ones(shape, dtype=dtype)

  # Pre-distribute the matrix shard onto devices
  sharded_matrix = jax.device_put(matrix, jax.sharding.NamedSharding(mesh, P("dcn", None, None)))

  jitted_op = jax.jit(psum_dcn_op)

  interface = get_default_interface()

  for _, name, apply_throttling, dcn_limit, dcn_burst in cases_to_run:
    if jax.process_index() == 0:
      print("\n==================================================")
      print(f"Running Case: {name}")
      if apply_throttling:
        print(f"  Throttling Config: limit={dcn_limit}, burst={dcn_burst}, latency={latency}")
      print("==================================================")

    if apply_throttling:
      config = SimpleNamespace(
          dcn_bandwidth_limit=dcn_limit,
          dcn_bandwidth_burst=dcn_burst,
          dcn_bandwidth_latency=latency,
          dcn_bandwidth_interface=interface,
      )
      train_utils.maybe_apply_dcn_throttling(config)
    else:
      config = None

    try:
      # Sync before starting benchmark
      jax.block_until_ready(jax.device_put(0.0) + 1.0)

      if jax.process_index() == 0:
        print(f"Starting benchmark for shape: {shape} ({matrix_size_gbyte * 1000:.1f} MB)")

      # Run time test
      time_ms = simple_timeit(jitted_op, sharded_matrix, task=f"psum_dcn_{shape}")

      # Calculate Bandwidth
      achieved_bandwidth_gbyte_s = matrix_size_gbyte * (dcn_size - 1) * 2 / dcn_size / dcn_size / (time_ms / 1e3)
      achieved_bandwidth_gbps = achieved_bandwidth_gbyte_s * 8.0

      if jax.process_index() == 0:
        print(f"Results for {name}:")
        print(f"  Avg Latency: {time_ms:.2f} ms")
        print(
            f"  Achieved DCN Bandwidth: {achieved_bandwidth_gbyte_s:.3f} GB/s ({achieved_bandwidth_gbps:.2f} Gbps) per slice"
        )
    finally:
      if apply_throttling and config:
        if jax.process_index() == 0:
          print(f"Cleaning up throttling for {name}...")
        train_utils.maybe_cleanup_dcn_throttling(config)

      # Sync after cleanup
      jax.block_until_ready(jax.device_put(0.0) + 1.0)


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="DCN Bandwidth Benchmark")
  parser.add_argument(
      "--case", choices=["none", "100g", "50g"], help="Predefined throttling case (optional, runs all if omitted)"
  )
  parser.add_argument("--limit", help="Custom DCN bandwidth limit (e.g. 100gbit)")
  parser.add_argument("--burst", help="Custom DCN bandwidth burst (e.g. 600mb)")
  parser.add_argument("--latency", default="50ms", help="DCN bandwidth latency (default: 50ms)")
  parsed_args = parser.parse_args()

  run_dcn_benchmark(case=parsed_args.case, limit=parsed_args.limit, burst=parsed_args.burst, latency=parsed_args.latency)
