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

"""Experimental script to benchmark weight transfers across multi-slice TPU topologies using tpu-raiden.

Exercises transfer of weights with different JAX Sharding configurations (DP, TP, TP->FSDP)
from one host/slice to another on 2 TPU v5p-8 slices, with performance timing and JAX profiling.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import socket
import sys
import time
from typing import Any, Dict, List, Tuple

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
  sys.setdlopenflags(sys.getdlopenflags() & ~ctypes.RTLD_GLOBAL | ctypes.RTLD_LOCAL)
except Exception:  # pylint: disable=broad-exception-caught
  pass

try:
  import google.protobuf.runtime_version

  google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except (ImportError, AttributeError):
  pass

try:
  from tpu_raiden.api.jax import weight_synchronizer
  from tpu_raiden.frameworks.jax import resharding_planner
  from tpu_raiden.rpc import raiden_service_pb2

  if weight_synchronizer._weight_synchronizer is None:  # pylint: disable=protected-access
    raise RuntimeError("tpu-raiden native C++ extension is not available!")
except Exception as e:  # pylint: disable=broad-exception-caught
  raise RuntimeError(f"tpu-raiden is required for transfer_weights_raiden.py but could not be imported: {e}") from e

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np


def str2bool(v: Any) -> bool:
  """Parses boolean values from string CLI arguments."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(raw_args: list[str] | None = None) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Benchmark tpu-raiden multi-slice weight transfers and sharding performance."
  )
  parser.add_argument(
      "--weight_size_mb",
      type=int,
      default=1024,
      help="Total payload size in MB for model weights.",
  )
  parser.add_argument(
      "--num_layers",
      type=int,
      default=12,
      help="Number of synthetic Transformer layers.",
  )
  parser.add_argument(
      "--iterations",
      type=int,
      default=10,
      help="Number of benchmark iterations.",
  )
  parser.add_argument(
      "--warmup_iterations",
      type=int,
      default=3,
      help="Number of warmup iterations before profiling.",
  )
  parser.add_argument(
      "--profile_dir",
      type=str,
      default="",
      help="Directory to save JAX profiler trace outputs.",
  )
  parser.add_argument(
      "--source_ip",
      type=str,
      default="0.0.0.0",
      help="Bind IP address for source WeightSynchronizer.",
  )
  parser.add_argument(
      "--dest_ip",
      type=str,
      default="127.0.0.1",
      help="Target IP address for destination WeightSynchronizer.",
  )
  parser.add_argument(
      "--dest_port",
      type=int,
      default=8080,
      help="Target port for destination WeightSynchronizer.",
  )
  parser.add_argument(
      "--verify_correctness",
      type=str2bool,
      default=True,
      help="Whether to verify transferred weight arrays against source.",
  )
  args, _ = parser.parse_known_args(raw_args)
  return args


def setup_multi_slice_devices() -> Tuple[List[Any], List[Any]]:
  """Sets up source and destination device lists across 2 TPU slices/processes."""
  try:
    if jax.process_count() > 1:
      devices = jax.local_devices(backend="tpu")
      src_devices = devices
      dst_devices = devices
    else:
      devices = jax.devices("tpu")
      num_devices = len(devices)
      half = max(1, num_devices // 2)
      src_devices = devices[:half]
      dst_devices = devices[half:]
  except Exception:  # pylint: disable=broad-exception-caught
    if jax.process_count() > 1:
      devices = jax.local_devices()
      src_devices = devices
      dst_devices = devices
    else:
      devices = jax.devices()
      num_devices = len(devices)
      half = max(1, num_devices // 2)
      src_devices = devices[:half]
      dst_devices = devices[half:]
  return src_devices, dst_devices


def calculate_layer_dimensions(target_total_bytes: int, num_layers: int) -> Tuple[int, int]:
  bytes_per_layer = target_total_bytes / num_layers
  elements_per_layer = bytes_per_layer / 4
  hidden_dim = 4096
  raw_intermediate_dim = int(elements_per_layer / (6 * hidden_dim))
  # Round up to multiple of 16 to ensure even division across mesh axes
  intermediate_dim = ((raw_intermediate_dim + 15) // 16) * 16
  return hidden_dim, max(16, intermediate_dim)


def create_synthetic_weights(num_layers: int, total_size_mb: int, mesh: Mesh, sharding_spec: P) -> Dict[str, Any]:
  """Generates synthetic Transformer model weight PyTrees matching MaxText conventions."""
  total_bytes = total_size_mb * 1024 * 1024
  dim0, dim1 = calculate_layer_dimensions(total_bytes, num_layers)

  sharding = NamedSharding(mesh, sharding_spec)
  replicated_sharding = NamedSharding(mesh, P())
  weights = {}

  for i in range(num_layers):
    layer_name = f"layer_{i}"
    raw_array = jnp.ones((dim0, dim1), dtype=jnp.float32) * (i + 1.0)
    sharded_array = jax.device_put(raw_array, sharding)
    bias_array = jax.device_put(jnp.zeros((dim1,), dtype=jnp.float32), replicated_sharding)
    weights[layer_name] = {
        "kernel": sharded_array,
        "bias": bias_array,
    }
  return weights


def get_dst_sharding_for_array(arr: jax.Array, dst_mesh: Mesh, dst_sharding_spec: P) -> NamedSharding:
  """Returns appropriate NamedSharding for an array based on its rank."""
  if arr.ndim == 2:
    return NamedSharding(dst_mesh, dst_sharding_spec)
  else:
    return NamedSharding(dst_mesh, P())


def build_resharding_start_request(
    flat_src: List[Any],
    src_sharding: NamedSharding,
    dst_sharding: NamedSharding,
    dest_ip: str,
    dest_port: int,
) -> raiden_service_pb2.StartTransferRequest:
  """Builds a StartTransferRequest protobuf containing reshard_push_schedules if needed."""
  start_req = raiden_service_pb2.StartTransferRequest(is_sender=True)
  has_reshard = False

  for arr in flat_src:
    if arr.ndim == 2:
      global_shape = arr.shape
      try:
        chunks = resharding_planner.make_resharding_plan(global_shape, src_sharding, dst_sharding)
      except Exception:  # pylint: disable=broad-exception-caught
        continue

      has_reshard = True
      src_devices = src_sharding.mesh.devices.flatten()
      dst_devices = dst_sharding.mesh.devices.flatten()

      src_map = src_sharding.devices_indices_map(global_shape)
      dst_map = dst_sharding.devices_indices_map(global_shape)

      itemsize = arr.dtype.itemsize

      for chunk in chunks:
        src_dev = src_devices[chunk.src_device_id]
        dst_dev = dst_devices[chunk.dst_device_id]

        _, src_col_slice = src_map[src_dev]
        _, dst_col_slice = dst_map[dst_dev]

        src_cols = (
            (src_col_slice.stop - src_col_slice.start) if src_col_slice.stop and src_col_slice.start else global_shape[1]
        )
        dst_cols = (
            (dst_col_slice.stop - dst_col_slice.start) if dst_col_slice.stop and dst_col_slice.start else global_shape[1]
        )

        r_start, _, c_start, _ = chunk.src_slice
        d_r_start, _, d_c_start, _ = chunk.dst_slice
        c_rows, c_cols = chunk.shape

        schedule = start_req.shard_push_schedules[chunk.src_device_id]
        entry = schedule.entries.add()
        entry.dst_peer = f"{dest_ip}:{dest_port}"
        entry.dst_shard_idx = chunk.dst_device_id
        entry.src_offset_bytes = (r_start * src_cols + c_start) * itemsize
        entry.dst_offset_bytes = (d_r_start * dst_cols + d_c_start) * itemsize
        entry.size_bytes = c_cols * itemsize
        if hasattr(entry, "src_stride_bytes"):
          entry.src_stride_bytes = src_cols * itemsize
          entry.dst_stride_bytes = dst_cols * itemsize
          entry.count = c_rows
      break

  return start_req if has_reshard else raiden_service_pb2.StartTransferRequest(is_sender=True)


def trigger_raiden_transfer(
    ws_source: weight_synchronizer.WeightSynchronizer,
    dest_ip: str,
    ws_dest: weight_synchronizer.WeightSynchronizer,
    start_transfer_req: raiden_service_pb2.StartTransferRequest | None = None,
) -> None:
  """Sends ControlRequest protobuf COMMAND_START_TRANSFER to ws_source.listener_port."""
  if start_transfer_req is None:
    start_transfer_req = raiden_service_pb2.StartTransferRequest(is_sender=True)
  else:
    start_transfer_req.is_sender = True

  req = raiden_service_pb2.ControlRequest(
      command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
      peers=[f"{dest_ip}:{ws_dest.local_port}"],
      start_transfer_request=start_transfer_req,
  )
  payload = req.SerializeToString()

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    sock.connect(("127.0.0.1", ws_source.listener_port))
  except (socket.error, OverflowError):
    sock.close()
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.connect(("::1", ws_source.listener_port))

  sock.sendall(len(payload).to_bytes(4, "big") + payload)
  resp_len = int.from_bytes(sock.recv(4), "big")
  resp_bytes = sock.recv(resp_len)
  resp = raiden_service_pb2.ControlResponse()
  resp.ParseFromString(resp_bytes)
  sock.close()
  if not resp.success:
    raise RuntimeError(f"Raiden transfer trigger failed: {resp.message}")


def transfer_and_benchmark(
    name: str,
    src_weights: Dict[str, Any],
    src_sharding: NamedSharding,
    dst_sharding: NamedSharding,
    args: argparse.Namespace,
) -> Dict[str, Any]:
  """Executes weight transfer benchmark for a given sharding configuration."""
  flat_src, treedef = jax.tree.flatten(src_weights)

  total_bytes = sum(arr.nbytes for arr in flat_src)
  total_mb = total_bytes / (1024 * 1024)

  flat_dst_init = [
      jax.device_put(
          jnp.zeros_like(arr),
          get_dst_sharding_for_array(arr, dst_sharding.mesh, dst_sharding.spec),
      )
      for arr in flat_src
  ]
  jax.tree.map(lambda x: x.block_until_ready(), flat_dst_init)

  print(f"\n--- Running Benchmark: {name} ---")
  print(f"Total payload size: {total_mb:.2f} MB ({total_bytes} bytes)")

  start_req = build_resharding_start_request(flat_src, src_sharding, dst_sharding, args.dest_ip, args.dest_port)

  dev_runtime = list(flat_src[0].devices())[0].client.runtime_type
  print(
      f"DEBUG ARRAY DIAGNOSTICS: type={type(flat_src[0])},"
      f" devices={[d.platform for d in flat_src[0].devices()]},"
      f" dev_client_runtime_type={dev_runtime}"
  )

  for w in range(args.warmup_iterations):
    syncer_src = weight_synchronizer.WeightSynchronizer(flat_src, bind_ip=args.source_ip)
    syncer_dst = weight_synchronizer.WeightSynchronizer(flat_dst_init, local_port=args.dest_port + w)
    syncer_src.d2h()
    trigger_raiden_transfer(syncer_src, args.dest_ip, syncer_dst, start_req)
    syncer_dst.h2d()
    transferred_flat = flat_dst_init
    jax.tree.map(lambda x: x.block_until_ready(), transferred_flat)

  if args.profile_dir:
    print(f"Starting JAX profiler trace output to: {args.profile_dir}")
    jax.profiler.start_trace(args.profile_dir)

  latencies = []

  for it in range(args.iterations):
    t0 = time.perf_counter()

    port_offset = args.warmup_iterations + it
    syncer_src = weight_synchronizer.WeightSynchronizer(flat_src, bind_ip=args.source_ip)
    syncer_dst = weight_synchronizer.WeightSynchronizer(flat_dst_init, local_port=args.dest_port + port_offset)
    syncer_src.d2h()
    trigger_raiden_transfer(syncer_src, args.dest_ip, syncer_dst, start_req)
    syncer_dst.h2d()
    transferred_flat = flat_dst_init

    jax.tree.map(lambda x: x.block_until_ready(), transferred_flat)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    latencies.append(elapsed_ms)

  if args.profile_dir:
    jax.profiler.stop_trace()
    print("Stopped JAX profiler trace.")

  avg_latency_ms = float(np.mean(latencies))
  min_latency_ms = float(np.min(latencies))
  throughput_gbps = (total_bytes / 1e9) / (avg_latency_ms / 1000.0) if avg_latency_ms > 0 else 0.0

  print(f"Results for {name}:")
  print(f"  Avg Latency: {avg_latency_ms:.3f} ms")
  print(f"  Min Latency: {min_latency_ms:.3f} ms")
  print(f"  Throughput : {throughput_gbps:.3f} GB/s")

  if args.verify_correctness:
    transferred_weights = jax.tree.unflatten(treedef, transferred_flat)
    for k in src_weights.keys():
      src_k = np.array(src_weights[k]["kernel"])
      dst_k = np.array(transferred_weights[k]["kernel"])
      np.testing.assert_allclose(src_k, dst_k, rtol=1e-5)

      src_b = np.array(src_weights[k]["bias"])
      dst_b = np.array(transferred_weights[k]["bias"])
      np.testing.assert_allclose(src_b, dst_b, rtol=1e-5)
    print("  Correctness: VERIFIED PASSED")

  return {
      "name": name,
      "payload_mb": total_mb,
      "avg_latency_ms": avg_latency_ms,
      "throughput_gbps": throughput_gbps,
  }


def main(raw_args: list[str] | None = None):
  args = parse_args(raw_args)
  print("=========================================================")
  print("MaxText TPU Raiden Multi-Slice Weight Transfer Benchmark")
  print("=========================================================")
  print(f"Layers: {args.num_layers}, Weight Size Target: {args.weight_size_mb} MB")
  print(f"Iterations: {args.iterations}, Warmup Iterations: {args.warmup_iterations}")

  try:
    jax.distributed.initialize()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"jax.distributed.initialize info: {e}")

  src_devices, dst_devices = setup_multi_slice_devices()

  results = []

  src_mesh_dp = Mesh(np.array(src_devices).reshape((1, len(src_devices))), axis_names=("data", "model"))
  dst_mesh_dp = Mesh(np.array(dst_devices).reshape((1, len(dst_devices))), axis_names=("data", "model"))
  src_dp_sharding = NamedSharding(src_mesh_dp, P())
  dst_dp_sharding = NamedSharding(dst_mesh_dp, P())

  src_dp_weights = create_synthetic_weights(args.num_layers, args.weight_size_mb, src_mesh_dp, P())
  res_dp = transfer_and_benchmark(
      "DP -> DP (Replicated)",
      src_dp_weights,
      src_dp_sharding,
      dst_dp_sharding,
      args,
  )
  results.append(res_dp)

  src_mesh_tp = Mesh(np.array(src_devices).reshape((1, len(src_devices))), axis_names=("data", "model"))
  dst_mesh_tp = Mesh(np.array(dst_devices).reshape((1, len(dst_devices))), axis_names=("data", "model"))
  src_tp_sharding = NamedSharding(src_mesh_tp, P(None, "model"))
  dst_tp_sharding = NamedSharding(dst_mesh_tp, P(None, "model"))

  src_tp_weights = create_synthetic_weights(args.num_layers, args.weight_size_mb, src_mesh_tp, P(None, "model"))
  res_tp = transfer_and_benchmark(
      "TP -> TP (Column Sharded)",
      src_tp_weights,
      src_tp_sharding,
      dst_tp_sharding,
      args,
  )
  results.append(res_tp)

  dst_mesh_fsdp = Mesh(np.array(dst_devices).reshape((len(dst_devices), 1)), axis_names=("data", "model"))
  dst_fsdp_sharding = NamedSharding(dst_mesh_fsdp, P("data", None))

  res_tp_fsdp = transfer_and_benchmark(
      "TP -> FSDP (Axis 1 -> Axis 0)",
      src_tp_weights,
      src_tp_sharding,
      dst_fsdp_sharding,
      args,
  )
  results.append(res_tp_fsdp)

  print("\n=========================================================")
  print("BENCHMARK SUMMARY RESULTS")
  print("=========================================================")
  print(f"{'Transfer Scenario':<30} | {'Payload (MB)':<12} | {'Latency (ms)':<12} | {'Throughput (GB/s)':<15}")
  print("-" * 77)
  for r in results:
    print(f"{r['name']:<30} | {r['payload_mb']:<12.1f} | {r['avg_latency_ms']:<12.3f} | {r['throughput_gbps']:<15.3f}")
  print("=========================================================")


if __name__ == "__main__":
  main()
