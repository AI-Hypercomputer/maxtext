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

"""Simplified Multi-Slice Weight Transfer for MaxText using TPU-Raiden with Persistent ACK Server."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import queue
import socket
import threading
import time
from typing import Any, Dict, List, Tuple

# Default host platform devices if XLA_FLAGS is not set
if "XLA_FLAGS" not in os.environ:
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

try:
  import google.protobuf.runtime_version

  google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except (ImportError, AttributeError):
  pass

try:
  from tpu_raiden.api.jax import weight_synchronizer
  from tpu_raiden.rpc import raiden_controller
  from tpu_raiden.rpc import raiden_service_pb2

  if weight_synchronizer._weight_synchronizer is None:  # pylint: disable=protected-access
    raise RuntimeError("tpu-raiden native C++ extension is not available!")
except Exception as e:  # pylint: disable=broad-exception-caught
  raise RuntimeError(
      f"tpu-raiden is required for transfer_weights_raiden_simple.py but could not be imported: {e}"
  ) from e


def str2bool(v: Any) -> bool:
  """Parses boolean values from string CLI arguments."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  if v.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(raw_args: list[str] | None = None) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(description="Simplified tpu-raiden weight transfer with ACK server & stage metrics.")
  parser.add_argument(
      "--weight_size_mb",
      type=int,
      default=8192,
      help="Total payload size in MB for model weights (default: 8192 MB / 8 GB).",
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
      default=5,
      help="Number of benchmark iterations.",
  )
  parser.add_argument(
      "--warmup_iterations",
      type=int,
      default=3,
      help="Number of warmup iterations.",
  )
  parser.add_argument(
      "--source_ip",
      type=str,
      default="0.0.0.0",
      help="Source host IP address (Sender).",
  )
  parser.add_argument(
      "--dest_ip",
      type=str,
      default="127.0.0.1",
      help="Destination host IP address (Receiver).",
  )
  parser.add_argument(
      "--dest_port",
      type=int,
      default=29500,
      help="Destination Raiden service port.",
  )
  parser.add_argument(
      "--verify_correctness",
      type=str2bool,
      default=True,
      help="Verify values after transfer.",
  )
  return parser.parse_args(raw_args)


def setup_multi_slice_devices() -> Tuple[List[Any], List[Any]]:
  """Sets up source and destination device lists across 2 TPU slices/processes."""
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
  """Calculates layer 2D matrix shape to match target byte size."""
  bytes_per_layer = target_total_bytes / num_layers
  elements_per_layer = bytes_per_layer / 4
  hidden_dim = 4096
  raw_intermediate_dim = int(elements_per_layer / hidden_dim)
  intermediate_dim = ((raw_intermediate_dim + 15) // 16) * 16
  return hidden_dim, max(16, intermediate_dim)


def create_synthetic_weights(num_layers: int, total_size_mb: int, mesh: Mesh, sharding_spec: P) -> List[jax.Array]:
  """Generates synthetic 2D weight arrays with given sharding spec."""
  total_bytes = total_size_mb * 1024 * 1024
  dim0, dim1 = calculate_layer_dimensions(total_bytes, num_layers)

  sharding = NamedSharding(mesh, sharding_spec)
  arrays = []

  for i in range(num_layers):
    raw_array = jnp.ones((dim0, dim1), dtype=jnp.float32) * (i + 1.0)
    sharded_array = jax.device_put(raw_array, sharding)
    arrays.append(sharded_array)
  return arrays


def get_dst_sharding_for_array(_arr: jax.Array, dst_mesh: Mesh, dst_sharding_spec: P) -> NamedSharding:
  """Creates destination sharding for an array."""
  return NamedSharding(dst_mesh, dst_sharding_spec)


def create_variable_protos(
    flat_src: List[jax.Array],
) -> List[raiden_service_pb2.VariableMetadataProto]:
  """Constructs a list of VariableMetadataProto for synthetic weight matrices."""
  protos = []
  for idx, arr in enumerate(flat_src):
    sharding = arr.sharding
    if isinstance(sharding, NamedSharding):
      mesh = sharding.mesh
      spec = sharding.spec
      mesh_shape = []
      for d in range(len(arr.shape)):
        if d < len(spec) and spec[d] is not None:
          axis_spec = spec[d]
          if isinstance(axis_spec, str):
            mesh_shape.append(mesh.shape[axis_spec])
          elif isinstance(axis_spec, tuple):
            mesh_shape.append(math.prod(mesh.shape[axis] for axis in axis_spec))
          else:
            mesh_shape.append(1)
        else:
          mesh_shape.append(1)
    else:
      mesh_shape = [1] * len(arr.shape)

    layout = list(reversed(range(len(arr.shape))))
    proto = raiden_service_pb2.VariableMetadataProto(
        name=f"weights_{idx}",
        shape=list(arr.shape),
        mesh_shape=mesh_shape,
        layout=layout,
        item_size=arr.dtype.itemsize,
        layer_idx=idx,
    )
    protos.append(proto)
  return protos


def send_completion_ack(dest_ip: str, port: int):
  """Sends a single TCP ACK to notify peer of iteration completion."""
  try:
    with socket.create_connection((dest_ip, port), timeout=10.0) as sock:
      sock.sendall(b"ACK\n")
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Warning: Failed to send completion ACK to {dest_ip}:{port}: {e}", flush=True)


class PersistentACKServer:
  """Single persistent TCP ACK server listening on a single port across all iterations."""

  def __init__(self, port: int):
    self.port = port
    self.queue = queue.Queue()
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.sock.bind(("0.0.0.0", port))
    self.sock.listen(128)
    self.running = True
    self.thread = threading.Thread(target=self._listen, daemon=True)
    self.thread.start()

  def _listen(self):
    """Background worker loop accepting TCP ACK connections."""
    while self.running:
      try:
        self.sock.settimeout(1.0)
        conn, _ = self.sock.accept()
        with conn:
          data = conn.recv(1024)
          if b"ACK" in data:
            self.queue.put(True)
      except socket.timeout:
        continue
      except Exception as e:  # pylint: disable=broad-exception-caught
        if self.running:
          print(f"Warning: ACK server error on port {self.port}: {e}", flush=True)

  def wait_for_ack(self, timeout: float = 120.0) -> bool:
    """Waits for an incoming ACK signal."""
    try:
      return self.queue.get(timeout=timeout)
    except queue.Empty:
      print(f"Warning: Timed out waiting for completion ACK on port {self.port}", flush=True)
      return False

  def stop(self):
    """Stops the ACK server thread."""
    self.running = False
    try:
      self.sock.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def wait_for_port(ip: str, port: int, timeout: float = 60.0) -> bool:
  """Waits until the destination port is open and listening."""
  t0 = time.time()
  print(f"  [wait_for_port] Waiting for {ip}:{port} to be ready...", flush=True)
  while time.time() - t0 < timeout:
    try:
      with socket.create_connection((ip, port), timeout=1.0):
        print(f"  [wait_for_port] Destination {ip}:{port} is open and ready!", flush=True)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
      time.sleep(0.5)
  print(f"  [wait_for_port] Timed out waiting for {ip}:{port}", flush=True)
  return False


# pylint: disable=too-many-positional-arguments,too-many-arguments
def transfer_and_benchmark(
    args: argparse.Namespace,
    src_devices: List[Any],
    dst_devices: List[Any],
    src_sharding_spec: P,
    dst_sharding_spec: P,
    scenario_name: str,
    source_ip: str,
    dest_ip: str,
    dest_port: int,
    is_sender: bool,
    is_receiver: bool,
) -> Dict[str, Any]:
  """Performs weight synchronization and benchmarks stage-by-stage latency/throughput with ACK server."""
  src_mesh = Mesh(np.array(src_devices), axis_names=("devices",))
  dst_mesh = Mesh(np.array(dst_devices), axis_names=("devices",))

  total_bytes = args.weight_size_mb * 1024 * 1024
  dim0, dim1 = calculate_layer_dimensions(total_bytes, args.num_layers)
  actual_bytes_per_layer = dim0 * dim1 * 4
  total_payload_bytes = actual_bytes_per_layer * args.num_layers
  total_payload_mb = total_payload_bytes / (1024 * 1024)

  print(f"\n--- Running Simplified Benchmark: {scenario_name} ---", flush=True)
  print(f"Total payload size: {total_payload_mb:.2f} MB ({total_payload_bytes} bytes)", flush=True)

  flat_src = create_synthetic_weights(args.num_layers, args.weight_size_mb, src_mesh, src_sharding_spec)
  dst_shardings = [get_dst_sharding_for_array(arr, dst_mesh, dst_sharding_spec) for arr in flat_src]

  flat_dst_init = [
      jax.device_put(jnp.zeros((dim0, dim1), dtype=jnp.float32), dst_sharding) for dst_sharding in dst_shardings
  ]

  sender_listener_port = dest_port + 100
  receiver_listener_port = dest_port + 150

  syncer_src = None
  syncer_dst = None

  sender_ack_port = dest_port + 200
  receiver_ack_port = dest_port + 300

  ack_server = None
  controller = None
  src_unit_id = raiden_controller.RaidenId(job_name="source", job_replica_id="0", data_name="weights", data_replica_idx=0)
  dst_unit_id = raiden_controller.RaidenId(
      job_name="destination", job_replica_id="0", data_name="weights", data_replica_idx=0
  )

  try:
    if is_sender and is_receiver:
      syncer_src = weight_synchronizer.WeightSynchronizer(
          flat_src, bind_ip=source_ip, listener_port=sender_listener_port, parallelism=8
      )
      syncer_dst = weight_synchronizer.WeightSynchronizer(
          flat_dst_init, local_port=dest_port, listener_port=receiver_listener_port, parallelism=8
      )
      ack_server = None
    elif is_sender:
      syncer_src = weight_synchronizer.WeightSynchronizer(
          flat_src, bind_ip=source_ip, listener_port=sender_listener_port, parallelism=8
      )
      ack_server = PersistentACKServer(sender_ack_port)
      wait_for_port(dest_ip, dest_port, timeout=60.0)
      wait_for_port(dest_ip, receiver_listener_port, timeout=60.0)
      wait_for_port(dest_ip, receiver_ack_port, timeout=60.0)
    elif is_receiver:
      syncer_dst = weight_synchronizer.WeightSynchronizer(
          flat_dst_init, local_port=dest_port, listener_port=receiver_listener_port, parallelism=8
      )
      ack_server = PersistentACKServer(receiver_ack_port)
      wait_for_port(source_ip, sender_ack_port, timeout=60.0)

    if ack_server:
      while not ack_server.queue.empty():
        try:
          ack_server.queue.get_nowait()
        except queue.Empty:
          break

    if is_sender:
      src_protos = create_variable_protos(flat_src)
      dst_protos = create_variable_protos(flat_dst_init)

      worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient()
      controller = raiden_controller.RaidenController(
          port=0,
          worker_rpc_client=worker_rpc_client,
      )
      src_local_port = syncer_src.local_port if syncer_src and syncer_src.local_port else sender_listener_port
      controller.register_work_unit(
          src_unit_id,
          shards=[f"{source_ip}:{src_local_port}"] * len(src_devices),
          control_plane_rpc_address=f"{source_ip}:{sender_listener_port}",
          variables=src_protos,
      )
      controller.register_work_unit(
          dst_unit_id,
          shards=[f"{dest_ip}:{dest_port}"] * len(dst_devices),
          control_plane_rpc_address=f"{dest_ip}:{receiver_listener_port}",
          variables=dst_protos,
      )

    # 1. Warmup Iterations
    for w in range(args.warmup_iterations):
      print(f"  Warmup iteration {w + 1}/{args.warmup_iterations}...", flush=True)
      if is_sender and is_receiver:
        future = controller.start_transfer(
            src_units=[src_unit_id],
            dst_units=[dst_unit_id],
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            use_block_chunks=True,
            is_sender=True,
        )
        asyncio.run(future.wait())
        syncer_dst.h2d()
      elif is_sender:
        future = controller.start_transfer(
            src_units=[src_unit_id],
            dst_units=[dst_unit_id],
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            use_block_chunks=True,
            is_sender=True,
        )
        asyncio.run(future.wait())
        send_completion_ack(dest_ip, receiver_ack_port)
        if ack_server:
          ack_server.wait_for_ack(timeout=600.0)
      elif is_receiver:
        if ack_server:
          ack_server.wait_for_ack(timeout=600.0)
        syncer_dst.h2d()
        send_completion_ack(source_ip, sender_ack_port)

    e2e_ms_list = []

    # 2. Benchmark Iterations
    for it in range(args.iterations):
      e2e_sec = 0.0
      if is_sender and is_receiver:
        t_start = time.perf_counter()
        future = controller.start_transfer(
            src_units=[src_unit_id],
            dst_units=[dst_unit_id],
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            use_block_chunks=True,
            is_sender=True,
        )
        asyncio.run(future.wait())
        syncer_dst.h2d()
        t_end = time.perf_counter()
        e2e_sec = t_end - t_start
      elif is_sender:
        t_start = time.perf_counter()
        future = controller.start_transfer(
            src_units=[src_unit_id],
            dst_units=[dst_unit_id],
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            use_block_chunks=True,
            is_sender=True,
        )
        asyncio.run(future.wait())
        send_completion_ack(dest_ip, receiver_ack_port)
        if ack_server:
          ack_server.wait_for_ack(timeout=600.0)
        t_end = time.perf_counter()
        e2e_sec = t_end - t_start
      elif is_receiver:
        if ack_server:
          ack_server.wait_for_ack(timeout=600.0)
        t_start = time.perf_counter()
        syncer_dst.h2d()
        t_end = time.perf_counter()
        h2d_sec = t_end - t_start
        e2e_sec = h2d_sec
        send_completion_ack(source_ip, sender_ack_port)

      e2e_ms = e2e_sec * 1000.0
      e2e_ms_list.append(e2e_ms)
      e2e_mbps = total_payload_mb / e2e_sec if e2e_sec > 0 else 0.0

      print(
          f"  Iteration {it + 1}/{args.iterations}: " f"E2E: {e2e_ms:.2f} ms ({e2e_mbps:.2f} MB/s)",
          flush=True,
      )

    if is_sender and not is_receiver:
      send_completion_ack(dest_ip, receiver_ack_port)
    if is_receiver and not is_sender and ack_server:
      ack_server.wait_for_ack(timeout=60.0)

  finally:
    if ack_server:
      ack_server.stop()

  # 3. Correctness Verification
  if is_receiver and args.verify_correctness:
    print("Verifying transferred weight correctness on receiver...", flush=True)
    for idx, (expected_arr, dst_arr) in enumerate(zip(flat_src, flat_dst_init)):
      expected_data = np.asarray(expected_arr)
      actual_data = np.asarray(dst_arr)
      if not np.allclose(expected_data, actual_data, rtol=1e-5, atol=1e-5):
        print(f"Correctness Mismatch on Layer {idx}:", flush=True)
        print(
            f"  Expected shape={expected_data.shape}, min={expected_data.min()},"
            f" max={expected_data.max()}, sample={expected_data[:2, :5]}",
            flush=True,
        )
        print(
            f"  Actual   shape={actual_data.shape}, min={actual_data.min()},"
            f" max={actual_data.max()}, sample={actual_data[:2, :5]}",
            flush=True,
        )
        raise ValueError(f"Correctness check failed on layer {idx}!")
    print("VERIFICATION PASSED: All weights matched expected values!", flush=True)

  avg_e2e_ms = float(np.mean(e2e_ms_list))
  min_e2e_ms = float(np.min(e2e_ms_list))
  avg_e2e_mbps = total_payload_mb / (avg_e2e_ms / 1000.0) if avg_e2e_ms > 0 else 0.0
  min_e2e_mbps = total_payload_mb / (min_e2e_ms / 1000.0) if min_e2e_ms > 0 else 0.0

  print(f"\nLatency & Throughput Summary for {scenario_name}:", flush=True)
  print(
      f"  E2E - Avg: {avg_e2e_ms:.3f} ms ({avg_e2e_mbps:.2f} MB/s), Min: {min_e2e_ms:.3f} ms ({min_e2e_mbps:.2f} MB/s)",
      flush=True,
  )

  return {
      "scenario": scenario_name,
      "payload_bytes": total_payload_bytes,
      "payload_mb": total_payload_mb,
      "avg_e2e_ms": avg_e2e_ms,
      "min_e2e_ms": min_e2e_ms,
      "avg_e2e_mbps": avg_e2e_mbps,
      "min_e2e_mbps": min_e2e_mbps,
  }


def main(raw_args: list[str] | None = None):
  """Main entry point for multi-slice weight transfer benchmark with ACK server."""
  args = parse_args(raw_args)

  print("=========================================================", flush=True)
  print("MaxText TPU Raiden Multi-Slice Weight Transfer Benchmark", flush=True)
  print("=========================================================", flush=True)
  print(f"Layers: {args.num_layers}, Weight Size Target: {args.weight_size_mb} MB", flush=True)
  print(f"Iterations: {args.iterations}, Warmup Iterations: {args.warmup_iterations}", flush=True)

  src_devices, dst_devices = setup_multi_slice_devices()

  source_ip = args.source_ip
  dest_ip = args.dest_ip
  dest_port = args.dest_port

  is_sender = False
  is_receiver = False

  if jax.process_count() > 1:
    local_host_ips = socket.gethostbyname_ex(socket.gethostname())[2]
    local_host_ips.append(socket.gethostname())
    local_host_ips.append("127.0.0.1")

    if source_ip in local_host_ips or source_ip == "0.0.0.0":
      is_sender = True
    if dest_ip in local_host_ips:
      is_receiver = True

    if not is_sender and not is_receiver:
      if jax.process_index() == 0:
        is_sender = True
      else:
        is_receiver = True
  else:
    is_sender = True
    is_receiver = True

  print(
      f"  [Role Check] Hostname: {socket.gethostname()}, Local IP: {socket.gethostbyname(socket.gethostname())},"
      f" Source IP: {source_ip}, Dest IP: {dest_ip}",
      flush=True,
  )
  print(f"  [Role Result] is_sender={is_sender}, is_receiver={is_receiver}", flush=True)

  results = []
  scenarios = [
      (
          "2D 4-way FSDP (Host 0) -> 2D 4-way TP (Host 1)",
          P("devices", None),
          P(None, "devices"),
      )
  ]

  for scenario_name, src_sharding_spec, dst_sharding_spec in scenarios:
    res = transfer_and_benchmark(
        args,
        src_devices,
        dst_devices,
        src_sharding_spec,
        dst_sharding_spec,
        scenario_name,
        source_ip,
        dest_ip,
        dest_port,
        is_sender,
        is_receiver,
    )
    results.append(res)

  print(
      "\n==========================================================================================================",
      flush=True,
  )
  print("BENCHMARK SUMMARY RESULT", flush=True)
  print(
      "==========================================================================================================",
      flush=True,
  )
  print(
      f"{'Transfer Scenario':<45} | {'Payload(MB)':<11} | {'E2E Avg(ms)':<11} |"
      f" {'E2E Min(ms)':<11} | {'E2E (MB/s)':<11}"
  )
  print("-" * 100)
  for r in results:
    print(
        f"{r['scenario']:<45} | {r['payload_mb']:<11.2f} | {r['avg_e2e_ms']:<11.2f} |"
        f" {r['min_e2e_ms']:<11.2f} | {r['avg_e2e_mbps']:<11.2f}"
    )
  print(
      "==========================================================================================================",
      flush=True,
  )


if __name__ == "__main__":
  main()
