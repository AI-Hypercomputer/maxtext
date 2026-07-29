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

"""Experimental script to benchmark 8GB weight transfer from 4-way FSDP on Host 0 to 4-way TP on Host 1 using tpu-raiden."""

from __future__ import annotations

import argparse
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
  from tpu_raiden.frameworks.jax import resharding_planner
  from tpu_raiden.rpc import raiden_service_pb2

  if weight_synchronizer._weight_synchronizer is None:  # pylint: disable=protected-access
    raise RuntimeError("tpu-raiden native C++ extension is not available!")
except Exception as e:  # pylint: disable=broad-exception-caught
  raise RuntimeError(f"tpu-raiden is required for transfer_weights_raiden.py but could not be imported: {e}") from e


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
  parser = argparse.ArgumentParser(description="Benchmark tpu-raiden multi-slice 8GB FSDP to TP weight transfer.")
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
      help="Number of warmup iterations before profiling.",
  )
  parser.add_argument(
      "--profile_dir",
      type=str,
      default="",
      help="Directory to write JAX profiler trace.",
  )
  parser.add_argument(
      "--source_ip",
      type=str,
      default="0.0.0.0",
      help="IP address of source host (Sender).",
  )
  parser.add_argument(
      "--dest_ip",
      type=str,
      default="127.0.0.1",
      help="IP address of destination host (Receiver).",
  )
  parser.add_argument(
      "--dest_port",
      type=int,
      default=29500,
      help="Base TCP port for destination listener.",
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


def get_dst_sharding_for_array(arr: jax.Array, dst_mesh: Mesh, dst_sharding_spec: P) -> NamedSharding:
  """Returns a NamedSharding for the destination array."""
  return NamedSharding(dst_mesh, dst_sharding_spec)


def build_resharding_start_request(
    flat_src: List[Any],
    src_sharding: NamedSharding,
    dst_sharding: NamedSharding,
    dest_ip: str,
    dest_port: int,
) -> raiden_service_pb2.StartTransferRequest:
  """Builds a StartTransferRequest protobuf containing reshard_push_schedules."""
  start_req = raiden_service_pb2.StartTransferRequest(is_sender=True)
  has_reshard = False

  for idx, arr in enumerate(flat_src):
    if arr.ndim == 2:
      global_shape = arr.shape
      try:
        chunks = resharding_planner.make_resharding_plan(global_shape, src_sharding, dst_sharding)
      except Exception:  # pylint: disable=broad-exception-caught
        continue

      if not chunks:
        continue

      has_reshard = True
      unit = start_req.src_units.add()
      unit.data_name = str(idx)

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

        has_src_cols = src_col_slice.stop is not None and src_col_slice.start is not None
        src_cols = (src_col_slice.stop - src_col_slice.start) if has_src_cols else global_shape[1]

        has_dst_cols = dst_col_slice.stop is not None and dst_col_slice.start is not None
        dst_cols = (dst_col_slice.stop - dst_col_slice.start) if has_dst_cols else global_shape[1]

        r_start, _, c_start, _ = chunk.src_slice
        d_r_start, _, d_c_start, _ = chunk.dst_slice
        c_rows, c_cols = chunk.shape

        schedule = start_req.shard_push_schedules[chunk.src_device_id]
        entry = schedule.entries.add()
        entry.dst_peer = f"{dest_ip}:{dest_port}"
        entry.dst_shard_idx = chunk.dst_device_id
        entry.src_block_id = idx
        entry.dst_block_id = idx
        if hasattr(entry, "layer_idx"):
          entry.layer_idx = idx
        entry.src_offset_bytes = (r_start * src_cols + c_start) * itemsize
        entry.dst_offset_bytes = (d_r_start * dst_cols + d_c_start) * itemsize
        entry.size_bytes = c_cols * itemsize
        if hasattr(entry, "src_stride_bytes"):
          entry.src_stride_bytes = src_cols * itemsize
          entry.dst_stride_bytes = dst_cols * itemsize
          entry.count = c_rows

  return start_req if has_reshard else raiden_service_pb2.StartTransferRequest(is_sender=True)


def trigger_raiden_transfer(
    ws_source: weight_synchronizer.WeightSynchronizer,
    dest_ip: str,
    ws_dest: weight_synchronizer.WeightSynchronizer | int | None = None,
    start_transfer_req: raiden_service_pb2.StartTransferRequest | None = None,
    dest_port: int | None = None,
):
  """Triggers Raiden weight transfer over network via native C++ listener."""
  dest_port_val = dest_port if dest_port is not None else 29500
  req = raiden_service_pb2.ControlRequest(
      command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
      peers=[f"{dest_ip}:{dest_port_val}"],
      start_transfer_request=start_transfer_req,
  )
  payload = req.SerializeToString()
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(("127.0.0.1", ws_source.listener_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)
    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(f"Raiden transfer failed: {resp.message}")


def send_completion_ack(dest_ip: str, port: int):
  """Sends a single TCP ACK to notify peer of iteration completion."""
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.settimeout(10.0)
      s.connect((dest_ip, port))
      s.sendall(b"ACK")
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
          conn.recv(1024)
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


def resolve_slice_ips(job_name: str, fallback_source: str, fallback_dest: str) -> tuple[str, str]:
  """Resolves Host 0 (Sender) and Host 1 (Receiver) IPs via Kubernetes DNS with retries."""
  sender_ip = None
  receiver_ip = None
  if job_name:
    for _ in range(30):
      try:
        if not sender_ip:
          sender_ip = socket.gethostbyname(f"{job_name}-slice-job-0-0.{job_name}")
        if not receiver_ip:
          receiver_ip = socket.gethostbyname(f"{job_name}-slice-job-1-0.{job_name}")
        if sender_ip and receiver_ip:
          print(f"  [DNS] Resolved slice-job-0-0: {sender_ip}, slice-job-1-0: {receiver_ip}", flush=True)
          return sender_ip, receiver_ip
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      time.sleep(1.0)
  return fallback_source, fallback_dest


# pylint: disable-next=too-many-positional-arguments
def transfer_and_benchmark(
    name: str,
    src_weights: List[Any],
    src_sharding: NamedSharding,
    dst_sharding: NamedSharding,
    args: argparse.Namespace,
) -> Dict[str, Any]:
  """Executes weight transfer benchmark for FSDP -> TP sharding configuration."""
  job_name = os.environ.get("JOB_NAME", os.environ.get("JOBSET_NAME", ""))
  source_ip, dest_ip = resolve_slice_ips(job_name, args.source_ip, args.dest_ip)

  flat_src, _ = jax.tree.flatten(src_weights)

  total_bytes = sum(arr.nbytes for arr in flat_src)
  total_mb = total_bytes / (1024 * 1024)

  flat_dst_init = [
      jax.device_put(
          jnp.zeros_like(arr),
          get_dst_sharding_for_array(arr, dst_sharding.mesh, dst_sharding.spec),
      )
      for arr in flat_src
  ]
  jax.tree.map(lambda x: x.block_until_ready(), flat_src)
  jax.tree.map(lambda x: x.block_until_ready(), flat_dst_init)

  dest_port = args.dest_port
  listener_port = dest_port + 100

  print(f"\n--- Running Benchmark: {name} ---", flush=True)
  print(f"Total payload size: {total_mb:.2f} MB ({total_bytes} bytes)", flush=True)

  start_req = build_resharding_start_request(flat_src, src_sharding, dst_sharding, dest_ip, dest_port)

  hostname = socket.gethostname()
  local_ip = socket.gethostbyname(hostname)
  print(
      f"  [Role Check] Hostname: {hostname}, Local IP: {local_ip}, Source IP: {source_ip}, Dest IP: {dest_ip}", flush=True
  )

  if local_ip == source_ip or "slice-job-0" in hostname or "-0-0" in hostname:
    is_sender = True
    is_receiver = False
  elif local_ip == dest_ip or "slice-job-1" in hostname or "-1-0" in hostname:
    is_sender = False
    is_receiver = True
  else:
    is_sender = source_ip != dest_ip
    is_receiver = source_ip == dest_ip or dest_ip in ("127.0.0.1", "0.0.0.0")

  print(f"  [Role Result] is_sender={is_sender}, is_receiver={is_receiver}", flush=True)

  syncer_src = None
  syncer_dst = None

  if is_sender:
    syncer_src = weight_synchronizer.WeightSynchronizer(flat_src, bind_ip=source_ip, listener_port=listener_port)

  if is_receiver:
    syncer_dst = weight_synchronizer.WeightSynchronizer(flat_dst_init, local_port=dest_port)

  sender_ack_port = dest_port + 200
  receiver_ack_port = dest_port + 300

  ack_server = None
  if is_sender:
    ack_server = PersistentACKServer(sender_ack_port)
    wait_for_port(dest_ip, dest_port, timeout=60.0)
    wait_for_port(dest_ip, receiver_ack_port, timeout=60.0)

  if is_receiver:
    ack_server = PersistentACKServer(receiver_ack_port)
    wait_for_port(source_ip, sender_ack_port, timeout=60.0)

  # 1. Warmup Iterations
  for w in range(args.warmup_iterations):
    print(f"  Warmup iteration {w + 1}/{args.warmup_iterations}...", flush=True)
    if is_sender:
      trigger_raiden_transfer(syncer_src, dest_ip, dest_port=dest_port, start_transfer_req=start_req)
      send_completion_ack(dest_ip, receiver_ack_port)
      ack_server.wait_for_ack(timeout=600.0)

    if is_receiver:
      ack_server.wait_for_ack(timeout=600.0)
      syncer_dst.h2d()
      send_completion_ack(source_ip, sender_ack_port)

  # 2. JAX Profiler Trace Start
  if args.profile_dir:
    os.makedirs(args.profile_dir, exist_ok=True)
    jax.profiler.start_trace(args.profile_dir)
    print(f"Started JAX profiler trace in {args.profile_dir}", flush=True)

  latencies = []

  # 3. Main Benchmark Iterations
  for it in range(args.iterations):
    t_start = time.perf_counter()
    if is_sender:
      trigger_raiden_transfer(syncer_src, dest_ip, dest_port=dest_port, start_transfer_req=start_req)
      send_completion_ack(dest_ip, receiver_ack_port)
      ack_server.wait_for_ack(timeout=600.0)

    if is_receiver:
      ack_server.wait_for_ack(timeout=600.0)
      syncer_dst.h2d()
      send_completion_ack(source_ip, sender_ack_port)

    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000.0
    latencies.append(elapsed_ms)
    print(f"  Iteration {it + 1}/{args.iterations}: {elapsed_ms:.2f} ms", flush=True)

  print("  Finalizing scenario benchmark...", flush=True)

  if is_sender and ack_server:
    time.sleep(1.0)
    send_completion_ack(dest_ip, receiver_ack_port)
    ack_server.wait_for_ack(timeout=60.0)

  if is_receiver and ack_server:
    ack_server.wait_for_ack(timeout=60.0)
    send_completion_ack(source_ip, sender_ack_port)

  if ack_server:
    ack_server.stop()

  if args.profile_dir:
    jax.profiler.stop_trace()
    print("Stopped JAX profiler trace.", flush=True)

  avg_latency_ms = float(np.mean(latencies))
  min_latency_ms = float(np.min(latencies))
  throughput_gbps = (total_bytes / 1e9) / (avg_latency_ms / 1000.0)

  print(f"Results for {name}:", flush=True)
  print(f"  Avg Latency: {avg_latency_ms:.3f} ms", flush=True)
  print(f"  Min Latency: {min_latency_ms:.3f} ms", flush=True)
  print(f"  Throughput : {throughput_gbps:.3f} GB/s", flush=True)

  if args.verify_correctness:
    if jax.process_count() == 1 or jax.process_index() == 1:
      for idx, arr in enumerate(flat_dst_init):
        dst_np = np.array(arr)
        expected_val = float(idx + 1.0)
        np.testing.assert_allclose(dst_np, expected_val, rtol=1e-5)
      print("  Correctness: VERIFIED PASSED", flush=True)

  time.sleep(2.0)

  return {
      "name": name,
      "payload_mb": total_mb,
      "avg_latency_ms": avg_latency_ms,
      "min_latency_ms": min_latency_ms,
      "throughput_gbps": throughput_gbps,
  }


def main():
  """Main entry point for multi-slice 8GB FSDP to TP weight transfer benchmark."""
  args = parse_args()
  print("=========================================================")
  print("MaxText TPU Raiden Multi-Slice 8GB Weight Transfer Benchmark")
  print("=========================================================")
  print(f"Layers: {args.num_layers}, Weight Size Target: {args.weight_size_mb} MB")
  print(f"Iterations: {args.iterations}, Warmup Iterations: {args.warmup_iterations}")

  src_devices, dst_devices = setup_multi_slice_devices()

  # 4-way FSDP on Host 0 (Data Parallel across 4 devices, size 4 along axis 0)
  src_mesh_fsdp = Mesh(np.array(src_devices).reshape((len(src_devices), 1)), axis_names=("data", "model"))
  src_fsdp_sharding = NamedSharding(src_mesh_fsdp, P("data", None))

  # 4-way TP on Host 1 (Tensor Parallel / Column Sharded across 4 devices, size 4 along axis 1)
  dst_mesh_tp = Mesh(np.array(dst_devices).reshape((1, len(dst_devices))), axis_names=("data", "model"))
  dst_tp_sharding = NamedSharding(dst_mesh_tp, P(None, "model"))

  src_fsdp_weights = create_synthetic_weights(args.num_layers, args.weight_size_mb, src_mesh_fsdp, P("data", None))

  res = transfer_and_benchmark(
      "4-way FSDP (Host 0) -> 4-way TP (Host 1)",
      src_fsdp_weights,
      src_fsdp_sharding,
      dst_tp_sharding,
      args,
  )

  print("\n=========================================================")
  print("BENCHMARK SUMMARY RESULT")
  print("=========================================================")
  print(f"{'Transfer Scenario':<40} | {'Payload (GB)':<12} | {'Latency (ms)':<12} | {'Throughput (GB/s)':<15}")
  print("-" * 87)
  payload_gb = res["payload_mb"] / 1024.0
  print(f"{res['name']:<40} | {payload_gb:<12.2f} | {res['avg_latency_ms']:<12.3f} |" f" {res['throughput_gbps']:<15.3f}")
  print("=========================================================")


if __name__ == "__main__":
  main()
