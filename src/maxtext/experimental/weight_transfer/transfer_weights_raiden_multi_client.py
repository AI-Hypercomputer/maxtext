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

"""Multi-Client JAX Benchmark Script for TPU Raiden in OSS mode."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from typing import Any, List, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

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
except Exception as e:
  raise RuntimeError(
      "tpu-raiden is required for transfer_weights_raiden_multi_client.py " f"but could not be imported: {e}"
  ) from e


_ROLE = flags.DEFINE_enum(
    "role",
    "controller",
    ["controller", "source", "destination", "all"],
    "Role of this process: 'controller', 'source', 'destination', or 'all'.",
)
_CONTROLLER_PORT = flags.DEFINE_integer(
    "controller_port",
    29500,
    "Port for the central Raiden controller server.",
)
_CONTROLLER_ADDRESS = flags.DEFINE_string(
    "controller_address",
    "127.0.0.1:29500",
    "Address of the central Raiden controller (host:port).",
)
_LOCAL_IP = flags.DEFINE_string(
    "local_ip",
    "",
    "Explicit local IP address to bind/advertise. If empty, auto-resolved.",
)
_DATA_PORT = flags.DEFINE_integer(
    "data_port",
    0,
    "Data TCP port for C++ worker data sockets (0 for ephemeral).",
)
_LISTENER_PORT = flags.DEFINE_integer(
    "listener_port",
    0,
    "Listener port for C++ worker listener (0 for ephemeral).",
)
_ITERATIONS = flags.DEFINE_integer(
    "iterations",
    5,
    "Number of benchmark iterations.",
)
_WARMUP_ITERATIONS = flags.DEFINE_integer(
    "warmup_iterations",
    3,
    "Number of warmup iterations.",
)
_VERIFY = flags.DEFINE_bool(
    "verify",
    True,
    "Whether to verify weight values on destination after transfer.",
)
_PROFILE_DIR = flags.DEFINE_string(
    "profile_dir",
    "",
    "Directory to write JAX profiler trace.",
)
_VOCAB_SIZE = flags.DEFINE_integer(
    "vocab_size",
    256000,
    "Vocabulary size for Gemma2 2B model embeddings.",
)
_NUM_LAYERS = flags.DEFINE_integer(
    "num_layers",
    26,
    "Number of Transformer layers for Gemma2 2B model.",
)


# pylint: disable=too-many-positional-arguments,too-many-arguments
def get_gemma2_2b_specs(
    vocab_size: int = 256000,
    num_layers: int = 26,
    hidden_size: int = 2304,
    intermediate_size: int = 9216,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 256,
) -> List[Tuple[str, Tuple[int, ...]]]:
  """Generates parameter names and shapes for the Gemma2 2B model."""
  q_dim = num_heads * head_dim  # 2048
  kv_dim = num_kv_heads * head_dim  # 1024

  specs: List[Tuple[str, Tuple[int, ...]]] = [("embedder.input_embedding", (vocab_size, hidden_size))]
  for layer_idx in range(num_layers):
    specs.extend(
        [
            (f"layers.{layer_idx}.q_proj", (hidden_size, q_dim)),
            (f"layers.{layer_idx}.k_proj", (hidden_size, kv_dim)),
            (f"layers.{layer_idx}.v_proj", (hidden_size, kv_dim)),
            (f"layers.{layer_idx}.o_proj", (q_dim, hidden_size)),
            (f"layers.{layer_idx}.gate_proj", (hidden_size, intermediate_size)),
            (f"layers.{layer_idx}.up_proj", (hidden_size, intermediate_size)),
            (f"layers.{layer_idx}.down_proj", (intermediate_size, hidden_size)),
            (f"layers.{layer_idx}.input_layernorm", (hidden_size,)),
            (f"layers.{layer_idx}.post_attention_layernorm", (hidden_size,)),
            (f"layers.{layer_idx}.post_feedforward_layernorm", (hidden_size,)),
        ]
    )
  specs.append(("final_norm", (hidden_size,)))
  return specs


def resolve_local_ip() -> str:
  """Resolves local host IP address via outbound UDP probe with fallback."""
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect(("8.8.8.8", 80))
      return s.getsockname()[0]
  except Exception:  # pylint: disable=broad-exception-caught
    return socket.gethostbyname(socket.gethostname())


def create_sharding_for_shape(shape: Tuple[int, ...], mesh: Mesh) -> NamedSharding:
  """Creates a NamedSharding for a given shape partitioned across mesh."""
  num_devices = len(mesh.devices.flatten())
  if len(shape) == 2:
    if shape[0] % num_devices == 0:
      return NamedSharding(mesh, P("data", None))
    if shape[1] % num_devices == 0:
      return NamedSharding(mesh, P(None, "data"))
    return NamedSharding(mesh, P(None, None))
  if len(shape) == 1:
    if shape[0] % num_devices == 0:
      return NamedSharding(mesh, P("data"))
    return NamedSharding(mesh, P(None))
  return NamedSharding(mesh, P(None))


def create_source_synthetic_weights(specs: Sequence[Tuple[str, Tuple[int, ...]]], mesh: Mesh) -> List[jax.Array]:
  """Creates synthetic JAX sharded arrays for source worker."""
  arrays: List[jax.Array] = []
  for idx, (_, shape) in enumerate(specs):
    sharding = create_sharding_for_shape(shape, mesh)
    arr = jax.device_put(jnp.ones(shape, dtype=jnp.float32) * (idx + 1.0), sharding)
    arrays.append(arr)
  jax.tree.map(lambda x: x.block_until_ready(), arrays)
  return arrays


def create_destination_synthetic_weights(specs: Sequence[Tuple[str, Tuple[int, ...]]], dst_mesh: Mesh) -> List[jax.Array]:
  """Creates synthetic JAX sharded zero arrays for destination worker."""
  arrays: List[jax.Array] = []
  for _, shape in specs:
    dst_sharding = create_sharding_for_shape(shape, dst_mesh)
    arr = jax.device_put(jnp.zeros(shape, dtype=jnp.float32), dst_sharding)
    arrays.append(arr)
  jax.tree.map(lambda x: x.block_until_ready(), arrays)
  return arrays


# pylint: disable=too-many-positional-arguments,too-many-arguments
def run_controller(
    controller_port: int,
    total_bytes: int,
    num_tensors: int,
    iterations: int,
    warmup_iterations: int,
    profile_dir: str = "",
    timeout: float = 300.0,
) -> dict[str, Any]:
  """Runs the RaidenController lifecycle, transfers, and benchmark measurements."""
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient()
  controller = raiden_controller.RaidenController(
      port=controller_port,
      worker_rpc_client=worker_rpc_client,
  )
  server = raiden_controller.RaidenControllerServer(controller)
  bound_port = server.start()
  logging.info(
      "RaidenControllerServer listening on port %d (requested: %d)",
      bound_port,
      controller_port,
  )

  src_unit = raiden_controller.RaidenId(
      job_name="source",
      job_replica_id="0",
      data_name="weights",
      data_replica_idx=0,
  )
  dst_unit = raiden_controller.RaidenId(
      job_name="destination",
      job_replica_id="0",
      data_name="weights",
      data_replica_idx=0,
  )

  logging.info("Waiting for source and destination workers to register...")
  t_wait_start = time.time()
  while time.time() - t_wait_start < timeout:
    metadata = controller.get_all_metadata()
    registered_units = {
        raiden_controller.RaidenId(
            m.unit.job_name,
            m.unit.job_replica_id,
            m.unit.data_name,
            m.unit.data_replica_idx,
        )
        for m in metadata
    }
    if src_unit in registered_units and dst_unit in registered_units:
      logging.info("Both source and destination workers successfully registered with" " controller.")
      break
    time.sleep(1.0)
  else:
    server.stop()
    raise TimeoutError("Timed out waiting for worker self-registrations.")

  # 1. Warmup transfers
  for w in range(warmup_iterations):
    logging.info("Running warmup transfer %d/%d...", w + 1, warmup_iterations)
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        req_id=f"warmup_{w}",
        uuid=w + 1,
        expected_block_count=num_tensors,
    )
    if future.try_start():
      asyncio.run(future.wait())
    else:
      future.wait_threadsafe()

  # 2. Profiling (optional)
  if profile_dir:
    os.makedirs(profile_dir, exist_ok=True)
    jax.profiler.start_trace(profile_dir)
    logging.info("Started JAX profiler trace in %s", profile_dir)

  # 3. Timed benchmark transfers
  latencies: List[float] = []
  total_mb = total_bytes / (1024 * 1024)
  logging.info(
      "\n--- Starting Benchmark Transfers (Payload: %.2f MB / %.2f GB) ---",
      total_mb,
      total_bytes / 1e9,
  )

  for it in range(iterations):
    t_start = time.perf_counter()
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        req_id=f"bench_{it}",
        uuid=100 + it,
        expected_block_count=num_tensors,
    )
    if future.try_start():
      asyncio.run(future.wait())
    else:
      future.wait_threadsafe()
    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000.0
    latencies.append(elapsed_ms)
    iter_bw_gbps = (total_bytes / 1e9) / (elapsed_ms / 1000.0)
    logging.info(
        "  Iteration %d/%d: %.2f ms | D2H+H2H Bandwidth: %.2f GB/s",
        it + 1,
        iterations,
        elapsed_ms,
        iter_bw_gbps,
    )

  if profile_dir:
    jax.profiler.stop_trace()
    logging.info("Stopped JAX profiler trace.")

  avg_latency_ms = float(np.mean(latencies))
  min_latency_ms = float(np.min(latencies))
  max_latency_ms = float(np.max(latencies))
  avg_bw_gbps = (total_bytes / 1e9) / (avg_latency_ms / 1000.0)
  peak_bw_gbps = (total_bytes / 1e9) / (min_latency_ms / 1000.0)

  print("\n=========================================================")
  print("TPU Raiden Multi-Client Weight Transfer Benchmark Summary")
  print("=========================================================")
  print(f"Payload Size           : {total_mb:.2f} MB ({total_bytes / 1e9:.3f} GB)")
  print(f"Iterations             : {iterations} (Warmup: {warmup_iterations})")
  print(f"Avg Latency            : {avg_latency_ms:.3f} ms")
  print(f"Min Latency            : {min_latency_ms:.3f} ms")
  print(f"Max Latency            : {max_latency_ms:.3f} ms")
  print(f"Avg D2H+H2H Bandwidth  : {avg_bw_gbps:.3f} GB/s")
  print(f"Peak D2H+H2H Bandwidth : {peak_bw_gbps:.3f} GB/s")
  print("=========================================================\n")

  # 4. Shutdown workers and stop server
  controller_network_client = controller.worker_rpc_client
  logging.info("Sending shutdown command to all registered workers...")
  asyncio.run(controller_network_client.shutdown_workers())
  server.stop()
  logging.info("Controller server stopped successfully.")

  return {
      "payload_bytes": total_bytes,
      "avg_latency_ms": avg_latency_ms,
      "min_latency_ms": min_latency_ms,
      "avg_bw_gbps": avg_bw_gbps,
      "peak_bw_gbps": peak_bw_gbps,
  }


def run_source(
    controller_address: str,
    specs: Sequence[Tuple[str, Tuple[int, ...]]],
    local_ip: str,
    data_port: int = 0,
    listener_port: int = 0,
) -> None:
  """Initializes synthetic weights, registers source worker, and waits."""
  devices = jax.local_devices()
  mesh = Mesh(np.array(devices), axis_names=("data",))
  logging.info("Source worker creating synthetic sharded arrays on mesh %s...", mesh)
  jax_arrays = create_source_synthetic_weights(specs, mesh)

  ws = weight_synchronizer.WeightSynchronizer(
      jax_arrays,
      local_port=data_port,
      listener_port=listener_port,
      bind_ip=local_ip,
      unsafe_skip_buffer_lock=True,
  )
  logging.info(
      "Source WeightSynchronizer initialized (data_port=%s, listener_port=%s)",
      ws.local_port,
      ws.listener_port,
  )

  src_unit = raiden_controller.RaidenId(
      job_name="source",
      job_replica_id="0",
      data_name="weights",
      data_replica_idx=0,
  )
  client_facade = raiden_controller.RaidenControllerClientFacade(controller_address)
  client_facade.register_work_unit(
      unit=src_unit,
      shards=[f"{local_ip}:{ws.local_port}"],
      control_plane_rpc_address=f"{local_ip}:{ws.listener_port}",
  )
  logging.info("Source worker registered with controller at %s", controller_address)

  logging.info("Source worker active. Waiting for transfer and shutdown signals...")
  while ws.is_listener_active:
    time.sleep(1)
  logging.info("Source worker received shutdown command. Exiting cleanly.")


# pylint: disable=too-many-positional-arguments,too-many-arguments
def run_destination(
    controller_address: str,
    specs: Sequence[Tuple[str, Tuple[int, ...]]],
    local_ip: str,
    total_bytes: int,
    data_port: int = 0,
    listener_port: int = 0,
    verify: bool = True,
) -> None:
  """Initializes destination weights, registers worker, awaits transfers, runs H2D."""
  dst_mesh = Mesh(np.array(jax.local_devices()), axis_names=("data",))
  logging.info("Destination worker creating initial zero arrays on mesh %s...", dst_mesh)
  jax_arrays = create_destination_synthetic_weights(specs, dst_mesh)

  ws = weight_synchronizer.WeightSynchronizer(
      jax_arrays,
      local_port=data_port,
      listener_port=listener_port,
      bind_ip=local_ip,
      unsafe_skip_buffer_lock=True,
  )
  logging.info(
      "Destination WeightSynchronizer initialized (data_port=%s," " listener_port=%s)",
      ws.local_port,
      ws.listener_port,
  )

  dst_unit = raiden_controller.RaidenId(
      job_name="destination",
      job_replica_id="0",
      data_name="weights",
      data_replica_idx=0,
  )
  client_facade = raiden_controller.RaidenControllerClientFacade(controller_address)
  client_facade.register_work_unit(
      unit=dst_unit,
      shards=[f"{local_ip}:{ws.local_port}"],
      control_plane_rpc_address=f"{local_ip}:{ws.listener_port}",
  )
  logging.info("Destination worker registered with controller at %s", controller_address)

  logging.info("Destination worker active. Waiting for transfer and shutdown signals...")
  while ws.is_listener_active:
    time.sleep(1)
  logging.info("Destination worker received shutdown command. Running H2D...")

  t0 = time.perf_counter()
  ws.h2d()
  jax.tree.map(lambda x: x.block_until_ready(), jax_arrays)
  h2d_duration = time.perf_counter() - t0
  h2d_bw_gbps = (total_bytes / 1e9) / h2d_duration
  logging.info(
      "Destination H2D copy completed in %.2f ms (Bandwidth: %.2f GB/s)",
      h2d_duration * 1000.0,
      h2d_bw_gbps,
  )

  if verify:
    logging.info("Verifying destination weight values for correctness...")
    for idx, arr in enumerate(jax_arrays):
      np_arr = np.asarray(arr)
      expected_val = float(idx + 1.0)
      np.testing.assert_allclose(np_arr, expected_val, rtol=1e-5)
    logging.info("Destination correctness verification PASSED.")

  logging.info("Destination worker exiting cleanly.")


# pylint: disable=too-many-positional-arguments,too-many-arguments
def run_all(
    controller_port: int,
    specs: Sequence[Tuple[str, Tuple[int, ...]]],
    total_bytes: int,
    iterations: int,
    warmup_iterations: int,
    verify: bool = True,
    profile_dir: str = "",
) -> None:
  """Runs controller, destination worker, and source worker in separate threads."""
  local_ip = "127.0.0.1"
  controller_address = f"{local_ip}:{controller_port}"

  thread_dst = threading.Thread(
      target=run_destination,
      args=(
          controller_address,
          specs,
          local_ip,
          total_bytes,
          0,
          0,
          verify,
      ),
      daemon=True,
  )
  thread_src = threading.Thread(
      target=run_source,
      args=(
          controller_address,
          specs,
          local_ip,
          0,
          0,
      ),
      daemon=True,
  )

  server_ready = threading.Event()

  def _controller_runner():
    server_ready.set()
    run_controller(
        controller_port=controller_port,
        total_bytes=total_bytes,
        num_tensors=len(specs),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        profile_dir=profile_dir,
    )

  thread_ctrl = threading.Thread(target=_controller_runner)
  thread_ctrl.start()
  server_ready.wait()
  time.sleep(0.5)

  thread_dst.start()
  thread_src.start()

  thread_ctrl.join()
  thread_dst.join(timeout=30.0)
  thread_src.join(timeout=30.0)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  local_ip = _LOCAL_IP.value or resolve_local_ip()
  specs = get_gemma2_2b_specs(
      vocab_size=_VOCAB_SIZE.value,
      num_layers=_NUM_LAYERS.value,
  )
  total_bytes = sum(int(np.prod(shape)) * np.dtype(jnp.float32).itemsize for _, shape in specs)
  total_mb = total_bytes / (1024 * 1024)

  logging.info("Role                 : %s", _ROLE.value)
  logging.info("Local IP             : %s", local_ip)
  logging.info(
      "Gemma2 2B Total Bytes: %d bytes (%.2f MB / %.2f GB)",
      total_bytes,
      total_mb,
      total_bytes / 1e9,
  )
  logging.info("Total Model Tensors  : %d", len(specs))

  if _ROLE.value == "controller":
    run_controller(
        controller_port=_CONTROLLER_PORT.value,
        total_bytes=total_bytes,
        num_tensors=len(specs),
        iterations=_ITERATIONS.value,
        warmup_iterations=_WARMUP_ITERATIONS.value,
        profile_dir=_PROFILE_DIR.value,
    )
  elif _ROLE.value == "source":
    run_source(
        controller_address=_CONTROLLER_ADDRESS.value,
        specs=specs,
        local_ip=local_ip,
        data_port=_DATA_PORT.value,
        listener_port=_LISTENER_PORT.value,
    )
  elif _ROLE.value == "destination":
    run_destination(
        controller_address=_CONTROLLER_ADDRESS.value,
        specs=specs,
        local_ip=local_ip,
        total_bytes=total_bytes,
        data_port=_DATA_PORT.value,
        listener_port=_LISTENER_PORT.value,
        verify=_VERIFY.value,
    )
  elif _ROLE.value == "all":
    run_all(
        controller_port=_CONTROLLER_PORT.value,
        specs=specs,
        total_bytes=total_bytes,
        iterations=_ITERATIONS.value,
        warmup_iterations=_WARMUP_ITERATIONS.value,
        verify=_VERIFY.value,
        profile_dir=_PROFILE_DIR.value,
    )


if __name__ == "__main__":
  app.run(main)
