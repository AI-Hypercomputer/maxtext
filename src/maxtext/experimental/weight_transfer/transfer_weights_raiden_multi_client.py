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

"""Distributed Multi-Process TPU Raiden 2-Slice Benchmark.

Orchestrates real TPU weight transfer across independent processes (Controller,
Source / Trainer Worker, Destination / Sampler Worker) across 2 TPU slices on GKE.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import socket
import threading
import time
from typing import List, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

if "XLA_FLAGS" not in os.environ:
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

try:
  import google.protobuf.runtime_version

  google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except (ImportError, AttributeError):
  pass

from tpu_raiden.api.jax import weight_synchronizer
from tpu_raiden.frameworks.jax import resharding_planner
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2


FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "role",
    "controller",
    ["controller", "source", "destination", "all"],
    "Role of this process: 'controller', 'source', 'destination', or 'all'.",
)
flags.DEFINE_integer("controller_port", 29500, "Port for the central Raiden controller server.")
flags.DEFINE_string(
    "controller_address",
    "127.0.0.1:29500",
    "Address of the central Raiden controller (host:port).",
)
flags.DEFINE_string(
    "local_ip",
    "",
    "Explicit local IP address to bind/advertise. If empty, auto-resolved.",
)
flags.DEFINE_integer("num_decoder_layers", 26, "Number of Transformer decoder layers.")
flags.DEFINE_integer("benchmark_iterations", 3, "Number of timed benchmark iterations.")
flags.DEFINE_integer("warmup_iterations", 1, "Number of warmup iterations before profiling.")
flags.DEFINE_integer("hidden_dim", 4096, "Model hidden dimension.")
flags.DEFINE_integer("mlp_dim", 14336, "Model MLP intermediate dimension.")
flags.DEFINE_integer("vocab_size", 128256, "Vocabulary size.")
flags.DEFINE_integer("group_size", 128, "Number of weights to group per transfer request.")
flags.DEFINE_integer("parallelism", 16, "Number of parallel TCP socket stream workers.")
flags.DEFINE_integer(
    "raiden_transport_coalesce_window_bytes",
    67108864,
    "Host-side coalescing buffer size (bytes) for network transfers.",
)
flags.DEFINE_bool("verify_parity", True, "Whether to verify 100% numerical parity.")


@dataclass
class DmaEventRecord:
  name: str
  shape: Tuple[int, ...]
  byte_size: int
  stage: str  # 'D2H' or 'H2D'
  start_time_s: float
  end_time_s: float
  duration_ms: float
  bandwidth_gb_s: float


def resolve_local_ip() -> str:
  """Resolves local host IP address via outbound UDP probe with fallback."""
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect(("8.8.8.8", 80))
      return s.getsockname()[0]
  except Exception:  # pylint: disable=broad-exception-caught
    return socket.gethostbyname(socket.gethostname())


def build_transformer_specs(
    num_layers: int,
    hidden_dim: int,
    mlp_dim: int,
    vocab_size: int,
) -> list[tuple[tuple[int, ...], P, str]]:
  """Builds tensor shapes, sharding specs, and names for a Transformer model."""
  specs: list[tuple[tuple[int, ...], P, str]] = [
      ((vocab_size, hidden_dim), P("fsdp", "tp"), "tok_embeddings"),
  ]
  for layer_idx in range(num_layers):
    specs.extend(
        [
            ((hidden_dim, hidden_dim), P("fsdp", "tp"), f"layer_{layer_idx}.attention.wq"),
            ((hidden_dim, hidden_dim), P("fsdp", "tp"), f"layer_{layer_idx}.attention.wk"),
            ((hidden_dim, hidden_dim), P("fsdp", "tp"), f"layer_{layer_idx}.attention.wv"),
            ((hidden_dim, hidden_dim), P("tp", "fsdp"), f"layer_{layer_idx}.attention.wo"),
            ((hidden_dim, mlp_dim), P("fsdp", "tp"), f"layer_{layer_idx}.mlp.gate_proj"),
            ((hidden_dim, mlp_dim), P("fsdp", "tp"), f"layer_{layer_idx}.mlp.up_proj"),
            ((mlp_dim, hidden_dim), P("tp", "fsdp"), f"layer_{layer_idx}.mlp.down_proj"),
            ((hidden_dim,), P("fsdp"), f"layer_{layer_idx}.input_layernorm"),
            ((hidden_dim,), P("fsdp"), f"layer_{layer_idx}.post_attention_layernorm"),
            ((hidden_dim,), P("fsdp"), f"layer_{layer_idx}.post_feedforward_layernorm"),
        ]
    )
  specs.append(((hidden_dim,), P("fsdp"), "final_norm"))
  return specs


def build_variable_protos(
    arrays: Sequence[jax.Array],
    shardings: Sequence[NamedSharding],
    names: Sequence[str],
) -> list[raiden_service_pb2.VariableMetadataProto]:
  """Constructs Raiden VariableMetadataProto list for controller registration."""
  protos: list[raiden_service_pb2.VariableMetadataProto] = []
  for idx, (arr, shd, name) in enumerate(zip(arrays, shardings, names)):
    g_shape = arr.shape
    l_shape = shd.shard_shape(g_shape)
    s_shape = [g // l for g, l in zip(g_shape, l_shape)]
    layout = tuple(range(len(g_shape) - 1, -1, -1))
    spec_axes: list[str] = []
    if hasattr(shd, "spec"):
      for axis in shd.spec:
        if axis is None:
          spec_axes.append("")
        elif isinstance(axis, str):
          spec_axes.append(axis)
        else:
          spec_axes.append(",".join(axis))
    proto = raiden_service_pb2.VariableMetadataProto(
        name=name,
        shape=g_shape,
        mesh_shape=s_shape,
        layout=layout,
        item_size=arr.dtype.itemsize,
        layer_idx=idx,
        sharding_spec=spec_axes,
    )
    protos.append(proto)
  return protos


def print_dma_size_distribution(
    names: List[str],
    dma_events: List[DmaEventRecord],
    total_bytes: int,
):
  """Computes and prints bucketed size distribution and aggregated DMA metrics."""
  buckets = [
      ("< 1 MB", lambda b: b < 1e6),
      ("1 MB - 10 MB", lambda b: 1e6 <= b < 10e6),
      ("10 MB - 50 MB", lambda b: 10e6 <= b < 50e6),
      ("50 MB - 100 MB", lambda b: 50e6 <= b < 100e6),
      ("> 100 MB", lambda b: b >= 100e6),
  ]
  d2h_map = {e.name: e for e in dma_events if e.stage == "D2H"}
  h2d_map = {e.name: e for e in dma_events if e.stage == "H2D"}

  print("\n" + "=" * 115)
  print("STAGE 6: DMA CALLS SIZE DISTRIBUTION & AGGREGATE THROUGHPUT")
  print("=" * 115)
  print(
      f"{'Size Bucket':<16} | {'Count':<5} | {'Total (MB)':<11} |"
      f" {'Payload %':<9} | {'Avg D2H (ms)':<12} | {'D2H BW (GB/s)':<13} |"
      f" {'Avg H2D (ms)':<12} | {'H2D BW (GB/s)':<13}"
  )
  print("-" * 115)

  for label, predicate in buckets:
    bucket_tensors = [name for name in names if predicate(d2h_map[name].byte_size)]
    count = len(bucket_tensors)
    if count == 0:
      continue
    b_bytes = sum(d2h_map[name].byte_size for name in bucket_tensors)
    b_mb = b_bytes / 1e6
    pct = (b_bytes / total_bytes) * 100.0

    avg_d2h_ms = float(np.mean([d2h_map[name].duration_ms for name in bucket_tensors]))
    avg_d2h_bw = float(np.mean([d2h_map[name].bandwidth_gb_s for name in bucket_tensors]))
    avg_h2d_ms = float(np.mean([h2d_map[name].duration_ms for name in bucket_tensors]))
    avg_h2d_bw = float(np.mean([h2d_map[name].bandwidth_gb_s for name in bucket_tensors]))

    print(
        f"{label:<16} | {count:<5} | {b_mb:>11.2f} | {pct:>8.2f}% |"
        f" {avg_d2h_ms:>12.3f} | {avg_d2h_bw:>13.2f} | {avg_h2d_ms:>12.3f} |"
        f" {avg_h2d_bw:>13.2f}"
    )
  print("=" * 115 + "\n")


def print_dma_timeline_visualization(dma_events: List[DmaEventRecord]):
  """Renders a text Gantt chart visualizing execution timeline of DMA calls."""
  if not dma_events:
    return

  t_base = min(e.start_time_s for e in dma_events)
  t_end_max = max(e.end_time_s for e in dma_events)
  total_span_ms = (t_end_max - t_base) * 1000.0
  bar_width = 40

  print("\n" + "=" * 125)
  print("STAGE 7: TEMPORAL DMA TIMELINE VISUALIZATION (Total Profiling Window:" f" {total_span_ms:.2f} ms)")
  print("=" * 125)
  print(
      f"{'Operation / Layer':<42} | {'Stage':<5} | {'Start (ms)':<10} |"
      f" {'End (ms)':<10} | {'Dur (ms)':<9} | Timeline Visualization"
  )
  print("-" * 125)

  for e in dma_events:
    start_ms = (e.start_time_s - t_base) * 1000.0
    end_ms = (e.end_time_s - t_base) * 1000.0
    dur_ms = e.duration_ms

    start_col = int((start_ms / total_span_ms) * (bar_width - 1))
    end_col = int((end_ms / total_span_ms) * (bar_width - 1))
    end_col = max(end_col, start_col)

    char_sym = "#" if e.stage == "D2H" else "="
    timeline_chars = [" "] * bar_width
    for c in range(start_col, end_col + 1):
      timeline_chars[c] = char_sym
    bar_str = "[" + "".join(timeline_chars) + "]"

    print(f"{e.name:<42} | {e.stage:<5} | {start_ms:>10.2f} | {end_ms:>10.2f} |" f" {dur_ms:>9.3f} | {bar_str}")
  print("=" * 125)
  print("Legend: [#] = Device-to-Host (D2H) Staging | [=] = Host-to-Device" " (H2D) Ingestion\n")


def run_controller(
    controller_port: int,
    total_bytes: int,
    iterations: int,
    warmup_iterations: int,
    timeout: float = 300.0,
) -> None:
  """Runs the central Controller server coordinating cross-slice transfers."""
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

  src_unit = raiden_controller.RaidenId("trainer", "0", "weights")
  dst_unit = raiden_controller.RaidenId("sampler", "0", "weights")

  logging.info("Waiting for Trainer and Sampler slice workers to register...")
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
      logging.info("Both Trainer (Slice 0) and Sampler (Slice 1) workers registered" " successfully!")
      break
    time.sleep(1.0)
  else:
    server.stop()
    raise TimeoutError("Timed out waiting for slice workers to register.")

  # Warmup Transfers
  for w in range(warmup_iterations):
    logging.info("Running warmup transfer %d/%d...", w + 1, warmup_iterations)
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=0,
        uuid=10 + w,
        req_id=f"warmup_{w}",
        group_size=FLAGS.group_size,
    )
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(future.wait())
    finally:
      loop.close()

  # Timed Benchmark Transfers
  total_gb = total_bytes / 1e9
  h2h_latencies: List[float] = []

  print("\n" + "=" * 90)
  print(f"STAGE 3: EXECUTING {iterations} TIMED CROSS-SLICE H2H TCP TRANSFERS")
  print("=" * 90)

  for it in range(iterations):
    t_start = time.perf_counter()
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=0,
        uuid=100 + it,
        req_id=f"bench_{it}",
        group_size=FLAGS.group_size,
    )
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(future.wait())
    finally:
      loop.close()
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    h2h_latencies.append(elapsed)
    print(
        f"  Iteration {it + 1}/{iterations}: {elapsed * 1000.0:.2f} ms | H2H" f" Bandwidth: {total_gb / elapsed:.2f} GB/s"
    )

  mean_h2h = float(np.mean(h2h_latencies))
  print("-" * 90)
  print(
      f"Mean H2H Cross-Slice Transfer Time : {mean_h2h * 1000.0:.3f} ms |" f" Bandwidth: {total_gb / mean_h2h:.2f} GB/s"
  )
  print("=" * 90 + "\n")

  logging.info("Sending shutdown command to slice workers...")
  asyncio.run(controller.worker_rpc_client.shutdown_workers())
  server.stop()
  logging.info("Controller server stopped successfully.")


def run_source(
    controller_address: str,
    specs: Sequence[Tuple[Tuple[int, ...], P, str]],
    local_ip: str,
) -> None:
  """Runs the Source (Trainer) worker on Slice 0 with local PjRt TPUs."""
  devices = jax.local_devices()
  mesh = Mesh(np.array(devices).reshape(-1, 1), ("fsdp", "tp"))
  logging.info("Slice 0 Source Worker initialized on Mesh %s", mesh)

  src_arrays: List[jax.Array] = []
  src_shardings: List[NamedSharding] = []
  names: List[str] = []
  rng = np.random.RandomState(42)

  for shape, pspec, name in specs:
    names.append(name)
    shd = NamedSharding(mesh, pspec)
    src_shardings.append(shd)
    np_val = rng.standard_normal(shape).astype(np.float32)
    arr = jax.device_put(np_val, shd)
    arr.block_until_ready()
    src_arrays.append(arr)

  ws = weight_synchronizer.WeightSynchronizer(
      jax_arrays=src_arrays,
      local_port=0,
      parallelism=FLAGS.parallelism,
      listener_port=0,
      bind_ip=local_ip,
      unsafe_skip_buffer_lock=True,
  )
  logging.info(
      "Slice 0 WeightSynchronizer initialized (local_port=%d," " listener_port=%d)",
      ws.local_port,
      ws.listener_port,
  )

  src_unit = raiden_controller.RaidenId("trainer", "0", "weights")
  client_facade = raiden_controller.RaidenControllerClientFacade(controller_address)
  client_facade.register_work_unit(
      unit=src_unit,
      shards=[f"{local_ip}:{ws.local_port}"] * len(devices),
      control_plane_rpc_address=f"{local_ip}:{ws.listener_port}",
      mesh_shape=[mesh.shape["fsdp"], mesh.shape["tp"]],
      variables=build_variable_protos(src_arrays, src_shardings, names),
      mesh_axes=list(mesh.axis_names),
  )
  logging.info("Slice 0 Worker registered with controller at %s", controller_address)

  # Stage 2: Device-to-Host (D2H) Staging
  t0 = time.perf_counter()
  ws.d2h()
  t1 = time.perf_counter()
  total_bytes = sum(int(np.prod(s[0])) * 4 for s in specs)
  d2h_ms = (t1 - t0) * 1000.0
  print(f"Slice 0 D2H Staging Completed in {d2h_ms:.2f} ms | Bandwidth:" f" {(total_bytes/1e9)/(t1-t0):.2f} GB/s")

  logging.info("Slice 0 active. Serving transfer requests...")
  while ws.is_listener_active:
    time.sleep(1)
  logging.info("Slice 0 worker received shutdown signal. Exiting.")


def run_destination(
    controller_address: str,
    specs: Sequence[Tuple[Tuple[int, ...], P, str]],
    local_ip: str,
) -> None:
  """Runs the Destination (Sampler) worker on Slice 1 with local PjRt TPUs."""
  devices = jax.local_devices()
  mesh = Mesh(np.array(devices).reshape(1, -1), ("fsdp", "tp"))
  logging.info("Slice 1 Destination Worker initialized on Mesh %s", mesh)

  dst_arrays: List[jax.Array] = []
  dst_shardings: List[NamedSharding] = []
  names: List[str] = []

  for shape, pspec, name in specs:
    names.append(name)
    shd = NamedSharding(mesh, pspec)
    dst_shardings.append(shd)
    arr = jax.device_put(np.zeros(shape, dtype=np.float32), shd)
    arr.block_until_ready()
    dst_arrays.append(arr)

  ws = weight_synchronizer.WeightSynchronizer(
      jax_arrays=dst_arrays,
      local_port=0,
      parallelism=FLAGS.parallelism,
      listener_port=0,
      bind_ip=local_ip,
      unsafe_skip_buffer_lock=True,
  )
  logging.info(
      "Slice 1 WeightSynchronizer initialized (local_port=%d," " listener_port=%d)",
      ws.local_port,
      ws.listener_port,
  )

  dst_unit = raiden_controller.RaidenId("sampler", "0", "weights")
  client_facade = raiden_controller.RaidenControllerClientFacade(controller_address)
  client_facade.register_work_unit(
      unit=dst_unit,
      shards=[f"{local_ip}:{ws.local_port}"] * len(devices),
      control_plane_rpc_address=f"{local_ip}:{ws.listener_port}",
      mesh_shape=[mesh.shape["fsdp"], mesh.shape["tp"]],
      variables=build_variable_protos(dst_arrays, dst_shardings, names),
      mesh_axes=list(mesh.axis_names),
  )
  logging.info("Slice 1 Worker registered with controller at %s", controller_address)

  logging.info("Slice 1 active. Awaiting transfer requests...")
  while ws.is_listener_active:
    time.sleep(1)

  # Stage 4: Host-to-Device (H2D) Ingestion
  logging.info("Slice 1 received completion signal. Running H2D Ingestion...")
  t0 = time.perf_counter()
  ws.h2d()
  t1 = time.perf_counter()
  total_bytes = sum(int(np.prod(s[0])) * 4 for s in specs)
  h2d_ms = (t1 - t0) * 1000.0
  print(f"Slice 1 H2D Ingestion Completed in {h2d_ms:.2f} ms | Bandwidth:" f" {(total_bytes/1e9)/(t1-t0):.2f} GB/s")

  # Stage 5: Numerical Parity Verification
  if FLAGS.verify_parity:
    print("\nVerifying 100% numerical parity on Slice 1...")
    rng = np.random.RandomState(42)
    for idx, (shape, _, name) in enumerate(specs):
      expected = rng.standard_normal(shape).astype(np.float32)
      actual = np.asarray(jax.device_get(dst_arrays[idx]))
      np.testing.assert_allclose(
          actual,
          expected,
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"Divergence detected in tensor {name} on Slice 1!",
      )
    print(f"✓ [SUCCESS] 100% numerical parity verified across all {len(specs)} tensors.")


def run_all(
    controller_port: int,
    specs: Sequence[Tuple[Tuple[int, ...], P, str]],
    total_bytes: int,
    iterations: int,
    warmup_iterations: int,
) -> None:
  """Runs controller, source worker, and destination worker in separate threads."""
  local_ip = "127.0.0.1"
  controller_address = f"{local_ip}:{controller_port}"

  thread_src = threading.Thread(
      target=run_source,
      args=(controller_address, specs, local_ip),
      daemon=True,
  )
  thread_dst = threading.Thread(
      target=run_destination,
      args=(controller_address, specs, local_ip),
      daemon=True,
  )

  ready = threading.Event()

  def _controller():
    ready.set()
    run_controller(
        controller_port=controller_port,
        total_bytes=total_bytes,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )

  thread_ctrl = threading.Thread(target=_controller)
  thread_ctrl.start()
  ready.wait()
  time.sleep(0.5)

  thread_src.start()
  thread_dst.start()

  thread_ctrl.join()
  thread_src.join(timeout=30.0)
  thread_dst.join(timeout=30.0)


def main(argv: Sequence[str]) -> None:
  del argv
  os.environ["RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES"] = str(FLAGS.raiden_transport_coalesce_window_bytes)

  local_ip = FLAGS.local_ip or resolve_local_ip()
  specs = build_transformer_specs(
      FLAGS.num_decoder_layers,
      FLAGS.hidden_dim,
      FLAGS.mlp_dim,
      FLAGS.vocab_size,
  )
  total_bytes = sum(int(np.prod(s[0])) * 4 for s in specs)

  # Stage 1: Resharding Plan Generation Overhead
  if FLAGS.role in ("controller", "all"):
    print("=" * 90)
    print("STAGE 1: RESHARDING PLAN GENERATION OVERHEAD")
    print("=" * 90)
    devices = jax.local_devices()
    half = max(len(devices) // 2, 1)
    s_mesh = Mesh(np.array(devices[:half]).reshape(-1, 1), ("fsdp", "tp"))
    d_mesh = Mesh(np.array(devices[half : 2 * half]).reshape(1, -1), ("fsdp", "tp"))
    t0 = time.perf_counter()
    total_chunks = 0
    for shape, pspec, _ in specs:
      if len(shape) == 2:
        s_shd = NamedSharding(s_mesh, pspec)
        d_shd = NamedSharding(d_mesh, pspec)
        plan = resharding_planner.make_resharding_plan(shape, s_shd, d_shd)
        total_chunks += len(plan)
    plan_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Total Variables Evaluated : {len(specs)}")
    print(f"Total 2D Reshard Chunks    : {total_chunks}")
    print(f"Total Planning Time       : {plan_ms:.3f} ms" f" ({plan_ms*1000.0/len(specs):.2f} us/var)")
    print("=" * 90 + "\n")

  if FLAGS.role == "controller":
    run_controller(
        controller_port=FLAGS.controller_port,
        total_bytes=total_bytes,
        iterations=FLAGS.benchmark_iterations,
        warmup_iterations=FLAGS.warmup_iterations,
    )
  elif FLAGS.role == "source":
    run_source(
        controller_address=FLAGS.controller_address,
        specs=specs,
        local_ip=local_ip,
    )
  elif FLAGS.role == "destination":
    run_destination(
        controller_address=FLAGS.controller_address,
        specs=specs,
        local_ip=local_ip,
    )
  elif FLAGS.role == "all":
    run_all(
        controller_port=FLAGS.controller_port,
        specs=specs,
        total_bytes=total_bytes,
        iterations=FLAGS.benchmark_iterations,
        warmup_iterations=FLAGS.warmup_iterations,
    )


if __name__ == "__main__":
  app.run(main)
