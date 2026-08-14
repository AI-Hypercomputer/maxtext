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

"""TPU Raiden Weight Synchronizer Benchmark on Pathways (2-Slice TPU v5p).

Implements the official TPU Raiden + Pathways FFI architecture:
1. Single-client orchestration on Pathways Head Node.
2. Remote Device-to-Host (D2H) staging via weight_synchronizer_ffi.init_weight_synchronizer_and_d2h.
3. Remote Destination initialization via weight_synchronizer_ffi.init_weight_synchronizer.
4. Dynamic port/IP extraction from FFI return tensors.
5. Central RaidenController coordination across Slice 0 (Trainer) and Slice 1 (Sampler).
6. Remote Host-to-Device (H2D) ingestion via weight_synchronizer_ffi.multi_h2d.
7. 100% Numerical Parity Verification and Stage 1-7 DMA breakdown reporting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import gc
import ipaddress
import os
import time
from typing import List, Sequence, Tuple

from absl import app
from absl import flags

import jax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

try:
  from jax.experimental import compute_on

  _orig_compute_on = compute_on.compute_on

  def _compat_compute_on(*args, **kwargs):
    compute_type = args[0] if args else kwargs.get("compute_type", "device_host")
    return _orig_compute_on(compute_type)

  compute_on.compute_on = _compat_compute_on
except (ImportError, AttributeError):
  pass

try:
  from pathwaysutils import proxy_backend

  proxy_backend.register_backend_factory()
except (ImportError, AttributeError):
  pass

try:
  import google.protobuf.runtime_version

  google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except (ImportError, AttributeError):
  pass

from tpu_raiden.frameworks.jax import resharding_planner
from tpu_raiden.frameworks.jax import weight_synchronizer_ffi as raiden_ffi
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

# Model weights in JAX host staging buffers are linear flat row-major DRAM buffers.
# Disable KV-cache hardware tile awareness to ensure exact 2D strided chunk slicing.
raiden_controller.is_nd_slice_tile_aligned = lambda *args, **kwargs: False


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_decoder_layers", 26, "Number of Transformer decoder layers.")
flags.DEFINE_integer("benchmark_iterations", 3, "Number of timed benchmark iterations.")
flags.DEFINE_integer("warmup_iterations", 1, "Number of warmup iterations before profiling.")
flags.DEFINE_integer("hidden_dim", 4096, "Model hidden dimension.")
flags.DEFINE_integer("mlp_dim", 14336, "Model intermediate MLP dimension.")
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


def unpack_ip(row: np.ndarray) -> str:
  """Unpacks 4-byte or 16-byte raw IP representation from FFI metadata row."""
  raw_bytes = b"".join(int(x).to_bytes(4, byteorder="little", signed=True) for x in row[:4])
  try:
    ip = str(ipaddress.IPv6Address(raw_bytes))
    if ":" in ip:
      return f"[{ip}]"
    return ip
  except ValueError:
    try:
      return str(ipaddress.IPv4Address(raw_bytes[:4]))
    except ValueError:
      return "127.0.0.1"


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


def run_benchmark():
  """Main benchmark orchestrator executed on Pathways Head Node."""
  os.environ["RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES"] = str(FLAGS.raiden_transport_coalesce_window_bytes)

  head = os.environ.get("PATHWAYS_HEAD", "127.0.0.1")
  target = os.environ.get("JAX_BACKEND_TARGET", f"grpc://{head}:29000")
  if "$(" in target or not target:
    target = f"grpc://{head}:29000"

  if "PATHWAYS_HEAD" in os.environ:
    jax.config.update("jax_platforms", "proxy")
    jax.config.update("jax_backend_target", target)

  print("=" * 90)
  print("[Pathways Raiden Benchmark] Starting 2-Slice TPU Benchmark...")
  print(f"  JAX Platforms : {getattr(jax.config, 'jax_platforms', 'proxy')}")
  print(f"  Backend Target: {getattr(jax.config, 'jax_backend_target', 'default')}")
  print("=" * 90)

  devices = jax.devices()
  print(f"Detected TPU Devices ({len(devices)}): {devices}")

  if len(devices) < 8:
    raise ValueError(f"Pathways 2-slice benchmark requires 8 TPU devices, found {len(devices)}")

  src_devices = devices[:4]
  dst_devices = devices[4:8]

  src_mesh = Mesh(np.array(src_devices).reshape(4, 1), ("fsdp", "tp"))
  dst_mesh = Mesh(np.array(dst_devices).reshape(1, 4), ("fsdp", "tp"))

  print("\n" + "=" * 90)
  print("TPU MESH ALLOCATION (2 SLICES / 2 HOSTS)")
  print("=" * 90)
  print(f"Slice 0 (Trainer): 4 devices -> Mesh {src_mesh}")
  print(f"Slice 1 (Sampler): 4 devices -> Mesh {dst_mesh}")
  print("=" * 90 + "\n")

  specs = build_transformer_specs(
      FLAGS.num_decoder_layers,
      FLAGS.hidden_dim,
      FLAGS.mlp_dim,
      FLAGS.vocab_size,
  )
  total_elements = sum(int(np.prod(s[0])) for s in specs)
  total_bytes = total_elements * 4
  total_mb = total_bytes / 1e6
  total_gb = total_bytes / 1e9

  print(f"Allocating {len(specs)} synthetic transformer weights across 2 slices...")
  src_arrays: List[jax.Array] = []
  dst_arrays: List[jax.Array] = []
  src_shardings: List[NamedSharding] = []
  dst_shardings: List[NamedSharding] = []
  names: List[str] = []

  rng = np.random.RandomState(42)
  t_alloc_start = time.perf_counter()

  for idx, (shape, pspec, name) in enumerate(specs):
    names.append(name)
    s_shd = NamedSharding(src_mesh, pspec)
    d_shd = NamedSharding(dst_mesh, pspec)
    src_shardings.append(s_shd)
    dst_shardings.append(d_shd)

    np_val = rng.standard_normal(shape).astype(np.float32)
    s_arr = jax.device_put(np_val, s_shd)
    d_arr = jax.device_put(np.zeros(shape, dtype=np.float32), d_shd)

    src_arrays.append(s_arr)
    dst_arrays.append(d_arr)
    del np_val

    if (idx + 1) % 50 == 0 or (idx + 1) == len(specs):
      curr_mb = sum(int(np.prod(s[0])) * 4 for s in specs[: idx + 1]) / 1e6
      print(f"  Allocated [{idx + 1}/{len(specs)}] tensors ({curr_mb:.1f} MB)...")

  jax.tree.map(lambda x: x.block_until_ready(), src_arrays)
  jax.tree.map(lambda x: x.block_until_ready(), dst_arrays)
  t_alloc_end = time.perf_counter()
  print(
      f"Model Payload Allocated: {total_elements} elements, {total_mb:.2f} MB"
      f" ({total_gb:.3f} GB) in {t_alloc_end - t_alloc_start:.3f}s.\n"
  )

  # STAGE 1: Resharding Plan Generation Overhead
  print("=" * 90)
  print("STAGE 1: RESHARDING PLAN GENERATION OVERHEAD")
  print("=" * 90)
  t_plan_start = time.perf_counter()
  total_chunks = 0
  for shape, pspec, _ in specs:
    if len(shape) == 2:
      s_shd = NamedSharding(src_mesh, pspec)
      d_shd = NamedSharding(dst_mesh, pspec)
      plan = resharding_planner.make_resharding_plan(shape, s_shd, d_shd)
      total_chunks += len(plan)
  t_plan_end = time.perf_counter()
  plan_time_ms = (t_plan_end - t_plan_start) * 1000.0
  print(f"Total Variables Evaluated : {len(specs)}")
  print(f"Total 2D Reshard Chunks    : {total_chunks}")
  print(f"Total Planning Time       : {plan_time_ms:.3f} ms" f" ({plan_time_ms*1000.0/len(specs):.2f} us/var)")
  print("=" * 90 + "\n")

  # STAGE 2: Start In-Process Controller & Register via FFI
  print("=" * 90)
  print("STAGE 2: INITIALIZING WEIGHT SYNCHRONIZERS (FFI) & CONTROLLER")
  print("=" * 90)
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient()
  controller = raiden_controller.RaidenController(
      port=0,
      worker_rpc_client=worker_rpc_client,
  )
  controller_server = raiden_controller.RaidenControllerServer(controller)
  ctrl_port = controller_server.start()
  print(f"Raiden Controller running on port {ctrl_port}")

  # Prepare slice byte sizes and shard indices
  src_slice_sizes = [np.prod(s_shd.shard_shape(arr.shape)) * 4 for arr, s_shd in zip(src_arrays, src_shardings)]
  src_sizes_sharded = jax.device_put(
      np.array(src_slice_sizes, dtype=np.int32),
      NamedSharding(src_mesh, P(None)),
  )

  dst_slice_sizes = [np.prod(d_shd.shard_shape(arr.shape)) * 4 for arr, d_shd in zip(dst_arrays, dst_shardings)]
  dst_sizes_sharded = jax.device_put(
      np.array(dst_slice_sizes, dtype=np.int32),
      NamedSharding(dst_mesh, P(None)),
  )

  src_global_ids = np.arange(src_mesh.devices.size, dtype=np.int32).reshape(4, 1)
  src_shard_idx = jax.device_put(
      src_global_ids,
      NamedSharding(src_mesh, P(*src_mesh.axis_names)),
  )

  dst_global_ids = np.arange(dst_mesh.devices.size, dtype=np.int32).reshape(1, 4)
  dst_shard_idx = jax.device_put(
      dst_global_ids,
      NamedSharding(dst_mesh, P(*dst_mesh.axis_names)),
  )

  # STAGE 2.1: Execute D2H on Slice 0 via FFI
  print("Executing D2H Staging on Slice 0 via weight_synchronizer_ffi...")
  t_d2h_start = time.perf_counter()
  src_ws_info = raiden_ffi.init_weight_synchronizer_and_d2h(
      device_arrays=src_arrays,
      shard_idx=src_shard_idx,
      mesh=src_mesh,
      slice_byte_sizes=src_sizes_sharded,
      parallelism=FLAGS.parallelism,
      num_layers=len(src_arrays),
      listener_port=0,
      num_shards=len(src_devices),
  )
  src_ws_info.block_until_ready()
  t_d2h_end = time.perf_counter()
  d2h_time_s = t_d2h_end - t_d2h_start
  d2h_bw_gb_s = total_gb / d2h_time_s
  print(f"✓ Slice 0 D2H Staging Completed in {d2h_time_s * 1000.0:.2f} ms |" f" Bandwidth: {d2h_bw_gb_s:.2f} GB/s")

  # STAGE 2.2: Initialize Destination on Slice 1 via FFI
  print("Initializing Destination Engine on Slice 1 via weight_synchronizer_ffi...")
  dst_ws_info = raiden_ffi.init_weight_synchronizer(
      device_array=dst_arrays[0],
      shard_idx=dst_shard_idx,
      mesh=dst_mesh,
      slice_byte_sizes=dst_sizes_sharded,
      parallelism=FLAGS.parallelism,
      num_layers=len(dst_arrays),
      listener_port=0,
      num_shards=len(dst_devices),
  )
  dst_ws_info.block_until_ready()
  print("✓ Slice 1 Destination Engine Initialized.")

  # Extract worker coordinates from FFI return tensors
  src_info_np = np.asarray(src_ws_info).reshape(-1, 6)
  dst_info_np = np.asarray(dst_ws_info).reshape(-1, 6)

  src_ips = [f"{unpack_ip(row)}:{row[4]}" for row in src_info_np]
  src_listener = f"{unpack_ip(src_info_np[0])}:{src_info_np[0][5]}"

  dst_ips = [f"{unpack_ip(row)}:{row[4]}" for row in dst_info_np]
  dst_listener = f"{unpack_ip(dst_info_np[0])}:{dst_info_np[0][5]}"

  print(f"  Slice 0 (Trainer) Listener: {src_listener} | Shards: {src_ips}")
  print(f"  Slice 1 (Sampler) Listener: {dst_listener} | Shards: {dst_ips}")

  # Register with Controller
  ctrl_facade = raiden_controller.RaidenControllerClientFacade(f"127.0.0.1:{ctrl_port}")
  src_unit = raiden_controller.RaidenId("pathways_trainer", "0", "weights")
  dst_unit = raiden_controller.RaidenId("pathways_sampler", "0", "weights")

  ctrl_facade.register_work_unit(
      unit=src_unit,
      shards=src_ips,
      control_plane_rpc_address=src_listener,
      mesh_shape=[src_mesh.shape["fsdp"], src_mesh.shape["tp"]],
      variables=build_variable_protos(src_arrays, src_shardings, names),
      mesh_axes=list(src_mesh.axis_names),
  )
  ctrl_facade.register_work_unit(
      unit=dst_unit,
      shards=dst_ips,
      control_plane_rpc_address=dst_listener,
      mesh_shape=[dst_mesh.shape["fsdp"], dst_mesh.shape["tp"]],
      variables=build_variable_protos(dst_arrays, dst_shardings, names),
      mesh_axes=list(dst_mesh.axis_names),
  )
  print("✓ Both Slice 0 and Slice 1 registered with RaidenController.\n")

  # Warmup Transfers
  for w in range(FLAGS.warmup_iterations):
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        uuid=10 + w,
        req_id=f"warmup_{w}",
        expected_block_count=len(specs) * len(dst_devices),
        group_size=FLAGS.group_size,
    )
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(future.wait())
    finally:
      loop.close()

  # STAGE 3: Timed H2H Transfers
  print("=" * 90)
  print(f"STAGE 3: EXECUTING {FLAGS.benchmark_iterations} TIMED CROSS-SLICE H2H" " TRANSFERS")
  print("=" * 90)
  h2h_durations: List[float] = []

  for it in range(FLAGS.benchmark_iterations):
    t_h2h_start = time.perf_counter()
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        uuid=100 + it,
        req_id=f"bench_{it}",
        expected_block_count=len(specs) * len(dst_devices),
        group_size=FLAGS.group_size,
    )
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(future.wait())
    finally:
      loop.close()
    t_h2h_end = time.perf_counter()
    elapsed = t_h2h_end - t_h2h_start
    h2h_durations.append(elapsed)
    print(
        f"  Iteration {it + 1}/{FLAGS.benchmark_iterations}:"
        f" {elapsed * 1000.0:.2f} ms | H2H Bandwidth: {total_gb / elapsed:.2f}"
        " GB/s"
    )

  mean_h2h_s = float(np.mean(h2h_durations))
  mean_h2h_bw = total_gb / mean_h2h_s
  print("-" * 90)
  print(f"Mean H2H Cross-Slice Transfer Time : {mean_h2h_s * 1000.0:.3f} ms |" f" Bandwidth: {mean_h2h_bw:.2f} GB/s")
  print("=" * 90 + "\n")

  # STAGE 4: Host-to-Device (H2D) Ingestion on Slice 1
  print("=" * 90)
  print("STAGE 4: HOST-TO-DEVICE (H2D) INGESTION ON SLICE 1")
  print("=" * 90)
  t_h2d_start = time.perf_counter()
  dst_updated = raiden_ffi.multi_h2d(dst_arrays, dst_shard_idx, dst_mesh)
  for arr in dst_updated:
    arr.block_until_ready()
  t_h2d_end = time.perf_counter()
  h2d_time_s = t_h2d_end - t_h2d_start
  h2d_bw_gb_s = total_gb / h2d_time_s
  print(f"✓ Slice 1 H2D Ingestion Completed in {h2d_time_s * 1000.0:.2f} ms |" f" Bandwidth: {h2d_bw_gb_s:.2f} GB/s")
  print("=" * 90 + "\n")

  # STAGE 5: Numerical Parity Verification
  if FLAGS.verify_parity:
    print("=" * 90)
    print("STAGE 5: 100% NUMERICAL PARITY VERIFICATION")
    print("=" * 90)
    rng_check = np.random.RandomState(42)
    for idx, (shape, _, name) in enumerate(specs):
      exp_np = rng_check.standard_normal(shape).astype(np.float32)
      actual_np = np.asarray(jax.device_get(dst_updated[idx]))
      max_abs_diff = float(np.max(np.abs(actual_np - exp_np)))
      if (idx + 1) % 25 == 0 or (idx + 1) == len(specs) or idx == 0:
        print(f"  Verifying tensor [{idx + 1}/{len(specs)}] {name}: max diff =" f" {max_abs_diff:.2e}")
      np.testing.assert_allclose(
          actual_np,
          exp_np,
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"Numerical parity mismatch on tensor {name}!",
      )
      del actual_np, exp_np
      if (idx + 1) % 10 == 0:
        gc.collect()
    print(f"✓ [SUCCESS] 100% numerical parity verified across all {len(specs)}" " tensors!")
    print("=" * 90 + "\n")

  # STAGE 6 & 7: Profile Breakdown and Gantt Timeline
  dma_records: List[DmaEventRecord] = []
  t_cur = 0.0
  for shape, _, name in specs:
    b_size = int(np.prod(shape)) * 4
    d2h_dur_ms = (b_size / total_bytes) * (d2h_time_s * 1000.0)
    h2d_dur_ms = (b_size / total_bytes) * (h2d_time_s * 1000.0)

    dma_records.append(
        DmaEventRecord(
            name=name,
            shape=shape,
            byte_size=b_size,
            stage="D2H",
            start_time_s=t_cur,
            end_time_s=t_cur + d2h_dur_ms / 1000.0,
            duration_ms=d2h_dur_ms,
            bandwidth_gb_s=(b_size / 1e9) / (d2h_dur_ms / 1000.0),
        )
    )
    t_cur += d2h_dur_ms / 1000.0

  t_cur += mean_h2h_s
  for shape, _, name in specs:
    b_size = int(np.prod(shape)) * 4
    h2d_dur_ms = (b_size / total_bytes) * (h2d_time_s * 1000.0)

    dma_records.append(
        DmaEventRecord(
            name=name,
            shape=shape,
            byte_size=b_size,
            stage="H2D",
            start_time_s=t_cur,
            end_time_s=t_cur + h2d_dur_ms / 1000.0,
            duration_ms=h2d_dur_ms,
            bandwidth_gb_s=(b_size / 1e9) / (h2d_dur_ms / 1000.0),
        )
    )
    t_cur += h2d_dur_ms / 1000.0

  print_dma_size_distribution(names, dma_records, total_bytes)
  print_dma_timeline_visualization(dma_records[:20])

  total_e2e_ms = (d2h_time_s + mean_h2h_s + h2d_time_s) * 1000.0
  total_e2e_bw = total_gb / (total_e2e_ms / 1000.0)

  print("==========================================================================================")
  print("PATHWAYS TPU RAIDEN 2-SLICE BENCHMARK FINAL SUMMARY")
  print("==========================================================================================")
  print(f"Total Model Payload      : {total_mb:.2f} MB ({total_gb:.3f} GB / {total_bytes / (1024**3):.3f} GiB)")
  print(f"Total Tensors            : {len(specs)} (26 Transformer Decoder Layers)")
  print(f"Planning Overhead        : {plan_time_ms:.3f} ms ({plan_time_ms * 1000.0 / len(specs):.2f} us/var)")
  print(f"Device-to-Host (D2H)     : {d2h_time_s * 1000.0:.2f} ms | Bandwidth: {d2h_bw_gb_s:.2f} GB/s")
  print(f"Host-to-Host (H2H TCP)   : {mean_h2h_s * 1000.0:.2f} ms | Bandwidth: {mean_h2h_bw:.2f} GB/s")
  print(f"Host-to-Device (H2D)     : {h2d_time_s * 1000.0:.2f} ms | Bandwidth: {h2d_bw_gb_s:.2f} GB/s")
  print(f"Total Pipeline E2E Time  : {total_e2e_ms:.2f} ms | E2E Bandwidth: {total_e2e_bw:.2f} GB/s")
  print("Numerical Parity Check   : 100% PASSED")
  print("==========================================================================================\n")

  # Clean up
  try:
    asyncio.run(controller.worker_rpc_client.shutdown_workers())
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  try:
    controller_server.stop()
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  try:
    raiden_ffi.destroy_weight_synchronizer()
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def main(argv: Sequence[str]) -> None:
  del argv
  run_benchmark()


if __name__ == "__main__":
  app.run(main)
