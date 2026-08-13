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

"""In-process end-to-end performance and profiling benchmark for TPU Raiden weight
synchronization with 2D resharding (FSDP x TP -> a differently-shaped FSDP x TP mesh).

OSS port of the internal 2D-resharding perf test: same D2H / H2H / H2D staged
benchmark, per-tensor DMA profiling (size-bucket + timeline), and numerical
parity verification, adapted to run against the public `tpu-raiden` package
(single process, CPU-emulated devices by default).

Usage:
  python transfer_weights_raiden_2d_resharding_bench.py \
      --num_decoder_layers=26 --benchmark_iterations=3
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import time
from typing import Any, List, Tuple

# Default host platform devices if XLA_FLAGS is not set, so this can run on
# CPU without a real TPU slice.
if "XLA_FLAGS" not in os.environ:
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from absl import app
from absl import flags
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
  from tpu_sync.rpc import raiden_controller
  from tpu_sync.rpc import raiden_service_pb2
except Exception as e:  # pylint: disable=broad-exception-caught
  raise RuntimeError(
      "tpu-raiden is required for transfer_weights_raiden_2d_resharding_bench.py "
      f"but could not be imported: {e}"
  ) from e


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "num_decoder_layers",
    26,
    "Number of decoder layers to benchmark.",
)
flags.DEFINE_integer(
    "benchmark_iterations", 3, "Number of timed benchmark iterations."
)
flags.DEFINE_integer(
    "group_size",
    128,
    "Number of weights to group per transfer request.",
)
flags.DEFINE_integer(
    "parallelism",
    16,
    "Number of parallel network stream TCP socket workers per"
    " WeightSynchronizer.",
)
flags.DEFINE_integer(
    "raiden_transport_coalesce_window_bytes",
    67108864,
    "Host-side coalescing buffer size (bytes) for network transfers. This"
    " maps to a native ABSL_FLAG (raiden_transport_coalesce_window_bytes,"
    " tpu_sync/transport/block_transport.cc) that the JAX nanobind"
    " extension never reaches via absl::ParseCommandLine, so Python cannot"
    " set it through the flag/argv path. This script instead exports it as"
    " the RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES environment variable"
    " before constructing any WeightSynchronizer, which BlockTransport"
    " reads as an override -- this requires a tpu-raiden build containing"
    " that env-var read (upstream branch"
    " expose-coalesce-window-bytes-config); on older builds lacking it,"
    " this value is exported but has no effect.",
)


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


def get_synthetic_transformer_specs(
    num_layers: int = 26,
) -> List[Tuple[Tuple[int, ...], P, str]]:
  """Generates representative transformer parameter specifications."""
  specs = []
  # Embedding
  specs.append(((256128, 2304), P("tp", "fsdp"), "embedder.input_embedding"))

  if num_layers > 0:
    # Transformer Decoder Layers
    for l in range(num_layers):
      specs.append(((2304, 2048), P("fsdp", "tp"), f"layer_{l}.attn.q_proj"))
      specs.append(((2304, 1024), P("fsdp", "tp"), f"layer_{l}.attn.k_proj"))
      specs.append(((2304, 1024), P("fsdp", "tp"), f"layer_{l}.attn.v_proj"))
      specs.append(((2048, 2304), P("tp", "fsdp"), f"layer_{l}.attn.o_proj"))

      specs.append(((2304, 9216), P("fsdp", "tp"), f"layer_{l}.mlp.gate_proj"))
      specs.append(((2304, 9216), P("fsdp", "tp"), f"layer_{l}.mlp.up_proj"))
      specs.append(((9216, 2304), P("tp", "fsdp"), f"layer_{l}.mlp.down_proj"))

      specs.append(((2304,), P("fsdp"), f"layer_{l}.input_layernorm"))
      specs.append(((2304,), P("fsdp"), f"layer_{l}.post_attention_layernorm"))
      specs.append(((2304,), P("fsdp"), f"layer_{l}.post_feedforward_layernorm"))

    # Final Norm
    specs.append(((2304,), P("fsdp"), "final_norm"))
  return specs


def build_variable_protos(
    arrays: List[jax.Array],
    shardings: List[NamedSharding],
    names: List[str],
) -> List[Any]:
  """Builds VariableMetadataProto entries describing each array's global shape/sharding."""
  protos = []
  for idx, (arr, sharding, name) in enumerate(zip(arrays, shardings, names)):
    global_shape = arr.shape
    local_shard_shape = sharding.shard_shape(global_shape)
    sharding_shape = [g // l for g, l in zip(global_shape, local_shard_shape)]
    layout = tuple(range(len(global_shape) - 1, -1, -1))

    spec_axes = []
    if hasattr(sharding, "spec"):
      for axis in sharding.spec:
        if axis is None:
          spec_axes.append("")
        elif isinstance(axis, str):
          spec_axes.append(axis)
        else:
          spec_axes.append(",".join(axis))

    protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=name,
            shape=global_shape,
            mesh_shape=sharding_shape,
            layout=layout,
            item_size=arr.dtype.itemsize,
            layer_idx=idx,
            sharding_spec=spec_axes,
        )
    )
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

    avg_d2h_ms = np.mean([d2h_map[name].duration_ms for name in bucket_tensors])
    avg_d2h_bw = np.mean([d2h_map[name].bandwidth_gb_s for name in bucket_tensors])
    avg_h2d_ms = np.mean([h2d_map[name].duration_ms for name in bucket_tensors])
    avg_h2d_bw = np.mean([h2d_map[name].bandwidth_gb_s for name in bucket_tensors])

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
  print(
      "STAGE 7: TEMPORAL DMA TIMELINE VISUALIZATION (Total Profiling Window:"
      f" {total_span_ms:.2f} ms)"
  )
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

    print(
        f"{e.name:<42} | {e.stage:<5} | {start_ms:>10.2f} | {end_ms:>10.2f} |"
        f" {dur_ms:>9.3f} | {bar_str}"
    )
  print("=" * 125)
  print(
      "Legend: [#] = Device-to-Host (D2H) Staging | [=] = Host-to-Device"
      " (H2D) Ingestion\n"
  )


def main(argv):
  del argv  # Unused.

  # BlockTransport reads this env var (if the linked tpu-raiden build
  # supports it) as an override for the coalesce-window ABSL_FLAG, which is
  # otherwise unreachable from Python. Must be set before any
  # WeightSynchronizer is constructed.
  os.environ["RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES"] = str(
      FLAGS.raiden_transport_coalesce_window_bytes
  )

  try:
    all_devices = jax.devices("tpu")
  except RuntimeError:
    all_devices = jax.devices()

  if len(all_devices) < 2:
    raise RuntimeError(
        f"This benchmark requires at least 2 devices, found {len(all_devices)}. "
        "Set XLA_FLAGS=--xla_force_host_platform_device_count=8 to emulate on CPU."
    )

  # Disjoint source/destination sub-slices on the same physical / simulated
  # host, split evenly across the available devices.
  half = len(all_devices) // 2
  src_devices = all_devices[:half]
  dst_devices = all_devices[half : 2 * half]

  # Source Mesh: pure FSDP (FSDP=half, TP=1).
  src_mesh = Mesh(np.array(src_devices).reshape(half, 1), ("fsdp", "tp"))
  # Destination Mesh: pure TP (FSDP=1, TP=half) -> non-trivial 2D resharding.
  dst_mesh = Mesh(np.array(dst_devices).reshape(1, half), ("fsdp", "tp"))

  print(f"Source Mesh initialized: {src_mesh.shape} on devices {[d.id for d in src_devices]}")
  print(f"Destination Mesh initialized: {dst_mesh.shape} on devices {[d.id for d in dst_devices]}")

  num_layers = FLAGS.num_decoder_layers
  specs = get_synthetic_transformer_specs(num_layers=num_layers)
  total_weights = len(specs)
  print(f"Generating {total_weights} synthetic transformer weight tensors (layers={num_layers})...")

  key = jax.random.PRNGKey(42)
  src_arrays = []
  dst_arrays = []
  src_shardings = []
  dst_shardings = []
  names = []
  expected_numpy = []

  total_elements = 0
  total_bytes = 0

  # 1. Allocate and initialize synthetic weights on source and destination sub-slices.
  for shape, pspec, name in specs:
    names.append(name)
    s_shd = NamedSharding(src_mesh, pspec)
    d_shd = NamedSharding(dst_mesh, pspec)
    src_shardings.append(s_shd)
    dst_shardings.append(d_shd)

    key, subkey = jax.random.split(key)
    src_arr = jax.jit(
        lambda k: jax.random.normal(k, shape=shape, dtype=jnp.float32),
        out_shardings=s_shd,
    )(subkey)
    src_arr.block_until_ready()
    src_arrays.append(src_arr)

    dst_arr = jax.jit(
        lambda: jnp.zeros(shape, dtype=jnp.float32),
        out_shardings=d_shd,
    )()
    dst_arr.block_until_ready()
    dst_arrays.append(dst_arr)

    expected_numpy.append(np.asarray(jax.device_get(src_arr)))

    num_elem = int(np.prod(shape))
    total_elements += num_elem
    total_bytes += num_elem * 4

  total_size_mb = total_bytes / 1e6
  total_size_gb = total_bytes / 1e9
  total_size_gib = total_bytes / (1024**3)

  print(
      f"Model payload allocated: {total_elements} elements, {total_size_mb:.2f} MB"
      f" ({total_size_gb:.3f} GB / {total_size_gib:.3f} GiB)"
  )

  # 2. Measure Resharding Plan Generation Overhead.
  print("Measuring Resharding Plan Generation Overhead...")
  t_plan_start = time.perf_counter()
  total_chunks = 0
  for shape, s_shd, d_shd in zip([s[0] for s in specs], src_shardings, dst_shardings):
    if len(shape) == 2:
      plan = resharding_planner.make_resharding_plan(shape, s_shd, d_shd)
      total_chunks += len(plan)
  t_plan_end = time.perf_counter()
  plan_total_time_ms = (t_plan_end - t_plan_start) * 1000.0
  plan_per_var_us = (plan_total_time_ms / total_weights) * 1000.0

  print("\n" + "=" * 80)
  print("STAGE 1: RESHARDING PLAN GENERATION OVERHEAD")
  print("=" * 80)
  print(f"Total Variables Evaluated : {total_weights}")
  print(f"Total 2D Reshard Chunks    : {total_chunks}")
  print(f"Total Plan Generation Time: {plan_total_time_ms:.3f} ms")
  print(f"Average Planning Overhead : {plan_per_var_us:.2f} us/variable")
  print("=" * 80 + "\n")

  # 3. Setup in-process Controller and WeightSynchronizer instances.
  print("Starting in-process RaidenController...")
  controller_network_client = raiden_controller.WeightSyncWorkerRpcClient()
  controller = raiden_controller.RaidenController(
      port=0,
      worker_rpc_client=controller_network_client,
  )
  controller_server = raiden_controller.RaidenControllerServer(controller)
  controller_server.start()
  controller_port = controller_server.port
  print(f"Controller running on port {controller_port}")
  print(f"WeightSynchronizer parallelism: {FLAGS.parallelism}")
  print(f"Transfer group_size: {FLAGS.group_size}")
  print(
      "raiden_transport_coalesce_window_bytes:"
      f" {FLAGS.raiden_transport_coalesce_window_bytes} (exported as"
      " RAIDEN_TRANSPORT_COALESCE_WINDOW_BYTES; applied only if the linked"
      " tpu-raiden build reads that env var -- see"
      " expose-coalesce-window-bytes-config)"
  )

  ws_source = weight_synchronizer.WeightSynchronizer(
      jax_arrays=src_arrays,
      local_port=0,
      parallelism=FLAGS.parallelism,
      listener_port=0,
      unsafe_skip_buffer_lock=False,
      bind_ip="127.0.0.1",
  )
  ws_dest = weight_synchronizer.WeightSynchronizer(
      jax_arrays=dst_arrays,
      local_port=0,
      parallelism=FLAGS.parallelism,
      listener_port=0,
      unsafe_skip_buffer_lock=False,
      bind_ip="127.0.0.1",
  )

  src_unit = raiden_controller.RaidenId("trainer", "0", "weights")
  dst_unit = raiden_controller.RaidenId("sampler", "0", "weights")

  src_variable_protos = build_variable_protos(src_arrays, src_shardings, names)
  dst_variable_protos = build_variable_protos(dst_arrays, dst_shardings, names)

  ctrl_client = raiden_controller.RaidenControllerClientFacade(f"127.0.0.1:{controller_port}")
  ctrl_client.register_work_unit(
      src_unit,
      [f"127.0.0.1:{ws_source.local_port}"] * len(src_devices),
      f"127.0.0.1:{ws_source.listener_port}",
      mesh_shape=[src_mesh.shape["fsdp"], src_mesh.shape["tp"]],
      variables=src_variable_protos,
      mesh_axes=list(src_mesh.axis_names),
  )
  ctrl_client.register_work_unit(
      dst_unit,
      [f"127.0.0.1:{ws_dest.local_port}"] * len(dst_devices),
      f"127.0.0.1:{ws_dest.listener_port}",
      mesh_shape=[dst_mesh.shape["fsdp"], dst_mesh.shape["tp"]],
      variables=dst_variable_protos,
      mesh_axes=list(dst_mesh.axis_names),
  )

  # 4. Multi-iteration Timed Benchmark Run.
  num_iters = FLAGS.benchmark_iterations
  d2h_latencies = []
  h2h_latencies = []
  h2d_latencies = []
  e2e_latencies = []

  print(f"Executing {num_iters} benchmark iterations...")

  for it in range(num_iters):
    print(f"--- Benchmark Iteration {it + 1}/{num_iters} ---")

    # Stage 2: Device-to-Host (D2H) Staging.
    t0 = time.perf_counter()
    ws_source.d2h()
    t1 = time.perf_counter()
    d2h_elapsed = t1 - t0
    d2h_latencies.append(d2h_elapsed)

    # Stage 3: Host-to-Host (H2H) P2P Loopback Transfer.
    t2 = time.perf_counter()
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=0,
        uuid=1000 + it,
        req_id=f"bench_iter_{it}",
        group_size=FLAGS.group_size,
    )
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(future.wait())
    finally:
      loop.close()
    t3 = time.perf_counter()
    h2h_elapsed = t3 - t2
    h2h_latencies.append(h2h_elapsed)

    # Stage 4: Host-to-Device (H2D) Ingestion.
    t4 = time.perf_counter()
    ws_dest.h2d()
    t5 = time.perf_counter()
    h2d_elapsed = t5 - t4
    h2d_latencies.append(h2d_elapsed)

    e2e_elapsed = d2h_elapsed + h2h_elapsed + h2d_elapsed
    e2e_latencies.append(e2e_elapsed)

  # 5. End-to-End Numerical Parity Verification.
  print("Verifying 100% numerical parity between Source and Destination tensors...")
  for idx, (name, dst_arr, exp_np) in enumerate(zip(names, dst_arrays, expected_numpy)):
    actual_np = np.asarray(jax.device_get(dst_arr))
    max_abs_diff = np.max(np.abs(actual_np - exp_np))
    if idx % 5 == 0:
      print(f"Verifying tensor [{idx + 1}/{total_weights}] {name}: max diff = {max_abs_diff:.2e}")
    np.testing.assert_allclose(
        actual_np,
        exp_np,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Tensor numerical divergence detected in {name} after 2D resharding!",
    )

  print(f"[SUCCESS] 100% numerical parity verified across all {total_weights} tensors after 2D resharding.")

  # 6. Measure Individual DMA Latencies & Timeline Interleaving.
  print("Profiling individual layer DMA calls & temporal timeline...")
  dma_events: List[DmaEventRecord] = []

  for name, s_arr, d_shd in zip(names, src_arrays, dst_shardings):
    single_bytes = int(np.prod(s_arr.shape)) * s_arr.dtype.itemsize
    # Single-tensor D2H.
    ws_single_src = weight_synchronizer.WeightSynchronizer(
        jax_arrays=[s_arr],
        local_port=0,
        parallelism=FLAGS.parallelism,
        listener_port=0,
        unsafe_skip_buffer_lock=False,
        bind_ip="127.0.0.1",
    )
    t_d2h_s = time.perf_counter()
    ws_single_src.d2h()
    t_d2h_e = time.perf_counter()
    d2h_dur_ms = (t_d2h_e - t_d2h_s) * 1000.0
    d2h_bw = (single_bytes / 1e9) / max(t_d2h_e - t_d2h_s, 1e-9)

    dma_events.append(
        DmaEventRecord(
            name=name,
            shape=s_arr.shape,
            byte_size=single_bytes,
            stage="D2H",
            start_time_s=t_d2h_s,
            end_time_s=t_d2h_e,
            duration_ms=d2h_dur_ms,
            bandwidth_gb_s=d2h_bw,
        )
    )

    # Single-tensor H2D (using isolated scratch array on destination mesh).
    scratch_d_arr = jax.jit(
        lambda: jnp.zeros(s_arr.shape, dtype=s_arr.dtype),
        out_shardings=d_shd,
    )()
    scratch_d_arr.block_until_ready()

    ws_single_dst = weight_synchronizer.WeightSynchronizer(
        jax_arrays=[scratch_d_arr],
        local_port=0,
        parallelism=FLAGS.parallelism,
        listener_port=0,
        unsafe_skip_buffer_lock=False,
        bind_ip="127.0.0.1",
    )
    t_h2d_s = time.perf_counter()
    ws_single_dst.h2d()
    t_h2d_e = time.perf_counter()
    h2d_dur_ms = (t_h2d_e - t_h2d_s) * 1000.0
    h2d_bw = (single_bytes / 1e9) / max(t_h2d_e - t_h2d_s, 1e-9)

    dma_events.append(
        DmaEventRecord(
            name=name,
            shape=s_arr.shape,
            byte_size=single_bytes,
            stage="H2D",
            start_time_s=t_h2d_s,
            end_time_s=t_h2d_e,
            duration_ms=h2d_dur_ms,
            bandwidth_gb_s=h2d_bw,
        )
    )

  # 7. Print Formatted Performance Report & Timeline Summary.
  mean_d2h = np.mean(d2h_latencies)
  mean_h2h = np.mean(h2h_latencies)
  mean_h2d = np.mean(h2d_latencies)
  mean_e2e = np.mean(e2e_latencies)

  d2h_bw_gb = total_size_gb / mean_d2h
  d2h_bw_gib = total_size_gib / mean_d2h
  h2h_bw_gb = total_size_gb / mean_h2h
  h2d_bw_gb = total_size_gb / mean_h2d
  h2d_bw_gib = total_size_gib / mean_h2d
  e2e_bw_gb = total_size_gb / mean_e2e

  print("\n" + "=" * 100)
  print("STAGE 2 - 4: END-TO-END WEIGHT SYNCHRONIZATION BENCHMARK RESULTS")
  print("=" * 100)
  print(
      f"Model Parameters Payload: {total_size_mb:.2f} MB ({total_size_gb:.3f}"
      f" GB / {total_size_gib:.3f} GiB)"
  )
  print(f"Benchmark Iterations    : {num_iters}")
  print("-" * 100)
  print(
      f"Device-to-Host (D2H)    : {mean_d2h*1000.0:.3f} ms | Throughput:"
      f" {d2h_bw_gb:.2f} GB/s ({d2h_bw_gib:.2f} GiB/s)"
  )
  print(f"Host-to-Host (H2H Loop) : {mean_h2h*1000.0:.3f} ms | Throughput: {h2h_bw_gb:.2f} GB/s")
  print(
      f"Host-to-Device (H2D)    : {mean_h2d*1000.0:.3f} ms | Throughput:"
      f" {h2d_bw_gb:.2f} GB/s ({h2d_bw_gib:.2f} GiB/s)"
  )
  print(
      f"Total Pipeline E2E Time : {mean_e2e*1000.0:.3f} ms | Effective"
      f" Aggregate BW: {e2e_bw_gb:.2f} GB/s"
  )
  print("=" * 100)

  # 8. Print Individual DMA Call Latency and Interleaving Matrix.
  print("\n" + "=" * 115)
  print("STAGE 5: INDIVIDUAL DMA CALLS BREAKDOWN & TEMPORAL INTERLEAVING")
  print("=" * 115)
  print(
      f"{'Layer Name':<42} | {'Shape':<16} | {'Size (MB)':<9} |"
      f" {'D2H (ms)':<9} | {'D2H (GB/s)':<11} | {'H2D (ms)':<9} |"
      f" {'H2D (GB/s)':<11}"
  )
  print("-" * 115)

  d2h_map = {e.name: e for e in dma_events if e.stage == "D2H"}
  h2d_map = {e.name: e for e in dma_events if e.stage == "H2D"}

  for name in names:
    d2h_e = d2h_map[name]
    h2d_e = h2d_map[name]
    size_mb = d2h_e.byte_size / 1e6
    shape_str = str(list(d2h_e.shape))
    print(
        f"{name:<42} | {shape_str:<16} | {size_mb:>9.2f} |"
        f" {d2h_e.duration_ms:>9.3f} | {d2h_e.bandwidth_gb_s:>11.2f} |"
        f" {h2d_e.duration_ms:>9.3f} | {h2d_e.bandwidth_gb_s:>11.2f}"
    )
  print("=" * 115 + "\n")

  # 9. Print DMA Size Distribution Analysis.
  print_dma_size_distribution(names, dma_events, total_bytes)

  # 10. Print DMA Timeline Visualization along Time Axis.
  print_dma_timeline_visualization(dma_events)

  # 11. Clean Shutdown.
  controller_server.stop()


if __name__ == "__main__":
  app.run(main)
