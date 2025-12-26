
import argparse
import os
import time

import jax
from jax import sharding
import numpy as np
import pandas as pd


libtpu_init_args = [
    "--xla_tpu_dvfs_p_state=7",
]
os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "65536000000"
os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "65536000000"

BASE_PATH = "./bucket/202512_h2dd2h"


def mesh_shape_type(s):
  try:
    dims = [int(d) for d in s.split("x")]
    if len(dims) < 1 or len(dims) > 2:
      raise argparse.ArgumentTypeError(
          f"Mesh shape must be in 'N' or 'XxY' format: {s}"
      )
    return tuple(dims)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"Mesh shape must be in 'N' or 'XxY' format: {s}"
    ) from exc


def get_tpu_devices(num_devices):
  devices = jax.devices()
  print(f"Device found: {devices}")
  if len(devices) < num_devices:
    raise RuntimeError(
        f"Require at least {num_devices} TPU cores for this mesh"
        f" configuration, but found {len(devices)}"
    )
  if devices[0].platform != "tpu":
    raise RuntimeError("First device is not TPU, maybe install libtpu?")
  return devices[:num_devices]


def benchmark_transfer(
    root_path, tpu_devices, mesh_shape, data_size, n_iter=20
):
  """Benchmarks Host-to-Device and Device-to-Host transfer performance.

  Args:
    root_path: The root path for saving results and traces.
    tpu_devices: A list of TPU devices to use for the benchmark.
    mesh_shape: A tuple representing mesh shape (e.g., (2, 4)).
    data_size: The size of the data to transfer in MB.
    n_iter: The number of iterations to run the benchmark.

  Returns:
    A dictionary containing the median times and throughputs for H2D and
    different D2H transfer methods (standard, iterated, parallel).
  """
  num_devices = len(tpu_devices)
  data_size_mb = data_size
  rows = 1024 * data_size_mb // 4
  host_data = np.random.normal(size=(rows, 8, 128)).astype(np.float32)

  print(
      f"Benchmarking Transfer with Data Size: {data_size_mb} MB on"
      f" {num_devices} devices"
  )

  # --- WARM UP ---
  # JAX often has initialization overhead on the first run
  print(f"Warming up data transfer ({data_size_mb} MB)...")
  if len(mesh_shape) == 1:
    mesh = sharding.Mesh(
        np.array(tpu_devices).reshape(mesh_shape), axis_names=("x",)
    )
    data_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec("x"))
  else:
    mesh = sharding.Mesh(
        np.array(tpu_devices).reshape(mesh_shape), axis_names=("x", "y")
    )
    data_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec(("x", "y"))
    )
  data_on_device = jax.device_put(host_data, data_sharding)
  data_on_device.block_until_ready()
  _ = jax.device_get(data_on_device)

  h2d_perf, std_d2h_perf, iter_d2h_perf, para_d2h_perf, para_h2d_perf = [], [], [], [], []

  def h2d():
    data_on_device = jax.device_put(host_data, data_sharding)
    data_on_device.block_until_ready()
    return data_on_device

  mesh_str = "x".join(map(str, mesh_shape))
  with jax.profiler.trace(f"{root_path}/traces/{mesh_str}/{data_size}MB"):
    for _ in range(n_iter):
      # CASE 1: Standard device_get
      # --- BENCHMARK HOST -> DEVICE (H2D) ---
      h2d_start = time.perf_counter()
      data_on_device = h2d()
      h2d_end = time.perf_counter()
      h2d_perf.append(h2d_end - h2d_start)
      # --- BENCHMARK DEVICE -> HOST (D2H) ---
      std_d2h_start = time.perf_counter()
      _ = jax.device_get(data_on_device)
      std_d2h_end = time.perf_counter()
      std_d2h_perf.append(std_d2h_end - std_d2h_start)
      data_on_device.delete()

      # CASE 2: Iterated device_get
      data_on_device = h2d()
      iter_d2h_start = time.perf_counter()
      _ = [
          jax.device_get(shard.data)
          for shard in data_on_device.addressable_shards
      ]
      iter_d2h_end = time.perf_counter()
      iter_d2h_perf.append(iter_d2h_end - iter_d2h_start)
      data_on_device.delete()

      # CASE 3: Parallel device_get
      data_on_device = h2d()
      para_d2h_start = time.perf_counter()
      addressable_shards = data_on_device.addressable_shards
      shards_on_host = jax.device_get([s.data for s in data_on_device.addressable_shards])
      # Get a map of which device maps to which host array
      shard_device_map = {
          s.device: arr for s, arr in zip(addressable_shards, shards_on_host)
      }
      para_d2h_end = time.perf_counter()
      para_d2h_perf.append(para_d2h_end - para_d2h_start)
      data_on_device.delete()

      # CASE 4: Bring back addressable shards from host to device
      para_h2d_start = time.perf_counter()
      
      # 1. Identify which devices in the mesh are local to this process
      # (In single-host, this is all of them. In multi-host, it's a subset)
      global_mesh_devices = data_sharding.mesh.devices.flat
      local_mesh_devices = [
          d for d in global_mesh_devices 
          if d.process_index == jax.process_index()
      ]
      
      # 2. Reorder our host shards to match the local mesh order strictly
      # This handles cases where addressable_shards order != mesh order
      ordered_host_shards = [shard_device_map[d] for d in local_mesh_devices]

      # 3. Transfer each shard to its specific device
      single_device_arrays = [
          jax.device_put(arr, dev) 
          for arr, dev in zip(ordered_host_shards, local_mesh_devices)
      ]

      # 4. Stitch them together
      # Note: jax.make_array_from_single_device_arrays expects the list of arrays
      # to match the order of 'local' devices in the mesh.
      restored_device_array = jax.make_array_from_single_device_arrays(
          host_data.shape,
          data_sharding,
          single_device_arrays
      )
      restored_device_array.block_until_ready()
      
      para_h2d_end = time.perf_counter()
      para_h2d_perf.append(para_h2d_end - para_h2d_start)
      restored_device_array.delete()

  def time2bandwidth_gb(t):
    return (data_size_mb / 1024) / t

  h2d_median = np.median(h2d_perf)
  print(
      f"Host -> TPU: {h2d_median:.4f} s | Bandwidth:"
      f" {time2bandwidth_gb(h2d_median):.2f} GB/s"
  )

  std_d2h_median = np.median(std_d2h_perf)
  print(
      f"STANDARD TPU -> Host: {std_d2h_median:.4f} s | Bandwidth:"
      f" {time2bandwidth_gb(std_d2h_median):.2f} GB/s"
  )

  iter_d2h_median = np.median(iter_d2h_perf)
  print(
      f"ITERATED TPU -> Host: {iter_d2h_median:.4f} s | Bandwidth:"
      f" {time2bandwidth_gb(iter_d2h_median):.2f} GB/s"
  )

  para_d2h_median = np.median(para_d2h_perf)
  print(
      f"PARALLEL TPU -> Host: {para_d2h_median:.4f} s | Bandwidth:"
      f" {time2bandwidth_gb(para_d2h_median):.2f} GB/s"
  )

  para_h2d_median = np.median(para_h2d_perf)
  print(
      f"PARALLEL HOST -> TPU: {para_h2d_median:.4f} s | Brandwidth:"
      f" {time2bandwidth_gb(para_h2d_median):.2f} GB/s"
  )

  return {
      "h2d_time (s)": h2d_median,
      "h2d_throughput (GB/s)": time2bandwidth_gb(h2d_median),
      "std_d2h_time (s)": std_d2h_median,
      "std_d2h_throughput (GB/s)": time2bandwidth_gb(std_d2h_median),
      "iter_d2h_time (s)": iter_d2h_median,
      "iter_d2h_throughput (GB/s)": time2bandwidth_gb(iter_d2h_median),
      "para_d2h_time (s)": para_d2h_median,
      "para_d2h_throughput (GB/s)": time2bandwidth_gb(para_d2h_median),
      "para_h2d_time (s)": para_h2d_median,
      "para_h2d_throughput (GB/s)": time2bandwidth_gb(para_h2d_median),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=(
          "Benchmarks Host-to-Device and Device-to-Host transfer performance"
          " on TPU devices with a specific mesh configuration."
      )
  )
  parser.add_argument(
      "--mesh",
      type=mesh_shape_type,
      default="1x2",
      help="TPU mesh shape in format N or XxY, e.g., 8, 1x2 or 2x4.",
  )
  parser.add_argument(
      "--exp_name",
      type=str,
      default="mesh",
      help="Experiment name, used as the directory name for results.",
  )
  args = parser.parse_args()
  mesh_shape = (
      args.mesh if isinstance(args.mesh, tuple) else mesh_shape_type(args.mesh)
  )
  num_devices_needed = np.prod(mesh_shape)
  root_path = f"{BASE_PATH}/{args.exp_name}"
  mesh_str = "x".join(map(str, mesh_shape))

  dir_path = f"{root_path}/traces/{mesh_str}"
  os.makedirs(dir_path, exist_ok=True)
  print(f"Folder {dir_path} and traces are created")

  tpu_devices = get_tpu_devices(num_devices_needed)

  data_range = [1 << 10]
  result = {}
  for data in data_range:
    result[f"{data}MB"] = benchmark_transfer(
        root_path, tpu_devices, mesh_shape, data
    )
  result = pd.DataFrame(result).T
  result.index.name = "data size"
  result.to_csv(f"{root_path}/result_{mesh_str}.csv")
  print(f"Result is saved to {root_path}/result_{mesh_str}.csv")