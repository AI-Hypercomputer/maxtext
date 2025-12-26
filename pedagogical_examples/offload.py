import os
import time
import jax
import jax.numpy as jnp
import numpy as np

libtpu_init_args = [
    "--xla_tpu_dvfs_p_state=7",
]
os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "65536000000"
os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "65536000000"

def create_default_mesh(axis_shapes, axis_names):
  """Creates a JAX device mesh with the default device order."""
  try:
    num_required_devices = np.prod(axis_shapes)
    devices = np.array(jax.devices())
    if len(devices) < num_required_devices:
      print(f"Expected at least {num_required_devices} devices, but only found {len(devices)}. This script requires more devices.")
      return None

    device_array = devices[:num_required_devices].reshape(axis_shapes)
    return jax.sharding.Mesh(device_array, axis_names)
  except RuntimeError:
    print("No TPU devices found. This script must be run on a TPU node.")
    return None

def run_optimized_benchmark():
  axis_names = ("data", "model")
  axis_shapes = (1, 4)
  mesh = create_default_mesh(axis_shapes, axis_names)
  if mesh is None: return

  # Parameters
  data_dtype = jnp.bfloat16
  num_layers = 10
  num_blocks = 1024
  block_size = 64
  # Fold num_layers into the first dimension for a single contiguous buffer
  # combined_shape = (num_layers, num_blocks, block_size, 8, 2, 128)
  combined_shape = (num_layers, num_blocks, block_size, 8, 2, 128)
  
  # Sharding - keeping your "model" axis parallelism
  # partition_spec = jax.sharding.PartitionSpec(None, None, None, "model", None, None)
  partition_spec = jax.sharding.PartitionSpec(None, None, None, "model", None, None)

  device_sharding = jax.sharding.NamedSharding(mesh, partition_spec)
  host_sharding = jax.sharding.NamedSharding(mesh, partition_spec, memory_kind='pinned_host')

  kv_cache_host = np.random.normal(size=combined_shape).astype(np.float32)
  kv_cache_full = jax.device_put(kv_cache_host, device_sharding).block_until_ready()

  # 2. Optimized Transfer Function
  # We avoid jax.tree.map inside JIT to prevent overhead
  @jax.jit
  def d2h_transfer(arr):
    return jax.device_put(arr, host_sharding)

  def smart_d2h_transfer(sharded_array):
    src_shard_map = {s.device: s.data for s in sharded_array.addressable_shards}
      
    pinned_shards = []
    
    # 2. Iterate strictly in Mesh Order to satisfy make_array requirements
    # We use the mesh from the target sharding to ensure alignment
    for device in host_sharding.mesh.devices.flat:
      if device in src_shard_map:
        # CRITICAL FIX: Target "Pinned Host" memory for THIS specific TPU.
        # This keeps the data in the driver's DMA-accessible region.
        # We create a sharding that says: "This array lives on 'device', but in Host RAM".
        pinned_spec = jax.sharding.SingleDeviceSharding(device, memory_kind='pinned_host')
        
        # Async DMA Transfer (TPU HBM -> TPU Pinned Host)
        # This happens in parallel across all chips if you have multiple
        shard_future = jax.device_put(src_shard_map[device], pinned_spec)
        pinned_shards.append(shard_future)
    
    # 3. Assemble without copying
    # JAX simply creates a new metadata wrapper pointing to the pinned buffers we just created.
    return jax.make_array_from_single_device_arrays(
      sharded_array.shape, 
      host_sharding, 
      pinned_shards
    )

  def h2d_transfer(arr):
    return jax.device_put(arr, device_sharding)

  # 4. Benchmark
  n_trials = 16
  total_bytes = np.prod(combined_shape) * kv_cache_full.dtype.itemsize
  h2d_perf, d2h_perf = [], []
  for _ in range(n_trials):
    # 1. put host data on device using device_put
    h2d_start = time.perf_counter()
    on_device = jax.device_put(kv_cache_host, device_sharding).block_until_ready()
    h2d_end = time.perf_counter()
    h2d_perf.append(h2d_end - h2d_start)
    # 2. put device data on host using device_get
    d2h_start = time.perf_counter()
    _ = jax.device_get([s.data for s in on_device.addressable_shards])
    d2h_end = time.perf_counter()
    d2h_perf.append(d2h_end - d2h_start)


  # with jax.profiler.trace("gs://runner-maxtext-logs/offload/12162025/"):
  #   h2d_transfer(kv_cache_host).block_until_ready()

  median_h2d_t = np.median(h2d_perf)
  h2d_bw_gbps = (total_bytes / (1024**3)) / median_h2d_t
  median_d2h_t = np.median(d2h_perf)
  d2h_bw_gbps = (total_bytes / (1024**3)) / median_d2h_t
  

  print(f"\n--- Results ---")
  print(f"Total Transfer Size: {total_bytes / (1024**2):.2f} MB")
  print(f"Measured D2H Bandwidth: {d2h_bw_gbps:.2f} GiB/s")
  print(f"Measured H2D Bandwidth: {h2d_bw_gbps:.2f} GiB/s")

if __name__ == "__main__":
  run_optimized_benchmark()