import datetime
import functools
import subprocess
import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

def apply_throttling(rate="100mbit", burst="500kb", latency="400ms"):
  try:
    route_output = subprocess.check_output("ip route show", shell=True, text=True)
    interface = None
    for line in route_output.splitlines():
      if "default" in line:
        interface = line.split("dev")[1].strip().split()[0]
        break
    if not interface:
      interface = "eth0"
    
    print(f"Applying tc egress limit of {rate} on {interface}...")
    subprocess.run(f"tc qdisc del dev {interface} root || true", shell=True)
    subprocess.run(
        f"tc qdisc add dev {interface} root tbf rate {rate} burst {burst} latency {latency}",
        shell=True,
        check=True,
    )
    print("Throttling applied successfully.")
  except Exception as e:
    print(f"Error applying throttling: {e}")

def cleanup_throttling():
  try:
    route_output = subprocess.check_output("ip route show", shell=True, text=True)
    interface = None
    for line in route_output.splitlines():
      if "default" in line:
        interface = line.split("dev")[1].strip().split()[0]
        break
    if not interface:
      interface = "eth0"
      
    print(f"Cleaning up tc egress limit on {interface}...")
    subprocess.run(f"tc qdisc del dev {interface} root || true", shell=True)
    print("Throttling cleaned up successfully.")
  except Exception as e:
    print(f"Error cleaning up throttling: {e}")

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
    raise ValueError(
        f"Need {dcn_size * ici_size} devices, but found {total_devices}"
    )
  mesh_devices = mesh_utils.create_hybrid_device_mesh(
      ici_parallelism, dcn_parallelism, devices=jax.devices()
  )
  mesh = Mesh(mesh_devices, ("dcn", "ici"))
  return mesh

def run_dcn_benchmark():
  print(f"JAX process index: {jax.process_index()} / {jax.process_count()}")
  print(f"Total devices: {jax.device_count()}, local devices: {jax.local_device_count()}")
  
  dcn_size = 2
  ici_size = 16
  mesh = create_mesh(dcn_size, ici_size)
  
  # Matrix sizes to benchmark: 64MB, 256MB, 1GB (matrix_dim = 4096, 8192, 16384 for bfloat16)
  matrix_dims = [4096, 8192, 16384]
  dtype = jnp.bfloat16
  
  apply_throttling(rate="28gbit", burst="10mb", latency="50ms")
  
  try:
    print("\n=== Starting DCN Bandwidth Microbenchmark ===")
    for dim in matrix_dims:
      matrix = jnp.ones((dim, dim), dtype=dtype)
      matrix_size_gbyte = dim * dim * dtype.dtype.itemsize / 1e9
      
      # We define shard map collective psum along the DCN axis.
      # Input x is sharded across 'dcn' axis, meaning local shard size is matrix_size_gbyte / 2
      @functools.partial(shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P(None))
      def psum_dcn_op(x):
        return jax.lax.psum(x, "dcn")
        
      # Pre-distribute the matrix shard onto devices
      sharded_matrix = jax.device_put(
          matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
      )
      
      jitted_op = jax.jit(psum_dcn_op)
      
      # Run time test
      time_ms = simple_timeit(jitted_op, sharded_matrix, task=f"psum_dcn_{dim}x{dim}")
      
      # Calculate Bandwidth
      # volume = 2 * (D-1) * M / (D * D) bytes
      # D = 2 slices. volume = 2 * (1) * M / 4 = M / 2 bytes per slice.
      # Bandwidth (GB/s) = (M / 2 GB) / (time_ms / 1000) = M * 500 / time_ms
      achieved_bandwidth_gbyte_s = (
          matrix_size_gbyte
          * (dcn_size - 1)
          * 2
          / dcn_size
          / dcn_size
          / (time_ms / 1e3)
      )
      
      # Convert GB/s to Gbps: 1 GB/s = 8 Gbps
      achieved_bandwidth_gbps = achieved_bandwidth_gbyte_s * 8.0
      
      if jax.process_index() == 0:
        print(f"Matrix: {dim}x{dim} ({matrix_size_gbyte * 1000:.1f} MB)")
        print(f"  Avg Latency: {time_ms:.2f} ms")
        print(f"  Achieved DCN Bandwidth: {achieved_bandwidth_gbyte_s:.3f} GB/s ({achieved_bandwidth_gbps:.2f} Gbps) per slice")
        print("-" * 50)
  finally:
    cleanup_throttling()

if __name__ == "__main__":
  run_dcn_benchmark()
