import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import time

def main():
  # Initialize JAX distributed
  try:
    jax.distributed.initialize()
    print("JAX distributed initialized successfully.")
  except Exception as e:
    print(f"JAX distributed init failed (probably running single-host local): {e}")

  devices = jax.devices()
  num_devices = len(devices)
  print(f"Found {num_devices} devices: {devices}")

  # Create 1D mesh for FSDP
  mesh = Mesh(devices, ('fsdp',))

  # Shapes
  B = 8192
  H1 = 1024
  H2 = 2048
  H3 = 4096
  H4 = 8192
  H5 = 16384

  # Shardings
  x_sharding = NamedSharding(mesh, P('fsdp', None))
  w1_sharding = NamedSharding(mesh, P('fsdp', None))
  w2_sharding = NamedSharding(mesh, P('fsdp', None))
  w3_sharding = NamedSharding(mesh, P('fsdp', None))
  w4_sharding = NamedSharding(mesh, P('fsdp', None))

  # Initialize inputs and weights
  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)

  print("Initializing arrays...")
  x = jax.device_put(jax.random.normal(k1, (B, H1), dtype=jnp.bfloat16), x_sharding)
  w1 = jax.device_put(jax.random.normal(k2, (H1, H2), dtype=jnp.bfloat16), w1_sharding)
  w2 = jax.device_put(jax.random.normal(k3, (H2, H3), dtype=jnp.bfloat16), w2_sharding)
  w3 = jax.device_put(jax.random.normal(k4, (H3, H4), dtype=jnp.bfloat16), w3_sharding)
  w4 = jax.device_put(jax.random.normal(k5, (H4, H5), dtype=jnp.bfloat16), w4_sharding)

  # Define the computation
  @jax.jit
  def forward(x, w1, w2, w3, w4):
    y1 = jnp.matmul(x, w1)
    y2 = jnp.matmul(y1, w2)
    y3 = jnp.matmul(y2, w3)
    y4 = jnp.matmul(y3, w4)
    return y4

  # Warmup
  print("Warming up...")
  out = forward(x, w1, w2, w3, w4)
  jax.block_until_ready(out)
  print("Warmup done.")

  # Profiling setup
  trace_dir = f"gs://mattdavidow-maxtext-br/jetski/profile_{int(time.time())}"
  print(f"Profiling to {trace_dir}")

  # Time and Profile
  print("Running with profiling...")
  jax.profiler.start_trace(trace_dir)
  
  t0 = time.time()
  steps = 10
  for _ in range(steps):
    out = forward(x, w1, w2, w3, w4)
  jax.block_until_ready(out)
  t1 = time.time()
  
  jax.profiler.stop_trace()
  print("Profiling stopped.")
  print(f"Average time: {(t1-t0)/steps * 1000:.3f} ms")

  # Print output shape and sharding to verify
  print(f"Output shape: {out.shape}")
  print(f"Output sharding: {out.sharding}")

if __name__ == "__main__":
  main()
