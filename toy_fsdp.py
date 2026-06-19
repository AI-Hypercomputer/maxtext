import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax import shard_map
import functools
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
  B = 32 * 1024  # Increased batch size
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

  # Define the computation using shard_map
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=(P('fsdp', None), P('fsdp', None), P('fsdp', None), P('fsdp', None), P('fsdp', None)),
      out_specs=P('fsdp', None),
  )
  def forward_shard_map(x_local, w1_local, w2_local, w3_local, w4_local):
    # Layer 1: All-gather W1, then matmul
    w1_gathered = jax.lax.all_gather(w1_local, axis_name='fsdp', axis=0)
    y1_local = jnp.matmul(x_local, w1_gathered)

    # Layer 2: All-gather W2, then matmul
    w2_gathered = jax.lax.all_gather(w2_local, axis_name='fsdp', axis=0)
    y2_local = jnp.matmul(y1_local, w2_gathered)

    # Layer 3: All-gather W3, then matmul
    w3_gathered = jax.lax.all_gather(w3_local, axis_name='fsdp', axis=0)
    y3_local = jnp.matmul(y2_local, w3_gathered)

    # Layer 4: All-gather W4, then matmul
    w4_gathered = jax.lax.all_gather(w4_local, axis_name='fsdp', axis=0)
    y4_local = jnp.matmul(y3_local, w4_gathered)

    return y4_local

  @jax.jit
  def run_forward(x, w1, w2, w3, w4):
    return forward_shard_map(x, w1, w2, w3, w4)

  # Warmup
  print("Warming up...")
  out = run_forward(x, w1, w2, w3, w4)
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
    out = run_forward(x, w1, w2, w3, w4)
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
