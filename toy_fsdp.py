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

  # Shapes (roughly 8k, incremented by 256)
  B = 32 * 1024
  H1 = 8192
  H2 = 8192 + 256       # 8448
  H3 = 8192 + 2 * 256   # 8704
  H4 = 8192 + 3 * 256   # 8960
  H5 = 8192 + 4 * 256   # 9216

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

  # Helper function for overlapped compute and gather
  def compute_and_gather(y_current, w_current_gathered, w_next_local):
    # Compute current layer: depends on y_current and w_current_gathered
    y_next = jnp.matmul(y_current, w_current_gathered)
    
    # Gather next layer's weight: depends ONLY on w_next_local
    w_next_gathered = jax.lax.all_gather(w_next_local, axis_name='fsdp', axis=0)
    w_next_gathered = w_next_gathered.reshape(-1, w_next_gathered.shape[-1])
    
    return y_next, w_next_gathered

  # Define the computation using shard_map with overlap
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=(P('fsdp', None), P('fsdp', None), P('fsdp', None), P('fsdp', None), P('fsdp', None)),
      out_specs=P('fsdp', None),
  )
  def forward_shard_map(x_local, w1_local, w2_local, w3_local, w4_local):
    # 1. Start: Gather W1 (no prior compute)
    w1_gathered = jax.lax.all_gather(w1_local, axis_name='fsdp', axis=0)
    w1_gathered = w1_gathered.reshape(-1, w1_gathered.shape[-1])

    # 2. Step 1: Compute Layer 1 AND Gather W2
    y1_local, w2_gathered = compute_and_gather(x_local, w1_gathered, w2_local)

    # 3. Step 2: Compute Layer 2 AND Gather W3
    y2_local, w3_gathered = compute_and_gather(y1_local, w2_gathered, w3_local)

    # 4. Step 3: Compute Layer 3 AND Gather W4
    y3_local, w4_gathered = compute_and_gather(y2_local, w3_gathered, w4_local)

    # 5. End: Compute Layer 4 (no more gathers)
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
