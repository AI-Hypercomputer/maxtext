import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import shard_map
from jax.experimental import xla_metadata
import jax.experimental.pallas as pl
import functools
import time
import numpy as np

# Import Tokamax Splash Attention
try:
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash_kernel
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask
  print("Successfully imported Tokamax Splash Attention.")
except ImportError as e:
  print(f"Failed to import Tokamax: {e}")
  # Fallback placeholders for syntax checking if run locally on workstation without tokamax
  class tokamax_splash_kernel:
    QKVLayout = None
    SplashConfig = None
    @staticmethod
    def make_splash_mha(*args, **kwargs):
      return lambda q, k, v, *a: jnp.zeros((q.shape[0], q.shape[1], q.shape[2])) # dummy
  class tokamax_splash_mask:
    CausalMask = None

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
  B = 16
  H = 16
  S = 2048
  D = 128
  H_in = H * D # 2048
  H_out = 8192

  # Shardings
  # Shard batch dimension for activations, replicate others
  act_sharding = NamedSharding(mesh, P('fsdp', None, None, None))
  # Shard input dimension for weights (FSDP)
  w_sharding = NamedSharding(mesh, P('fsdp', None))

  # Initialize inputs on host, then put to devices
  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)

  print("Initializing arrays...")
  q = jax.device_put(jax.random.normal(k1, (B, H, S, D), dtype=jnp.bfloat16), act_sharding)
  k = jax.device_put(jax.random.normal(k2, (B, H, S, D), dtype=jnp.bfloat16), act_sharding)
  v = jax.device_put(jax.random.normal(k3, (B, H, S, D), dtype=jnp.bfloat16), act_sharding)
  
  # W_proj: [H_in, H_out] sharded along H_in
  w_proj = jax.device_put(jax.random.normal(k4, (H_in, H_out), dtype=jnp.bfloat16), w_sharding)

  # Define Cost Estimate for Splash Attention
  # Local FLOPs per device: 2 * B_local * H * S^2 * D = 2 * 2 * 16 * 2048^2 * 128 = 68.7 GFLOPs
  # Local Bytes accessed: (3 * B_local * H * S * D + 1 * B_local * H * S * D) * 2 bytes = 67 MB
  cost_est = pl.CostEstimate(
      flops=68_000_000_000,
      transcendentals=0,
      bytes_accessed=67_000_000,
  )

  # Setup Splash Attention Config
  # We use HEAD_DIM_MINOR because shape is [H, S, D]
  # We use block size 2048 (entire sequence) for maximum performance
  sa_config = tokamax_splash_kernel.SplashConfig(
      block_q=2048,
      block_kv=2048,
      block_kv_compute=2048,
      q_layout=tokamax_splash_kernel.QKVLayout.HEAD_DIM_MINOR if tokamax_splash_kernel.QKVLayout else None,
      k_layout=tokamax_splash_kernel.QKVLayout.HEAD_DIM_MINOR if tokamax_splash_kernel.QKVLayout else None,
      v_layout=tokamax_splash_kernel.QKVLayout.HEAD_DIM_MINOR if tokamax_splash_kernel.QKVLayout else None,
      fwd_cost_estimate=cost_est,
  )
  
  # Causal mask for the sequence length
  mask = tokamax_splash_mask.CausalMask(shape=(S, S)) if tokamax_splash_mask.CausalMask else None

  # Make the splash attention kernel
  # Note: q_seq_shards=1 because we are not sharding the sequence dimension, only batch.
  if tokamax_splash_mask.CausalMask:
    splash_kernel = tokamax_splash_kernel.make_splash_mha(
        mask=mask,
        config=sa_config,
        q_seq_shards=1,
    )
  else:
    splash_kernel = None

  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=(
          P('fsdp', None, None, None), # q
          P('fsdp', None, None, None), # k
          P('fsdp', None, None, None), # v
          P('fsdp', None),             # w_proj
      ),
      out_specs=P('fsdp', None),       # output y: sharded along batch
      check_vma=False,
  )
  def forward_shard_map(q_local, k_local, v_local, w_proj_local):
    # We vmap over the batch dimension (axis 0 of local inputs)
    attn_fn = jax.vmap(lambda q, k, v: splash_kernel(q, k, v), in_axes=(0, 0, 0))

    # Use scheduling group to force overlap of Splash Attention and All-Gather
    with xla_metadata.set_xla_metadata(_scheduling_group_id=0):
      # Gather W_proj
      w_proj_gathered = jax.lax.all_gather(w_proj_local, axis_name='fsdp', axis=0)
      w_proj_gathered = w_proj_gathered.reshape(-1, w_proj_gathered.shape[-1])
      
      # Run Splash Attention
      attn_out_local = attn_fn(q_local, k_local, v_local)

    # 3. Reshape attention output for Projection Matmul
    # We need to transpose to [B_local, S, H, D] then reshape to [B_local * S, H * D]
    # H * D = H_in = 2048.
    attn_out_local = jnp.transpose(attn_out_local, (0, 2, 1, 3)) # [B_local, S, H, D]
    attn_out_local_flat = attn_out_local.reshape(-1, H_in)       # [B_local * S, H_in]

    # 4. Projection Matmul
    # attn_out_local_flat is [B_local * S, H_in] = [2 * 2048, 2048] = [4096, 2048]
    # w_proj_gathered is [H_in, H_out] = [2048, 8192]
    # y_local will be [B_local * S, H_out] = [4096, 8192]
    y_local = jnp.matmul(attn_out_local_flat, w_proj_gathered)

    # We return y_local. The out_spec is P('fsdp', None), which matches y_local's sharding
    # because the first dimension of y_local is B_local * S = (B/8) * S = (B * S)/8,
    # which is sharded along the 'fsdp' axis (since B is sharded).
    return y_local

  @jax.jit
  def run_forward(q, k, v, w_proj):
    return forward_shard_map(q, k, v, w_proj)

  # Warmup
  print("Warming up...")
  out = run_forward(q, k, v, w_proj)
  jax.block_until_ready(out)
  print("Warmup done.")

  # Profiling setup
  trace_dir = f"gs://mattdavidow-maxtext-br/jetski/profile_{int(time.time())}"
  print(f"Profiling to {trace_dir}")

  # Run with profiling
  print("Running with profiling...")
  jax.profiler.start_trace(trace_dir)
  
  # Run a few steps to get a good average and trace
  steps = 10
  start_time = time.time()
  for _ in range(steps):
    out = run_forward(q, k, v, w_proj)
    jax.block_until_ready(out)
  end_time = time.time()
  
  jax.profiler.stop_trace()
  
  avg_time = (end_time - start_time) / steps * 1000
  print(f"Average time: {avg_time:.3f} ms")

  print(f"Output shape: {out.shape}")
  print(f"Output sharding: {out.sharding}")

if __name__ == "__main__":
  main()
