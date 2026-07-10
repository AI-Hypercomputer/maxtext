#!/usr/bin/env python3
"""Standalone OSS JAX Script: Multi-Slice TPU Host Offloading on Pathways.

This script replaces internal Google3 test utilities with standard OSS JAX primitives.

It demonstrates:
1. Implicit TPU HBM <-> Host CPU offloading using `@compute_on('device_host')` within `@jax.jit`.
2. Explicit memory sharding via `.with_memory_kind('pinned_host')`.

Note: When running via Pathways (SPS or XManager), `jax.distributed.initialize()` is not needed
because coordination and discovery are managed by the Pathways Resource Manager and proxy server.
"""

import jax
from jax import lax
from jax import sharding
from jax.experimental import compute_on
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np
import optax

# Type aliases
Mesh = sharding.Mesh
NamedSharding = sharding.NamedSharding
PartitionSpec = sharding.PartitionSpec


# ==============================================================================
# 1. Implicit Host Offloading & Pinned Host Memory (@compute_on)
# ==============================================================================


def run_host_offloading_example(tpu_mesh: Mesh):
  """Demonstrates implicit TPU <-> CPU offloading and memory kind sharding on multi-slice TPUs."""
  print("\n--- 1. Testing @compute_on('device_host') & pinned_host Sharding ---")

  # Define TPU HBM sharding and Host CPU RAM sharding over the TPU mesh
  device_sharding = NamedSharding(tpu_mesh, PartitionSpec("dp", None))
  host_sharding = device_sharding.with_memory_kind("pinned_host")

  # Initialize weights on TPU HBM ('device') and optimizer state on Host ('pinned_host')
  params = jax.device_put(jnp.ones((tpu_mesh.devices.size * 32, 128), dtype=jnp.float32), device_sharding)
  grads = jax.device_put(jnp.full_like(params, 0.1), device_sharding)

  def put_on_host(x):
    spec = PartitionSpec("dp", None) if x.ndim >= 2 else PartitionSpec(*([None] * x.ndim))
    return jax.device_put(x, NamedSharding(tpu_mesh, spec).with_memory_kind("pinned_host"))

  optimizer = optax.adam(learning_rate=1e-3)
  opt_state = jax.tree.map(put_on_host, optimizer.init(params))

  print(f"Initial params memory kind: {params.sharding.memory_kind}")
  print(f"Initial opt_state memory kind: {jax.tree.leaves(opt_state)[0].sharding.memory_kind}")

  @compute_on.compute_on("device_host")
  @jax.jit
  def host_optimizer_update(p, opt_st, g):
    """Executes entirely on host CPU RAM during TPU execution, saving TPU HBM."""
    updates, new_opt_st = optimizer.update(g, opt_st)
    updates_device = jax.tree.map(lambda x: jax.device_put(x, device_sharding), updates)
    new_p = optax.apply_updates(p, updates_device)
    return new_p, new_opt_st

  @jax.jit
  def tpu_train_step(p, opt_st, g):
    # 1. All-reduce gradients across TPU slices on HBM
    g_accum = shard_map.shard_map(
        lambda x: lax.psum(x, axis_name="dp"),
        tpu_mesh,
        in_specs=PartitionSpec("dp", None),
        out_specs=PartitionSpec("dp", None),
    )(g)

    # 2. Automatically offload parameter update to Host CPU memory space
    g_host = jax.device_put(g_accum, host_sharding)
    new_p, new_opt_st = host_optimizer_update(p, opt_st, g_host)
    return new_p, new_opt_st

  new_params, new_opt_state = tpu_train_step(params, opt_state, grads)
  jax.block_until_ready((new_params, new_opt_state))
  print("✓ @compute_on('device_host') training step completed successfully.")
  print(f"  New params memory kind: {new_params.sharding.memory_kind}")
  print(f"  New opt_state memory kind: {jax.tree.leaves(new_opt_state)[0].sharding.memory_kind}")

  # Explicit data transfer demonstration from TPU HBM to Pinned Host
  explicit_host_array = jax.device_put(new_params, host_sharding)
  print(f"✓ Explicit transfer to host RAM completed. Memory kind: {explicit_host_array.sharding.memory_kind}")


# ==============================================================================
# Main Execution
# ==============================================================================


def main():
  tpu_devices = jax.devices() if jax.default_backend() in ("tpu", "proxy") else []
  print(f"Total TPU devices detected: {len(tpu_devices)}")

  # Execute TPU <-> Host CPU offloading if TPU devices are available
  if tpu_devices:
    tpu_mesh = Mesh(np.array(tpu_devices).reshape(-1, 1), ("dp", "model"))
    with tpu_mesh:
      run_host_offloading_example(tpu_mesh)
  else:
    print("\nSkipping TPU host offloading tests (no TPU devices detected in current environment).")


if __name__ == "__main__":
  main()
