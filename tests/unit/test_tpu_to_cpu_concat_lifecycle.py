"""Unit test script verifying the full tiling lifecycle using colocated_cpu_devices:

1. Create tensors on TPU mesh -> Prove they explicitly carry hardware tiling ({...:T(8,128)}).
2. Move tensors to colocated_cpu_devices(tpu_mesh) -> Inspect layout on colocated CPU mesh.
3. Concatenate CPU tensors via pw_client.concatenate_by_mesh_axis -> Inspect layout after concat.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
from jax import sharding
from jax.experimental import colocated_python
from pathwaysutils.experimental.concatenate_by_mesh_axis import concatenate_by_mesh_axis

def print_arr(name, arr):
  layout = getattr(arr, "_pjrt_layout", "NO_ATTRIBUTE")
  print(f"{name:<50} | shape={str(arr.shape):<15} | layout={layout}")

def _expand_dim_and_mesh_colocated(x: jax.Array, axis_name: str) -> jax.Array:
  sharding_val = x.sharding
  submesh = sharding_val.mesh
  expanded_devices = np.expand_dims(np.array(submesh.devices), axis=0)
  expanded_mesh = jax.sharding.Mesh(expanded_devices, axis_names=(axis_name,) + submesh.axis_names)
  expanded_sharding = jax.sharding.NamedSharding(
      expanded_mesh, jax.sharding.PartitionSpec(axis_name, *sharding_val.spec), memory_kind=sharding_val.memory_kind
  )
  local_arrays = [shard.data.reshape((1,) + shard.data.shape) for shard in x.addressable_shards]
  return jax.make_array_from_single_device_arrays(
      shape=(1,) + x.shape,
      sharding=expanded_sharding,
      arrays=local_arrays,
  )

def main():
  print("================ TPU -> COLOCATED CPU -> CONCAT TILING LIFECYCLE ================")
  tpu_devices = jax.devices("tpu")
  assert len(tpu_devices) >= 2, "Need >=2 TPU devices"
  
  # Use 2 TPU devices so colocated_cpu_devices succeeds on single-host VM
  tpu_devices_2 = tpu_devices[:2]
  tpu_mesh = jax.sharding.Mesh(np.array(tpu_devices_2).reshape(2, 1), axis_names=("diloco", "data"))
  cpu_mesh = colocated_python.colocated_cpu_devices(tpu_mesh)

  # Partition mesh into 2 submeshes along 'diloco' axis (each size 1)
  tpu_submeshes = [
      jax.sharding.Mesh(np.array([tpu_devices_2[i]]), axis_names=("data",)) for i in range(2)
  ]
  cpu_submeshes = [
      colocated_python.colocated_cpu_devices(subm) for subm in tpu_submeshes
  ]

  shape2d = (64, 128)
  @jax.jit
  def make_tpu_array(val):
    return jnp.full(shape2d, val, dtype=jnp.float32)

  expanded_cpu_trees = []
  for i in range(2):
    print(f"\n--- Replica {i} Lifecycle (Colocated CPU) ---")
    # Step 1: Create array on TPU submesh i (native XLA tiled layout)
    tpu_sharding = jax.sharding.NamedSharding(tpu_submeshes[i], jax.sharding.PartitionSpec("data"))
    tpu_arr = jax.device_put(make_tpu_array(float(i + 1)), tpu_sharding)
    print_arr(f"1. Replica {i} TPU Array (Created on TPU)", tpu_arr)

    # Prove it explicitly has tiling on TPU
    layout_tpu = str(getattr(tpu_arr, "_pjrt_layout", ""))
    assert "T(" in layout_tpu, f"Expected TILED layout on TPU, got: {layout_tpu}"
    print(f"   [VERIFIED] TPU array explicitly HAS TILING: {layout_tpu}")

    # Step 2: Move tensor to colocated CPU submesh
    cpu_sharding = jax.sharding.NamedSharding(cpu_submeshes[i], jax.sharding.PartitionSpec("data"))
    cpu_arr = jax.device_put(tpu_arr, cpu_sharding)
    print_arr(f"2. Replica {i} CPU Array (Moved to colocated CPU)", cpu_arr)

    layout_cpu = str(getattr(cpu_arr, "_pjrt_layout", ""))
    if "T(" in layout_cpu:
      print(f"   [OBSERVED] Moving to colocated CPU preserved tiling: {layout_cpu}")
    else:
      print(f"   [OBSERVED] Moving to colocated CPU stripped tiling: {layout_cpu}")

    exp_cpu = _expand_dim_and_mesh_colocated(cpu_arr, "diloco")
    expanded_cpu_trees.append(exp_cpu)

  # Step 3: Concatenate CPU submesh tensors via pw_client.concatenate_by_mesh_axis
  print("\n--- Step 3: Executing pw_client.concatenate_by_mesh_axis on Colocated CPU ---")
  concatenated_cpu_arr = concatenate_by_mesh_axis(expanded_cpu_trees, mesh_axis="diloco")
  print_arr("3. Concatenated Colocated CPU Result (_pjrt_layout)", concatenated_cpu_arr)

  layout_concat = str(getattr(concatenated_cpu_arr, "_pjrt_layout", ""))
  if "T(" in layout_concat:
    print(f"   [OBSERVED] Concatenation output is TILED: {layout_concat}")
  else:
    print(f"   [OBSERVED] Concatenation output is UNTILED (Null layout): {layout_concat}")

  print("\n[SUCCESS] Completed full TPU -> Colocated CPU -> Concatenation tiling lifecycle test!")

if __name__ == "__main__":
  main()

"""
================ LIVE EXECUTION LOG (task-666 on jzuo-dev-v4) ================

================ TPU -> COLOCATED CPU -> CONCAT TILING LIFECYCLE ================

--- Replica 0 Lifecycle (Colocated CPU) ---
1. Replica 0 TPU Array (Created on TPU)            | shape=(64, 128)       | layout={1,0:T(8,128)}
   [VERIFIED] TPU array explicitly HAS TILING: {1,0:T(8,128)}
2. Replica 0 CPU Array (Moved to colocated CPU)    | shape=(64, 128)       | layout={1,0}
   [OBSERVED] Moving to colocated CPU stripped tiling: {1,0}

--- Replica 1 Lifecycle (Colocated CPU) ---
1. Replica 1 TPU Array (Created on TPU)            | shape=(64, 128)       | layout={1,0:T(8,128)}
   [VERIFIED] TPU array explicitly HAS TILING: {1,0:T(8,128)}
2. Replica 1 CPU Array (Moved to colocated CPU)    | shape=(64, 128)       | layout={1,0}
   [OBSERVED] Moving to colocated CPU stripped tiling: {1,0}

--- Step 3: Executing pw_client.concatenate_by_mesh_axis on Colocated CPU ---
3. Concatenated Colocated CPU Result (_pjrt_layout) | shape=(2, 64, 128)    | layout={2,1,0}
   [OBSERVED] Concatenation output is UNTILED (Null layout): {2,1,0}

[SUCCESS] Completed full TPU -> Colocated CPU -> Concatenation tiling lifecycle test!
"""