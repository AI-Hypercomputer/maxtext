import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
from jax.experimental.shard_map import shard_map

def main():
    # Parameters
    e = 128
    l = 94
    m = 2
    n = 128  # Defined n
    input_shape = (e, l, m, n)

    # 1. Setup Devices
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    
    if len(devices) < 8:
        print("Warning: Fewer than 8 devices found. This script is designed for 8 TPUs.")
        # Proceeding with available devices might fail mesh creation if not adjusted, 
        # but assuming the environment provides 8.
        if len(devices) == 0:
            return

    # 2. Create Mesh
    # We distribute the 8 devices across 'expert' and 'stage' axes.
    # To shard on 'expert' axis efficiently with 8 devices, we can set expert=8, stage=1.
    num_experts_axis = 8
    num_stages_axis = 1
    
    # Ensure we use exactly 8 devices if available, or fewer if testing locally
    num_used_devices = min(len(devices), num_experts_axis * num_stages_axis)
    mesh_devices = np.array(devices[:num_used_devices]).reshape(num_experts_axis, num_stages_axis)
    
    mesh = Mesh(mesh_devices, axis_names=('expert', 'stage'))
    print(f"Mesh created: {mesh}")

    # 3. Define Sharding
    # The tensor (e, l, m, n) is sharded on the 'expert' axis (dimension 0).
    # PartitionSpec maps tensor dimensions to mesh axes.
    # (expert, None, None, None) means:
    #   dim 0 (e) -> 'expert' axis
    #   dim 1 (l) -> None (replicated/unsharded on that dimension)
    #   dim 2 (m) -> None
    #   dim 3 (n) -> None
    sharding = NamedSharding(mesh, PartitionSpec('expert', None, None, None))

    # 4. Create Tensor
    print(f"Creating tensor {input_shape} with sharding {sharding}...")

    # Create a random tensor on host and push to device with sharding
    x_host = jax.random.normal(jax.random.PRNGKey(42), input_shape)
    x_sharded = jax.device_put(x_host, sharding)
    x_sharded.block_until_ready()  # Ensure the tensor is fully transferred and sharded before proceeding
    
    # 5. Verify Sharding
    print(f"Resulting sharding: {x_sharded.sharding}")
    
    # Check the shape of the shard on the first device
    # With e=128 and expert axis=8, each device should have 128/8 = 16 for dim 0.
    if x_sharded.addressable_shards:
        shard_shape = x_sharded.addressable_shards[0].data.shape
        print(f"Local shard shape: {shard_shape}")
        expected_shard_shape = (e // num_experts_axis, l, m, n)
        assert shard_shape == expected_shard_shape, \
            f"Expected {expected_shard_shape}, got {shard_shape}"
        print("Sharding verified successfully.")

    # 6. Shard Map Transpose
    print("\nRunning shard_map transpose...")
    permute_fn = shard_map(
        lambda x: jnp.transpose(x, (1, 0, 3, 2)),
        mesh=mesh,
        in_specs=PartitionSpec('expert', None, None, None),
        out_specs=PartitionSpec(None, 'expert', None, None)
    )
    y = permute_fn(x_sharded)
    print(f"Output shape: {y.shape}")
    print(f"Output sharding: {y.sharding}")

if __name__ == "__main__":
    main()
