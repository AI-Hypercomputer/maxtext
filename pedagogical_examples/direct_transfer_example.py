"""
A pedagogical example of creating a JAX array on a training mesh
and then directly transferring it to a separate inference mesh.

This version is hardcoded for a 32-device environment, splitting them
into two 16-device meshes for training and inference.
"""
import pathwaysutils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np


pathwaysutils.initialize()


def run_direct_transfer_example():
  """
  Demonstrates creating and directly transferring a JAX array between two different meshes.
  """
  num_devices = len(jax.devices())
  if num_devices != 32:
    print(f"This example is hardcoded for 32 devices, but found {num_devices} devices.")
    return

  # 1. Set up two distinct 16-device meshes.
  all_devices = np.array(jax.devices())[:32]
  train_mesh_shape = (4, 4)
  inference_mesh_shape = (4, 4)
  axis_names = ('data', 'model')

  # First 16 devices for the training mesh.
  train_devices = all_devices[:16]
  train_mesh = Mesh(train_devices.reshape(train_mesh_shape), axis_names)

  # Next 16 devices for the inference mesh.
  inference_devices = all_devices[16:32]
  inference_mesh = Mesh(inference_devices.reshape(inference_mesh_shape), axis_names)

  print(f"Total devices available: {num_devices}")
  print(f"Training mesh using devices 0-15. Shape: {train_mesh.shape}")
  print(f"Inference mesh using devices 16-31. Shape: {inference_mesh.shape}")
  print("-" * 50)

  # 2. Generate a large array and distribute it across the training mesh.
  large_array = jnp.arange(4096 * 4096, dtype=jnp.float32).reshape(4096, 4096)

  train_sharding = NamedSharding(train_mesh, PartitionSpec('data', 'model'))
  array_on_train_mesh = jax.device_put(large_array, train_sharding)

  print("Array placed on the training mesh.")
  print(f"Sharding on training mesh: {array_on_train_mesh.sharding}")
  print(f"Global shape: {array_on_train_mesh.shape}")
  print("-" * 50)

  # 3. Assign this array to the inference mesh.
  inference_sharding = NamedSharding(inference_mesh, PartitionSpec('data', 'model'))

  with (
    jax.transfer_guard_device_to_host("disallow_explicit"),
    jax.transfer_guard_host_to_device("disallow_explicit"),
  ):
    print("Putting array_on_train_mesh on inference_mesh. It will fail if it goes through the host.")

    # Manual direct transfer implementation.

    # 1. Get the shards from the source array.
    source_shards = array_on_train_mesh.addressable_shards

    # 2. Create a list of single-device arrays, each placed on a target device.
    # This assumes a simple 1:1 mapping of device order, which works for this example.
    single_device_arrays = []
    for i, shard in enumerate(source_shards):
      target_device = inference_mesh.devices.flat[i]
      single_device_arrays.append(jax.device_put(shard.data, target_device))

    # 3. Construct the new GlobalDeviceArray on the target mesh.
    array_on_inference_mesh = jax.make_array_from_single_device_arrays(
      array_on_train_mesh.shape, inference_sharding, single_device_arrays
    )


  print("Array placed on the inference mesh.")
  print("-" * 50)

  # Verify the contents are the same.
  np.testing.assert_allclose(np.array(array_on_train_mesh), np.array(array_on_inference_mesh))
  print("Verified: Array contents are identical after direct transfer.")


if __name__ == "__main__":
  run_direct_transfer_example()
