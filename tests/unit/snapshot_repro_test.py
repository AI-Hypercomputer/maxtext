import functools
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from pathwaysutils.experimental import mock_fault_injection
from pathwaysutils.experimental import split_by_mesh_axis
from maxtext.utils import snapshot as maxtext_snapshot


class ReproducibleNoActiveReplicasTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.simulator = mock_fault_injection.DeviceLossSimulator()
    self.simulator.start()

    orig_split = split_by_mesh_axis.split_by_mesh_axis

    def contaminated_split(arrays, *args, **kwargs):
      flat_inputs, _ = jax.tree.flatten(arrays)
      has_data_loss = any(
          self.simulator.is_array_corrupted(x) for x in flat_inputs
      )
      outputs = orig_split(arrays, *args, **kwargs)
      if has_data_loss:
        flat_outputs, _ = jax.tree.flatten(outputs)
        for leaf in flat_outputs:
          self.simulator.track_array(leaf)
          self.simulator._corrupted_array_ids.add(id(leaf))
      return outputs

    self.split_patcher = functools.partial(contaminated_split)
    self.orig_split_ref = split_by_mesh_axis.split_by_mesh_axis
    split_by_mesh_axis.split_by_mesh_axis = self.split_patcher

  def tearDown(self):
    split_by_mesh_axis.split_by_mesh_axis = self.orig_split_ref
    self.simulator.stop()
    super().tearDown()

  def test_reproduce_no_active_replicas_failure(self):
    devices, slice_to_devices = mock_fault_injection.create_mock_devices(
        num_devices=4, num_slices=2
    )
    mesh = jax.sharding.Mesh(np.array(devices).reshape(2, 2), ("data", "model"))

    with jax.sharding.Mesh(mesh.devices, mesh.axis_names):
      sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec("data", None)
      )
      arr = jax.device_put(jnp.ones((4, 8)), sharding)
      self.simulator.bind_array_to_devices(arr, devices)

      snapshotter = maxtext_snapshot.Snapshotter(replica_axis_index=0)
      snapshotter.save(0, {"weights": arr})
      snapshotter.join()

      self.simulator.mark_slice_lost(0, slice_to_devices)
      self.simulator.mark_slice_connected(0, slice_to_devices)

      with self.assertRaises(RuntimeError) as ctx:
        _ = snapshotter.load({"weights": arr})
      self.assertIn("No active replicas found.", str(ctx.exception))

  def test_non_defensive_active_slice_recovery_fix(self):
    devices, slice_to_devices = mock_fault_injection.create_mock_devices(
        num_devices=4, num_slices=2
    )
    mesh = jax.sharding.Mesh(np.array(devices).reshape(2, 2), ("data", "model"))

    with jax.sharding.Mesh(mesh.devices, mesh.axis_names):
      sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec("data", None)
      )
      arr = jax.device_put(jnp.ones((4, 8)), sharding)
      self.simulator.bind_array_to_devices(arr, devices)

      snapshotter = maxtext_snapshot.Snapshotter(replica_axis_index=0)
      snapshotter.save(0, {"weights": arr})
      snapshotter.join()

      self.simulator.mark_slice_lost(0, slice_to_devices)
      self.simulator.mark_slice_connected(0, slice_to_devices)

      active_slice_idx = 1
      slice_devices = slice_to_devices[active_slice_idx]
      submesh = jax.sharding.Mesh(
          np.array(slice_devices).reshape(1, 2), ("data", "model")
      )
      submesh_sharding = jax.sharding.NamedSharding(
          submesh, jax.sharding.PartitionSpec("data", None)
      )

      restored_slice = jax.device_put(arr[2:4, :], submesh_sharding)
      self.assertFalse(self.simulator.is_array_corrupted(restored_slice))
      np.testing.assert_array_equal(np.asarray(restored_slice), np.ones((2, 8)))


if __name__ == "__main__":
  absltest.main()
