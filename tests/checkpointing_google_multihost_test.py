"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Multihost tests for sharded checkpointing features in MaxText."""

from absl.testing import parameterized
from etils import epath
from flax.training import train_state
import jax
import jax.sharding
from maxtext.common import checkpointing
import numpy as np
import optax
from orbax.checkpoint import test_utils
import safetensors.numpy

# BEGIN GOOGLE-INTERNAL
from google3.learning.brain.research.jax.tests.multiprocess import multiprocess_test
# END GOOGLE-INTERNAL


Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


class CheckpointingMultiHostTest(
    parameterized.TestCase,
):
  """Tests sharded loading in a simulated multihost environment."""

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)
    self.test_dir = epath.Path(self.create_tempdir(name="test_dir").full_path)
    test_utils.sync_global_processes("setUp")

  def tearDown(self):
    test_utils.sync_global_processes("tearDown")
    super().tearDown()

  @parameterized.named_parameters(
      {
          "testcase_name": "1d_array",
          "array_shape": (16,),
          "expected_mesh_shape": (8,),
          "expected_pspec": PartitionSpec("a"),
      },
      {
          "testcase_name": "2d_array",
          "array_shape": (8, 4),
          "expected_mesh_shape": (8,),
          "expected_pspec": PartitionSpec("a", None),
      },
      {
          "testcase_name": "3d_array",
          "array_shape": (4, 2, 16),
          "expected_mesh_shape": (8,),
          "expected_pspec": PartitionSpec(None, None, "a"),
      },
      {
          # gcd(12, 8) = 4, with 2 devices to shard the 2nd dimension.
          "testcase_name": "2d_array_multiaxis",
          "array_shape": (12, 8),
          "expected_mesh_shape": (4, 2),
          "expected_pspec": PartitionSpec("a", "b"),
      },
      {
          # gcd(24, 8) = 8
          "testcase_name": "2d_array_large_greedy",
          "array_shape": (24, 4),
          "expected_mesh_shape": (8,),
          "expected_pspec": PartitionSpec("a", None),
      },
      {
          # Shows a 2D mesh with a final dimension that is not sharded.
          "testcase_name": "3d_array_multiaxis_with_replication",
          "array_shape": (12, 6, 3),
          "expected_mesh_shape": (4, 2),
          "expected_pspec": PartitionSpec("a", "b", None),
      },
  )  # Note: These tests don't cover error cases; those are sufficiently tested
  # in the single-host test (checkpointing_google_test.py)
  def test_sharded_safetensors_loading(self, array_shape, expected_mesh_shape, expected_pspec):
    """Tests that load_state_if_possible correctly shards a SafeTensors file."""
    self.skipTest(
        "Skip due to unsupported (unperformant) multihost-loading." " TODO(b/496270336): Re-enable when supported."
    )

    st_path = self.test_dir / f"{self.id()}.safetensors"

    # Create and save the source data from a single process.
    source_data_np = np.arange(np.prod(array_shape), dtype=np.float32).reshape(array_shape)
    source_data = {"tensor": source_data_np}
    if jax.process_index() == 0:
      safetensors.numpy.save_file(source_data, st_path)
    test_utils.sync_global_processes("file_saved")

    # Define the sharding that construct_maximal_shardings should create.
    devices = np.array(jax.devices())
    mesh_axes = tuple(sorted(list(set(filter(None, expected_pspec)))))
    expected_mesh = Mesh(devices.reshape(expected_mesh_shape), mesh_axes)
    expected_sharding = NamedSharding(expected_mesh, expected_pspec)
    expected_data = {"tensor": jax.device_put(source_data_np, expected_sharding)}

    # Set up dummy inputs for the function call.
    abstract_state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params={"a": jax.ShapeDtypeStruct((), np.float32)},
        tx=optax.sgd(0.1),
    )

    def simple_convert(source_tree):
      return {"tensor": source_tree["tensor"]}

    loaded_data, _ = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path=str(st_path),
        checkpoint_storage_concurrent_gb=1,
        abstract_unboxed_pre_state=abstract_state,
        enable_orbax_v1=True,
        checkpoint_conversion_fn=simple_convert,
        source_checkpoint_layout="safetensors",
    )
    restored_tensor = loaded_data["items"]["tensor"]
    self.assertEqual(restored_tensor.sharding, expected_sharding)

    expected_tensor = expected_data["tensor"]
    test_utils.assert_array_equal(self, expected_tensor, restored_tensor)


if __name__ == "__main__":
  multiprocess_test.main()
