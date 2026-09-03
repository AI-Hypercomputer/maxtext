"""
Copyright 2025 Google LLC

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

"""Tests for Orbax checkpointing features in MaxText."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax.training import train_state
import jax
from jax import sharding
from maxtext.common import checkpointing
import numpy as np
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint import test_utils
from orbax.checkpoint import v1 as ocp_v1
import safetensors.numpy


NamedSharding = sharding.NamedSharding
Mesh = jax.sharding.Mesh
PartitionSpec = jax.sharding.PartitionSpec
Path = epath.Path
options_lib = ocp_v1.options


class SourceCheckpointLoadingTest(parameterized.TestCase):
  """Tests for the `load_state_if_possible` function."""

  def setUp(self):
    """Sets up the test environment."""
    super().setUp()
    # Create a dummy mesh for sharding definitions
    self.mesh = Mesh(np.array(jax.devices()), axis_names=("x",))
    self.sharding = NamedSharding(self.mesh, PartitionSpec())

    self.tmp_dir = Path(self.create_tempdir().full_path)
    self.v0_root_directory = self.tmp_dir / "v0_checkpoints"
    self.v1_orbax_ckpt_path = self.tmp_dir / "v1_orbax_ckpt"
    self.safetensors_ckpt_path = self.tmp_dir / "model.safetensors"
    self.invalid_ckpt_path = self.tmp_dir / "invalid.txt"

    # State for v0 checkpoint manager
    self.state_zero = {
        "a": {"x": jax.device_put(np.arange(4, dtype=np.float32), self.sharding)},
        "b": {"y": jax.device_put(np.array(0.01), self.sharding)},
    }
    # State for v1 orbax checkpoint
    self.state_one = {
        "a": {"x": jax.device_put(np.arange(4, 8, dtype=np.float32), self.sharding)},
        "b": {"y": jax.device_put(np.array(0.02), self.sharding)},
    }
    # State for safetensors checkpoint
    self.state_two = {
        "c": np.arange(8, 12, dtype=np.float32),
        "d": np.array(0.03, dtype=np.float32),
    }

    # Create a dummy TrainState object for testing
    self.abstract_state = train_state.TrainState.create(apply_fn=lambda x: x, params=self.state_one, tx=optax.sgd(0.1))

  def test_load_full_state_from_v0_checkpoint_manager(self):
    """Tests loading a full v0 state via a CheckpointManager."""
    save_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir=str(self.v0_root_directory),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
    )
    assert save_manager is not None
    with save_manager:
      checkpointing.save_checkpoint(save_manager, 0, self.state_zero, None)

    # Create a new manager for loading
    load_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir=str(self.v0_root_directory),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
    )
    assert load_manager is not None
    with load_manager:
      abstract_state_zero = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state_zero)
      loaded_data, loaded_vars = checkpointing.load_state_if_possible(
          checkpoint_manager=load_manager,
          data_iterator=None,
          load_parameters_from_path="",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=1,
          abstract_unboxed_pre_state=abstract_state_zero,
      )

      self.assertIsNone(loaded_vars)
      test_utils.assert_tree_equal(self, self.state_zero, loaded_data["items"])

  @parameterized.named_parameters(
      ("_orbax", "orbax"),
      ("_safetensors", "safetensors"),
  )
  def test_load_full_state_from_v1_checkpoint(self, ckpt_type):
    """Tests loading a full state from a v1 checkpoint."""
    if ckpt_type == "orbax":
      path = self.v1_orbax_ckpt_path
      ocp_v1.save_pytree(path, self.state_one)
      expected_data = self.state_one
      abstract_unboxed_pre_state = jax.tree_util.tree_map(
          lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state_one
      )
      load_full_path = str(path)
    elif ckpt_type == "safetensors":
      safetensors.numpy.save_file(self.state_two, self.safetensors_ckpt_path)
      expected_data = jax.device_put(self.state_two)

      abstract_unboxed_pre_state = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state_two)
      load_full_path = str(self.safetensors_ckpt_path)
    else:
      raise ValueError(f"Unknown ckpt_type: {ckpt_type}")

    loaded_data, loaded_vars = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path=load_full_path,
        checkpoint_storage_concurrent_gb=1,
        abstract_unboxed_pre_state=abstract_unboxed_pre_state,
        enable_orbax_v1=True,
        checkpoint_conversion_fn=lambda x: x,
        source_checkpoint_layout=ckpt_type,
    )

    self.assertIsNone(loaded_vars)
    test_utils.assert_tree_equal(self, expected_data, loaded_data["items"])

  def test_load_full_state_orbax_v1_disabled(self):
    """Tests loading a v0 checkpoint from a path with enable_orbax_v1=False."""

    v0_path = self.tmp_dir / "v0_direct_ckpt"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(v0_path, self.state_zero)

    abstract_state_zero = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state_zero)

    loaded_data, loaded_vars = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path=str(v0_path),
        checkpoint_storage_concurrent_gb=1,
        abstract_unboxed_pre_state=abstract_state_zero,
        enable_orbax_v1=False,
        checkpoint_conversion_fn=None,
    )

    self.assertIsNone(loaded_vars)
    test_utils.assert_tree_equal(self, self.state_zero, loaded_data["items"])

  def test_load_full_state_from_empty_directory(self):
    """Tests that create_orbax_checkpoint_manager is still utilized when the root directory is not None but is empty."""

    manager_path = self.tmp_dir / "empty_manager_dir"
    v1_save_path = self.tmp_dir / "v1_orbax_ckpt"

    new_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir=str(manager_path),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
    )

    ocp_v1.save_pytree(v1_save_path, self.state_one)
    abstract_state_one = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state_one)
    loaded_data, loaded_vars = checkpointing.load_state_if_possible(
        checkpoint_manager=new_manager,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path=str(v1_save_path),
        checkpoint_storage_concurrent_gb=1,
        abstract_unboxed_pre_state=abstract_state_one,
        enable_orbax_v1=True,
        checkpoint_conversion_fn=None,
        source_checkpoint_layout="orbax",
    )

    self.assertIsNone(loaded_vars)
    self.assertIsNotNone(loaded_data)
    test_utils.assert_tree_equal(self, self.state_one, loaded_data["items"])

  def test_load_safetensors_with_conversion(self):
    """Tests SafeTensors conversion with a conversion function."""
    safetensors.numpy.save_file(self.state_two, self.safetensors_ckpt_path)

    def convert_func(s):
      return {
          "a": {"x": s["c"] + 1},
          "b": {"y": s["d"] + 1},
      }

    sharded_data = jax.device_put(self.state_two)
    expected_data = convert_func(sharded_data)
    source_checkpoint_layout = "safetensors"
    loaded_data, loaded_vars = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path=str(self.safetensors_ckpt_path),
        checkpoint_storage_concurrent_gb=1,
        abstract_unboxed_pre_state=self.abstract_state,
        enable_orbax_v1=True,
        checkpoint_conversion_fn=convert_func,
        source_checkpoint_layout=source_checkpoint_layout,
    )

    self.assertIsNone(loaded_vars)
    test_utils.assert_tree_equal(self, expected_data, loaded_data["items"])

  def test_load_invalid_format_raises_error(self):
    """Tests that loading an invalid file format raises an error."""
    self.invalid_ckpt_path.write_text("this is not a checkpoint")

    source_checkpoint_layout = "safetensors"
    with self.assertRaises(ocp_v1.errors.InvalidLayoutError):
      checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="",
          load_full_state_from_path=str(self.invalid_ckpt_path),
          checkpoint_storage_concurrent_gb=1,
          abstract_unboxed_pre_state=self.abstract_state,
          enable_orbax_v1=True,
          checkpoint_conversion_fn=lambda x: x,
          source_checkpoint_layout=source_checkpoint_layout,
      )


if __name__ == "__main__":
  absltest.main()
