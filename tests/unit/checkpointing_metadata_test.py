# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for checkpoint custom-metadata reading, resume-step discovery, and the
pre-restore weight checks. custom_metadata is written at the *step* directory."""

import shutil
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from maxtext.common import checkpointing


def _save_manager_checkpoint(root, step, custom_metadata):
  """Saves a tiny pytree the way MaxText's save_checkpoint does (Composite items=...)."""
  state = {"params": {"params": {"w": np.ones((2, 2), np.float32)}}}
  manager = ocp.CheckpointManager(root)
  manager.save(
      step,
      args=ocp.args.Composite(items=ocp.args.PyTreeSave(item=state)),
      custom_metadata=custom_metadata,
  )
  manager.wait_until_finished()
  manager.close()


class LoadCheckpointMetadataTest(unittest.TestCase):
  """load_checkpoint_metadata must see manager-saved custom_metadata from either path form."""

  def setUp(self):
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_reads_metadata_at_step_dir(self):
    _save_manager_checkpoint(self.root, 0, {"scan_layers": True})
    metadata = checkpointing.load_checkpoint_metadata(f"{self.root}/0")
    self.assertEqual(metadata.get("scan_layers"), True)

  def test_reads_metadata_at_items_dir_via_parent(self):
    """The documented load_parameters_path form (…/<step>/items) must also find the metadata."""
    _save_manager_checkpoint(self.root, 0, {"scan_layers": False})
    metadata = checkpointing.load_checkpoint_metadata(f"{self.root}/0/items")
    self.assertEqual(metadata.get("scan_layers"), False)

  def test_missing_checkpoint_returns_empty(self):
    metadata = checkpointing.load_checkpoint_metadata(f"{self.root}/does_not_exist")
    self.assertEqual(metadata, {})


class LatestCheckpointStepDirTest(unittest.TestCase):
  """latest_checkpoint_step_dir must return the newest step directory, or None."""

  def setUp(self):
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_returns_newest_step(self):
    _save_manager_checkpoint(self.root, 0, {"scan_layers": True})
    _save_manager_checkpoint(self.root, 3, {"scan_layers": True})
    step_dir = checkpointing.latest_checkpoint_step_dir(self.root)
    self.assertIsNotNone(step_dir)
    self.assertEqual(step_dir.name, "3")

  def test_empty_dir_returns_none(self):
    self.assertIsNone(checkpointing.latest_checkpoint_step_dir(self.root))

  def test_missing_dir_returns_none(self):
    self.assertIsNone(checkpointing.latest_checkpoint_step_dir(f"{self.root}/nope"))

  def test_empty_path_returns_none(self):
    self.assertIsNone(checkpointing.latest_checkpoint_step_dir(""))

  def test_step_dir_metadata_roundtrip(self):
    """The step dir found by discovery carries the metadata the run was saved with."""
    _save_manager_checkpoint(self.root, 0, {"scan_layers": True})
    step_dir = checkpointing.latest_checkpoint_step_dir(self.root)
    metadata = checkpointing.load_checkpoint_metadata(step_dir)
    self.assertEqual(metadata.get("scan_layers"), True)


class PreRestoreWeightCheckTest(unittest.TestCase):
  """The pre-restore checks must name shape AND missing mismatches from metadata alone."""

  def setUp(self):
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    _save_manager_checkpoint(self.root, 0, {"scan_layers": True})
    # item_metadata materializes only when the manager knows the item's handler,
    # mirroring create_orbax_checkpoint_manager's registration.
    manager = ocp.CheckpointManager(
        self.root, item_names=("items",), item_handlers={"items": ocp.PyTreeCheckpointHandler()}
    )
    self.addCleanup(manager.close)
    # Indexed, not attribute access: Composite.items is a method.
    self.stored_params = manager.item_metadata(0)["items"].tree["params"]["params"]

  def test_shape_change_is_named_before_restore(self):
    want = {"w": jax.ShapeDtypeStruct((4, 4), jnp.float32)}  # stored is (2, 2)
    with self.assertRaises(ValueError) as ctx:
      checkpointing._raise_on_weight_mismatch(want, self.stored_params)  # pylint: disable=protected-access
    message = str(ctx.exception)
    self.assertIn("Checkpoint does not match the model", message)
    self.assertIn("'w'", message)
    self.assertIn("(2, 2)", message)

  def test_missing_weight_is_named_before_restore(self):
    want = {
        "w": jax.ShapeDtypeStruct((2, 2), jnp.float32),
        "w_extra": jax.ShapeDtypeStruct((2, 2), jnp.float32),
    }
    with self.assertRaises(ValueError) as ctx:
      checkpointing._raise_on_weight_mismatch(want, self.stored_params)  # pylint: disable=protected-access
    message = str(ctx.exception)
    self.assertIn("Checkpoint does not match the model", message)
    self.assertIn("'w_extra'", message)
    self.assertIn("missing", message)

  def test_matching_weights_pass(self):
    want = {"w": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    checkpointing._raise_on_weight_mismatch(want, self.stored_params)  # pylint: disable=protected-access

  def test_params_only_path_metadata_names_shape_change(self):
    """load_params_from_path's pre-check form: Checkpointer.metadata at the items dir."""
    ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    self.addCleanup(ckptr.close)
    stored_tree = ckptr.metadata(f"{self.root}/0/items").item_metadata.tree
    want = {"w": jax.ShapeDtypeStruct((4, 4), jnp.float32)}  # stored is (2, 2)
    with self.assertRaises(ValueError) as ctx:
      checkpointing._raise_on_weight_mismatch(want, stored_tree["params"]["params"])  # pylint: disable=protected-access
    message = str(ctx.exception)
    self.assertIn("Checkpoint does not match the model", message)
    self.assertIn("(2, 2)", message)

  def test_pre_restore_check_skips_when_collection_not_located(self):
    """An empty stored collection (unfamiliar layout) must skip, not report every weight missing."""
    want = {"w": jax.ShapeDtypeStruct((2, 2), jnp.float32)}
    checkpointing._pre_restore_weight_check(want, {})  # pylint: disable=protected-access

  def test_pre_restore_check_still_raises_on_real_mismatch(self):
    """A located-but-mismatched collection must still fail before restore."""
    want = {
        "w": jax.ShapeDtypeStruct((2, 2), jnp.float32),
        "w_extra": jax.ShapeDtypeStruct((2, 2), jnp.float32),  # not in the checkpoint
    }
    with self.assertRaises(ValueError) as ctx:
      checkpointing._pre_restore_weight_check(want, self.stored_params)  # pylint: disable=protected-access
    self.assertIn("'w_extra'", str(ctx.exception))


if __name__ == "__main__":
  unittest.main()
