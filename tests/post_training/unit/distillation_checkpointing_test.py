# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Distillation Checkpointing logic."""

import pytest

pytest.importorskip("tunix")
pytestmark = [pytest.mark.tpu_only, pytest.mark.post_training]

import json
import os
from types import SimpleNamespace
from etils import epath
import jax.numpy as jnp
import optax
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
import grain
import jax
from flax import nnx
import orbax.checkpoint as ocp
from maxtext.common import checkpointing
from maxtext.trainers.post_train.distillation import distillation_utils


class FakeGrainIterator(grain.DatasetIterator):
  """A simple iterator that mimics Grain's stateful interface."""

  def __init__(self):
    super().__init__()
    # Initialize _closed to satisfy grain.DatasetIterator.__del__
    self._closed = False
    self.counter = 0

  def __next__(self):
    self.counter += 1
    return self.counter

  def get_state(self):
    return {"current_count": self.counter}

  def set_state(self, state):
    self.counter = state["current_count"]

  @property
  def element_spec(self):
    return int


class DummyModel(nnx.Module):
  """Minimal NNX module to generate non-empty params for Orbax."""

  def __init__(self, rngs):
    self.layer = nnx.Linear(1, 1, rngs=rngs)


class MaxTextCheckpointManagerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)

  def tearDown(self):
    if os.path.exists(self.test_dir):
      shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_save_and_restore_iterator(self):
    """Verifies that the iterator state is saved to JSON and restored correctly."""

    # 1. Setup Iterator and Advance
    iterator = FakeGrainIterator()
    for _ in range(10):
      next(iterator)
    self.assertEqual(iterator.counter, 10)

    # 2. Save Checkpoint
    mock_student_config = mock.Mock()
    mock_student_config.learn_to_init_mode = False
    mock_student_config.scan_layers = True
    mock_student_config.lora = None
    mock_student_config.checkpoint_storage_concurrent_gb = None
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=iterator, root_directory=self.test_dir, student_config=mock_student_config, options=self.options
    )

    # Create dummy model so 'model_params' is not empty
    model = mock.Mock()
    model.student_model = DummyModel(nnx.Rngs(0))

    # Mock jax.process_index/count to simulate single host
    with mock.patch.object(jax, "process_index", return_value=0), mock.patch.object(jax, "process_count", return_value=1):
      # Pass the dummy model here
      saved = manager.save(step=100, model=model, optimizer=None, force=True)

    manager.wait_until_finished()
    self.assertTrue(saved)

    # 3. Verify File Structure
    # MaxText GrainHandler saves as: <dir>/<step>/iter/process_0-of-1.json
    expected_file = os.path.join(self.test_dir, "100", "iter", "process_0-of-1.json")
    self.assertTrue(os.path.exists(expected_file), f"Expected file {expected_file} not found")

    with open(expected_file, "r", encoding="utf-8") as f:
      content = json.load(f)
      self.assertEqual(content["current_count"], 10)

    # 4. Restore into New Iterator
    new_iterator = FakeGrainIterator()
    self.assertEqual(new_iterator.counter, 0)

    mock_student_config_restore = mock.Mock()
    mock_student_config_restore.learn_to_init_mode = False
    mock_student_config_restore.scan_layers = True
    mock_student_config_restore.lora = None
    mock_student_config_restore.checkpoint_storage_concurrent_gb = None
    restore_manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=new_iterator,
        root_directory=self.test_dir,
        student_config=mock_student_config_restore,
        options=self.options,
    )

    with mock.patch.object(jax, "process_index", return_value=0), mock.patch.object(jax, "process_count", return_value=1):
      restored_iter = restore_manager.restore_iterator()

    self.assertIsNotNone(restored_iter)
    self.assertEqual(new_iterator.counter, 10)

  def test_restore_returns_none_if_no_checkpoint(self):
    """Verifies restore_iterator returns None gracefully if no checkpoint exists."""
    iterator = FakeGrainIterator()
    mock_student_config_restore = mock.Mock()
    mock_student_config_restore.learn_to_init_mode = False
    mock_student_config_restore.scan_layers = True
    mock_student_config_restore.lora = None
    mock_student_config_restore.checkpoint_storage_concurrent_gb = None
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=iterator,
        root_directory=self.test_dir,
        student_config=mock_student_config_restore,
        options=self.options,
    )

    # No save called
    result = manager.restore_iterator()
    self.assertIsNone(result)


class MaxTextCheckpointManagerLayoutTest(absltest.TestCase):
  """Distillation checkpoints only the student, in MaxText's on-disk layout."""

  class Bundle(nnx.Module):
    """Stand-in for the teacher/student ModelBundle the trainer holds."""

    def __init__(self, student, teacher):
      self.student_model = student
      self.teacher_model = teacher

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)

  def tearDown(self):
    if os.path.exists(self.test_dir):
      shutil.rmtree(self.test_dir)
    super().tearDown()

  def _save(self, learn_to_init_mode=False):
    """Saves a checkpoint and returns its on-disk leaf paths."""
    student, teacher = DummyModel(nnx.Rngs(0)), DummyModel(nnx.Rngs(1))
    bundle = self.Bundle(student, teacher)
    optimizer = nnx.Optimizer(student, optax.adamw(1e-3), wrt=nnx.Param)
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=None,
        root_directory=self.test_dir,
        student_config=SimpleNamespace(learn_to_init_mode=learn_to_init_mode, scan_layers=True, lora=None),
        options=self.options,
    )
    self.assertTrue(manager.save(1, bundle, optimizer, force=True))
    manager.wait_until_finished()
    manager.close()

    metadata = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).metadata(epath.Path(self.test_dir) / "1" / "items")
    tree = getattr(metadata.item_metadata, "tree", metadata.item_metadata)
    return list(ocp.tree.to_flat_dict(tree, sep="/"))

  def test_saves_the_student_in_maxtext_layout(self):
    keys = self._save()

    self.assertTrue(any(k.startswith("params/params/layer/") for k in keys), keys)
    self.assertTrue(any(k.startswith("opt_state/") for k in keys), keys)
    # The bundle's wrapper level and the teacher stay out of the checkpoint.
    self.assertEqual([k for k in keys if "student_model" in k.split("/") or "teacher_model" in k.split("/")], [])

  def test_the_students_scan_setting_is_recorded(self):
    """Distillation checkpoints the student, so the metadata has to describe the student."""
    student, teacher = DummyModel(nnx.Rngs(0)), DummyModel(nnx.Rngs(1))
    bundle = self.Bundle(student, teacher)
    optimizer = nnx.Optimizer(student, optax.adamw(1e-3), wrt=nnx.Param)
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=None,
        root_directory=self.test_dir,
        student_config=SimpleNamespace(learn_to_init_mode=False, scan_layers=False, lora=None),
        options=self.options,
    )
    manager.save(1, bundle, optimizer, force=True)
    manager.wait_until_finished()
    manager.close()

    metadata = checkpointing.load_checkpoint_metadata(os.path.join(self.test_dir, "1", "items"))
    self.assertIs(metadata.get("scan_layers"), False)

  def test_learn_to_init_mode_leaves_the_optimizer_out(self):
    keys = self._save(learn_to_init_mode=True)

    self.assertTrue(any(k.startswith("params/params/layer/") for k in keys), keys)
    self.assertEqual([k for k in keys if k.startswith("opt_state")], [])

  def test_restores_the_student(self):
    student, teacher = DummyModel(nnx.Rngs(0)), DummyModel(nnx.Rngs(1))
    bundle = self.Bundle(student, teacher)
    optimizer = nnx.Optimizer(student, optax.adamw(1e-3), wrt=nnx.Param)
    optimizer.update(student, jax.tree.map(jnp.ones_like, nnx.state(student, nnx.Param)))
    trained = jnp.asarray(student.layer.kernel[...])

    def manager():
      return distillation_utils.MaxTextCheckpointManager(
          raw_iterator=None,
          root_directory=self.test_dir,
          student_config=SimpleNamespace(learn_to_init_mode=False),
          options=self.options,
      )

    saver = manager()
    saver.save(1, bundle, optimizer, force=True)
    saver.wait_until_finished()
    saver.close()

    fresh_student = DummyModel(nnx.Rngs(0))
    fresh_bundle = self.Bundle(fresh_student, teacher)
    fresh_optimizer = nnx.Optimizer(fresh_student, optax.adamw(1e-3), wrt=nnx.Param)
    restorer = manager()
    step, _ = restorer.maybe_restore(fresh_bundle, fresh_optimizer)
    restorer.close()

    self.assertEqual(step, 1)
    self.assertTrue(jnp.array_equal(trained, fresh_student.layer.kernel[...]))


if __name__ == "__main__":
  absltest.main()
