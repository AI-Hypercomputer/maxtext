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
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
import grain
import jax
from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
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
    mock_student_config.distill_save_trainable_params_only = False
    mock_student_config.distill_skip_optimizer_checkpoint = False
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
    mock_student_config_restore.distill_save_trainable_params_only = False
    mock_student_config_restore.distill_skip_optimizer_checkpoint = False
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
    mock_student_config_restore.distill_save_trainable_params_only = False
    mock_student_config_restore.distill_skip_optimizer_checkpoint = False
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=iterator,
        root_directory=self.test_dir,
        student_config=mock_student_config_restore,
        options=self.options,
    )

    # No save called
    result = manager.restore_iterator()
    self.assertIsNone(result)


class _Student(nnx.Module):
  """Student with trainable and frozen params plus non-Param state."""

  def __init__(self, seed):
    rngs = nnx.Rngs(seed)
    self.trainable = nnx.Linear(2, 2, rngs=rngs)
    self.frozen = nnx.Linear(2, 2, rngs=rngs)
    self.stat = nnx.BatchStat(jnp.full((2,), float(seed)))


def _is_trainable(path, x):
  return isinstance(x, nnx.Param) and path[0] == "trainable"


def _student_config(save_trainable_params_only=False, skip_optimizer_checkpoint=False):
  config = mock.Mock()
  config.learn_to_init_mode = False
  config.distill_save_trainable_params_only = save_trainable_params_only
  config.distill_skip_optimizer_checkpoint = skip_optimizer_checkpoint
  return config


class TrainableOnlyCheckpointTest(absltest.TestCase):
  """Save/restore round trips for the distillation checkpoint options."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True, enable_async_checkpointing=False)

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)
    super().tearDown()

  def _trained_student(self):
    model = _Student(0)
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=_is_trainable)
    grads = jax.tree.map(jnp.ones_like, nnx.state(model, _is_trainable))
    optimizer.update(model, grads)
    return model, optimizer

  def _manager(self, config):
    return distillation_utils.MaxTextCheckpointManager(
        raw_iterator=None,
        root_directory=self.test_dir,
        student_config=config,
        options=self.options,
        trainable_param_filter=_is_trainable,
    )

  def _save(self, config):
    model, optimizer = self._trained_student()
    manager = self._manager(config)
    self.assertTrue(manager.save(step=1, model=model, optimizer=optimizer, force=True))
    manager.wait_until_finished()
    manager.close()
    return model

  def _restore(self, config):
    model = _Student(1)
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=_is_trainable)
    manager = self._manager(config)
    step, _ = manager.maybe_restore(model, optimizer)
    restored_optimizer_state = manager.restored_optimizer_state
    manager.close()
    return step, model, optimizer, restored_optimizer_state

  def test_default_round_trip_restores_full_state(self):
    saved = self._save(_student_config())
    step, model, optimizer, restored_opt = self._restore(_student_config())
    self.assertEqual(step, 1)
    self.assertTrue(restored_opt)
    np.testing.assert_array_equal(model.trainable.kernel.get_value(), saved.trainable.kernel.get_value())
    np.testing.assert_array_equal(model.frozen.kernel.get_value(), saved.frozen.kernel.get_value())
    # Non-Param state is part of the default checkpoint.
    np.testing.assert_array_equal(model.stat.get_value(), saved.stat.get_value())
    self.assertEqual(optimizer.step.get_value(), 1)

  def test_trainable_only_round_trip(self):
    config = _student_config(save_trainable_params_only=True)
    saved = self._save(config)
    metadata = self._manager(config)._checkpointer.checkpointables_metadata(1)  # pylint: disable=protected-access
    saved_leaves = jax.tree_util.tree_leaves_with_path(metadata.metadata["model_params"])
    saved_paths = [jax.tree_util.keystr(p) for p, _ in saved_leaves]
    self.assertNotEmpty(saved_paths)
    self.assertTrue(all("trainable" in p for p in saved_paths), saved_paths)

    step, model, optimizer, restored_opt = self._restore(config)
    self.assertEqual(step, 1)
    np.testing.assert_array_equal(model.trainable.kernel.get_value(), saved.trainable.kernel.get_value())
    # Frozen params and non-Param state keep the values of the freshly initialized model.
    np.testing.assert_array_equal(model.frozen.kernel.get_value(), _Student(1).frozen.kernel.get_value())
    np.testing.assert_array_equal(model.stat.get_value(), _Student(1).stat.get_value())
    # The optimizer state saved with the trainable params is restored as well.
    self.assertTrue(restored_opt)
    self.assertEqual(optimizer.step.get_value(), 1)

  def test_skip_optimizer_checkpoint(self):
    for trainable_only in (False, True):
      with self.subTest(trainable_only=trainable_only):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        config = _student_config(save_trainable_params_only=trainable_only, skip_optimizer_checkpoint=True)
        self._save(config)
        step, _, optimizer, restored_opt = self._restore(config)
        self.assertEqual(step, 1)
        self.assertFalse(restored_opt)
        self.assertEqual(optimizer.step.get_value(), 0)


if __name__ == "__main__":
  absltest.main()
