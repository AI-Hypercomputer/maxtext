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
pytestmark = [pytest.mark.tpu_only]

import json
import os
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
import grain
import jax
from flax import nnx
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
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=iterator, root_directory=self.test_dir, options=self.options
    )

    # Create dummy model so 'model_params' is not empty
    model = DummyModel(nnx.Rngs(0))

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

    restore_manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=new_iterator, root_directory=self.test_dir, options=self.options
    )

    with mock.patch.object(jax, "process_index", return_value=0), mock.patch.object(jax, "process_count", return_value=1):
      restored_iter = restore_manager.restore_iterator()

    self.assertIsNotNone(restored_iter)
    self.assertEqual(new_iterator.counter, 10)

  def test_restore_returns_none_if_no_checkpoint(self):
    """Verifies restore_iterator returns None gracefully if no checkpoint exists."""
    iterator = FakeGrainIterator()
    manager = distillation_utils.MaxTextCheckpointManager(
        raw_iterator=iterator, root_directory=self.test_dir, options=self.options
    )

    # No save called
    result = manager.restore_iterator()
    self.assertIsNone(result)


if __name__ == "__main__":
  absltest.main()
