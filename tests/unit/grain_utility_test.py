# Copyright 2026 Google LLC
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

"""Unit tests for the consolidated grain v1 ``GrainCheckpointable``."""

import asyncio
import json
import pathlib
import tempfile
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import grain
from grain import experimental as grain_experimental
import grain.sharding
from maxtext.common import grain_utility
from orbax.checkpoint import v1 as ocp


ElasticIterator = grain_experimental.ElasticIterator
GrainCheckpointable = grain_utility.GrainCheckpointable_v1


def _std_iter() -> Any:
  return iter(grain.MapDataset.range(10).to_iter_dataset())


def _elastic_iter():
  return ElasticIterator(
      grain.MapDataset.range(40),
      global_batch_size=2,
      shard_options=grain.sharding.ShardOptions(shard_index=0, shard_count=1),
  )


class _FakeRemote(grain_utility.RemoteIteratorWrapper):
  """Stands in for RemoteIteratorWrapper (colocated-python save/restore by step)."""

  def __init__(self):
    # pylint: disable=super-init-not-called
    self.saved_step = None
    self.restored_step = None

  def save_state(self, step):
    self.saved_step = step

  def restore_state(self, step):
    self.restored_step = step


async def _drive(stateful_coro):
  background = await stateful_coro
  await background


class TestGrainCheckpointable(unittest.TestCase):
  """Tests for the GrainCheckpointable class."""

  def test_standard_delegates_to_grain_native(self):
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "ckpt"
      it = _std_iter()
      for _ in range(4):
        next(it)
      expected = it.get_state()
      ocp.save_checkpointables(str(path), {"iter": GrainCheckpointable(it)})
      # grain-native writes its own per-process file name.
      self.assertTrue((path / "iter" / "process_0-of-1.json").exists())
      restored = _std_iter()
      ocp.load_checkpointables(str(path), {"iter": GrainCheckpointable(restored)})
      self.assertEqual(restored.get_state(), expected)

  def test_elastic_writes_single_shared_file(self):
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "ckpt"
      it = _elastic_iter()
      for _ in range(3):
        next(it)
      expected = it.get_state()
      ocp.save_checkpointables(str(path), {"iter": GrainCheckpointable(it)})
      self.assertTrue((path / "iter" / "process_0.json").exists())  # reshard-safe
      restored = _elastic_iter()
      ocp.load_checkpointables(str(path), {"iter": GrainCheckpointable(restored)})
      self.assertEqual(restored.get_state(), expected)

  def test_scaled_list_explicit_index_count(self):
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "ckpt"
      a, b = _std_iter(), _std_iter()
      for _ in range(2):
        next(a)
      for _ in range(6):
        next(b)
      state_a, state_b = a.get_state(), b.get_state()
      ocp.save_checkpointables(str(path), {"iter": GrainCheckpointable([(a, 0, 2), (b, 1, 2)])})
      self.assertTrue((path / "iter" / "process_0-of-2.json").exists())
      self.assertTrue((path / "iter" / "process_1-of-2.json").exists())
      ra, rb = _std_iter(), _std_iter()
      ocp.load_checkpointables(
          str(path),
          {"iter": GrainCheckpointable([ra, rb], restore_process_index=[0, 1], restore_process_count=2)},
      )
      self.assertEqual(ra.get_state(), state_a)
      self.assertEqual(rb.get_state(), state_b)

  def test_scale_up_single_with_explicit_count(self):
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "ckpt"
      it = _std_iter()
      for _ in range(7):
        next(it)
      expected = it.get_state()
      ocp.save_checkpointables(str(path), {"iter": GrainCheckpointable([(it, 0, 1)])})
      restored = _std_iter()
      ocp.load_checkpointables(
          str(path), {"iter": GrainCheckpointable(restored, restore_process_index=0, restore_process_count=1)}
      )
      self.assertEqual(restored.get_state(), expected)

  def test_remote_forwards_step(self):
    wrapper = _FakeRemote()
    asyncio.run(_drive(GrainCheckpointable(wrapper, step=7).save(None)))
    asyncio.run(_drive(GrainCheckpointable(wrapper, step=9).load(None)))
    self.assertEqual(wrapper.saved_step, 7)
    self.assertEqual(wrapper.restored_step, 9)


# pylint: disable=protected-access
class TestGrainUtilityFactories(unittest.TestCase):
  """Tests for the high-level factories for_save and for_restore."""

  def test_for_save_remote(self):
    wrapper = _FakeRemote()
    c = grain_utility.for_save(7, wrapper, 1)
    self.assertEqual(c._step, 7)
    self.assertIs(c._item, wrapper)

  def test_for_save_elastic(self):
    it = _elastic_iter()
    # Mock data_iterator to have .local_iterator
    mock_iter = mock.MagicMock()
    mock_iter.local_iterator = it
    c = grain_utility.for_save(0, mock_iter, 1)
    self.assertIs(c._item, it)

  def test_for_save_standard(self):
    it = _std_iter()
    c = grain_utility.for_save(0, it, 1)
    self.assertIs(c._item, it)

  def test_for_save_scaled_list(self):
    a, b = _std_iter(), _std_iter()
    c = grain_utility.for_save(0, [a, b], 1)
    self.assertIsInstance(c._item, list)
    self.assertEqual(len(c._item), 2)
    # specs: (item, index, total)
    self.assertIs(c._item[0][0], a)
    self.assertEqual(c._item[0][1], 0)
    self.assertEqual(c._item[0][2], 2)

  def test_for_restore_remote(self):
    wrapper = _FakeRemote()
    c = grain_utility.for_restore(None, 7, wrapper, 1)
    self.assertEqual(c._step, 7)
    self.assertIs(c._item, wrapper)

  def test_for_restore_elastic(self):
    it = _elastic_iter()
    mock_iter = mock.MagicMock()
    mock_iter.local_iterator = it
    c = grain_utility.for_restore(None, 0, mock_iter, 1)
    self.assertIs(c._item, it)


# pylint: enable=protected-access


# ------------------------------------------------------------------------------
# TODO(b/532274266): Remove everything below this line once distillation_utils
# supports the new GrainCheckpointHandler.
# ------------------------------------------------------------------------------


class GrainCheckpointableEquivalenceTest(parameterized.TestCase):
  """Tests to ensure GrainCheckpointable is equivalent to GrainCheckpointHandler."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = epath.Path(self.create_tempdir().full_path)

  def test_save_restore_equivalence_single_item(self):
    class FakeIterator:
      """A fake iterator for testing."""

      def __init__(self, state=0):
        self.state = state

      def get_state(self):
        return json.dumps({"state": self.state}).encode()

      def set_state(self, state):
        self.state = json.loads(state.decode())["state"]

      def __next__(self):
        self.state += 1
        return self.state

    iterator_v0 = FakeIterator(10)
    iterator_v1 = FakeIterator(10)

    step = 100
    v0_path = self.tmp_dir / str(step) / "iter_v0"
    v1_path = self.tmp_dir / str(step) / "iter_v1"

    # v0 Save
    handler = grain_utility.GrainCheckpointHandler()
    v0_path.mkdir(parents=True, exist_ok=True)
    handler.save(v0_path, item=iterator_v0)

    # v1 Save
    wrapper = GrainCheckpointable(iterator_v1)

    class MockDirectory:

      async def await_creation(self):
        v1_path.mkdir(parents=True, exist_ok=True)
        return v1_path

    commit_func = asyncio.run(wrapper.save(MockDirectory()))
    if commit_func:
      asyncio.run(commit_func)

    # Verify files are identical
    v0_file = v0_path / "process_0-of-1.json"
    v1_file = v1_path / "process_0-of-1.json"

    self.assertTrue(v0_file.exists())
    self.assertTrue(v1_file.exists())
    self.assertEqual(v0_file.read_text(), v1_file.read_text())

    # v0 Restore
    restored_iterator_v0 = FakeIterator(0)
    args_v0 = grain_utility.GrainCheckpointRestore(item=restored_iterator_v0)
    handler.restore(v0_path, args=args_v0)
    self.assertEqual(restored_iterator_v0.state, 10)

    # v1 Restore
    restored_iterator_v1 = FakeIterator(0)
    wrapper_restore = GrainCheckpointable(restored_iterator_v1)

    load_func = asyncio.run(wrapper_restore.load(v1_path))
    asyncio.run(load_func)
    self.assertEqual(restored_iterator_v1.state, 10)

  def test_save_restore_equivalence_list_item(self):
    class FakeIterator:
      """A fake iterator for testing."""

      def __init__(self, state=0):
        self.state = state

      def get_state(self):
        return json.dumps({"state": self.state}).encode()

      def set_state(self, state):
        self.state = json.loads(state.decode())["state"]

    iterator_a = FakeIterator(10)
    iterator_b = FakeIterator(20)

    item_v0 = [(iterator_a, 0, 2), (iterator_b, 1, 2)]
    item_v1 = [(iterator_a, 0, 2), (iterator_b, 1, 2)]

    step = 100
    v0_path = self.tmp_dir / str(step) / "iter_v0"
    v1_path = self.tmp_dir / str(step) / "iter_v1"

    # v0 Save
    handler = grain_utility.GrainCheckpointHandler()
    v0_path.mkdir(parents=True, exist_ok=True)
    handler.save(v0_path, item=item_v0)

    # v1 Save
    wrapper = GrainCheckpointable(item_v1)

    class MockDirectory:

      async def await_creation(self):
        v1_path.mkdir(parents=True, exist_ok=True)
        return v1_path

    commit_func = asyncio.run(wrapper.save(MockDirectory()))
    if commit_func:
      asyncio.run(commit_func)

    # Verify files are identical
    v0_file_0 = v0_path / "process_0-of-2.json"
    v1_file_0 = v1_path / "process_0-of-2.json"
    v0_file_1 = v0_path / "process_1-of-2.json"
    v1_file_1 = v1_path / "process_1-of-2.json"

    self.assertTrue(v0_file_0.exists())
    self.assertTrue(v1_file_0.exists())
    self.assertEqual(v0_file_0.read_text(), v1_file_0.read_text())

    self.assertTrue(v0_file_1.exists())
    self.assertTrue(v1_file_1.exists())
    self.assertEqual(v0_file_1.read_text(), v1_file_1.read_text())

    # v0 Restore
    iterators_restore_v0 = [FakeIterator(0), FakeIterator(0)]
    args_v0 = grain_utility.GrainCheckpointRestore(item=iterators_restore_v0, process_index=[0, 1], process_count=2)
    handler.restore(v0_path, args=args_v0)

    self.assertEqual(iterators_restore_v0[0].state, 10)
    self.assertEqual(iterators_restore_v0[1].state, 20)

    # v1 Restore
    iterators_restore_v1 = [FakeIterator(0), FakeIterator(0)]
    wrapper_restore = GrainCheckpointable(iterators_restore_v1, restore_process_index=[0, 1], restore_process_count=2)
    load_func = asyncio.run(wrapper_restore.load(v1_path))
    asyncio.run(load_func)
    self.assertEqual(iterators_restore_v1[0].state, 10)
    self.assertEqual(iterators_restore_v1[1].state, 20)


if __name__ == "__main__":
  absltest.main()
