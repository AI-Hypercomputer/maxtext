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

import grain
from grain import experimental as grain_experimental
import grain.sharding
from maxtext.common import grain_utility
from orbax.checkpoint import v1 as ocp


ElasticIterator = grain_experimental.ElasticIterator
GrainCheckpointable = grain_utility.GrainCheckpointable


def _std_iter() -> Any:
  return iter(grain.MapDataset.range(10).to_iter_dataset())


def _elastic_iter():
  return ElasticIterator(
      grain.MapDataset.range(40),
      global_batch_size=2,
      shard_options=grain.sharding.ShardOptions(shard_index=0, shard_count=1),
  )


class _FakeRemote(grain_utility.RemoteIteratorWrapper):
  """Stands in for RemoteIteratorWrapper.

  With `elastic=True` the controller gets and sets the global position; otherwise the colocated hosts save and restore
  their own state files in the directory they are given.
  """

  def __init__(self, events=None, elastic=False, state=None):
    # pylint: disable=super-init-not-called
    self.elastic = elastic
    self.state = state
    self.saved_directory = None
    self.restored_directory = None
    self.events = [] if events is None else events

  def save_state(self, directory):
    self.saved_directory = directory
    self.events.append("saved")

  def restore_state(self, directory):
    self.restored_directory = directory

  def get_state(self):
    return self.state

  def set_state(self, state):
    self.state = state


class _FakeDirectory:
  """Stands in for the directory Orbax hands to `save`, which Orbax may not have created yet."""

  def __init__(self, path, events):
    self.path = path
    self.events = events

  async def await_creation(self):
    self.events.append("created")
    return self.path


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

  def test_remote_saves_into_the_created_directory_in_the_blocking_phase(self):
    """The colocated hosts write into the directory Orbax creates (e.g. its temporary one), before `save` returns,
    so that the saved position matches the step. A restore reads from the directory it is given."""
    events = []
    wrapper = _FakeRemote(events)
    created = pathlib.Path("/ckpt/7.orbax-checkpoint-tmp/iter")

    async def save():
      background = await GrainCheckpointable(wrapper).save(_FakeDirectory(created, events))
      events.append("returned")
      await background

    asyncio.run(save())
    asyncio.run(_drive(GrainCheckpointable(wrapper).load(pathlib.Path("/other_ckpt/9/iter"))))

    self.assertEqual(events, ["created", "saved", "returned"])
    self.assertEqual(wrapper.saved_directory, created)
    self.assertEqual(wrapper.restored_directory, pathlib.Path("/other_ckpt/9/iter"))

  def test_elastic_remote_position_is_written_by_the_controller_in_the_background(self):
    """Like ElasticIterator: the position is taken in the blocking phase and written once the directory exists."""
    with tempfile.TemporaryDirectory() as d:
      created = pathlib.Path(d)
      wrapper = _FakeRemote(elastic=True, state={"global_next_index": 42})

      async def save():
        background = await GrainCheckpointable(wrapper).save(_FakeDirectory(created, wrapper.events))
        self.assertEqual(wrapper.events, [])  # Neither created nor written before `save` returns.
        wrapper.state = {"global_next_index": 50}  # Training goes on; the checkpoint keeps the position of its step.
        await background

      asyncio.run(save())

      self.assertEqual(wrapper.events, ["created"])  # The colocated hosts don't write.
      self.assertEqual(json.loads((created / "process_0.json").read_text()), {"global_next_index": 42})

      restored = _FakeRemote(elastic=True, state={"global_next_index": 0})
      asyncio.run(_drive(GrainCheckpointable(restored).load(created)))
      self.assertEqual(restored.state, {"global_next_index": 42})
      self.assertIsNone(restored.restored_directory)


# pylint: disable=protected-access
class TestGrainUtilityFactories(unittest.TestCase):
  """Tests for the high-level factories for_save and for_restore."""

  def test_for_save_remote(self):
    wrapper = _FakeRemote()
    c = grain_utility.for_save(wrapper, 1)
    self.assertIs(c._item, wrapper)

  def test_for_save_elastic(self):
    it = _elastic_iter()
    # Mock data_iterator to have .local_iterator
    mock_iter = mock.MagicMock()
    mock_iter.local_iterator = it
    c = grain_utility.for_save(mock_iter, 1)
    self.assertIs(c._item, it)

  def test_for_save_standard(self):
    it = _std_iter()
    c = grain_utility.for_save(it, 1)
    self.assertIs(c._item, it)

  def test_for_save_scaled_list(self):
    a, b = _std_iter(), _std_iter()
    c = grain_utility.for_save([a, b], 1)
    self.assertIsInstance(c._item, list)
    self.assertEqual(len(c._item), 2)
    # specs: (item, index, total)
    self.assertIs(c._item[0][0], a)
    self.assertEqual(c._item[0][1], 0)
    self.assertEqual(c._item[0][2], 2)

  def test_for_restore_remote(self):
    wrapper = _FakeRemote()
    c = grain_utility.for_restore(None, 7, wrapper, 1)
    self.assertIs(c._item, wrapper)

  def test_for_restore_elastic(self):
    it = _elastic_iter()
    mock_iter = mock.MagicMock()
    mock_iter.local_iterator = it
    c = grain_utility.for_restore(None, 0, mock_iter, 1)
    self.assertIs(c._item, it)


# pylint: enable=protected-access
