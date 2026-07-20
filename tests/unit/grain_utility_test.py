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
      ocp.load_checkpointables(str(path), {"iter": GrainCheckpointable([ra, rb], process_index=[0, 1], process_count=2)})
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
      ocp.load_checkpointables(str(path), {"iter": GrainCheckpointable(restored, process_index=0, process_count=1)})
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
