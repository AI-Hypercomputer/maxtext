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

# pylint: disable=missing-module-docstring, missing-function-docstring, line-too-long, g-generic-assert, protected-access, import-outside-toplevel, super-init-not-called, missing-class-docstring
import itertools
import json

import pathlib
import sys
import tempfile


from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import mesh_utils
import jax.experimental.colocated_python
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.input_pipeline import multihost_dataloading
from tests.utils.test_helpers import get_test_base_output_directory
from tests.utils.test_helpers import get_test_config_path
from tests.utils.test_helpers import get_test_dataset_path
import numpy as np
import pytest

# Mock jax.experimental.colocated_python before it is imported by
# multihost_dataloading
mock.patch.object(
    jax.experimental.colocated_python,
    "colocated_python_class",
    lambda cls: cls,
).start()
mock.patch.object(
    jax.experimental.colocated_python,
    "colocated_cpu_devices",
    lambda x: x,
).start()


class MockIterator:
  """Mock iterator for testing dataloading state saving/restoring."""

  def __init__(self, mesh_size):
    self.state = 0
    self.mesh_size = mesh_size

  def __next__(self):
    self.state += 1
    return np.full((self.mesh_size, 1), self.state, dtype=np.int32)

  def get_state(self) -> dict[str, int]:
    return {"state": self.state}

  def set_state(self, state: dict[str, int]):
    self.state = state["state"]


class MockDataloader:
  """Mock dataloader for testing."""

  def __init__(self, mesh_size):
    self.mesh_size = mesh_size

  def __iter__(self) -> MockIterator:
    return MockIterator(self.mesh_size)


def _get_test_mesh_shapes_named():
  return [
      ("1_device", (1, 1)),
      ("2_devices", (2, 1)),
      ("4_devices", (2, 2)),
  ]


class MultihostDataloadingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Note: this test uses gs://max-experiments/ (not runner logs) in cloud mode
    base_output_directory = get_test_base_output_directory(cloud_path="gs://max-experiments/")
    dataset_path = get_test_dataset_path(cloud_path="gs://maxtext-dataset/")
    batch_size = len(jax.devices())
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        base_output_directory=base_output_directory,
        dataset_path=dataset_path,
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        enable_checkpointing=False,
    )
    mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), config.mesh_axes)
    # Create 2 distinct batches and cycle through them infinitely.
    global_data = np.arange(batch_size * 2 * config.max_target_length, dtype=np.int32).reshape(
        (batch_size * 2, config.max_target_length)
    )
    data_batches = [global_data[:batch_size], global_data[batch_size:]]
    self.multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(itertools.cycle(data_batches), self.mesh)

  @pytest.mark.tpu_only
  def test_batch_sharded_data_pipeline(self):
    first_batch = next(self.multihost_gen)
    sec_batch = next(self.multihost_gen)
    self.assertFalse(np.array_equal(first_batch, sec_batch, equal_nan=True))

  @parameterized.named_parameters(*_get_test_mesh_shapes_named())
  def test_remote_iterator_wrapper_save_state(self, mesh_shape):
    if jax.default_backend() == "gpu":
      self.skipTest("Skipping colocated python tests on GPU")
    mesh_size = mesh_shape[0] * mesh_shape[1]
    if mesh_size > len(jax.devices()):
      self.skipTest(
          f"Skipping test because available devices ({len(jax.devices())}) is"
          f" less than required mesh size ({mesh_size}) for shape {mesh_shape}."
      )

    devs = jax.devices()[:mesh_size]
    devices = mesh_utils.create_device_mesh(mesh_shape, devs)
    mesh = Mesh(devices, ("x", "y"))

    def get_ds_fn(dataloading_host_index, dataloading_host_count):
      del dataloading_host_index, dataloading_host_count
      return MockDataloader(mesh_size)

    def preprocessing_fn(dataset):
      return dataset

    global_shape = (mesh_size, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
      wrapper = multihost_dataloading.RemoteIteratorWrapper(
          get_ds_fn=get_ds_fn,
          preprocessing_fn=preprocessing_fn,
          global_mesh=mesh,
          global_shape=global_shape,
          checkpoint_path=tmpdir,
          elastic=False,
      )
      # Advance state once so the value is 1
      next(wrapper)

      wrapper.save_state(step=5)

      # Verify that a file was written in the tempdir containing {"state": 1}
      json_files = list(pathlib.Path(tmpdir).glob("**/*.json"))
      self.assertEqual(len(json_files), 1, f"Expected 1 JSON file, found: {json_files}")
      written_content = json_files[0].read_text()
      self.assertEqual(json.loads(written_content), {"state": 1})

  @parameterized.named_parameters(*_get_test_mesh_shapes_named())
  def test_remote_iterator_wrapper_restore_state(self, mesh_shape):
    if jax.default_backend() == "gpu":
      self.skipTest("Skipping colocated python tests on GPU")
    mesh_size = mesh_shape[0] * mesh_shape[1]
    if mesh_size > len(jax.devices()):
      self.skipTest(
          f"Skipping test because available devices ({len(jax.devices())}) is"
          f" less than required mesh size ({mesh_size}) for shape {mesh_shape}."
      )

    devs = jax.devices()[:mesh_size]
    devices = mesh_utils.create_device_mesh(mesh_shape, devs)
    mesh = Mesh(devices, ("x", "y"))

    def get_ds_fn(dataloading_host_index, dataloading_host_count):
      del dataloading_host_index, dataloading_host_count
      return MockDataloader(mesh_size)

    def preprocessing_fn(dataset):
      return dataset

    global_shape = (mesh_size, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
      step = 5
      state_dir = pathlib.Path(tmpdir) / str(step) / "iter"
      state_dir.mkdir(parents=True, exist_ok=True)
      state_file = state_dir / "process_0-of-1.json"
      state_file.write_text('{"state": 10}')

      wrapper = multihost_dataloading.RemoteIteratorWrapper(
          get_ds_fn=get_ds_fn,
          preprocessing_fn=preprocessing_fn,
          global_mesh=mesh,
          global_shape=global_shape,
          checkpoint_path=tmpdir,
          elastic=False,
      )

      wrapper.restore_state(step=5)
      val = next(wrapper)

      # Next value should be 11 (state 10 + 1)
      self.assertEqual(val.addressable_data(0)[0], 11)

  def test_singleton_object_store_cleanup_preserves_mtc_objects(self):
    """Verifies that RemoteIterator initialization preserves MTC objects in SINGLETON_OBJECT_STORE."""
    from jax.experimental.colocated_python.obj_backend import SINGLETON_OBJECT_STORE, _ObjectState

    class MockMtcCheckpointer:

      def __init__(self):
        self.closed = False

      def close(self):
        self.closed = True

    class MockRemoteIterator(multihost_dataloading.RemoteIterator):

      def __init__(self):
        self.closed = False

      def close(self, dummy_array=None):
        self.closed = True

    mtc_obj = MockMtcCheckpointer()
    old_iter = MockRemoteIterator()

    with SINGLETON_OBJECT_STORE._lock:
      SINGLETON_OBJECT_STORE._storage.clear()
      SINGLETON_OBJECT_STORE._storage["mtc_checkpointer_uid"] = _ObjectState(is_being_initialized=False, obj=mtc_obj)
      SINGLETON_OBJECT_STORE._storage["old_iterator_uid"] = _ObjectState(is_being_initialized=False, obj=old_iter)

    def dummy_get_ds(dataloading_host_index, dataloading_host_count):
      del dataloading_host_index, dataloading_host_count
      return MockDataloader(1)

    _ = multihost_dataloading.RemoteIterator(
        get_ds_fn=dummy_get_ds,
        preprocessing_fn=lambda dataset: dataset,
        global_shape=(1, 1),
        checkpoint_path="/tmp",
        elastic=False,
    )

    with SINGLETON_OBJECT_STORE._lock:
      # MTC object must still be in SINGLETON_OBJECT_STORE and not closed
      self.assertIn("mtc_checkpointer_uid", SINGLETON_OBJECT_STORE._storage)
      self.assertIs(SINGLETON_OBJECT_STORE._storage["mtc_checkpointer_uid"].obj, mtc_obj)
      self.assertFalse(mtc_obj.closed)

      # Old RemoteIterator must be evicted and closed
      self.assertNotIn("old_iterator_uid", SINGLETON_OBJECT_STORE._storage)
      self.assertTrue(old_iter.closed)
      SINGLETON_OBJECT_STORE._storage.clear()

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("os.listdir")
  @mock.patch("builtins.open")
  @mock.patch("os.kill")
  def test_orphan_cleanup_preserves_healthy_multiprocessing_processes(
      self, mock_kill, mock_open_fn, mock_listdir, mock_exists
  ):
    """Verifies that orphan cleanup terminates only PPid==1 processes and preserves healthy ones."""
    del mock_exists
    mock_listdir.return_value = ["101", "102", "103"]

    def fake_open(filepath, *args, **kwargs):
      del args, kwargs
      if filepath == "/proc/101/cmdline":
        return mock.mock_open(read_data=b"python -m multiprocessing.spawn").return_value
      elif filepath == "/proc/101/status":
        return mock.mock_open(read_data="Name:\tpython\nPPid:\t1\n").return_value
      elif filepath == "/proc/102/cmdline":
        return mock.mock_open(read_data=b"python -m multiprocessing.spawn").return_value
      elif filepath == "/proc/102/status":
        return mock.mock_open(read_data="Name:\tpython\nPPid:\t50\n").return_value
      elif filepath == "/proc/103/cmdline":
        return mock.mock_open(read_data=b"/bin/bash").return_value
      elif filepath == "/proc/103/status":
        return mock.mock_open(read_data="Name:\tbash\nPPid:\t1\n").return_value
      raise FileNotFoundError(filepath)

    mock_open_fn.side_effect = fake_open

    multihost_dataloading._cleanup_orphaned_multiprocessing_processes()

    # Verify that only PID 101 (orphan) was sent kill signals
    killed_pids = [call.args[0] for call in mock_kill.call_args_list]
    self.assertIn(101, killed_pids)
    self.assertNotIn(102, killed_pids)
    self.assertNotIn(103, killed_pids)

  @mock.patch("os.getpid", return_value=1)
  @mock.patch("os.path.exists")
  @mock.patch("os.kill")
  def test_orphan_cleanup_skips_when_running_as_pid_1(self, mock_kill, mock_exists, mock_getpid):
    """Verifies that orphan cleanup does not kill direct child workers when running as container PID 1."""
    del mock_exists, mock_getpid
    multihost_dataloading._cleanup_orphaned_multiprocessing_processes()
    mock_kill.assert_not_called()

  def test_terminate_iterator_workers_and_close(self):
    """Verifies targeted worker termination on iterator/dataloader objects."""

    class MockWorker:

      def __init__(self):
        self.terminated = False
        self.joined = False

      def terminate(self):
        self.terminated = True

      def join(self, timeout=None):
        del timeout
        self.joined = True

    class MockPool:

      def __init__(self):
        self.terminated = False

      def terminate(self):
        self.terminated = True

    class MockParent:

      def __init__(self):
        self._workers = [MockWorker(), MockWorker()]
        self._pool = MockPool()

    class MockIterWithParent:

      def __init__(self):
        self._parent = MockParent()
        self.closed = False

      def close(self):
        self.closed = True

    it = MockIterWithParent()
    multihost_dataloading._close_iterator(it)
    multihost_dataloading._terminate_iterator_workers(it)

    self.assertTrue(it.closed)
    for w in it._parent._workers:
      self.assertTrue(w.terminated)
      self.assertTrue(w.joined)
    self.assertTrue(it._parent._pool.terminated)


if __name__ == "__main__":
  absltest.main()
