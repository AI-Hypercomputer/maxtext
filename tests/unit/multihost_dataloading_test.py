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

# pylint: disable=missing-module-docstring, missing-function-docstring
import itertools
import json
import os
import pathlib
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.input_pipeline import multihost_dataloading
from tests.utils.test_helpers import get_test_base_output_directory
from tests.utils.test_helpers import get_test_config_path
from tests.utils.test_helpers import get_test_dataset_path
import numpy as np
import pytest


class MockIterator:

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
    base_output_directory = get_test_base_output_directory(
        cloud_path="gs://max-experiments/"
    )
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
    self.mesh = Mesh(
        mesh_utils.create_device_mesh(mesh_shape_1d), config.mesh_axes
    )
    # Create 2 distinct batches and cycle through them infinitely.
    global_data = np.arange(
        batch_size * 2 * config.max_target_length, dtype=np.int32
    ).reshape((batch_size * 2, config.max_target_length))
    data_batches = [global_data[:batch_size], global_data[batch_size:]]
    self.multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        itertools.cycle(data_batches), self.mesh
    )

  @pytest.mark.tpu_only
  def test_batch_sharded_data_pipeline(self):
    first_batch = next(self.multihost_gen)
    sec_batch = next(self.multihost_gen)
    self.assertFalse(np.array_equal(first_batch, sec_batch, equal_nan=True))

  @parameterized.named_parameters(*_get_test_mesh_shapes_named())
  def test_remote_iterator_wrapper_save_state(self, mesh_shape):
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

    preprocessing_fn = lambda dataset: dataset
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
      self.assertEqual(
          len(json_files), 1, f"Expected 1 JSON file, found: {json_files}"
      )
      written_content = json_files[0].read_text()
      self.assertEqual(json.loads(written_content), {"state": 1})

  @parameterized.named_parameters(*_get_test_mesh_shapes_named())
  def test_remote_iterator_wrapper_restore_state(self, mesh_shape):
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

    preprocessing_fn = lambda dataset: dataset
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


if __name__ == "__main__":
  absltest.main()
