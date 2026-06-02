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
import sys
import tempfile

from absl.testing import absltest
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


class MultihostDataloadingTest(absltest.TestCase):

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

  def test_remote_iterator_wrapper_save_restore_state(self):
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

    num_devices = len(jax.devices())
    if num_devices >= 4:
      mesh_shape = (2, num_devices // 2)
    elif num_devices >= 2:
      mesh_shape = (2, 1)
    else:
      mesh_shape = (1, 1)

    mesh_size = mesh_shape[0] * mesh_shape[1]
    devices = mesh_utils.create_device_mesh(
        mesh_shape, jax.devices()[:mesh_size]
    )
    mesh = Mesh(devices, ("x", "y"))

    def get_ds_fn(dataloading_host_index, dataloading_host_count):
      del dataloading_host_index, dataloading_host_count
      return MockDataloader(mesh_size)

    preprocessing_fn = lambda dataset: dataset

    with tempfile.TemporaryDirectory() as tmpdir:
      global_shape = (mesh_size, 1)
      wrapper = multihost_dataloading.RemoteIteratorWrapper(
          get_ds_fn=get_ds_fn,
          preprocessing_fn=preprocessing_fn,
          global_mesh=mesh,
          global_shape=global_shape,
          checkpoint_path=tmpdir,
          elastic=False,
      )

      # 1. Advance the iterator state by calling next()
      val = next(wrapper)
      # The value retrieved is a JAX array sharded across cpu/tpu sharding.
      # Let's check the first element of the addressable data.
      self.assertEqual(val.addressable_data(0)[0], 1)

      # 2. Save state at step 5
      wrapper.save_state(step=5)

      # 3. Verify that the state is written to the filesystem and has value 1
      expected_file = os.path.join(
          tmpdir,
          "5",
          "iter",
          f"process_{jax.process_index()}-of-{jax.process_count()}.json",
      )
      self.assertTrue(os.path.exists(expected_file))

      with open(expected_file, "r") as f:
        saved_state = json.load(f)
      self.assertEqual(saved_state["state"], 1)

      # 4. Advance the state again
      val = next(wrapper)
      self.assertEqual(val.addressable_data(0)[0], 2)

      # 5. Restore state from step 5
      wrapper.restore_state(step=5)

      # 6. Advance the state again. It should have been restored to 1, so
      # calling next() should return 2!
      val = next(wrapper)
      self.assertEqual(val.addressable_data(0)[0], 2)


if __name__ == "__main__":
  absltest.main()
