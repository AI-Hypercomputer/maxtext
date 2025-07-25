"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=missing-module-docstring, missing-function-docstring
import sys
import unittest
import os.path

import pytest

import numpy as np

import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec

import tensorflow as tf

from MaxText import pyconfig
from MaxText import multihost_dataloading
from MaxText.globals import PKG_DIR, tpu_present


class MultihostDataloadingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    batch_size = 4
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        base_output_directory="gs://max-experiments/",
        dataset_path="gs://maxtext-dataset/",
        enable_checkpointing=False,
    )
    global_data_shape = PartitionSpec(batch_size, config.max_target_length)
    mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), config.mesh_axes)
    # creating 2 batches of data
    global_data = np.arange(np.prod(global_data_shape) * 2).reshape((batch_size * 2, config.max_target_length))

    dataset = tf.data.Dataset.from_tensor_slices(global_data)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    self.multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataset, self.mesh)

  @pytest.mark.tpu_only
  @unittest.skipIf(not tpu_present, "TPU only test")
  def test_batch_sharded_data_pipeline(self):
    first_batch = next(self.multihost_gen)
    sec_batch = next(self.multihost_gen)
    self.assertTrue(not np.array_equal(first_batch, sec_batch, equal_nan=True))


if __name__ == "__main__":
  unittest.main()
