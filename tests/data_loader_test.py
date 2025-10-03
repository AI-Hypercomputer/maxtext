# Copyright 2023–2025 Google LLC
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

"""Tests for data_loader.py"""

import unittest
import os.path
import numpy as np

import jax

from unittest.mock import MagicMock
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from MaxText.data_loader import DataLoader
from MaxText import exceptions
from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR


class DataLoaderTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = self.get_test_config(reuse_example_batch=False)
    self.config_reuse_example = self.get_test_config(reuse_example_batch=True)
    self.mesh_shape_1d = (len(jax.devices()),)
    self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
    self.mock_data_iterator = MagicMock()

  def get_test_config(self, reuse_example_batch):
    return pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1,
        run_name="test",
        mesh_axes=["data"],
        logical_axis_rules=[["batch", "data"]],
        data_sharding=["data"],
        enable_checkpointing=False,
        reuse_example_batch=reuse_example_batch,
    )

  def test_load_next_batch_success(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    expected_batch = {"inputs": np.zeros(expected_shape, dtype=int)}
    self.mock_data_iterator.__next__.return_value = expected_batch

    data_loader = DataLoader(self.config, self.mesh, self.mock_data_iterator, None)
    batch = data_loader.load_next_batch()

    self.assertEqual(batch["inputs"].shape, expected_batch["inputs"].shape)
    self.assertTrue((batch["inputs"] == expected_batch["inputs"]).all())
    self.assertEqual(data_loader.last_batch["inputs"].shape, expected_batch["inputs"].shape)
    self.assertTrue((data_loader.last_batch["inputs"] == expected_batch["inputs"]).all())
    self.mock_data_iterator.__next__.assert_called_once()

  def test_load_next_batch_reuse_true(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    expected_batch = {"inputs": np.zeros(expected_shape, dtype=int)}
    self.mock_data_iterator.__next__.return_value = expected_batch

    data_loader = DataLoader(self.config_reuse_example, self.mesh, self.mock_data_iterator, None)

    batch_1 = data_loader.load_next_batch()
    self.assertEqual(batch_1["inputs"].shape, expected_batch["inputs"].shape)
    self.assertTrue((batch_1["inputs"] == expected_batch["inputs"]).all())
    self.assertEqual(data_loader.last_batch["inputs"].shape, expected_batch["inputs"].shape)
    self.assertTrue((data_loader.last_batch["inputs"] == expected_batch["inputs"]).all())
    self.mock_data_iterator.__next__.assert_called_once()  # Called first time

    batch_2 = data_loader.load_next_batch()  # Should reuse batch
    self.assertEqual(batch_2["inputs"].shape, expected_batch["inputs"].shape)
    self.assertTrue((batch_2["inputs"] == expected_batch["inputs"]).all())
    self.assertEqual(data_loader.last_batch["inputs"].shape, expected_batch["inputs"].shape)
    self.assertTrue((data_loader.last_batch["inputs"] == expected_batch["inputs"]).all())
    self.mock_data_iterator.__next__.assert_called_once()  # Still called only once

  def test_load_next_batch_reuse_false(self):
    expected_shape = [jax.device_count(), self.config.max_target_length]
    expected_batch_1 = {"inputs": np.zeros(expected_shape, dtype=int)}
    expected_batch_2 = {"inputs": np.ones(expected_shape, dtype=int)}
    self.mock_data_iterator.__next__.side_effect = [expected_batch_1, expected_batch_2]

    data_loader = DataLoader(self.config, self.mesh, self.mock_data_iterator, None)

    batch_1 = data_loader.load_next_batch()
    self.assertEqual(batch_1["inputs"].shape, expected_batch_1["inputs"].shape)
    self.assertTrue((batch_1["inputs"] == expected_batch_1["inputs"]).all())
    self.assertEqual(data_loader.last_batch["inputs"].shape, expected_batch_1["inputs"].shape)
    self.assertTrue((data_loader.last_batch["inputs"] == expected_batch_1["inputs"]).all())

    batch_2 = data_loader.load_next_batch()  # Should not reuse batch
    self.assertEqual(batch_2["inputs"].shape, expected_batch_2["inputs"].shape)
    self.assertTrue((batch_2["inputs"] == expected_batch_2["inputs"]).all())
    self.assertEqual(data_loader.last_batch["inputs"].shape, expected_batch_2["inputs"].shape)
    self.assertTrue((data_loader.last_batch["inputs"] == expected_batch_2["inputs"]).all())

    self.assertEqual(self.mock_data_iterator.__next__.call_count, 2)

  def test_load_next_batch_throws_exception(self):
    self.mock_data_iterator.__next__.side_effect = StopIteration("generator raised StopIteration")

    data_loader = DataLoader(self.config, self.mesh, self.mock_data_iterator, None)
    with self.assertRaises(exceptions.StopTraining) as e:
      _ = data_loader.load_next_batch()
    self.assertTrue(str(e.exception).startswith("You may have run out of training data."))


if __name__ == "__main__":
  unittest.main()
