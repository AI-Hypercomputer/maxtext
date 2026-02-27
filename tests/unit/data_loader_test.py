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

import numpy as np
import pytest

import jax

from unittest.mock import MagicMock
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.common.data_loader import DataLoader, RampUpDataLoader
from maxtext.utils import exceptions
from maxtext.utils.maxtext_utils import create_device_mesh
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.utils.rampup_batch import RampupBatchManager
from tests.utils.test_helpers import get_test_config_path


class DataLoaderTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = self.get_test_config(reuse_example_batch=False, per_device_batch_size=1)
    self.config_reuse_example = self.get_test_config(reuse_example_batch=True, per_device_batch_size=1)
    self.config_rampup = self.get_test_config(
        reuse_example_batch=False,
        per_device_batch_size=4.0,  # This is the 'end' batch size
        enable_rampup_batch_size=True,
        per_device_batch_size_start=1.0,
        per_device_batch_size_increment=1.0,
        # global_rampup_samples: (rampup increment number) * (Samples for initial 5 steps)
        global_rampup_samples=3 * (1 * jax.device_count() * 5),
    )
    self.mesh = Mesh(create_device_mesh(self.config), self.config.mesh_axes)
    self.mock_data_iterator = MagicMock()

  def get_test_config(self, reuse_example_batch, **kwargs):
    """Generate config for tests"""
    args = {
        "run_name": "test",
        "enable_checkpointing": False,
        "reuse_example_batch": reuse_example_batch,
    }
    args.update(kwargs)

    # In decoupled mode, adapt mesh/ICI parallelism so that the
    # product of ICI parallelism matches the available devices for
    # this test only.
    if is_decoupled():
      args.setdefault("mesh_axes", ["data"])
      args.setdefault("ici_data_parallelism", -1)

    return pyconfig.initialize(
        [None, get_test_config_path()],
        **args,
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

  @pytest.mark.external_serving
  def test_rampup_data_loader(self):
    """Tests that RampUpLoader correctly slices and increment."""
    # Mock iterator returns a FULL batch (size 4)
    full_batch_size = int(self.config_rampup.per_device_batch_size * self.config_rampup.num_target_devices)
    full_shape = [full_batch_size, self.config_rampup.max_target_length]
    full_batch = {"inputs": np.ones(full_shape, dtype=int)}
    self.mock_data_iterator.__next__.return_value = full_batch

    # Create the RampUpDataLoader
    rampup_manager = RampupBatchManager(self.config_rampup, -1)
    data_loader = RampUpDataLoader(self.config_rampup, self.mesh, self.mock_data_iterator, None)

    # Expected batch sizes based on test config.
    # The end global batch size is self.num_devices * per_device_batch_size
    # The rampup of per_device_batch_size should be:
    #   5 steps of size 1, 3 steps of size 2, 2 steps of size 3, then size 4.
    multipliers = [1] * 5 + [2] * 3 + [3] * 2 + [4] * 2
    expected_batch_sizes = [m * self.config_rampup.num_target_devices for m in multipliers]
    for i, expected_size in enumerate(expected_batch_sizes):
      batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
      expected_shape = (expected_size, self.config_rampup.max_target_length)
      self.assertEqual(
          batch["inputs"].shape,
          expected_shape,
          f"Mismatch at step {i+1}: expected {expected_shape}, got {batch['inputs'].shape}",
      )
      self.assertTrue((batch["inputs"] == 1).all())

  def test_rampup_data_loader_from_checkpointing(self):
    """Tests that RampUpLoader correctly slices and increment resumed from checkpointing."""
    # Mock iterator returns a FULL batch (size 4)
    full_batch_size = int(self.config_rampup.per_device_batch_size * self.config_rampup.num_target_devices)
    full_shape = [full_batch_size, self.config_rampup.max_target_length]
    full_batch = {"inputs": np.ones(full_shape, dtype=int)}
    self.mock_data_iterator.__next__.return_value = full_batch
    # We assume rampup batch size resuming from step 5
    checkpoint_step = 5
    rampup_manager = RampupBatchManager(self.config_rampup, checkpoint_step)

    # Create the RampUpDataLoader
    data_loader = RampUpDataLoader(self.config_rampup, self.mesh, self.mock_data_iterator, None)

    # Expected batch sizes based on test config.
    # The end global batch size is self.num_devices * per_device_batch_size.
    # In decoupled mode, derive the schedule from a fresh RampupBatchManager
    # so it matches the actual global batch sizes on the host.
    if is_decoupled():
      tmp_manager = RampupBatchManager(self.config_rampup, checkpoint_step)
      expected_batch_sizes = []
      # Collect sizes for the ramp-up phase.
      while True:
        expected_batch_sizes.append(tmp_manager.global_batch_size_current)
        rampup_active = tmp_manager.update()
        if not rampup_active:
          break
      # Add a couple of post-ramp-up steps at the final size, mirroring
      # the original test's intent.
      for _ in range(2):
        expected_batch_sizes.append(tmp_manager.global_batch_size_current)
        tmp_manager.update()
    else:
      # The end global batch size is self.num_devices * per_device_batch_size
      # The rampup of per_device_batch_size should be:
      #   3 steps of size 2, 2 steps of size 3, then size 4.
      multipliers = [2] * 3 + [3] * 2 + [4] * 2
      expected_batch_sizes = [m * self.config_rampup.num_target_devices for m in multipliers]
    for i, expected_size in enumerate(expected_batch_sizes):
      batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
      expected_shape = (expected_size, self.config_rampup.max_target_length)
      self.assertEqual(
          batch["inputs"].shape,
          expected_shape,
          f"Mismatch at step {i+1}: expected {expected_shape}, got {batch['inputs'].shape}",
      )
      self.assertTrue((batch["inputs"] == 1).all())


if __name__ == "__main__":
  unittest.main()
