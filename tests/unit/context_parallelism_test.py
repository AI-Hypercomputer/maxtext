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

"""Tests for Attentions."""

import sys
import unittest
import os.path

import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from maxtext.utils import maxtext_utils


class ContextParallelismTest(unittest.TestCase):
  # We are using the same config from AttentionTest to get the same mesh and other config
  # This is a test for context parallelism, so we will not be
  # testing the attention mechanism itself which is done in AttentionTest,
  # but rather how the data is sharded and how the context parallelism is applied.
  # This test is only relevant for TPU devices, so we will skip it for
  # other devices.
  # The test will check if the data is sharded correctly across the devices in the
  # mesh and if the context parallelism is applied correctly.

  # Note: if you are changing these configs, please make sure to change the configs in
  # attention_test.py as well, since we are using the same configs for both
  # tests to get the same mesh and other config
  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_prefill_predict_length": 16,
      "max_target_length": 512,
      "sa_block_q": 128,
      "sa_block_kv": 128,
      "sa_block_kv_compute": 128,
      "sa_block_q_dkv": 128,
      "sa_block_kv_dkv": 128,
      "sa_block_kv_dkv_compute": 128,
      "sa_block_q_dq": 128,
      "sa_block_kv_dq": 128,
  }

  def setUp(self):
    config_cp = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
        ici_context_parallelism=4,  # use context parallelism of 4
        context_parallel_load_balance=False,  # set load_balancing to False such that
        # there's no need for reordering the input/output
    )
    self.cfg_cp = config_cp
    devices_array_cp = maxtext_utils.create_device_mesh(self.cfg_cp)  # for context parallelism
    self.mesh_cp = Mesh(devices_array_cp, self.cfg_cp.mesh_axes)  # for context parallelism

  @pytest.mark.tpu_only
  def test_context_parallelism_sharding(self):
    # Ensure data is sharded following context parallelism axis when enabled
    # Global array: 8x3. Sharded along first dim (8) across 4 devices, it becomes four 2x3 shards.
    num_rows_per_device = 2
    num_cols = 3
    global_data_2d = jnp.arange(4 * num_rows_per_device * num_cols).reshape(4 * num_rows_per_device, num_cols)

    sharding_named_cp = NamedSharding(self.mesh_cp, PartitionSpec("context", None))

    sharded_array_2d_cp = jax.device_put(global_data_2d, sharding_named_cp)
    self.assertEqual(len(sharded_array_2d_cp.global_shards), jax.device_count())
    jax.debug.visualize_array_sharding(sharded_array_2d_cp)
    # Define expected indices for data distributed across the logical mesh devices
    # The second dimension uses slice(None, None, None) as it's replicated and that's what shard.index often shows.
    expected_indices_per_logical_device = [
        (slice(0, num_rows_per_device * 1, None), slice(None, None, None)),  # Shard for logical device 0
        (
            slice(num_rows_per_device * 1, num_rows_per_device * 2, None),
            slice(None, None, None),
        ),  # Shard for logical device 1
        (
            slice(num_rows_per_device * 2, num_rows_per_device * 3, None),
            slice(None, None, None),
        ),  # Shard for logical device 2
        (
            slice(num_rows_per_device * 3, num_rows_per_device * 4, None),
            slice(None, None, None),
        ),  # Shard for logical device 3
    ]
    expected_shard_shape_2d = (num_rows_per_device, num_cols)

    # Create a mapping from the expected device (in logical mesh order) to its expected properties
    # current_test_devices defines the logical order of devices in the mesh
    current_test_devices = self.mesh_cp.devices.flatten()

    expected_map = {
        current_test_devices[i]: {
            "index": expected_indices_per_logical_device[i],
            "shape": expected_shard_shape_2d,
            "data": global_data_2d[expected_indices_per_logical_device[i]],
        }
        for i in range(len(current_test_devices))
    }

    found_devices_in_shards = set()

    for shard in sharded_array_2d_cp.global_shards:
      actual_device = shard.device
      actual_index = shard.index
      actual_shape = shard.data.shape
      actual_data = shard.data

      self.assertIn(
          actual_device,
          expected_map,
          f"Shard found on device {actual_device} which was not in the mesh's expected devices.",
      )

      expected_props = expected_map[actual_device]

      self.assertEqual(
          actual_index,
          expected_props["index"],
          f"Index mismatch for device {actual_device}. Got {actual_index}, expected {expected_props['index']}.",
      )
      self.assertEqual(
          actual_shape,
          expected_props["shape"],
          f"Shape mismatch for device {actual_device}. Got {actual_shape}, expected {expected_props['shape']}.",
      )
      np.testing.assert_array_equal(actual_data, expected_props["data"], f"Data mismatch for device {actual_device}.")

      found_devices_in_shards.add(actual_device)

    # Ensure all expected devices from the mesh were found in the shards
    self.assertEqual(
        found_devices_in_shards, set(current_test_devices), "Not all expected mesh devices were found among the shards."
    )
