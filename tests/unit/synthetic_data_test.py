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

"""Tests for synthetic data sharding."""

import sys
from types import SimpleNamespace
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator, SyntheticDataIterator
from maxtext.utils import sharding as maxtext_sharding
from tests.utils.test_helpers import get_test_config_path


class SyntheticDataShardingTest(parameterized.TestCase):

  @parameterized.product(
      mesh_shape=[(1, 1), (2, 1), (2, 2)],
      shard_mode=["auto", "explicit"],
  )
  def test_synthetic_data_sharding(self, mesh_shape, shard_mode):
    devices = jax.devices()
    num_devices = len(devices)

    target_devices = mesh_shape[0] * mesh_shape[1]

    if num_devices < target_devices:
      self.skipTest(f"Not enough devices. Required: {target_devices}, Available: {num_devices}")

    mesh_devices = devices[:target_devices]
    devices_array = mesh_utils.create_device_mesh(mesh_shape, mesh_devices)
    mesh = Mesh(devices_array, ["data", "fsdp"])

    # Initialize config
    init_kwargs = {
        "per_device_batch_size": 2.0,
        "num_target_devices": len(mesh_devices),
        "run_name": "test",
        "enable_checkpointing": False,
        "dataset_type": "synthetic",
        "model_name": "llama3.1-8b",
        "max_target_length": 16,
        "shard_mode": shard_mode,
    }
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **init_kwargs,
    )

    iterator = SyntheticDataIterator(config, mesh)
    batch = next(iterator)

    self.assertIn("inputs", batch)
    inputs = batch["inputs"]

    self.assertEqual(inputs.shape, (config.global_batch_size_to_load, config.max_target_length))

    # Expected sharding: mapping of logical axes to physical mesh
    expected_sharding = maxtext_sharding.get_input_data_sharding(config, mesh)
    self.assertEqual(inputs.sharding, expected_sharding)


class PlaceHolderDataTest(absltest.TestCase):
  """Tests placeholder batch metadata for non-loading hosts."""

  def test_block_diffusion_placeholder_has_zero_loss_masks(self):
    config = SimpleNamespace(
        global_batch_size_to_load=jax.process_count(),
        max_target_length=8,
        training_objective="block_diffusion",
    )

    batch = next(PlaceHolderDataIterator.get_place_holder_synthetic_data(config))

    self.assertIn("completion_mask", batch)
    self.assertIn("corruption_mask", batch)
    self.assertIn("targets_loss_mask", batch)
    self.assertFalse(batch["completion_mask"].any())
    self.assertFalse(batch["corruption_mask"].any())
    self.assertFalse(batch["targets_loss_mask"].any())


if __name__ == "__main__":
  absltest.main()
