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

import os
import subprocess
import sys
import types
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from maxtext.configs import pyconfig
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.input_pipeline.synthetic_data_processing import PlaceHolderDataIterator
from maxtext.input_pipeline.synthetic_data_processing import SyntheticDataIterator
from maxtext.utils import elastic_utils
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

  def test_constant_padded_masks_train_tail_and_uses_unpadded_eval_batch(self):
    live_devices = list(jax.devices())
    num_live = len(live_devices)
    if num_live < 2:
      if not sys.executable:
        self.skipTest("Needs >= 2 devices.")
      self.assertNotEqual(os.environ.get("_MAXTEXT_CONSTANT_PADDED_SUBPROC"), "1")
      env = {
          **os.environ,
          "_MAXTEXT_CONSTANT_PADDED_SUBPROC": "1",
          "JAX_PLATFORMS": "cpu",
          "XLA_FLAGS": f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=2".strip(),
          "PYTHONPATH": os.pathsep.join(filter(None, [*sys.path, os.environ.get("PYTHONPATH", "")])),
      }
      cmd = [sys.executable, os.path.abspath(__file__), f"{type(self).__name__}.{self._testMethodName}"]
      res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
      self.assertEqual(res.returncode, 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
      return

    all_devices = live_devices + [mock.MagicMock(slice_index=99)]

    with (
        mock.patch("jax.devices", return_value=all_devices),
        mock.patch.object(elastic_utils, "ensure_elastic_manager_initialized"),
        mock.patch.object(elastic_utils, "live_devices", return_value=live_devices),
    ):
      config = pyconfig.initialize(
          [sys.argv[0], get_test_config_path()],
          run_name="test_synthetic_constant_padded",
          enable_checkpointing=False,
          enable_single_controller=True,
          elastic_enabled=True,
          elastic_global_batch_size_mode="constant_padded",
          per_device_batch_size=1.0,
          eval_per_device_batch_size=1.0,
          eval_interval=5,
          dataset_type="synthetic",
          packing=False,
          mesh_axes=["data"],
          logical_axis_rules=[["batch", "data"]],
          data_sharding=["data"],
          max_target_length=8,
      )

    mesh = Mesh(mesh_utils.create_device_mesh((num_live,), live_devices), config.mesh_axes)
    train_iter, eval_iter = input_pipeline_interface.create_data_iterator(config, mesh)
    self.assertIsNotNone(eval_iter)

    train_batch = next(train_iter)
    self.assertEqual(train_batch["inputs"].shape, (config.global_batch_size_to_load, 8))
    in_one = jnp.all(train_batch["inputs_segmentation"] == 1, axis=-1)
    in_zero = jnp.all(train_batch["inputs_segmentation"] == 0, axis=-1)
    self.assertEqual(int(jnp.sum(in_one)), config.logical_global_batch_size_to_load)
    self.assertEqual(int(jnp.sum(in_zero)), config.global_batch_size_to_load - config.logical_global_batch_size_to_load)
    self.assertGreater(int(jnp.sum(in_zero)), 0)
    self.assertTrue(bool(jnp.array_equal(train_batch["inputs_segmentation"], train_batch["targets_segmentation"])))

    eval_batch = next(eval_iter)
    self.assertEqual(eval_batch["inputs"].shape, (config.global_batch_size_to_load_eval, 8))
    self.assertTrue(bool(jnp.all(eval_batch["inputs_segmentation"] == 1)))
    self.assertTrue(bool(jnp.all(eval_batch["targets_segmentation"] == 1)))


class SyntheticDataPaddedRowsTest(parameterized.TestCase):
  """The logical rows of a padded synthetic batch match the rows of the unpadded full-job batch."""

  def _data(self, batch_size, logical_batch_size=None, packing=True, num_replicas=1, replica_index=0):
    """The (tokens, positions, segmentation) rows a `SyntheticDataIterator` builds for one replica."""
    config = types.SimpleNamespace(
        global_batch_size_to_load=batch_size,
        global_batch_size_to_load_eval=batch_size,
        logical_global_batch_size_to_load=logical_batch_size,
        logical_axis_rules_for_eval=None,
        max_target_length=16,
        max_segments_per_seq=4,
        vocab_size=128,
        packing=packing,
        num_data_replicas_per_process=num_replicas,
        data_replica_index=replica_index,
        enable_diloco=False,
    )
    batch = SyntheticDataIterator.raw_generate_synthetic_data(config, is_training=True)
    return batch["inputs"], batch["inputs_position"], batch["inputs_segmentation"]

  @parameterized.product(
      packing=[True, False], replica=[(1, 0), (2, 0), (2, 1)], prng_impl=["unsafe_rbg", "threefry2x32"]
  )
  def test_logical_rows_match_unpadded_batch(self, packing, replica, prng_impl):
    # `base.yml` defaults to `prng_impl: unsafe_rbg`; on TPU a smaller draw is not a row-prefix of a larger one.
    self.addCleanup(jax.config.update, "jax_default_prng_impl", jax.config.jax_default_prng_impl)
    jax.config.update("jax_default_prng_impl", prng_impl)
    num_replicas, replica_index = replica
    full = self._data(4, packing=packing, num_replicas=num_replicas, replica_index=replica_index)
    padded = self._data(6, 4, packing=packing, num_replicas=num_replicas, replica_index=replica_index)
    for full_part, padded_part in zip(full, padded, strict=True):
      self.assertTrue(bool(jnp.array_equal(padded_part[:4], full_part)))
    tokens, _, segmentation = padded
    self.assertTrue(bool(jnp.all(segmentation[4:] == 0)))
    self.assertTrue(bool(jnp.all(tokens[4:] == 0)))
    if packing:
      self.assertGreater(int(jnp.max(full[2])), 1)  # Packed: several segments per row.

  @parameterized.parameters((1, 4), (2, 8))
  def test_packed_segment_ids_are_generated_for_the_logical_rows_only(self, num_replicas, logical_rows):
    # On CPU a smaller draw is a row-prefix of a larger one for both PRNGs, so check the generated shape directly.
    with mock.patch.object(
        synthetic_data_processing,
        "_make_packed_segment_ids",
        wraps=synthetic_data_processing._make_packed_segment_ids,  # pylint: disable=protected-access
    ) as make_segment_ids:
      self._data(6, 4, num_replicas=num_replicas)
    make_segment_ids.assert_called_once_with(logical_rows, 16, 4)


class PlaceHolderDataIteratorTest(absltest.TestCase):
  """Tests for the placeholder iterator used by the non loading hosts."""

  def _config(self):
    return types.SimpleNamespace(
        global_batch_size_to_load=16,
        global_batch_size_to_load_eval=8,
        max_target_length=4,
    )

  def test_eval_placeholder_uses_eval_batch_size(self):
    data = next(PlaceHolderDataIterator.get_place_holder_synthetic_data(self._config(), is_training=False))
    self.assertEqual(data["inputs"].shape, (8 // jax.process_count(), 4))

  def test_train_placeholder_uses_train_batch_size(self):
    data = next(PlaceHolderDataIterator.get_place_holder_synthetic_data(self._config()))
    self.assertEqual(data["inputs"].shape, (16 // jax.process_count(), 4))


if __name__ == "__main__":
  absltest.main()
