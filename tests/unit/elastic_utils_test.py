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

"""Unit tests for Elastic Training utility functions."""

import unittest
from unittest.mock import create_autospec, Mock


from maxtext.utils import elastic_utils
from maxtext.utils import gcs_utils
import pathwaysutils
from pathwaysutils.elastic.manager import ScaleUpSignalError


class FakeDevice:
  """Fake Device object."""

  def __init__(self, slice_index=0, process_index=0, task_id=0):
    self.slice_index = slice_index
    self.process_index = process_index
    self.task_id = task_id


class FakeConfig:
  """Fake configuration object."""

  def __init__(self):
    self.elastic_enabled = True
    self.checkpoint_dir = "gs://test_bucket/checkpoints"
    self.elastic_max_retries = 3
    self.elastic_timeout_seconds = 100
    self.global_batch_size_to_load = 64
    self.per_device_batch_size = 4
    self.elastic_min_slice_count = 1


class ElasticUtilsTest(unittest.TestCase):
  """Unit tests for Elastic Training utility functions."""

  def setUp(self):
    super().setUp()
    # Save original dependencies
    self.original_pathwaysutils = elastic_utils.pathwaysutils
    self.original_jax = elastic_utils.jax
    self.original_gcs_utils = elastic_utils.gcs_utils
    self.original_max_logging = elastic_utils.max_logging
    self.original_manager_class = pathwaysutils.elastic.manager.Manager
    self.original_scale_up_signal_error = getattr(pathwaysutils.elastic.manager, "ScaleUpSignalError", None)

    # Initialize fakes as mocks
    self.fake_gcs_utils = create_autospec(gcs_utils)
    self.fake_gcs_utils.add_trailing_slash.side_effect = gcs_utils.add_trailing_slash
    self.fake_pathwaysutils = create_autospec(pathwaysutils)
    self.fake_logging = create_autospec(self.original_max_logging)
    self.fake_jax = create_autospec(self.original_jax)
    self.fake_manager = create_autospec(self.original_manager_class, instance=True)
    self.fake_manager.new_slice_event = Mock()

    # Configure default behaviors if needed
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    self.fake_jax.process_index.return_value = 0

    # Inject fakes into elastic_utils namespace
    elastic_utils.pathwaysutils = self.fake_pathwaysutils
    elastic_utils.jax = self.fake_jax
    elastic_utils.gcs_utils = self.fake_gcs_utils
    elastic_utils.max_logging = self.fake_logging

    # Hook up pathwaysutils.elastic.manager.Manager to return our fake_manager
    pathwaysutils.elastic.manager.Manager = lambda *args, **kwargs: self.fake_manager
    pathwaysutils.elastic.manager.ScaleUpSignalError = ScaleUpSignalError

    # Reset global state for testing is no longer needed

  def tearDown(self):
    # Restore original dependencies
    elastic_utils.pathwaysutils = self.original_pathwaysutils
    elastic_utils.jax = self.original_jax
    elastic_utils.gcs_utils = self.original_gcs_utils
    elastic_utils.max_logging = self.original_max_logging
    pathwaysutils.elastic.manager.Manager = self.original_manager_class
    pathwaysutils.elastic.manager.ScaleUpSignalError = self.original_scale_up_signal_error
    elastic_utils.elastic_manager = None
    super().tearDown()

  def test_elastic_enabled(self):
    config = FakeConfig()
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    config.elastic_enabled = True
    self.assertTrue(elastic_utils.elastic_enabled(config))

    config.elastic_enabled = False
    self.assertFalse(elastic_utils.elastic_enabled(config))

    config.elastic_enabled = True
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = False
    self.assertFalse(elastic_utils.elastic_enabled(config))

  def test_clean_up_checkpoints_no_checkpoints(self):
    self.fake_gcs_utils.gcs_list_directories.return_value = []
    elastic_utils.clean_up_checkpoints("gs://test_bucket/checkpoints")
    self.fake_gcs_utils.gcs_delete_directory.assert_not_called()

  def test_clean_up_checkpoints_incomplete(self):
    """Tests clean_up_checkpoints when the latest checkpoint is incomplete."""
    checkpoint_dir = "gs://test_bucket/checkpoints"
    self.fake_gcs_utils.gcs_list_directories.return_value = ["1", "2", "10"]
    self.fake_gcs_utils.gcs_glob_pattern.return_value = []
    # No commit_success for "10"
    elastic_utils.clean_up_checkpoints(checkpoint_dir)
    self.fake_gcs_utils.gcs_delete_directory.assert_called_once_with(f"{checkpoint_dir}/10/")

  def test_clean_up_checkpoints_complete(self):
    """Tests clean_up_checkpoints when the latest checkpoint is complete."""
    checkpoint_dir = "gs://test_bucket/checkpoints"
    self.fake_gcs_utils.gcs_list_directories.return_value = ["1", "2", "10"]
    self.fake_gcs_utils.gcs_glob_pattern.return_value = [f"{checkpoint_dir}/10/commit_success_0"]
    elastic_utils.clean_up_checkpoints(checkpoint_dir)
    self.fake_gcs_utils.gcs_delete_directory.assert_not_called()

  def test_live_devices_no_pathways(self):
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = False
    device0 = FakeDevice(slice_index=0)
    self.fake_jax.devices.return_value = [device0]

    config = FakeConfig()
    devices = elastic_utils.live_devices(config)
    self.assertEqual(devices, [device0])

  def test_live_devices_pathways(self):
    """Tests live_devices when pathways is used."""
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    device0 = FakeDevice(slice_index=0)
    device1 = FakeDevice(slice_index=1)
    self.fake_jax.devices.return_value = [device0, device1]
    self.fake_manager.active_slice_indices = {0}

    config = FakeConfig()
    devices = elastic_utils.live_devices(config)
    self.assertEqual(devices, [device0])

  def test_elastic_retry_disabled(self):
    """Tests elastic_retry when disabled but pathways is used."""
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    config = FakeConfig()
    config.elastic_enabled = False
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled: False, pathways"
        " backend used: True"
    )
    with self.assertRaisesRegex(ValueError, msg):
      elastic_utils.elastic_retry(config)

  def test_elastic_retry_no_pathways(self):
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = False
    config = FakeConfig()
    config.elastic_enabled = True
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled: True, pathways"
        " backend used: False"
    )
    with self.assertRaisesRegex(ValueError, msg):
      elastic_utils.elastic_retry(config)

  def test_chain_callbacks(self):
    # Test with no functions
    chained_fn_empty = elastic_utils.chain_callbacks()
    chained_fn_empty()  # Should not fail

    # Test with multiple functions
    call_order = []

    def fn1():
      call_order.append(1)

    def fn2():
      call_order.append(2)

    chained_fn = elastic_utils.chain_callbacks(fn1, fn2)
    chained_fn()
    self.assertEqual(call_order, [1, 2])

  def test_get_local_batch_size_elastic(self):
    config = FakeConfig()
    config.elastic_enabled = True
    config.per_device_batch_size = 4

    device0 = FakeDevice(slice_index=0, process_index=0)
    self.fake_jax.devices.return_value = [device0]
    self.fake_manager.all_slice_indices = {0}
    self.fake_manager.active_slice_indices = {0}

    batch_size = elastic_utils.get_local_batch_size(config)
    self.assertEqual(batch_size, 4)

  def test_get_local_batch_size_non_elastic(self):
    config = FakeConfig()
    config.elastic_enabled = False
    config.global_batch_size_to_load = 64
    self.fake_jax.process_count.return_value = 2
    # Provide 8 devices to yield devices_per_host = 8, so 4 * 8 = 32
    self.fake_jax.devices.return_value = [FakeDevice(slice_index=0, process_index=0, task_id=0) for _ in range(8)]
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = False

    batch_size = elastic_utils.get_local_batch_size(config)
    self.assertEqual(batch_size, 32)

  def test_live_slice_indices(self):
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = False
    device0 = FakeDevice(slice_index=0)
    device1 = FakeDevice(slice_index=1)
    self.fake_jax.devices.return_value = [device0, device1]

    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0, 1}
    indices = elastic_utils.live_slice_indices(config)
    self.assertEqual(indices, {0, 1})

  def test_get_devices_per_host(self):
    device0 = FakeDevice(slice_index=0, process_index=0, task_id=0)
    device1 = FakeDevice(slice_index=0, process_index=0, task_id=0)
    device2 = FakeDevice(slice_index=0, process_index=1, task_id=1)
    device3 = FakeDevice(slice_index=0, process_index=1, task_id=1)
    self.fake_jax.devices.return_value = [device0, device1, device2, device3]
    self.fake_manager.all_slice_indices = {0}
    self.fake_manager.active_slice_indices = {0}

    config = FakeConfig()
    count = elastic_utils.get_devices_per_host(config)
    self.assertEqual(count, 2)

  def test_maybe_elastic_scale_up(self):
    config = FakeConfig()
    config.elastic_enabled = True

    class FakeCheckpointManager:

      def __init__(self):
        self.wait_called = False

      def wait_until_finished(self):
        self.wait_called = True

    cm = FakeCheckpointManager()

    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.new_slice_event.is_set.return_value = True

    with self.assertRaises(ScaleUpSignalError):
      elastic_utils.maybe_elastic_scale_up(config, cm)

    self.assertTrue(cm.wait_called)

  def test_elastic_retry_default_min_slices(self):
    """Tests that elastic_retry passes None when elastic_min_slice_count is -1."""
    config = FakeConfig()
    config.elastic_enabled = True
    config.elastic_min_slice_count = -1

    elastic_utils.elastic_manager = self.fake_manager

    elastic_utils.elastic_retry(config)

    self.fake_manager.elastic_retry.assert_called_once()
    kwargs = self.fake_manager.elastic_retry.call_args.kwargs
    self.assertIsNone(kwargs["minimum_slice_count"])

  def test_ensure_elastic_manager_initialized_readonly_config(self):
    """Tests that ensure_elastic_manager_initialized works with read-only config."""

    class ReadOnlyConfig:
      elastic_manager = None

      def __init__(self):
        object.__setattr__(self, "elastic_enabled", True)

      def __setattr__(self, name, value):
        raise ValueError("Configuration is read-only")

    config = ReadOnlyConfig()
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True

    # Should not raise ValueError
    elastic_utils.ensure_elastic_manager_initialized(config)
    self.assertEqual(elastic_utils.elastic_manager, self.fake_manager)


if __name__ == "__main__":
  unittest.main()
