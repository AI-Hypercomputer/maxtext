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
from unittest.mock import Mock, create_autospec
from absl.testing import parameterized
from maxtext.common import checkpointing
from maxtext.utils import elastic_utils
from maxtext.utils import gcs_utils
import pathwaysutils
from pathwaysutils.elastic.manager import ScaleUpSignalError


class MockJaxRuntimeError(Exception):
  """Fake JAX Runtime Error class for unit tests."""


class FakeDevice:
  """Fake Device object."""

  def __init__(self, slice_index=0, process_index=0, task_id=0, device_id=0):
    self.slice_index = slice_index
    self.process_index = process_index
    self.task_id = task_id
    self.id = device_id
    self.platform = "tpu"
    self.device_kind = "TPU"
    self.client = Mock()


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


class ElasticUtilsTest(parameterized.TestCase):
  """Unit tests for Elastic Training utility functions."""

  def setUp(self):
    """Set up the test environment."""
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
    self.fake_manager.available_inactive_slices = set()    # Configure default behaviors if needed
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    self.fake_jax.process_index.return_value = 0
    self.fake_manager.slice_to_devices = {0: [FakeDevice(slice_index=0)]}
    self.fake_manager.active_slice_indices = {0}
    self.original_wait_for_slices = elastic_utils.elastic.wait_for_slices
    self.original_get_active_slice_indices = elastic_utils.elastic.get_active_slice_indices
    elastic_utils.elastic.wait_for_slices = Mock(return_value={0})
    elastic_utils.elastic.get_active_slice_indices = Mock(return_value={0})

    # Inject fakes into elastic_utils namespace
    elastic_utils.pathwaysutils = self.fake_pathwaysutils
    elastic_utils.jax = self.fake_jax
    self.fake_jax.errors.JaxRuntimeError = MockJaxRuntimeError
    elastic_utils.gcs_utils = self.fake_gcs_utils
    elastic_utils.max_logging = self.fake_logging

    # Hook up pathwaysutils.elastic.manager.Manager to return our fake_manager
    pathwaysutils.elastic.manager.Manager = lambda *args, **kwargs: self.fake_manager  # pyrefly: ignore[bad-assignment]
    pathwaysutils.elastic.manager.ScaleUpSignalError = ScaleUpSignalError

    # Reset global state for testing is no longer needed

  def tearDown(self):
    """Restore original dependencies"""
    elastic_utils.pathwaysutils = self.original_pathwaysutils
    elastic_utils.jax = self.original_jax
    elastic_utils.gcs_utils = self.original_gcs_utils
    elastic_utils.max_logging = self.original_max_logging
    elastic_utils.elastic.wait_for_slices = self.original_wait_for_slices
    elastic_utils.elastic.get_active_slice_indices = self.original_get_active_slice_indices
    pathwaysutils.elastic.manager.Manager = self.original_manager_class
    pathwaysutils.elastic.manager.ScaleUpSignalError = (  # pyrefly: ignore[bad-assignment]
        self.original_scale_up_signal_error
    )
    elastic_utils.elastic_manager = None
    elastic_utils.pending_reinit_recorder = None
    elastic_utils.pending_elastic_event_type = None
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

  def test_live_devices_disabled(self):
    """Tests live_devices when pathways is used but elastic is disabled."""
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    device0 = FakeDevice(slice_index=0)
    self.fake_jax.devices.return_value = [device0]

    config = FakeConfig()
    config.elastic_enabled = False
    devices = elastic_utils.live_devices(config)
    self.assertEqual(devices, [device0])
    self.assertIsNone(elastic_utils.elastic_manager)

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

  def _base_mtc_keys(self, **overrides):
    keys = {
        "elastic_enabled": True,
        "mtc_data_parallelism": 1,
        "num_slices": 2,
    }
    keys.update(overrides)
    return keys

  def test_single_controller_mtc_init_kwargs_uses_active_elastic_devices(self):
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    device0 = FakeDevice(slice_index=0)
    device1 = FakeDevice(slice_index=1)
    self.fake_jax.devices.return_value = [device0, device1]
    self.fake_manager.active_slice_indices = {0}

    kwargs = elastic_utils.single_controller_mtc_init_kwargs(self._base_mtc_keys(mtc_data_parallelism=0))

    self.assertEqual(kwargs["devices"], (device0,))
    self.assertEqual(kwargs["num_slices"], 1)
    self.assertEqual(kwargs["data_parallelism"], 1)

  def test_single_controller_mtc_init_kwargs_raises_if_empty(self):
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    device0 = FakeDevice(slice_index=0)
    self.fake_jax.devices.return_value = [device0]
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {1}

    with self.assertRaisesRegex(ValueError, "Elastic single-controller MTC initialization found no active devices."):
      elastic_utils.single_controller_mtc_init_kwargs(self._base_mtc_keys())

  def test_single_controller_mtc_init_kwargs_non_elastic(self):
    kwargs = elastic_utils.single_controller_mtc_init_kwargs(
        self._base_mtc_keys(elastic_enabled=False, mtc_data_parallelism=3, num_slices=4)
    )

    self.assertEqual(kwargs, {"data_parallelism": 3, "num_slices": 4})
    self.assertIsNone(elastic_utils.elastic_manager)

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

      def wait(self):
        self.wait_called = True

    cm = FakeCheckpointManager()

    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.available_inactive_slices = {1}

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

  def test_elastic_retry_pre_callback_none_by_default(self):
    """pre_callback executes wait_for_devices_placed when pre_callback_fn is not supplied."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager

    with unittest.mock.patch.object(elastic_utils, "wait_for_devices_placed") as mock_wait:
      elastic_utils.elastic_retry(config)
      kwargs = self.fake_manager.elastic_retry.call_args.kwargs
      self.assertTrue(callable(kwargs["pre_callback"]))
      kwargs["pre_callback"]()
      mock_wait.assert_called_once_with(config)

  def test_elastic_retry_pre_callback_forwarded(self):
    """pre_callback_fn must be invoked by effective pre_callback along with wait_for_devices_placed."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager

    fake_pre_callback = Mock()
    with unittest.mock.patch.object(elastic_utils, "wait_for_devices_placed") as mock_wait:
      elastic_utils.elastic_retry(config, pre_callback_fn=fake_pre_callback)
      kwargs = self.fake_manager.elastic_retry.call_args.kwargs
      self.assertTrue(callable(kwargs["pre_callback"]))
      kwargs["pre_callback"]()
      mock_wait.assert_called_once_with(config)
      fake_pre_callback.assert_called_once()

  def test_record_elastic_event_start(self):
    """Tests recording an elastic slice down start."""
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.available_inactive_slices = set()
    fake_recorder = Mock()
    config = FakeConfig()

    elastic_utils.record_elastic_event_start(fake_recorder, config)

    fake_recorder.record_elastic_wait_start_time.assert_called_once_with(
        event_type="elastic_slice_down"
    )
    self.assertEqual(elastic_utils.pending_elastic_event_type, "elastic_slice_down")

  def test_record_elastic_event_start_scale_up(self):
    """Tests recording an elastic slice scale up start."""
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.available_inactive_slices = {1}
    fake_recorder = Mock()
    config = FakeConfig()

    elastic_utils.record_elastic_event_start(fake_recorder, config)

    fake_recorder.record_elastic_wait_start_time.assert_called_once_with(
        event_type="elastic_scale_up"
    )

  def test_record_elastic_wait_end_and_reinit_start_noop_on_first_attempt(self):
    """Tests recording elastic event end and elastic reinit start."""
    elastic_utils.pending_elastic_event_type = None
    fake_recorder = Mock()

    elastic_utils.record_elastic_wait_end_and_reinit_start(fake_recorder)

    fake_recorder.record_elastic_wait_end_time.assert_not_called()
    fake_recorder.record_elastic_reinit_start_time.assert_not_called()
    self.assertIsNone(elastic_utils.pending_reinit_recorder)

  def test_record_elastic_wait_end_and_reinit_start(self):
    """Test recording end of slice down and start of reinit."""
    elastic_utils.pending_elastic_event_type = "elastic_slice_down"  # pyrefly: ignore[bad-assignment]
    fake_recorder = Mock()

    elastic_utils.record_elastic_wait_end_and_reinit_start(fake_recorder)

    fake_recorder.record_elastic_wait_end_time.assert_called_once_with(
        event_type="elastic_slice_down"
    )
    fake_recorder.record_elastic_reinit_start_time.assert_called_once_with()
    self.assertIs(elastic_utils.pending_reinit_recorder, fake_recorder)
    self.assertIsNone(elastic_utils.pending_elastic_event_type)

  def test_record_elastic_reinit_end(self):
    """Tests recording end of elastic reinit."""
    fake_recorder = Mock()
    elastic_utils.pending_reinit_recorder = fake_recorder

    elastic_utils.record_elastic_reinit_end()

    fake_recorder.record_elastic_reinit_end_time.assert_called_once_with()
    self.assertIsNone(elastic_utils.pending_reinit_recorder)

  def test_record_elastic_reinit_end_on_cold_start(self):
    """Tests recording end of elastic reinit on cold start."""
    elastic_utils.pending_reinit_recorder = None

    elastic_utils.record_elastic_reinit_end()

  def test_record_slice_state_calls_recorder(self):
    """Tests that record_slice_state computes and forwards the right slice counts."""
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    elastic_utils.elastic.get_active_slice_indices = Mock(return_value={0, 1})
    original_get_slice_to_devices = elastic_utils.elastic.get_slice_to_devices
    elastic_utils.elastic.get_slice_to_devices = Mock(
        return_value={0: [FakeDevice(slice_index=0)], 1: [FakeDevice(slice_index=1)]}
    )
    fake_recorder = Mock()

    try:
      elastic_utils.record_slice_state(fake_recorder)
    finally:
      elastic_utils.elastic.get_slice_to_devices = original_get_slice_to_devices

    fake_recorder.record_elastic_slice_counts.assert_called_once_with(
        available_slices=2, active_slices=1, total_slices=2
    )

  def test_record_slice_state_active_slices_override(self):
    """Tests that an explicit active_slices_override is forwarded instead of the live count."""
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0, 1}
    elastic_utils.elastic.get_active_slice_indices = Mock(return_value={0, 1})
    original_get_slice_to_devices = elastic_utils.elastic.get_slice_to_devices
    elastic_utils.elastic.get_slice_to_devices = Mock(
        return_value={0: [FakeDevice(slice_index=0)], 1: [FakeDevice(slice_index=1)]}
    )
    fake_recorder = Mock()

    try:
      elastic_utils.record_slice_state(fake_recorder, active_slices_override=0)
    finally:
      elastic_utils.elastic.get_slice_to_devices = original_get_slice_to_devices

    fake_recorder.record_elastic_slice_counts.assert_called_once_with(
        available_slices=2, active_slices=0, total_slices=2
    )

  def test_record_slice_state_noop_recorder_missing_attr(self):
    """Tests that record_slice_state no-ops for a recorder without record_elastic_slice_counts."""
    elastic_utils.elastic_manager = self.fake_manager
    fake_recorder = Mock(spec=[])

    elastic_utils.record_slice_state(fake_recorder)

    self.assertFalse(hasattr(fake_recorder, "record_elastic_slice_counts"))

  def test_record_slice_state_noop_not_pathways(self):
    """Tests that record_slice_state no-ops when not on the Pathways backend."""
    elastic_utils.elastic_manager = self.fake_manager
    elastic_utils.pathwaysutils.is_pathways_backend_used.return_value = False
    fake_recorder = Mock()

    elastic_utils.record_slice_state(fake_recorder)

    fake_recorder.record_elastic_slice_counts.assert_not_called()

  def test_record_slice_state_handles_health_check_failure(self):
    """A health-check failure must not propagate."""
    elastic_utils.elastic_manager = self.fake_manager
    elastic_utils.elastic.get_active_slice_indices = Mock(
        side_effect=MockJaxRuntimeError("unrecognized backend error")
    )
    fake_recorder = Mock()

    # Should not raise.
    elastic_utils.record_slice_state(fake_recorder)

    fake_recorder.record_elastic_slice_counts.assert_not_called()

  def test_record_slice_state_noop_no_elastic_manager(self):
    """Tests that record_slice_state no-ops when elastic_manager is uninitialized."""
    elastic_utils.elastic_manager = None
    fake_recorder = Mock()

    elastic_utils.record_slice_state(fake_recorder)

    fake_recorder.record_elastic_slice_counts.assert_not_called()

  def test_record_slice_state_noop_recorder_none(self):
    """Tests that record_slice_state no-ops when the recorder is None."""
    elastic_utils.elastic_manager = self.fake_manager

    # Should not raise.
    elastic_utils.record_slice_state(None)

  def test_elastic_event_lifecycle_records_slice_counts(self):
    """End-to-end: start -> wait-end/reinit-start -> reinit-end each log slice counts."""
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.available_inactive_slices = set()
    self.fake_manager.active_slice_indices = {0}
    elastic_utils.elastic.get_active_slice_indices = Mock(return_value={0})
    original_get_slice_to_devices = elastic_utils.elastic.get_slice_to_devices
    elastic_utils.elastic.get_slice_to_devices = Mock(
        return_value={0: [FakeDevice(slice_index=0)]}
    )
    fake_recorder = Mock()
    config = FakeConfig()

    try:
      elastic_utils.record_elastic_event_start(fake_recorder, config)
      elastic_utils.record_elastic_wait_end_and_reinit_start(fake_recorder)
      elastic_utils.record_elastic_reinit_end()
    finally:
      elastic_utils.elastic.get_slice_to_devices = original_get_slice_to_devices

    fake_recorder.record_elastic_wait_start_time.assert_called_once_with(
        event_type="elastic_slice_down"
    )
    fake_recorder.record_elastic_wait_end_time.assert_called_once_with(
        event_type="elastic_slice_down"
    )
    fake_recorder.record_elastic_reinit_start_time.assert_called_once_with()
    fake_recorder.record_elastic_reinit_end_time.assert_called_once_with()
    # active_slices=0 is forced on event start (we've lost/are waiting on slices),
    # then reflects the live count once wait ends and once reinit ends.
    self.assertEqual(
        [c.kwargs["active_slices"] for c in fake_recorder.record_elastic_slice_counts.call_args_list],
        [0, 1, 1],
    )

  def test_ensure_elastic_manager_initialized_readonly_config(self):
    """Tests that ensure_elastic_manager_initialized works with read-only config."""

    class ReadOnlyConfig:

      def __init__(self):
        object.__setattr__(self, "elastic_enabled", True)
        object.__setattr__(self, "elastic_min_slice_count", 1)
        object.__setattr__(self, "num_slices", 1)
        object.__setattr__(self, "elastic_timeout_seconds", 100)

      def __setattr__(self, name, value):
        raise ValueError("Configuration is read-only")

    config = ReadOnlyConfig()
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True

    # Should not raise ValueError
    elastic_utils.ensure_elastic_manager_initialized(config)
    self.assertEqual(elastic_utils.elastic_manager, self.fake_manager)

  @parameterized.parameters(
      # Positive cases
      ({1}, True),
      ({0}, True),
      ({1, 2}, True),
      ({0, 3, 6}, True),
      ({10, 25}, True),
      # Negative cases
      (set(), False),
  )
  def test_is_scale_up_event_with_set(self, available_inactive_slices, expected):
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager

    self.fake_manager.available_inactive_slices = available_inactive_slices
    self.assertEqual(elastic_utils.is_scale_up_event(config), expected)

  def test_maybe_bubble_elastic_exception_bubbles_on_elastic_errors(self):
    """Tests that elastic exceptions are bubbled up, while other exceptions are returned normally."""
    config = FakeConfig()
    config.elastic_enabled = True
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True

    # Scenario 1: Elastic JAX error propagates
    with self.assertRaises(MockJaxRuntimeError):
      elastic_utils.maybe_bubble_elastic_exception(config, MockJaxRuntimeError("TPU offline"))

    # Scenario 2: ScaleUpSignalError propagates
    with self.assertRaises(ScaleUpSignalError):
      elastic_utils.maybe_bubble_elastic_exception(config, ScaleUpSignalError())

    # Scenario 3: Non-elastic error is returned/ignored
    elastic_utils.maybe_bubble_elastic_exception(config, ValueError("Disk full"))

  def test_maybe_bubble_elastic_exception_disabled_does_not_bubble(self):
    """If elasticity is disabled, JaxRuntimeError should not bubble."""
    config = FakeConfig()
    config.elastic_enabled = False  # Disabled
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True

    # Executes normally (no exception raised)
    elastic_utils.maybe_bubble_elastic_exception(config, MockJaxRuntimeError("JAX error but elasticity disabled"))

  def test_checkpoint_exception_guard_checks_scale_up_on_success(self):
    """Signals ScaleUpSignalError if scale-up is active when save completes."""
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    self.fake_manager.available_inactive_slices = {1}  # Trigger scale-up

    mock_checkpoint_manager = Mock(spec=["wait_until_finished"])

    # Successful checkpoint save block raises ScaleUpSignalError to trigger restart
    with self.assertRaises(ScaleUpSignalError):
      with checkpointing.checkpoint_exception_guard(config, mock_checkpoint_manager):
        pass

    mock_checkpoint_manager.wait_until_finished.assert_called_once()

  def test_checkpoint_exception_guard_none_manager(self):
    """Checks that checkpoint_manager=None doesn't raise AttributeError on scale-up."""
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    self.fake_manager.available_inactive_slices = {1}  # Trigger scale-up

    with self.assertRaises(ScaleUpSignalError):
      with checkpointing.checkpoint_exception_guard(config, checkpoint_manager=None):
        pass

  def test_checkpoint_exception_guard_skips_scale_up_on_failure(self):
    """If checkpoint save fails, scale-up check should be skipped, and exception handled."""
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    self.fake_manager.available_inactive_slices = {1}

    handler_called = False

    def handler(_err):
      nonlocal handler_called
      handler_called = True

    with checkpointing.checkpoint_exception_guard(config, self.fake_manager, handler):
      raise ValueError("Save failed")

    self.assertTrue(handler_called)

  def test_wait_for_devices_placed_success(self):
    """Tests wait_for_devices_placed returns active devices when placement succeeds."""
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    devices = [FakeDevice(slice_index=0, process_index=0, task_id=0, device_id=0)]
    self.fake_jax.devices.return_value = devices

    mock_arr = Mock()
    self.fake_jax.device_put.return_value = mock_arr

    res = elastic_utils.wait_for_devices_placed(config, timeout=5.0, poll_interval=0.01)
    self.assertEqual(res, devices)
    self.fake_jax.device_put.assert_called_once()
    self.fake_jax.block_until_ready.assert_called_once_with(mock_arr)
    mock_arr.delete.assert_called_once()

  def test_wait_for_devices_placed_transient_error_recovers(self):
    """Tests wait_for_devices_placed retries and succeeds when first probe encounters transient error."""
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    devices = [FakeDevice(slice_index=0, process_index=0, task_id=0, device_id=0)]
    self.fake_jax.devices.return_value = devices

    mock_arr = Mock()
    # First call raises JaxRuntimeError, second call succeeds
    self.fake_jax.device_put.side_effect = [MockJaxRuntimeError("Placement in progress"), mock_arr]

    res = elastic_utils.wait_for_devices_placed(config, timeout=5.0, poll_interval=0.01)
    self.assertEqual(res, devices)
    self.assertEqual(self.fake_jax.device_put.call_count, 2)
    self.fake_jax.block_until_ready.assert_called_once_with(mock_arr)

  def test_wait_for_devices_placed_slice_drops_mid_poll(self):
    """Tests wait_for_devices_placed shrinks to surviving slice when a slice drops mid-poll."""
    config = FakeConfig()
    config.elastic_enabled = True
    config.elastic_min_slice_count = 1
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0, 1}
    self.fake_manager.slice_to_devices = {
        0: [FakeDevice(slice_index=0, process_index=0, task_id=0, device_id=0)],
        1: [FakeDevice(slice_index=1, process_index=1, task_id=1, device_id=1)],
    }
    all_devices = [
        FakeDevice(slice_index=0, process_index=0, task_id=0, device_id=0),
        FakeDevice(slice_index=1, process_index=1, task_id=1, device_id=1),
    ]
    self.fake_jax.devices.return_value = all_devices

    mock_arr = Mock()
    # First attempt on {0, 1} fails (slice 1 died), second attempt on {0} succeeds
    self.fake_jax.device_put.side_effect = [MockJaxRuntimeError("Slice 1 died"), mock_arr]

    # When get_active_slice_indices is called after failure, simulate slice 1 gone
    with unittest.mock.patch("pathwaysutils.elastic.elastic.get_active_slice_indices", return_value={0}):
      res = elastic_utils.wait_for_devices_placed(config, timeout=5.0, poll_interval=0.01)

  def test_wait_for_devices_placed_multiple_transient_errors(self):
    """Tests wait_for_devices_placed recovers after multiple consecutive transient errors."""
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    devices = [FakeDevice(slice_index=0, process_index=0, task_id=0, device_id=0)]
    self.fake_jax.devices.return_value = devices

    mock_arr = Mock()
    # 2 transient errors then success
    self.fake_jax.device_put.side_effect = [
        MockJaxRuntimeError("Placement in progress 1"),
        MockJaxRuntimeError("Placement in progress 2"),
        mock_arr,
    ]

    res = elastic_utils.wait_for_devices_placed(config, timeout=5.0, poll_interval=0.01)
    self.assertEqual(res, devices)
    self.assertEqual(self.fake_jax.device_put.call_count, 3)

  def test_wait_for_devices_placed_timeout_returns_live_devices(self):
    """Tests wait_for_devices_placed falls back to live_devices when timeout expires."""
    config = FakeConfig()
    config.elastic_enabled = True
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    devices = [FakeDevice(slice_index=0, process_index=0, task_id=0, device_id=0)]
    self.fake_jax.devices.return_value = devices

    self.fake_jax.device_put.side_effect = MockJaxRuntimeError("Persistent placement error")

    res = elastic_utils.wait_for_devices_placed(config, timeout=0.05, poll_interval=0.01)
    self.assertEqual(res, devices)

  def test_scale_up_signal_error_bubbles_for_checkpoint_mode(self):
    """Verifies ScaleUpSignalError raises for checkpoint mode to let elastic_retry handle scaling."""
    config = FakeConfig()
    config.elastic_enabled = True
    config.elastic_backup_kind = "checkpoint"
    err = ScaleUpSignalError("Scale up during initialization")

    # maybe_bubble_elastic_exception should raise ScaleUpSignalError
    with self.assertRaises(ScaleUpSignalError):
      elastic_utils.maybe_bubble_elastic_exception(config, err)

  def test_scale_up_signal_error_snapshot_mode_detection(self):
    """Verifies elastic_snapshot distinguishes snapshot vs checkpoint mode."""
    config = FakeConfig()
    config.elastic_enabled = True
    config.elastic_backup_kind = "snapshot"
    self.assertTrue(elastic_utils.elastic_snapshot(config))

    config.elastic_backup_kind = "checkpoint"
    self.assertFalse(elastic_utils.elastic_snapshot(config))


if __name__ == "__main__":
  unittest.main()
