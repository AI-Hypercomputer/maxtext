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
    self.elastic_backup_kind = "checkpoint"
    self.dataset_type = "grain"
    self.grain_use_elastic_iterator = True


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
    self.original_elastic = elastic_utils.elastic
    self.original_manager_class = pathwaysutils.elastic.manager.Manager
    self.original_scale_up_signal_error = getattr(pathwaysutils.elastic.manager, "ScaleUpSignalError", None)

    # Initialize fakes as mocks
    self.fake_gcs_utils = create_autospec(gcs_utils)
    self.fake_gcs_utils.add_trailing_slash.side_effect = gcs_utils.add_trailing_slash
    self.fake_pathwaysutils = create_autospec(pathwaysutils)
    self.fake_logging = create_autospec(self.original_max_logging)
    self.fake_jax = create_autospec(self.original_jax)
    self.fake_manager = create_autospec(self.original_manager_class, instance=True)
    self.fake_manager.available_inactive_slices = set()
    self.fake_manager.total_slice_count = 2
    self.fake_elastic = create_autospec(self.original_elastic)

    # Configure default behaviors if needed
    self.fake_pathwaysutils.is_pathways_backend_used.return_value = True
    self.fake_elastic.get_active_slice_indices.return_value = [0, 1]
    self.fake_elastic.get_slice_to_devices.return_value = {
        0: [FakeDevice()],
        1: [FakeDevice()],
    }
    self.fake_jax.process_index.return_value = 0

    # Inject fakes into elastic_utils namespace
    elastic_utils.pathwaysutils = self.fake_pathwaysutils
    elastic_utils.jax = self.fake_jax
    self.fake_jax.errors.JaxRuntimeError = MockJaxRuntimeError
    elastic_utils.gcs_utils = self.fake_gcs_utils
    elastic_utils.max_logging = self.fake_logging
    elastic_utils.elastic = self.fake_elastic

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
    elastic_utils.elastic = self.original_elastic
    pathwaysutils.elastic.manager.Manager = self.original_manager_class
    pathwaysutils.elastic.manager.ScaleUpSignalError = (  # pyrefly: ignore[bad-assignment]
        self.original_scale_up_signal_error,
    )
    elastic_utils.elastic_manager = None
    elastic_utils.pending_reinit_recorder = None
    elastic_utils.pending_elastic_event_type = None
    super().tearDown()

  def test_record_slice_state(self):
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    self.fake_manager.slice_to_devices = {0: [FakeDevice()], 1: [FakeDevice()]}
    self.fake_elastic.get_active_slice_indices.return_value = {0, 1}

    fake_recorder = Mock()
    fake_recorder.record_elastic_slice_counts = Mock()

    elastic_utils.record_slice_state(fake_recorder)

    fake_recorder.record_elastic_slice_counts.assert_called_once_with(available_slices=2, active_slices=1, total_slices=2)

    fake_recorder.record_elastic_slice_counts.reset_mock()
    elastic_utils.record_slice_state(fake_recorder, active_slices_override=0)
    fake_recorder.record_elastic_slice_counts.assert_called_once_with(available_slices=2, active_slices=0, total_slices=2)

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
    elastic_utils.clean_up_incomplete_checkpoints("gs://test_bucket/checkpoints")
    self.fake_gcs_utils.gcs_delete_directory.assert_not_called()

  def test_clean_up_checkpoints_incomplete(self):
    """Tests clean_up_incomplete_checkpoints when the latest checkpoint is incomplete."""
    checkpoint_dir = "gs://test_bucket/checkpoints"
    self.fake_gcs_utils.gcs_list_directories.return_value = ["1", "2", "10"]
    self.fake_gcs_utils.gcs_glob_pattern.return_value = []
    # No commit_success for "10"
    elastic_utils.clean_up_incomplete_checkpoints(checkpoint_dir)
    self.fake_gcs_utils.gcs_delete_directory.assert_called_once_with(f"{checkpoint_dir}/10/")

  def test_clean_up_checkpoints_complete(self):
    """Tests clean_up_incomplete_checkpoints when the latest checkpoint is complete."""
    checkpoint_dir = "gs://test_bucket/checkpoints"
    self.fake_gcs_utils.gcs_list_directories.return_value = ["1", "2", "10"]
    self.fake_gcs_utils.gcs_glob_pattern.return_value = [f"{checkpoint_dir}/10/commit_success_0"]
    elastic_utils.clean_up_incomplete_checkpoints(checkpoint_dir)
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

      def wait(self):
        self.wait_called = True

    cm = FakeCheckpointManager()

    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.available_inactive_slices = {1}

    with self.assertRaises(ScaleUpSignalError):
      elastic_utils.maybe_elastic_scale_up(config, cm)

    self.assertTrue(cm.wait_called)

  def test_elastic_retry_rejects_min_slices_above_total(self):
    config = FakeConfig()
    config.elastic_min_slice_count = 3
    elastic_utils.elastic_manager = self.fake_manager

    with self.assertRaisesRegex(ValueError, "larger than the number of slices"):
      elastic_utils.elastic_retry(config)

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

  def test_elastic_retry_pre_callback_without_pre_callback_fn(self):
    """pre_callback still works when pre_callback_fn is not supplied."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0, 1}

    elastic_utils.elastic_retry(config)

    kwargs = self.fake_manager.elastic_retry.call_args.kwargs
    kwargs["pre_callback"]()  # Must not raise.

  def test_elastic_retry_pre_callback_forwarded(self):
    """The manager's pre_callback must call pre_callback_fn."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0, 1}

    fake_pre_callback = Mock()
    elastic_utils.elastic_retry(config, pre_callback_fn=fake_pre_callback)

    kwargs = self.fake_manager.elastic_retry.call_args.kwargs
    kwargs["pre_callback"]()
    fake_pre_callback.assert_called_once()

  @parameterized.named_parameters(("all_slices", -1, True), ("min_equals_total", 2, True), ("min_below_total", 1, False))
  def test_is_pause_resume(self, min_slice_count, expected):
    config = FakeConfig()
    config.elastic_min_slice_count = min_slice_count
    elastic_utils.elastic_manager = self.fake_manager

    self.assertEqual(elastic_utils.is_pause_resume(config), expected)

  @parameterized.named_parameters(
      ("resize_regular_grain_iterator", 1, "grain", False, True),
      ("resize_elastic_iterator", 1, "grain", True, False),
      ("resize_non_grain", 1, "tfds", False, False),
      ("pause_resume_regular_grain_iterator", -1, "grain", False, False),
  )
  def test_elastic_retry_resize_requires_elastic_iterator(
      self, min_slice_count, dataset_type, use_elastic_iterator, expect_error
  ):
    """A replica resize with the regular grain iterator is rejected: its state can't move to another host count."""
    config = FakeConfig()
    config.elastic_min_slice_count = min_slice_count
    config.dataset_type = dataset_type
    config.grain_use_elastic_iterator = use_elastic_iterator
    elastic_utils.elastic_manager = self.fake_manager

    if expect_error:
      with self.assertRaisesRegex(ValueError, "grain_use_elastic_iterator=True"):
        elastic_utils.elastic_retry(config)
    else:
      elastic_utils.elastic_retry(config)
      self.fake_manager.elastic_retry.assert_called_once()

  def _fake_pathways_retry_with_one_retry(self):
    """Makes the fake manager retry once after an elastic error, running the event callback in between.

    Like pathwaysutils, it lets an error from the last attempt propagate.
    """

    def fake_pathways_retry(**kwargs):
      on_elastic_event_callback = kwargs["on_elastic_event_callback"]

      def decorator(func):
        def run_with_one_retry():
          try:
            return func()
          except (ScaleUpSignalError, MockJaxRuntimeError):
            on_elastic_event_callback()
          return func()

        return run_with_one_retry

      return decorator

    self.fake_manager.elastic_retry.side_effect = fake_pathways_retry
    self.fake_gcs_utils.gcs_list_directories.return_value = []

  @parameterized.named_parameters(("scale_up", ScaleUpSignalError, True), ("slice_down", MockJaxRuntimeError, False))
  def test_elastic_retry_tells_callback_the_event_kind(self, error_cls, expected_scale_up):
    """The event callback learns from the failed attempt itself whether it was a scale-up or a slice-down."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager
    callback_fn = Mock()
    attempts = []

    def train():
      attempts.append(len(attempts))
      if len(attempts) == 1:
        raise error_cls()

    self._fake_pathways_retry_with_one_retry()

    elastic_utils.elastic_retry(config, callback_fn=callback_fn)(train)()

    callback_fn.assert_called_once_with(scale_up=expected_scale_up)
    self.assertLen(attempts, 2)

  def test_elastic_retry_scale_up_does_not_leak_into_next_run(self):
    """A scale-up that ended one run (retries exhausted) doesn't label the first event of the next run."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager
    callback_fn = Mock()
    errors = [ScaleUpSignalError(), ScaleUpSignalError(), MockJaxRuntimeError(), None]

    def train():
      error = errors.pop(0)
      if error is not None:
        raise error

    self._fake_pathways_retry_with_one_retry()
    train_func = elastic_utils.elastic_retry(config, callback_fn=callback_fn)(train)

    with self.assertRaises(ScaleUpSignalError):
      train_func()  # Both attempts are scale-ups; the second one propagates.
    callback_fn.reset_mock()

    train_func()  # Slice-down, then success.

    callback_fn.assert_called_once_with(scale_up=False)

  def test_retry_cache_class(self):
    """A fresh cache is off; entries survive attempts on the same slices and are dropped on a slice change."""
    cache = elastic_utils.RetryCache()
    self.assertFalse(cache.enabled)
    cache.put("mesh", "ignored")
    self.assertIsNone(cache.get("mesh"))

    cache.reset(enabled=True)
    cache.clear_if_slices_changed(frozenset({0, 1}))
    cache.put("mesh", "mesh_a")
    cache.clear_if_slices_changed(frozenset({0, 1}))
    self.assertEqual(cache.get("mesh"), "mesh_a")

    cache.clear_if_slices_changed(frozenset({0}))
    self.assertIsNone(cache.get("mesh"))
    cache.put("mesh", "mesh_b")
    self.assertEqual(cache.get("mesh"), "mesh_b")

    cache.reset(enabled=False)
    self.assertFalse(cache.enabled)
    self.assertIsNone(cache.get("mesh"))

  def test_retry_cache_snapshotter_survives_reset_and_slice_change(self):
    """The snapshotter slot is not a cache entry: neither a slice change nor a reset clears it."""
    cache = elastic_utils.RetryCache()
    self.assertIsNone(cache.snapshotter)

    snapshotter = Mock()
    cache.snapshotter = snapshotter
    cache.reset(enabled=True)
    cache.clear_if_slices_changed(frozenset({0, 1}))
    cache.clear_if_slices_changed(frozenset({0}))
    cache.reset(enabled=False)

    self.assertIs(cache.snapshotter, snapshotter)
    self.assertIsNone(cache.get("snapshotter"))

  def test_get_or_build(self):
    """Builds every time without a cache; with one, builds on a miss and reuses on a hit."""
    builds = []

    def build_mesh():
      builds.append(len(builds))
      return f"mesh_{builds[-1]}"

    self.assertEqual(elastic_utils.get_or_build(None, "mesh", build_mesh), "mesh_0")
    self.assertEqual(elastic_utils.get_or_build(None, "mesh", build_mesh), "mesh_1")

    cache = elastic_utils.RetryCache()
    cache.reset(enabled=True)
    self.assertEqual(elastic_utils.get_or_build(cache, "mesh", build_mesh), "mesh_2")
    self.assertEqual(elastic_utils.get_or_build(cache, "mesh", build_mesh), "mesh_2")
    self.assertEqual(builds, [0, 1, 2])

  def test_get_or_build_caches_none(self):
    """A build that returns None is cached too, so it is not repeated on the next attempt."""
    builds = []

    def build_none():
      # Returns None implicitly: the point of the test is that None is a cacheable value.
      builds.append(len(builds))

    cache = elastic_utils.RetryCache()
    cache.reset(enabled=True)
    self.assertFalse(cache.contains("optional"))
    self.assertIsNone(elastic_utils.get_or_build(cache, "optional", build_none))
    self.assertTrue(cache.contains("optional"))
    self.assertIsNone(elastic_utils.get_or_build(cache, "optional", build_none))
    self.assertEqual(builds, [0])

    cache.reset(enabled=False)
    self.assertFalse(cache.contains("optional"))

  def _run_three_attempts_through_fake_pathways_retry(self):
    """Makes the fake manager run the decorated function on slices {0, 1}, {0, 1} and then {0}."""

    def fake_pathways_retry(**kwargs):
      pre_callback = kwargs["pre_callback"]

      def decorator(func):
        def run_three_attempts():
          for active_slices in ({0, 1}, {0, 1}, {0}):
            self.fake_manager.active_slice_indices = active_slices
            pre_callback()
            func()

        return run_three_attempts

      return decorator

    self.fake_manager.elastic_retry.side_effect = fake_pathways_retry

  def test_retry_cache_reused_on_same_slices_and_rebuilt_after_resize(self):
    """Attempts on the same slices reuse cached objects, and an attempt on other slices rebuilds them."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager
    retry_cache = elastic_utils.RetryCache()
    built_meshes = []

    def setup_mesh():
      self.assertTrue(retry_cache.enabled)
      mesh = retry_cache.get("mesh")
      if mesh is None:
        mesh = f"mesh_{len(built_meshes)}"
        built_meshes.append(mesh)
        retry_cache.put("mesh", mesh)

    self._run_three_attempts_through_fake_pathways_retry()

    elastic_utils.elastic_retry(config, retry_cache=retry_cache)(setup_mesh)()

    self.assertEqual(built_meshes, ["mesh_0", "mesh_1"])
    # The cache is turned off again once the elastic run is over.
    self.assertFalse(retry_cache.enabled)
    self.assertIsNone(retry_cache.get("mesh"))

  def test_elastic_retry_without_retry_cache(self):
    """Without a cache every attempt runs the decorated function in full."""
    config = FakeConfig()
    elastic_utils.elastic_manager = self.fake_manager
    attempts = []

    def setup_mesh():
      attempts.append(set(self.fake_manager.active_slice_indices))

    self._run_three_attempts_through_fake_pathways_retry()

    elastic_utils.elastic_retry(config)(setup_mesh)()

    self.assertEqual(attempts, [{0, 1}, {0, 1}, {0}])

  def test_record_elastic_event_start(self):
    """Tests recording an elastic slice down start."""
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.slice_to_devices = {0: [FakeDevice()], 1: [FakeDevice()]}
    fake_recorder = Mock()

    elastic_utils.record_elastic_event_start(fake_recorder, scale_up=False)

    fake_recorder.record_elastic_wait_start_time.assert_called_once_with(event_type="elastic_slice_down")
    fake_recorder.record_elastic_slice_counts.assert_called_once()
    self.assertEqual(elastic_utils.pending_elastic_event_type, "elastic_slice_down")

  def test_record_elastic_event_start_scale_up(self):
    """Tests recording an elastic slice scale up start."""
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.slice_to_devices = {0: [FakeDevice()], 1: [FakeDevice()]}
    fake_recorder = Mock()

    elastic_utils.record_elastic_event_start(fake_recorder, scale_up=True)

    fake_recorder.record_elastic_wait_start_time.assert_called_once_with(event_type="elastic_scale_up")
    fake_recorder.record_elastic_slice_counts.assert_called_once()

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
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    self.fake_manager.slice_to_devices = {0: [FakeDevice()], 1: [FakeDevice()]}
    fake_recorder = Mock()

    elastic_utils.record_elastic_wait_end_and_reinit_start(fake_recorder)

    fake_recorder.record_elastic_wait_end_time.assert_called_once_with(event_type="elastic_slice_down")
    fake_recorder.record_elastic_reinit_start_time.assert_called_once()
    fake_recorder.record_elastic_slice_counts.assert_called_once()
    self.assertIs(elastic_utils.pending_reinit_recorder, fake_recorder)
    self.assertIsNone(elastic_utils.pending_elastic_event_type)

  def test_record_elastic_reinit_end(self):
    """Tests recording end of elastic reinit."""
    fake_recorder = Mock()
    elastic_utils.pending_reinit_recorder = fake_recorder
    elastic_utils.elastic_manager = self.fake_manager
    self.fake_manager.active_slice_indices = {0}
    self.fake_manager.slice_to_devices = {0: [FakeDevice()], 1: [FakeDevice()]}

    elastic_utils.record_elastic_reinit_end()

    fake_recorder.record_elastic_reinit_end_time.assert_called_once()
    fake_recorder.record_elastic_slice_counts.assert_called_once()
    self.assertIsNone(elastic_utils.pending_reinit_recorder)

  def test_record_elastic_reinit_end_on_cold_start(self):
    """Tests recording end of elastic reinit on cold start."""
    elastic_utils.pending_reinit_recorder = None

    elastic_utils.record_elastic_reinit_end()

  def test_record_elastic_event_start_non_elastic_recorder_noop(self):
    """A recorder lacking the elastic API (e.g. the ImportError fallback) must not raise."""
    elastic_utils.elastic_manager = self.fake_manager
    non_elastic_recorder = Mock(spec=[])  # No record_elastic_* attributes.

    elastic_utils.record_elastic_event_start(non_elastic_recorder, scale_up=False)  # Must not raise.

    self.assertEqual(elastic_utils.pending_elastic_event_type, "elastic_slice_down")

  def test_record_elastic_wait_end_and_reinit_start_non_elastic_recorder_noop(self):
    """A recorder lacking the elastic API must not raise, but is still tracked as pending."""
    elastic_utils.pending_elastic_event_type = "elastic_slice_down"  # pyrefly: ignore[bad-assignment]
    non_elastic_recorder = Mock(spec=[])

    elastic_utils.record_elastic_wait_end_and_reinit_start(non_elastic_recorder)  # Must not raise.

    self.assertIs(elastic_utils.pending_reinit_recorder, non_elastic_recorder)
    self.assertIsNone(elastic_utils.pending_elastic_event_type)

  def test_record_elastic_reinit_end_non_elastic_recorder_noop(self):
    """A recorder lacking the elastic API must not raise, and pending state is still cleared."""
    non_elastic_recorder = Mock(spec=[])
    elastic_utils.pending_reinit_recorder = non_elastic_recorder

    elastic_utils.record_elastic_reinit_end()  # Must not raise.

    self.assertIsNone(elastic_utils.pending_reinit_recorder)

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

    mock_checkpoint_manager = Mock(spec=checkpointing.ocp.training.Checkpointer)

    # Successful checkpoint save block raises ScaleUpSignalError to trigger restart
    with self.assertRaises(ScaleUpSignalError):
      with checkpointing.checkpoint_exception_guard(config, mock_checkpoint_manager):
        pass

    mock_checkpoint_manager.wait.assert_called_once()

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


if __name__ == "__main__":
  unittest.main()
