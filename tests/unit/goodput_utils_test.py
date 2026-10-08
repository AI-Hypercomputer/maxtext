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

"""Tests for goodput_utils.py"""

import os
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pytest

ml_goodput_measurement = pytest.importorskip("ml_goodput_measurement")
goodput = ml_goodput_measurement.goodput
goodput_elastic = ml_goodput_measurement.goodput_elastic
monitoring = ml_goodput_measurement.monitoring
monitoring_elastic = ml_goodput_measurement.monitoring_elastic
checkpoint_badput_calculator = ml_goodput_measurement.checkpoint_badput_calculator


import jax
import jax.numpy as jnp
from maxtext.common import checkpointing

from maxtext.configs import pyconfig
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    _construct_goodput_monitor,
    create_goodput_recorder,
    get_goodput_job_name,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from tests.utils.test_helpers import get_test_config_path, get_test_base_output_directory

pytestmark = [pytest.mark.external_training]


class _ConfigOverride:
  """Wrapper exposing overridden attributes on top of a read-only config."""

  def __init__(self, base_config, **overrides):
    self._base_config = base_config
    self._overrides = overrides

  def __getattr__(self, name):
    if name in self._overrides:
      return self._overrides[name]
    return getattr(self._base_config, name)


class GoodputUtilsTest(unittest.TestCase):
  """Tests for Goodput monitoring and recording."""

  def setUp(self):
    super().setUp()
    base_output_directory = get_test_base_output_directory()
    self.config = pyconfig.initialize(
        [None, get_test_config_path()],
        base_output_directory=base_output_directory,
        run_name="runner_test",
        enable_checkpointing=False,
        monitor_goodput=True,
        enable_goodput_recording=True,
        monitor_step_time_deviation=True,
    )

  @mock.patch("ml_goodput_measurement.goodput.GoodputRecorder.record_job_end_time")
  @mock.patch("ml_goodput_measurement.goodput.GoodputRecorder.record_job_start_time")
  @mock.patch("google.cloud.logging.Client")
  def test_record_goodput(self, mock_cloud_logger, mock_record_job_start_time, mock_record_job_end_time):
    mock_cloud_logger.return_value = mock.MagicMock()
    mock_record_job_start_time.return_value = mock.MagicMock()
    mock_record_job_end_time.return_value = mock.MagicMock()

    recorder = create_goodput_recorder(self.config)
    with maybe_record_goodput(recorder, GoodputEvent.JOB):
      pass

    mock_cloud_logger.return_value.logger.assert_called()
    mock_record_job_start_time.assert_called()
    mock_record_job_end_time.assert_called()

    class TestException(BaseException):
      pass

    mock_record_job_start_time.reset_mock()
    mock_record_job_end_time.reset_mock()
    with self.assertRaises(TestException):
      with maybe_record_goodput(recorder, GoodputEvent.JOB):
        mock_record_job_start_time.assert_called_once()
        raise TestException()

    mock_record_job_start_time.assert_called_once()
    mock_record_job_end_time.assert_not_called()

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_goodput(self, mock_start_goodput_uploader, mock_stop_goodput_uploader, *unused_rolling_mocks):
    mock_start_goodput_uploader.return_value = mock.MagicMock()

    with maybe_monitor_goodput(self.config):
      mock_start_goodput_uploader.assert_called()
    mock_stop_goodput_uploader.assert_called()

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_rolling_window_goodput(
      self, mock_start_goodput, mock_stop_goodput, mock_start_rolling, mock_stop_rolling
  ):
    """Both uploaders start; on exit the rolling window stops before the cumulative one."""
    windows = [3600, 86400]
    config = _ConfigOverride(self.config, enable_rolling_window_goodput=True, rolling_windows_seconds=windows)
    calls = mock.Mock()
    calls.attach_mock(mock_stop_rolling, "stop_rolling")
    calls.attach_mock(mock_stop_goodput, "stop_goodput")

    with maybe_monitor_goodput(config):
      mock_start_goodput.assert_called_once()
      mock_start_rolling.assert_called_once_with(windows)
      mock_stop_rolling.assert_not_called()

    self.assertEqual(calls.mock_calls, [mock.call.stop_rolling(), mock.call.stop_goodput()])

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_rolling_window_goodput_disabled(
      self, mock_start_goodput, mock_stop_goodput, mock_start_rolling, mock_stop_rolling
  ):
    """enable_rolling_window_goodput=False only runs the cumulative uploader."""
    config = _ConfigOverride(self.config, enable_rolling_window_goodput=False)

    with maybe_monitor_goodput(config):
      mock_start_goodput.assert_called_once()

    mock_stop_goodput.assert_called_once()
    mock_start_rolling.assert_not_called()
    mock_stop_rolling.assert_not_called()

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_rolling_window_goodput_uploader")
  @mock.patch(
      "ml_goodput_measurement.monitoring.GoodputMonitor.start_rolling_window_goodput_uploader",
      side_effect=RuntimeError("start boom"),
  )
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_rolling_window_goodput_start_failure_degrades_gracefully(
      self, unused_mock_start_goodput, mock_stop_goodput, mock_start_rolling, mock_stop_rolling
  ):
    """A failing rolling window start is swallowed; the body still runs and nothing is stopped twice."""
    config = _ConfigOverride(self.config, enable_rolling_window_goodput=True)
    body_ran = False

    with maybe_monitor_goodput(config):  # Must not raise.
      body_ran = True

    self.assertTrue(body_ran)
    mock_start_rolling.assert_called_once()
    mock_stop_rolling.assert_not_called()  # Never started, so never stopped.
    mock_stop_goodput.assert_called_once()  # Cumulative monitoring is unaffected.

  @mock.patch(
      "ml_goodput_measurement.monitoring.GoodputMonitor.stop_rolling_window_goodput_uploader",
      side_effect=RuntimeError("stop boom"),
  )
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_rolling_window_goodput_stop_failure_degrades_gracefully(
      self, unused_mock_start_goodput, mock_stop_goodput, unused_mock_start_rolling, mock_stop_rolling
  ):
    """A failing rolling window stop is swallowed and the cumulative uploader is still stopped."""
    config = _ConfigOverride(self.config, enable_rolling_window_goodput=True)

    with maybe_monitor_goodput(config):  # Must not raise on exit.
      pass

    mock_stop_rolling.assert_called_once()
    mock_stop_goodput.assert_called_once()

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_rolling_window_goodput_propagates_body_exception(
      self, unused_mock_start_goodput, mock_stop_goodput, unused_mock_start_rolling, mock_stop_rolling
  ):
    """Graceful degradation must not swallow errors raised by training itself."""
    config = _ConfigOverride(self.config, enable_rolling_window_goodput=True)

    class TrainingError(Exception):
      pass

    with self.assertRaises(TrainingError):
      with maybe_monitor_goodput(config):
        raise TrainingError()

    mock_stop_rolling.assert_called_once()
    mock_stop_goodput.assert_called_once()

  def test_rolling_windows_seconds_rejects_empty_list(self):
    """An empty window list would spawn an idle uploader process, so config validation rejects it."""
    with self.assertRaisesRegex(Exception, "rolling_windows_seconds"):
      pyconfig.initialize(
          [None, get_test_config_path()],
          base_output_directory=get_test_base_output_directory(),
          run_name="runner_test",
          enable_checkpointing=False,
          rolling_windows_seconds=[],
      )

  def test_job_recording_constants(self):
    """Constants must map to the recorder method names."""
    self.assertEqual(RECORD_JOB_START_TIME, "record_job_start_time")
    self.assertEqual(RECORD_JOB_END_TIME, "record_job_end_time")

  def _common_monitor_kwargs(self):
    """Helper to construct goodput monitor."""
    return {
        "job_name": self.config.run_name,
        "logger_name": f"goodput_{self.config.run_name}",
        "tensorboard_dir": tempfile.mkdtemp(),
        "upload_interval": self.config.goodput_upload_interval_seconds,
        "monitoring_enabled": True,
        "include_badput_breakdown": True,
        "include_step_deviation": self.config.monitor_step_time_deviation,
        "step_deviation_interval_seconds": self.config.step_deviation_interval_seconds,
        "gcp_options": monitoring.GCPOptions(),
    }

  @mock.patch("google.cloud.logging.Client")
  def test_construct_goodput_monitor_non_elastic(self, mock_cloud_logger):
    """A McJAX config must construct the base monitor."""
    mock_cloud_logger.return_value = mock.MagicMock()
    self.assertFalse(self.config.elastic_enabled)

    monitor = _construct_goodput_monitor(self.config, self._common_monitor_kwargs())

    self.assertIsInstance(monitor, monitoring.GoodputMonitor)
    self.assertNotIsInstance(monitor, monitoring_elastic.ElasticGoodputMonitor)

  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  @mock.patch("google.cloud.logging.Client")
  def test_construct_goodput_monitor_elastic(self, mock_cloud_logger, mock_should_use_elastic):
    """elastic_enabled=True on an actual Pathways run must construct the elastic monitor."""
    mock_cloud_logger.return_value = mock.MagicMock()
    mock_should_use_elastic.return_value = True
    config = _ConfigOverride(self.config, elastic_enabled=True)

    monitor = _construct_goodput_monitor(config, self._common_monitor_kwargs())

    self.assertIsInstance(monitor, monitoring_elastic.ElasticGoodputMonitor)

  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  @mock.patch("google.cloud.logging.Client")
  def test_construct_goodput_monitor_elastic_enabled_not_pathways_falls_back(
      self, mock_cloud_logger, mock_should_use_elastic
  ):
    """elastic_enabled=True but not actually on Pathways must fall back to the base monitor."""
    mock_cloud_logger.return_value = mock.MagicMock()
    mock_should_use_elastic.return_value = False
    config = _ConfigOverride(self.config, elastic_enabled=True)

    monitor = _construct_goodput_monitor(config, self._common_monitor_kwargs())

    self.assertIsInstance(monitor, monitoring.GoodputMonitor)
    self.assertNotIsInstance(monitor, monitoring_elastic.ElasticGoodputMonitor)

  @mock.patch("ml_goodput_measurement.monitoring_elastic.ElasticGoodputMonitor", side_effect=RuntimeError("boom"))
  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  @mock.patch("google.cloud.logging.Client")
  def test_construct_goodput_monitor_elastic_construction_failure_falls_back(
      self, mock_cloud_logger, mock_should_use_elastic, unused_mock_elastic_monitor_cls
  ):
    """A failure constructing the elastic monitor must fall back rather than propagate."""
    mock_cloud_logger.return_value = mock.MagicMock()
    mock_should_use_elastic.return_value = True
    config = _ConfigOverride(self.config, elastic_enabled=True)

    monitor = _construct_goodput_monitor(config, self._common_monitor_kwargs())  # Must not raise.

    # Note: not asserting assertNotIsInstance(monitor, monitoring_elastic.ElasticGoodputMonitor)
    # here - that name is itself patched to a Mock for this test, so it isn't a usable type.
    self.assertIsInstance(monitor, monitoring.GoodputMonitor)

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_rolling_window_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring_elastic.ElasticGoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring_elastic.ElasticGoodputMonitor.start_goodput_uploader")
  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  def test_monitor_goodput_elastic(
      self, mock_should_use_elastic, mock_start_goodput_uploader, mock_stop_goodput_uploader, *unused_rolling_mocks
  ):
    """maybe_monitor_goodput actually starts/stops an ElasticGoodputMonitor when elastic is active."""
    mock_should_use_elastic.return_value = True
    mock_start_goodput_uploader.return_value = mock.MagicMock()
    config = _ConfigOverride(self.config, elastic_enabled=True)

    with maybe_monitor_goodput(config):
      mock_start_goodput_uploader.assert_called()
    mock_stop_goodput_uploader.assert_called()

  @mock.patch("google.cloud.logging.Client")
  def test_create_goodput_recorder_non_elastic(self, mock_cloud_logger):
    """Regular (non-Pathways/McJAX) config must get the base recorder."""
    mock_cloud_logger.return_value = mock.MagicMock()
    self.assertFalse(self.config.elastic_enabled)

    recorder = create_goodput_recorder(self.config)

    self.assertNotIsInstance(recorder, goodput_elastic.ElasticGoodputRecorder)

  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  @mock.patch("google.cloud.logging.Client")
  def test_create_goodput_recorder_elastic(self, mock_cloud_logger, mock_should_use_elastic):
    """elastic_enabled=True on an actual Pathways run must get the elastic recorder."""
    mock_cloud_logger.return_value = mock.MagicMock()
    mock_should_use_elastic.return_value = True
    config = _ConfigOverride(self.config, elastic_enabled=True)

    recorder = create_goodput_recorder(config)

    self.assertIsInstance(recorder, goodput_elastic.ElasticGoodputRecorder)

  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  @mock.patch("google.cloud.logging.Client")
  def test_create_goodput_recorder_elastic_enabled_not_pathways_falls_back(
      self, mock_cloud_logger, mock_should_use_elastic
  ):
    """elastic_enabled=True but not actually on Pathways (e.g. McJAX) must get the base recorder."""
    mock_cloud_logger.return_value = mock.MagicMock()
    mock_should_use_elastic.return_value = False
    config = _ConfigOverride(self.config, elastic_enabled=True)

    recorder = create_goodput_recorder(config)

    self.assertNotIsInstance(recorder, goodput_elastic.ElasticGoodputRecorder)

  @mock.patch("ml_goodput_measurement.goodput_elastic.ElasticGoodputRecorder", side_effect=RuntimeError("boom"))
  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  @mock.patch("google.cloud.logging.Client")
  def test_create_goodput_recorder_elastic_construction_failure_falls_back(
      self, mock_cloud_logger, mock_should_use_elastic, unused_mock_elastic_recorder_cls
  ):
    """A failure constructing the elastic recorder must fall back rather than propagate."""
    mock_cloud_logger.return_value = mock.MagicMock()
    mock_should_use_elastic.return_value = True
    config = _ConfigOverride(self.config, elastic_enabled=True)

    recorder = create_goodput_recorder(config)  # Must not raise.

    self.assertIsInstance(recorder, goodput.GoodputRecorder)

  @mock.patch("ml_goodput_measurement.goodput.GoodputRecorder.record_job_end_time")
  @mock.patch("ml_goodput_measurement.goodput.GoodputRecorder.record_job_start_time")
  @mock.patch("google.cloud.logging.Client")
  def test_explicit_job_recording_graceful_completion(
      self, mock_cloud_logger, mock_record_job_start_time, mock_record_job_end_time
  ):
    """Both start and end are recorded when the job completes gracefully."""
    mock_cloud_logger.return_value = mock.MagicMock()
    recorder = create_goodput_recorder(self.config)

    record_goodput(recorder, RECORD_JOB_START_TIME)
    _job_completed_gracefully = False
    try:
      _job_completed_gracefully = True
    finally:
      if _job_completed_gracefully:
        record_goodput(recorder, RECORD_JOB_END_TIME)

    mock_record_job_start_time.assert_called_once()
    mock_record_job_end_time.assert_called_once()

  @mock.patch("ml_goodput_measurement.goodput.GoodputRecorder.record_job_end_time")
  @mock.patch("ml_goodput_measurement.goodput.GoodputRecorder.record_job_start_time")
  @mock.patch("google.cloud.logging.Client")
  def test_explicit_job_recording_elastic_restart(
      self, mock_cloud_logger, mock_record_job_start_time, mock_record_job_end_time
  ):
    """Only start is recorded when the elastic manager handles the error internally.

    This simulates the elastic-restart scenario: the manager catches the JAX
    exception inside train_loop, so the loop exits without raising.  The
    _job_completed_gracefully flag is never set, so record_job_end_time must
    not be called.
    """
    mock_cloud_logger.return_value = mock.MagicMock()
    recorder = create_goodput_recorder(self.config)

    record_goodput(recorder, RECORD_JOB_START_TIME)
    _job_completed_gracefully = False
    try:
      pass  # Elastic manager caught and suppressed the exception.
    finally:
      if _job_completed_gracefully:
        record_goodput(recorder, RECORD_JOB_END_TIME)

    mock_record_job_start_time.assert_called_once()
    mock_record_job_end_time.assert_not_called()

  @mock.patch("google.cloud.logging.Client")
  def test_goodput_job_name_override(self, mock_cloud_logger):
    """goodput_job_name overrides run_name for GoodputRecorder and GoodputMonitor when set."""
    mock_cloud_logger.return_value = mock.MagicMock()
    self.assertEqual(get_goodput_job_name(self.config), "runner_test")

    overridden_config = _ConfigOverride(self.config, goodput_job_name="gemma3-4b-pre-runner_test")
    self.assertEqual(get_goodput_job_name(overridden_config), "gemma3-4b-pre-runner_test")

    recorder = create_goodput_recorder(overridden_config)
    self.assertIsNotNone(recorder)
    self.assertEqual(mock_cloud_logger.return_value.logger.call_args[0][0], "goodput_gemma3-4b-pre-runner_test")

  @mock.patch.dict("os.environ", {"M_GOODPUT_JOB_NAME": "llama3-pre-shared-run"}, clear=False)
  def test_goodput_job_name_env_override(self):
    """M_GOODPUT_JOB_NAME populates config.goodput_job_name via pyconfig."""
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        base_output_directory=get_test_base_output_directory(),
        run_name="shared-run",
        enable_checkpointing=False,
    )
    self.assertEqual(cfg.run_name, "shared-run")
    self.assertEqual(cfg.goodput_job_name, "llama3-pre-shared-run")
    self.assertEqual(get_goodput_job_name(cfg), "llama3-pre-shared-run")


class _RecordingLogger:
  """Minimal Orbax `AbstractLogger` that records every logged entry."""

  def __init__(self):
    self.entries = []

  def log_entry(self, entry):
    self.entries.append(entry)


def _entries_of_type(logger, event_type):
  return [e for e in logger.entries if e.get("event_type") == event_type]


class CheckpointGoodputLoggerTest(unittest.TestCase):
  """Checkpoint step statistics must reach the Goodput logger (b/568044767).

  With `enable_checkpoint_cloud_logger=true`, Orbax save/restore step statistics
  feed Goodput's checkpoint save/restore badput. The Orbax v1 migration silently
  dropped them, so these tests drive a real on-disk save -> restore through
  create_orbax_checkpoint_manager + load_state_if_possible.
  """

  def setUp(self):
    super().setUp()
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
    self.state = {"w": jnp.arange(8, dtype=jnp.float32)}
    self.abstract_state = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state)

  def _manager(self, logger, use_async=False):
    """Creates an Orbax v1 Checkpointer in the test dir with `logger` attached."""
    return checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._dir, "ckpt"),
        enable_checkpointing=True,
        use_async=use_async,
        save_interval_steps=1,
        orbax_logger=logger,
    )

  def _restore(self, manager):
    """Restores the latest step through load_state_if_possible."""
    restored, _ = checkpointing.load_state_if_possible(
        manager,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=self.abstract_state,
        dataset_type="tfds",
    )
    return restored

  def _save_and_restore(self, logger, use_async=False, step=0):
    """Saves `self.state` at `step`, then restores it with a fresh manager like a resumed run."""
    manager = self._manager(logger, use_async=use_async)
    self.assertTrue(checkpointing.save_checkpoint(manager, step, self.state))
    checkpointing.wait_until_finished(manager)
    return self._restore(self._manager(logger, use_async=use_async))

  def _assert_step_statistics_emitted(self, use_async):
    """Asserts one save and one restore entry with positive durations reach the logger."""
    logger = _RecordingLogger()

    restored = self._save_and_restore(logger, use_async=use_async, step=3)

    self.assertTrue(jnp.array_equal(restored["items"]["w"], self.state["w"]))
    saves = _entries_of_type(logger, "save")
    restores = _entries_of_type(logger, "restore")
    self.assertEqual(len(saves), 1, f"expected one save entry, got {logger.entries}")
    self.assertEqual(len(restores), 1, f"expected one restore entry, got {logger.entries}")
    self.assertEqual(saves[0]["step"], 3)
    self.assertEqual(restores[0]["step"], 3)
    self.assertGreater(saves[0]["checkpoint_manager_blocking_duration_secs"], 0)
    self.assertGreater(restores[0]["checkpoint_manager_duration_secs"], 0)
    self.assertGreater(restores[0]["checkpointer_duration_secs"], 0)

  def test_sync_save_and_restore_emit_step_statistics(self):
    self._assert_step_statistics_emitted(use_async=False)

  def test_async_save_and_restore_emit_step_statistics(self):
    self._assert_step_statistics_emitted(use_async=True)

  def test_entries_produce_goodput_checkpoint_badput(self):
    logger = _RecordingLogger()
    self._save_and_restore(logger)

    # Same wiring as GoodputCalculator: entries read from the Goodput log are
    # handed to the checkpoint badput calculator.
    calc = checkpoint_badput_calculator.CheckpointBadputCalculator(
        checkpoint_badput_calculator.CheckpointLoggerOptions(use_goodput_logger=True)
    )
    calc.entries = logger.entries
    save_stats = calc.calculate_save_operation_checkpoint_manager_blocking_time()
    restore_stats = calc.calculate_restore_operation_checkpoint_manager_blocking_time()

    self.assertGreater(save_stats.total_checkpoint_manager_blocking_time, 0)
    self.assertGreater(restore_stats.total_checkpoint_manager_time, 0)

  def test_no_logger_emits_nothing_and_still_restores(self):
    manager = self._manager(None)
    self.assertIsNone(getattr(manager, "orbax_logger", None))
    self.assertTrue(checkpointing.save_checkpoint(manager, 0, self.state))
    checkpointing.wait_until_finished(manager)

    restored = self._restore(manager)

    self.assertTrue(jnp.array_equal(restored["items"]["w"], self.state["w"]))

  def test_setup_checkpoint_logger_flag_off_returns_none(self):
    config = SimpleNamespace(enable_checkpoint_cloud_logger=False, run_name="run")
    self.assertIsNone(checkpointing.setup_checkpoint_logger(config))

  def test_setup_checkpoint_logger_flag_on_uses_goodput_log(self):
    config = SimpleNamespace(enable_checkpoint_cloud_logger=True, run_name="run")
    with (
        mock.patch.object(checkpointing.ocp_logging, "CloudLogger", create=True) as cloud_logger,
        mock.patch.object(checkpointing.ocp_logging, "CloudLoggerOptions", create=True) as options,
    ):
      logger = checkpointing.setup_checkpoint_logger(config)

    self.assertIs(logger, cloud_logger.return_value)
    options.assert_called_once_with(job_name="run", logger_name="goodput_run")


if __name__ == "__main__":
  unittest.main()
