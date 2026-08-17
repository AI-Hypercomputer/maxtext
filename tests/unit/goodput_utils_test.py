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

import tempfile
import unittest
from unittest import mock

import pytest

goodput = pytest.importorskip("ml_goodput_measurement.goodput")
goodput_elastic = pytest.importorskip("ml_goodput_measurement.goodput_elastic")
monitoring = pytest.importorskip("ml_goodput_measurement.monitoring")
monitoring_elastic = pytest.importorskip("ml_goodput_measurement.monitoring_elastic")

from maxtext.configs import pyconfig
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    _construct_goodput_monitor,
    create_goodput_recorder,
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

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_goodput(self, mock_start_goodput_uploader, mock_stop_goodput_uploader):
    mock_start_goodput_uploader.return_value = mock.MagicMock()

    with maybe_monitor_goodput(self.config):
      mock_start_goodput_uploader.assert_called()
    mock_stop_goodput_uploader.assert_called()

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

  @mock.patch("ml_goodput_measurement.monitoring_elastic.ElasticGoodputMonitor.stop_goodput_uploader")
  @mock.patch("ml_goodput_measurement.monitoring_elastic.ElasticGoodputMonitor.start_goodput_uploader")
  @mock.patch("maxtext.utils.elastic_utils.should_use_elastic")
  def test_monitor_goodput_elastic(
      self, mock_should_use_elastic, mock_start_goodput_uploader, mock_stop_goodput_uploader
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


if __name__ == "__main__":
  unittest.main()
