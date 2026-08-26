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

import unittest
from unittest import mock

import pytest

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

  def _make_elastic_config(self, run_name):
    return pyconfig.initialize(
        [None, get_test_config_path()],
        base_output_directory=get_test_base_output_directory(),
        run_name=run_name,
        enable_checkpointing=False,
        enable_goodput_recording=True,
        elastic_enabled=True,
        enable_single_controller=True,
    )

  @mock.patch("google.cloud.logging.Client")
  def test_create_goodput_recorder_elastic(self, mock_cloud_logger):
    """create_goodput_recorder builds an ElasticGoodputRecorder and seeds slice state."""
    mock_cloud_logger.return_value = mock.MagicMock()
    elastic_config = self._make_elastic_config("runner_test_elastic")

    with (
        mock.patch(
            "maxtext.utils.elastic_utils.pathwaysutils.is_pathways_backend_used",
            return_value=True,
        ),
        mock.patch("maxtext.utils.elastic_utils.record_slice_state") as mock_seed,
    ):
      recorder = create_goodput_recorder(elastic_config)

    from ml_goodput_measurement import goodput_elastic  # pylint: disable=g-import-not-at-top

    self.assertIsInstance(recorder, goodput_elastic.ElasticGoodputRecorder)
    mock_seed.assert_called_once_with(recorder)

  @mock.patch("google.cloud.logging.Client")
  def test_create_goodput_recorder_elastic_fallback_on_error(self, mock_cloud_logger):
    """Falls back to the base GoodputRecorder if elastic recorder construction raises."""
    mock_cloud_logger.return_value = mock.MagicMock()
    elastic_config = self._make_elastic_config("runner_test_elastic_fallback")

    with (
        mock.patch(
            "maxtext.utils.elastic_utils.pathwaysutils.is_pathways_backend_used",
            return_value=True,
        ),
        mock.patch(
            "ml_goodput_measurement.goodput_elastic.ElasticGoodputRecorder",
            side_effect=RuntimeError("boom"),
        ),
    ):
      recorder = create_goodput_recorder(elastic_config)

    from ml_goodput_measurement import goodput as base_goodput  # pylint: disable=g-import-not-at-top

    self.assertIs(type(recorder), base_goodput.GoodputRecorder)

  def test_create_goodput_recorder_not_elastic_when_backend_absent(self):
    """elastic_enabled=True alone isn't enough: falls back without the Pathways backend."""
    elastic_config = self._make_elastic_config("runner_test_elastic_no_pathways")

    with mock.patch("google.cloud.logging.Client") as mock_cloud_logger:
      mock_cloud_logger.return_value = mock.MagicMock()
      recorder = create_goodput_recorder(elastic_config)

    from ml_goodput_measurement import goodput as base_goodput  # pylint: disable=g-import-not-at-top

    self.assertIs(type(recorder), base_goodput.GoodputRecorder)

  def test_construct_goodput_monitor_elastic(self):
    """_construct_goodput_monitor prefers ElasticGoodputMonitor when elastic training applies."""
    elastic_config = self._make_elastic_config("runner_test_monitor_elastic")
    common_kwargs = {"job_name": "test"}

    with (
        mock.patch("maxtext.utils.elastic_utils.should_use_elastic", return_value=True),
        mock.patch("ml_goodput_measurement.monitoring_elastic.ElasticGoodputMonitor") as mock_elastic_monitor,
    ):
      mock_elastic_monitor.return_value = mock.MagicMock()
      monitor = _construct_goodput_monitor(elastic_config, common_kwargs)

    mock_elastic_monitor.assert_called_once_with(include_slice_efficiency=True, **common_kwargs)
    self.assertIs(monitor, mock_elastic_monitor.return_value)

  def test_construct_goodput_monitor_non_elastic(self):
    """_construct_goodput_monitor uses the base GoodputMonitor when elastic training is off."""
    common_kwargs = {"job_name": "test"}

    with mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor:
      mock_monitor.return_value = mock.MagicMock()
      monitor = _construct_goodput_monitor(self.config, common_kwargs)

    mock_monitor.assert_called_once_with(pathway_enabled=self.config.enable_pathways_goodput, **common_kwargs)
    self.assertIs(monitor, mock_monitor.return_value)

  def test_construct_goodput_monitor_elastic_fallback_on_error(self):
    """Falls back to the base GoodputMonitor if elastic monitor construction raises."""
    elastic_config = self._make_elastic_config("runner_test_monitor_elastic_fallback")
    common_kwargs = {"job_name": "test"}

    with (
        mock.patch("maxtext.utils.elastic_utils.should_use_elastic", return_value=True),
        mock.patch(
            "ml_goodput_measurement.monitoring_elastic.ElasticGoodputMonitor",
            side_effect=RuntimeError("boom"),
        ),
        mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor") as mock_monitor,
    ):
      mock_monitor.return_value = mock.MagicMock()
      monitor = _construct_goodput_monitor(elastic_config, common_kwargs)

    mock_monitor.assert_called_once_with(pathway_enabled=elastic_config.enable_pathways_goodput, **common_kwargs)
    self.assertIs(monitor, mock_monitor.return_value)


if __name__ == "__main__":
  unittest.main()
