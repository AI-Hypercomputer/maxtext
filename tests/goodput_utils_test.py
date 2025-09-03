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

"""Tests for goodput_utils.py"""

import os
import unittest
from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from unittest import mock
from MaxText.utils.goodput_utils import create_goodput_recorder, maybe_monitor_goodput, maybe_record_goodput, GoodputEvent


class GoodputUtilsTest(unittest.TestCase):
  """Tests for Goodput monitoring and recording."""

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        base_output_directory="gs://runner-src/MaxText-logs",
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

  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_step_deviation_uploader")
  @mock.patch("ml_goodput_measurement.monitoring.GoodputMonitor.start_goodput_uploader")
  def test_monitor_goodput(self, mock_start_goodput_uploader, mock_start_step_deviation_uploader):
    mock_start_goodput_uploader.return_value = mock.MagicMock()
    mock_start_step_deviation_uploader.return_value = mock.MagicMock()

    maybe_monitor_goodput(self.config)

    mock_start_goodput_uploader.assert_called()
    mock_start_step_deviation_uploader.assert_called()


if __name__ == "__main__":
  unittest.main()
