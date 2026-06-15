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

"""Unit tests for ManagedMLDiagnostics."""

import unittest
from unittest import mock

from maxtext.common import managed_mldiagnostics
from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
import pytest


@pytest.mark.cpu_only
class ManagedMLDiagnosticsTest(unittest.TestCase):
  # pylint: disable=protected-access

  def setUp(self):
    super().setUp()
    # Reset singleton instance between tests
    ManagedMLDiagnostics._instance = None

  def test_not_enabled_noop(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = False

    with mock.patch.object(managed_mldiagnostics.mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      mock_run.assert_not_called()

  def test_enabled_empty_region_passes_none(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = ""
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = "test_group"
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.get_keys.return_value = {"key1": "val1"}

    with mock.patch.object(managed_mldiagnostics.mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      mock_run.assert_called_once_with(
          name="test_run",
          run_group="test_group",
          configs={"key1": "val1"},
          gcs_path="gs://test_dir",
          region=None,
      )

  def test_enabled_populated_region_passes_region(self):
    mock_config = mock.MagicMock()
    mock_config.managed_mldiagnostics = True
    mock_config.managed_mldiagnostics_region = "us-east1"
    mock_config.run_name = "test_run"
    mock_config.managed_mldiagnostics_run_group = "test_group"
    mock_config.managed_mldiagnostics_dir = "gs://test_dir"
    mock_config.get_keys.return_value = {"key1": "val1"}

    with mock.patch.object(managed_mldiagnostics.mldiag, "machinelearning_run") as mock_run:
      ManagedMLDiagnostics(mock_config)
      mock_run.assert_called_once_with(
          name="test_run",
          run_group="test_group",
          configs={"key1": "val1"},
          gcs_path="gs://test_dir",
          region="us-east1",
      )


if __name__ == "__main__":
  unittest.main()
