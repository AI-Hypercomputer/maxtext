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
from unittest import mock

import jax
import pathwaysutils
import pytest

from maxtext.utils import elastic_utils
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging


@pytest.mark.cpu_only
class ElasticUtilsTest(unittest.TestCase):
  """Unit tests for Elastic Training utility functions."""

  def test_elastic_enabled(self):
    """Tests elastic_enabled."""
    config = mock.MagicMock()
    config.elastic_enabled = True
    with mock.patch.object(pathwaysutils, "is_pathways_backend_used", return_value=True):
      self.assertTrue(elastic_utils.elastic_enabled(config))

    config.elastic_enabled = False
    with mock.patch.object(pathwaysutils, "is_pathways_backend_used", return_value=True):
      self.assertFalse(elastic_utils.elastic_enabled(config))

    config.elastic_enabled = True
    with mock.patch.object(pathwaysutils, "is_pathways_backend_used", return_value=False):
      self.assertFalse(elastic_utils.elastic_enabled(config))

  @mock.patch.object(gcs_utils, "gcs_list_directories")
  @mock.patch.object(gcs_utils, "gcs_glob_pattern")
  @mock.patch.object(gcs_utils, "gcs_delete_directory")
  @mock.patch.object(max_logging, "log")
  def test_clean_up_checkpoints_no_checkpoints(self, mock_log, mock_delete, mock_glob, mock_list):
    """Tests clean_up_checkpoints when no checkpoints exist."""
    mock_list.return_value = []
    elastic_utils.clean_up_checkpoints("gs://test_bucket/checkpoints")
    mock_log.assert_any_call("Found no existing checkpoints. Continuing")
    mock_delete.assert_not_called()

  @mock.patch.object(gcs_utils, "gcs_list_directories")
  @mock.patch.object(gcs_utils, "gcs_glob_pattern")
  @mock.patch.object(gcs_utils, "gcs_delete_directory")
  @mock.patch.object(max_logging, "log")
  def test_clean_up_checkpoints_incomplete(self, mock_log, mock_delete, mock_glob, mock_list):
    """Tests clean_up_checkpoints when the latest checkpoint is incomplete."""
    mock_list.return_value = ["10", "20", "not_a_step"]
    mock_glob.return_value = []  # No commit_success file
    elastic_utils.clean_up_checkpoints("gs://test_bucket/checkpoints")
    mock_delete.assert_called_with("gs://test_bucket/checkpoints/20/")
    mock_log.assert_any_call("No commit_success file found. Deleting gs://test_bucket/checkpoints/20/...")

  @mock.patch.object(gcs_utils, "gcs_list_directories")
  @mock.patch.object(gcs_utils, "gcs_glob_pattern")
  @mock.patch.object(gcs_utils, "gcs_delete_directory")
  @mock.patch.object(max_logging, "log")
  def test_clean_up_checkpoints_complete(self, mock_log, mock_delete, mock_glob, mock_list):
    """Tests clean_up_checkpoints when the latest checkpoint is complete."""
    mock_list.return_value = ["10", "20"]
    mock_glob.return_value = ["gs://test_bucket/checkpoints/20/commit_success_0"]
    elastic_utils.clean_up_checkpoints("gs://test_bucket/checkpoints")
    mock_delete.assert_not_called()
    mock_log.assert_any_call("Found commit_success file. Keeping gs://test_bucket/checkpoints/20/.")

  @mock.patch.object(pathwaysutils, "is_pathways_backend_used")
  @mock.patch.object(jax, "devices")
  @mock.patch("pathwaysutils.elastic.manager.Manager")
  def test_live_devices_pathways(self, mock_manager, mock_devices, mock_is_pathways):
    """Tests live_devices when pathways is used."""
    mock_is_pathways.return_value = True
    device0 = mock.MagicMock(slice_index=0)
    device1 = mock.MagicMock(slice_index=1)
    mock_devices.return_value = [device0, device1]

    mock_mgr_instance = mock_manager.return_value
    mock_mgr_instance.active_slice_indices = {0}

    # Reset global state for testing
    elastic_utils.elastic_manager = None

    devices = elastic_utils.live_devices()
    self.assertEqual(devices, [device0])

  @mock.patch.object(pathwaysutils, "is_pathways_backend_used")
  @mock.patch.object(jax, "devices")
  def test_live_devices_no_pathways(self, mock_devices, mock_is_pathways):
    """Tests live_devices when pathways is not used."""
    mock_is_pathways.return_value = False
    device0 = mock.MagicMock()
    mock_devices.return_value = [device0]

    devices = elastic_utils.live_devices()
    self.assertEqual(devices, [device0])

  def test_elastic_retry_disabled(self):
    """Tests elastic_retry when disabled."""
    config = mock.MagicMock()
    config.elastic_enabled = False
    decorator = elastic_utils.elastic_retry(config)

    def test_fn(x):
      return x

    self.assertEqual(decorator(test_fn), test_fn)

  @mock.patch.object(pathwaysutils, "is_pathways_backend_used")
  @mock.patch("pathwaysutils.elastic.manager.Manager")
  def test_elastic_retry_enabled(self, mock_manager, mock_is_pathways):
    """Tests elastic_retry when enabled."""
    mock_is_pathways.return_value = True
    config = mock.MagicMock()
    config.elastic_enabled = True
    config.checkpoint_dir = "gs://test_bucket/checkpoints"
    config.elastic_max_retries = 3
    config.elastic_timeout_seconds = 100

    mock_mgr_instance = mock_manager.return_value

    # Reset global state for testing
    elastic_utils.elastic_manager = None

    elastic_utils.elastic_retry(config)

    mock_mgr_instance.elastic_retry.assert_called_once()
    _, kwargs = mock_mgr_instance.elastic_retry.call_args
    self.assertEqual(kwargs["max_retries"], 3)
    self.assertEqual(kwargs["timeout"], 100)
    self.assertTrue(callable(kwargs["on_elastic_event_callback"]))

  @mock.patch.object(pathwaysutils, "is_pathways_backend_used")
  def test_elastic_retry_no_pathways(self, mock_is_pathways):
    """Tests elastic_retry when enabled but pathways is not used."""
    mock_is_pathways.return_value = False
    config = mock.MagicMock()
    config.elastic_enabled = True
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled: True, pathways"
        " backend used: False"
    )

    with self.assertRaisesRegex(ValueError, msg):
      elastic_utils.elastic_retry(config)


if __name__ == "__main__":
  unittest.main()
