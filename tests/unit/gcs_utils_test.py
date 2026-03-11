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

"""Unit tests for GCS utility functions."""

import unittest
from unittest import mock
import os
import tempfile
import pytest

# Module to be tested
from maxtext.utils import gcs_utils


@pytest.mark.cpu_only
class GcsUtilsTest(unittest.TestCase):
  """Unit tests for GCS utility functions."""

  def test_add_trailing_slash(self):
    """Tests the simple add_trailing_slash utility."""
    self.assertEqual(gcs_utils.add_trailing_slash("a/b"), "a/b/")
    self.assertEqual(gcs_utils.add_trailing_slash("a/b/"), "a/b/")

  def test_mkdir_non_existing_dir(self):
    """Tests that a non-existing directory is created and is empty."""
    with tempfile.TemporaryDirectory() as temp_dir:
      new_dir_path = os.path.join(temp_dir, "new_dir")
      self.assertFalse(os.path.exists(new_dir_path))

      # Act
      gcs_utils.mkdir_and_check_permissions(new_dir_path)

      # Assert
      self.assertTrue(os.path.isdir(new_dir_path))
      self.assertEqual(os.listdir(new_dir_path), [])

  def test_mkdir_existing_non_empty_dir(self):
    """Tests that an existing, non-empty directory's contents are unmodified."""
    with tempfile.TemporaryDirectory() as temp_dir:
      existing_dir_path = os.path.join(temp_dir, "existing_dir")
      os.makedirs(existing_dir_path)
      dummy_file_path = os.path.join(existing_dir_path, "dummy.txt")
      with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write("test")

      # Act
      gcs_utils.mkdir_and_check_permissions(existing_dir_path)

      # Assert
      self.assertTrue(os.path.isdir(existing_dir_path))
      self.assertEqual(os.listdir(existing_dir_path), ["dummy.txt"])

  def test_mkdir_existing_read_only_dir(self):
    """Tests that a PermissionError is raised for a read-only directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
      read_only_dir_path = os.path.join(temp_dir, "read_only_dir")
      os.makedirs(read_only_dir_path)
      os.chmod(read_only_dir_path, 0o555)
      gcs_utils.mkdir_and_check_permissions(read_only_dir_path)
      self.assertTrue(os.path.isdir(read_only_dir_path))

  def test_mkdir_read_only_parent_dir(self):
    """Tests that a PermissionError is raised when the parent is read-only."""
    with tempfile.TemporaryDirectory() as temp_dir:
      parent_dir_path = os.path.join(temp_dir, "read_only_parent")
      os.makedirs(parent_dir_path)
      os.chmod(parent_dir_path, 0o555)
      new_dir_path = os.path.join(parent_dir_path, "new_dir")
      gcs_utils.mkdir_and_check_permissions(new_dir_path)
      self.assertFalse(os.path.isdir(new_dir_path))

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_mkdir_gcs_no_such_bucket(self, mock_storage_client):
    """Tests that an exception is raised for a non-existent GCS bucket."""
    mock_client_instance = mock_storage_client.return_value
    mock_client_instance.get_bucket.side_effect = Exception("Bucket not found!")
    gcs_path = "gs://no_such_bucket"

    with self.assertRaises(FileNotFoundError):
      gcs_utils.mkdir_and_check_permissions(gcs_path)
    mock_client_instance.get_bucket.assert_called_with("no_such_bucket")

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_mkdir_gcs_no_such_bucket_with_path(self, mock_storage_client):
    """Tests an exception for a non-existent bucket with a subdirectory."""
    mock_client_instance = mock_storage_client.return_value
    mock_client_instance.get_bucket.side_effect = Exception("Bucket not found!")
    gcs_path = "gs://no_such_bucket/some/dir"

    with self.assertRaises(FileNotFoundError):
      gcs_utils.mkdir_and_check_permissions(gcs_path)
    mock_client_instance.get_bucket.assert_called_with("no_such_bucket")

  @mock.patch("maxtext.utils.gcs_utils.epath.Path")
  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_mkdir_gcs_valid_bucket(self, mock_storage_client, mock_epath):
    """Tests that a valid GCS path is handled correctly without errors."""
    # Arrange: Mock the GCS client to simulate a valid bucket
    mock_client_instance = mock_storage_client.return_value

    # Arrange: Mock epath to prevent real GCS calls
    mock_path_instance = mock.MagicMock()
    mock_path_instance.as_posix.return_value = "gs://valid_bucket/some/dir"
    mock_path_instance.parts = ["gs:", "", "valid_bucket", "some", "dir"]
    mock_path_instance.exists.return_value = True

    mock_temp_file_instance = mock.MagicMock()
    mock_path_instance.__truediv__.return_value = mock_temp_file_instance

    mock_epath.return_value = mock_path_instance
    gcs_path = "gs://valid_bucket/some/dir"

    # Act
    gcs_utils.mkdir_and_check_permissions(gcs_path)

    # Assert
    mock_client_instance.get_bucket.assert_called_with("valid_bucket")
    mock_path_instance.mkdir.assert_called_with(exist_ok=True, parents=True)
    mock_path_instance.exists.assert_called_once()
    mock_temp_file_instance.write_text.assert_called_once_with("test")
    mock_temp_file_instance.unlink.assert_called_once()
