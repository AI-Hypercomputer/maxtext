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
import yaml

# Module to be tested
from maxtext.utils import gcs_utils


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
    # Ideally, we would create a temporary directory here and mark it read-only. Unfortunately it does not work when the
    # tests run inside a GitHub action. I think that those tests are executed as sudo and they ignore the permissions.
    # Instead we use "/sys", which is a universally read-only directory, even for superusers.
    read_only_dir_path = "/sys"
    with self.assertRaises((PermissionError, OSError)):
      gcs_utils.mkdir_and_check_permissions(read_only_dir_path)

  def test_mkdir_read_only_parent_dir(self):
    """Tests that a PermissionError is raised when the parent is read-only."""
    # Ideally, we would create a temporary directory here and mark it read-only. Unfortunately it does not work when the
    # tests run inside a GitHub action. I think that those tests are executed as sudo and they ignore the permissions.
    # Instead we use "/sys", which is a universally read-only directory, even for superusers.
    new_dir_path = "/sys/new_dir"
    with self.assertRaises((PermissionError, OSError)):
      gcs_utils.mkdir_and_check_permissions(new_dir_path)

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
    # `exists()` is meaningless for GCS prefixes, so it must not gate the result.
    mock_path_instance.exists.assert_not_called()
    mock_temp_file_instance.write_text.assert_called_once_with("test")
    mock_temp_file_instance.unlink.assert_called_once()

  @mock.patch("maxtext.utils.gcs_utils.epath.Path")
  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_mkdir_gcs_writable_when_mkdir_is_noop(self, mock_storage_client, mock_epath):
    """Tests that a writable GCS path succeeds even when `mkdir` leaves nothing behind.

    Regression test: the `gcsfs` backend (used whenever TensorFlow is not installed) is a no-op when
    asked to create a sub-path, because GCS has no real directories. The previous `exists()` check
    then raised a spurious PermissionError on paths the caller could write to just fine.
    """
    # Arrange: Mock the GCS client to simulate a valid bucket
    mock_client_instance = mock_storage_client.return_value

    # Arrange: simulate the gcsfs backend, where `mkdir` does nothing and `exists()` stays False.
    mock_path_instance = mock.MagicMock()
    mock_path_instance.as_posix.return_value = "gs://valid_bucket/some/dir"
    mock_path_instance.parts = ["gs:", "", "valid_bucket", "some", "dir"]
    mock_path_instance.exists.return_value = False

    # The temp file write succeeds, i.e. the path really is writable.
    mock_temp_file_instance = mock.MagicMock()
    mock_path_instance.__truediv__.return_value = mock_temp_file_instance

    mock_epath.return_value = mock_path_instance
    gcs_path = "gs://valid_bucket/some/dir"

    # Act: must not raise.
    result = gcs_utils.mkdir_and_check_permissions(gcs_path)

    # Assert: the real write test is what determines success.
    self.assertIs(result, mock_path_instance)
    mock_client_instance.get_bucket.assert_called_with("valid_bucket")
    mock_temp_file_instance.write_text.assert_called_once_with("test")
    mock_temp_file_instance.unlink.assert_called_once()

  @mock.patch("maxtext.utils.gcs_utils.epath.Path")
  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_mkdir_gcs_not_writable(self, mock_storage_client, mock_epath):
    """Tests that a PermissionError is raised when the GCS path cannot be written to."""
    del mock_storage_client  # Bucket lookup succeeds; only the write test should fail.

    mock_path_instance = mock.MagicMock()
    mock_path_instance.as_posix.return_value = "gs://valid_bucket/some/dir"
    mock_path_instance.parts = ["gs:", "", "valid_bucket", "some", "dir"]
    mock_path_instance.exists.return_value = False

    # Arrange: writing the temp file fails, which is a genuine permission problem.
    mock_temp_file_instance = mock.MagicMock()
    mock_temp_file_instance.write_text.side_effect = Exception("403 Forbidden")
    mock_path_instance.__truediv__.return_value = mock_temp_file_instance

    mock_epath.return_value = mock_path_instance

    with self.assertRaises(PermissionError):
      gcs_utils.mkdir_and_check_permissions("gs://valid_bucket/some/dir")

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_gcs_glob_pattern_lists_only_under_literal_prefix(self, mock_storage_client):
    """Tests that the listing starts at the text before the first glob character."""
    mock_list_blobs = mock_storage_client.return_value.list_blobs
    expected_prefixes = {
        "ckpt/20/commit_success*": "ckpt/20/commit_success",
        "ckpt/20/commit_succes?": "ckpt/20/commit_succes",
        "ckpt/20/commit_succes[s]": "ckpt/20/commit_succes",
        "ckpt/20/{commit_success,other}": "ckpt/20/",
        "ckpt/20/commit\\*success": "ckpt/20/commit",
        "ckpt/*/commit_success*": "ckpt/",
        "ckpt/20/commit_success.txt": "ckpt/20/commit_success.txt",
    }

    for glob_pattern, expected_prefix in expected_prefixes.items():
      with self.subTest(glob_pattern=glob_pattern):
        gcs_utils.gcs_glob_pattern(f"gs://bucket/{glob_pattern}")

        mock_list_blobs.assert_called_with("bucket", prefix=expected_prefix, match_glob=glob_pattern)

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_gcs_glob_pattern_without_literal_prefix(self, mock_storage_client):
    """Tests that a pattern starting with a glob character lists the whole bucket."""
    mock_list_blobs = mock_storage_client.return_value.list_blobs

    gcs_utils.gcs_glob_pattern("gs://bucket/*.array_record")

    mock_list_blobs.assert_called_once_with("bucket", prefix=None, match_glob="*.array_record")

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  def test_gcs_glob_pattern_returns_full_gcs_paths(self, mock_storage_client):
    """Tests that the matches are returned as full `gs://` paths."""
    blob = mock.MagicMock()
    blob.name = "ckpt/20/commit_success.txt"
    mock_storage_client.return_value.list_blobs.return_value = [blob]

    data_files = gcs_utils.gcs_glob_pattern("gs://bucket/ckpt/20/commit_success*")

    self.assertEqual(data_files, ["gs://bucket/ckpt/20/commit_success.txt"])

  def test_write_config_raw_keys_for_gcs_skips_secret_keys(self):
    """Tests that secret keys in KEYS_NO_LOGGING are omitted from config.yml."""
    secret = "hf_gcs_secret_456"
    raw_keys = {
        "save_config_to_gcs": True,
        "base_output_directory": "gs://bucket/out",
        "run_name": "run",
        "steps": 100,
        "hf_access_token": secret,
    }
    uploaded_contents = []

    def _capture_upload(dest_gcs_name, source_file_name):
      del dest_gcs_name
      with open(source_file_name, "rt", encoding="utf8") as f:
        uploaded_contents.append(f.read())

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        mock.patch("jax.process_index", return_value=0),
        mock.patch.object(gcs_utils, "_gcs_guard", return_value=True),
        mock.patch.object(gcs_utils, "upload_blob", side_effect=_capture_upload) as mock_upload_blob,
    ):
      prev_cwd = os.getcwd()
      os.chdir(tmp_dir)
      try:
        gcs_utils.write_config_raw_keys_for_gcs(raw_keys)
      finally:
        os.chdir(prev_cwd)

    mock_upload_blob.assert_called_once_with("gs://bucket/out/run/config.yml", "config.yml")
    self.assertEqual(len(uploaded_contents), 1)
    written = uploaded_contents[0]
    parsed = yaml.safe_load(written)
    self.assertEqual(parsed["steps"], 100)
    self.assertNotIn("hf_access_token", parsed)
    self.assertNotIn("hf_access_token", written)
    self.assertNotIn(secret, written)

  def _populate_dump_dir(self, local_dir):
    os.makedirs(os.path.join(local_dir, "nested"), exist_ok=True)
    with open(os.path.join(local_dir, "module_0001.jit_train_step.txt"), "w", encoding="utf-8") as f:
      f.write("hlo-train-step")
    with open(os.path.join(local_dir, "module_0002.jit_other.txt"), "w", encoding="utf-8") as f:
      f.write("hlo-other")
    with open(os.path.join(local_dir, "nested", "sub.jit_train_step.txt"), "w", encoding="utf-8") as f:
      f.write("hlo-nested")

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_local_target_copies_matching_and_cleans_staging(self, _mock_proc, mock_storage_client):
    """Non-GCS target copies matching files without invoking GCS and preserves final artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "staging")
      target_dir = os.path.join(tmp_dir, "mounted_out", "run1", "xla_dump")
      self._populate_dump_dir(local_dir)

      gcs_utils.upload_dump(
          local_dir,
          target_dir,
          module_name="jit_train_step",
          delete_local_after=True,
          all_host_upload=False,
      )

      mock_storage_client.assert_not_called()
      self.assertFalse(os.path.exists(local_dir))
      self.assertTrue(os.path.isfile(os.path.join(target_dir, "module_0001.jit_train_step.txt")))
      self.assertTrue(os.path.isfile(os.path.join(target_dir, "nested", "sub.jit_train_step.txt")))
      self.assertFalse(os.path.exists(os.path.join(target_dir, "module_0002.jit_other.txt")))
      with open(os.path.join(target_dir, "module_0001.jit_train_step.txt"), encoding="utf-8") as f:
        self.assertEqual(f.read(), "hlo-train-step")

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_local_target_keeps_local_when_delete_false(self, _mock_proc, mock_storage_client):
    """When delete_local_after=False, both staging local_dir and target_dir retain artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "staging")
      target_dir = os.path.join(tmp_dir, "mounted_out")
      self._populate_dump_dir(local_dir)

      gcs_utils.upload_dump(
          local_dir,
          target_dir,
          module_name="",
          delete_local_after=False,
          all_host_upload=False,
      )

      mock_storage_client.assert_not_called()
      self.assertTrue(os.path.isfile(os.path.join(local_dir, "module_0001.jit_train_step.txt")))
      self.assertTrue(os.path.isfile(os.path.join(target_dir, "module_0001.jit_train_step.txt")))
      self.assertTrue(os.path.isfile(os.path.join(target_dir, "module_0002.jit_other.txt")))

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_local_same_path_and_symlink_not_deleted(self, _mock_proc, mock_storage_client):
    """When target_dir equals local_dir (including via symlink), files are kept in place and not deleted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "same_dump")
      symlink_target = os.path.join(tmp_dir, "link_dump")
      self._populate_dump_dir(local_dir)
      os.symlink(local_dir, symlink_target)

      for target_dir in (local_dir, local_dir + os.sep, symlink_target):
        with self.subTest(target_dir=target_dir):
          gcs_utils.upload_dump(
              local_dir,
              target_dir,
              module_name="jit_train_step",
              delete_local_after=True,
              all_host_upload=False,
          )
          self.assertTrue(os.path.isfile(os.path.join(local_dir, "module_0001.jit_train_step.txt")))
          self.assertTrue(os.path.isfile(os.path.join(local_dir, "module_0002.jit_other.txt")))

      mock_storage_client.assert_not_called()

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_local_target_inside_local_dir_not_deleted(self, _mock_proc, mock_storage_client):
    """When target_dir is nested inside local_dir, files are copied safely and local_dir is not rmtreed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "dump_root")
      target_dir = os.path.join(local_dir, "saved_sub")
      self._populate_dump_dir(local_dir)

      gcs_utils.upload_dump(
          local_dir,
          target_dir,
          module_name="jit_train_step",
          delete_local_after=True,
          all_host_upload=False,
      )

      mock_storage_client.assert_not_called()
      self.assertTrue(os.path.isdir(local_dir))
      self.assertTrue(os.path.isfile(os.path.join(target_dir, "module_0001.jit_train_step.txt")))
      self.assertFalse(os.path.exists(os.path.join(target_dir, "saved_sub")))

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("socket.gethostname", return_value="worker-host-1")
  def test_upload_dump_local_multihost_all_host_upload_toggle(self, _mock_host, mock_storage_client):
    """Non-zero process skips when all_host_upload=False, and writes under hostname subdir when True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "staging")
      target_dir = os.path.join(tmp_dir, "shared_mount")
      self._populate_dump_dir(local_dir)

      with mock.patch("jax.process_index", return_value=1):
        gcs_utils.upload_dump(
            local_dir,
            target_dir,
            module_name="jit_train_step",
            delete_local_after=True,
            all_host_upload=False,
        )
        self.assertFalse(os.path.exists(target_dir))
        self.assertTrue(os.path.isdir(local_dir))

        gcs_utils.upload_dump(
            local_dir,
            target_dir,
            module_name="jit_train_step",
            delete_local_after=True,
            all_host_upload=True,
        )

      mock_storage_client.assert_not_called()
      self.assertFalse(os.path.exists(local_dir))
      self.assertTrue(os.path.isfile(os.path.join(target_dir, "worker-host-1", "module_0001.jit_train_step.txt")))

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_local_works_when_gcs_decoupled(self, _mock_proc, mock_storage_client):
    """Non-GCS target still copies artifacts even when GCS storage is unavailable / decoupled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "staging")
      target_dir = os.path.join(tmp_dir, "local_out")
      self._populate_dump_dir(local_dir)

      with mock.patch.object(gcs_utils, "_gcs_guard", return_value=False) as mock_guard:
        gcs_utils.upload_dump(
            local_dir,
            target_dir,
            module_name="jit_train_step",
            delete_local_after=True,
            all_host_upload=False,
        )

      mock_guard.assert_not_called()
      mock_storage_client.assert_not_called()
      self.assertTrue(os.path.isfile(os.path.join(target_dir, "module_0001.jit_train_step.txt")))

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("socket.gethostname", return_value="tpu-host-0")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_gcs_target_regression(self, _mock_proc, _mock_host, mock_storage_client):
    """Regression test: gs:// target preserves GCS upload, filtering, host subdir, and delete_local_after."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "staging")
      self._populate_dump_dir(local_dir)

      mock_bucket = mock_storage_client.return_value.get_bucket.return_value
      uploaded = {}

      def _fake_blob(blob_name):
        blob = mock.MagicMock()
        blob.upload_from_filename.side_effect = lambda src: uploaded.__setitem__(blob_name, src)
        return blob

      mock_bucket.blob.side_effect = _fake_blob

      with mock.patch.object(gcs_utils, "_gcs_guard", return_value=True):
        gcs_utils.upload_dump(
            local_dir,
            "gs://my-bucket/out/run/xla_dump/",
            module_name="jit_train_step",
            delete_local_after=True,
            all_host_upload=True,
        )

      mock_storage_client.return_value.get_bucket.assert_called_once_with("my-bucket")
      self.assertEqual(
          set(uploaded.keys()),
          {
              "out/run/xla_dump/tpu-host-0/module_0001.jit_train_step.txt",
              "out/run/xla_dump/tpu-host-0/nested/sub.jit_train_step.txt",
          },
      )
      self.assertFalse(os.path.exists(local_dir))

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_local_jaxpr_file_filter_and_preservation(self, _mock_proc, mock_storage_client):
    """Simulates maybe_dump_jaxpr saving train_step.jaxpr to a non-GCS base_output_directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "jaxpr_staging")
      target_dir = os.path.join(tmp_dir, "mounted_out", "run1", "jaxpr_dump")
      os.makedirs(local_dir, exist_ok=True)
      with open(os.path.join(local_dir, "train_step.jaxpr"), "w", encoding="utf-8") as f:
        f.write("{ lambda ; a:f32[2,3]. let b:f32[2,3] = add a 1.0 in (b,) }")

      gcs_utils.upload_dump(
          local_dir,
          target_dir,
          module_name="train_step.jaxpr",
          delete_local_after=True,
          all_host_upload=False,
      )

      mock_storage_client.assert_not_called()
      self.assertFalse(os.path.exists(local_dir))
      dumped = os.path.join(target_dir, "train_step.jaxpr")
      self.assertTrue(os.path.isfile(dumped))
      with open(dumped, encoding="utf-8") as f:
        self.assertIn("add a 1.0", f.read())

  @mock.patch("maxtext.utils.gcs_utils.storage.Client")
  @mock.patch("jax.process_index", return_value=0)
  def test_upload_dump_empty_target_dir_is_noop(self, _mock_proc, mock_storage_client):
    """Empty target_dir returns without touching GCS or deleting local_dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      local_dir = os.path.join(tmp_dir, "staging")
      self._populate_dump_dir(local_dir)

      gcs_utils.upload_dump(local_dir, "", module_name="jit_train_step", delete_local_after=True)

      mock_storage_client.assert_not_called()
      self.assertTrue(os.path.isfile(os.path.join(local_dir, "module_0001.jit_train_step.txt")))
