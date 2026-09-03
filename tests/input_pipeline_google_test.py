# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Google-specific input pipeline and filesystem patches."""

import glob
import pathlib
from unittest import mock

from absl.testing import absltest
# Placeholder: internal
from maxtext.input_pipeline import grain_data_processing

from google3.pyglib import gfile


class InputPipelineGoogleTest(absltest.TestCase):

  def test_patched_open_local_file(self):
    temp_dir = self.create_tempdir()
    test_file = temp_dir.create_file("sample.txt", "hello world")
    with maxtext_google.patched_open(test_file.full_path, "r") as f:
      content = f.read()
    self.assertEqual(content, "hello world")

  def test_patched_open_cns_file(self):
    with mock.patch.object(gfile, "Open") as mock_open:
      mock_open.return_value = mock.MagicMock()
      maxtext_google.patched_open("/cns/test-cell/home/data.txt", "r")
      mock_open.assert_called_once_with(
          "/cns/test-cell/home/data.txt", "r"
      )

  def test_patched_glob_local_files(self):
    temp_dir = self.create_tempdir()
    temp_dir.create_file("a.txt", "a")
    temp_dir.create_file("b.txt", "b")
    pattern = f"{temp_dir.full_path}/*.txt"
    files = maxtext_google.patched_glob(pattern)
    self.assertLen(files, 2)

  def test_patched_glob_cns_files(self):
    with mock.patch.object(gfile, "Glob") as mock_glob:
      mock_glob.return_value = [
          "/cns/test-cell/home/data_0.arrayrecord",
          "/cns/test-cell/home/data_1.arrayrecord",
      ]
      files = maxtext_google.patched_glob(
          "/cns/test-cell/home/*.arrayrecord"
      )
      mock_glob.assert_called_once_with(
          "/cns/test-cell/home/*.arrayrecord"
      )
      self.assertEqual(
          files,
          [
              "/cns/test-cell/home/data_0.arrayrecord",
              "/cns/test-cell/home/data_1.arrayrecord",
          ],
      )

  def test_patched_iglob_cns_files(self):
    with mock.patch.object(gfile, "Glob") as mock_glob:
      mock_glob.return_value = [
          "/cns/test-cell/home/data_0.arrayrecord",
      ]
      iterator = maxtext_google.patched_iglob(
          "/cns/test-cell/home/*.arrayrecord"
      )
      files = list(iterator)
      mock_glob.assert_called_once_with(
          "/cns/test-cell/home/*.arrayrecord"
      )
      self.assertEqual(
          files,
          ["/cns/test-cell/home/data_0.arrayrecord"],
      )

  def test_patched_open_cns_pathlib_path(self):
    with mock.patch.object(gfile, "Open") as mock_open:
      mock_open.return_value = mock.MagicMock()
      maxtext_google.patched_open(
          pathlib.Path("/cns/test-cell/home/data.txt"), "r"
      )
      mock_open.assert_called_once_with(
          "/cns/test-cell/home/data.txt", "r"
      )

  def test_patched_glob_cns_pathlib_path(self):
    with mock.patch.object(gfile, "Glob") as mock_glob:
      mock_glob.return_value = [
          "/cns/test-cell/home/data_0.arrayrecord",
      ]
      files = maxtext_google.patched_glob(
          pathlib.Path("/cns/test-cell/home/*.arrayrecord")
      )
      mock_glob.assert_called_once_with(
          "/cns/test-cell/home/*.arrayrecord"
      )
      self.assertEqual(
          files,
          ["/cns/test-cell/home/data_0.arrayrecord"],
      )

  def test_find_data_files_with_cns_and_patched_glob(self):
    import tensorflow as tf
    with mock.patch.object(gfile, "Glob") as mock_glob:
      mock_glob.return_value = [
          "/cns/test-cell/home/train-00000.parquet",
          "/cns/test-cell/home/train-00001.parquet",
      ]
      with mock.patch.object(
          glob, "glob", side_effect=maxtext_google.patched_glob
      ), mock.patch.object(tf.io.gfile, "glob", side_effect=maxtext_google.patched_glob):
        files = grain_data_processing.find_data_files(
            "/cns/test-cell/home/*.parquet"
        )
      mock_glob.assert_called_once_with(
          "/cns/test-cell/home/*.parquet"
      )
      self.assertEqual(
          files,
          [
              "/cns/test-cell/home/train-00000.parquet",
              "/cns/test-cell/home/train-00001.parquet",
          ],
      )


if __name__ == "__main__":
  absltest.main()
