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

"""Smoke tests for train_tokenizer file format support."""

import os
import unittest
import pytest

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.trainers.tokenizer import train_tokenizer
from tests.utils.test_helpers import get_test_dataset_path


class TrainTokenizerFormatTest(unittest.TestCase):
  """Smoke-tests that train_tokenizer runs end-to-end for each supported file format."""

  def _run_format_test(self, file_pattern, file_type):
    """Uses a tiny corpus; the resulting tokenizer is not stored — only verify
    it can be loaded and used for encode/decode.
    """
    output_path = os.path.join("tests", f"test_tokenizer_{file_type}")
    try:
      dataset_iter = train_tokenizer.build_grain_iterator(file_pattern, file_type)
      train_tokenizer.train_tokenizer(
          dataset_iter,
          vocab_path=output_path,
          vocab_size=512,
          max_corpus_chars=10_000,
      )
      tok = input_pipeline_utils.get_tokenizer(output_path, "sentencepiece", add_bos=False, add_eos=False)
      text = "This is a test"
      tokens = tok.encode(text)
      self.assertGreater(len(tokens), 0)
      self.assertEqual(tok.decode(tokens), text)
    finally:
      if os.path.exists(output_path):
        os.remove(output_path)

  @pytest.mark.cpu_only
  def test_parquet(self):
    path = os.path.join(get_test_dataset_path(), "hf", "c4", "c4-train-00000-of-01637.parquet")
    self._run_format_test(path, "parquet")

  @pytest.mark.cpu_only
  def test_arrayrecord(self):
    dataset_root = get_test_dataset_path()
    if is_decoupled():
      path = os.path.join(dataset_root, "c4", "en", "3.0.1", "c4-train.array_record-00000-of-00008")
    else:
      path = os.path.join(dataset_root, "array-record", "c4", "en", "3.0.1", "c4-train.array_record-00000-of-01024")
    self._run_format_test(path, "arrayrecord")

  @pytest.mark.cpu_only
  def test_tfrecord(self):
    dataset_root = get_test_dataset_path()
    if is_decoupled():
      path = os.path.join(dataset_root, "c4", "en", "3.0.1", "__local_c4_builder-train.tfrecord-00000-of-00008")
    else:
      path = os.path.join(dataset_root, "c4", "en", "3.0.1", "c4-train.tfrecord-00000-of-01024")
    self._run_format_test(path, "tfrecord")


if __name__ == "__main__":
  unittest.main()
