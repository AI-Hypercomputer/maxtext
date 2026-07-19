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

""" Tests for tokenizer
"""

import unittest

import grain.python as grain
import numpy as np
from maxtext.input_pipeline import grain_tokenizer
from maxtext.input_pipeline import input_pipeline_utils
from numpy.testing import assert_array_equal


class MockTokenizer:
  """
  Mocks a tokenizer by splitting on space and mapping letters to simple ints.
  e.g., "a b c" -> [1, 2, 3]
  """

  def encode(self, text: str) -> list[int]:
    if not text:
      return []
    # Simple 'a'=1, 'b'=2, ... mapping
    return [ord(c) - ord("a") + 1 for c in text.split(" ")]


class TokenizerTransformTest(unittest.TestCase):
  """Tests for chunking, trimming, and padding transformations."""

  def setUp(self):
    self.max_len = 5
    self.pad_length = 7
    self.pad_id = 0
    self.feature_names = "text"
    self.mock_tokenizer = MockTokenizer()
    self.source_data = [{"text": "a b c"}, {"text": "d e f g h i j"}, {"text": ""}, {"text": "k l m n o p q r s t"}]
    self.base_ds = grain.MapDataset.source(self.source_data).to_iter_dataset()

  def test_tokenize_and_trim(self):
    """Tests the 1:1 MapTransform (truncation) logic."""
    trim_op = grain_tokenizer.TokenizeAndTrim(
        feature_names=self.feature_names, sequence_length=self.max_len, tokenizer=self.mock_tokenizer
    )
    trim_ds = self.base_ds.map(trim_op)
    results = list(trim_ds)
    self.assertEqual(len(results), len(self.source_data))
    expected_inputs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7, 8], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([11, 12, 13, 14, 15], dtype=np.int32),
    ]
    result_inputs = [r["text"] for r in results]
    self.assertEqual(len(result_inputs), len(expected_inputs))
    for res, exp in zip(result_inputs, expected_inputs):
      assert_array_equal(res, exp)

  def test_tokenize_and_chunk(self):
    """Tests the 1:N FlatMapTransform (chunking) logic."""
    chunk_op = grain_tokenizer.TokenizeAndChunk(
        feature_names=self.feature_names, sequence_length=self.max_len, tokenizer=self.mock_tokenizer
    )
    chunk_ds = self.base_ds.apply(chunk_op)
    results = list(chunk_ds)
    self.assertEqual(len(results), 5)
    expected_inputs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7, 8], dtype=np.int32),
        np.array([9, 10], dtype=np.int32),
        np.array([11, 12, 13, 14, 15], dtype=np.int32),
        np.array([16, 17, 18, 19, 20], dtype=np.int32),
    ]
    result_inputs = [r["text"] for r in results]
    self.assertEqual(len(result_inputs), len(expected_inputs))
    for res, exp in zip(result_inputs, expected_inputs):
      assert_array_equal(res, exp)

  def test_trim_and_pad_chaining(self):
    """Tests chaining TokenizeAndTrim.map() -> PadOrTrimToMaxLength.map()"""
    trim_op = grain_tokenizer.TokenizeAndTrim(
        feature_names=self.feature_names, sequence_length=self.max_len, tokenizer=self.mock_tokenizer
    )
    pad_op = input_pipeline_utils.PadOrTrimToMaxLength(max_length=self.pad_length, pad_id=self.pad_id)
    chained_ds = self.base_ds.map(trim_op).map(pad_op)
    results = list(chained_ds)
    self.assertEqual(len(results), len(self.source_data))
    expected_inputs = [
        np.array([1, 2, 3, 0, 0, 0, 0], dtype=np.int32),
        np.array([4, 5, 6, 7, 8, 0, 0], dtype=np.int32),
        np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        np.array([11, 12, 13, 14, 15, 0, 0], dtype=np.int32),
    ]
    result_inputs = [r["text"] for r in results]
    self.assertEqual(len(result_inputs), len(expected_inputs))
    for res, exp in zip(result_inputs, expected_inputs):
      assert_array_equal(res, exp)

  def test_chunk_and_pad_chaining(self):
    """Tests chaining TokenizeAndChunk.apply() -> PadOrTrimToMaxLength.map()"""
    chunk_op = grain_tokenizer.TokenizeAndChunk(
        feature_names=self.feature_names, sequence_length=self.max_len, tokenizer=self.mock_tokenizer
    )
    pad_op = input_pipeline_utils.PadOrTrimToMaxLength(max_length=self.pad_length, pad_id=self.pad_id)
    chained_ds = self.base_ds.apply(chunk_op).map(pad_op)
    results = list(chained_ds)
    self.assertEqual(len(results), 5)
    expected_inputs = [
        np.array([1, 2, 3, 0, 0, 0, 0], dtype=np.int32),
        np.array([4, 5, 6, 7, 8, 0, 0], dtype=np.int32),
        np.array([9, 10, 0, 0, 0, 0, 0], dtype=np.int32),
        np.array([11, 12, 13, 14, 15, 0, 0], dtype=np.int32),
        np.array([16, 17, 18, 19, 20, 0, 0], dtype=np.int32),
    ]
    result_inputs = [r["text"] for r in results]
    self.assertEqual(len(result_inputs), len(expected_inputs))
    for res, exp in zip(result_inputs, expected_inputs):
      assert_array_equal(res, exp)


if __name__ == "__main__":
  unittest.main()
