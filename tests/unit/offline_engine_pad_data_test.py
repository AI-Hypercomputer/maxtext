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

"""Tests for OfflineEngine.pad_data."""

import unittest
from unittest import mock

import numpy as np
import pytest

from maxtext.common.gcloud_stub import is_decoupled

pytestmark = [pytest.mark.cpu_only, pytest.mark.external_serving]

# offline_engine raises on import when decoupled and jetstream is stubbed.
if not is_decoupled():
  from maxtext.inference.offline_engine import InputData, OfflineEngine
else:
  InputData = OfflineEngine = None  # Never used: the external_serving marker deselects these tests.


class PadDataTest(unittest.TestCase):
  """OfflineEngine.pad_data pads each prompt to a prefill length.
  Command: pytest tests/unit/offline_engine_pad_data_test.py
  """

  def setUp(self):
    super().setUp()
    self.engine = OfflineEngine.__new__(OfflineEngine)  # pad_data reads only the prefill lengths
    self.engine.prefill_lengths = [16, 64]
    self.engine.max_prefill_length = 64

  def test_a_prompt_longer_than_the_largest_prefill_length_keeps_the_length_it_was_cut_to(self):
    tokens = np.arange(1, 66, dtype=np.int32)
    with mock.patch("maxtext.inference.offline_engine.max_logging.log") as log:
      padded = self.engine.pad_data([InputData(id="long", tokens=tokens, true_length=65)])[0]
    self.assertEqual(padded.true_length, 64)
    np.testing.assert_array_equal(padded.tokens, tokens[:64])
    log.assert_called_once()
    self.assertIn("only its first 64 tokens are used", log.call_args.args[0])

  def test_a_prompt_that_fits_is_padded_and_keeps_its_length(self):
    tokens = np.arange(1, 11, dtype=np.int32)
    with mock.patch("maxtext.inference.offline_engine.max_logging.log") as log:
      padded = self.engine.pad_data([InputData(id="short", tokens=tokens, true_length=10)])[0]
    self.assertEqual(padded.true_length, 10)
    self.assertEqual(len(padded.tokens), 16)
    np.testing.assert_array_equal(padded.tokens[:10], tokens)
    np.testing.assert_array_equal(padded.tokens[10:], np.zeros(6, np.int32))
    log.assert_not_called()

  def test_a_prompt_of_exactly_the_largest_length_is_unchanged(self):
    tokens = np.arange(1, 65, dtype=np.int32)
    padded = self.engine.pad_data([InputData(id="exact", tokens=tokens, true_length=64)])[0]
    self.assertEqual(padded.true_length, 64)
    np.testing.assert_array_equal(padded.tokens, tokens)


if __name__ == "__main__":
  unittest.main()
