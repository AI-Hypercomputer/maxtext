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
"""Tests that MegatronSplitInputsTargets passes dataset_id through as [S] when enabled."""

import unittest

import numpy as np

from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets


class MegatronSplitDatasetIdTest(unittest.TestCase):

  def _element(self, with_id=True):
    el = {"text": np.arange(9, dtype=np.int32)}  # seq_len = 8 after split
    if with_id:
      el["dataset_id"] = np.int32(3)
    return el

  def test_emits_per_token_dataset_id_when_enabled(self):
    t = MegatronSplitInputsTargets(eod_id=0, emit_dataset_id=True)
    result = t.map(self._element())
    self.assertIn("dataset_id", result)
    self.assertEqual(result["dataset_id"].shape, (8,))
    self.assertTrue(np.all(result["dataset_id"] == 3))
    self.assertEqual(result["dataset_id"].dtype, np.int32)

  def test_omits_dataset_id_when_disabled(self):
    t = MegatronSplitInputsTargets(eod_id=0, emit_dataset_id=False)
    self.assertNotIn("dataset_id", t.map(self._element()))

  def test_omits_when_element_has_no_dataset_id(self):
    t = MegatronSplitInputsTargets(eod_id=0, emit_dataset_id=True)
    self.assertNotIn("dataset_id", t.map(self._element(with_id=False)))


if __name__ == "__main__":
  unittest.main()
