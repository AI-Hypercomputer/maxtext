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

"""Unit tests for GLM-5.2 Training-Aware IndexShare (Cross-Layer IndexCache)."""

import unittest
from maxtext.utils import index_share_utils


class GLM52IndexSharePatternTest(unittest.TestCase):
  """Tests for IndexShare pattern utilities."""

  def test_pattern_expansion_and_validation(self):
    # Test periodic expansion for 78 layers (GLM-5.1/5.2 default)
    pattern = index_share_utils.parse_index_share_pattern("FSSS", 78)
    self.assertEqual(len(pattern), 78)
    self.assertEqual(pattern[0], "F")
    self.assertEqual(pattern[1], "S")
    self.assertEqual(pattern[2], "S")
    self.assertEqual(pattern[3], "S")
    self.assertEqual(pattern[4], "F")

    # Count F and S layers (1/4 retention)
    num_f = sum(1 for p in pattern if p == "F")
    num_s = sum(1 for p in pattern if p == "S")
    self.assertEqual(num_f, 20)  # ceil(78/4)
    self.assertEqual(num_s, 58)

  def test_donor_mapping(self):
    pattern = index_share_utils.parse_index_share_pattern("FSSS", 8)
    donors = index_share_utils.get_donor_layer_indices(pattern)
    self.assertEqual(donors, (0, 0, 0, 0, 4, 4, 4, 4))

  def test_group_sizes(self):
    pattern = index_share_utils.parse_index_share_pattern("FSSS", 8)
    sizes = index_share_utils.get_served_group_sizes(pattern)
    self.assertEqual(sizes, (4, 4, 4, 4, 4, 4, 4, 4))

  def test_invalid_pattern_raises(self):
    with self.assertRaises(ValueError):
      index_share_utils.parse_index_share_pattern("SFFF", 4)
    with self.assertRaises(ValueError):
      index_share_utils.parse_index_share_pattern("FABCS", 5)
    with self.assertRaises(ValueError):
      index_share_utils.parse_index_share_pattern("", 4)


if __name__ == "__main__":
  unittest.main()
