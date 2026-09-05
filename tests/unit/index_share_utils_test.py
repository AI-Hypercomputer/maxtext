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

"""Unit tests for IndexShare pattern utilities."""

import unittest
from maxtext.utils import index_share_utils


class IndexShareUtilsTest(unittest.TestCase):

  def test_parse_index_share_pattern_periodic(self):
    pattern = index_share_utils.parse_index_share_pattern("FSSS", 10)
    self.assertEqual(len(pattern), 10)
    self.assertEqual(pattern, ("F", "S", "S", "S", "F", "S", "S", "S", "F", "S"))

  def test_parse_index_share_pattern_with_commas_and_spaces(self):
    pattern = index_share_utils.parse_index_share_pattern("f, s, s, s", 8)
    self.assertEqual(pattern, ("F", "S", "S", "S", "F", "S", "S", "S"))

  def test_parse_index_share_pattern_exact(self):
    pattern = index_share_utils.parse_index_share_pattern("FSFSS", 5)
    self.assertEqual(pattern, ("F", "S", "F", "S", "S"))

  def test_invalid_first_layer(self):
    with self.assertRaises(ValueError) as ctx:
      index_share_utils.parse_index_share_pattern("SFFF", 4)
    self.assertIn("First layer (Layer 0) must always be 'F'", str(ctx.exception))

  def test_invalid_characters(self):
    with self.assertRaises(ValueError) as ctx:
      index_share_utils.parse_index_share_pattern("FXSS", 4)
    self.assertIn("Invalid characters", str(ctx.exception))

  def test_donor_indices(self):
    pattern = ("F", "S", "S", "S", "F", "S", "S")
    donors = index_share_utils.get_donor_layer_indices(pattern)
    self.assertEqual(donors, (0, 0, 0, 0, 4, 4, 4))

  def test_group_sizes(self):
    pattern = ("F", "S", "S", "S", "F", "S", "S")
    sizes = index_share_utils.get_served_group_sizes(pattern)
    # Layer 0 serves 4 layers (0, 1, 2, 3) -> size 4
    # Layer 4 serves 3 layers (4, 5, 6) -> size 3
    self.assertEqual(sizes, (4, 4, 4, 4, 3, 3, 3))

  def test_pattern_expansion_78_layers(self):
    pattern = index_share_utils.parse_index_share_pattern("FSSS", 78)
    self.assertEqual(len(pattern), 78)
    self.assertEqual(pattern[0], "F")
    self.assertEqual(pattern[1], "S")
    self.assertEqual(pattern[2], "S")
    self.assertEqual(pattern[3], "S")
    self.assertEqual(pattern[4], "F")
    num_f = sum(1 for p in pattern if p == "F")
    num_s = sum(1 for p in pattern if p == "S")
    self.assertEqual(num_f, 20)
    self.assertEqual(num_s, 58)

  def test_checkpoint_donor_resolution(self):
    pattern = index_share_utils.parse_index_share_pattern("FSSS", 12)
    for l in range(4):
      self.assertEqual(index_share_utils.get_donor_layer_idx(l, pattern), 0)
    for l in range(4, 8):
      self.assertEqual(index_share_utils.get_donor_layer_idx(l, pattern), 4)
    for l in range(8, 12):
      self.assertEqual(index_share_utils.get_donor_layer_idx(l, pattern), 8)

  def test_is_shared_layer(self):
    pattern = ("F", "S", "S", "F")
    self.assertFalse(index_share_utils.is_shared_layer(0, pattern))
    self.assertTrue(index_share_utils.is_shared_layer(1, pattern))
    self.assertTrue(index_share_utils.is_shared_layer(2, pattern))
    self.assertFalse(index_share_utils.is_shared_layer(3, pattern))
    self.assertFalse(index_share_utils.is_shared_layer(4, pattern))


if __name__ == "__main__":
  unittest.main()
