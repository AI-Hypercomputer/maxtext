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

"""Unit tests for GLM-5.2 IndexShare pattern utilities."""

import unittest

from maxtext.utils import index_share_utils


class IndexShareUtilsTest(unittest.TestCase):
  """Tests for pattern parsing, donor resolution, and served group sizes."""

  def test_parse_index_share_pattern_periodic_divisible(self):
    pattern = index_share_utils.parse_index_share_pattern("FSSS", 8)
    self.assertEqual(pattern, ("F", "S", "S", "S", "F", "S", "S", "S"))

  def test_parse_index_share_pattern_with_commas_spaces_and_sequence(self):
    pattern = index_share_utils.parse_index_share_pattern("f, s, s, s", 8)
    self.assertEqual(pattern, ("F", "S", "S", "S", "F", "S", "S", "S"))

    seq_pattern = index_share_utils.parse_index_share_pattern(["f", "s"], 4)
    self.assertEqual(seq_pattern, ("F", "S", "F", "S"))

  def test_parse_index_share_pattern_exact(self):
    pattern = index_share_utils.parse_index_share_pattern("FFSSFS", 6)
    self.assertEqual(pattern, ("F", "F", "S", "S", "F", "S"))

  def test_parse_glm52_78_layer_pattern(self):
    glm52_str = "FFF" + "SSSF" * 18 + "SSS"
    self.assertEqual(len(glm52_str), 78)
    pattern = index_share_utils.parse_index_share_pattern(glm52_str, 78)
    self.assertEqual(len(pattern), 78)

    full_indices = [idx for idx, role in enumerate(pattern) if role == "F"]
    shared_indices = [idx for idx, role in enumerate(pattern) if role == "S"]
    expected_full = [0, 1, 2] + list(range(6, 78, 4))
    self.assertEqual(len(full_indices), 21)
    self.assertEqual(len(shared_indices), 57)
    self.assertEqual(full_indices, expected_full)

    group_sizes = index_share_utils.get_served_group_sizes(pattern)
    self.assertEqual(group_sizes[:2], (1, 1))
    self.assertEqual(group_sizes[2:], (4,) * 76)

  def test_non_divisible_pattern_raises_value_error(self):
    for num_layers in (10, 78):
      with self.subTest(num_layers=num_layers):
        with self.assertRaisesRegex(ValueError, "must evenly divide"):
          index_share_utils.parse_index_share_pattern("FSSS", num_layers)

  def test_invalid_first_layer_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "First layer \\(Layer 0\\) must always be 'F'"):
      index_share_utils.parse_index_share_pattern("SFFF", 4)

  def test_invalid_characters_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "Invalid characters"):
      index_share_utils.parse_index_share_pattern("FXSS", 4)

  def test_empty_pattern_raises_value_error(self):
    for empty_val in ("", " ,   ", []):
      with self.subTest(pattern=empty_val):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
          index_share_utils.parse_index_share_pattern(empty_val, 4)

  def test_pattern_exceeding_num_layers_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "cannot exceed num_decoder_layers"):
      index_share_utils.parse_index_share_pattern("FSSSFSSS", 4)

  def test_non_positive_num_layers_raises_value_error(self):
    for num_layers in (0, -4):
      with self.subTest(num_layers=num_layers):
        with self.assertRaisesRegex(ValueError, "num_layers must be positive"):
          index_share_utils.parse_index_share_pattern("FSSS", num_layers)

  def test_donor_indices_and_checkpoint_donor_resolution(self):
    pattern = ("F", "F", "S", "S", "F", "S")
    donors = index_share_utils.get_donor_layer_indices(pattern)
    self.assertEqual(donors, (0, 1, 1, 1, 4, 4))

    pattern_12 = index_share_utils.parse_index_share_pattern("FSSS", 12)
    for layer_idx in range(12):
      self.assertEqual(index_share_utils.get_donor_layer_idx(layer_idx, pattern_12), (layer_idx // 4) * 4)

  def test_served_group_sizes(self):
    pattern = ("F", "S", "S", "S", "F", "S", "S")
    sizes = index_share_utils.get_served_group_sizes(pattern)
    self.assertEqual(sizes, (4, 4, 4, 4, 3, 3, 3))

  def test_is_shared_layer(self):
    pattern = ("F", "S", "S", "F")
    self.assertFalse(index_share_utils.is_shared_layer(-1, pattern))
    self.assertFalse(index_share_utils.is_shared_layer(0, pattern))
    self.assertTrue(index_share_utils.is_shared_layer(1, pattern))
    self.assertTrue(index_share_utils.is_shared_layer(2, pattern))
    self.assertFalse(index_share_utils.is_shared_layer(3, pattern))
    self.assertFalse(index_share_utils.is_shared_layer(4, pattern))


if __name__ == "__main__":
  unittest.main()
