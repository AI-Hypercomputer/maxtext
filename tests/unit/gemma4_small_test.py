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

"""Unit tests for Gemma 4 small (E2B / E4B) layer-pattern helpers."""

import unittest

from maxtext.common.common_types import AttentionType
from maxtext.models import gemma4_small


L = AttentionType.LOCAL_SLIDING
G = AttentionType.GLOBAL


class Gemma4SmallAttentionPatternTest(unittest.TestCase):
  """Per-variant attention-type pattern dispatch."""

  def test_e2b_attention_pattern_period_5(self):
    self.assertEqual(gemma4_small.get_attention_pattern("gemma4-e2b"), (L, L, L, L, G))

  def test_e4b_attention_pattern_period_6(self):
    self.assertEqual(gemma4_small.get_attention_pattern("gemma4-e4b"), (L, L, L, L, L, G))

  def test_default_pattern_period_6(self):
    self.assertEqual(len(gemma4_small.get_attention_pattern(None)), 6)


class Gemma4SmallLayerTypesTest(unittest.TestCase):
  """Per-layer attention-type list across the full stack."""

  def test_e2b_full_layer_types(self):
    layer_types = gemma4_small.build_layer_types(35, "gemma4-e2b")
    self.assertEqual(len(layer_types), 35)
    self.assertEqual(layer_types[0], L)
    # 35 = 7 * 5, so layers 4, 9, ..., 34 are GLOBAL.
    for i in range(4, 35, 5):
      self.assertEqual(layer_types[i], G)

  def test_e4b_full_layer_types(self):
    layer_types = gemma4_small.build_layer_types(42, "gemma4-e4b")
    self.assertEqual(len(layer_types), 42)
    for i in range(42):
      expected = G if i % 6 == 5 else L
      self.assertEqual(layer_types[i], expected, f"layer {i}")


class Gemma4SmallKvSharingTest(unittest.TestCase):
  """KV-sharing donor / shared-layer mapping."""

  def test_e2b_first_kv_shared_layer(self):
    self.assertEqual(gemma4_small.first_kv_shared_layer_idx(35, 20), 15)
    self.assertFalse(gemma4_small.is_kv_shared_layer(14, 35, 20))
    self.assertTrue(gemma4_small.is_kv_shared_layer(15, 35, 20))
    self.assertTrue(gemma4_small.is_kv_shared_layer(34, 35, 20))

  def test_e4b_first_kv_shared_layer(self):
    self.assertEqual(gemma4_small.first_kv_shared_layer_idx(42, 18), 24)
    self.assertFalse(gemma4_small.is_kv_shared_layer(23, 42, 18))
    self.assertTrue(gemma4_small.is_kv_shared_layer(24, 42, 18))

  def test_e2b_kv_donor_mapping(self):
    # E2B has 15 non-shared layers with pattern (L,L,L,L,G). Layer 13 is the
    # last LOCAL_SLIDING and layer 14 the last GLOBAL before sharing starts.
    layer_types = gemma4_small.build_layer_types(35, "gemma4-e2b")
    self.assertEqual(gemma4_small.kv_donor_layer_idx(15, layer_types, 20), 13)
    self.assertEqual(gemma4_small.kv_donor_layer_idx(19, layer_types, 20), 14)
    self.assertIsNone(gemma4_small.kv_donor_layer_idx(0, layer_types, 20))
    self.assertIsNone(gemma4_small.kv_donor_layer_idx(14, layer_types, 20))

  def test_e4b_kv_donor_mapping(self):
    # E4B has 24 non-shared layers with pattern (L,L,L,L,L,G). Layer 22 is
    # the last LOCAL_SLIDING and layer 23 the last GLOBAL before sharing.
    layer_types = gemma4_small.build_layer_types(42, "gemma4-e4b")
    self.assertEqual(gemma4_small.kv_donor_layer_idx(24, layer_types, 18), 22)
    self.assertEqual(gemma4_small.kv_donor_layer_idx(29, layer_types, 18), 23)
    self.assertIsNone(gemma4_small.kv_donor_layer_idx(23, layer_types, 18))

  def test_donor_layer_flag(self):
    layer_types = gemma4_small.build_layer_types(35, "gemma4-e2b")
    self.assertTrue(gemma4_small.is_kv_donor_layer(13, layer_types, 20))
    self.assertTrue(gemma4_small.is_kv_donor_layer(14, layer_types, 20))
    self.assertFalse(gemma4_small.is_kv_donor_layer(12, layer_types, 20))

  def test_no_kv_sharing_when_num_kv_shared_zero(self):
    layer_types = gemma4_small.build_layer_types(10, None)
    self.assertEqual(gemma4_small.first_kv_shared_layer_idx(10, 0), 10)
    for i in range(10):
      self.assertFalse(gemma4_small.is_kv_shared_layer(i, 10, 0))
      self.assertIsNone(gemma4_small.kv_donor_layer_idx(i, layer_types, 0))
      self.assertFalse(gemma4_small.is_kv_donor_layer(i, layer_types, 0))


if __name__ == "__main__":
  unittest.main()
