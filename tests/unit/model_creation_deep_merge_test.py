# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the recursive collection merge used when restoring a checkpoint."""

import unittest

import pytest

from maxtext.utils.model_creation_utils import _deep_merge

pytestmark = pytest.mark.cpu_only


class DeepMergeTest(unittest.TestCase):
  """Tests that _deep_merge recurses instead of shallow-overwriting subtrees."""

  def test_nested_subtrees_are_merged_not_replaced(self):
    target = {"layers_0": {"mlp": {"w0": 1, "w1": 2}}}

    _deep_merge(target, {"layers_0": {"mlp": {"w2": 3}, "attn": {"q": 4}}})

    self.assertEqual(target, {"layers_0": {"mlp": {"w0": 1, "w1": 2, "w2": 3}, "attn": {"q": 4}}})

  def test_sibling_collections_are_preserved(self):
    target = {"params": {"decoder": {"w": 1}}}

    _deep_merge(target, {"Tid2EidVar": {"decoder": {"tid2eid": 7}}})

    self.assertEqual(target, {"params": {"decoder": {"w": 1}}, "Tid2EidVar": {"decoder": {"tid2eid": 7}}})

  def test_leaf_values_are_overwritten(self):
    target = {"decoder": {"w": 1}}

    _deep_merge(target, {"decoder": {"w": 2}})

    self.assertEqual(target, {"decoder": {"w": 2}})

  def test_dict_replaces_a_non_dict_leaf(self):
    target = {"decoder": 1}

    _deep_merge(target, {"decoder": {"w": 2}})

    self.assertEqual(target, {"decoder": {"w": 2}})

  def test_non_dict_replaces_a_dict_subtree(self):
    target = {"decoder": {"w": 1}}

    _deep_merge(target, {"decoder": 2})

    self.assertEqual(target, {"decoder": 2})

  def test_empty_source_is_a_no_op(self):
    target = {"decoder": {"w": 1}}

    _deep_merge(target, {})

    self.assertEqual(target, {"decoder": {"w": 1}})


if __name__ == "__main__":
  unittest.main()
