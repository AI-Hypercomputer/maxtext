# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prefix Cache Test"""

from prefix_cache import PrefixCacheTrie, Value

import unittest
import jax
import jax.numpy as jnp


class ValueTest(unittest.TestCase):
  """Test for Value."""

  def _create_default_value(self, prefix=None, true_length=1, padded_length=1, tokens=None) -> Value:
    if tokens is None:
      tokens = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    return Value(
        prefix=prefix,
        true_length=true_length,
        padded_length=padded_length,
        tokens=tokens,
    )

  def test_get_the_set_value(self):
    true_length = 5
    tokens = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    padded_length = 10
    prefix = {
        "decoder": {
            "layer_0": {
                "cached_prefill_key": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.int8),
                "cached_prefill_value": jnp.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=jnp.float32),
            },
            "layer_1": {
                "cached_prefill_key": jnp.array([11, 22, 33, 44, 55, 66, 77, 88, 99, 1010], dtype=jnp.float16),
                "cached_prefill_value": jnp.array(
                    [1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020],
                    dtype=jnp.int32,
                ),
            },
        }
    }
    # array len * byte size(int8, + float32 + float16 + int32) in the prefix
    prefix_byte_size = 10 * (1 + 4 + 2 + 4)
    value = Value(
        prefix=prefix,
        true_length=true_length,
        padded_length=padded_length,
        tokens=tokens,
    )
    assert value.prefix == prefix
    assert value.true_length == true_length
    assert value.padded_length == padded_length
    assert jnp.array_equal(value.tokens, tokens)
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, value.prefix, prefix))
    assert value.prefix_size_bytes == prefix_byte_size

  def test_set_none_prefix_without_error(self):
    value = self._create_default_value(
        prefix=None,
    )
    assert value.prefix == None
    assert value.prefix_size_bytes == 0

  def test_set_prefix_tree_with_non_jax_array_member_ignore_the_bytes(self):
    prefix = {
        "non_jax_array": "member",
        "jax_array": jnp.array([1, 2, 3], dtype=jnp.int8),
    }
    prefix_size_bytes = 3
    value = self._create_default_value(prefix=prefix)
    assert value.prefix == prefix
    assert value.prefix_size_bytes == prefix_size_bytes

  def test_adjust_true_length_shorter_equal_than_tokens(self):
    value = self._create_default_value(true_length=100, tokens=jnp.array([1, 2, 3]))
    assert value.true_length == 3


class PrefixCacheTrieTest(unittest.TestCase):
  """Test for PrefixCacheTrie."""

  def test_get_longest_common_prefix_key(self):
    trie = PrefixCacheTrie()
    key = (1, 2, 3, 4)
    trie.insert(key)
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4, 5)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 6, 7)) == key
    assert trie.get_longest_common_prefix_key((2, 3, 4, 5)) is None

  def test_insert_longer_key_replace_shorter_key(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 3, 4))
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == (1, 2, 3, 4)

  def test_insert_key_with_different_suffix_will_store_another_one(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3, 4))
    trie.insert((1, 2, 3, 5))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert trie.get_longest_common_prefix_key((1, 2, 3, 5)) == (1, 2, 3, 5)

  def test_insert_shorter_key_will_not_replace_longer_key(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3, 4))
    trie.insert((1, 2, 3))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == (1, 2, 3, 4)

  def test_insert_multiple_key_and_get_longest_common_prefix(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 4))
    trie.insert((11, 2, 3))
    trie.insert((11, 2))
    trie.insert((11, 3, 4))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4, 5))[:3] == (1, 2, 3)
    assert trie.get_longest_common_prefix_key((1, 2, 4, 5, 6))[:3] == (1, 2, 4)
    assert trie.get_longest_common_prefix_key((11, 2, 3, 4))[:3] == (11, 2, 3)
    assert trie.get_longest_common_prefix_key((11, 3, 6))[:2] == (11, 3)

  def test_erase_matched_key(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.erase((1, 2, 3))
    assert trie.get_longest_common_prefix_key((1, 2, 3)) is None

  def test_erase_shorter_or_longer_key_will_not_effect(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.erase((1, 2, 3, 4))
    trie.erase((1, 2))
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == (1, 2, 3)

  def test_erase_key_will_change_to_another_longest_common_prefix_key(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 4))
    first_matched_key = trie.get_longest_common_prefix_key((1, 2))
    trie.erase(first_matched_key)
    second_matched_key = trie.get_longest_common_prefix_key((1, 2))
    assert second_matched_key[:2] == (1, 2)
    trie.erase(second_matched_key)
    assert trie.get_longest_common_prefix_key((1, 2)) is None


if __name__ == "__main__":
  unittest.main()
