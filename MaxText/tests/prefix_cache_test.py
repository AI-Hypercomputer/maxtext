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

from prefix_cache import HBMCache, LRUStrategy, PrefixCache, PrefixCacheTrie, Value

import pytest
import unittest
import jax
import jax.numpy as jnp


def create_default_value(prefix=None, true_length=1, padded_length=1, tokens=None) -> Value:
  if prefix is None:
    prefix = {"decoder": jnp.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])}
  if tokens is None:
    tokens = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
  return Value(
      prefix=prefix,
      true_length=true_length,
      padded_length=padded_length,
      tokens=tokens,
  )


class ValueTest(unittest.TestCase):
  """Test for Value."""

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
    prefix_size_byte = 10 * (1 + 4 + 2 + 4)
    value = Value(
        prefix=prefix,
        true_length=true_length,
        padded_length=padded_length,
        tokens=tokens,
    )
    assert value.true_length == true_length
    assert value.padded_length == padded_length
    assert value.tokens == tokens
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, value.prefix, prefix))
    assert value.prefix_size_bytes == prefix_size_byte

  def test_set_none_prefix_without_error(self):
    value = Value(
        prefix=None,
        true_length=1,
        padded_length=2,
        tokens=[1],
    )
    assert value.prefix is None
    assert value.prefix_size_bytes == 0

  def test_set_prefix_tree_with_non_jax_array_member_ignore_the_bytes(self):
    prefix = {
        "non_jax_array": "member",
        "jax_array": jnp.array([1, 2, 3], dtype=jnp.int8),
    }
    prefix_size_bytes = 3
    value = create_default_value(prefix=prefix)
    assert value.prefix == prefix
    assert value.prefix_size_bytes == prefix_size_bytes

  def test_adjust_true_length_shorter_equal_than_tokens(self):
    value = create_default_value(true_length=100, tokens=jnp.array([1, 2, 3]))
    assert value.true_length == 3

  def test_equal(self):
    prefix1 = {
        "decoder": jnp.array([1, 2, 3], dtype=jnp.int8),
    }
    prefix2 = {
        "decoder": jnp.array([1, 2, 3, 4, 5], dtype=jnp.int8),
    }
    value1_1 = create_default_value(prefix=prefix1)
    value1_2 = create_default_value(prefix=prefix1)
    assert value1_1 != 42
    assert value1_1 == value1_2
    assert value1_1.true_length == value1_2.true_length
    assert value1_1.padded_length == value1_2.padded_length
    assert jnp.array_equal(value1_1.tokens, value1_2.tokens)
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, value1_1.prefix, value1_2.prefix))
    assert value1_1.prefix_size_bytes == value1_2.prefix_size_bytes
    value1_2 = create_default_value(prefix=prefix2)
    assert value1_1 != value1_2

  def test_clone_for_copy_jax_array(self):
    """jax array in value prefix should be different or may destroy the array in cache."""
    prefix = {
        "decoder": jnp.array([1, 2, 3], dtype=jnp.int8),
    }
    value1_1 = create_default_value(prefix=prefix)
    value1_2 = value1_1
    assert value1_1.prefix is value1_2.prefix
    assert value1_1.prefix["decoder"] is value1_2.prefix["decoder"]
    value2 = value1_1.clone()
    assert value2.prefix is not value1_1.prefix
    assert value2.prefix["decoder"] is not value1_1.prefix["decoder"]
    assert value2 == value1_1


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


class HBMCacheTest(unittest.TestCase):

  def test_is_enough_space_remain(self):
    value = create_default_value()
    hbm_cache = HBMCache(max_size_bytes=value.prefix_size_bytes)
    assert hbm_cache.is_enough_space_remain(value) is True
    hbm_cache = HBMCache(max_size_bytes=value.prefix_size_bytes - 1)
    assert hbm_cache.is_enough_space_remain(value) is False

  def test_add_to_cache_and_fetch_with_key_exactly_matched(self):
    key = (1, 2, 3)
    value = create_default_value()
    hbm_cache = HBMCache(max_size_bytes=value.prefix_size_bytes)
    assert hbm_cache.add_to_cache(key, value) is True
    assert hbm_cache.retrieve_from_cache(key) == value

  def test_add_to_cache_fail_if_not_enough_space(self):
    value = create_default_value()
    hbm_cache = HBMCache(max_size_bytes=value.prefix_size_bytes * 2 - 1)
    assert hbm_cache.add_to_cache((1), value) is True
    # The second one will exceed 1 bytes
    assert hbm_cache.add_to_cache((2), value) is False
    assert hbm_cache.retrieve_from_cache((2)) is None

  def test_cannot_retrieve_not_exactly_matched_key(self):
    key = (1, 2, 3)
    value = create_default_value()
    hbm_cache = HBMCache(max_size_bytes=value.prefix_size_bytes)
    assert hbm_cache.add_to_cache(key, value) is True
    assert hbm_cache.retrieve_from_cache((1, 2, 4)) is None
    assert hbm_cache.retrieve_from_cache((1, 2)) is None

  def test_add_and_retrieve_multiple_keys(self):
    hbm_cache = HBMCache(max_size_bytes=1_000_000)
    value1 = create_default_value(tokens=[1])
    hbm_cache.add_to_cache((1), value1)
    value2 = create_default_value(tokens=[2])
    hbm_cache.add_to_cache((2), value2)
    assert hbm_cache.retrieve_from_cache((1)) == value1
    assert hbm_cache.retrieve_from_cache((2)) == value2

  def test_evict_cache(self):
    value = create_default_value()
    hbm_cache = HBMCache(max_size_bytes=value.prefix_size_bytes)
    hbm_cache.add_to_cache((1), value)
    # memory is not enough
    assert hbm_cache.add_to_cache((2), create_default_value()) is False
    evict_value = hbm_cache.evict_cache((1))
    assert evict_value == value
    assert hbm_cache.retrieve_from_cache((1)) is None
    assert hbm_cache.evict_cache((1)) is None
    # should add another after evict
    value2 = create_default_value()
    assert hbm_cache.add_to_cache((2), value2) is True
    assert hbm_cache.retrieve_from_cache((2)) == value2


class LRUStrategyTest(unittest.TestCase):

  def test_evict_none_if_no_use(self):
    strategy = LRUStrategy()
    assert strategy.evict() is None

  def test_evict_FIFO(self):
    strategy = LRUStrategy()
    strategy.use((1))
    strategy.use((2))
    strategy.use((3))
    assert strategy.evict() == (1)
    assert strategy.evict() == (2)
    assert strategy.evict() == (3)
    assert strategy.evict() is None

  def test_use_in_the_middle_update_to_the_last(self):
    strategy = LRUStrategy()
    strategy.use((1))
    strategy.use((2))
    strategy.use((1))
    assert strategy.evict() == (2)
    assert strategy.evict() == (1)
    assert strategy.evict() is None


class PrefixCacheTest(unittest.TestCase):

  def test_cache_miss_save_hit_load(self):
    hbm_bytes = 64 * 1024 * 1024 * 1024  # 64 GB
    prefix_cache = PrefixCache(hbm_bytes=hbm_bytes)
    tokens = (1, 2, 3)
    no_matched_key = prefix_cache.fetch_longest_common_prefix_key(tokens)
    # first seen prefix will not match any key
    assert no_matched_key is None
    # Use dummy cache which should be returned from prefill: cache, _ = prefill(tokens)
    kv_cache = {
        "decoder": {
            "layer_0": {
                "cached_prefill_key": jnp.array([1, 2, 3, 0, 0, 0, 0, 0], dtype=jnp.bfloat16),
                "cached_prefill_value": jnp.array([1, 2, 3, 0, 0, 0, 0, 0], dtype=jnp.bfloat16),
            },
            "layer_1": {
                "cached_prefill_key": jnp.array([1, 2, 3, 0, 0, 0, 0, 0], dtype=jnp.bfloat16),
                "cached_prefill_value": jnp.array([1, 2, 3, 0, 0, 0, 0, 0], dtype=jnp.bfloat16),
            },
        }
    }
    # Should copy kv_cache before saved if the kv_cache is used after saved.
    kv_cache_copy = jax.tree_util.tree_map(lambda x: x.copy(), kv_cache)
    saved_value = Value(prefix=kv_cache_copy, true_length=len(tokens), padded_length=len(tokens), tokens=tokens)
    prefix_cache.save(tokens, saved_value)

    tokens_with_common_prefix = tokens + (4, 5, 6)
    matched_key = prefix_cache.fetch_longest_common_prefix_key(tokens_with_common_prefix)
    assert matched_key == tokens
    loaded_value = prefix_cache.load(matched_key)
    assert loaded_value == saved_value

  @pytest.mark.tpu_only
  def test_hbm_memory_usage(self):
    """Test HBM memory change.
    Create the class instance will not pre allocate memory.
    Save the cache will move without copy.
    Load the cache will copy from cache.
    Clear the cache will clear the saved cache.
    """
    device = jax.local_devices(0)[0]

    def get_byte_in_use():
      jax.clear_caches()
      return device.memory_stats()["bytes_in_use"]

    pre_bytes_in_use = get_byte_in_use()
    hbm_bytes = 1024
    prefix_cache = PrefixCache(hbm_bytes=hbm_bytes)
    # prefix_cache will not pre allocate the memory
    assert pre_bytes_in_use == get_byte_in_use()
    prefix = {"cached_prefill_key": jnp.array([1, 2, 3, 4], dtype=jnp.bfloat16, device=device)}
    prefix_create_bytes_in_use = get_byte_in_use()
    # memory would be allocated with chunked minimum size
    prefix_bytes = prefix_create_bytes_in_use - pre_bytes_in_use
    assert prefix_create_bytes_in_use == pre_bytes_in_use + prefix_bytes
    key = (1, 2, 3)
    prefix_cache.save(key, create_default_value(prefix=prefix))
    # cache save will not copy
    assert prefix_create_bytes_in_use == get_byte_in_use()
    del prefix
    # prefix move into the cache
    assert prefix_create_bytes_in_use == get_byte_in_use()
    loaded_value = prefix_cache.load(key)
    assert loaded_value is not None
    loaded_bytes_in_use = get_byte_in_use()
    # load cache copy from cache
    assert loaded_bytes_in_use == prefix_create_bytes_in_use + prefix_bytes
    del loaded_value
    del_loaded_bytes_in_use = get_byte_in_use()
    assert del_loaded_bytes_in_use == loaded_bytes_in_use - prefix_bytes
    prefix_cache.clear()
    clear_bytes_in_use = get_byte_in_use()
    assert clear_bytes_in_use == del_loaded_bytes_in_use - prefix_bytes
    assert prefix_cache.load(key) is None


if __name__ == "__main__":
  unittest.main()
