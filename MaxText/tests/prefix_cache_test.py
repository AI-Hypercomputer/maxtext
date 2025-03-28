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

from MaxText import prefix_cache

import pytest
import unittest
import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def create_default_value(prefix=None, true_length=1, padded_length=1, tokens=None) -> prefix_cache.Value:
  if prefix is None:
    prefix = {"decoder": jnp.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])}
  if tokens is None:
    tokens = (1, 2, 3, 4, 5, 0, 0, 0, 0, 0)
  return prefix_cache.Value(
      prefix=prefix,
      true_length=true_length,
      padded_length=padded_length,
      tokens=tokens,
  )


def get_byte_in_use(device_idx: int = 0) -> int:
  jax.clear_caches()
  device = jax.local_devices()[device_idx]
  memory_stats = device.memory_stats()
  if memory_stats is None:
    pytest.skip("Cannot get device memory stats. Does the test run with TPU?")

  return memory_stats["bytes_in_use"]


class ValueTest(unittest.TestCase):
  """Test for Value."""

  def test_get_the_set_value(self):
    true_length = 5
    tokens = (1, 2, 3, 4, 5, 0, 0, 0, 0, 0)
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
    value = prefix_cache.Value(
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
    value = prefix_cache.Value(
        prefix=None,
        true_length=1,
        padded_length=2,
        tokens=(1,),
    )
    assert value.prefix is None
    assert value.prefix_size_bytes == 0

  def test_throw_exception_if_prefix_containing_non_array(self):
    with self.assertRaises(Exception):
      create_default_value(prefix={"a": "abc"})

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
    assert value1_1.tokens == value1_2.tokens
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, value1_1.prefix, value1_2.prefix))
    assert value1_1.prefix_size_bytes == value1_2.prefix_size_bytes
    value1_2 = create_default_value(prefix=prefix2)
    assert value1_1 != value1_2

  @pytest.mark.tpu_only
  def test_device_saved_as_prefix_tree(self):
    local_devices = jax.local_devices()
    num_devices = jax.local_device_count()
    mesh_shape1 = (num_devices,)
    device_mesh1 = mesh_utils.create_device_mesh(mesh_shape1, devices=local_devices)
    mesh1 = Mesh(device_mesh1, axis_names=("x",))
    partition_spec1_1 = PartitionSpec("x", None)
    partition_spec1_2 = PartitionSpec(None, "x")
    sharding1_1 = NamedSharding(mesh1, partition_spec1_1)
    sharding1_2 = NamedSharding(mesh1, partition_spec1_2)

    prefix = {
        "a": jnp.ones((512, 512), device=local_devices[0]),
        "b": jnp.ones((mesh_shape1[0], 512, 512), device=sharding1_1),
        "c": jnp.ones((512, mesh_shape1[0], 512), device=sharding1_2),
    }
    expected_device = {
        "a": local_devices[0],
        "b": sharding1_1,
        "c": sharding1_2,
    }
    value = create_default_value(prefix=prefix)
    assert value.device == expected_device

  @pytest.mark.tpu_only
  def test_device_saved_as_prefix_tree_with_sharding_multiple_dimension(self):
    local_devices = jax.local_devices()
    num_devices = jax.local_device_count()
    # assume number of devices will be multiple of 2
    if num_devices % 2 != 0:
      pytest.skip("Need multiple of 2 devices on testing environment.")
    mesh_shape2 = (num_devices // 2, 2)
    device_mesh2 = mesh_utils.create_device_mesh(mesh_shape2, devices=local_devices)
    mesh2 = Mesh(device_mesh2, axis_names=("x", "y"))
    partition_spec2 = PartitionSpec("x", "y", None)
    sharding2 = NamedSharding(mesh2, partition_spec2)

    prefix = {
        "a": jnp.ones((512, 512), device=local_devices[0]),
        "d": jnp.ones((mesh_shape2[0], mesh_shape2[1], 512), device=sharding2),
    }
    expected_device = {
        "a": local_devices[0],
        "d": sharding2,
    }
    value = create_default_value(prefix=prefix)
    assert value.device == expected_device


class PrefixCacheTrieTest(unittest.TestCase):
  """Test for PrefixCacheTrie."""

  def test_get_longest_common_prefix_key(self):
    trie = prefix_cache.PrefixCacheTrie()
    key = (1, 2, 3, 4)
    trie.insert(key)
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4, 5)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 6, 7)) == key
    assert trie.get_longest_common_prefix_key((2, 3, 4, 5)) is None

  def test_insert_longer_key_replace_shorter_key(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 3, 4))
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == (1, 2, 3, 4)

  def test_insert_key_with_different_suffix_will_store_another_one(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3, 4))
    trie.insert((1, 2, 3, 5))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert trie.get_longest_common_prefix_key((1, 2, 3, 5)) == (1, 2, 3, 5)

  def test_insert_shorter_key_will_not_replace_longer_key(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3, 4))
    trie.insert((1, 2, 3))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == (1, 2, 3, 4)

  def test_insert_multiple_key_and_get_longest_common_prefix(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 4))
    trie.insert((11, 2, 3))
    trie.insert((11, 2))
    trie.insert((11, 3, 4))
    result1 = trie.get_longest_common_prefix_key((1, 2, 3, 4, 5))
    assert result1 is not None
    assert result1[:3] == (1, 2, 3)
    result2 = trie.get_longest_common_prefix_key((1, 2, 4, 5, 6))
    assert result2 is not None
    assert result2[:3] == (1, 2, 4)
    result3 = trie.get_longest_common_prefix_key((11, 2, 3, 4))
    assert result3 is not None
    assert result3[:3] == (11, 2, 3)
    result4 = trie.get_longest_common_prefix_key((11, 3, 6))
    assert result4 is not None and result4[:2] == (11, 3)

  def test_erase_matched_key(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.erase((1, 2, 3))
    assert trie.get_longest_common_prefix_key((1, 2, 3)) is None

  def test_erase_shorter_or_longer_key_will_not_effect(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.erase((1, 2, 3, 4))
    trie.erase((1, 2))
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == (1, 2, 3)

  def test_erase_key_will_change_to_another_longest_common_prefix_key(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 4))
    first_matched_key = trie.get_longest_common_prefix_key((1, 2))
    assert first_matched_key is not None
    trie.erase(first_matched_key)
    second_matched_key = trie.get_longest_common_prefix_key((1, 2))
    assert second_matched_key is not None
    assert second_matched_key[:2] == (1, 2)
    trie.erase(second_matched_key)
    assert trie.get_longest_common_prefix_key((1, 2)) is None

  def test_insert_after_erase_to_empty(self):
    trie = prefix_cache.PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.erase((1, 2, 3))
    trie.insert((4, 5, 6))
    assert trie.get_longest_common_prefix_key((4, 5, 6)) == (4, 5, 6)


class BasicStorageTest(unittest.TestCase):

  def test_is_enough_space_remain(self):
    value = create_default_value()
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.has_enough_space(value.prefix_size_bytes) is True
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes - 1)
    assert storage.has_enough_space(value.prefix_size_bytes) is False

  def test_add_and_fetch_with_key_exactly_matched(self):
    key = (1, 2, 3)
    value = create_default_value()
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.add(key, value) is True
    assert storage.retrieve(key) == value

  def test_add_fail_if_not_enough_space(self):
    value = create_default_value()
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes * 2 - 1)
    assert storage.add((1,), value) is True
    # The second one will exceed 1 bytes
    assert storage.add((2,), value) is False
    assert storage.retrieve((2,)) is None

  def test_cannot_retrieve_not_exactly_matched_key(self):
    key = (1, 2, 3)
    value = create_default_value()
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.add(key, value) is True
    assert storage.retrieve((1, 2, 4)) is None
    assert storage.retrieve((1, 2)) is None

  def test_add_and_retrieve_multiple_keys(self):
    storage = prefix_cache.BasicStorage(max_size_bytes=1_000_000)
    value1 = create_default_value(tokens=(1,))
    storage.add((1,), value1)
    value2 = create_default_value(tokens=(2,))
    storage.add((2,), value2)
    assert storage.retrieve((1,)) == value1
    assert storage.retrieve((2,)) == value2

  def test_evict_return_evicted_value(self):
    value = create_default_value()
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes)
    storage.add((1,), value)
    evict_value = storage.evict((1,))
    assert evict_value == value

  def test_evict_will_release_the_memory_usage_and_cannot_retrieve_and_evict_after_evict(self):
    value = create_default_value()
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes)
    storage.add((1,), value)
    # memory is not enough
    assert storage.add((2,), create_default_value()) is False
    storage.evict((1,))
    assert storage.retrieve((1,)) is None
    assert storage.evict((1,)) is None
    # should add another after evict
    value2 = create_default_value()
    assert storage.add((2,), value2) is True
    assert storage.retrieve((2,)) == value2

  def test_evict_multiple_values(self):
    key1 = (1,)
    value1 = create_default_value(tokens=key1)
    key2 = (
        1,
        2,
    )
    value2 = create_default_value(tokens=key2)
    storage = prefix_cache.BasicStorage(max_size_bytes=value1.prefix_size_bytes * 2)
    storage.add(key1, value1)
    storage.add(key2, value2)
    evict_value = storage.evict(key2)
    assert evict_value == value2
    assert storage.retrieve(key2) is None
    assert storage.retrieve(key1) is not None
    evict_value = storage.evict(key1)
    assert storage.retrieve(key1) is None

  def test_get_max_size_bytes(self):
    storage = prefix_cache.BasicStorage(max_size_bytes=100)
    assert storage.get_max_size_bytes() == 100

  def test_contains_key(self):
    value = create_default_value()
    storage = prefix_cache.BasicStorage(max_size_bytes=value.prefix_size_bytes)
    key = (1,)
    assert storage.contains(key) is False
    storage.add(key, value)
    assert storage.contains(key) is True
    storage.evict(key)
    assert storage.contains(key) is False


class HBMStorageTest(unittest.TestCase):
  """Test roughly for HBMStorage while HBMStorage is a wrapper of BasicStorage."""

  def test_basic_usage(self):
    """Test basic usage of HBMStorage checking all functions work."""
    key = (1,)
    value = create_default_value()
    storage = prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.get_max_size_bytes() == value.prefix_size_bytes
    assert storage.contains(key) is False
    assert storage.has_enough_space(value.prefix_size_bytes) is True
    assert storage.add(key, value) is True
    assert storage.contains(key) is True
    assert storage.retrieve(key) == value
    # Only have one value size, and cannot afford the second.
    assert storage.has_enough_space(value.prefix_size_bytes) is False
    assert storage.evict(key) == value
    assert storage.contains(key) is False
    assert storage.retrieve(key) is None
    assert storage.evict(key) is None
    # After evict, it should have enough space
    assert storage.has_enough_space(value.prefix_size_bytes) is True
    assert storage.add(key, value) is True

  @pytest.mark.tpu_only
  def test_move_value_to_and_from_different_device(self):
    local_devices = jax.local_devices()
    if len(local_devices) < 2:
      pytest.skip("Need to test with multiple devices")

    key = (1,)

    device_0_bytes_before = get_byte_in_use(0)
    value = create_default_value(prefix={"a": jnp.ones((4, 4), device=local_devices[0])})
    device_0_bytes_create_value = get_byte_in_use(0)
    prefix_actually_used_byte = device_0_bytes_create_value - device_0_bytes_before
    # Storage on device 1
    storage = prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes, device=local_devices[1])
    device_1_bytes_before = get_byte_in_use(1)
    assert storage.add(key, value) is True
    # The value will not removed from device 0 until async saved complete.
    # Block the retrieved value to make sure the save complete.
    retrieved_back_value = storage.retrieve(key)
    assert retrieved_back_value is not None
    jax.block_until_ready(retrieved_back_value.prefix)
    del retrieved_back_value
    del value
    device_1_byte_after_add = get_byte_in_use(1)
    # The prefix value is saved in device 1, and delete in device 0
    assert device_1_byte_after_add - device_1_bytes_before == prefix_actually_used_byte
    assert device_0_bytes_before == get_byte_in_use(0)
    # Retrieve back to the original device
    retrieved_back_value = storage.retrieve(key)
    assert retrieved_back_value is not None
    assert device_0_bytes_create_value == get_byte_in_use(0)

  @pytest.mark.tpu_only
  def test_retrieve_to_specific_device(self):
    local_devices = jax.local_devices()
    if len(local_devices) < 2:
      pytest.skip("Need to test with multiple devices")

    key = (1,)

    device_0_bytes_before = get_byte_in_use(0)
    value = create_default_value(prefix={"a": jnp.ones((4, 4), device=local_devices[0])})
    device_0_bytes_create_value = get_byte_in_use(0)
    prefix_actually_used_byte = device_0_bytes_create_value - device_0_bytes_before
    # Storage without move device
    storage = prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.add(key, value) is True

    device_1_byte_before = get_byte_in_use(1)
    # Retrieve to device 1 from storage device 0
    retrieved_back_value = storage.retrieve(key, device=local_devices[1])
    assert retrieved_back_value is not None
    device_1_byte_after = get_byte_in_use(1)
    assert retrieved_back_value.prefix["a"].device == local_devices[1]
    assert device_1_byte_after - device_1_byte_before == prefix_actually_used_byte


class DRAMStorageTest(unittest.TestCase):
  """Test basic usage of DRAMStorage only since DRAMStorage is wrapper of BasicStorage with device_get and device_put."""

  def test_basic_usage(self):
    """Test basic usage of DRAMStorage checking all functions work."""
    key = (1,)
    value = create_default_value()
    storage = prefix_cache.DRAMStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.get_max_size_bytes() == value.prefix_size_bytes
    assert storage.has_enough_space(value.prefix_size_bytes) is True
    assert storage.contains(key) is False
    assert storage.add(key, value) is True
    assert storage.contains(key) is True
    assert storage.retrieve(key) == value
    # Only have one value size, and cannot afford the second.
    assert storage.has_enough_space(value.prefix_size_bytes) is False
    assert storage.evict(key) == value
    assert storage.contains(key) is False
    assert storage.retrieve(key) is None
    assert storage.evict(key) is None
    # After evict, it should have enough space
    assert storage.has_enough_space(value.prefix_size_bytes) is True
    assert storage.add(key, value) is True

  @pytest.mark.tpu_only
  def test_move_value_between_device_and_host(self):
    origin_hbm_byte = get_byte_in_use()
    key = (1,)
    value = create_default_value()
    storage = prefix_cache.DRAMStorage(max_size_bytes=value.prefix_size_bytes)
    value_on_device_hbm_byte = get_byte_in_use()
    assert storage.add(key, value) is True
    # add to cache will not copy another in HBM
    assert value_on_device_hbm_byte == get_byte_in_use()
    del value
    value_on_host_hbm_byte = get_byte_in_use()
    # after del the value on device, hbm memory should release
    assert value_on_host_hbm_byte == origin_hbm_byte
    device_value = storage.retrieve(key)
    # copy the value back to device
    assert value_on_device_hbm_byte == get_byte_in_use()
    del device_value

  @pytest.mark.tpu_only
  def test_value_retrieve_back_to_the_same_device_sharding(self):
    local_devices = jax.local_devices()
    num_devices = jax.local_device_count()
    mesh_shape = (num_devices,)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape, devices=local_devices)
    mesh = Mesh(device_mesh, axis_names=("x",))
    partition_spec = PartitionSpec("x", None)
    sharding = NamedSharding(mesh, partition_spec)

    key = (1,)
    prefix = {
        "cache": jnp.ones((mesh_shape[0], 512, 512), device=sharding),
    }
    value = create_default_value(prefix=prefix)
    storage = prefix_cache.DRAMStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.add(key, value)
    del value
    retrieved_value = storage.retrieve(key)
    assert retrieved_value is not None
    assert retrieved_value.prefix["cache"].device == sharding

  @pytest.mark.tpu_only
  def test_retrieve_to_specific_device(self):
    local_devices = jax.local_devices()
    if len(local_devices) < 2:
      pytest.skip("Need to test with multiple devices")

    key = (1,)
    device_0_bytes_before = get_byte_in_use(0)
    value = create_default_value(prefix={"a": jnp.ones((4, 4), device=local_devices[0])})
    device_0_bytes_create_value = get_byte_in_use(0)
    prefix_actually_used_byte = device_0_bytes_create_value - device_0_bytes_before
    # Storage without move device
    storage = prefix_cache.DRAMStorage(max_size_bytes=value.prefix_size_bytes)
    assert storage.add(key, value) is True

    device_1_byte_before = get_byte_in_use(1)
    # Retrieve to device 1 from storage device 0
    retrieved_back_value = storage.retrieve(key, device=local_devices[1])
    assert retrieved_back_value is not None
    device_1_byte_after = get_byte_in_use(1)
    assert retrieved_back_value.prefix["a"].device == local_devices[1]
    assert device_1_byte_after - device_1_byte_before == prefix_actually_used_byte


class LRUStrategyTest(unittest.TestCase):

  def test_evict_none_if_no_use(self):
    strategy = prefix_cache.LRUStrategy()
    assert strategy.evict() is None

  def test_evict_FIFO(self):
    strategy = prefix_cache.LRUStrategy()
    strategy.use((1,))
    strategy.use((2,))
    strategy.use((3,))
    assert strategy.evict() == (1,)
    assert strategy.evict() == (2,)
    assert strategy.evict() == (3,)
    assert strategy.evict() is None

  def test_use_in_the_middle_update_to_the_last(self):
    strategy = prefix_cache.LRUStrategy()
    strategy.use((1,))
    strategy.use((2,))
    strategy.use((1,))
    assert strategy.evict() == (2,)
    assert strategy.evict() == (1,)
    assert strategy.evict() is None


class HierarchicalCacheTest(unittest.TestCase):

  def test_add_to_all_layers(self):
    value = create_default_value()
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    key = (1,)
    assert cache.add(key, value)[0] is True
    assert layers[0].contains(key)
    assert layers[1].contains(key)

  def test_cannot_add_if_first_layer_max_size_is_not_enough(self):
    value = create_default_value()
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes - 1),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    key = (1,)
    assert cache.add(key, value)[0] is False
    assert not layers[0].contains(key)
    assert not layers[1].contains(key)

  def test_add_evict_to_enough_space_by_lru(self):
    value = create_default_value()
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 2),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    assert cache.add((1,), value)[0] is True
    assert cache.add((2,), value)[0] is True
    ok, evicted = cache.add((3,), value)
    assert ok is True
    assert len(evicted) == 1
    assert (1,) in evicted
    assert not layers[0].contains((1,))
    assert not layers[0].contains((2,))
    assert layers[0].contains((3,))
    assert not layers[1].contains((1,))
    assert layers[1].contains((2,))
    assert layers[1].contains((3,))

  def test_retrieve_will_from_all_layer_and_add_to_all_layer(self):
    value = create_default_value()
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 2),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    assert cache.add((1,), value)[0] is True
    assert cache.add((2,), value)[0] is True
    assert not layers[0].contains((1,))
    assert layers[1].contains((1,))
    assert cache.retrieve((1,)) == value
    assert layers[0].contains((1,))
    assert not layers[0].contains((2,))

  def test_retrieve_not_exist_in_any_layers_return_none(self):
    value = create_default_value()
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 2),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    assert cache.add((1,), value)[0] is True
    assert cache.add((2,), value)[0] is True
    assert cache.add((3,), value)[0] is True
    # Key (1,) is evicted since LRU
    assert cache.retrieve((1,)) is None

  def test_add_will_happen_to_all_layers_even_if_some_layers_already_contains_the_key(self):
    value = create_default_value()
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 2),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    assert cache.add((1,), value)[0] is True
    assert cache.add((2,), value)[0] is True
    # layers[0] do not have (1,) now since LRU
    assert not layers[0].contains((1,))
    assert layers[1].contains((1,))
    ok, evicted = cache.add((1,), value)
    assert ok is True
    # (1,) is not evicted from the second layer
    assert not evicted
    assert layers[0].contains((1,))
    assert not layers[0].contains((2,))
    assert layers[1].contains((1,))
    assert layers[1].contains((2,))

  def test_lru_affect_all_layers_when_add_and_retrieve(self):
    value = create_default_value()
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 2),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 5),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    # There is a LRU queue length of 5. The first 2 will also at the first layer
    assert cache.add((1,), value)[0] is True
    assert cache.add((2,), value)[0] is True
    assert cache.retrieve((1,)) is not None
    assert cache.add((3,), value)[0] is True
    assert cache.add((4,), value)[0] is True
    assert cache.add((5,), value)[0] is True
    assert cache.retrieve((4,)) is not None
    ok, evicted = cache.add((6,), value)
    assert ok is True
    assert len(evicted) == 1
    assert (2,) in evicted
    assert cache.retrieve((3,)) is not None
    # Now [3, 6, 4, 5, 1] in LRU
    assert layers[0].contains((3,))
    assert layers[0].contains((6,))
    assert layers[1].contains((3,))
    assert layers[1].contains((6,))
    assert layers[1].contains((4,))
    assert layers[1].contains((5,))
    assert layers[1].contains((1,))
    # 2 is evicted
    assert not layers[1].contains((2,))

  @pytest.mark.tpu_only
  def test_retrieve_to_specific_device(self):
    local_devices = jax.local_devices()
    if len(local_devices) < 2:
      pytest.skip("Need to test with multiple devices")

    key = (1,)
    value = create_default_value(prefix=jnp.ones((512, 512), device=local_devices[0]))
    layers = (
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 2),
        prefix_cache.HBMStorage(max_size_bytes=value.prefix_size_bytes * 5),
    )
    cache = prefix_cache.HierarchicalCache(layers)
    assert cache.add(key, value)[0]
    retrieved_value = cache.retrieve(key, device=local_devices[1])
    assert retrieved_value is not None
    assert retrieved_value.prefix.device == local_devices[1]
    # The device in the Value remain the original before saved
    assert retrieved_value.device == local_devices[0]


class PrefixCacheTest(unittest.TestCase):

  def test_cache_miss_save_hit_load(self):
    max_bytes = 64 * 1024 * 1024 * 1024  # 64 GB
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
    tokens = (1, 2, 3)
    no_matched_key = prefix_cache_inst.fetch_longest_common_prefix_key(tokens)
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
    saved_value = prefix_cache.Value(prefix=kv_cache_copy, true_length=len(tokens), padded_length=len(tokens), tokens=tokens)
    prefix_cache_inst.save(tuple(tokens), saved_value)

    tokens_with_common_prefix = tokens + (4, 5, 6)
    matched_key = prefix_cache_inst.fetch_longest_common_prefix_key(tokens_with_common_prefix)
    assert matched_key is not None
    assert matched_key == tokens
    loaded_value = prefix_cache_inst.load(matched_key)
    assert loaded_value == saved_value

  def test_clear_cache(self):
    tokens = (1, 2, 3)
    value = create_default_value(tokens=tokens)
    max_bytes = value.prefix_size_bytes
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
    assert prefix_cache_inst.save(tokens, value) is True
    assert prefix_cache_inst.load(tokens) == value
    prefix_cache_inst.clear()
    assert prefix_cache_inst.load(tokens) is None

  def test_evict_cache_with_LRU_strategy(self):
    tokens1 = (1,)
    value1 = create_default_value(tokens=tokens1)
    tokens2 = (2,)
    value2 = create_default_value(tokens=tokens2)
    tokens3 = (3,)
    value3 = create_default_value(tokens=tokens3)
    # cache with 2 values size, and will evict with LRU
    max_bytes = value1.prefix_size_bytes * 2
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
    assert prefix_cache_inst.save(tokens1, value1) is True
    assert prefix_cache_inst.save(tokens2, value2) is True
    assert prefix_cache_inst.load(tokens1) == value1
    assert prefix_cache_inst.save(tokens3, value3) is True
    assert prefix_cache_inst.load(tokens2) is None

  def test_fetch_longest_common_prefix_key_does_not_affect_LRU(self):
    tokens1 = (1,)
    value1 = create_default_value(tokens=tokens1)
    tokens2 = (2,)
    value2 = create_default_value(tokens=tokens2)
    tokens3 = (3,)
    value3 = create_default_value(tokens=tokens3)
    # cache with 2 values size, and will evict with LRU
    max_bytes = value1.prefix_size_bytes * 2
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
    assert prefix_cache_inst.save(tokens1, value1) is True
    assert prefix_cache_inst.save(tokens2, value2) is True
    assert prefix_cache_inst.fetch_longest_common_prefix_key(tokens1) == tokens1
    assert prefix_cache_inst.save(tokens3, value3) is True
    assert prefix_cache_inst.fetch_longest_common_prefix_key(tokens1) is None
    assert prefix_cache_inst.load(tokens1) is None

  def test_evict_cache_until_feasible_to_new_cache(self):
    # value2 is double size of value1
    value1 = create_default_value(prefix=[jnp.array([1, 2, 3]) for _ in range(1)])
    value2 = create_default_value(prefix=[jnp.array([1, 2, 3]) for _ in range(2)])

    # Only feasible for two value1 or one value2
    max_bytes = value1.prefix_size_bytes * 2
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
    assert prefix_cache_inst.save(key=(1,), value=value1) is True
    assert prefix_cache_inst.save(key=(2,), value=value1) is True
    assert prefix_cache_inst.load(key=(1,)) == value1
    assert prefix_cache_inst.load(key=(2,)) == value1
    assert prefix_cache_inst.save(key=(3,), value=value2) is True
    assert prefix_cache_inst.load(key=(1,)) is None
    assert prefix_cache_inst.load(key=(2,)) is None
    assert prefix_cache_inst.load(key=(3,)) == value2

  def test_will_not_evict_if_whole_cache_is_not_feasible(self):
    # value2 is triple size of value1
    value1 = create_default_value(prefix=[jnp.array([1, 2, 3]) for _ in range(1)])
    value2 = create_default_value(prefix=[jnp.array([1, 2, 3]) for _ in range(3)])

    # Only feasible for two value1 and never value2
    max_bytes = value1.prefix_size_bytes * 2
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
    assert prefix_cache_inst.save(key=(1,), value=value1) is True
    assert prefix_cache_inst.save(key=(2,), value=value1) is True
    assert prefix_cache_inst.load(key=(1,)) == value1
    assert prefix_cache_inst.load(key=(2,)) == value1
    assert prefix_cache_inst.save(key=(3,), value=value2) is False
    assert prefix_cache_inst.load(key=(1,)) == value1
    assert prefix_cache_inst.load(key=(2,)) == value1
    assert prefix_cache_inst.load(key=(3,)) is None

  def test_cannot_fetch_longest_common_prefix_key_while_fully_evicted_from_dram(self):
    value = create_default_value()
    max_bytes = value.prefix_size_bytes
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes * 2)
    assert (
        prefix_cache_inst.save(
            (1, 2),
            value,
        )
        is True
    )
    assert (
        prefix_cache_inst.save(
            (1, 3),
            value,
        )
        is True
    )
    # (1,2,) is only in DRAM now
    assert prefix_cache_inst.fetch_longest_common_prefix_key((1, 2, 3)) == (1, 2)
    assert (
        prefix_cache_inst.save(
            (2, 3),
            value,
        )
        is True
    )
    # (1,2,) is fully evicted now
    assert prefix_cache_inst.fetch_longest_common_prefix_key((1, 2, 3)) == (1, 3)

  @pytest.mark.tpu_only
  def test_hbm_memory_usage(self):
    """Test HBM memory change."""
    pre_bytes_in_use = get_byte_in_use()
    hbm_bytes = 1024
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=hbm_bytes, dram_bytes=hbm_bytes)
    # prefix_cache will not pre allocate the memory
    assert pre_bytes_in_use == get_byte_in_use()
    prefix = {"cached_prefill_key": jnp.array([1, 2, 3, 4], dtype=jnp.bfloat16)}
    prefix_create_bytes_in_use = get_byte_in_use()
    # memory would be allocated with chunked minimum size
    prefix_bytes = prefix_create_bytes_in_use - pre_bytes_in_use
    assert prefix_create_bytes_in_use == pre_bytes_in_use + prefix_bytes
    key = (1, 2, 3)
    prefix_cache_inst.save(key, create_default_value(prefix=prefix))
    # cache save will not copy
    assert prefix_create_bytes_in_use == get_byte_in_use()
    del prefix
    # prefix move into the cache
    assert prefix_create_bytes_in_use == get_byte_in_use()
    loaded_value = prefix_cache_inst.load(key)
    assert loaded_value is not None
    loaded_bytes_in_use = get_byte_in_use()
    # load cache will not copy from cache
    assert loaded_bytes_in_use == prefix_create_bytes_in_use
    del loaded_value
    del_loaded_bytes_in_use = get_byte_in_use()
    assert del_loaded_bytes_in_use == loaded_bytes_in_use
    prefix_cache_inst.clear()
    # clear the cache will clear the saved cache.
    clear_bytes_in_use = get_byte_in_use()
    assert clear_bytes_in_use == del_loaded_bytes_in_use - prefix_bytes
    assert prefix_cache_inst.load(key) is None

  @pytest.mark.tpu_only
  def test_cache_move_between_device_and_host_with_hierarchical_cache(self):
    value1 = create_default_value()
    prefix_size_bytes = value1.prefix_size_bytes
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=prefix_size_bytes, dram_bytes=prefix_size_bytes * 2)
    assert prefix_cache_inst.save((1,), value1)
    del value1
    first_save_bytes_in_use = get_byte_in_use()
    value2 = create_default_value()
    assert prefix_size_bytes == value2.prefix_size_bytes
    assert prefix_cache_inst.save((2,), value2)
    del value2
    # the first saved cache (1,) is evicted since the size of HBM is one value.
    assert first_save_bytes_in_use == get_byte_in_use()

    # loaded value will move to HBM and (2,) in the HBM layer is evicted. The loaded value is the value in the HBM layer
    loaded_value = prefix_cache_inst.load((1,))
    assert loaded_value is not None
    assert first_save_bytes_in_use == get_byte_in_use()
    del loaded_value
    assert first_save_bytes_in_use == get_byte_in_use()

    # switch the cache again
    loaded_value = prefix_cache_inst.load((2,))
    assert loaded_value is not None
    assert first_save_bytes_in_use == get_byte_in_use()
    del loaded_value
    assert first_save_bytes_in_use == get_byte_in_use()

  @pytest.mark.tpu_only
  def test_load_to_specific_device(self):
    local_devices = jax.local_devices()
    if len(local_devices) < 2:
      pytest.skip("Need to test with multiple devices")

    key = (1,)
    value = create_default_value(prefix=jnp.ones((512, 512), device=local_devices[0]))
    prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=value.prefix_size_bytes, dram_bytes=value.prefix_size_bytes * 2)
    assert prefix_cache_inst.save(key, value)
    loaded_value = prefix_cache_inst.load(key, device=local_devices[1])
    assert loaded_value is not None
    assert loaded_value.prefix.device == local_devices[1]
    assert loaded_value.device == local_devices[0]


if __name__ == "__main__":
  unittest.main()
