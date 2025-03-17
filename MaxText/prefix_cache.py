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

"""Implementation of PrefixCache with LRU cache and Trie based key lookup.

i) Initialize cache

# Prefix return by prefill function of mixtral-8x22b model with max_prefill_length=1024,
# int8 quantize_kvcache is 235_930_060 bytes, nearly 256 MB.
# It need about 256 * 256 MB = 64GB to caching 256 prompts prefill.
hbm_bytes = 64_000_000_000  # 64 GB
dram_bytes = 640_000_000_000  # 640 GB
prefix_cache = PrefixCache(hbm_bytes=hbm_bytes, dram_bytes=dram_bytes)

ii) Read from Cache:

# From request
prompt: str = "blah"
full_key: Key = tokenizer.tokenize(prompt)

# Inside prefill call
orig_key_len = len(full_key)

matched_key = prefix_cache.fetch_longest_common_prefix_key(key)
if matched_key is not None and load_cache_is_efficient_enough(matched_key):
  cache_results = prefix_cache.load(matched_key)
else:
  cache_results = None

if cache_results:
   # load cache successfully
   matched_len = calculate_matched_len(full_key, matched_key)
   cached_prefix = cache_results.prefix
else:
   cached_prefix = None

iii) Run prefill un-cached prompt and write to cache

full_kv_cache = prefill(prompt, cached_prefix, matched_len)

# if the cache didn't contain prefix or if it was a partial match
if cached_prefix is None or matched_len != orig_key_len:
    prefix_cache.save(full_key, Value(
        prefix=full_kv_cache,
        true_length=true_length,
        padded_length=padded_length,
        tokens=full_key,
    ))

"""

from collections import OrderedDict
from typing import Any, Optional, Tuple
import abc
import dataclasses
import jax
import jax.numpy as jnp
import logging
import threading

logger = logging.getLogger(__name__)

Token = int
# Tuple of tokens from prompt
Key = Tuple[Token, ...]
Prefix = Any  # KVCache for one prompt


class Value:
  """Object stored contains the actual KVcache

  Attributes:
    prefix:
      Readonly. Prefix Cache using in model. Should be pytree of all jax.Array.
    true_length:
      Readonly. True length of tokens calculate prefix. Should be <= than len(tokens).
      true_length will be min(true_length, len(tokens))
    padded_length:
      Readonly. Length of tokens including padding calculate prefix.
    tokens:
      Readonly. Tokens calculate prefix. may include partial of padding.
    prefix_size_bytes:
      Readonly. bytes of prefix.
    device:
      Readonly. Devices of prefix. The same structure of pytree to prefix.
      The device may be different from actually prefix is in.
      It used for retrieved back to original device.
  """

  def __init__(
      self,
      *,
      prefix: Prefix,
      true_length: int,
      padded_length: int,
      tokens: tuple[Token, ...],
      prefix_size_bytes: Optional[int] = None,
      device=None,
  ):
    """Init Value with attributes.

    If true_length shorter than len(tokens), true_length will adjust to len(tokens).
    If prefix_size_bytes is not provided, calculate automatically.
    prefix should be pytree of all jax.Array. It may raise exception if there is anything not a jax.Array,
    If either prefix_size_bytes and device is None, get from prefix.
    """
    self._prefix = prefix
    self._true_length = self._maybe_adjust_true_length(true_length, tokens)
    self._padded_length = padded_length
    self._tokens = tokens
    if prefix_size_bytes is None:
      self._prefix_size_bytes: int = jax.tree.reduce(
          lambda acc, array: acc + array.nbytes,
          prefix,
          0,
      )
    else:
      self._prefix_size_bytes = prefix_size_bytes

    if device is None:
      self._device = jax.tree.map(lambda x: x.device, prefix)
    else:
      self._device = device

  @property
  def prefix(self) -> Prefix:
    return self._prefix

  @property
  def true_length(self) -> int:
    return self._true_length

  @property
  def padded_length(self) -> int:
    return self._padded_length

  @property
  def tokens(self) -> tuple[Token, ...]:
    return self._tokens

  @property
  def prefix_size_bytes(self) -> int:
    return self._prefix_size_bytes

  @property
  def device(self) -> int:
    return self._device

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Value):
      return False
    return (
        other.padded_length == self.padded_length
        and other.tokens == self.tokens
        and jax.tree.all(jax.tree.map(jnp.array_equal, other.prefix, self.prefix))
        and other.prefix_size_bytes == self.prefix_size_bytes
    )

  def _maybe_adjust_true_length(self, true_length: int, tokens: tuple[Token, ...]) -> int:
    if true_length > len(tokens):
      logger.warning("true_length=%d should <= len(tokens)=%d.", true_length, len(tokens))

    return min(true_length, len(tokens))


def device_put_value(value: Value, device: Any = None) -> Value:
  """Create a new value with prefix put to device.

  If the device is the same as value.prefix, we expect no copy here in jax.device_put.

  Args:
    value: Value to put.
    device:
      The same as the jax.device_put device to put the value.prefix.
      if None, put to the value.device.
  Returns:
    Values with prefix put to device.
  """
  put_device = device
  if put_device is None:
    put_device = value.device
  return Value(
      prefix=jax.device_put(value.prefix, put_device),
      true_length=value.true_length,
      padded_length=value.padded_length,
      tokens=value.tokens,
      prefix_size_bytes=value.prefix_size_bytes,
      device=value.device,
  )


class PrefixCacheTrie:
  """Stores prefix tokens as a trie for fast lookup index of PrefixCache store in cache.

  Insert longer Key replace shorter key to be the longest common prefix key.
  The shorter key will never be returned even if longer key is erased, and should got evicted in the future.

  Assume Key is equal length to tokens, which can be used to slice prompt and cache Value.
  Should check the return key common prefix length by the caller.

  If erase the Key not the leaf, nothing will happen.
  If erased key match at a leaf, delete the node and ancestors would be the leaf after deleted.
  """

  @dataclasses.dataclass
  class Node:
    """Trie Node."""

    parent: Optional["PrefixCacheTrie.Node"] = None
    token: Optional[Token] = None
    children: dict[Token, "PrefixCacheTrie.Node"] = dataclasses.field(default_factory=dict)

    def is_leaf(self):
      return len(self.children) == 0

    def get_one_child_token(self) -> Optional[Token]:
      if len(self.children) == 0:
        return None
      return next(iter(self.children.keys()))

  def __init__(self):
    self._saved_keys: list[Key] = []
    self._root = PrefixCacheTrie.Node()

  def insert(self, key: Key):
    """Insert key into the trie."""
    node = self._root
    for token in key:
      if token not in node.children:
        node.children[token] = PrefixCacheTrie.Node(parent=node, token=token)
      node = node.children[token]

  def get_longest_common_prefix_key(self, key: Key) -> Optional[Key]:
    """Get the key with longest common prefix.
    If not found at least one token match, return None."""
    result_tokens: list[Token] = []

    node = self._root
    for token in key:
      if token not in node.children:
        break
      node = node.children[token]
      result_tokens.append(token)

    if len(result_tokens) == 0:
      return None

    while not node.is_leaf():
      token = node.get_one_child_token()
      if token is None:
        break
      result_tokens.append(token)
      node = node.children[token]

    return tuple(result_tokens)

  def erase(self, key: Key) -> None:
    """Erase key in trie if it is leaf."""
    node = self._root
    for token in key:
      if token not in node.children:
        return
      node = node.children[token]

    while node.is_leaf():
      parent = node.parent
      if parent is None or node.token not in parent.children:
        return
      del parent.children[node.token]
      node = parent


class ValueStorageInterface(abc.ABC):
  """Interface for Value storage."""

  @abc.abstractmethod
  def get_max_size_bytes(self) -> int:
    """Get the max size bytes in storage."""

  @abc.abstractmethod
  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""

  @abc.abstractmethod
  def add(self, key: Key, value: Value) -> bool:
    """Add value and return True. If storage is full, return False."""

  @abc.abstractmethod
  def retrieve(self, key: Key, device: Any = None) -> Optional[Value]:
    """Return value from storage or None if not found.

    Args:
      key: key to retrieve value.
      device:
        The same as device in jax.device_put. Retrieve the value to device.
        If device is None, retrieve the value to it's original device while saved.
    Returns:
      Value retrieved from storage or None if not found.
    """

  @abc.abstractmethod
  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage."""

  @abc.abstractmethod
  def contains(self, key: Key) -> bool:
    """If there is key in storage."""


class BasicStorage:
  """Basic implement calculating size and save value into dict without modify."""

  def __init__(self, max_size_bytes: int):
    """
    Args:
      max_size_bytes: Maximum bytes use
    """
    self._max_size_bytes = max_size_bytes
    self._remain_size_bytes = max_size_bytes
    self._saved_values: dict[Key, Value] = {}

  def get_max_size_bytes(self) -> int:
    return self._max_size_bytes

  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""
    return self._remain_size_bytes >= needed_bytes

  def add(self, key: Key, value: Value) -> bool:
    """Add value and return True. If storage is full, return False.

    The value will not copied. Be aware not to modify the value after add to storage.
    Storage is expected to have enough space.
    """
    if not self.has_enough_space(value.prefix_size_bytes):
      logger.warning(
          "should check enough space before add to storage, but remain=%d not enough for value=%d",
          self._remain_size_bytes,
          value.prefix_size_bytes,
      )
      return False

    self._saved_values[key] = value
    self._remain_size_bytes -= value.prefix_size_bytes
    return True

  def retrieve(self, key: Key) -> Optional[Value]:
    """Return value from storage or None if not found.

    Be aware the storage is not return a copy. Clone the Value first if additional modification needed.
    """
    if key not in self._saved_values:
      logger.warning("key=%r should exist in storage before retrieve, but not found", key)
      return None
    return self._saved_values[key]

  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage.
    Key is expected to be found.
    """
    if key not in self._saved_values:
      logger.warning("key=%r should exist in storage before evict, but not found", key)
      return None
    value = self._saved_values.pop(key)
    self._remain_size_bytes += value.prefix_size_bytes
    return value

  def contains(self, key: Key) -> bool:
    """If there is key in storage."""
    return key in self._saved_values


class HBMStorage(ValueStorageInterface):
  """Stores kv storage values in HBM.

  Store the Value into the specific HBM device, which is the same type as device in jax.device_put.
  The Value would be jax.device_put to the HBM device after add, and retrieve back to the original device.
  """

  def __init__(self, max_size_bytes: int, device: Any = None):
    """Init the HBMStorage with max size limit and device to store the Value.

    Args:
      max_size_bytes: Maximum bytes of HBM to use for storage
      device:
        the same type as jax.device_put. It is used to store the cache Value.
        If None, do not move the Value.
    """
    self._storage = BasicStorage(max_size_bytes)
    self._device = device

  def get_max_size_bytes(self) -> int:
    return self._storage.get_max_size_bytes()

  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""
    return self._storage.has_enough_space(needed_bytes)

  def add(self, key: Key, value: Value) -> bool:
    """Add key/value pair into the cache.

    Depend on jax.device_put,
    the Value will not be copied if the device storing the cache is the same as the origin device of Value.
    Storage is expected to have enough space.

    Args:
      key: key of cache index.
      value: Value to store.
    Returns:
      True if successful. False if failed due to not enough space.
    """
    hbm_value = device_put_value(value, self._device)
    return self._storage.add(key, hbm_value)

  def retrieve(self, key: Key, device: Any = None) -> Optional[Value]:
    """Retrieve value back to the original device or None if not found.

    Be aware the storage may not return a copy if the original devices is the same as depend on jax.device_put.
    Key is expected to be found.
    """
    hbm_value = self._storage.retrieve(key)
    if hbm_value is None:
      return None

    return device_put_value(hbm_value, device)

  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage.
    Key is expected to be found.
    """
    return self._storage.evict(key)

  def contains(self, key: Key) -> bool:
    """If there is key in storage."""
    return self._storage.contains(key)


class DRAMStorage(ValueStorageInterface):
  """Stores KV Cache values in host DRAM."""

  def __init__(self, max_size_bytes: int):
    """
    Args:
      max_size_bytes: Maximum bytes of host DRAM to use for storage
    """
    self._storage = BasicStorage(max_size_bytes)

  def get_max_size_bytes(self) -> int:
    return self._storage.get_max_size_bytes()

  def has_enough_space(self, needed_bytes: int) -> bool:
    """Calculate if needed_bytes size can add to storage."""
    return self._storage.has_enough_space(needed_bytes)

  def add(self, key: Key, value: Value) -> bool:
    """Add value into host DRAM.

    Return false if storage does not have enough space.
    Do not use this function to check if has enough space.
    This function will first move to host DRAM before check the space.
    The storage will copy to the host DRAM if originally on device,
    or with the same reference to the value if originally on host.
    Do not use the value after this function if originally on host since the value will not copy.
    """
    host_value = Value(
        prefix=jax.device_get(value.prefix),
        true_length=value.true_length,
        padded_length=value.padded_length,
        tokens=value.tokens,
        prefix_size_bytes=value.prefix_size_bytes,
        device=value.device,
    )

    return self._storage.add(key, host_value)

  def retrieve(self, key: Key, device: Any = None) -> Optional[Value]:
    """Return value from storage to the original device or None if not found.

    If the original device save in the storage is cpu, the storage will not copied.
    Do not modify the storage prefix retrieved.
    """
    host_value = self._storage.retrieve(key)
    if host_value is None:
      return None

    return device_put_value(host_value, device)

  def evict(self, key: Key) -> Optional[Value]:
    """Evict and return value, or None if key is not in storage."""
    return self._storage.evict(key)

  def contains(self, key: Key) -> bool:
    """If there is key in storage."""
    return self._storage.contains(key)


class LRUStrategy:
  """Least recently used cache strategy manage key."""

  def __init__(self):
    self._order: OrderedDict[Key, None] = OrderedDict()

  def evict(self) -> Optional[Key]:
    """Return and pop the least recently used key."""
    if len(self._order) == 0:
      return None
    return self._order.popitem(last=False)[0]

  def use(self, key: Key) -> None:
    """Updated the usage history."""
    if key not in self._order:
      self._order[key] = None
    else:
      self._order.move_to_end(key, last=True)


@dataclasses.dataclass
class StorageWithStrategy:
  """Storage with corresponding strategy"""

  storage: ValueStorageInterface
  strategy: LRUStrategy


class HierarchicalCache:
  """Hierarchical Cache contains two layers of ValueStorageInterface.

  The first layer contains subset fo key / value pairs of the second layer.
  The second storage max size bytes should >= first storage max size bytes.
  Use LRU for each layer.
  Add the Value will save to all layers.
  Retrieve the Value will retrieve to HBM and then saved to all layers.
  The added value size should less than the first layer max size.
  If the first layer max size cannot contains the added Value, add will failed.
  """

  def __init__(self, layers: Tuple[ValueStorageInterface, ValueStorageInterface]):
    assert (
        layers[0].get_max_size_bytes() <= layers[1].get_max_size_bytes()
    ), "Bottom layer of storage need to be larger than top."

    self._layers = [StorageWithStrategy(storage, LRUStrategy()) for storage in layers]

  def add(self, key: Key, value: Value) -> tuple[bool, dict[Key, Value]]:
    """Add to all layers and return (ok, dict[fully evicted from hierarchical cache key value pair]).

    Beware in some error case, there may be not ok but have some evicted key value.
    """
    needed_bytes = value.prefix_size_bytes
    if self._layers[0].storage.get_max_size_bytes() < needed_bytes:
      logging.warning(
          "Trying to add value larger than top layer max size. need_bytes=%d, max_size_bytes=%d",
          needed_bytes,
          self._layers[0].storage.get_max_size_bytes(),
      )
      return False, {}

    # Only return last layers evicted key value pair which is fully evicted from hierarchical cache.
    all_ok = True
    last_layer_evicted_key_values: dict[Key, Value] = {}
    for layer in self._layers:
      if layer.storage.contains(key):
        last_layer_evicted_key_values = {}
        continue

      ok, last_layer_evicted_key_values = self._evict_to_enough_space(layer, needed_bytes)
      all_ok = all_ok and ok

    if not all_ok:
      logging.error("Cannot evict enough space after checking max_size is enough for bytes=%d.", needed_bytes)
      return False, last_layer_evicted_key_values

    for layer in self._layers:
      if not layer.storage.contains(key):
        if not layer.storage.add(key, value):
          logging.error("Cannot add to storage. key=%r, needed_bytes=%d", key, needed_bytes)
          return False, last_layer_evicted_key_values

      layer.strategy.use(key)

    return True, last_layer_evicted_key_values

  def retrieve(self, key: Key, device: Any = None) -> Optional[Value]:
    """Retrieve from all layers and add to all layers.

    Args:
      key: key to retrieve.
      device:
        The same type as the device in jax.device_put. Return the Value put on the device.
        If None, the Value will be put on the Value.device.
    Returns:
      Value retrieved from all layers or None if not found.
      The Value.device is not changed to device retrieved.
    """
    value: Optional[Value] = None
    for layer in self._layers:
      if layer.storage.contains(key):
        value = layer.storage.retrieve(key, device)
        break

    if value is None:
      logging.warning("Should check key exist before retrieve, but fail for key=%r", key)
      return None

    for layer in self._layers:
      if not layer.storage.contains(key):
        if not self._evict_to_enough_space(layer, value.prefix_size_bytes):
          logging.error("Cannot evict enough space for retrieved Value to other layers.")
          continue

        if not layer.storage.add(key, value):
          logging.error("Cannot add retrieved Value to other layers.")
          continue

      layer.strategy.use(key)

    return value

  def _evict_to_enough_space(self, layer: StorageWithStrategy, needed_bytes: int) -> tuple[bool, dict[Key, Value]]:
    """Evict layer to enough bytes for add and return (ok, dict[evicted key, evicted value])."""
    evicted_key_values: dict[Key, Value] = {}
    while not layer.storage.has_enough_space(needed_bytes):
      evicted_key = layer.strategy.evict()
      if evicted_key is None:
        logging.error("Cannot evict enough space for bytes=%d.", needed_bytes)
        return False, evicted_key_values

      evicted_value = layer.storage.evict(evicted_key)
      if evicted_value is None:
        logging.error("Key should in storage before evict but not. key=%r", evicted_key)
        continue

      evicted_key_values[evicted_key] = evicted_value

    return True, evicted_key_values


class PrefixCache:
  """Store Prefix KV cache.

  Use hierarchical cache of two layers the first in the HBM and the second in the host DRAM.
  Assuming HBM is available, or the cache would degrade to two layers on DRAM.
  If cache is full, evict least-recently used entries (LRU).
  LRU strategy is apply to all layers.
  The cache in HBM will be subset of the cache in host DRAM.
  For example:
    For HBM can contain 2 values, and DRAM can contain 5 values,
    [1, 2, 3, 4, 5] LRU history, the [1, 2] will in HBM, and [1, 2, 3, 4, 5] will in DRAM.
  Always return cache after load into HBM.
  The value need to be <= to the max size in HBM.
  DRAM max size need to be >= than HBM max size.
  """

  def __init__(self, hbm_bytes: int, dram_bytes: int):
    """
    dram_bytes >= hbm_bytes
    Args:
      hbm_bytes: Total amount of HBM to use for cache.
      dram_bytes: Total amount of DRAM to use for cache.
    """
    # TODO(yuyanpeng): way to disable DRAM cache
    assert dram_bytes >= hbm_bytes, "DRAM max size need to be >= than HBM max size."
    self._lock = threading.Lock()
    self._hbm_bytes = hbm_bytes
    self._dram_bytes = dram_bytes
    # init in clear()
    self._trie: PrefixCacheTrie
    self._cache: HierarchicalCache
    self.clear()

  def fetch_longest_common_prefix_key(self, key: Key) -> Optional[Key]:
    """Returns key with longest common prefix matched or None if not found."""
    logger.debug("fetch_longest_common_prefix_key, key=%r", key)
    with self._lock:
      matched_key = self._trie.get_longest_common_prefix_key(key)
      logger.debug("matched_key=%r", matched_key)
      return matched_key

  def save(self, key: Key, value: Value) -> bool:
    """Save key/value to the cache."""
    logger.debug("save key=%r", key)
    with self._lock:
      ok, evicted = self._cache.add(key, value)
      for evicted_key in evicted.keys():
        self._trie.erase(evicted_key)
      if not ok:
        logger.warning("Cannot add to cache")
        return False
      self._trie.insert(key)
      return True

  def load(self, key: Key, device: Any = None) -> Optional[Value]:
    """Returns Value stored with key or None if not found.

    Args:
      key: key to load.
      device:
        The same type as device in the jax.device_put. Load the Value on the device.
        If None, load the Value on the original device Value.device.
    Return:
      Value stored with key or None if not found.
      The Value.device is not changed to device loaded on.
    """
    logger.debug("load key=%r", key)
    with self._lock:
      value = self._cache.retrieve(key, device)
      if value is None:
        logger.warning(
            "The key should fetched by fetch_longest_common_prefix_key, load key=%r should be valid but not.", key
        )
        return None
      return value

  def clear(self):
    """Clear entire cache."""
    logger.debug("clear cache")
    with self._lock:
      self._trie = PrefixCacheTrie()
      self._cache = HierarchicalCache(
          layers=(
              HBMStorage(self._hbm_bytes),
              DRAMStorage(self._dram_bytes),
          )
      )
