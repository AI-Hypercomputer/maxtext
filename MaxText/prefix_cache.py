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

# HBM size with Trillium 8 core use about 17 / 31.25 GB per core after loading models.
# To utilize HBM memory to 80%, about (30 * 0.8 - 16) * 8 = 64GB
# Prefix return by prefill function of mixtral-8x22b model with max_prefill_length=1024,
# int8 quantize_kvcache is 235_930_060 bytes, nearly 256 MB.
# HBM cache can store about 64GB / 256 MB = 256 prompts.
hbm_bytes = 64 * 1024 * 1024 * 1024  # 64 GB
prefix_cache = PrefixCache(hbm_bytes)

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
   matched_len, cached_prefix, tokens = cache_results
   key = full_key[match_len:]
else:
   cached_prefix = None

iii) Run prefill un-cached prompt and write to cache

full_kv_cache = prefill(prompt, cached_prefix)

# if the cache didn't contain prefix or if it was a partial match
if cached_prefix is None or matched_len != orig_key_len:
    prefix_cache.save(full_key, Value(full_kv_cache, orig_key_len))

"""

from typing import Tuple, Any, Optional
import dataclasses
import jax
import jax.numpy as jnp

import max_logging

Token = int
# Tuple of tokens from prompt
Key = Tuple[Token, ...]
Prefix = Any  # KVCache for one prompt


class Value:
  """Object stored contains the actual KVcache

  Attributes:
    prefix:
      Readonly. Prefix Cache using in model. Should be dictionary of jnp.array.
    true_length:
      Readonly. True length of tokens calculate prefix. Should be <= than len(tokens).
      true_length will be min(true_length, len(tokens))
    padded_length:
      Readonly. Length of tokens including padding calculate prefix.
    tokens:
      Readonly. Tokens calculate prefix. may include partial of padding.
    prefix_size_bytes:
      Readonly. bytes of prefix.
  """

  def __init__(self, *, prefix: Prefix, true_length: int, padded_length: int, tokens: list[int]):
    self._prefix = prefix
    self._prefix_size_bytes: int = self._calculate_prefix_bytes(prefix)
    self._true_length = self._maybe_adjust_true_length(true_length, tokens)
    self._padded_length = padded_length
    self._tokens = tokens

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
  def tokens(self) -> list[int]:
    return self._tokens

  @property
  def prefix_size_bytes(self) -> int:
    return self._prefix_size_bytes

  def clone(self) -> "Value":
    """Clone to prevent use the same jax array."""
    copied_prefix = jax.tree.map(lambda x: x.copy() if isinstance(x, jax.Array) else x, self._prefix)
    return Value(prefix=copied_prefix, true_length=self._true_length, padded_length=self._padded_length, tokens=self._tokens)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Value):
      return False
    return (
        other.padded_length == self.padded_length
        and other.tokens == self.tokens
        and jax.tree.all(jax.tree.map(jnp.array_equal, other.prefix, self.prefix))
        and other.prefix_size_bytes == self.prefix_size_bytes
    )

  def _calculate_prefix_bytes(self, prefix: Prefix) -> int:
    def has_nbytes_int(obj) -> bool:
      return hasattr(obj, "nbytes") and isinstance(obj.nbytes, int)

    # calculate all bytes of jnp.array in the prefix
    return jax.tree.reduce(
        lambda acc, array: acc + (array.nbytes if has_nbytes_int(array) else 0),
        prefix,
        0,
    )

  def _maybe_adjust_true_length(self, true_length: int, tokens: list[int]) -> int:
    if true_length > len(tokens):
      max_logging.log(f"WARNING: {true_length=} should <= {len(tokens)=}.")

    return min(true_length, len(tokens))


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
        node.children[token] = PrefixCacheTrie.Node()
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

    def erase_from(node: PrefixCacheTrie.Node, idx: int) -> bool:
      """
      Erase node traverse from `node` and start from key index `idx`. Return true if the `node` is leaf needed deleted.
      """
      if idx >= len(key):
        return node.is_leaf()

      if key[idx] not in node.children:
        return False

      if erase_from(node.children[key[idx]], idx + 1):
        del node.children[key[idx]]

      return node.is_leaf()

    erase_from(self._root, 0)


class HBMCache:
  """Stores kv cache values in HBM and supports eviction using LRU.

  Cache is distributed across all devices on the VM.
  """

  def __init__(self, max_size_bytes: int):
    """
    Args:
      max_size_bytes: Maximum bytes of HBM to use for cache
    """
    raise NotImplementedError

  def is_enough_space_remain(self, value: Value) -> bool:
    """Calculate if value size can add to cache."""
    raise NotImplementedError

  def add_to_cache(self, key: Key, value: Value) -> bool:
    """
    Return False if cache is full.
    """
    raise NotImplementedError

  def retrieve_from_cache(self, key: Key) -> Optional[Value]:
    """
    Return None if key is not found.
    """
    raise NotImplementedError


class PrefixCache:
  """Store Prefix KV cache.

  If cache is full, evict least-recently used entries (LRU).
  """

  def __init__(self, hbm_bytes: int):
    """
    Args:
      hbm_bytes: Total amount of HBM to use for cache.
    """
    self.hbm_cache = HBMCache(max_size_bytes=hbm_bytes)
    self.trie = PrefixCacheTrie()

  def fetch_longest_common_prefix_key(self, key: Key) -> Optional[Key]:
    """Returns key with longest common prefix matched or None if not found."""
    return self.trie.get_longest_common_prefix_key(key)

  def save(self, key: Key, value: Value) -> bool:
    """Save key/value to the cache."""
    if not self.hbm_cache.is_enough_space_remain(value):
      self._evict_least_recently_used_cache()
    if not self.hbm_cache.add_to_cache(key, value):
      return False
    return self.trie.insert(key)

  def load(self, key: Key) -> Optional[Value]:
    """Returns Value stored with key or None if not found."""
    return self.hbm_cache.retrieve_from_cache(key)

  def clear(self):
    """Clear entire cache."""
    raise NotImplementedError

  def _evict_least_recently_used_cache(self):
    raise NotImplementedError
