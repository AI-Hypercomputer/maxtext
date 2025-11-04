
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable: bool = False

    def __init__(self):
        self.keys: Optional[Array] = None
        self.values: Optional[Array] = None
        self.device: Optional[jax.Device] = None

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: Array): ...

    @abstractmethod
    def update(
        self, key_states: Array, value_states: Array, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> Tuple[Array, Array]: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: Array) -> Tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.keys is not None:
            self.keys = jax.device_put(self.keys, jax.devices("cpu")[0])
            self.values = jax.device_put(self.values, jax.devices("cpu")[0])

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if self.keys is not None and self.device is not None and self.keys.device() != self.device:
            self.keys = jax.device_put(self.keys, self.device)
            self.values = jax.device_put(self.values, self.device)

    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""
        if self.keys is not None:
            self.keys = jnp.zeros_like(self.keys)
            self.values = jnp.zeros_like(self.values)
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0

    def reorder_cache(self, beam_idx: Array) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            # JAX arrays are immutable, so we create new arrays by indexing.
            # The device placement of beam_idx is handled automatically by JAX.
            self.keys = self.keys[beam_idx]
            self.values = self.values[beam_idx]

from typing import Any, List, Optional, Type

import jax
import jax.numpy as jnp

# Re-used module.
# from generated_code.Qwen3MoeForCausalLM.cache_utils import CacheLayerMixin
from generated_code.Qwen3MoeForCausalLM.cache_utils import CacheLayerMixin


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Parameters:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
    """

    def __init__(
        self,
        layers: Optional[List[CacheLayerMixin]] = None,
        layer_class_to_replicate: Optional[Type[CacheLayerMixin]] = None,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers is not None else []
        self.layer_class_to_replicate = layer_class_to_replicate

    def __repr__(self):
        return f"{self.__class__.__name__}(layers={self.layers})"

    def update(
        self,
        key_states: jax.Array,
        value_states: jax.Array,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`jax.Array`):
                The new key states to cache.
            value_states (`jax.Array`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        return keys, values

    def early_initialization(self, batch_size: int, num_heads: int, head_dim: int, dtype: jnp.dtype):
        """
        Initialize all the layers in advance (it's otherwise lazily initialized on the first `update` call).
        This is useful for our `export` recipes, as `export` needs everything in advance.
        """
        # Note that the initialization needs all dimensions (except -2), as well as dtype, so we use
        # this fake tensor approach. It has size 0 on the -2 dimension, so it does not allocate any data (it only
        # creates an empty tensor with correct shape and dtype), which is very efficient and practical
        fake_keys_tensor = jnp.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype)
        # Init all layers
        for layer in self.layers:
            layer.lazy_initialization(fake_keys_tensor)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: jax.Array, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, the size is
        # simply the shape of `cache_position`
        if layer_idx >= len(self.layers):
            return cache_position.shape[0], 0
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length."""
        # For DynamicCache, where the layers are created at runtime -> if it was not yet created, return -1
        # as DynamicLayer does
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def reset(self):
        """Recursively reset all layers tensors"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reset()

    def reorder_cache(self, beam_idx: jax.Array):
        """Reorder the cache for beam search"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].reorder_cache(beam_idx)

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].crop(max_length)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat and interleave the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: jax.Array):
        """Select indices from the cache"""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].batch_select_indices(indices)

    @property
    def max_batch_size(self) -> int:
        """Return the maximum batch size of the cache"""
        values = [layer.max_batch_size for layer in self.layers]
        if len(set(values)) > 1:
            raise ValueError(f"Max batch size is not consistent across layers: {values}")
        return values[0]

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache"""
        values = [layer.max_cache_len for layer in self.layers]
        return max(values)

    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        # For DynamicCache dispatching the layers lazily (otherwise, all([]) is True)
        if len(self.layers) == 0:
            return False
        return all(layer.is_compileable for layer in self.layers)

    @property
    def is_sliding(self) -> List[bool]:
        """Return whether the layers of the cache are sliding window"""
        return [getattr(layer, "is_sliding", False) for layer in self.layers]

    def __getitem__(self, layer_idx: int) -> tuple[jax.Array, jax.Array]:
        """
        Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_values` iteration, e.g. `for x in past_key_values:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layers[layer_idx].keys, self.layers[layer_idx].values)

    def __len__(self):
        """
        This value corresponds to the number of layers in the model.
        """
        # Note: for DynamicCache, layers are initialized lazily, so this will not be accurate before the first
        # forward through all the layers
        return len(self.layers)

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

# Re-used from 'generated_code.Qwen3MoeForCausalLM.cache_utils.CacheLayerMixin'
from .cache_utils import CacheLayerMixin


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """

    is_sliding: bool = False

    def lazy_initialization(self, key_states: jax.Array):
        self.dtype, self.device = key_states.dtype, key_states.device
        empty_shape = (*key_states.shape[:-2], 0, key_states.shape[-1])
        self.keys = jnp.zeros(empty_shape, dtype=self.dtype)
        self.values = jnp.zeros(empty_shape, dtype=self.dtype)

    def update(
        self,
        key_states: jax.Array,
        value_states: jax.Array,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Update the key and value caches in-place, and return the necessary kes and value states.

        Args:
            key_states (`jax.Array`): The new key states to cache.
            value_states (`jax.Array`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`jax.Array`, `jax.Array`]: The key and value states.
        """
        # Lazy initialization
        if self.keys is None:
            self.lazy_initialization(key_states)

        self.keys = jnp.concatenate([self.keys, key_states], axis=-2)
        self.values = jnp.concatenate([self.values, value_states], axis=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: jax.Array) -> Tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.size == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        """Returns the maximum sequence length of the cache object. DynamicLayer does not have a maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be negative
        to remove `max_length` tokens.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat the cache `repeats` times in the batch dimension."""
        if self.get_seq_length() > 0:
            self.keys = jnp.repeat(self.keys, repeats, axis=0)
            self.values = jnp.repeat(self.values, repeats, axis=0)

    def batch_select_indices(self, indices: jax.Array) -> None:
        """Only keep the `indices` in the batch dimension of the cache."""
        if self.get_seq_length() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]

# Copyright 2024 Google LLC
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

"""A JAX port of the DynamicSlidingWindowLayer from HuggingFace."""

from typing import Any, Optional

import jax.numpy as jnp
from jax import Array

# Re-used module from generated_code.Qwen3MoeForCausalLM.cache_utils.DynamicLayer
from .dynamic_layer import DynamicLayer


class DynamicSlidingWindowLayer(DynamicLayer):
  """
  A cache layer that grows dynamically as more tokens are generated, up until
  the sliding window size. It stores the key and value states as tensors of
  shape `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
  """

  is_sliding = True

  def __init__(self, sliding_window: int):
    super().__init__()
    self.sliding_window: int = sliding_window
    self.cumulative_length: int = 0

  def update(
      self,
      key_states: Array,
      value_states: Array,
      cache_kwargs: Optional[dict[str, Any]] = None,
  ) -> tuple[Array, Array]:
    """
    Update the key and value caches in-place, and return the necessary kes and
    value states.

    Args:
        key_states (`Array`): The new key states to cache.
        value_states (`Array`): The new value states to cache.
        cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for
          the cache.

    Returns:
        tuple[`Array`, `Array`]: The key and value states.
    """
    # Lazy initialization
    if self.keys is None:
      self.lazy_initialization(key_states)

    self.cumulative_length += key_states.shape[-2]

    # Compute the full states
    full_key_states = jnp.concatenate([self.keys, key_states], axis=-2)
    full_value_states = jnp.concatenate([self.values, value_states], axis=-2)
    # Only cache the last `self.sliding_window - 1` tokens (or all of them if
    # lower than that)
    self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

    # Return the full states
    return full_key_states, full_value_states

  def get_mask_sizes(self, cache_position: Array) -> tuple[int, int]:
    """Return the length and offset of the cache, used to generate the attention mask"""
    query_length = cache_position.shape[0]
    first_cache_position = cache_position[0]

    kv_offset = jnp.clip(
        first_cache_position - self.sliding_window + 1, a_min=0
    )

    if self.get_seq_length() >= self.sliding_window:
      kv_length = self.sliding_window - 1 + query_length
    else:
      kv_length = self.get_seq_length() + query_length

    return kv_length, int(kv_offset)

  def get_seq_length(self) -> int:
    """Returns the sequence length of the cached states."""
    return self.cumulative_length

  def get_max_cache_shape(self) -> int:
    """Return the maximum cache shape of the cache"""
    return self.sliding_window

  def crop(self, max_length: int) -> None:
    """
    Crop the past key values up to a new `max_length` in terms of tokens.
    `max_length` can also be negative to remove `max_length` tokens.
    """
    if self.get_seq_length() >= self.sliding_window:
      raise ValueError(
          "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more"
          " tokens than its sliding window (otherwise some states are lost)"
      )
    super().crop(max_length)
    self.cumulative_length = self.keys.shape[-2]

# Copyright 2024 Google LLC
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

"""A JAX port of the DynamicCache from HuggingFace."""

from typing import Any, Iterable, Optional, Tuple

from jax import Array

# Re-used modules from generated_code.Qwen3MoeForCausalLM.cache_utils
from .cache_utils import Cache, DynamicLayer, DynamicSlidingWindowLayer


class DynamicCache(Cache):
  """
    A cache that grows dynamically as more tokens are generated. This is the
    default for generative models. It stores the key and value states as a list
    of `CacheLayer`, one for each layer. The expected shape for each tensor in
    the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`. If a
    config is passed, it will additionally check for sliding or hybrid cache
    structure, greatly reducing the memory requirement of the cached tensors to
    `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache
    classes.

    Example:

    