
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import jax
from jax import numpy as jnp

Array = jax.Array


class CacheLayerMixin(ABC):
  """Base, abstract class for a single layer's cache."""

  is_compileable = False

  def __init__(self):
    self.keys: Optional[Array] = None
    self.values: Optional[Array] = None

  def __repr__(self):
    return f"{self.__class__.__name__}"

  @abstractmethod
  def lazy_initialization(self, key_states: Array):
    ...

  @abstractmethod
  def update(
      self,
      key_states: Array,
      value_states: Array,
      cache_kwargs: Optional[dict[str, Any]] = None,
  ) -> Tuple[Array, Array]:
    ...

  @abstractmethod
  def get_mask_sizes(self, cache_position: Array) -> Tuple[int, int]:
    ...

  @abstractmethod
  def get_seq_length(self) -> int:
    ...

  @abstractmethod
  def get_max_cache_shape(self) -> int:
    ...

  # offload and prefetch are PyTorch-specific device management features.
  # They don't have a direct, idiomatic equivalent in JAX for this context.
  # JAX manages device placement more explicitly, often at a higher level.
  # Omitting these methods.

  def reset(self) -> None:
    """Resets the cache values while preserving the objects."""
    if self.keys is not None:
      # JAX arrays are immutable, so we can't use in-place operations like .zero_()
      # Instead, we create new zeroed arrays of the same shape and type.
      self.keys = jnp.zeros_like(self.keys)
      self.values = jnp.zeros_like(self.values)
    # This attribute is set on several Layers
    if hasattr(self, "cumulative_length"):
      self.cumulative_length = 0

  def reorder_cache(self, beam_idx: Array) -> None:
    """Reorders this layer's cache for beam search."""
    if self.get_seq_length() > 0:
      # PyTorch's index_select is equivalent to JAX's array indexing.
      # Device placement is handled implicitly by JAX.
      self.keys = self.keys[beam_idx]
      self.values = self.values[beam_idx]

from typing import Any, List, Optional, Tuple, Type

import jax.numpy as jnp
from jax import Array

# Reused from Qwen3ForCausalLM.cache.CacheLayerMixin
from MaxText.layers.qwen3.cache import CacheLayerMixin


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
      raise ValueError("You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache.")
    self.layers = layers if layers is not None else []
    self.layer_class_to_replicate = layer_class_to_replicate
    # Offloading logic is PyTorch-specific and not translated.

  def __repr__(self):
    return f"{self.__class__.__name__}(layers={self.layers})"

  def update(
      self,
      key_states: Array,
      value_states: Array,
      layer_idx: int,
      cache_kwargs: Optional[dict[str, Any]] = None,
  ) -> Tuple[Array, Array]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`Array`):
            The new key states to cache.
        value_states (`Array`):
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

  def get_mask_sizes(self, cache_position: Array, layer_idx: int) -> Tuple[int, int]:
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

  def reorder_cache(self, beam_idx: Array):
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

  def batch_select_indices(self, indices: Array):
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

  def __getitem__(self, layer_idx: int) -> Tuple[Array, Array]:
    """
    Support for backwards-compatible `past_key_values` indexing, e.g. `past_key_values[0][0].shape[2]` to get the
    sequence length.
    """
    if layer_idx < len(self.layers):
      return self.layers[layer_idx].keys, self.layers[layer_idx].values
    else:
      raise KeyError(f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}")

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
