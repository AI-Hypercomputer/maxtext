#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Page Managers for implementing paged attention in MaxText.

This module provides the `PageManager` class and associated `PageState` dataclass
for managing the paged attention mechanism. The paging system allows efficient
handling of variable-length sequences by dividing the attention context into
fixed-size pages, similar to virtual memory systems in operating systems. Each
sequence is assigned to a request_id, and pages are allocated to these requests
as needed.
"""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
import common_types

Config = common_types.Config


@struct.dataclass
class PageState:
  """Represents the state of memory pages managed by the `PageManager`.

  This dataclass tracks the allocation status of each page, the mapping of
  pages to request groups, and the current position within each sequence's pages.

  The first dimension of all arrays is the layer index, as each layer has its
  own independent set of pages and page management state.

  Attributes:
    page_status: A `jnp.ndarray` of shape `[num_layers, num_pages]`.  Each
      element indicates whether the corresponding page is free (0) or allocated (1).
    page_map: A `jnp.ndarray` of shape `[num_layers, max_page_groups,
      max_pages_per_group]`.  This array maps each page group to the indices of
      its allocated pages.  A value of -1 indicates an unassigned page index.
    sequence_lengths: A `jnp.ndarray` of shape `[num_layers, max_page_groups]`.
      This array stores the current length of each sequence (in tokens).
    num_pages_used: A `jnp.ndarray` of shape `[num_layers, max_page_groups]`.
      This array tracks the number of pages currently allocated to each page group.
    current_page: A `jnp.ndarray` of shape `[num_layers, max_page_groups]`.
      This array stores the index of the *currently active* page for each page
      group. A value of -1 indicates that no page is currently active (e.g.,
      before prefill or after the sequence has finished).
    current_page_position: A `jnp.ndarray` of shape `[num_layers,
      max_page_groups]`. This array stores the index (offset) of the next
      available token index within the `current_page` for each page group.
  """

  page_status: jnp.ndarray
  page_map: jnp.ndarray
  sequence_lengths: jnp.ndarray
  num_pages_used: jnp.ndarray
  current_page: jnp.ndarray
  current_page_position: jnp.ndarray


def initialize_page_state(
    num_layers: int,
    num_pages: int,
    max_page_groups: int,
    max_pages_per_group: int,
) -> PageState:
  """Creates and initializes a `PageState` object.

  All pages are initially marked as free, and no pages are assigned to any group.

  Args:
    num_layers: The number of layers in the model.
    num_pages: The total number of available pages.
    max_page_groups: The maximum number of page groups.
    max_pages_per_group: The maximum number of pages that can be allocated to
      a single page group.

  Returns:
    An initialized `PageState` object.

  Example:
    ```python
    # Initialize a page state for a model with 2 layers
    state = initialize_page_state(
        num_layers=2,
        num_pages=128,
        max_page_groups=4,
        max_pages_per_group=32
    )
    ```
  """
  return PageState(
      page_status=jnp.zeros((num_layers, num_pages), dtype=jnp.int32),
      page_map=jnp.full((num_layers, max_page_groups, max_pages_per_group), -1, dtype=jnp.int32),
      sequence_lengths=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
      num_pages_used=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
      current_page=jnp.full((num_layers, max_page_groups), -1, dtype=jnp.int32),
      current_page_position=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
  )


def _find_next_free_page_index(page_status: jnp.ndarray) -> jnp.ndarray:
  """Finds the index of the next available free page in a single layer.

  Args:
    page_status: A 1D `jnp.ndarray` representing the status of pages in a
      single layer (0 for free, 1 for allocated).

  Returns:
    The index of the next free page, or -1 if no free pages are found.  If
    multiple pages are free, the one with the lowest index is returned.
  """
  free_mask = page_status == 0
  next_free = jnp.argmax(free_mask)  # Returns 0 if all False (all allocated)
  has_free = jnp.any(free_mask)
  return jnp.where(has_free, next_free, -1)


def _update_prefill_pages_for_group(
    page_state: PageState,
    page_group_id: int,
    true_length: int,
    layer_id: int,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Reserves pages for a page group during the prefill stage for a specific layer.

  This function allocates the required number of pages for a given page group
  to store the initial sequence (prefill). If there are not enough free pages 
  available, no new pages are allocated and the state is returned with the
  existing pages for this group released.

  Args:
    page_state: The current `PageState`.
    page_group_id: The ID of the page group to reserve pages for.
    true_length: The actual length of the sequence (in tokens).
    layer_id: The index of the layer to allocate pages in.
    tokens_per_page: The number of tokens that can be stored in a single page.
    max_pages_per_group: The maximum number of pages allowed per group.

  Returns:
    The updated `PageState` after reserving pages.
  """
  # First, release any existing pages for this group in this layer
  layer_page_status = page_state.page_status[layer_id]
  layer_page_map = page_state.page_map[layer_id]
  
  def release_existing_pages(index: int, state: Tuple[jnp.ndarray, jnp.ndarray]):
    current_status, current_map = state
    page_index = current_map[page_group_id, index]
    
    # Use jax.lax.cond instead of jnp.where for better traceability
    updated_status = jax.lax.cond(
        page_index >= 0,
        lambda _: current_status.at[page_index].set(0),
        lambda _: current_status,
        None
    )
    
    updated_map = jax.lax.cond(
        page_index >= 0,
        lambda _: current_map.at[page_group_id, index].set(-1),
        lambda _: current_map,
        None
    )
    
    return (updated_status, updated_map)

  layer_page_status, layer_page_map = jax.lax.fori_loop(
      0, max_pages_per_group, release_existing_pages, (layer_page_status, layer_page_map)
  )
  
  # Handle zero length case
  def handle_zero_length(_):
    return PageState(
        page_status=page_state.page_status.at[layer_id].set(layer_page_status),
        page_map=page_state.page_map.at[layer_id].set(layer_page_map),
        sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(0),
        num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(0),
        current_page=page_state.current_page.at[layer_id, page_group_id].set(-1),
        current_page_position=page_state.current_page_position.at[layer_id, page_group_id].set(0),
    )
  
  # Handle normal case
  def handle_normal_case(_):
    # Calculate how many pages we need
    num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
    last_page_position = (true_length - 1) % tokens_per_page

    # Check if we have enough free pages
    num_free_pages = jnp.sum(layer_page_status == 0)
    has_enough_pages = num_free_pages >= num_pages_needed
    
    # Allocate new pages if we have enough free pages
    def allocate_pages(_):
      def allocate_new_page(index: int, state: Tuple[jnp.ndarray, jnp.ndarray]):
        current_status, current_map = state
        next_free_page = _find_next_free_page_index(current_status)
        
        # Check if we should allocate this page
        should_allocate = jnp.logical_and(index < num_pages_needed, next_free_page >= 0)
        
        # Update status and map using jax.lax.cond
        updated_status = jax.lax.cond(
            should_allocate,
            lambda _: current_status.at[next_free_page].set(1),
            lambda _: current_status,
            None
        )
        
        updated_map = jax.lax.cond(
            should_allocate,
            lambda _: current_map.at[page_group_id, index].set(next_free_page),
            lambda _: current_map,
            None
        )
        
        return (updated_status, updated_map)

      new_page_status, new_page_map = jax.lax.fori_loop(
          0, max_pages_per_group, allocate_new_page, (layer_page_status, layer_page_map)
      )

      # For the current page, use the last allocated page
      last_page_idx = jnp.minimum(num_pages_needed, max_pages_per_group) - 1
      
      # Get the page index at the last used index
      last_page_index = new_page_map[page_group_id, last_page_idx]
      
      # Make sure we handle the case where we have no pages
      last_page_index = jax.lax.cond(
          num_pages_needed > 0,
          lambda _: last_page_index,
          lambda _: jnp.array(-1, dtype=jnp.int32),
          None
      )

      return PageState(
          page_status=page_state.page_status.at[layer_id].set(new_page_status),
          page_map=page_state.page_map.at[layer_id].set(new_page_map),
          sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(true_length),
          num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(
              jnp.minimum(num_pages_needed, max_pages_per_group)
          ),
          current_page=page_state.current_page.at[layer_id, page_group_id].set(last_page_index),
          current_page_position=page_state.current_page_position.at[layer_id, page_group_id].set(last_page_position),
      )
    
    # If we don't have enough pages, just return a state with cleared resources
    def return_cleared_state(_):
      return PageState(
          page_status=page_state.page_status.at[layer_id].set(layer_page_status),
          page_map=page_state.page_map.at[layer_id].set(layer_page_map),
          sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(0),
          num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(0),
          current_page=page_state.current_page.at[layer_id, page_group_id].set(-1),
          current_page_position=page_state.current_page_position.at[layer_id, page_group_id].set(0),
      )
    
    return jax.lax.cond(has_enough_pages, allocate_pages, return_cleared_state, None)
  
  return jax.lax.cond(true_length == 0, handle_zero_length, handle_normal_case, None)


def _update_decode_pages_for_layer(
    page_state: PageState,
    layer_id: int,
    tokens_per_page: int,
) -> PageState:
  """Updates pages for autoregressive decoding for a specific layer.

  During decoding, one token is generated at a time. This function checks if
  the current page for each active page group has space for the next token. If
  the current page is full, a new page is allocated (if available). If no new
  page is available, the page group continues to use its current page, which
  may result in earlier tokens being overwritten.

  Args:
    page_state: The current `PageState`.
    layer_id: The index of the layer to allocate pages in.
    tokens_per_page: The number of tokens that can be stored in a single page.

  Returns:
    The updated `PageState` after incrementing sequence lengths and allocating
    new pages as needed.
  """
  layer_page_status = page_state.page_status[layer_id]
  layer_page_map = page_state.page_map[layer_id]
  layer_sequence_lengths = page_state.sequence_lengths[layer_id]
  layer_pages_used = page_state.num_pages_used[layer_id]
  layer_current_page = page_state.current_page[layer_id]
  layer_current_position = page_state.current_page_position[layer_id]

  # Update sequence length and position within the page.
  # Only increment sequence length for active requests (current_page >= 0)
  new_sequence_lengths = layer_sequence_lengths + jnp.where(layer_current_page >= 0, 1, 0)
  new_current_position = (new_sequence_lengths - 1) % tokens_per_page
  new_pages_needed = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page

  # Create a new PageState for this layer
  new_page_status = layer_page_status
  new_page_map = layer_page_map
  new_pages_used = layer_pages_used
  new_current_page = layer_current_page

  # Process each active group
  max_page_groups = layer_sequence_lengths.shape[0]

  def update_group(group_index: int, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]):
    """Updates page allocation for a single group."""
    current_status, current_map, current_pages, pages_used = state

    # Check if a new page is needed for this group.
    needs_new_page = jnp.logical_and(
        new_pages_needed[group_index] > pages_used[group_index], 
        current_pages[group_index] >= 0
    )
    
    # Find next free page if needed
    next_free_page = jax.lax.cond(
        needs_new_page,
        lambda _: _find_next_free_page_index(current_status),
        lambda _: jnp.array(-1, dtype=jnp.int32),
        None
    )
    
    # Allocate a new page only if needed and available
    should_allocate = next_free_page >= 0
    
    # Update status if allocating a new page
    updated_status = jax.lax.cond(
        should_allocate, 
        lambda _: current_status.at[next_free_page].set(1), 
        lambda _: current_status,
        None
    )
    
    # Update map if allocating a new page
    updated_map = jax.lax.cond(
        should_allocate, 
        lambda _: current_map.at[group_index, pages_used[group_index]].set(next_free_page), 
        lambda _: current_map,
        None
    )
    
    # Update current page if allocating a new page
    updated_pages = jax.lax.cond(
        should_allocate, 
        lambda _: current_pages.at[group_index].set(next_free_page), 
        lambda _: current_pages,
        None
    )
    
    # Update pages used if allocating a new page
    updated_used = jax.lax.cond(
        should_allocate, 
        lambda _: pages_used.at[group_index].set(pages_used[group_index] + 1), 
        lambda _: pages_used,
        None
    )
    
    return (updated_status, updated_map, updated_pages, updated_used)

  # Apply the update to all groups in the layer
  new_page_status, new_page_map, new_current_page, new_pages_used = jax.lax.fori_loop(
      0, max_page_groups, update_group, (new_page_status, new_page_map, new_current_page, new_pages_used)
  )

  # Update the overall PageState with the modified layer
  return PageState(
      page_status=page_state.page_status.at[layer_id].set(new_page_status),
      page_map=page_state.page_map.at[layer_id].set(new_page_map),
      sequence_lengths=page_state.sequence_lengths.at[layer_id].set(new_sequence_lengths),
      num_pages_used=page_state.num_pages_used.at[layer_id].set(new_pages_used),
      current_page=page_state.current_page.at[layer_id].set(new_current_page),
      current_page_position=page_state.current_page_position.at[layer_id].set(new_current_position),
  )


def _release_page_group_in_layer(
    page_state: PageState,
    page_group_id: int,
    layer_id: int,
) -> PageState:
  """Releases all pages associated with a given page group in a specific layer.

  This function iterates through all pages assigned to the specified page group
  in the given layer and marks them as free in the page_status array. It also
  resets the page_map entries and other state variables for the page group.

  Args:
    page_state: The current `PageState`.
    page_group_id: The ID of the page group to release resources for.
    layer_id: The layer in which to release pages.

  Returns:
    The updated `PageState` after releasing the specified page group's resources.
  """
  # Get current layer state
  layer_page_status = page_state.page_status[layer_id]
  layer_page_map = page_state.page_map[layer_id]
  group_pages = layer_page_map[page_group_id]
  
  # For each used page, update the page status to mark it as free
  def release_page(i, page_status):
    page_idx = group_pages[i]
    return jax.lax.cond(
        page_idx >= 0,
        lambda _: page_status.at[page_idx].set(0),
        lambda _: page_status,
        None
    )
  
  new_layer_page_status = jax.lax.fori_loop(
      0, group_pages.shape[0], release_page, layer_page_status
  )
  
  # Reset the page map for this group to -1 (no pages assigned)
  new_layer_page_map = layer_page_map.at[page_group_id].set(jnp.full_like(group_pages, -1))
  
  # Reset other group-specific state
  return PageState(
      page_status=page_state.page_status.at[layer_id].set(new_layer_page_status),
      page_map=page_state.page_map.at[layer_id].set(new_layer_page_map),
      sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(0),
      num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(0),
      current_page=page_state.current_page.at[layer_id, page_group_id].set(-1),
      current_page_position=page_state.current_page_position.at[layer_id, page_group_id].set(0),
  )


class PageManager:
  """Manages the allocation and release of pages for paged attention.

  This class provides an interface for reserving pages during prefill and
  decoding, and for releasing pages when a sequence is complete. It encapsulates
  the logic for tracking page allocation and managing the `PageState`.

  Example:
    ```python
    # Initialize a PageManager
    page_manager = PageManager(config)

    # Get initial page state
    state = page_manager.get_initial_page_state()

    # Update pages for prefill
    state = page_manager.update_prefill_pages(
        page_state=state,
        request_id=0,
        true_length=16
    )

    # Update pages for decode step
    state = page_manager.update_decode_pages(state)

    # Release pages
    state = page_manager.release_pages(
        page_state=state,
        request_id=0
    )
    ```
  """

  def __init__(self, config: Config):
    """Initializes the `PageManager` from a configuration object.

    Args:
      config: A `Config` object containing the necessary parameters, including:
        * `num_decoder_layers`: The number of decoder layers in the model.
        * `max_target_length`: The maximum sequence length.
        * `pagedattn_num_pages`: The total number of pages PER LAYER. Each layer
            has its own independent pool of pages - you do not need to multiply
            by the number of layers yourself.
        * `pagedattn_tokens_per_page`: The number of tokens per page.
        * `pagedattn_max_pages_per_group`: The maximum number of total pages per request.
        * `pagedattn_max_page_groups`: The maximum number of total requests.

    Raises:
      ValueError: If the configuration parameters are invalid.
    """
    self.num_layers = config.num_decoder_layers
    self.num_pages = config.pagedattn_num_pages
    self.tokens_per_page = config.pagedattn_tokens_per_page
    self.max_target_length = config.max_target_length
    self.max_pages_per_group = config.pagedattn_max_pages_per_group
    self.max_page_groups = config.pagedattn_max_page_groups

    self._validate_init_params()

  def _validate_init_params(self) -> None:
    """Validates initialization parameters.

    This method checks that the configuration parameters are valid and logically
    consistent. It ensures that there are enough pages, that the page size is
    reasonable, and that the maximum pages per group can accommodate the maximum
    target length.

    Raises:
      ValueError: If any of the configuration parameters are invalid or inconsistent.
    """
    if self.max_pages_per_group < self.max_target_length // self.tokens_per_page:
      raise ValueError(
          f"`max_pages_per_group` ({self.max_pages_per_group}) must be at "
          f"least {self.max_target_length // self.tokens_per_page} "
          f"to accommodate `max_target_length` ({self.max_target_length}) "
          f"with `tokens_per_page` ({self.tokens_per_page})."
      )
    if self.num_pages <= 0:
      raise ValueError(f"Invalid `num_pages`: {self.num_pages}")
    if self.tokens_per_page <= 0:
      raise ValueError(f"Invalid `tokens_per_page`: {self.tokens_per_page}")
    if self.max_page_groups <= 0 and self.max_page_groups != -1:
      raise ValueError(f"Invalid `max_page_groups`: {self.max_page_groups}")
    if self.max_pages_per_group <= 0:
      raise ValueError(f"Invalid `max_pages_per_group`: {self.max_pages_per_group}")

  def update_prefill_pages(self, page_state: PageState, request_id: int, true_length: int) -> PageState:
    """Reserves pages for a request during prefill across all layers.

    This method allocates the necessary pages to store a sequence of the specified length
    in each layer of the model for the given request. If there are not enough free pages
    available in a layer, no pages will be allocated for that layer.

    Args:
      page_state: The current `PageState`.
      request_id: The ID of the request to allocate pages for.
      true_length: The sequence length to allocate pages for.

    Returns:
      The updated `PageState` with pages allocated for the sequence.

    Example:
      ```python
      # Reserve pages for a 16-token sequence in request 0
      state = page_manager.update_prefill_pages(
          page_state=state,
          request_id=0,
          true_length=16
      )
      ```
    """
    # Use JAX primitives for validation rather than Python conditionals
    is_valid_request = jnp.logical_and(request_id >= 0, request_id < self.max_page_groups)
    is_valid_length = jnp.logical_and(true_length >= 0, true_length <= self.max_target_length)
    is_valid = jnp.logical_and(is_valid_request, is_valid_length)
    
    # Use JAX conditional primitive
    def process_valid_request(_):
      # Use JAX fori_loop to iterate through layers
      def update_layer(layer_id, current_state):
        return _update_prefill_pages_for_group(
            current_state, 
            request_id,  # Use request_id as page_group_id internally
            true_length, 
            layer_id, 
            self.tokens_per_page, 
            self.max_pages_per_group
        )
      
      return jax.lax.fori_loop(0, self.num_layers, update_layer, page_state)
    
    def return_original_state(_):
      return page_state
    
    return jax.lax.cond(is_valid, process_valid_request, return_original_state, None)

  def update_decode_pages(self, page_state: PageState) -> PageState:
    """Updates pages for autoregressive decoding across all layers.

    This method increments the sequence length for all active requests and
    allocates new pages as necessary when page boundaries are crossed.

    Args:
      page_state: The current `PageState`.

    Returns:
      The updated `PageState` after the decode step.

    Example:
      ```python
      # Update pages for the next decode step
      state = page_manager.update_decode_pages(state)
      ```
    """
    # Use JAX fori_loop to iterate through layers
    def update_layer(layer_id, current_state):
      return _update_decode_pages_for_layer(current_state, layer_id, self.tokens_per_page)
    
    return jax.lax.fori_loop(0, self.num_layers, update_layer, page_state)

  def release_pages(self, page_state: PageState, request_id: int) -> PageState:
    """Releases all pages associated with a given request across all layers.

    This method frees all pages allocated to the specified request in all layers
    and resets the corresponding state entries.

    Args:
      page_state: The current `PageState`.
      request_id: The ID of the request to release.

    Returns:
      The updated `PageState` after releasing the pages.

    Example:
      ```python
      # Release all pages for request 0
      state = page_manager.release_pages(
          page_state=state,
          request_id=0
      )
      ```
    """
    is_valid = jnp.logical_and(request_id >= 0, request_id < self.max_page_groups)
    
    def process_valid_request(_):
      def release_layer(layer_id, current_state):
        return _release_page_group_in_layer(current_state, request_id, layer_id)
      
      return jax.lax.fori_loop(0, self.num_layers, release_layer, page_state)
    
    def return_original_state(_):
      return page_state
    
    return jax.lax.cond(is_valid, process_valid_request, return_original_state, None)

  def get_initial_page_state(self) -> PageState:
    """Creates and returns an initial `PageState`.

    This is a convenience method that calls `initialize_page_state` with
    the parameters specified during initialization.

    Returns:
      An initialized `PageState` object.

    Example:
      ```python
      # Get an initial page state
      state = page_manager.get_initial_page_state()
      ```
    """
    return initialize_page_state(
        self.num_layers, 
        self.num_pages, 
        self.max_page_groups, 
        self.max_pages_per_group
    )