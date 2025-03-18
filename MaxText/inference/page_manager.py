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
      its allocated pages.
    num_pages_used: A `jnp.ndarray` of shape `[num_layers, max_page_groups]`.
      This array tracks the number of pages currently allocated to each page group.
    sequence_lengths: A `jnp.ndarray` of shape `[num_layers, max_page_groups]`.
      This array stores the current length of each sequence (in tokens).
    active_page: A `jnp.ndarray` of shape `[num_layers, max_page_groups]`.
      This array stores the index of the *currently active* page for each page group.
    has_active_page: A `jnp.ndarray` of shape `[num_layers, max_page_groups]`.
      Boolean mask indicating which active_page entries represent valid pages.
    active_page_position: A `jnp.ndarray` of shape `[num_layers,
      max_page_groups]`. This array stores the index (offset) of the next
      available token index within the `active_page` for each page group.
  """

  page_status: jnp.ndarray
  page_map: jnp.ndarray
  num_pages_used: jnp.ndarray
  sequence_lengths: jnp.ndarray
  active_page: jnp.ndarray
  has_active_page: jnp.ndarray
  active_page_position: jnp.ndarray


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
  """
  return PageState(
      page_status=jnp.zeros((num_layers, num_pages), dtype=jnp.int32),
      page_map=jnp.zeros((num_layers, max_page_groups, max_pages_per_group), dtype=jnp.int32),
      num_pages_used=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
      sequence_lengths=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
      active_page=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
      has_active_page=jnp.zeros((num_layers, max_page_groups), dtype=jnp.bool_),
      active_page_position=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
  )

def get_valid_page_assignments(state, layer_id):
    max_groups = state.page_map.shape[1]
    max_pages_per_group = state.page_map.shape[2]
    page_indices = jnp.arange(max_pages_per_group)
    
    # This creates a 2D validity mask [groups, pages]
    validity_mask = page_indices[None, :] < state.num_pages_used[layer_id, :, None]
    
    # Get all page indices from page_map
    all_pages = state.page_map[layer_id]
    
    # Create a flattened list of indices with validity mask applied
    # Replace invalid entries with -1
    masked_pages = jnp.where(validity_mask, all_pages, -1)
    
    # Flatten and filter out the -1 values 
    flat_masked = masked_pages.flatten()
    return flat_masked[flat_masked >= 0]

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
  """Reserves pages for a page group during the prefill stage for a specific layer."""

  # --- IMPORTANT: Create copies to avoid in-place modification ---
  layer_page_status = page_state.page_status[layer_id].copy()
  layer_page_map = page_state.page_map[layer_id].copy()
  num_pages_used = page_state.num_pages_used[layer_id].copy()

  def release_existing_pages(index: int, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
    current_status, current_map, pages_used = state

    # Only consider pages within the used range
    is_valid = index < pages_used[page_group_id]
    page_index = current_map[page_group_id, index]

    # Free the page if it's within the valid range
    updated_status = jax.lax.cond(
        is_valid, 
        lambda: current_status.at[page_index].set(0), 
        lambda: current_status
    )

    return (updated_status, current_map, pages_used)

  # Reset number of pages used for this group to 0
  num_pages_used = num_pages_used.at[page_group_id].set(0)

  # Release existing pages for this group
  layer_page_status, layer_page_map, _ = jax.lax.fori_loop(
      0,
      max_pages_per_group,
      release_existing_pages,
      (layer_page_status, layer_page_map, page_state.num_pages_used[layer_id]),
  )

  # Handle zero length case (return NEW PageState)
  def handle_zero_length(_):
    return PageState(
        page_status=page_state.page_status.at[layer_id].set(layer_page_status),
        page_map=page_state.page_map.at[layer_id].set(layer_page_map),
        num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(0),
        sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(0),
        active_page=page_state.active_page.at[layer_id, page_group_id].set(0),
        has_active_page=page_state.has_active_page.at[layer_id, page_group_id].set(False),
        active_page_position=page_state.active_page_position.at[layer_id, page_group_id].set(0),
    )

  def allocate_pages(state, num_pages_needed):
    current_status, current_map, pages_used = state

    def allocate_new_page(index: int, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
      current_status, current_map, pages_used = state

      # Find next free page
      next_free_page = _find_next_free_page_index(current_status)

      # Check if we need this page and if one is available
      should_allocate = jnp.logical_and(index < num_pages_needed, next_free_page >= 0)

      # Update page status and map if we should allocate
      current_status = jax.lax.cond(
          should_allocate, 
          lambda: current_status.at[next_free_page].set(1), 
          lambda: current_status
      )
      current_map = jax.lax.cond(
          should_allocate, 
          lambda: current_map.at[page_group_id, index].set(next_free_page), 
          lambda: current_map
      )

      # Increment pages_used if a page was allocated
      pages_used = jax.lax.cond(
        should_allocate, 
        lambda: pages_used.at[page_group_id].add(1), 
        lambda: pages_used
      )

      return (current_status, current_map, pages_used)

    new_page_status, new_page_map, new_pages_used = jax.lax.fori_loop(
        0, 
        max_pages_per_group, 
        allocate_new_page, (current_status, current_map, pages_used)
    )
    return new_page_status, new_page_map, new_pages_used

  # If we don't have enough pages, just return a state with cleared resources
  def return_cleared_state(_):
    return PageState(
        page_status=page_state.page_status.at[layer_id].set(layer_page_status),
        page_map=page_state.page_map.at[layer_id].set(layer_page_map),
        num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(0),
        sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(0),
        active_page=page_state.active_page.at[layer_id, page_group_id].set(0),
        has_active_page=page_state.has_active_page.at[layer_id, page_group_id].set(False),
        active_page_position=page_state.active_page_position.at[layer_id, page_group_id].set(0),
    )

  # Handle normal case
  def handle_normal_case(_):
    # Calculate how many pages we need
    num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
    last_page_position = (true_length - 1) % tokens_per_page

    # Check if we have enough free pages
    num_free_pages = jnp.sum(layer_page_status == 0)
    has_enough_pages = num_free_pages >= num_pages_needed

    def allocate_and_return_state(state):
      layer_page_status, layer_page_map, num_pages_used = state
      new_page_status, new_page_map, new_pages_used = allocate_pages(
          (layer_page_status, layer_page_map, num_pages_used), num_pages_needed
      )

      # For the current page, use the last allocated page
      last_page_idx = jnp.minimum(num_pages_needed, max_pages_per_group) - 1
      last_page_valid = num_pages_needed > 0
      last_page_index = jax.lax.cond(
          last_page_valid, 
          lambda: new_page_map[page_group_id, last_page_idx], 
          lambda: jnp.array(0, dtype=jnp.int32)
      )

      return PageState(
          page_status=page_state.page_status.at[layer_id].set(new_page_status),
          page_map=page_state.page_map.at[layer_id].set(new_page_map),
          num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(
              jnp.minimum(new_pages_used[page_group_id], max_pages_per_group)
          ),
          sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(true_length),
          active_page=page_state.active_page.at[layer_id, page_group_id].set(last_page_index),
          has_active_page=page_state.has_active_page.at[layer_id, page_group_id].set(last_page_valid),
          active_page_position=page_state.active_page_position.at[layer_id, page_group_id].set(last_page_position),
      )

    return jax.lax.cond(
        has_enough_pages,
        lambda x: allocate_and_return_state(x),
        lambda x: return_cleared_state(x),
        (layer_page_status, layer_page_map, num_pages_used),
    )

  return jax.lax.cond(true_length == 0, handle_zero_length, handle_normal_case, None)


def _update_decode_pages_for_layer(
    page_state: PageState,
    layer_id: int,
    tokens_per_page: int,
) -> PageState:
  """Updates pages for autoregressive decoding for a specific layer."""

  # --- IMPORTANT: Create copies to avoid in-place modification ---
  layer_page_status = page_state.page_status[layer_id].copy()
  layer_page_map = page_state.page_map[layer_id].copy()
  layer_sequence_lengths = page_state.sequence_lengths[layer_id].copy()
  layer_pages_used = page_state.num_pages_used[layer_id].copy()
  layer_active_page = page_state.active_page[layer_id].copy()
  layer_has_active_page = page_state.has_active_page[layer_id].copy()
  layer_active_position = page_state.active_page_position[layer_id].copy()

  # Update sequence length and position within the page.
  # Only increment sequence lengths for active pages
  new_sequence_lengths = layer_sequence_lengths + jnp.where(layer_has_active_page, 1, 0)

  # Calculate new position for each page
  new_current_position = jnp.where(
      layer_has_active_page, 
      (new_sequence_lengths - 1) % tokens_per_page, 
      layer_active_position
  )

  # Determine which slots need new pages
  new_pages_needed = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page

  # Initialize variables for updated state
  new_page_status = layer_page_status
  new_page_map = layer_page_map
  new_pages_used = layer_pages_used
  new_active_page = layer_active_page
  new_has_active_page = layer_has_active_page

  max_page_groups = layer_sequence_lengths.shape[0]

  def update_group(group_index: int, state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]):
    current_status, current_map, active_pages, has_current, pages_used = state

    # Need new page if:
    # 1. We need more pages than we're using
    # 2. We have an active current page
    needs_new_page = jnp.logical_and(
      new_pages_needed[group_index] > pages_used[group_index], 
      has_current[group_index]
    )

    # Find next free page if needed
    next_free_page = jax.lax.cond(
        needs_new_page, 
        lambda: _find_next_free_page_index(current_status), 
        lambda: jnp.array(0, dtype=jnp.int32)
    )

    # Determine if we can allocate a new page
    should_allocate = jnp.logical_and(needs_new_page, next_free_page >= 0)

    # Update page status to allocate new page
    updated_status = jax.lax.cond(
      should_allocate, 
      lambda: current_status.at[next_free_page].set(1), 
      lambda: current_status
    )

    # Add page to page map
    updated_map = jax.lax.cond(
        should_allocate,
        lambda: current_map.at[group_index, pages_used[group_index]].set(next_free_page),
        lambda: current_map,
    )

    # Update current page
    updated_pages = jax.lax.cond(
        should_allocate, 
        lambda: active_pages.at[group_index].set(next_free_page), 
        lambda: active_pages
    )

    # Update count of pages used
    updated_used = jax.lax.cond(
        should_allocate, 
        lambda: pages_used.at[group_index].set(pages_used[group_index] + 1), 
        lambda: pages_used
    )

    # Current page is always valid when we have an active sequence
    updated_has_current = has_current

    return (updated_status, updated_map, updated_pages, updated_has_current, updated_used)

  # Update each group's state
  (new_page_status, 
   new_page_map, 
   new_active_page, 
   new_has_active_page, 
   new_pages_used) = jax.lax.fori_loop(
      0,
      max_page_groups,
      update_group,
      (new_page_status, new_page_map, new_active_page, new_has_active_page, new_pages_used),
  )

  # Return a NEW PageState
  return PageState(
      page_status=page_state.page_status.at[layer_id].set(new_page_status),
      page_map=page_state.page_map.at[layer_id].set(new_page_map),
      num_pages_used=page_state.num_pages_used.at[layer_id].set(new_pages_used),
      sequence_lengths=page_state.sequence_lengths.at[layer_id].set(new_sequence_lengths),
      active_page=page_state.active_page.at[layer_id].set(new_active_page),
      has_active_page=page_state.has_active_page.at[layer_id].set(new_has_active_page),
      active_page_position=page_state.active_page_position.at[layer_id].set(new_current_position),
  )


def _release_page_group_in_layer(
    page_state: PageState,
    page_group_id: int,
    layer_id: int,
) -> PageState:
  """Releases all pages associated with a given page group in a specific layer."""
  layer_page_status = page_state.page_status[layer_id].copy()
  layer_page_map = page_state.page_map[layer_id].copy()

  # Get valid pages based on num_pages_used
  num_valid_pages = page_state.num_pages_used[layer_id, page_group_id]

  def release_page(i, page_status):
    # Only process pages that are within the valid range
    is_valid = i < num_valid_pages
    page_idx = layer_page_map[page_group_id, i]

    # Only modify page_status if the page is valid
    return jax.lax.cond(
      is_valid, 
      lambda: page_status.at[page_idx].set(0), 
      lambda: page_status
    )

  # Free all valid pages in this group
  new_layer_page_status = jax.lax.fori_loop(
    0, 
    layer_page_map.shape[1], 
    release_page, 
    layer_page_status
  )

  # Return a NEW PageState with cleared state for this page group
  return PageState(
      page_status=page_state.page_status.at[layer_id].set(new_layer_page_status),
      page_map=page_state.page_map.at[layer_id].set(layer_page_map),
      num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(0),
      sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(0),
      active_page=page_state.active_page.at[layer_id, page_group_id].set(0),
      has_active_page=page_state.has_active_page.at[layer_id, page_group_id].set(False),
      active_page_position=page_state.active_page_position.at[layer_id, page_group_id].set(0),
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
    is_valid_request = jnp.logical_and(request_id >= 0, request_id < self.max_page_groups)
    is_valid_length = jnp.logical_and(true_length >= 0, true_length <= self.max_target_length)
    is_valid = jnp.logical_and(is_valid_request, is_valid_length)

    def process_valid_request(page_state):
      def update_layer(layer_id, page_state):
        return _update_prefill_pages_for_group(
            page_state,
            request_id,
            true_length,
            layer_id,
            self.tokens_per_page,
            self.max_pages_per_group,
        )

      # Critically, the PageState is not updated in place
      new_page_state = jax.lax.fori_loop(0, self.num_layers, update_layer, page_state)
      return new_page_state  # Return the modified state

    def return_original_state(page_state):
      return page_state

    return jax.lax.cond(is_valid, process_valid_request, return_original_state, page_state)

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

    def update_layer(layer_id, page_state):
      return _update_decode_pages_for_layer(page_state, layer_id, self.tokens_per_page)

    # Critically, the PageState is not updated in place
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

    def process_valid_request(page_state):
      def release_layer(layer_id, page_state):
        return _release_page_group_in_layer(page_state, request_id, layer_id)

      # Critically, the PageState is not updated in place
      return jax.lax.fori_loop(0, self.num_layers, release_layer, page_state)

    def return_original_state(page_state):
      return page_state

    return jax.lax.cond(is_valid, process_valid_request, return_original_state, page_state)  # Pass page_state

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
    return initialize_page_state(self.num_layers, self.num_pages, self.max_page_groups, self.max_pages_per_group)
