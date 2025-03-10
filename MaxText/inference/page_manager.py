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

WARNING: THIS FILE IS A WORK IN PROGRESS.

This module provides the PageManager class and associated PageState dataclass for
managing the paged attention mechanism. The paging system allows efficient handling
of variable-length sequences by dividing the attention context into fixed-size pages,
similar to virtual memory systems.
"""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
from functools import partial
import common_types


Config = common_types.Config


@struct.dataclass
class PageState:
  """Represents the state of memory pages managed by the PageManager.

  Attributes:
    page_status: jnp.ndarray [num_layers, num_pages] | 0: free, 1: allocated.
      Array indicating whether each page is in use (1) or free (0).
    page_map: jnp.ndarray [num_layers, max_page_groups, max_pages_per_group].
      Array mapping a page group id to their assigned pages.  The value -1 is used
      to indicate an unassigned page.
    sequence_lengths: jnp.ndarray [num_layers, max_page_groups].
      Array containing the current length of each sequence.
    num_pages_used: jnp.ndarray [num_layers, max_page_groups].
      Array tracking how many pages each group is using.
    current_page: jnp.ndarray [num_layers, max_page_groups].
      Array indicating the current active page for each group. A value of -1
      indicates that no page is currently active (e.g., before prefill).
    current_page_position: jnp.ndarray [num_layers, max_page_groups].
      Array tracking the position within the current active page for each group.
  """

  page_status: jnp.ndarray  # [num_layers, num_pages] | 0: free, 1: allocated
  page_map: jnp.ndarray  # [num_layers, max_page_groups, max_pages_per_group]
  sequence_lengths: jnp.ndarray  # [num_layers, max_page_groups]
  num_pages_used: jnp.ndarray  # [num_layers, max_page_groups]
  current_page: jnp.ndarray  # [num_layers, max_page_groups]
  current_page_position: jnp.ndarray  # [num_layers, max_page_groups]


def validate_page_group_id(page_group_id: int, max_page_groups: int) -> bool:
  """Checks if a page group ID is in the valid range [0, max_page_groups)."""
  return jnp.logical_and(page_group_id >= 0, page_group_id < max_page_groups)


def validate_sequence_length(length: int, max_target_length: int) -> bool:
  """Checks if a sequence length is in the valid range [0, max_target_length]."""
  return jnp.logical_and(length >= 0, length <= max_target_length)


def initialize_page_state(
    num_layers: int,
    num_pages: int,
    max_page_groups: int,
    max_pages_per_group: int,
) -> PageState:
  """Creates and initializes a PageState object."""
  return PageState(
      page_status=jnp.zeros((num_layers, num_pages), dtype=jnp.int32),
      page_map=jnp.full((num_layers, max_page_groups, max_pages_per_group), -1, dtype=jnp.int32),
      sequence_lengths=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
      num_pages_used=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
      current_page=jnp.full((num_layers, max_page_groups), -1, dtype=jnp.int32),
      current_page_position=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
  )


def _find_next_free_page_index(page_status: jnp.ndarray) -> int:
  """Finds the index of the next available free page in a single layer."""
  free_mask = page_status == 0
  next_free = jnp.argmax(free_mask)  # Returns 0 if all False
  has_free = jnp.any(free_mask)
  return jnp.where(has_free, next_free, -1)


def reserve_prefill_pages_for_group(
    page_state: PageState,
    page_group_id: int,
    true_length: int,
    layer_id: int,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Reserves pages for a page group during prefill."""

  # Special case for zero length
  def handle_zero_length(_):
    layer_page_status = page_state.page_status[layer_id]
    layer_page_map = page_state.page_map[layer_id]

    # Release any existing pages for this group
    def release_existing_pages(index: int, state: Tuple[jnp.ndarray, jnp.ndarray]):
      current_status, current_map = state
      page_index = current_map[page_group_id, index]
      updated_status = jnp.where(page_index >= 0, current_status.at[page_index].set(0), current_status)
      updated_map = jnp.where(page_index >= 0, current_map.at[page_group_id, index].set(-1), current_map)
      return (updated_status, updated_map)

    layer_page_status, layer_page_map = jax.lax.fori_loop(
        0, max_pages_per_group, release_existing_pages, (layer_page_status, layer_page_map)
    )

    # For zero-length, set all fields appropriately
    return page_state.replace(
        page_status=page_state.page_status.at[layer_id].set(layer_page_status),
        page_map=page_state.page_map.at[layer_id].set(layer_page_map),
        sequence_lengths=page_state.sequence_lengths.at[layer_id, page_group_id].set(0),
        num_pages_used=page_state.num_pages_used.at[layer_id, page_group_id].set(0),
        current_page=page_state.current_page.at[layer_id, page_group_id].set(-1),
        current_page_position=page_state.current_page_position.at[layer_id, page_group_id].set(0),
    )

  # Normal case for non-zero length
  def handle_normal_case(_):
    layer_page_status = page_state.page_status[layer_id]
    layer_page_map = page_state.page_map[layer_id]
    layer_sequence_lengths = page_state.sequence_lengths[layer_id]
    layer_pages_used = page_state.num_pages_used[layer_id]
    layer_current_page = page_state.current_page[layer_id]
    layer_current_position = page_state.current_page_position[layer_id]

    num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
    last_page_position = (true_length - 1) % tokens_per_page

    num_free_pages = jnp.sum(layer_page_status == 0)
    has_enough_pages = num_free_pages >= num_pages_needed

    def release_existing_pages(index: int, state: Tuple[jnp.ndarray, jnp.ndarray]):
      current_status, current_map = state
      page_index = current_map[page_group_id, index]
      updated_status = jnp.where(page_index >= 0, current_status.at[page_index].set(0), current_status)
      updated_map = jnp.where(page_index >= 0, current_map.at[page_group_id, index].set(-1), current_map)
      return (updated_status, updated_map)

    layer_page_status, layer_page_map = jax.lax.fori_loop(
        0, max_pages_per_group, release_existing_pages, (layer_page_status, layer_page_map)
    )

    def do_allocation(_):
      def allocate_new_page(index: int, state: Tuple[jnp.ndarray, jnp.ndarray]):
        current_status, current_map = state
        next_free_page = _find_next_free_page_index(current_status)
        should_allocate = jnp.logical_and(index < num_pages_needed, next_free_page >= 0)
        updated_status = jnp.where(should_allocate, current_status.at[next_free_page].set(1), current_status)
        updated_map = jnp.where(should_allocate, current_map.at[page_group_id, index].set(next_free_page), current_map)
        return (updated_status, updated_map)

      new_page_status, new_page_map = jax.lax.fori_loop(
          0, max_pages_per_group, allocate_new_page, (layer_page_status, layer_page_map)
      )
      new_sequence_lengths = layer_sequence_lengths.at[page_group_id].set(true_length)
      new_pages_used = layer_pages_used.at[page_group_id].set(jnp.minimum(num_pages_needed, max_pages_per_group))
      last_page_index = new_page_map[page_group_id, jnp.minimum(num_pages_needed, max_pages_per_group) - 1]
      new_current_page = layer_current_page.at[page_group_id].set(last_page_index)
      new_current_position = layer_current_position.at[page_group_id].set(last_page_position)

      return PageState(
          page_status=new_page_status,
          page_map=new_page_map,
          sequence_lengths=new_sequence_lengths,
          num_pages_used=new_pages_used,
          current_page=new_current_page,
          current_page_position=new_current_position,
      )

    def keep_current_state(_):
      return PageState(
          page_status=layer_page_status,
          page_map=layer_page_map,
          sequence_lengths=layer_sequence_lengths,
          num_pages_used=layer_pages_used,
          current_page=layer_current_page,
          current_page_position=layer_current_position,
      )

    new_layer_state = jax.lax.cond(has_enough_pages, do_allocation, keep_current_state, None)

    return page_state.replace(
        page_status=page_state.page_status.at[layer_id].set(new_layer_state.page_status),
        page_map=page_state.page_map.at[layer_id].set(new_layer_state.page_map),
        sequence_lengths=page_state.sequence_lengths.at[layer_id].set(new_layer_state.sequence_lengths),
        num_pages_used=page_state.num_pages_used.at[layer_id].set(new_layer_state.num_pages_used),
        current_page=page_state.current_page.at[layer_id].set(new_layer_state.current_page),
        current_page_position=page_state.current_page_position.at[layer_id].set(new_layer_state.current_page_position),
    )

  # Choose the right branch based on true_length
  return jax.lax.cond(true_length == 0, handle_zero_length, handle_normal_case, None)


def reserve_decode_pages_for_group(
    page_state: PageState,
    layer_id: int,
    tokens_per_page: int,
    max_page_groups: int,
) -> PageState:
  """Reserves pages for autoregressive decoding."""
  layer_page_status = page_state.page_status[layer_id]
  layer_page_map = page_state.page_map[layer_id]
  layer_sequence_lengths = page_state.sequence_lengths[layer_id]
  layer_pages_used = page_state.num_pages_used[layer_id]
  layer_current_page = page_state.current_page[layer_id]
  layer_current_position = page_state.current_page_position[layer_id]

  new_sequence_lengths = layer_sequence_lengths + jnp.where(layer_current_page >= 0, 1, 0)
  new_current_position = (new_sequence_lengths - 1) % tokens_per_page
  new_pages_needed = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page

  def update_page_group(group_index: int, state: PageState):
    """Updates page allocation for a single group."""
    current_status = state.page_status
    current_map = state.page_map
    current_pages = state.current_page
    pages_used = state.num_pages_used

    needs_new_page = jnp.logical_and(
        new_pages_needed[group_index] > pages_used[group_index], current_pages[group_index] >= 0
    )
    next_free_page = jnp.where(needs_new_page, _find_next_free_page_index(current_status), -1)

    def allocate_page(next_free_page: int, state: PageState, group_index: int):
      """Allocates a new page if needed and available."""
      current_status = state.page_status
      current_map = state.page_map
      current_pages = state.current_page
      pages_used = state.num_pages_used

      updated_status = current_status.at[next_free_page].set(1)
      updated_map = current_map.at[group_index, pages_used[group_index]].set(next_free_page)
      updated_pages = current_pages.at[group_index].set(next_free_page)
      updated_used = pages_used.at[group_index].set(pages_used[group_index] + 1)
      return PageState(
          page_status=updated_status,
          page_map=updated_map,
          sequence_lengths=state.sequence_lengths,
          num_pages_used=updated_used,
          current_page=updated_pages,
          current_page_position=state.current_page_position,
      )

    def no_allocation(_next_free_page: int, state: PageState, _group_index: int):
      """Keeps the current state if no allocation is needed/possible."""
      return state

    updated_state = jax.lax.cond(
        next_free_page >= 0,
        partial(allocate_page, next_free_page, state, group_index),
        partial(no_allocation, next_free_page, state, group_index),
    )

    return updated_state

  layer_state = PageState(
      page_status=layer_page_status,
      page_map=layer_page_map,
      sequence_lengths=layer_sequence_lengths,
      num_pages_used=layer_pages_used,
      current_page=layer_current_page,
      current_page_position=layer_current_position,
  )

  updated_layer_state = jax.lax.fori_loop(0, max_page_groups, update_page_group, layer_state)

  return page_state.replace(
      page_status=page_state.page_status.at[layer_id].set(updated_layer_state.page_status),
      page_map=page_state.page_map.at[layer_id].set(updated_layer_state.page_map),
      sequence_lengths=page_state.sequence_lengths.at[layer_id].set(new_sequence_lengths),
      num_pages_used=page_state.num_pages_used.at[layer_id].set(updated_layer_state.num_pages_used),
      current_page=page_state.current_page.at[layer_id].set(updated_layer_state.current_page),
      current_page_position=page_state.current_page_position.at[layer_id].set(new_current_position),
  )


def release_page_group(
    page_state: PageState,
    page_group_id: int,
    max_page_groups: int,
) -> PageState:
  """Releases all pages associated with a given page group."""
  is_valid = validate_page_group_id(page_group_id, max_page_groups)

  def do_release(page_state: PageState):
    """Releases pages if the page group ID is valid."""
    new_page_status = page_state.page_status
    new_page_map = page_state.page_map
    new_sequence_lengths = page_state.sequence_lengths
    new_num_pages_used = page_state.num_pages_used
    new_current_page = page_state.current_page
    new_current_page_position = page_state.current_page_position

    for layer_id in range(page_state.page_status.shape[0]):
      layer_pages_used = page_state.num_pages_used[layer_id, page_group_id]

      def release_page(index: int, state: Tuple[jnp.ndarray, jnp.ndarray]):
        """Releases a single page within a layer."""
        current_status, current_map = state
        page_index = current_map[page_group_id, index]
        updated_status = jnp.where(page_index >= 0, current_status.at[page_index].set(0), current_status)
        return (updated_status, current_map)

      new_layer_status, new_layer_map = jax.lax.fori_loop(
          0, layer_pages_used, release_page, (new_page_status[layer_id], new_page_map[layer_id])
      )

      new_layer_map = new_layer_map.at[page_group_id].set(jnp.full(new_layer_map.shape[1], -1, dtype=jnp.int32))

      new_page_status = new_page_status.at[layer_id].set(new_layer_status)
      new_page_map = new_page_map.at[layer_id].set(new_layer_map)
      new_sequence_lengths = new_sequence_lengths.at[layer_id, page_group_id].set(0)
      new_num_pages_used = new_num_pages_used.at[layer_id, page_group_id].set(0)
      new_current_page = new_current_page.at[layer_id, page_group_id].set(-1)
      new_current_page_position = new_current_page_position.at[layer_id, page_group_id].set(0)

    return page_state.replace(
        page_status=new_page_status,
        page_map=new_page_map,
        sequence_lengths=new_sequence_lengths,
        num_pages_used=new_num_pages_used,
        current_page=new_current_page,
        current_page_position=new_current_page_position,
    )

  def keep_current_state(page_state: PageState):
    """Keeps the current state if the page group ID is invalid."""
    return page_state

  return jax.lax.cond(is_valid, do_release, keep_current_state, page_state)


class PageManager:
  """Manages the allocation and release of pages for paged attention."""

  def __init__(self, config: Config):
    """Initializes the PageManager from a configuration object.

    Args:
      config: A Config object containing the necessary parameters.
    """
    self.num_layers = config.num_decoder_layers
    self.num_pages = config.pagedattn_num_pages
    self.tokens_per_page = config.pagedattn_tokens_per_page
    self.max_target_length = config.max_target_length
    self.max_pages_per_group = config.pagedattn_max_pages_per_group
    self.max_page_groups = config.pagedattn_max_page_groups

    self._validate_init_params(config)

  def _validate_init_params(self, config: Config):
    """Validates initialization parameters."""

    if self.max_pages_per_group < self.max_target_length // self.tokens_per_page:
      raise ValueError(
          f"max_pages_per_group ({self.max_pages_per_group}) must be at least "
          f"{self.max_target_length // self.tokens_per_page} "
          f"to accommodate max_target_length ({self.max_target_length}) "
          f"with tokens_per_page ({self.tokens_per_page})."
      )
    if self.num_pages <= 0:
      raise ValueError(f"Invalid num_pages: {self.num_pages}")
    if self.tokens_per_page <= 0:
      raise ValueError(f"Invalid tokens_per_page: {self.tokens_per_page}")
    if self.max_page_groups <= 0 and self.max_page_groups != -1:
      raise ValueError(f"Invalid max_page_groups: {self.max_page_groups}")
    if self.max_pages_per_group <= 0:
      raise ValueError(f"Invalid max_pages_per_group: {self.max_pages_per_group}")

  def reserve_prefill_pages(self, page_state: PageState, page_group_id: int, true_length: int, layer_id: int) -> PageState:
    """Reserves pages for a page group during prefill."""
    return reserve_prefill_pages_for_group(
        page_state, page_group_id, true_length, layer_id, self.tokens_per_page, self.max_pages_per_group
    )

  def reserve_decode_pages(self, page_state: PageState, layer_id: int) -> PageState:
    """Reserves pages for a page group during decoding."""
    return reserve_decode_pages_for_group(page_state, layer_id, self.tokens_per_page, self.max_page_groups)

  def release_pages(self, page_state: PageState, page_group_id: int) -> PageState:
    """Releases all pages associated with a given page group."""
    return release_page_group(page_state, page_group_id, self.max_page_groups)

  def get_initial_page_state(self) -> PageState:
    """Creates and returns an initial PageState."""
    return initialize_page_state(self.num_layers, self.num_pages, self.max_page_groups, self.max_pages_per_group)
