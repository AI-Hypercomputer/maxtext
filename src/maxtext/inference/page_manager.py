# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Page Managers for implementing paged attention in MaxText.

This module provides the `PageManager` class and associated `PageState` dataclass
for managing the paged attention mechanism. The paging system allows efficient
handling of variable-length sequences by dividing the attention context into
fixed-size pages, similar to virtual memory systems. Pages are managed globally
(not per-layer) and assigned to page groups, where each group typically
represents an individual request or sequence.
"""

from functools import partial

import jax
import jax.numpy as jnp

from flax import struct

from jaxtyping import Array, Integer, Bool

from MaxText.common_types import Config

# Aliases using <Dims><Type><Rank>d convention
# We use string names for dimensions as they are symbolic within the type hints.
PagesInt1d = Integer[Array, "num_pages"]
GroupsPagesInt2d = Integer[Array, "max_page_groups max_pages_per_group"]
GroupsInt1d = Integer[Array, "max_page_groups"]
GroupsBool1d = Bool[Array, "max_page_groups"]
ScalarInt = Integer[Array, ""]
ScalarBool = Bool[Array, ""]


@struct.dataclass
class PageState:
  """Represents the global state of memory pages managed by the `PageManager`.

  This dataclass tracks the allocation status of each page across the entire system,
  the mapping of pages to page groups (requests), and the current position within
  each sequence's pages. State is managed globally, providing a single view
  across all potential layers using this manager.
  """

  page_status: PagesInt1d
  page_map: GroupsPagesInt2d
  num_pages_used: GroupsInt1d
  sequence_lengths: GroupsInt1d
  active_page: GroupsInt1d
  has_active_page: GroupsBool1d
  active_page_position: GroupsInt1d


def initialize_page_state(
    num_pages: int,
    max_page_groups: int,
    max_pages_per_group: int,
) -> PageState:
  """Creates and initializes a global `PageState` object.

  All pages in the global pool are initially marked as free (status 0), except
  for page 0 which is marked used as a workaround. No pages are assigned to any
  page group. Sequence lengths and page usage counts are initialized to zero.
  Active page tracking is also reset.

  Args:
    num_pages: The total number of available pages in the global pool.
    max_page_groups: The maximum number of page groups (concurrent sequences/requests)
      the system can track.
    max_pages_per_group: The maximum number of pages that can be allocated to
      a single page group (determines the size of the second dimension of `page_map`).

  Returns:
    An initialized `PageState` object with all values set to their defaults (zeros/False).
  """
  # TODO(patemotter): Produces garbage output for any request that uses page 0
  initial_page_status = jnp.zeros((num_pages,), dtype=jnp.int32)
  initial_page_status = initial_page_status.at[0].set(1)  # Workaround page 0
  return PageState(
      page_status=initial_page_status,
      page_map=jnp.zeros((max_page_groups, max_pages_per_group), dtype=jnp.int32),
      num_pages_used=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      sequence_lengths=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      active_page=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      has_active_page=jnp.zeros((max_page_groups,), dtype=jnp.bool_),
      active_page_position=jnp.zeros((max_page_groups,), dtype=jnp.int32),
  )


@jax.jit
def _find_next_free_page_index(page_status: PagesInt1d) -> ScalarInt:
  """Finds the index of the next available free page in the global pool.

  Searches the `page_status` array for the first occurrence of 0 (indicating
  a free page), skipping index 0 due to potential issues.

  Args:
    page_status: A 1D `jnp.ndarray` representing the global status of pages
      (0 for free, 1 for allocated). Should have shape [num_pages].

  Returns:
    A scalar `jnp.int32` array containing the index of the next free page
    (the lowest index >= 1 where `page_status` is 0).
    Returns -1 if no free pages (at index >= 1) are found.
  """
  # TODO(patemotter): Produces garbage output for any request that uses page 0
  search_status = page_status[1:]
  overall_free_mask = search_status == 0

  # argmax returns the index of the *first* True. If none are True, it returns 0.
  next_free_relative = jnp.argmax(overall_free_mask)
  # Add 1 to compensate for the slice [1:]
  next_free_overall = next_free_relative + 1
  # Check if a free page exists
  has_free_overall = jnp.any(overall_free_mask)
  # If a free page exists, return its index, otherwise return -1
  return jnp.where(has_free_overall, next_free_overall, -1)


@partial(jax.jit, static_argnames=("max_pages_per_group",))
def _release_pages_for_group(
    page_state: PageState,
    page_group_id: ScalarInt,
    max_pages_per_group: int,
) -> PageState:
  """Releases all pages associated with a given page group.

  This function iterates through the potential pages allocated to the specified
  `page_group_id` (up to `max_pages_per_group`). For each page index actually
  used by the group (determined by `num_pages_used`), it retrieves the global
  page index from `page_map` and resets its status to 0 (free) in the global
  `page_status` array. It also resets all state fields related to the
  `page_group_id` (length, count, active status, etc.) to their initial values.

  Args:
    page_state: The current global `PageState`.
    page_group_id: The index of the page group whose pages are to be released.
    max_pages_per_group: The maximum number of pages a group can hold (used as
      the loop bound).

  Returns:
    A new `PageState` object where the specified group's pages are marked as free
    in `page_status`, and the group's specific state entries are reset.
  """
  current_page_status = page_state.page_status
  current_page_map = page_state.page_map
  num_valid_pages = page_state.num_pages_used[page_group_id]

  def release_page(i: int, status: PagesInt1d) -> PagesInt1d:
    is_valid = i < num_valid_pages
    page_idx = current_page_map[page_group_id, i]
    # Only release if index 'i' points to a valid allocated page
    should_release = jnp.logical_and(is_valid, page_idx > 0)

    return jax.lax.cond(should_release, lambda s: s.at[page_idx].set(0), lambda s: s, status)

  new_page_status = jax.lax.fori_loop(0, max_pages_per_group, release_page, current_page_status)

  return page_state.replace(
      page_status=new_page_status,
      num_pages_used=page_state.num_pages_used.at[page_group_id].set(0),
      sequence_lengths=page_state.sequence_lengths.at[page_group_id].set(0),
      active_page=page_state.active_page.at[page_group_id].set(0),
      has_active_page=page_state.has_active_page.at[page_group_id].set(False),
      active_page_position=page_state.active_page_position.at[page_group_id].set(0),
  )


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _reserve_pages_for_group(
    released_state: PageState,
    page_group_id: ScalarInt,
    true_length: ScalarInt,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Reserves pages for a specific group, assuming true_length > 0.

  PRECONDITION: `true_length` must be > 0. This function assumes the caller
  (e.g., `PageManager.update_prefill_pages`) has validated this.

  Calculates the number of pages required for `true_length`. Checks if enough
  free pages exist globally and if the group has capacity based on the state
  provided in `released_state`. If resources are sufficient, it iteratively
  finds free pages, marks them allocated, records them in the map, and updates
  the group's state fields. If resources are insufficient, it returns the
  `released_state` unchanged (effectively leaving the group empty).

  Args:
      released_state: The global `PageState` after pages for `page_group_id`
          have already been released.
      page_group_id: The index of the page group to allocate pages for.
      true_length: The target sequence length for the prefill. MUST BE > 0.
      tokens_per_page: The capacity of each page.
      max_pages_per_group: The maximum number of pages the group can hold.

  Returns:
      A new `PageState` with pages allocated for the group and its state updated,
      or the input `released_state` if allocation failed due to resource limits.
  """
  num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
  last_token_abs_idx = true_length - 1
  last_page_position_idx = last_token_abs_idx % tokens_per_page
  next_write_position = (last_page_position_idx + 1) % tokens_per_page

  current_page_status = released_state.page_status
  current_page_map = released_state.page_map
  current_num_pages_used = released_state.num_pages_used

  num_free_pages = jnp.sum(current_page_status == 0)
  group_has_capacity = jax.lax.le(num_pages_needed, max_pages_per_group)
  sufficient_free_pages = jax.lax.ge(num_free_pages, num_pages_needed)
  has_enough_resources = jnp.logical_and(sufficient_free_pages, group_has_capacity)

  def allocate_and_update_state(initial_state_tuple: tuple[PagesInt1d, GroupsPagesInt2d, GroupsInt1d]) -> PageState:
    """Allocates pages iteratively if resources are sufficient."""
    initial_status, initial_map, initial_num_used = initial_state_tuple

    def allocate_one_page(
        page_idx_in_group: ScalarInt, loop_state_tuple: tuple[PagesInt1d, GroupsPagesInt2d, GroupsInt1d]
    ) -> tuple[PagesInt1d, GroupsPagesInt2d, GroupsInt1d]:
      """Allocates a single page within the fori_loop."""
      current_loop_status, current_loop_map, current_loop_num_used = loop_state_tuple
      next_free_page_global = _find_next_free_page_index(current_loop_status)
      page_allocated = jax.lax.ge(next_free_page_global, 0)

      new_loop_status = jax.lax.cond(
          page_allocated,
          lambda s: s.at[next_free_page_global].set(1),
          lambda s: s,
          current_loop_status,
      )
      new_loop_map = jax.lax.cond(
          page_allocated,
          lambda m: m.at[page_group_id, page_idx_in_group].set(next_free_page_global),
          lambda m: m,
          current_loop_map,
      )
      new_loop_num_used = jax.lax.cond(
          page_allocated,
          lambda n: n.at[page_group_id].add(1),
          lambda n: n,
          current_loop_num_used,
      )
      return new_loop_status, new_loop_map, new_loop_num_used

    final_page_status, final_page_map, final_num_pages_used = jax.lax.fori_loop(
        0,
        num_pages_needed,
        allocate_one_page,
        (initial_status, initial_map, initial_num_used),
    )
    active_page_global_index = final_page_map[page_group_id, num_pages_needed - 1]

    return released_state.replace(
        page_status=final_page_status,
        page_map=final_page_map,
        num_pages_used=final_num_pages_used,
        sequence_lengths=released_state.sequence_lengths.at[page_group_id].set(true_length),
        active_page=released_state.active_page.at[page_group_id].set(active_page_global_index),
        has_active_page=released_state.has_active_page.at[page_group_id].set(True),
        active_page_position=released_state.active_page_position.at[page_group_id].set(next_write_position),
    )

    # Conditionally perform allocation or return the released state

  final_state = jax.lax.cond(
      has_enough_resources,
      allocate_and_update_state,
      lambda _: released_state,
      operand=(current_page_status, current_page_map, current_num_pages_used),
  )
  return final_state


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _release_and_reserve_for_group(
    page_state: PageState,
    page_group_id: ScalarInt,
    true_length: ScalarInt,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Releases existing pages and reserves new pages for a group during prefill.

  Assumes true_length > 0. Caller MUST validate inputs.
  """
  released_state = _release_pages_for_group(page_state, page_group_id, max_pages_per_group)
  final_state = _reserve_pages_for_group(released_state, page_group_id, true_length, tokens_per_page, max_pages_per_group)
  return final_state


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _update_decode_pages_global(
    page_state: PageState,
    tokens_per_page: ScalarInt,
    max_pages_per_group: ScalarInt,
) -> PageState:
  """Updates pages globally for one step of autoregressive decoding.

  This function performs the following steps for all page groups simultaneously:
  1. Increments `sequence_lengths` for groups marked as `has_active_page`.
  2. Calculates the new `active_page_position` based on the incremented length.
  3. Determines which active groups now require a new page because their sequence
     length has crossed a page boundary (`required_pages > num_pages_used`) and
     they still have capacity (`required_pages <= max_pages_per_group`).
  4. For each group identified in step 3, it attempts to find a free page globally and
     allocate it, updating `page_status`, `page_map`, `num_pages_used`, and
     `active_page` for that group.

  Args:
    page_state: The current global `PageState`.
    tokens_per_page: The capacity of each page.
    max_pages_per_group: The maximum number of pages allowed per group.

  Returns:
    A new `PageState` object reflecting the state after the decode step, potentially
    with new pages allocated to groups that crossed page boundaries.
  """
  max_page_groups = page_state.sequence_lengths.shape[0]

  seq_len_increment = jnp.where(page_state.has_active_page, 1, 0)
  new_sequence_lengths = page_state.sequence_lengths + seq_len_increment

  new_active_page_position = jnp.where(
      page_state.has_active_page,
      (new_sequence_lengths - 1) % tokens_per_page,
      page_state.active_page_position,
  )

  required_pages_per_group = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page
  needs_new_page_mask = jnp.logical_and(page_state.has_active_page, required_pages_per_group > page_state.num_pages_used)
  has_capacity_mask = required_pages_per_group <= max_pages_per_group
  needs_allocation_mask = jnp.logical_and(needs_new_page_mask, has_capacity_mask)

  def allocate_for_group_if_needed(group_idx: ScalarInt, current_state: PageState) -> PageState:
    """Inner function for fori_loop to conditionally allocate a page."""
    current_status = current_state.page_status
    current_map = current_state.page_map
    current_num_used = current_state.num_pages_used
    current_active_page = current_state.active_page

    needs_alloc = needs_allocation_mask[group_idx]
    next_free_page_global = _find_next_free_page_index(current_status)
    can_allocate = jnp.logical_and(needs_alloc, next_free_page_global >= 0)

    new_status = jax.lax.cond(can_allocate, lambda s: s.at[next_free_page_global].set(1), lambda s: s, current_status)

    page_map_index = current_num_used[group_idx]
    new_map = jax.lax.cond(
        can_allocate, lambda m: m.at[group_idx, page_map_index].set(next_free_page_global), lambda m: m, current_map
    )
    new_num_used = jax.lax.cond(can_allocate, lambda n: n.at[group_idx].add(1), lambda n: n, current_num_used)
    new_active_page = jax.lax.cond(
        can_allocate, lambda a: a.at[group_idx].set(next_free_page_global), lambda a: a, current_active_page
    )

    # Reconstruct state for loop carry/return
    return current_state.replace(
        page_status=new_status,
        page_map=new_map,
        num_pages_used=new_num_used,
        active_page=new_active_page,
    )

    # Initialize loop state with pre-calculated lengths and positions

  initial_loop_state = page_state.replace(
      sequence_lengths=new_sequence_lengths,
      active_page_position=new_active_page_position,
  )

  # Apply conditional allocation across all groups
  final_state = jax.lax.fori_loop(0, max_page_groups, allocate_for_group_if_needed, initial_loop_state)
  return final_state


class PageManager:
  """Manages the global allocation and release of pages for paged attention.

  This class provides an interface for reserving pages during prefill and
  decoding, and for releasing pages when a sequence (page group) is complete.
  It encapsulates the logic for tracking page allocation globally and managing
  the `PageState`. It uses the concept of page groups, where each group typically
  corresponds to a single request or sequence being processed.

  Example:
    ```python
    # Initialize a PageManager from configuration
    config = YourConfig(...) # Set pagedattn_num_pages, etc.
    page_manager = PageManager(config)

    # Get initial page state (all pages free, except potentially page 0)
    state = page_manager.get_initial_page_state()

    # Update pages for prefill of a sequence in group 0 with length 16
    state = page_manager.update_prefill_pages(
        page_state=state,
        page_group_id=0,
        true_length=16
    )

    # Update pages for a single decode step (increments lengths, allocates if needed)
    state = page_manager.update_decode_pages(state)

    # Release pages associated with group 0 when the sequence is finished
    state = page_manager.release_pages(
        page_state=state,
        page_group_id=0
    )
    ```
  """

  def __init__(self, config: Config):
    """Initializes the `PageManager` from a configuration object.

    Args:
      config: A `Config` object containing the necessary parameters:
        * `max_target_length`: The maximum sequence length supported.
        * `pagedattn_num_pages`: The total number of pages available globally.
        * `pagedattn_tokens_per_page`: The number of tokens each page can hold.
        * `global_batch_size_to_load`: Used to determine the maximum number of concurrent
          page groups (`max_page_groups`) the system can manage.
        * `pagedattn_max_pages_per_group`: The maximum number of pages that can be
          allocated to a single page group.

    Raises:
      ValueError: If the configuration parameters are invalid (e.g., non-positive
        values, insufficient pages per group for max length).
    """
    self.num_pages: int = config.pagedattn_num_pages
    self.tokens_per_page: int = config.pagedattn_tokens_per_page
    self.max_target_length: int = config.max_target_length
    self.max_page_groups: int = config.global_batch_size_to_load
    self.max_pages_per_group: int = config.pagedattn_max_pages_per_group
    self._validate_init_params()

  def _validate_init_params(self) -> None:
    """Validates initialization parameters for logical consistency."""
    if self.max_pages_per_group <= 0:
      raise ValueError("`pagedattn_max_pages_per_group` must be positive.")
    min_required = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    if self.max_pages_per_group < min_required:
      raise ValueError(
          f"`pagedattn_max_pages_per_group` ({self.max_pages_per_group}) is insufficient for `max_target_length` "
          f"({self.max_target_length}). Needs {min_required}."
      )
      # Check > 1 due to potential page 0 workaround
    if self.num_pages <= 1:
      raise ValueError("`pagedattn_num_pages` must be greater than 1.")
    if self.tokens_per_page <= 0:
      raise ValueError("`pagedattn_tokens_per_page` must be positive.")
    if self.max_page_groups <= 0:
      raise ValueError("`pagedattn_max_page_groups` must be positive.")

  def update_prefill_pages(self, page_state: PageState, page_group_id: int, true_length: int) -> PageState:
    """Reserves pages for a specific page group during prefill (global state).

    This method first releases any pages currently allocated to the given
    `page_group_id`. It then attempts to allocate the necessary number of pages
    from the global pool to accommodate a sequence of `true_length`. If successful,
    it updates the `PageState` to reflect the new allocation and marks the group
    as active. If there are not enough free pages globally or the group exceeds
    its `max_pages_per_group` limit, the group's state remains cleared (as after
    the initial release). Input validation ensures `page_group_id` and `true_length`
    are within valid ranges.

    Args:
      page_state: The current global `PageState`.
      page_group_id: The ID of the page group (request) to allocate pages for. Must
        be between 0 and `max_page_groups - 1`.
      true_length: The sequence length to allocate pages for. Must be between 0
        and `max_target_length`.

    Returns:
      The updated `PageState`. If allocation fails due to resource limits, the
      returned state will have the specified `page_group_id` cleared.

    Raises:
      ValueError: If `page_group_id` or `true_length` are outside their valid ranges.

    Example:
      ```python
      # Reserve pages for a 16-token sequence in group 0
      state = page_manager.update_prefill_pages(
          page_state=state,
          page_group_id=0,
          true_length=16
      )
      ```
    """
    if page_group_id < 0 or page_group_id >= self.max_page_groups:
      raise ValueError(f"PageManager: page_group_id ({page_group_id}) out of range [0, {self.max_page_groups})")
    if true_length <= 0 or true_length > self.max_target_length:
      raise ValueError(f"PageManager: true_length ({true_length}) out of range (0, {self.max_target_length}]")

    return _release_and_reserve_for_group(
        page_state, page_group_id, true_length, self.tokens_per_page, self.max_pages_per_group
    )

  def update_decode_pages(self, page_state: PageState) -> PageState:
    """Updates pages globally for one step of autoregressive decoding.

    This method advances the state for all active page groups. It increments
    their sequence lengths by one and updates their position within the current
    active page. If this increment causes a sequence to cross a page boundary
    (i.e., it needs more pages than currently allocated), this method attempts
    to allocate a new page from the global pool, provided the group has not
    reached its `max_pages_per_group` limit and free pages are available.

    Args:
      page_state: The current global `PageState`.

    Returns:
      The updated `PageState` reflecting the state after the decode step.
      Sequence lengths and active positions are updated for all active groups.
      Groups that required and successfully obtained a new page will have their
      `num_pages_used`, `page_map`, and `active_page` updated.

    Example:
      ```python
      # Advance state for all active sequences by one decode step
      state = page_manager.update_decode_pages(state)
      ```
    """
    return _update_decode_pages_global(page_state, self.tokens_per_page, self.max_pages_per_group)

  def release_pages(self, page_state: PageState, page_group_id: int) -> PageState:
    """Releases all pages associated with a given page group (global state).

    This method identifies all pages currently allocated to the specified
    `page_group_id` using the `page_map` and `num_pages_used`. It marks these
    pages as free (status 0) in the global `page_status` array. It also resets
    all state information specific to the `page_group_id` (sequence length,
    page count, active status, etc.) to their initial zero/False values.
    Input validation ensures the `page_group_id` is within the valid range.

    Args:
      page_state: The current global `PageState`.
      page_group_id: The ID of the page group (request) to release. Must be
        between 0 and `max_page_groups - 1`.

    Returns:
      The updated `PageState` after releasing the pages and resetting the group's
      state.

    Raises:
      ValueError: If `page_group_id` is outside its valid range.

    Example:
      ```python
      # Release all pages currently held by group 0
      state = page_manager.release_pages(
          page_state=state,
          page_group_id=0
      )
      ```
    """
    if page_group_id < 0 or page_group_id >= self.max_page_groups:
      raise ValueError(f"PageManager: page_group_id ({page_group_id}) out of range [0, {self.max_page_groups})")
    return _release_pages_for_group(page_state, page_group_id, self.max_pages_per_group)

  def get_initial_page_state(self) -> PageState:
    """Creates and returns an initial global `PageState`.

    This is a convenience method that calls `initialize_page_state` with
    the parameters (`num_pages`, `max_page_groups`, `max_pages_per_group`)
    stored during the `PageManager` initialization.

    Returns:
      An initialized `PageState` object where all pages are free (except possibly 0)
      and no groups are active.

    Example:
      ```python
      # Get a fresh, empty page state
      initial_state = page_manager.get_initial_page_state()
      ```
    """
    return initialize_page_state(
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )
