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
fixed-size pages, similar to virtual memory systems. Pages are managed globally
(not per-layer) and assigned to page groups, where each group typically
represents an individual request or sequence.
"""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
import common_types

Config = common_types.Config
Array = common_types.Array


@struct.dataclass
class PageState:
  """Represents the global state of memory pages managed by the `PageManager`.

  This dataclass tracks the allocation status of each page across the entire system,
  the mapping of pages to page groups (requests), and the current position within
  each sequence's pages. State is managed globally, providing a single view
  across all potential layers using this manager.

  Attributes:
    page_status: A `jnp.ndarray` of shape `[num_pages]`. Each element
      indicates whether the corresponding page in the global pool is free (0)
      or allocated (1).
    page_map: A `jnp.ndarray` of shape `[max_page_groups, max_pages_per_group]`.
      This array maps each page group to the indices (within the global pool)
      of its allocated pages. Entries beyond `num_pages_used` for a group are invalid.
    num_pages_used: A `jnp.ndarray` of shape `[max_page_groups]`. This array
      tracks the number of pages currently allocated to each page group. This
      determines the valid entries in `page_map` for each group.
    sequence_lengths: A `jnp.ndarray` of shape `[max_page_groups]`. This array
      stores the current true length of each sequence (in tokens) associated
      with a page group.
    active_page: A `jnp.ndarray` of shape `[max_page_groups]`. This array
      stores the global index of the *currently active* page (the page where the
      next token will be written) for each page group. Only valid if the
      corresponding `has_active_page` is True.
    has_active_page: A `jnp.ndarray` of shape `[max_page_groups]`. Boolean mask
      indicating whether a page group currently represents an active sequence
      and thus whether its `active_page` and `active_page_position` entries
      are meaningful.
    active_page_position: A `jnp.ndarray` of shape `[max_page_groups]`. This array
      stores the index (offset, 0 to tokens_per_page-1) of the next available
      token *within the `active_page`* for each page group. Only valid if
      `has_active_page` is True.
  """

  page_status: Array
  page_map: Array
  num_pages_used: Array
  sequence_lengths: Array
  active_page: Array
  has_active_page: Array
  active_page_position: Array


def initialize_page_state(
    num_pages: int,
    max_page_groups: int,
    max_pages_per_group: int,
) -> PageState:
  """Creates and initializes a global `PageState` object.

  All pages in the global pool are initially marked as free (status 0), and
  no pages are assigned to any page group. Sequence lengths and page usage
  counts are initialized to zero. Active page tracking is also reset.

  Args:
    num_pages: The total number of available pages in the global pool.
    max_page_groups: The maximum number of page groups (concurrent sequences/requests)
      the system can track.
    max_pages_per_group: The maximum number of pages that can be allocated to
      a single page group (determines the size of the second dimension of `page_map`).

  Returns:
    An initialized `PageState` object with all values set to their defaults (zeros/False).
  """
  # Workaround for corrupted state in page 0, could be in kernel.
  initial_page_status = jnp.zeros((num_pages,), dtype=jnp.int32)
  initial_page_status = initial_page_status.at[0].set(1)
  return PageState(
      page_status=initial_page_status,
      page_map=jnp.zeros((max_page_groups, max_pages_per_group), dtype=jnp.int32),
      num_pages_used=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      sequence_lengths=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      active_page=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      has_active_page=jnp.zeros((max_page_groups,), dtype=jnp.bool_),
      active_page_position=jnp.zeros((max_page_groups,), dtype=jnp.int32),
  )


def _find_next_free_page_index(page_status: Array) -> Array:
  """Finds the index of the next available free page in the global pool.

  Searches the `page_status` array for the first occurrence of 0 (indicating
  a free page).

  Args:
    page_status: A 1D `jnp.ndarray` representing the global status of pages
      (0 for free, 1 for allocated).

  Returns:
    The index of the next free page (the lowest index where `page_status` is 0).
    Returns -1 if no free pages are found (i.e., `page_status` contains only 1s).
  """
  # Search for free pages starting from index 1, 0 is problematic?
  search_status = page_status[1:]
  overall_free_mask = search_status == 0
  # argmax on the sliced array gives index relative to the slice start (index 1)
  next_free_relative = jnp.argmax(overall_free_mask)
  next_free_overall = next_free_relative + 1

  # Check if any free page was found in the sliced array
  has_free_overall = jnp.any(overall_free_mask)

  # Return the found index (>= 1) if a free page exists, otherwise return -1.
  return jnp.where(has_free_overall, next_free_overall, -1)


def _release_pages_for_group(
    page_state: PageState,
    page_group_id: int,
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

  # Get the number of pages actually used by this group before resetting
  num_valid_pages = page_state.num_pages_used[page_group_id]

  def release_page(i, status):
    # 'i' is the index within the group's allocation (0 to max_pages_per_group-1)
    # Check if this index 'i' corresponds to a page actually used by the group
    is_valid = i < num_valid_pages
    # Get the global index of the page from the map
    page_idx = current_page_map[page_group_id, i]

    # Only modify page_status if the page was validly assigned to this group
    should_release = jnp.logical_and(is_valid, page_idx >= 0)  # Safety check for valid index

    # Set global page status to 0 (free) if it should be released
    return jax.lax.cond(
        should_release, lambda: status.at[page_idx].set(0), lambda: status  # Otherwise, return status unchanged
    )

  # Iterate up to the maximum possible pages per group, freeing the valid ones
  new_page_status = jax.lax.fori_loop(0, max_pages_per_group, release_page, current_page_status)

  # Return NEW PageState with updated global status and cleared state for this group.
  # Resetting num_pages_used implicitly invalidates the page_map entries for this group.
  return PageState(
      page_status=new_page_status,
      page_map=page_state.page_map,  # Map itself isn't cleared, but invalidated by num_pages_used
      num_pages_used=page_state.num_pages_used.at[page_group_id].set(0),
      sequence_lengths=page_state.sequence_lengths.at[page_group_id].set(0),
      active_page=page_state.active_page.at[page_group_id].set(0),
      has_active_page=page_state.has_active_page.at[page_group_id].set(False),
      active_page_position=page_state.active_page_position.at[page_group_id].set(0),
  )


def _update_prefill_pages_for_group(
    page_state: PageState,
    page_group_id: int,
    true_length: int,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Reserves pages for a page group during prefill, after releasing existing ones.

  This function first releases any pages currently held by the `page_group_id`.
  Then, it calculates the number of pages required for the `true_length`.
  It checks if enough free pages exist globally and if the group has capacity.
  If resources are sufficient, it iteratively finds free pages using
  `_find_next_free_page_index`, marks them as allocated in `page_status`, and
  records their indices in the `page_map` for the group. Finally, it updates
  the group's state (length, count, active page/position). If resources are
  insufficient, it returns the state immediately after release (group cleared).

  Args:
    page_state: The current global `PageState`.
    page_group_id: The index of the page group to allocate pages for.
    true_length: The target sequence length for the prefill.
    tokens_per_page: The capacity of each page.
    max_pages_per_group: The maximum number of pages the group can hold.

  Returns:
    A new `PageState` object with pages allocated for the group and its state updated,
    or the cleared state if allocation failed due to insufficient resources.
  """

  # 1. Release existing pages for the group (ensures clean state)
  released_state = _release_pages_for_group(page_state, page_group_id, max_pages_per_group)

  # --- Use state *after* release for the allocation logic ---
  current_page_status = released_state.page_status
  current_page_map = released_state.page_map
  # num_pages_used for this group is now 0 after release
  current_num_pages_used = released_state.num_pages_used

  # 2. Handle zero length: If true_length is 0, the release was sufficient.
  def handle_zero_length(_):
    # The group's state is already correctly reset by released_state.
    return released_state

  # 3. Handle normal allocation (true_length > 0)
  def handle_normal_case(_):
    # Calculate page requirements for the given length
    num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
    # Position within the *last* page (0-indexed)
    last_page_position = (true_length - 1) % tokens_per_page

    # Check resource availability: enough free pages globally AND group has capacity
    num_free_pages = jnp.sum(current_page_status == 0)
    group_has_capacity = num_pages_needed <= max_pages_per_group
    has_enough_resources = jnp.logical_and(num_free_pages >= num_pages_needed, group_has_capacity)

    # === Branch: Allocate if resources are sufficient ===
    def allocate_and_update_state(initial_state_tuple: Tuple[Array, Array, Array]):
      loop_page_status, loop_page_map, loop_num_pages_used = initial_state_tuple

      # Inner function to allocate a single page within the loop
      def allocate_one_page(page_idx_in_group, loop_state_tuple):
        # page_idx_in_group: 0, 1, ..., num_pages_needed-1
        current_loop_status, current_loop_map, current_loop_num_used = loop_state_tuple

        # Find the next globally available free page index
        next_free_page_global = _find_next_free_page_index(current_loop_status)

        # Check if a page was actually found (should be true if has_enough_resources)
        page_allocated = next_free_page_global >= 0

        # Update global status: Mark found page as allocated (1)
        new_loop_status = jax.lax.cond(
            page_allocated, lambda: current_loop_status.at[next_free_page_global].set(1), lambda: current_loop_status
        )
        # Update group's map: Record the global index at the correct position
        new_loop_map = jax.lax.cond(
            page_allocated,
            lambda: current_loop_map.at[page_group_id, page_idx_in_group].set(next_free_page_global),
            lambda: current_loop_map,
        )
        # Update group's count: Increment num_pages_used for this group
        new_loop_num_used = jax.lax.cond(
            page_allocated, lambda: current_loop_num_used.at[page_group_id].add(1), lambda: current_loop_num_used
        )
        # Return updated state tuple for the next loop iteration
        return new_loop_status, new_loop_map, new_loop_num_used

      # Loop `num_pages_needed` times, allocating one page per iteration
      final_page_status, final_page_map, final_num_pages_used = jax.lax.fori_loop(
          0,
          num_pages_needed,
          allocate_one_page,
          (loop_page_status, loop_page_map, loop_num_pages_used),  # Initial state for loop
      )

      # Determine the active page: the global index of the *last* page allocated
      # Its index within the group's map is num_pages_needed - 1
      active_page_global_index = final_page_map[page_group_id, num_pages_needed - 1]

      # Construct the final PageState object with all updates
      return PageState(
          page_status=final_page_status,
          page_map=final_page_map,
          num_pages_used=final_num_pages_used,
          # Update state specific to this group using released_state as base
          sequence_lengths=released_state.sequence_lengths.at[page_group_id].set(true_length),
          active_page=released_state.active_page.at[page_group_id].set(active_page_global_index),
          has_active_page=released_state.has_active_page.at[page_group_id].set(True),  # Mark as active
          active_page_position=released_state.active_page_position.at[page_group_id].set(last_page_position),
      )

    # === Branch: Return cleared state if resources are insufficient ===
    def return_cleared_state(_initial_state_tuple):
      # Optional: Add debug print for allocation failure
      # jax.debug.print("Prefill failed for page group {id}: Not enough resources.", id=page_group_id)
      # Return the state as it was immediately after releasing the group's pages
      return released_state

    # Conditionally execute allocation or return the cleared state based on resource check
    return jax.lax.cond(
        has_enough_resources,
        allocate_and_update_state,  # Function to call if True
        return_cleared_state,  # Function to call if False
        (current_page_status, current_page_map, current_num_pages_used),  # Argument passed to chosen function
    )

  # Main condition: Handle zero length or normal allocation case
  return jax.lax.cond(true_length == 0, handle_zero_length, handle_normal_case, None)


def _update_decode_pages_global(
    page_state: PageState,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Updates pages globally for one step of autoregressive decoding.

  This function performs the following steps for all page groups simultaneously:
  1. Increments `sequence_lengths` for groups marked as `has_active_page`.
  2. Calculates the new `active_page_position` based on the incremented length.
  3. Determines which active groups now require a new page because their sequence
     length has crossed a page boundary (`required_pages > num_pages_used`) and
     they still have capacity (`required_pages <= max_pages_per_group`).
  4. Iterates conceptually across all groups (using `jax.lax.fori_loop`). For
     groups identified in step 3, it attempts to find a free page globally and
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

  # 1. Increment sequence lengths for active groups
  seq_len_increment = jnp.where(page_state.has_active_page, 1, 0)
  new_sequence_lengths = page_state.sequence_lengths + seq_len_increment

  # 2. Calculate new position within the active page for active groups
  new_active_page_position = jnp.where(
      page_state.has_active_page,
      (new_sequence_lengths - 1) % tokens_per_page,  # New position after increment
      page_state.active_page_position,  # Keep old position for inactive groups
  )

  # 3. Determine which groups need a new page allocated
  # Calculate the total number of pages *required* after the length increment
  required_pages_per_group = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page

  # Identify groups needing allocation: Active AND require more pages than currently used AND have capacity
  needs_new_page_mask = jnp.logical_and(page_state.has_active_page, required_pages_per_group > page_state.num_pages_used)
  has_capacity_mask = required_pages_per_group <= max_pages_per_group
  # Final mask: groups that meet all conditions for needing allocation attempt
  needs_allocation_mask = jnp.logical_and(needs_new_page_mask, has_capacity_mask)

  # 4. Iterate through groups, allocating conditionally
  # This function is applied per group index within the fori_loop
  def allocate_for_group_if_needed(group_idx, current_state: PageState):
    # Unpack state relevant to allocation changes within the loop
    current_status = current_state.page_status
    current_map = current_state.page_map
    current_num_used = current_state.num_pages_used
    current_active_page = current_state.active_page

    # Check if *this specific group* needs allocation in this step
    needs_alloc = needs_allocation_mask[group_idx]

    # Attempt to find a free page *only if needed*.
    # Finding it inside the loop handles potential state changes if loop acts like a scan.
    next_free_page_global = _find_next_free_page_index(current_status)
    # Can we allocate? Needs allocation AND found a free page
    can_allocate = jnp.logical_and(needs_alloc, next_free_page_global >= 0)

    # --- Perform updates conditionally based on `can_allocate` ---

    # Update global page status: Mark the found page as allocated (1)
    new_status = jax.lax.cond(can_allocate, lambda: current_status.at[next_free_page_global].set(1), lambda: current_status)

    # Update group's page map: Assign the new page's global index.
    # The index within the group's map where the new page goes is the *old* num_pages_used.
    page_map_index = current_num_used[group_idx]
    new_map = jax.lax.cond(
        can_allocate, lambda: current_map.at[group_idx, page_map_index].set(next_free_page_global), lambda: current_map
    )

    # Update group's page count: Increment num_pages_used for this group
    new_num_used = jax.lax.cond(can_allocate, lambda: current_num_used.at[group_idx].add(1), lambda: current_num_used)

    # Update group's active page: Set to the newly allocated page's global index
    new_active_page = jax.lax.cond(
        can_allocate, lambda: current_active_page.at[group_idx].set(next_free_page_global), lambda: current_active_page
    )

    # --- Reconstruct state for next iteration/return ---
    # Sequence lengths and positions were updated *before* the loop.
    # has_active_page is assumed constant during this decode step update.
    return PageState(
        page_status=new_status,
        page_map=new_map,
        num_pages_used=new_num_used,
        sequence_lengths=current_state.sequence_lengths,  # Already updated
        active_page=new_active_page,
        has_active_page=current_state.has_active_page,
        active_page_position=current_state.active_page_position,  # Already updated
    )

  # --- Execute the loop ---
  # Initialize loop state with the pre-calculated new lengths and positions
  initial_loop_state = PageState(
      page_status=page_state.page_status,
      page_map=page_state.page_map,
      num_pages_used=page_state.num_pages_used,
      sequence_lengths=new_sequence_lengths,  # Use new lengths
      active_page=page_state.active_page,
      has_active_page=page_state.has_active_page,
      active_page_position=new_active_page_position,  # Use new positions
  )

  # Apply the conditional allocation logic across all groups
  final_state = jax.lax.fori_loop(
      0,  # Start index
      max_page_groups,  # End index (exclusive)
      allocate_for_group_if_needed,  # Function to apply
      initial_loop_state,  # Initial state for the loop
  )

  # Optional: Post-loop check for allocation failures (e.g., ran out of pages mid-loop)
  # Could compare `needs_allocation_mask` with actual changes in `num_pages_used`.

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

    # Get initial page state (all pages free)
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
        * `pagedattn_max_page_groups`: The maximum number of concurrent page groups
          (requests/sequences) that can be managed.
        * `pagedattn_max_pages_per_group`: The maximum number of pages that can be
          allocated to a single page group.

    Raises:
      ValueError: If the configuration parameters are invalid (e.g., non-positive
        values, insufficient pages per group for max length).
    """
    self.num_pages = config.pagedattn_num_pages
    self.tokens_per_page = config.pagedattn_tokens_per_page
    self.max_target_length = config.max_target_length
    self.max_page_groups = config.pagedattn_max_page_groups
    self.max_pages_per_group = config.pagedattn_max_pages_per_group

    self._validate_init_params()

  def _validate_init_params(self) -> None:
    """Validates initialization parameters for logical consistency."""
    if self.max_pages_per_group <= 0:
      raise ValueError(f"Invalid `pagedattn_max_pages_per_group`: {self.max_pages_per_group}. Must be positive.")
    # Check if max_pages_per_group is sufficient to hold the longest possible sequence
    min_required_pages = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    if self.max_pages_per_group < min_required_pages:
      raise ValueError(
          f"`pagedattn_max_pages_per_group` ({self.max_pages_per_group}) must be at "
          f"least {min_required_pages} "
          f"to accommodate `max_target_length` ({self.max_target_length}) "
          f"with `pagedattn_tokens_per_page` ({self.tokens_per_page})."
      )
    if self.num_pages <= 0:
      raise ValueError(f"Invalid `pagedattn_num_pages`: {self.num_pages}. Must be positive.")
    if self.tokens_per_page <= 0:
      raise ValueError(f"Invalid `pagedattn_tokens_per_page`: {self.tokens_per_page}. Must be positive.")
    if self.max_page_groups <= 0:
      raise ValueError(f"Invalid `pagedattn_max_page_groups`: {self.max_page_groups}. Must be positive.")

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
      returned state will have the specified `page_group_id` cleared. If input
      validation fails, the original `page_state` is returned unchanged.

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
    # Input Validation using jax.lax.cond for JAX compatibility
    is_valid_group = jnp.logical_and(page_group_id >= 0, page_group_id < self.max_page_groups)
    is_valid_length = jnp.logical_and(true_length >= 0, true_length <= self.max_target_length)
    is_valid_input = jnp.logical_and(is_valid_group, is_valid_length)

    # Define the action for valid input
    def process_valid_request(current_page_state):
      # Delegate to the internal helper function
      return _update_prefill_pages_for_group(
          current_page_state,
          page_group_id,
          true_length,
          self.tokens_per_page,
          self.max_pages_per_group,
      )

    # Define the action for invalid input (return state unchanged)
    def return_original_state(current_page_state):
      # Optional: Log or signal invalid input if required outside JAX tracing
      # jax.debug.print("Invalid prefill request: page_group_id={gid} or true_length={len}", gid=page_group_id, len=true_length)
      return current_page_state

    # Conditionally execute based on input validity
    return jax.lax.cond(is_valid_input, process_valid_request, return_original_state, page_state)

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
      The updated `PageState` reflecting the state after the decode step. Groups
      that required and successfully obtained a new page will have their
      `num_pages_used`, `page_map`, and `active_page` updated.

    Example:
      ```python
      # Advance state for all active sequences by one decode step
      state = page_manager.update_decode_pages(state)
      ```
    """
    # Delegate to the internal helper function that handles the global update logic
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
      state. If input validation fails, the original `page_state` is returned.

    Example:
      ```python
      # Release all pages currently held by group 0
      state = page_manager.release_pages(
          page_state=state,
          page_group_id=0
      )
      ```
    """
    is_valid_input = jnp.logical_and(page_group_id >= 0, page_group_id < self.max_page_groups)

    def process_valid_request(current_page_state):
      return _release_pages_for_group(current_page_state, page_group_id, self.max_pages_per_group)

    # Define action for invalid input
    def return_original_state(current_page_state):
      return current_page_state

    # Conditionally execute based on input validity
    return jax.lax.cond(is_valid_input, process_valid_request, return_original_state, page_state)

  def get_initial_page_state(self) -> PageState:
    """Creates and returns an initial global `PageState`.

    This is a convenience method that calls `initialize_page_state` with
    the parameters (`num_pages`, `max_page_groups`, `max_pages_per_group`)
    stored during the `PageManager` initialization.

    Returns:
      An initialized `PageState` object where all pages are free and no groups
      are active.

    Example:
      ```python
      # Get a fresh, empty page state
      initial_state = page_manager.get_initial_page_state()
      ```
    """
    return initialize_page_state(
        num_pages=self.num_pages, max_page_groups=self.max_page_groups, max_pages_per_group=self.max_pages_per_group
    )
