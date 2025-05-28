# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
from typing import Tuple

from flax import struct
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Integer

from MaxText import common_types

Config = common_types.Config

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

  Attributes:
    page_status: A `jnp.ndarray` of shape `[num_pages]`. Each element
      indicates whether the corresponding page in the global pool is free (0)
      or allocated (1). Page 0 is specially handled: it's initially marked
      allocated (1) as a workaround and is explicitly freed before certain
      allocation operations.
    page_map: A `jnp.ndarray` of shape `[max_page_groups, max_pages_per_group]`.
      This array maps each page group to the global indices of its allocated
      pages. Entries beyond `num_pages_used` for a group are invalid and
      typically contain 0 or placeholder values.
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
      token slot *within the `active_page`* for each page group. Only valid if
      `has_active_page` is True.
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

  All pages in the global pool are initially marked as free (status 0),
  except for page 0 which is marked as used (status 1). This special
  handling of page 0 is often a workaround for JAX behaviors (e.g., with
  `argmax` on all-zero arrays or to ensure valid indices for scatter updates
  if no other page is allocated).

  No pages are assigned to any page group initially. Sequence lengths, page
  usage counts for each group, and active page tracking information are all
  initialized to zero or False as appropriate.

  Args:
    num_pages: The total number of available pages in the global pool.
    max_page_groups: The maximum number of page groups (e.g., concurrent
      sequences or requests) that the system can track. This defines the first
      dimension of group-specific arrays in `PageState`.
    max_pages_per_group: The maximum number of pages that can be allocated to
      a single page group. This determines the size of the second dimension
      of the `page_map` attribute in `PageState`.

  Returns:
    A `PageState` object representing the initialized global page state.
    Specifically:
      - `page_status[0]` is 1, `page_status[1:]` are 0.
      - `page_map` is all zeros.
      - `num_pages_used`, `sequence_lengths`, `active_page`,
        `active_page_position` are all zeros.
      - `has_active_page` is all False.
  """
  # TODO(patemotter): Produces garbage output for any request that uses page 0
  initial_page_status = jnp.zeros((num_pages,), dtype=jnp.int32)
  initial_page_status = initial_page_status.at[0].set(1)  # Workaround page 0
  return PageState(
      page_status=initial_page_status,
      page_map=jnp.zeros(
          (max_page_groups, max_pages_per_group), dtype=jnp.int32
      ),
      num_pages_used=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      sequence_lengths=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      active_page=jnp.zeros((max_page_groups,), dtype=jnp.int32),
      has_active_page=jnp.zeros((max_page_groups,), dtype=jnp.bool_),
      active_page_position=jnp.zeros((max_page_groups,), dtype=jnp.int32),
  )


@jax.jit
def _find_next_free_page_index(page_status: PagesInt1d) -> ScalarInt:
  """Finds the index of the next available free page in the global pool.

  This function searches the `page_status` array for the first occurrence of 0
  (indicating a free page), starting its search from index 1. Page 0 is
  intentionally excluded from this direct search due to its special handling.
  If `page_status` has only one element (i.e., only page 0 exists),
  it returns -1.

  Args:
    page_status: A 1D `jnp.ndarray` of shape `[num_pages]` representing the
      current allocation status of all pages in the pool (0 for free, 1 for
      allocated).

  Returns:
    A scalar `jnp.int32` JAX array containing the index of the first free page
    found (at an index >= 1). If no free pages are available at index 1 or
    greater, or if only page 0 exists, it returns -1.
  """
  if page_status.shape[0] <= 1:  # Handle the edge case of only one page (page 0)
    return jnp.array(-1, dtype=jnp.int32)

  # Search for a free page, excluding page 0 by slicing.
  # `jnp.argmax(condition)` returns the index of the first True.
  # If no page is free in search_status (all are 1), `argmax` would return 0
  # for `search_status == 0`.
  search_status = page_status[1:]
  return jax.lax.cond(
      jnp.any(search_status == 0),  # Check if any page (from index 1 onwards) is free
      lambda: jnp.argmax(search_status == 0) + 1, # Found: return its global index (add 1 due to slice)
      lambda: jnp.array(-1, dtype=jnp.int32),    # Not found: return -1
  )


@jax.jit
def _free_page_zero(page_state: PageState) -> PageState:
  """Explicitly marks page zero as free in the `PageState.page_status`.

  This utility function is typically called before operations that might
  allocate pages (like prefill), allowing page 0 to be considered for allocation
  by `_find_next_free_page_index` if it happens to be the first available one.

  Args:
    page_state: The current `PageState`.

  Returns:
    The updated `PageState` with `page_status[0]` set to 0 (free).
  """
  new_page_status = page_state.page_status.at[0].set(0)
  return page_state.replace(page_status=new_page_status)


@partial(jax.jit, static_argnames=("max_pages_per_group",))
def _release_pages_for_group(
    page_state: PageState,
    page_group_id: ScalarInt,
    max_pages_per_group: int,
) -> PageState:
  """Releases all pages currently associated with a given page group.

  This function identifies pages allocated to `page_group_id` by consulting
  `page_state.page_map` up to the count specified by
  `page_state.num_pages_used[page_group_id]`. It then marks these global page
  indices as free (0) in `page_state.page_status`.

  Additionally, it resets all state attributes specific to this `page_group_id`
  in the `PageState` (e.g., `num_pages_used`, `sequence_lengths`, `active_page`,
  `page_map` row for this group) to their default (zero or False) values.

  Args:
    page_state: The current global `PageState`.
    page_group_id: The scalar `jnp.int32` ID of the page group whose pages are
      to be released.
    max_pages_per_group: The static maximum number of pages a group can hold,
      used to define array shapes (e.g., for `page_map`) and for masking.

  Returns:
    An updated `PageState` where pages previously allocated to the specified
    group are marked as free in `page_status`, and the group's specific state
    entries (like `page_map` row, `num_pages_used`, etc.) are reset.
  """
  current_page_status = page_state.page_status
  current_page_map = page_state.page_map
  num_valid_pages_for_group = page_state.num_pages_used[page_group_id]

  # Create a mask for valid page entries within this group's row in page_map
  pages_to_release_mask = (
      jnp.arange(max_pages_per_group) < num_valid_pages_for_group
  )
  # Get all global page indices potentially mapped to this group
  pages_mapped_to_group = current_page_map[page_group_id]

  # Select only the valid global page indices that this group actually uses.
  # Invalid entries (beyond num_valid_pages_for_group) are set to -1.
  actual_pages_to_release = jnp.where(
      pages_to_release_mask, pages_mapped_to_group, -1
  )
  # Further filter to ensure we only use non-negative indices for the status update.
  # This `valid_pages_to_release` array might still contain -1 where
  # `actual_pages_to_release` was -1.
  valid_pages_to_release = jnp.where(actual_pages_to_release >= 0,
                                       actual_pages_to_release, -1)

  # Update page_status: set to 0 for all valid_pages_to_release.
  # If `valid_pages_to_release` contains -1, JAX's scatter `at[indices].set()`
  # behavior with out-of-bounds indices can be backend-dependent or error-prone.
  # However, the `jnp.any(valid_pages_to_release >= 0)` condition ensures
  # this is only called if there's at least one valid positive index.
  # The original code relies on JAX's handling of this.
  new_page_status = jax.lax.cond(
      jnp.any(valid_pages_to_release >= 0),
      lambda: current_page_status.at[valid_pages_to_release].set(0),
      lambda: current_page_status,
  )

  # Clear the page_map row for this group by setting it to zeros.
  cleared_group_map_row = jnp.zeros(
      (max_pages_per_group,), dtype=page_state.page_map.dtype
  )
  new_page_map = current_page_map.at[page_group_id].set(cleared_group_map_row)

  # Reset all other state attributes for this page group.
  return page_state.replace(
      page_status=new_page_status,
      page_map=new_page_map,
      num_pages_used=page_state.num_pages_used.at[page_group_id].set(0),
      sequence_lengths=page_state.sequence_lengths.at[page_group_id].set(0),
      active_page=page_state.active_page.at[page_group_id].set(0),
      has_active_page=page_state.has_active_page.at[page_group_id].set(False),
      active_page_position=page_state.active_page_position.at[
          page_group_id
      ].set(0),
  )


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _reserve_pages_for_group(
    released_state: PageState,
    page_group_id: ScalarInt,
    true_length: ScalarInt,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Reserves the necessary number of pages for a given page group.

  This function assumes `true_length` is greater than 0. It calculates the
  number of pages (`num_pages_needed`) required for the given `true_length`.
  It then checks for resource availability:
  1.  If the global pool has at least `num_pages_needed` free pages (checked
      via `jnp.sum(current_page_status == 0)`).
  2.  If `num_pages_needed` does not exceed `max_pages_per_group`.

  If resources are adequate (`has_enough_resources` is True), it iteratively
  allocates pages:
  - In each step of a `fori_loop` (running `num_pages_needed` times):
    - It finds the next free global page index using `_find_next_free_page_index`.
    - Marks this page as allocated (1) in `page_status`.
    - Records the global page index in `page_map` for the `page_group_id`
      at the current iteration index `i`.
    - Increments `num_pages_used` for the `page_group_id`.
  - After the loop, it updates the group's `sequence_lengths`, sets `has_active_page`
    to True, sets `active_page` to the global index of the last page allocated,
    and calculates `active_page_position` (the next slot to write to in the active page).

  If resources are insufficient, the function returns the `released_state`
  unchanged, meaning the group remains un-allocated or in its previously
  released state.

  Args:
    released_state: The global `PageState`, typically after any pages previously
      held by `page_group_id` have been released (its `num_pages_used` should be 0).
      Page 0 might be free or used in this state.
    page_group_id: The scalar `jnp.int32` ID of the page group for which to
      reserve pages.
    true_length: The scalar `jnp.int32` true length of the sequence (in tokens)
      for which pages are being reserved. It is assumed `true_length > 0` by
      the caller logic (`_release_and_reserve_for_group`).
    tokens_per_page: The static integer number of tokens each page can hold.
    max_pages_per_group: The static integer maximum number of pages allowed
      per group.

  Returns:
    The updated `PageState` with pages allocated to the specified group if
    successful. If allocation fails due to insufficient resources, returns the
    input `released_state` unmodified.
  """
  num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
  # Calculate position details for the last token to determine next write slot
  last_token_abs_idx = true_length - 1
  last_page_position_idx = last_token_abs_idx % tokens_per_page
  next_write_position = (last_page_position_idx + 1) % tokens_per_page

  current_page_status = released_state.page_status
  current_page_map = released_state.page_map
  # `current_num_pages_used` for `page_group_id` should be 0 from `released_state`.
  current_num_pages_used = released_state.num_pages_used


  # Check for resource availability
  # Note: jnp.sum(current_page_status == 0) includes page 0 if it's free.
  num_free_pages = jnp.sum(current_page_status == 0)
  group_has_capacity = num_pages_needed <= max_pages_per_group
  sufficient_free_pages = num_free_pages >= num_pages_needed
  has_enough_resources = jnp.logical_and(
      sufficient_free_pages, group_has_capacity
  )

  def allocate_and_update_state():
    """Internal helper to perform allocation if resources are sufficient."""
    # These are the initial states for the allocation loop.
    # `new_num_used` for `page_group_id` is 0 at the start of this function's scope.
    new_status = current_page_status
    new_map = current_page_map
    new_num_used = current_num_pages_used # The full num_pages_used array

    def loop_body(i, loop_vars):
      """Body of the fori_loop for allocating one page at a time.
      'i' is the current page index relative to the group (0 to num_pages_needed-1).
      """
      current_status, current_map, current_num_used_for_group = loop_vars
      # Find next free page (searches from index 1, page 0 handled by _free_page_zero before call)
      next_free_page_global = _find_next_free_page_index(current_status)

      # Conditionally update based on whether a free page was found
      page_is_allocatable = next_free_page_global >= 0

      new_status_iter = jax.lax.cond(
          page_is_allocatable,
          lambda s: s.at[next_free_page_global].set(1), # Mark page as used
          lambda s: s, # No change if no page found
          current_status,
      )
      new_map_iter = jax.lax.cond(
          page_is_allocatable,
          # Record the global page index in the group's map at slot `i`
          lambda m: m.at[page_group_id, i].set(next_free_page_global),
          lambda m: m, # No change
          current_map,
      )
      # Increment `num_pages_used` for the specific `page_group_id`
      new_num_used_for_group_iter = jax.lax.cond(
          page_is_allocatable,
          lambda n: n.at[page_group_id].add(1),
          lambda n: n, # No change
          current_num_used_for_group, # This is the full num_pages_used array
      )
      return new_status_iter, new_map_iter, new_num_used_for_group_iter

    # Iteratively allocate pages
    # Initial `new_num_used` passed to loop_body contains 0 for page_group_id.
    final_status, final_map, final_num_used_array = jax.lax.fori_loop(
        0, num_pages_needed, loop_body, (new_status, new_map, new_num_used)
    )

    # Determine the active page (the last one allocated for this group)
    active_page_global_index = jax.lax.cond(
        num_pages_needed > 0,
        # Get the global index from the map at the last allocated slot for this group
        lambda: final_map[page_group_id, num_pages_needed - 1],
        lambda: jnp.array(0, dtype=jnp.int32), # Fallback (e.g. if num_pages_needed was 0)
    )

    return released_state.replace(
        page_status=final_status,
        page_map=final_map,
        num_pages_used=final_num_used_array, # The full updated array
        sequence_lengths=released_state.sequence_lengths.at[
            page_group_id
        ].set(true_length),
        active_page=released_state.active_page.at[page_group_id].set(
            active_page_global_index
        ),
        has_active_page=released_state.has_active_page.at[
            page_group_id
        ].set(True),
        active_page_position=released_state.active_page_position.at[
            page_group_id
        ].set(next_write_position),
    )

  # Conditionally perform allocation or return the state as it was.
  return jax.lax.cond(
      has_enough_resources,
      allocate_and_update_state, # Call the allocation function
      lambda: released_state     # If not enough resources, return state prior to this attempt
  )


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _release_and_reserve_for_group(
    page_state: PageState,
    page_group_id: ScalarInt,
    true_length: ScalarInt,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PageState:
  """Releases existing pages and then reserves new pages for a group.

  This function is typically used during prefill. It performs two main steps:
  1.  Calls `_release_pages_for_group` to free any pages currently allocated
      to `page_group_id` and reset its associated state fields.
  2.  If `true_length` is greater than 0, it then calls `_reserve_pages_for_group`
      to allocate the required number of new pages based on `true_length`.
      If `true_length` is 0, no new pages are reserved, and the group effectively
      remains in its cleared (empty) state after the release.

  Args:
    page_state: The current global `PageState`.
    page_group_id: The scalar `jnp.int32` ID of the page group to update.
    true_length: The scalar `jnp.int32` true length of the sequence (in tokens).
      If 0, the group is effectively just cleared of pages.
    tokens_per_page: The static integer number of tokens per page.
    max_pages_per_group: The static integer maximum number of pages allowed
      per group.

  Returns:
    The updated `PageState`. If `true_length` is 0, the group will be empty.
    If `true_length` > 0, pages will be reserved if resources permit; otherwise,
    the group might remain empty if the reservation step failed.
  """
  # Release any existing pages for this group first.
  released_state = _release_pages_for_group(
      page_state, page_group_id, max_pages_per_group
  )

  # Only reserve new pages if true_length > 0.
  # If true_length is 0, the group is simply cleared and no reservation is attempted.
  final_state = jax.lax.cond(
      true_length > 0,
      # If true_length > 0, attempt to reserve pages using the 'released_state'.
      lambda rs: _reserve_pages_for_group(
          rs, page_group_id, true_length, tokens_per_page, max_pages_per_group
      ),
      # If true_length is 0, no reservation needed; return the state after release.
      lambda rs: rs,
      released_state
  )
  return final_state


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _update_decode_pages_global(
    page_state: PageState,
    tokens_per_page: ScalarInt, # Keeping original ScalarInt type hint
    max_pages_per_group: ScalarInt, # Keeping original ScalarInt type hint
) -> PageState:
  """Updates pages globally for one step of autoregressive decoding.

  This function processes a single decode step for all active page groups
  simultaneously (vectorized where possible, with a loop for serialized allocation):
  1.  Increments `sequence_lengths` by 1 for all groups where `has_active_page`
      is True.
  2.  Calculates the new `active_page_position` for these active groups based
      on their new `sequence_lengths`. This is the token offset within the
      current active page.
  3.  Identifies active groups that now require a new page (`needs_allocation_mask`):
      a.  Their new `sequence_lengths` require more pages than `num_pages_used`.
      b.  They have not exceeded `max_pages_per_group`.
  4.  Iterates (using `jax.lax.fori_loop`) through all page groups (`max_page_groups` times).
      In each iteration `group_idx`:
      a.  Checks if `needs_allocation_mask[group_idx]` is True.
      b.  If so, attempts to find a free global page using `_find_next_free_page_index`
          (using the page status *carried over from the previous iteration*).
      c.  If allocation is possible for this group, its `page_status` (for the new page),
          `page_map`, `num_pages_used`, and `active_page` (set to the new page)
          are updated in the loop's carry state. This serialized allocation
          ensures that page status updates are seen by subsequent groups in the loop.
  5.  Returns the final `PageState` after the loop.

  Args:
    page_state: The current global `PageState`.
    tokens_per_page: The number of tokens per page (as a JAX scalar int).
      Treated as a static argument by `partial(jax.jit)`.
    max_pages_per_group: The maximum number of pages allowed per group (as a
      JAX scalar int). Treated as a static argument by `partial(jax.jit)`.

  Returns:
    An updated `PageState` reflecting changes after the decode step. This includes
    updated sequence lengths, active positions, and potentially newly allocated
    pages for groups that crossed a page boundary and had capacity/resources.
  """
  max_num_page_groups = page_state.sequence_lengths.shape[0]

  # Step 1: Increment sequence length for active groups
  seq_len_increment = jnp.where(page_state.has_active_page, 1, 0)
  new_sequence_lengths = page_state.sequence_lengths + seq_len_increment

  # Step 2: Update active page position for active groups
  # This is the position *within the current active page* where the new token lands.
  new_active_page_position = jnp.where(
      page_state.has_active_page,
      (new_sequence_lengths - 1) % tokens_per_page, # Position of the newly added token
      page_state.active_page_position, # No change for inactive groups
  )

  # Step 3: Determine which groups need a new page
  required_pages_per_group = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page
  # Needs a new page if it's active AND its new required pages > currently used pages
  needs_new_page_mask = jnp.logical_and(
      page_state.has_active_page,
      required_pages_per_group > page_state.num_pages_used
  )
  # Also, the group must have capacity for more pages
  has_capacity_mask = required_pages_per_group <= max_pages_per_group
  # Final mask: needs allocation if it needs a new page AND has capacity
  needs_allocation_mask = jnp.logical_and(needs_new_page_mask, has_capacity_mask)

  # Step 4: Inner function for fori_loop to conditionally allocate a page.
  # This loop ensures that page allocations are serialized if multiple groups need pages.
  def allocate_for_group_if_needed(
        group_idx: ScalarInt, current_carry_state: PageState
    ) -> PageState:
    """
    Loop body for `jax.lax.fori_loop` to conditionally allocate a page for a group.
    This function is applied sequentially for each `group_idx`. Page status
    updates from a previous iteration (earlier `group_idx`) are visible to
    subsequent iterations via `current_carry_state.page_status`.

    Args:
      group_idx: The index of the current page group being processed.
      current_carry_state: The `PageState` carried from the previous iteration
                           or the initial state for the first iteration. This state
                           already includes updated sequence_lengths and
                           active_page_positions.

    Returns:
      The `PageState` after potentially allocating a page for `group_idx`.
    """
    needs_alloc_for_this_group = needs_allocation_mask[group_idx]
    # Find the next free page using the *current* page_status from loop carry.
    # This ensures that allocations made for earlier groups in the loop are reflected.
    next_free_page_global = _find_next_free_page_index(current_carry_state.page_status)

    # Can allocate if this group needs it AND a free page is available globally
    can_allocate_for_this_group = jnp.logical_and(
        needs_alloc_for_this_group, next_free_page_global >= 0
    )

    # Conditionally update PageState fields if allocation happens for *this specific group*
    new_page_status_for_carry = jax.lax.cond(
        can_allocate_for_this_group,
        lambda s: s.at[next_free_page_global].set(1), # Mark page as used
        lambda s: s, # No change to page_status
        current_carry_state.page_status
    )
    # Index in the group's page_map row where the new global page index will go.
    # This is the current number of pages used by this group *before* this potential new allocation.
    page_map_index_for_this_group = current_carry_state.num_pages_used[group_idx]
    new_page_map_for_carry = jax.lax.cond(
        can_allocate_for_this_group,
        lambda m: m.at[group_idx, page_map_index_for_this_group].set(next_free_page_global),
        lambda m: m, # No change to page_map
        current_carry_state.page_map
    )
    new_num_pages_used_for_carry = jax.lax.cond(
        can_allocate_for_this_group,
        lambda n: n.at[group_idx].add(1), # Increment count for this group
        lambda n: n, # No change to num_pages_used
        current_carry_state.num_pages_used
    )
    # If a new page was allocated, it becomes the active page for this group.
    new_active_page_for_carry = jax.lax.cond(
        can_allocate_for_this_group,
        lambda a: a.at[group_idx].set(next_free_page_global),
        lambda a: a, # Active page for this group doesn't change if no new page allocated now
        current_carry_state.active_page
    )

    # Return the updated state for the next iteration or as the final result of the loop.
    # Note: sequence_lengths and active_page_position were updated *before* this loop started.
    return current_carry_state.replace(
        page_status=new_page_status_for_carry,
        page_map=new_page_map_for_carry,
        num_pages_used=new_num_pages_used_for_carry,
        active_page=new_active_page_for_carry,
    )

  # Initialize the loop's carry state.
  # Sequence lengths and active page positions are updated vectorially *before* the loop.
  initial_loop_state = page_state.replace(
      sequence_lengths=new_sequence_lengths,
      active_page_position=new_active_page_position,
  )

  # Apply conditional allocation across all groups via the loop.
  final_state_after_loop = jax.lax.fori_loop(
      0, max_num_page_groups, allocate_for_group_if_needed, initial_loop_state
  )
  return final_state_after_loop


class PageManager:
  """Manages the global allocation and release of pages for paged attention.

  This class provides an interface for:
  - Initializing the global page state (`get_initial_page_state`).
  - Reserving pages for sequences during prefill (`update_prefill_pages`).
  - Updating page allocations during autoregressive decoding steps
    (`update_decode_pages`).
  - Releasing pages when a sequence (represented by a page group) is completed
    or no longer needed (`release_pages`).

  It encapsulates the logic for tracking page allocation globally using the
  `PageState` dataclass. Page groups, typically corresponding to individual
  requests or sequences within a batch, are used to manage allocations for
  concurrently processed sequences.

  Static configuration parameters such as the total number of pages, tokens
  per page, maximum target sequence length, maximum number of page groups,
  and maximum pages allowed per group are provided during the `PageManager`'s
  initialization via a `Config` object.

  Attributes:
    num_pages: The total number of pages in the global memory pool.
    tokens_per_page: The number of tokens each page can hold.
    max_target_length: The maximum length of a sequence (in tokens) that can
      be processed. This influences the calculation of `max_pages_per_group`
      if not explicitly set.
    max_page_groups: The maximum number of page groups (e.g., concurrent
      sequences or requests) the PageManager is configured to handle. This is
      often derived from batch size configuration in `Config`.
    max_pages_per_group: The maximum number of pages that can be allocated to
      a single page group. If not explicitly provided in the configuration,
      a reasonable default is calculated based on `max_target_length` and
      `tokens_per_page`.

  Example:
    ```python
    # Assuming `config` is a MaxText Config object with paged attention settings:
    # config.pagedattn_num_pages = 1024
    # config.pagedattn_tokens_per_page = 16
    # config.max_target_length = 2048
    # # For max_page_groups, assuming global_batch_size_to_load is set:
    # config.global_batch_size_to_load = 4
    # # pagedattn_max_pages_per_group can be set or derived. If derived:
    # # min_pages = (2048 + 16 - 1) // 16 = 128
    # # config.pagedattn_max_pages_per_group = 128 (or some value >= min_pages)

    page_manager = PageManager(config)

    # Get initial page state
    current_page_state = page_manager.get_initial_page_state()

    # Prefill for page_group_id 0, with a sequence of true_length 30
    # Note: update_prefill_pages internally calls _free_page_zero.
    current_page_state = page_manager.update_prefill_pages(
        page_state=current_page_state,
        page_group_id=0,
        true_length=30
    )

    # Simulate a decode step (this affects all active sequences)
    current_page_state = page_manager.update_decode_pages(current_page_state)

    # When sequence in page_group_id 0 is finished, release its pages
    current_page_state = page_manager.release_pages(
        page_state=current_page_state,
        page_group_id=0
    )
    ```
  """

  def __init__(self, config: Config):
    """Initializes the `PageManager` from a configuration object.

    Args:
      config: A `Config` object (e.g., from `MaxText.common_types`)
        containing parameters relevant to paged attention. Expected attributes include:
        - `pagedattn_num_pages` (int): Total number of global pages available.
        - `pagedattn_tokens_per_page` (int): Number of tokens each page can hold.
        - `max_target_length` (int): Maximum sequence length the model supports.
        - `global_batch_size_to_load` (int, optional): Global batch size.
          Used to determine `max_page_groups`.
        - `per_device_batch_size` (int, optional): Per-device batch size.
          Used if `global_batch_size_to_load` is not available.
        - `pagedattn_max_pages_per_group` (int, optional): Max pages that can
          be allocated to a single sequence/group. If not set, it's
          calculated based on `max_target_length` and `pagedattn_tokens_per_page`.

    Raises:
      ValueError: If any critical configuration parameters are invalid (e.g.,
        non-positive values, insufficient pages per group for the max target
        length, or `num_pages` not being greater than 1).
    """
    self.num_pages: int = config.pagedattn_num_pages
    self.tokens_per_page: int = config.pagedattn_tokens_per_page
    self.max_target_length: int = config.max_target_length
    # Determine max_page_groups (concurrent sequences) from batch size config.
    # Original logic from user's code:
    self.max_page_groups: int = getattr(config, 'global_batch_size_to_load', getattr(config, 'per_device_batch_size', 1) * jax.device_count() if jax.device_count() > 0 else 1)

    # Calculate max_pages_per_group if not explicitly set in config.
    # This is the minimum number of pages required to hold the max_target_length.
    min_required_pages = (
        self.max_target_length + self.tokens_per_page - 1
    ) // self.tokens_per_page
    self.max_pages_per_group: int = getattr(
        config, 'pagedattn_max_pages_per_group', min_required_pages
    )

    self._validate_init_params()

  def _validate_init_params(self) -> None:
    """Validates initialization parameters for logical consistency.

    Ensures that page counts, tokens per page, group capacities, and other
    settings are positive and sufficient for the configured maximum target
    sequence length.

    Raises:
      ValueError: If any of the following conditions are not met:
        - `max_pages_per_group` must be positive.
        - `max_pages_per_group` must be sufficient to hold `max_target_length`.
        - `num_pages` must be greater than 1 (due to page 0's special handling).
        - `tokens_per_page` must be positive.
        - The derived `max_page_groups` (from batch sizes) must be positive.
    """
    if self.max_pages_per_group <= 0:
      raise ValueError("`pagedattn_max_pages_per_group` must be positive.")
    min_required = (
        self.max_target_length + self.tokens_per_page - 1
    ) // self.tokens_per_page
    if self.max_pages_per_group < min_required:
      raise ValueError(
          f"`pagedattn_max_pages_per_group` ({self.max_pages_per_group}) is "
          f"insufficient for `max_target_length` ({self.max_target_length}) "
          f"with `tokens_per_page` ({self.tokens_per_page}). Needs at least "
          f"{min_required} pages per group."
      )
    if self.num_pages <= 1:  # Check > 1 due to page 0 workaround/special handling
      raise ValueError(
          "`pagedattn_num_pages` must be greater than 1. Got: {self.num_pages}"
      )
    if self.tokens_per_page <= 0:
      raise ValueError(
          "`pagedattn_tokens_per_page` must be positive. Got: {self.tokens_per_page}"
      )
    if self.max_page_groups <= 0:
      raise ValueError(
          "Derived `max_page_groups` (from batch sizes) must be positive. "
          f"Got: {self.max_page_groups}"
      )

  def update_prefill_pages(
      self, page_state: PageState, page_group_id: int, true_length: int
  ) -> PageState:
    """Reserves pages for a specific page group, typically during sequence prefill.

    This method first validates that `page_group_id` and `true_length` are
    within their allowed ranges. It then prepares for page allocation by:
    1.  Converting `page_group_id` and `true_length` to JAX scalar arrays.
    2.  Calling `_free_page_zero` to ensure page 0 is marked as free and can be
        considered for allocation if it's the first available page.
    3.  Invoking the JIT-compiled `_release_and_reserve_for_group` function.
        This internal function first releases any pages currently held by
        `page_group_id` and then, if `true_length > 0`, attempts to reserve
        the new required number of pages.

    If `true_length` is 0, the group is effectively cleared of all pages, and
    no new pages are reserved. If `true_length > 0` but the reservation step
    fails (e.g., due to insufficient global free pages or the group exceeding
    its `max_pages_per_group` limit), the group will remain in its cleared state.

    Args:
      page_state: The current global `PageState`.
      page_group_id: The integer ID of the page group (e.g., a sequence index
        within a batch) for which to reserve pages. Must be in the range
        `[0, self.max_page_groups - 1]`.
      true_length: The integer true length of the sequence (in tokens) for
        which pages are to be reserved. Must be in the range
        `[0, self.max_target_length]`. If 0, the group is effectively cleared.

    Returns:
      The updated `PageState`. If `true_length > 0` and allocation was
      successful, the specified group will have newly allocated pages. Otherwise
      (e.g., `true_length` is 0, or allocation failed due to resource limits),
      the group will be in an empty/cleared state.

    Raises:
      ValueError: If `page_group_id` is out of its valid range.
      ValueError: If `true_length` is negative or exceeds `max_target_length`.
    """
    if not (0 <= page_group_id < self.max_page_groups):
      raise ValueError(
          f"PageManager: page_group_id ({page_group_id}) out of range "
          f"[0, {self.max_page_groups - 1}]"
      )
    if not (0 <= true_length <= self.max_target_length) : # Allow true_length == 0 for clearing
      raise ValueError(
          f"PageManager: true_length ({true_length}) exceeds max_target_length "
          f"({self.max_target_length}) or is negative."
      )

    page_group_id_jax = jnp.array(page_group_id, dtype=jnp.int32)
    true_length_jax = jnp.array(true_length, dtype=jnp.int32)

    # Explicitly free page 0 before attempting to reserve.
    # This makes page 0 available for allocation by `_reserve_pages_for_group`
    # if it's chosen by `_find_next_free_page_index`.
    state_after_freeing_zero = _free_page_zero(page_state)

    return _release_and_reserve_for_group(
        state_after_freeing_zero, # Use state where page 0 is potentially free
        page_group_id_jax,
        true_length_jax,
        self.tokens_per_page,
        self.max_pages_per_group,
    )

  def update_decode_pages(self, page_state: PageState) -> PageState:
    """Updates pages globally for one step of autoregressive decoding.

    This method invokes the JIT-compiled `_update_decode_pages_global` function.
    This function handles the logic for advancing the state of all active page
    groups by one token generation step. This includes:
    - Incrementing sequence lengths for active groups.
    - Updating their current write positions within their active pages.
    - If an active group's sequence crosses a page boundary (i.e., it now
      requires more pages than it currently has allocated):
        - An attempt is made to allocate a new page from the global pool,
          provided the group has not reached its `max_pages_per_group` limit
          and a free page is available.
        - Page allocations are handled serially within the JITted function
          to ensure correct accounting of free pages if multiple groups
          need new pages simultaneously.

    Page 0 is not explicitly freed here; `_find_next_free_page_index` within
    the JITted function will manage finding free pages (which might include
    page 0 if it was freed by a prior operation, like prefill, and remains free).

    Args:
      page_state: The current global `PageState`.

    Returns:
      The updated `PageState` reflecting sequence length increments, write
      position updates, and any new page allocations that occurred for groups
      that required them and for which resources were available.
    """
    # Pass self.tokens_per_page and self.max_pages_per_group directly
    # as Python integers, as they are static arguments for the JIT.
    return _update_decode_pages_global(
        page_state,
        self.tokens_per_page,
        self.max_pages_per_group
    )

  def release_pages(self, page_state: PageState, page_group_id: int) -> PageState:
    """Releases all pages currently associated with a given page group.

    This method first validates that the `page_group_id` is within the allowed
    range. It then converts `page_group_id` to a JAX scalar array and calls
    the JIT-compiled `_release_pages_for_group` function.

    The `_release_pages_for_group` function performs the core logic:
    - Identifies all global page indices that were allocated to the specified
      `page_group_id` by looking at its `page_map` entries up to its
      `num_pages_used`.
    - Marks these identified global pages as free (status 0) in the global
      `page_status` array.
    - Resets all state information specific to the `page_group_id` (such as
      its `sequence_length`, `num_pages_used`, `active_page` status, etc.)
      to their initial zero/False values.
    - Clears the `page_map` entries for this `page_group_id`.

    Args:
      page_state: The current global `PageState`.
      page_group_id: The integer ID of the page group (e.g., a sequence index
        within a batch) whose pages are to be released. Must be in the range
        `[0, self.max_page_groups - 1]`.

    Returns:
      The updated `PageState` after the specified group's pages have been
      successfully freed and its associated state has been reset.

    Raises:
      ValueError: If `page_group_id` is out of its valid range.
    """
    if not (0 <= page_group_id < self.max_page_groups):
      raise ValueError(
          f"PageManager: page_group_id ({page_group_id}) out of range "
          f"[0, {self.max_page_groups - 1}]"
      )

    page_group_id_jax = jnp.array(page_group_id, dtype=jnp.int32)
    return _release_pages_for_group(
        page_state, page_group_id_jax, self.max_pages_per_group
    )

  def get_initial_page_state(self) -> PageState:
    """Creates and returns an initial global `PageState`.

    This is a convenience method that calls the top-level `initialize_page_state`
    function. It uses the configuration parameters (`num_pages`,
    `max_page_groups`, `max_pages_per_group`) that were stored as attributes
    on the `PageManager` instance during its initialization.

    In the returned `PageState`:
    - Page 0 will be marked as used (allocated, status 1) as per the
      `initialize_page_state` logic.
    - All other pages will be marked as free (status 0).
    - No page groups will be active or have any pages assigned to them; their
      respective state fields (`num_pages_used`, `sequence_lengths`, etc.)
      will be initialized to zeros or False.

    Returns:
      A newly initialized `PageState` object, ready for use with the
      `PageManager`'s methods.
    """
    return initialize_page_state(
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )