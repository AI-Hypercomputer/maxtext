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

""" Tests for Page Manager. """

import sys
import unittest

import jax
import jax.numpy as jnp

from MaxText import pyconfig
from maxtext.inference.page_manager import PageManager, PageState
from tests.utils.test_helpers import get_test_config_path


class TestPageManager(unittest.TestCase):
  """Test page manager."""

  def setUp(self):
    super().setUp()
    self.num_pages = 128
    self.tokens_per_page = 8
    self.max_prefill_predict_length = 128
    self.max_target_length = 256
    self.max_pages_per_group = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page

    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_prefill_predict_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        pagedattn_num_pages=self.num_pages,
        pagedattn_tokens_per_page=self.tokens_per_page,
        pagedattn_max_pages_per_group=self.max_pages_per_group,
    )
    self.config = config
    self.max_page_groups = self.config.global_batch_size_to_load

    print("Note: Running PageManager tests locally without a JAX mesh.")

    self.key = jax.random.PRNGKey(0)
    self.pm = PageManager(config=self.config)

  def _validate_state_shapes(self, state: PageState):
    """Helper function to assert that all PageState arrays have correct global shapes."""
    # Expected shapes for the global state version
    expected_shapes = {
        "page_status": (self.num_pages,),
        "page_map": (self.max_page_groups, self.max_pages_per_group),
        "sequence_lengths": (self.max_page_groups,),
        "num_pages_used": (self.max_page_groups,),
        "active_page": (self.max_page_groups,),
        "has_active_page": (self.max_page_groups,),
        "active_page_position": (self.max_page_groups,),
    }

    for field, expected in expected_shapes.items():
      self.assertTrue(hasattr(state, field), f"State object missing field: {field}")
      actual_shape = getattr(state, field).shape
      self.assertEqual(
          actual_shape,
          expected,
          f"Shape mismatch for {field}: expected {expected}, got {actual_shape}",
      )

  def _assert_page_state_equal(self, state1: PageState, state2: PageState, msg=None):
    """Helper function to compare two PageState objects field by field."""
    self.assertIsInstance(state1, PageState)
    self.assertIsInstance(state2, PageState)
    for field in state1.__dataclass_fields__:
      field_name = field
      val1 = getattr(state1, field_name)
      val2 = getattr(state2, field_name)
      self.assertTrue(
          jnp.array_equal(val1, val2),
          msg=f"{msg or ''}: Field '{field_name}' mismatch.\nState1:\n{state1}\nState2:\n{state2}",
      )

  def test_initialization(self):
    """Tests the initialization of PageState: checks shapes and initial values (global state)."""
    state = self.pm.get_initial_page_state()
    self._validate_state_shapes(state)  # Verify all array shapes for global state
    self.assertTrue(jnp.all(state.page_status[1:] == 0), "All pages should be initially free (status 0)")
    self.assertTrue(jnp.all(state.num_pages_used[1:] == 0), "No pages should be used initially")
    self.assertTrue(jnp.all(state.sequence_lengths[1:] == 0), "Sequence lengths should be 0 initially")
    self.assertTrue(
        jnp.all(state.has_active_page[1:] == False),  # pylint: disable=singleton-comparison
        "No groups should be active initially",
    )

  def test_reserve_prefill_group(self):
    """Tests update_prefill_pages: standard case with sufficient space (global state)."""
    page_group_id = 0
    true_length = 12

    initial_state = self.pm.get_initial_page_state()

    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=true_length
    )

    # Calculate the expected number of pages needed
    pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page

    # Check global state (no layer loop)
    self.assertEqual(int(updated_state.sequence_lengths[page_group_id]), true_length)
    self.assertEqual(int(updated_state.num_pages_used[page_group_id]), pages_needed)
    self.assertTrue(bool(updated_state.has_active_page[page_group_id]))  # Should be active

    # Get the indices of allocated pages from the global page_map
    page_map_group = updated_state.page_map[page_group_id]
    num_used = updated_state.num_pages_used[page_group_id]
    # Slice the map to get only the used indices
    used_page_indices = page_map_group[:num_used]

    self.assertEqual(len(used_page_indices), pages_needed, "Incorrect number of page indices in map")
    # Ensure unique pages were allocated
    self.assertEqual(len(jnp.unique(used_page_indices)), pages_needed, "Allocated page indices are not unique")
    # Ensure no placeholder zeros remain if pages_needed > 0
    if pages_needed > 0:
      self.assertTrue(
          jnp.all(used_page_indices >= 0), "Valid page indices should be non-negative"
      )  # Assuming 0 is a valid page index

    # Verify that the allocated pages are marked as used in the global page_status
    allocated_status_sum = int(jnp.sum(updated_state.page_status))
    self.assertGreaterEqual(allocated_status_sum, pages_needed)  # Can be greater if other groups exist
    for page_idx in used_page_indices:
      # Check status of the specific allocated pages
      self.assertEqual(int(updated_state.page_status[page_idx]), 1, f"Page {page_idx} should be marked as used")

    # Verify position within the last page
    expected_pos = true_length % self.tokens_per_page
    self.assertEqual(int(updated_state.active_page_position[page_group_id]), expected_pos)

  def test_reserve_prefill_no_space(self):
    """Tests update_prefill_pages when there's no space available globally."""
    page_group_id = 0
    true_length = 12

    # Initialize and then set *all* global pages to be used.
    initial_state = self.pm.get_initial_page_state()
    initial_state = initial_state.replace(page_status=jnp.ones((self.num_pages,), dtype=jnp.int32))

    # Call update_prefill_pages; since no pages are free, the state for the group should be cleared/remain cleared.
    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=true_length
    )

    # Verify the state for the specific group is cleared (or unchanged from initial cleared state)
    # The global page_status remains all ones because allocation failed.
    self.assertTrue(jnp.all(updated_state.page_status == 1))
    self.assertEqual(int(updated_state.sequence_lengths[page_group_id]), 0)
    self.assertEqual(int(updated_state.num_pages_used[page_group_id]), 0)
    self.assertFalse(bool(updated_state.has_active_page[page_group_id]))

  def test_reserve_prefill_edge_cases(self):
    """Tests update_prefill_pages with edge cases: exact page multiple, partial page (global state)."""
    initial_state = self.pm.get_initial_page_state()

    # Test case 1: Exact page multiple - correct number of pages should be allocated.
    # Use a different group ID to avoid interference if state is carried over
    length_multiple = self.tokens_per_page * 2
    state_multiple = self.pm.update_prefill_pages(page_state=initial_state, page_group_id=1, true_length=length_multiple)
    self.assertEqual(int(state_multiple.sequence_lengths[1]), length_multiple)
    self.assertEqual(int(state_multiple.num_pages_used[1]), 2)
    self.assertTrue(bool(state_multiple.has_active_page[1]))
    self.assertEqual(int(state_multiple.active_page_position[1]), 0)

    # Test case 2: Partial page - only one page should be allocated.
    length_partial = 5
    state_partial = self.pm.update_prefill_pages(page_state=initial_state, page_group_id=2, true_length=length_partial)
    self.assertEqual(int(state_partial.sequence_lengths[2]), length_partial)
    self.assertEqual(int(state_partial.num_pages_used[2]), 1)
    self.assertTrue(bool(state_partial.has_active_page[2]))
    self.assertEqual(int(state_partial.active_page_position[2]), length_partial)

  def test_release_pages(self):
    """Tests the release_pages method of PageManager (global state)."""
    page_group_id = 1
    initial_length = 10

    initial_state = self.pm.get_initial_page_state()

    # First, allocate some pages.
    allocated_state = self.pm.update_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=initial_length
    )
    pages_needed = (initial_length + self.tokens_per_page - 1) // self.tokens_per_page
    self.assertEqual(int(allocated_state.num_pages_used[page_group_id]), pages_needed)

    # Get allocated page indices for later verification (global map)
    page_map_group = allocated_state.page_map[page_group_id]
    num_used = allocated_state.num_pages_used[page_group_id]
    allocated_page_indices = page_map_group[:num_used]
    # Ensure we captured some pages
    self.assertEqual(len(allocated_page_indices), pages_needed)
    initial_status_sum = jnp.sum(allocated_state.page_status)

    # Now, release the pages for this group
    released_state = self.pm.release_pages(page_state=allocated_state, page_group_id=page_group_id)

    # Verify that state for the specific group is reset.
    self.assertEqual(int(released_state.sequence_lengths[page_group_id]), 0)
    self.assertEqual(int(released_state.num_pages_used[page_group_id]), 0)
    self.assertFalse(bool(released_state.has_active_page[page_group_id]))

    # Verify that the previously allocated pages are now marked as free (status 0) in the global status.
    final_status_sum = jnp.sum(released_state.page_status)
    self.assertEqual(
        final_status_sum, initial_status_sum - pages_needed, "Global page status sum should decrease by released amount"
    )
    for page_idx in allocated_page_indices:
      self.assertEqual(int(released_state.page_status[page_idx]), 0, f"Page {page_idx} should be marked as free")

  def test_update_decode_pages(self):
    """Tests the update_decode_pages function (global state)."""
    initial_state = self.pm.get_initial_page_state()

    # Test case 1: Decode step with no active groups - state should not change.
    decode_state_inactive = self.pm.update_decode_pages(page_state=initial_state)
    self._assert_page_state_equal(initial_state, decode_state_inactive, "Decode with no active groups failed")

    # Test case 2: Decode step causing page boundary crossing.
    # Prefill exactly one page full.
    prefill_state_boundary = self.pm.update_prefill_pages(
        page_state=initial_state, page_group_id=0, true_length=self.tokens_per_page
    )
    self.assertEqual(int(prefill_state_boundary.num_pages_used[0]), 1)
    self.assertEqual(int(prefill_state_boundary.sequence_lengths[0]), self.tokens_per_page)
    self.assertEqual(int(prefill_state_boundary.active_page_position[0]), 0)

    # Perform decode step - should allocate a new page.
    decode_state_boundary = self.pm.update_decode_pages(page_state=prefill_state_boundary)

    self.assertEqual(
        int(decode_state_boundary.sequence_lengths[0]),
        self.tokens_per_page + 1,
        "Seq length incorrect after boundary cross",
    )
    self.assertEqual(int(decode_state_boundary.num_pages_used[0]), 2, "Page count incorrect after boundary cross")
    # Active page should be the newly allocated one (different from the first)
    first_page = prefill_state_boundary.page_map[0, 0]
    second_page = decode_state_boundary.page_map[0, 1]
    self.assertNotEqual(first_page, second_page, "Second page should have a different index")
    self.assertEqual(int(decode_state_boundary.active_page[0]), second_page, "Active page should be the new page index")
    # Position in the new page should be 0
    self.assertEqual(int(decode_state_boundary.active_page_position[0]), 0, "Position incorrect after boundary cross")

    # Test case 3: Decode step *not* crossing a page boundary.
    prefill_state_partial = self.pm.update_prefill_pages(page_state=initial_state, page_group_id=1, true_length=5)
    self.assertEqual(int(prefill_state_partial.num_pages_used[1]), 1)
    self.assertEqual(int(prefill_state_partial.sequence_lengths[1]), 5)
    self.assertEqual(int(prefill_state_partial.active_page_position[1]), 5)
    first_page_partial = prefill_state_partial.active_page[1]

    # Perform decode step
    decode_state_partial = self.pm.update_decode_pages(page_state=prefill_state_partial)

    self.assertEqual(int(decode_state_partial.sequence_lengths[1]), 6, "Seq length incorrect (no boundary cross)")
    # Should still only use one page
    self.assertEqual(int(decode_state_partial.num_pages_used[1]), 1, "Page count incorrect (no boundary cross)")
    # Active page should be the same
    self.assertEqual(
        int(decode_state_partial.active_page[1]), first_page_partial, "Active page incorrect (no boundary cross)"
    )
    # Position should advance
    self.assertEqual(int(decode_state_partial.active_page_position[1]), 5, "Position incorrect (no boundary cross)")

  def test_state_consistency(self):
    """Checks internal consistency of PageState after operations (global state)."""
    initial_state = self.pm.get_initial_page_state()

    # +1 is necessary due to page 0 always being "active"
    self.assertEqual(int(jnp.sum(initial_state.page_status)), 1)
    self.assertEqual(int(jnp.sum(initial_state.num_pages_used)), 0)

    page_group_id = 0
    true_length = 12

    # Perform a prefill.
    state_alloc = self.pm.update_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=true_length
    )

    # Verify that the total number of allocated pages (status 1) equals the total number of mapped pages used.
    total_allocated_pages = int(jnp.sum(state_alloc.page_status))
    total_mapped_pages = int(jnp.sum(state_alloc.num_pages_used))

    # +1 is necessary due to page 0 always being "active"
    self.assertEqual(total_allocated_pages, total_mapped_pages + 1, "Sum of page_status != sum of num_pages_used")

    # Check uniqueness of allocated pages within the group
    num_used = state_alloc.num_pages_used[page_group_id]
    page_map_group = state_alloc.page_map[page_group_id]
    used_page_indices = page_map_group[:num_used]
    if num_used > 0:
      self.assertEqual(len(jnp.unique(used_page_indices)), num_used, "Allocated pages for the group are not unique")

    # Perform a decode step
    state_decode = self.pm.update_decode_pages(page_state=state_alloc)
    total_allocated_pages_d = int(jnp.sum(state_decode.page_status))
    total_mapped_pages_d = int(jnp.sum(state_decode.num_pages_used))
    self.assertEqual(total_allocated_pages_d, total_mapped_pages_d + 1, "Consistency failed after decode step")

    # Release pages
    state_release = self.pm.release_pages(page_state=state_decode, page_group_id=page_group_id)
    total_allocated_pages_r = int(jnp.sum(state_release.page_status))
    total_mapped_pages_r = int(jnp.sum(state_release.num_pages_used))  # Should be less now
    self.assertEqual(total_allocated_pages_r, total_mapped_pages_r + 1, "Consistency failed after release step")
    # Check specific group is cleared
    self.assertEqual(int(state_release.num_pages_used[page_group_id]), 0)

  def test_page_group_id_boundaries(self):
    """Tests allocation at the boundaries: max page_group_id and max length (global state)."""
    initial_state = self.pm.get_initial_page_state()

    # Test allocation at the maximum valid page_group_id.
    max_group_idx = self.max_page_groups - 1
    state_max_group = self.pm.update_prefill_pages(page_state=initial_state, page_group_id=max_group_idx, true_length=1)
    self.assertEqual(int(state_max_group.sequence_lengths[max_group_idx]), 1)
    self.assertEqual(int(state_max_group.num_pages_used[max_group_idx]), 1)
    self.assertTrue(bool(state_max_group.has_active_page[max_group_idx]))

    # Test allocation at the maximum allowed length (limited by max_pages_per_group).
    page_group_id = 0
    # Calculate the actual max length that fits within max_pages_per_group
    # (Can't exceed max_target_length either, but validation handles that)
    max_possible_length = self.tokens_per_page * self.max_pages_per_group
    # Clamp to max_target_length specified in config
    effective_max_length = min(max_possible_length, self.max_target_length)

    state_max_len = self.pm.update_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=effective_max_length
    )
    expected_pages = (effective_max_length + self.tokens_per_page - 1) // self.tokens_per_page

    self.assertEqual(int(state_max_len.sequence_lengths[page_group_id]), effective_max_length)
    self.assertEqual(int(state_max_len.num_pages_used[page_group_id]), expected_pages)
    self.assertTrue(bool(state_max_len.has_active_page[page_group_id]))
    # Should not exceed max_pages_per_group
    self.assertLessEqual(int(state_max_len.num_pages_used[page_group_id]), self.max_pages_per_group)

  def test_multiple_group_interactions(self):
    """Tests allocation and deallocation across multiple page groups (global state)."""
    initial_state = self.pm.get_initial_page_state()
    current_state = initial_state
    num_test_groups = min(3, self.max_page_groups)

    for i in range(num_test_groups):
      page_group_id = i
      true_length = 10 + page_group_id * self.tokens_per_page

      current_state = self.pm.update_prefill_pages(
          page_state=current_state, page_group_id=page_group_id, true_length=true_length
      )

      self.assertEqual(int(current_state.sequence_lengths[page_group_id]), true_length)
      pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
      self.assertEqual(int(current_state.num_pages_used[page_group_id]), pages_needed)
      self.assertTrue(bool(current_state.has_active_page[page_group_id]))

      # +1 is necessary due to page 0 always being "active"
      self.assertEqual(int(jnp.sum(current_state.page_status)), int(jnp.sum(current_state.num_pages_used) + 1))

    # Test decode steps, affecting all active groups
    state_after_alloc = current_state
    decode_state = self.pm.update_decode_pages(page_state=state_after_alloc)

    for i in range(num_test_groups):
      page_group_id = i
      original_length = 10 + page_group_id * self.tokens_per_page
      # Check that sequence length was incremented for each active group
      self.assertEqual(
          int(decode_state.sequence_lengths[page_group_id]),
          original_length + 1,
          f"Group {page_group_id}: incorrect sequence length after decode step",
      )
      # Page count might or might not increase depending on boundary crossing
      new_pages_req = (original_length + 1 + self.tokens_per_page - 1) // self.tokens_per_page
      self.assertEqual(int(decode_state.num_pages_used[page_group_id]), new_pages_req)

    # Release one group and check others are unaffected
    group_to_release = 1
    if group_to_release < num_test_groups:
      released_one_state = self.pm.release_pages(page_state=decode_state, page_group_id=group_to_release)

      # Check released group is cleared
      self.assertEqual(int(released_one_state.num_pages_used[group_to_release]), 0)
      self.assertFalse(bool(released_one_state.has_active_page[group_to_release]))

      # Check other groups remain active and unchanged
      for i in range(num_test_groups):
        if i != group_to_release:
          self.assertEqual(int(released_one_state.sequence_lengths[i]), int(decode_state.sequence_lengths[i]))
          self.assertEqual(int(released_one_state.num_pages_used[i]), int(decode_state.num_pages_used[i]))
          self.assertTrue(bool(released_one_state.has_active_page[i]))

  def test_invalid_page_group_id(self):
    """Tests that operations with invalid page_group_id raise ValueError."""
    initial_state = self.pm.get_initial_page_state()

    # Test update_prefill_pages with invalid IDs
    with self.assertRaises(ValueError, msg="Prefill with ID -1 should raise ValueError"):
      self.pm.update_prefill_pages(page_state=initial_state, page_group_id=-1, true_length=10)

    # Check upper bound (index max_page_groups is out of bounds [0, max_page_groups-1])
    with self.assertRaises(ValueError, msg="Prefill with ID >= max should raise ValueError"):
      self.pm.update_prefill_pages(page_state=initial_state, page_group_id=self.max_page_groups, true_length=10)

    # First allocate some pages to a valid group for testing release
    # Use group 0 if max_page_groups is at least 1
    if self.max_page_groups > 0:
      valid_group_id = 0
      valid_state = self.pm.update_prefill_pages(page_state=initial_state, page_group_id=valid_group_id, true_length=10)

      with self.assertRaises(ValueError, msg="Release with ID -1 should raise ValueError"):
        self.pm.release_pages(page_state=valid_state, page_group_id=-1)

      with self.assertRaises(ValueError, msg="Release with ID >= max should raise ValueError"):
        self.pm.release_pages(page_state=valid_state, page_group_id=self.max_page_groups)
    else:
      # Handle the case where max_page_groups is 0, maybe skip the release part or raise config error
      print("Warning: Skipping release part of test_invalid_page_group_id as max_page_groups is 0.")

  def test_invalid_length(self):
    """Tests that prefill with invalid length raises ValueError."""
    initial_state = self.pm.get_initial_page_state()

    # Try prefill with negative length
    with self.assertRaises(ValueError, msg="Prefill with length -1 should raise ValueError"):
      self.pm.update_prefill_pages(page_state=initial_state, page_group_id=0, true_length=-1)

    # Try prefill with zero length (should also raise ValueError based on PageManager code)
    with self.assertRaises(ValueError, msg="Prefill with length 0 should raise ValueError"):
      self.pm.update_prefill_pages(page_state=initial_state, page_group_id=0, true_length=0)

    # Try prefill with length exceeding max_target_length
    with self.assertRaises(ValueError, msg="Prefill with length > max should raise ValueError"):
      self.pm.update_prefill_pages(page_state=initial_state, page_group_id=0, true_length=self.max_target_length + 1)

  def test_page_boundary_crossing(self):
    """Tests the specific case of crossing page boundaries during decoding (global state)."""
    initial_state = self.pm.get_initial_page_state()
    page_group_id = 0

    # Set up a sequence exactly at a page boundary
    current_state = self.pm.update_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=self.tokens_per_page
    )

    # Record the initial number of pages used for this group
    initial_pages_used = int(current_state.num_pages_used[page_group_id])
    self.assertEqual(initial_pages_used, 1)  # Should be exactly 1 page

    # Perform a decode step that should cross the boundary
    updated_state = self.pm.update_decode_pages(page_state=current_state)

    # Verify that a new page was allocated for this group
    self.assertEqual(
        int(updated_state.num_pages_used[page_group_id]),
        initial_pages_used + 1,
        "A new page should be allocated when crossing a page boundary",
    )
    # Check sequence length
    self.assertEqual(int(updated_state.sequence_lengths[page_group_id]), self.tokens_per_page + 1)

    # Check the current page position was reset for the new page
    self.assertEqual(
        int(updated_state.active_page_position[page_group_id]),
        0,
        "Current page position should be reset to 0 for the new page",
    )
    # Check active page index has updated
    self.assertNotEqual(int(current_state.active_page[page_group_id]), int(updated_state.active_page[page_group_id]))

  def test_repeated_allocation_deallocation(self):
    """Tests repeated allocation and deallocation to ensure stability (global state)."""
    initial_state = self.pm.get_initial_page_state()
    current_state = initial_state
    num_test_groups = min(3, self.max_page_groups)

    # Perform a series of allocations and deallocations
    for i in range(3):  # Repeat multiple times
      print(f"Repetition {i+1}")
      # Allocate to multiple groups
      for j in range(num_test_groups):
        page_group_id = j
        length = (page_group_id + 1) * 3 + i  # Vary lengths slightly each repetition
        print(f"  Allocating group {page_group_id} with length {length}")
        current_state = self.pm.update_prefill_pages(
            page_state=current_state, page_group_id=page_group_id, true_length=length
        )
        # Verify allocation for this group
        self.assertEqual(int(current_state.sequence_lengths[page_group_id]), length)
        self.assertTrue(bool(current_state.has_active_page[page_group_id]))

      # Perform some decode steps
      for k in range(2):
        print(f"  Decode step {k+1}")
        current_state = self.pm.update_decode_pages(page_state=current_state)

      # Release all groups used in this iteration
      for j in range(num_test_groups):
        page_group_id = j
        print(f"  Releasing group {page_group_id}")
        current_state = self.pm.release_pages(page_state=current_state, page_group_id=page_group_id)
        # Verify release for this group
        self.assertEqual(int(current_state.sequence_lengths[page_group_id]), 0)
        self.assertEqual(int(current_state.num_pages_used[page_group_id]), 0)
        self.assertFalse(bool(current_state.has_active_page[page_group_id]))

      # Check global state consistency after full release cycle
      self.assertEqual(int(jnp.sum(current_state.num_pages_used)), 0, f"Rep {i}: Not all groups released correctly")

      # 1 is necessary due to page 0 always being "active"
      self.assertEqual(int(jnp.sum(current_state.page_status)), 1, f"Rep {i}: Not all pages freed correctly")

    # Final check: Verify final state is clean (all pages should be free)
    # Offset necessary due to error with page 0
    self.assertTrue(
        jnp.all(current_state.page_status[1:] == 0), "After repeated allocation/deallocation, all pages should be free"
    )
    self.assertTrue(
        jnp.all(current_state.num_pages_used == 0), "After repeated cycles, num_pages_used should be all zero"
    )


if __name__ == "__main__":
  unittest.main()
