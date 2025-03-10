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

""" Tests for Page Manager."""
import sys
import unittest
import jax
import jax.numpy as jnp
import max_utils
from inference.page_manager import (
    PageManager,
    validate_page_group_id,
    validate_sequence_length,
    initialize_page_state,
    _find_next_free_page_index,
    release_page_group,
    reserve_prefill_pages_for_group,
    reserve_decode_pages_for_group,
)
import pyconfig


class TestPageManager(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_pages = 128
    self.tokens_per_page = 8
    self.max_page_groups = 4
    self.max_target_length = 256
    self.max_pages_per_group = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    self.num_layers = 2

    config = pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        pagedattn_num_pages=self.num_pages,
        pagedattn_tokens_per_page=self.tokens_per_page,
        pagedattn_max_page_groups=self.max_page_groups,
        max_target_length=self.max_target_length,
        pagedattn_max_pages_per_group=self.max_pages_per_group,
        base_num_decoder_layers=self.num_layers,
    )
    self.config = config
    self.key = jax.random.PRNGKey(0)

    devices_array = max_utils.create_device_mesh(self.config)
    self.mesh = jax.sharding.Mesh(devices_array, self.config.mesh_axes)
    self.pm = PageManager(config=self.config)

  def _validate_state_shapes(self, state):
    """Helper function to assert that all PageState arrays have correct shapes."""
    expected_shapes = {
        "page_status": (self.num_layers, self.num_pages),
        "page_map": (self.num_layers, self.max_page_groups, self.max_pages_per_group),
        "sequence_lengths": (
            self.num_layers,
            self.max_page_groups,
        ),
        "num_pages_used": (self.num_layers, self.max_page_groups),
        "current_page": (self.num_layers, self.max_page_groups),
        "current_page_position": (self.num_layers, self.max_page_groups),
    }

    for field, expected in expected_shapes.items():
      self.assertEqual(
          getattr(state, field).shape,
          expected,
          f"Shape mismatch for {field}: expected {expected}, got {getattr(state, field).shape}",
      )

  def _assert_page_state_equal(self, state1, state2, msg=None):
    """Helper function to compare two PageState objects field by field."""
    for field in state1.__dataclass_fields__:
      self.assertTrue(
          jnp.array_equal(getattr(state1, field), getattr(state2, field)), msg=f"{msg}: Field '{field}' mismatch"
      )

  def test_initialization(self):
    """Tests the initialize_page_state function: checks shapes and initial values."""
    state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )
    self._validate_state_shapes(state)  # Verify all array shapes
    self.assertTrue(jnp.all(state.page_status == 0))  # All pages should be initially free
    self.assertTrue(jnp.all(state.page_map == -1))  # No pages should be mapped initially

  def test_find_next_free_page(self):
    """Tests the find_next_free_page_index function with various page statuses."""

    # Case 1: Some pages are free; should return the index of the *first* free page.
    page_status = jnp.array([1, 1, 0, 0, 1, 0])
    next_free = _find_next_free_page_index(page_status=page_status)
    self.assertEqual(int(next_free), 2)

    # Case 2: No pages are free; should return -1.
    page_status = jnp.array([1, 1, 1, 1, 1, 1])
    next_free = _find_next_free_page_index(page_status=page_status)
    self.assertEqual(int(next_free), -1)

    # Case 3: All pages are free; should return 0 (the first page).
    page_status = jnp.array([0, 0, 0, 0, 0, 0])
    next_free = _find_next_free_page_index(page_status=page_status)
    self.assertEqual(int(next_free), 0)

  def test_jit_compatibility(self):
    """Tests that reserve_prefill_pages and reserve_decode_pages are JIT-compatible."""
    layer_id = 0
    page_group_id = 0
    true_length = 12

    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

    @jax.jit
    def jitted_prefill(page_state, page_group_id, true_length, layer_id):
      return reserve_prefill_pages_for_group(  # Call directly
          page_state=page_state,
          page_group_id=page_group_id,
          true_length=true_length,
          layer_id=layer_id,
          tokens_per_page=self.tokens_per_page,  # Pass required args
          max_pages_per_group=self.max_pages_per_group,
      )

    updated_state = jitted_prefill(
        page_state=initial_state,
        page_group_id=page_group_id,
        true_length=true_length,
        layer_id=layer_id,
    )
    self.assertEqual(int(updated_state.sequence_lengths[layer_id, page_group_id]), true_length)

    @jax.jit
    def jitted_autoregressive(page_state, layer_id):
      return reserve_decode_pages_for_group(  # Call directly
          page_state=page_state,
          layer_id=layer_id,
          tokens_per_page=self.tokens_per_page,  # Pass required args
          max_page_groups=self.max_page_groups,
      )

    ar_state = jitted_autoregressive(page_state=updated_state, layer_id=layer_id)
    self.assertEqual(int(ar_state.sequence_lengths[layer_id, page_group_id]), true_length + 1)

  def test_validation_functions(self):
    """Tests the validate_page_group_id and validate_sequence_length functions."""
    # Test validate_page_group_id with valid and invalid IDs
    is_valid = validate_page_group_id(page_group_id=1, max_page_groups=self.max_page_groups)
    self.assertTrue(is_valid)  # Use .item() for Python boolean

    is_invalid = validate_page_group_id(page_group_id=-1, max_page_groups=self.max_page_groups)
    self.assertFalse(is_invalid)  # Use .item() for Python boolean

    # Test validate_sequence_length with valid and invalid lengths
    is_valid = validate_sequence_length(length=100, max_target_length=self.max_target_length)
    self.assertTrue(is_valid)  # Use .item() for Python boolean

    is_invalid = validate_sequence_length(length=-1, max_target_length=self.max_target_length)
    self.assertFalse(is_invalid)  # Use .item() for Python boolean

  def test_reserve_prefill_page_group(self):
    """Tests the reserve_prefill_pages function: standard case with sufficient space."""
    layer_id = 0
    page_group_id = 0
    true_length = 12

    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

    updated_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=true_length, layer_id=layer_id
    )

    # Calculate the expected number of pages needed
    pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    self.assertEqual(int(updated_state.sequence_lengths[layer_id, page_group_id]), true_length)
    self.assertEqual(int(updated_state.num_pages_used[layer_id, page_group_id]), pages_needed)

    # Get the indices of allocated pages from the page_map
    page_map = updated_state.page_map[layer_id, page_group_id]
    used_page_indices = page_map[page_map >= 0]
    self.assertEqual(len(used_page_indices), pages_needed)  # Check the number of allocated pages
    self.assertEqual(len(jnp.unique(used_page_indices)), pages_needed)  # Ensure no duplicates

    # Verify that the allocated pages are marked as used in page_status
    for page_idx in used_page_indices:
      self.assertEqual(int(updated_state.page_status[layer_id, page_idx]), 1)

  def test_reserve_prefill_no_space(self):
    """Tests reserve_prefill_pages when there's no space available in the layer."""
    layer_id = 0
    page_group_id = 0
    true_length = 12
    # Initialize and then set *all* pages in the specified layer to be used.
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )
    initial_state = initial_state.replace(
        page_status=initial_state.page_status.at[layer_id].set(jnp.ones(self.num_pages, dtype=jnp.int32))
    )

    # Call reserve_prefill_pages; since no pages are free, the state should be unchanged.
    updated_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=true_length, layer_id=layer_id
    )

    # Use jnp.array_equal within assertTrue for JAX array comparisons
    self._assert_page_state_equal(initial_state, updated_state)

  def test_reserve_prefill_edge_cases(self):
    """Tests reserve_prefill_pages with edge cases: zero length, exact page multiple, partial page."""
    layer_id = 1  # Use a different layer to avoid conflicts with other tests
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

    # Test case 1: Zero length - no pages should be allocated.
    updated_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=0, true_length=0, layer_id=layer_id
    )
    self.assertEqual(int(updated_state.sequence_lengths[layer_id, 0]), 0)
    self.assertEqual(int(updated_state.num_pages_used[layer_id, 0]), 0)
    self.assertEqual(int(updated_state.current_page[layer_id, 0]), -1)
    self.assertEqual(int(updated_state.current_page_position[layer_id, 0]), 0)

    # Test case 2: Exact page multiple - correct number of pages should be allocated.
    updated_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=1, true_length=self.tokens_per_page * 2, layer_id=layer_id
    )
    self.assertEqual(int(updated_state.sequence_lengths[layer_id, 1]), self.tokens_per_page * 2)
    self.assertEqual(int(updated_state.num_pages_used[layer_id, 1]), 2)

    # Test case 3: Partial page - only one page should be allocated.
    updated_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=3, true_length=5, layer_id=layer_id
    )
    self.assertEqual(int(updated_state.sequence_lengths[layer_id, 3]), 5)
    self.assertEqual(int(updated_state.num_pages_used[layer_id, 3]), 1)

  def test_release_pages(self):
    """Tests the release method of PageManager using its public API."""
    page_group_id = 1
    initial_length = 10
    layer_id = 0
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

    # First, allocate some pages using reserve_prefill_pages.
    updated_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=initial_length, layer_id=layer_id
    )

    # Extract allocated page indices from the page_map.
    page_map = updated_state.page_map[layer_id]
    page_group_map = page_map[page_group_id]
    allocated_pages = page_group_map[page_group_map >= 0]

    # Now, release the pages using the public release method.
    released_state = self.pm.release_pages(page_state=updated_state, page_group_id=page_group_id)

    # Verify that sequence length and num_pages_used are reset to 0.
    self.assertEqual(int(released_state.sequence_lengths[layer_id, page_group_id]), 0)
    self.assertEqual(int(released_state.num_pages_used[layer_id, page_group_id]), 0)

    # Verify that the previously allocated pages are now marked as free (status 0).
    for page_idx in allocated_pages:
      self.assertEqual(int(released_state.page_status[layer_id, page_idx]), 0)

  def test_reserve_decode_step_pages(self):
    """Tests the reserve_decode_pages function."""
    layer_id = 0
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

    # Test case 1: Initial autoregressive step - no pages allocated yet.
    updated_state = self.pm.reserve_decode_pages(page_state=initial_state, layer_id=layer_id)
    self.assertEqual(jnp.sum(updated_state.sequence_lengths[layer_id]), 0)  # No allocation

    # Test case 2: After prefill - allocate one page, then another on the decode step.
    new_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=0, true_length=self.tokens_per_page, layer_id=layer_id
    )  # Allocate one page

    updated_state = self.pm.reserve_decode_pages(page_state=new_state, layer_id=layer_id)

    self.assertEqual(
        int(updated_state.sequence_lengths[layer_id, 0]), self.tokens_per_page + 1
    )  # Sequence length should be incremented
    self.assertEqual(int(updated_state.num_pages_used[layer_id, 0]), 2)  # Another page allocated
    self.assertGreater(int(updated_state.current_page[layer_id, 0]), -1)  # Current page should be valid

    # Test case 3: Partial page allocation - pages allocated only when needed.
    new_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=2, true_length=5, layer_id=layer_id
    )  # Allocate one page
    updated_state = self.pm.reserve_decode_pages(page_state=new_state, layer_id=layer_id)
    self.assertEqual(int(updated_state.sequence_lengths[layer_id, 2]), 6)  # Sequence length incremented
    self.assertEqual(int(updated_state.num_pages_used[layer_id, 2]), 1)  # Still only one page used

  def test_state_consistency(self):
    """Checks internal consistency of PageState after allocation operations."""
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )
    # Initially, no pages should be allocated or mapped.
    self.assertEqual(int(jnp.sum(initial_state.page_status)), 0)
    self.assertEqual(int(jnp.sum(initial_state.page_map != -1)), 0)

    page_group_id = 0
    true_length = 12
    layer_id = 0
    # Perform a prefill.
    state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=true_length, layer_id=layer_id
    )

    # Verify that the number of allocated pages (status 1) equals the number of mapped pages.
    allocated_pages = int(jnp.sum(state.page_status))
    mapped_pages = int(jnp.sum(state.page_map != -1))
    self.assertEqual(allocated_pages, mapped_pages)

    # Check for duplicate page assignments within the layer.
    page_assignments = state.page_map[layer_id][state.page_map[layer_id] != -1].flatten()
    self.assertEqual(len(page_assignments), len(jnp.unique(page_assignments)))

  def test_page_group_boundaries(self):
    """Tests allocation at the boundaries: max page group ID and max length."""
    layer_id = 0
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

    # Test allocation at the maximum page group ID.
    state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=self.max_page_groups - 1, true_length=1, layer_id=layer_id
    )
    self.assertEqual(int(state.sequence_lengths[layer_id, self.max_page_groups - 1]), 1)

    # Test allocation at the maximum length (within max_pages_per_group).
    page_group_id = 0
    max_length = self.tokens_per_page * self.max_pages_per_group
    state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=max_length, layer_id=layer_id
    )
    self.assertEqual(int(state.sequence_lengths[layer_id, page_group_id]), max_length)

  def test_multi_layer_consistency(self):
    """Tests allocation and deallocation across multiple layers."""
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

    # Allocate pages in different layers with different lengths.
    for layer_id in range(self.num_layers):
      true_length = 10 + layer_id * 5
      page_group_id = layer_id % self.max_page_groups

      updated_state = self.pm.reserve_prefill_pages(
          page_state=initial_state, page_group_id=page_group_id, true_length=true_length, layer_id=layer_id
      )

      self.assertEqual(
          updated_state.sequence_lengths[layer_id, page_group_id],
          true_length,
          f"Layer {layer_id}: incorrect sequence length",
      )

      pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
      self.assertEqual(
          updated_state.num_pages_used[layer_id, page_group_id],
          pages_needed,
          f"Layer {layer_id}: incorrect page count",
      )

      page_map = updated_state.page_map[layer_id, page_group_id]
      used_page_indices = page_map[page_map >= 0]
      for page_idx in used_page_indices:
        self.assertEqual(
            int(updated_state.page_status[layer_id, page_idx]),
            1,
            f"Layer {layer_id}, Page {page_idx} not marked as used",
        )
      initial_state = updated_state  # Carry over state to the next iteration

    # Test autoregressive steps for each layer, building on the previous state.
    for layer_id in range(self.num_layers):
      page_group_id = layer_id % self.max_page_groups
      updated_state = self.pm.reserve_decode_pages(page_state=initial_state, layer_id=layer_id)

      self.assertEqual(
          updated_state.sequence_lengths[layer_id, page_group_id],
          10 + layer_id * 5 + 1,  # Length should increment by 1
          f"Layer {layer_id}: incorrect sequence length after autoregressive step",
      )

      # Check if a new page was allocated (only if a page boundary was crossed).
      if (10 + layer_id * 5) % self.tokens_per_page == 0:
        self.assertEqual(
            updated_state.num_pages_used[layer_id, page_group_id],
            (10 + layer_id * 5 + self.tokens_per_page - 1) // self.tokens_per_page + 1,
            f"Layer {layer_id} page not incremented",
        )
      initial_state = updated_state  # Carry over state

  def test_release_invalid_page_group(self):
    """Tests releasing an invalid page group ID (should have no effect)."""
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )
    released_state = self.pm.release_pages(page_state=initial_state, page_group_id=-1)
    # All arrays in the state should remain unchanged.
    self._assert_page_state_equal(initial_state, released_state)

  def test_prefill_max_pages_per_group(self):
    """Tests prefilling with length > max_pages_per_group (should cap allocation)."""
    initial_state = initialize_page_state(
        num_layers=self.num_layers,
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )
    layer_id = 0
    page_group_id = 0
    true_length = (self.max_pages_per_group + 1) * self.tokens_per_page

    updated_state = self.pm.reserve_prefill_pages(
        page_state=initial_state, page_group_id=page_group_id, true_length=true_length, layer_id=layer_id
    )
    # Allocation should be capped at max_pages_per_group.
    self.assertEqual(int(updated_state.num_pages_used[layer_id, page_group_id]), self.max_pages_per_group)

  def test_release_page_group_direct(self):
    """Directly tests the release_page_group function."""
    max_pages_per_group = 10  # Use a smaller value for easier setup
    initial_state = initialize_page_state(
        num_layers=2, num_pages=20, max_page_groups=5, max_pages_per_group=max_pages_per_group
    )

    # 1. Setup: Manually create a PageState with some allocated pages.
    page_group_to_release = 2
    layer_to_test = 1
    # Manually allocate pages 2, 5, and 8 in layer 1, group 2.
    initial_state = initial_state.replace(
        page_status=initial_state.page_status.at[layer_to_test, [2, 5, 8]].set(1),
        page_map=initial_state.page_map.at[layer_to_test, page_group_to_release, [0, 1, 2]].set(
            jnp.array([2, 5, 8], dtype=jnp.int32)
        ),  # Ensure int32
        sequence_lengths=initial_state.sequence_lengths.at[layer_to_test, page_group_to_release].set(24),  # Arbitrary length
        num_pages_used=initial_state.num_pages_used.at[layer_to_test, page_group_to_release].set(3),
        current_page=initial_state.current_page.at[layer_to_test, page_group_to_release].set(8),
        current_page_position=initial_state.current_page_position.at[layer_to_test, page_group_to_release].set(7),
    )

    # 2. Call release_page_group with the prepared state.
    released_state = release_page_group(
        page_state=initial_state,
        page_group_id=page_group_to_release,
        max_page_groups=5,  # Use the test's max_page_groups
    )

    # 3. Assertions: Verify the state after releasing.
    # Check the specific layer and group where we released pages.
    self.assertEqual(int(released_state.sequence_lengths[layer_to_test, page_group_to_release]), 0)
    self.assertEqual(int(released_state.num_pages_used[layer_to_test, page_group_to_release]), 0)
    self.assertEqual(int(released_state.current_page[layer_to_test, page_group_to_release]), -1)
    self.assertEqual(int(released_state.current_page_position[layer_to_test, page_group_to_release]), 0)
    self.assertEqual(int(released_state.page_status[layer_to_test, 2]), 0)  # Page 2 should be free
    self.assertEqual(int(released_state.page_status[layer_to_test, 5]), 0)  # Page 5 should be free
    self.assertEqual(int(released_state.page_status[layer_to_test, 8]), 0)  # Page 8 should be free
    self.assertTrue(
        jnp.array_equal(
            released_state.page_map[layer_to_test, page_group_to_release],
            jnp.full(max_pages_per_group, -1, dtype=jnp.int32),  # page_map should be reset
        )
    )

    # Check other layers (should be unaffected).
    for layer_id in range(self.num_layers):
      # Only check that the *released* group is cleared in *all* layers
      self.assertEqual(int(released_state.sequence_lengths[layer_id, page_group_to_release]), 0)
      self.assertEqual(int(released_state.num_pages_used[layer_id, page_group_to_release]), 0)
      self.assertTrue(
          jnp.array_equal(
              released_state.page_map[layer_id, page_group_to_release], jnp.full(max_pages_per_group, -1, dtype=jnp.int32)
          )
      )

    # 4. Test releasing an invalid page group ID (should not change the state).
    invalid_released_state = release_page_group(
        page_state=initial_state,
        page_group_id=-1,
        max_page_groups=5,
    )
    self._assert_page_state_equal(initial_state, invalid_released_state)

    # 5. Test releasing a group that has no allocated pages (should have no effect).
    empty_released_state = release_page_group(
        page_state=initial_state,
        page_group_id=3,
        max_page_groups=5,
    )
    self._assert_page_state_equal(initial_state, empty_released_state)

  def test_get_initial_page_state(self):
    """Tests the PageManager's get_initial_page_state method."""
    state = self.pm.get_initial_page_state()
    self._validate_state_shapes(state)
    self.assertTrue(jnp.all(state.page_status == 0))
    self.assertTrue(jnp.all(state.page_map == -1))


if __name__ == "__main__":
  unittest.main()
