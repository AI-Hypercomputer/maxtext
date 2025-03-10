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
from inference.page_manager import PageManager, PageState
import pyconfig
import max_utils


class TestPageManager(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_pages = 128
    self.tokens_per_page = 8
    self.max_page_groups = 4
    self.max_prefill_predict_length = 128
    self.max_target_length = 256
    self.max_pages_per_group = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    self.num_layers = 2

    config = pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_prefill_predict_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        pagedattn_num_pages=self.num_pages,
        pagedattn_tokens_per_page=self.tokens_per_page,
        pagedattn_max_page_groups=self.max_page_groups,
        pagedattn_max_pages_per_group=self.max_pages_per_group,
        base_num_decoder_layers=self.num_layers,
    )
    self.config = config

    # Initialize JAX distributed system and mesh
    devices_array = max_utils.create_device_mesh(self.config)
    self.mesh = jax.sharding.Mesh(devices_array, self.config.mesh_axes)

    self.key = jax.random.PRNGKey(0)
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
    """Tests the initialization of PageState: checks shapes and initial values."""
    state = self.pm.get_initial_page_state()
    self._validate_state_shapes(state)  # Verify all array shapes
    self.assertTrue(jnp.all(state.page_status == 0))  # All pages should be initially free
    self.assertTrue(jnp.all(state.page_map == -1))  # No pages should be mapped initially

  def test_jit_compatibility(self):
    """Tests that PageManager methods are JIT-compatible."""
    concrete_request_id = 0
    concrete_true_length = 12

    initial_state = self.pm.get_initial_page_state()

    @jax.jit
    def jitted_prefill(page_state, request_id, true_length):
      return self.pm.update_prefill_pages(
          page_state=page_state,
          request_id=request_id,
          true_length=true_length,
      )

    updated_state = jitted_prefill(
        page_state=initial_state,
        request_id=concrete_request_id,
        true_length=concrete_true_length,
    )
    self.assertEqual(int(updated_state.sequence_lengths[0, concrete_request_id]), concrete_true_length)

    @jax.jit
    def jitted_autoregressive(page_state):
      return self.pm.update_decode_pages(page_state=page_state)

    ar_state = jitted_autoregressive(page_state=updated_state)
    self.assertEqual(int(ar_state.sequence_lengths[0, concrete_request_id]), concrete_true_length + 1)

  def test_reserve_prefill_request(self):
    """Tests update_prefill_pages: standard case with sufficient space."""
    request_id = 0
    true_length = 12

    initial_state = self.pm.get_initial_page_state()

    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=request_id, true_length=true_length
    )

    # Calculate the expected number of pages needed
    pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    
    # Check both layers have correct state
    for layer_id in range(self.num_layers):
      self.assertEqual(int(updated_state.sequence_lengths[layer_id, request_id]), true_length)
      self.assertEqual(int(updated_state.num_pages_used[layer_id, request_id]), pages_needed)

      # Get the indices of allocated pages from the page_map
      page_map = updated_state.page_map[layer_id, request_id]
      used_page_indices = page_map[page_map >= 0]
      self.assertEqual(len(used_page_indices), pages_needed)  # Check the number of allocated pages
      self.assertEqual(len(jnp.unique(used_page_indices)), pages_needed)  # Ensure no duplicates

      # Verify that the allocated pages are marked as used in page_status
      for page_idx in used_page_indices:
        self.assertEqual(int(updated_state.page_status[layer_id, page_idx]), 1)

  def test_reserve_prefill_no_space(self):
    """Tests update_prefill_pages when there's no space available in the layer."""
    request_id = 0
    true_length = 12
    
    # Initialize and then set *all* pages in all layers to be used.
    initial_state = self.pm.get_initial_page_state()
    initial_state = initial_state.replace(
        page_status=jnp.ones((self.num_layers, self.num_pages), dtype=jnp.int32)
    )

    # Call update_prefill_pages; since no pages are free, the state should be unchanged.
    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=request_id, true_length=true_length
    )

    # Verify the state is unchanged
    self.assertTrue(jnp.all(updated_state.sequence_lengths == 0))  # No sequences should be allocated
    self.assertTrue(jnp.all(updated_state.num_pages_used == 0))    # No pages should be used

  def test_reserve_prefill_edge_cases(self):
    """Tests update_prefill_pages with edge cases: zero length, exact page multiple, partial page."""
    initial_state = self.pm.get_initial_page_state()

    # Test case 1: Zero length - no pages should be allocated.
    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=0, true_length=0
    )
    for layer_id in range(self.num_layers):
      self.assertEqual(int(updated_state.sequence_lengths[layer_id, 0]), 0)
      self.assertEqual(int(updated_state.num_pages_used[layer_id, 0]), 0)
      self.assertEqual(int(updated_state.current_page[layer_id, 0]), -1)
      self.assertEqual(int(updated_state.current_page_position[layer_id, 0]), 0)

    # Test case 2: Exact page multiple - correct number of pages should be allocated.
    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=1, true_length=self.tokens_per_page * 2
    )
    for layer_id in range(self.num_layers):
      self.assertEqual(int(updated_state.sequence_lengths[layer_id, 1]), self.tokens_per_page * 2)
      self.assertEqual(int(updated_state.num_pages_used[layer_id, 1]), 2)

    # Test case 3: Partial page - only one page should be allocated.
    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=3, true_length=5
    )
    for layer_id in range(self.num_layers):
      self.assertEqual(int(updated_state.sequence_lengths[layer_id, 3]), 5)
      self.assertEqual(int(updated_state.num_pages_used[layer_id, 3]), 1)

  def test_release_pages(self):
    """Tests the release_pages method of PageManager."""
    request_id = 1
    initial_length = 10
    
    initial_state = self.pm.get_initial_page_state()

    # First, allocate some pages using update_prefill_pages.
    updated_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=request_id, true_length=initial_length
    )

    # Get allocated page indices for later verification
    allocated_pages = {}
    for layer_id in range(self.num_layers):
        page_map = updated_state.page_map[layer_id, request_id]
        allocated_pages[layer_id] = page_map[page_map >= 0]

    # Now, release the pages
    released_state = self.pm.release_pages(page_state=updated_state, request_id=request_id)

    # Verify that sequence length and num_pages_used are reset to 0.
    for layer_id in range(self.num_layers):
      self.assertEqual(int(released_state.sequence_lengths[layer_id, request_id]), 0)
      self.assertEqual(int(released_state.num_pages_used[layer_id, request_id]), 0)

      # Verify that the previously allocated pages are now marked as free (status 0).
      for page_idx in allocated_pages[layer_id]:
        self.assertEqual(int(released_state.page_status[layer_id, page_idx]), 0)

  def test_reserve_decode_step_pages(self):
    """Tests the update_decode_pages function."""
    initial_state = self.pm.get_initial_page_state()

    # Test case 1: Initial autoregressive step - no pages allocated yet.
    updated_state = self.pm.update_decode_pages(page_state=initial_state)
    self.assertEqual(jnp.sum(updated_state.sequence_lengths), 0)  # No allocation should happen

    # Test case 2: After prefill - allocate one page, then another on the decode step.
    new_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=0, true_length=self.tokens_per_page
    )

    updated_state = self.pm.update_decode_pages(page_state=new_state)

    for layer_id in range(self.num_layers):
      self.assertEqual(
          int(updated_state.sequence_lengths[layer_id, 0]), self.tokens_per_page + 1
      )  # Sequence length should be incremented
      self.assertEqual(int(updated_state.num_pages_used[layer_id, 0]), 2)  # Another page allocated
      self.assertGreater(int(updated_state.current_page[layer_id, 0]), -1)  # Current page should be valid

    # Test case 3: Partial page allocation - pages allocated only when needed.
    partial_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=2, true_length=5
    )
    partial_updated_state = self.pm.update_decode_pages(page_state=partial_state)
    
    for layer_id in range(self.num_layers):
      self.assertEqual(int(partial_updated_state.sequence_lengths[layer_id, 2]), 6)  # Sequence length incremented
      self.assertEqual(int(partial_updated_state.num_pages_used[layer_id, 2]), 1)  # Still only one page used

  def test_state_consistency(self):
    """Checks internal consistency of PageState after allocation operations."""
    initial_state = self.pm.get_initial_page_state()
    # Initially, no pages should be allocated or mapped.
    self.assertEqual(int(jnp.sum(initial_state.page_status)), 0)
    self.assertEqual(int(jnp.sum(initial_state.page_map != -1)), 0)

    request_id = 0
    true_length = 12
    
    # Perform a prefill.
    state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=request_id, true_length=true_length
    )

    # Verify that the number of allocated pages (status 1) equals the number of mapped pages.
    for layer_id in range(self.num_layers):
      allocated_pages = int(jnp.sum(state.page_status[layer_id]))
      mapped_pages = int(jnp.sum(state.page_map[layer_id] != -1))
      self.assertEqual(allocated_pages, mapped_pages)

      # Check for duplicate page assignments within the layer.
      page_assignments = state.page_map[layer_id][state.page_map[layer_id] != -1].flatten()
      self.assertEqual(len(page_assignments), len(jnp.unique(page_assignments)))

  def test_request_id_boundaries(self):
    """Tests allocation at the boundaries: max request_id and max length."""
    initial_state = self.pm.get_initial_page_state()

    # Test allocation at the maximum request_id.
    state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=self.max_page_groups - 1, true_length=1
    )
    for layer_id in range(self.num_layers):
      self.assertEqual(int(state.sequence_lengths[layer_id, self.max_page_groups - 1]), 1)

    # Test allocation at the maximum length (within max_pages_per_group).
    request_id = 0
    max_length = self.tokens_per_page * self.max_pages_per_group
    state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=request_id, true_length=max_length
    )
    for layer_id in range(self.num_layers):
      self.assertEqual(int(state.sequence_lengths[layer_id, request_id]), max_length)

  def test_multi_layer_consistency(self):
    """Tests allocation and deallocation across multiple layers."""
    initial_state = self.pm.get_initial_page_state()

    # Allocate pages in different requests with different lengths.
    for request_id in range(self.max_page_groups):
      true_length = 10 + request_id * 5
      
      updated_state = self.pm.update_prefill_pages(
          page_state=initial_state, request_id=request_id, true_length=true_length
      )

      for layer_id in range(self.num_layers):
        self.assertEqual(
            updated_state.sequence_lengths[layer_id, request_id],
            true_length,
            f"Layer {layer_id}: incorrect sequence length",
        )

        pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
        self.assertEqual(
            updated_state.num_pages_used[layer_id, request_id],
            pages_needed,
            f"Layer {layer_id}: incorrect page count",
        )

        page_map = updated_state.page_map[layer_id, request_id]
        used_page_indices = page_map[page_map >= 0]
        for page_idx in used_page_indices:
          self.assertEqual(
              int(updated_state.page_status[layer_id, page_idx]),
              1,
              f"Layer {layer_id}, Page {page_idx} not marked as used",
          )
      initial_state = updated_state  # Carry over state to the next iteration

    # Test autoregressive steps for each layer, building on the previous state.
    updated_state = self.pm.update_decode_pages(page_state=initial_state)
    
    for request_id in range(self.max_page_groups):
      for layer_id in range(self.num_layers):
        # Check that sequence length was incremented
        self.assertEqual(
            updated_state.sequence_lengths[layer_id, request_id],
            10 + request_id * 5 + 1,  # Length should increment by 1
            f"Layer {layer_id}: incorrect sequence length after autoregressive step",
        )

  def test_invalid_request_id(self):
    """Tests that operations with invalid request_id have no effect."""
    initial_state = self.pm.get_initial_page_state()
    
    # Try operations with invalid request_id
    updated_state = self.pm.update_prefill_pages(page_state=initial_state, request_id=-1, true_length=10)
    self._assert_page_state_equal(initial_state, updated_state)
    
    updated_state = self.pm.update_prefill_pages(page_state=initial_state, request_id=self.max_page_groups, true_length=10)
    self._assert_page_state_equal(initial_state, updated_state)
    
    # First allocate some pages to a valid request
    valid_state = self.pm.update_prefill_pages(page_state=initial_state, request_id=0, true_length=10)
    
    # Then try to release an invalid request
    released_state = self.pm.release_pages(page_state=valid_state, request_id=-1)
    self._assert_page_state_equal(valid_state, released_state)

  def test_invalid_length(self):
    """Tests that operations with invalid length have no effect."""
    initial_state = self.pm.get_initial_page_state()
    
    # Try operations with invalid length
    updated_state = self.pm.update_prefill_pages(page_state=initial_state, request_id=0, true_length=-1)
    self._assert_page_state_equal(initial_state, updated_state)
    
    updated_state = self.pm.update_prefill_pages(page_state=initial_state, request_id=0, true_length=self.max_target_length + 1)
    self._assert_page_state_equal(initial_state, updated_state)

  def test_page_boundary_crossing(self):
    """Tests the specific case of crossing page boundaries during decoding."""
    initial_state = self.pm.get_initial_page_state()

    # Set up a sequence exactly at a page boundary
    current_state = self.pm.update_prefill_pages(
        page_state=initial_state, request_id=0, true_length=self.tokens_per_page
    )

    # Record the initial number of pages used
    initial_pages_used = {}
    for layer_id in range(self.num_layers):
      initial_pages_used[layer_id] = int(current_state.num_pages_used[layer_id, 0])

    # Perform a decode step that should cross the boundary
    updated_state = self.pm.update_decode_pages(page_state=current_state)

    # Verify that a new page was allocated in each layer
    for layer_id in range(self.num_layers):
      self.assertEqual(
          int(updated_state.num_pages_used[layer_id, 0]),
          initial_pages_used[layer_id] + 1,
          f"Layer {layer_id}: A new page should be allocated when crossing a page boundary",
      )
      self.assertEqual(int(updated_state.sequence_lengths[layer_id, 0]), self.tokens_per_page + 1)

      # Check the current page position was reset for the new page
      self.assertEqual(
          int(updated_state.current_page_position[layer_id, 0]), 0, 
          f"Layer {layer_id}: Current page position should be reset to 0 for the new page"
      )

  def test_repeated_allocation_deallocation(self):
    """Tests repeated allocation and deallocation to ensure stability."""
    initial_state = self.pm.get_initial_page_state()
    current_state = initial_state

    # Perform a series of allocations and deallocations
    for i in range(3):  # Repeat multiple times
      # Allocate to multiple requests
      for request_id in range(min(3, self.max_page_groups)):
        length = (request_id + 1) * 5  # Different lengths for each request
        current_state = self.pm.update_prefill_pages(
            page_state=current_state, request_id=request_id, true_length=length
        )

        # Verify allocation
        for layer_id in range(self.num_layers):
          self.assertEqual(int(current_state.sequence_lengths[layer_id, request_id]), length)

      # Perform some autoregressive steps
      for _ in range(2):
        current_state = self.pm.update_decode_pages(page_state=current_state)

      # Release all requests
      for request_id in range(min(3, self.max_page_groups)):
        current_state = self.pm.release_pages(page_state=current_state, request_id=request_id)

        # Verify release
        for layer_id in range(self.num_layers):
          self.assertEqual(int(current_state.sequence_lengths[layer_id, request_id]), 0)
          self.assertEqual(int(current_state.num_pages_used[layer_id, request_id]), 0)

    # Verify final state is clean (all pages should be free)
    self.assertTrue(
        jnp.all(current_state.page_status == 0), 
        "After repeated allocation/deallocation, all pages should be free"
    )


if __name__ == "__main__":
  unittest.main()