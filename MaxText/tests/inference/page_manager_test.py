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

"""Tests for page_manager."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from MaxText.inference import page_manager


class PageManagerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'all_free_except_0',
          'page_status': jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
          'expected_index': 1,
      },
      {
          'testcase_name': 'fully_allocated',
          'page_status': jnp.array([1, 1, 1, 1, 1], dtype=jnp.int32),
          'expected_index': -1,
      },
      {
          'testcase_name': 'mixed_allocation',
          'page_status': jnp.array([1, 1, 0, 1, 0], dtype=jnp.int32),
          'expected_index': 2,
      },
      {
          'testcase_name': 'page_0_free_rest_used',
          'page_status': jnp.array([0, 1, 1, 1], dtype=jnp.int32),
          'expected_index': -1, # Searches page_status[1:]
      },
      {
          'testcase_name': 'page_0_used_page_1_free',
          'page_status': jnp.array([1, 0, 1, 1], dtype=jnp.int32),
          'expected_index': 1,
      },
      {
          'testcase_name': 'long_all_free_except_0',
          'page_status': jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
          'expected_index': 1,
      },
      {
          'testcase_name': 'last_page_free',
          'page_status': jnp.array([1, 1, 1, 1, 0], dtype=jnp.int32),
          'expected_index': 4,
      },
      {
          'testcase_name': 'only_page_0_no_others',
          'page_status': jnp.array([1], dtype=jnp.int32),
          'expected_index': -1,
      },
       {
          'testcase_name': 'two_pages_0_used_1_free',
          'page_status': jnp.array([1,0], dtype=jnp.int32),
          'expected_index': 1,
      }
  )
  def test_find_next_free_page_index(self, page_status, expected_index):
    next_free = page_manager._find_next_free_page_index(page_status)
    self.assertEqual(next_free, expected_index)

  def setUp(self):
    super().setUp()
    self.num_pages = 64 # Example value
    self.max_page_groups = 4 # Example value
    self.tokens_per_page = 16 # Static for _reserve_pages_for_group
    self.max_pages_per_group = 32 # Static for _reserve_pages_for_group and _release_pages_for_group

    self.initial_page_state = page_manager.initialize_page_state(
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group
    )

  def test_reserve_pages_successful_reservation(self):
    page_group_id = jnp.array(0, dtype=jnp.int32)
    num_pages_to_reserve = 5
    true_length = jnp.array(num_pages_to_reserve * self.tokens_per_page, dtype=jnp.int32)

    # _reserve_pages_for_group expects a state where the target group is "released"
    # For a fresh state, this is the initial_page_state.
    released_state = self.initial_page_state

    final_state = page_manager._reserve_pages_for_group(
        released_state,
        page_group_id,
        true_length,
        self.tokens_per_page, # static
        self.max_pages_per_group # static
    )

    self.assertEqual(final_state.num_pages_used[page_group_id], num_pages_to_reserve)
    self.assertEqual(final_state.sequence_lengths[page_group_id], true_length)

    # Page 0 is used by default, so reserved pages start from index 1
    expected_page_status = self.initial_page_state.page_status.at[1:num_pages_to_reserve+1].set(1)
    self.assertTrue(jnp.array_equal(final_state.page_status, expected_page_status))

    expected_page_map_row = jnp.arange(1, num_pages_to_reserve + 1, dtype=jnp.int32)
    self.assertTrue(jnp.array_equal(final_state.page_map[page_group_id, :num_pages_to_reserve], expected_page_map_row))
    self.assertTrue(jnp.all(final_state.page_map[page_group_id, num_pages_to_reserve:] == 0))

    self.assertEqual(final_state.active_page[page_group_id], expected_page_map_row[-1])
    self.assertTrue(final_state.has_active_page[page_group_id])
    self.assertEqual(final_state.active_page_position[page_group_id], 0) # true_length is multiple of tokens_per_page

    # Test with non-zero active_page_position
    true_length_plus_one = jnp.array(num_pages_to_reserve * self.tokens_per_page + 1, dtype=jnp.int32)
    # Release first for this group
    intermediate_state = page_manager._release_pages_for_group(
        final_state, page_group_id, self.max_pages_per_group
    )
    final_state_plus_one = page_manager._reserve_pages_for_group(
        intermediate_state,
        page_group_id,
        true_length_plus_one,
        self.tokens_per_page,
        self.max_pages_per_group
    )
    # num_pages_needed for true_length_plus_one is num_pages_to_reserve + 1
    self.assertEqual(final_state_plus_one.num_pages_used[page_group_id], num_pages_to_reserve + 1)
    self.assertEqual(final_state_plus_one.sequence_lengths[page_group_id], true_length_plus_one)
    self.assertEqual(final_state_plus_one.active_page_position[page_group_id], 1)
    # Pages will be allocated starting from 1 again.
    self.assertEqual(final_state_plus_one.active_page[page_group_id], num_pages_to_reserve + 1)


  def test_reserve_pages_insufficient_global_free_pages(self):
    page_group_id = jnp.array(1, dtype=jnp.int32)
    # Make most pages used, leave only 2 free pages (indices self.num_pages-2, self.num_pages-1)
    # Page 0 is already used. Indices 1 to self.num_pages-3 will be set to used.
    modified_status = self.initial_page_state.page_status.at[1:self.num_pages-2].set(1)
    released_state = self.initial_page_state.replace(page_status=modified_status)
    # Verify only 2 pages are free (excluding page 0)
    self.assertEqual(jnp.sum(released_state.page_status[1:] == 0), 2)

    true_length = jnp.array(3 * self.tokens_per_page, dtype=jnp.int32) # Attempt to reserve 3 pages

    final_state = page_manager._reserve_pages_for_group(
        released_state,
        page_group_id,
        true_length,
        self.tokens_per_page,
        self.max_pages_per_group
    )

    # Assertions: group state should remain as if it was just released (all zeros/False)
    self.assertEqual(final_state.num_pages_used[page_group_id], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id])
    # Global page status should be what it was before this specific failed attempt
    self.assertTrue(jnp.array_equal(final_state.page_status, released_state.page_status))

  def test_reserve_pages_insufficient_group_capacity(self):
    page_group_id = jnp.array(2, dtype=jnp.int32)
    # Attempt to reserve more pages than max_pages_per_group
    true_length = jnp.array((self.max_pages_per_group + 1) * self.tokens_per_page, dtype=jnp.int32)
    released_state = self.initial_page_state # Start with fresh state for this group

    final_state = page_manager._reserve_pages_for_group(
        released_state,
        page_group_id,
        true_length,
        self.tokens_per_page,
        self.max_pages_per_group
    )

    # Assertions: group state should remain as if it was just released
    self.assertEqual(final_state.num_pages_used[page_group_id], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id])
    # Global page status should be unchanged from initial if group had no pages
    self.assertTrue(jnp.array_equal(final_state.page_status, self.initial_page_state.page_status))


  def test_release_pages_release_active_group(self):
    page_group_id = jnp.array(0, dtype=jnp.int32)
    num_pages_to_reserve = 3
    true_length = jnp.array(num_pages_to_reserve * self.tokens_per_page, dtype=jnp.int32)

    # First, reserve some pages for the group
    reserved_state = page_manager._reserve_pages_for_group(
        self.initial_page_state,
        page_group_id,
        true_length,
        self.tokens_per_page,
        self.max_pages_per_group
    )
    # Ensure reservation was successful
    self.assertEqual(reserved_state.num_pages_used[page_group_id], num_pages_to_reserve)
    # Global page indices allocated were 1, 2, 3 because page 0 is pre-used.
    expected_allocated_indices = jnp.array([1, 2, 3], dtype=jnp.int32)
    self.assertTrue(jnp.all(reserved_state.page_status[expected_allocated_indices] == 1))

    # Now, release the pages
    final_state = page_manager._release_pages_for_group(
        reserved_state,
        page_group_id,
        self.max_pages_per_group # static
    )

    # Assertions for release
    self.assertTrue(jnp.all(final_state.page_status[expected_allocated_indices] == 0)) # Pages are free
    self.assertEqual(final_state.page_status[0], 1) # Page 0 remains used
    self.assertEqual(final_state.num_pages_used[page_group_id], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id])
    self.assertEqual(final_state.active_page[page_group_id], 0)
    self.assertEqual(final_state.active_page_position[page_group_id], 0)

  def test_release_pages_release_empty_group(self):
    page_group_id = jnp.array(1, dtype=jnp.int32) # A group that has no pages
    
    # Sanity check: group is indeed empty
    self.assertEqual(self.initial_page_state.num_pages_used[page_group_id], 0)

    final_state = page_manager._release_pages_for_group(
        self.initial_page_state,
        page_group_id,
        self.max_pages_per_group
    )

    # Assertions: No change to global page_status, group state remains cleared
    self.assertTrue(jnp.array_equal(final_state.page_status, self.initial_page_state.page_status))
    self.assertEqual(final_state.num_pages_used[page_group_id], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id])

if __name__ == '__main__':
  absltest.main()
