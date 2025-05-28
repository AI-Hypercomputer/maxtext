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
          'expected_index': -1,  # Searches page_status[1:]
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
          'testcase_name': 'only_page_0_no_others',  # page_status[1:] is empty
          'page_status': jnp.array([1], dtype=jnp.int32),
          'expected_index': -1,
      },
      {
          'testcase_name': 'two_pages_0_used_1_free',
          'page_status': jnp.array([1, 0], dtype=jnp.int32),
          'expected_index': 1,
      }
  )
  def test_find_next_free_page_index(self, page_status, expected_index):
    next_free = page_manager._find_next_free_page_index(page_status)
    self.assertEqual(next_free, expected_index)



  def setUp(self):
    super().setUp()
    self.num_pages = 64
    self.max_page_groups = 4
    self.tokens_per_page = 16
    self.max_pages_per_group = 32

    self.initial_page_state = page_manager.initialize_page_state(
        num_pages=self.num_pages,
        max_page_groups=self.max_page_groups,
        max_pages_per_group=self.max_pages_per_group,
    )

  def test_reserve_pages_successful_reservation(self):
    page_group_id_val = 0
    page_group_id = jnp.array(page_group_id_val, dtype=jnp.int32)
    num_pages_to_reserve = 5
    true_length_val = num_pages_to_reserve * self.tokens_per_page
    true_length = jnp.array(true_length_val, dtype=jnp.int32)

    released_state = self.initial_page_state

    final_state = page_manager._reserve_pages_for_group(
        released_state,
        page_group_id,
        true_length,
        self.tokens_per_page,
        self.max_pages_per_group,
    )

    self.assertEqual(final_state.num_pages_used[page_group_id_val], num_pages_to_reserve)
    self.assertEqual(final_state.sequence_lengths[page_group_id_val], true_length_val)

    # Check page_status: page 0 is used by default, then pages 1 to num_pages_to_reserve
    current_status_check = jnp.zeros_like(self.initial_page_state.page_status).at[0].set(1)
    current_status_check = current_status_check.at[1:num_pages_to_reserve+1].set(1)
    self.assertTrue(jnp.array_equal(final_state.page_status, current_status_check))

    # Check page_map: first num_pages_to_reserve slots should contain page indices 1 to num_pages_to_reserve
    expected_page_map_row = jnp.arange(1, num_pages_to_reserve + 1, dtype=jnp.int32)
    expected_page_map_full_row = expected_page_map_row
    self.assertTrue(jnp.array_equal(final_state.page_map[page_group_id_val, :num_pages_to_reserve], expected_page_map_full_row))

    self.assertEqual(final_state.active_page[page_group_id_val], expected_page_map_row[-1])
    self.assertTrue(final_state.has_active_page[page_group_id_val])
    self.assertEqual(final_state.active_page_position[page_group_id_val], 0)

    # Test with non-zero active_page_position
    true_length_plus_one_val = true_length_val + 1
    true_length_plus_one = jnp.array(true_length_plus_one_val, dtype=jnp.int32)

    intermediate_state = page_manager._release_pages_for_group(
        final_state, page_group_id, self.max_pages_per_group
    )
    final_state_plus_one = page_manager._reserve_pages_for_group(
        intermediate_state,
        page_group_id,
        true_length_plus_one,
        self.tokens_per_page,
        self.max_pages_per_group,
    )

    self.assertEqual(final_state_plus_one.num_pages_used[page_group_id_val], num_pages_to_reserve + 1)
    self.assertEqual(final_state_plus_one.sequence_lengths[page_group_id_val], true_length_plus_one_val)
    self.assertEqual(final_state_plus_one.active_page_position[page_group_id_val], 1)
    self.assertEqual(final_state_plus_one.active_page[page_group_id_val],  num_pages_to_reserve +1 )

  def test_reserve_pages_minimum_reservation(self):
      page_group_id_val = 1
      page_group_id = jnp.array(page_group_id_val, dtype=jnp.int32)
      num_pages_to_reserve = 1  # This is the minimum if true_length > 0
      true_length_val = 1  # Smallest possible true_length > 0 for this test
      true_length = jnp.array(true_length_val, dtype=jnp.int32)

      released_state = self.initial_page_state
      final_state = page_manager._reserve_pages_for_group(
          released_state,
          page_group_id,
          true_length,
          self.tokens_per_page,
          self.max_pages_per_group,
      )
      self.assertEqual(final_state.num_pages_used[page_group_id_val], num_pages_to_reserve)
      self.assertEqual(final_state.sequence_lengths[page_group_id_val], true_length_val)

      current_status_check = jnp.zeros_like(self.initial_page_state.page_status).at[0].set(1).at[1].set(1)
      self.assertTrue(jnp.array_equal(final_state.page_status, current_status_check))
      expected_page_map_row = jnp.array([1], dtype=jnp.int32)
      self.assertTrue(jnp.array_equal(final_state.page_map[page_group_id_val, :num_pages_to_reserve], expected_page_map_row))

      self.assertEqual(final_state.active_page[page_group_id_val], 1)
      self.assertTrue(final_state.has_active_page[page_group_id_val])
      self.assertEqual(final_state.active_page_position[page_group_id_val], 1)

  def test_reserve_pages_insufficient_global_free_pages(self):
    page_group_id_val = 1
    page_group_id = jnp.array(page_group_id_val, dtype=jnp.int32)

    modified_status = self.initial_page_state.page_status.at[1:self.num_pages-2].set(1)
    released_state = self.initial_page_state.replace(page_status=modified_status)
    self.assertEqual(jnp.sum(released_state.page_status[1:] == 0), 2) # Verify only 2 pages are free

    true_length = jnp.array(3 * self.tokens_per_page, dtype=jnp.int32) # Needs 3 pages

    final_state = page_manager._reserve_pages_for_group(
        released_state,
        page_group_id,
        true_length,
        self.tokens_per_page,
        self.max_pages_per_group
    )

    self.assertEqual(final_state.num_pages_used[page_group_id_val], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id_val], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id_val] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id_val])
    self.assertTrue(jnp.array_equal(final_state.page_status, released_state.page_status))

  def test_reserve_pages_insufficient_group_capacity(self):
    page_group_id_val = 2
    page_group_id = jnp.array(page_group_id_val, dtype=jnp.int32)
    
    true_length = jnp.array((self.max_pages_per_group + 1) * self.tokens_per_page, dtype=jnp.int32)
    released_state = self.initial_page_state

    final_state = page_manager._reserve_pages_for_group(
        released_state,
        page_group_id,
        true_length,
        self.tokens_per_page,
        self.max_pages_per_group
    )

    self.assertEqual(final_state.num_pages_used[page_group_id_val], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id_val], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id_val] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id_val])
    self.assertTrue(jnp.array_equal(final_state.page_status, self.initial_page_state.page_status))

  def test_release_pages_release_active_group(self):
    page_group_id_val = 0
    page_group_id = jnp.array(page_group_id_val, dtype=jnp.int32)
    num_pages_to_reserve = 3
    true_length = jnp.array(num_pages_to_reserve * self.tokens_per_page, dtype=jnp.int32)

    reserved_state = page_manager._reserve_pages_for_group(
        self.initial_page_state,
        page_group_id,
        true_length,
        self.tokens_per_page,
        self.max_pages_per_group
    )
    self.assertEqual(reserved_state.num_pages_used[page_group_id_val], num_pages_to_reserve)
    expected_allocated_indices = jnp.array([1, 2, 3], dtype=jnp.int32)
    self.assertTrue(jnp.all(reserved_state.page_status.take(expected_allocated_indices) == 1))

    final_state = page_manager._release_pages_for_group(
        reserved_state,
        page_group_id,
        self.max_pages_per_group
    )

    self.assertTrue(jnp.all(final_state.page_status.take(expected_allocated_indices) == 0))
    self.assertEqual(final_state.page_status[0], 1)  # Check page 0 remains used
    self.assertEqual(final_state.num_pages_used[page_group_id_val], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id_val], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id_val] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id_val])
    self.assertEqual(final_state.active_page[page_group_id_val], 0)
    self.assertEqual(final_state.active_page_position[page_group_id_val], 0)

  def test_release_pages_release_empty_group(self):
    page_group_id_val = 1
    page_group_id = jnp.array(page_group_id_val, dtype=jnp.int32)
    
    self.assertEqual(self.initial_page_state.num_pages_used[page_group_id_val], 0)

    final_state = page_manager._release_pages_for_group(
        self.initial_page_state,
        page_group_id,
        self.max_pages_per_group
    )

    self.assertTrue(jnp.array_equal(final_state.page_status, self.initial_page_state.page_status))
    self.assertEqual(final_state.num_pages_used[page_group_id_val], 0)
    self.assertEqual(final_state.sequence_lengths[page_group_id_val], 0)
    self.assertTrue(jnp.all(final_state.page_map[page_group_id_val] == 0))
    self.assertFalse(final_state.has_active_page[page_group_id_val])

if __name__ == '__main__':
  absltest.main()