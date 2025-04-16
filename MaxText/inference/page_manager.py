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

# TODO Need to update unit tests for this file under Maxtext/tests/page_manager.py.

import jax
import jax.numpy as jnp

from flax import struct

from MaxText import common_types

Array = common_types.Array
DType = common_types.DType
AxisNames = common_types.AxisNames
Config = common_types.Config


@struct.dataclass
class PageState:
  """Represents the current state of the paging system.

  Attributes:
    page_status: Array indicating whether each page is in use (1) or free (0)
    page_map: Array mapping slots to their assigned pages
    sequence_lengths: Array containing the current length of each sequence
    num_pages_used: Array tracking how many pages each slot is using
    current_page: Array indicating the current active page for each slot
    current_page_position: Array tracking position within current pages
  """

  page_status: Array
  page_map: Array
  sequence_lengths: Array
  num_pages_used: Array
  current_page: Array
  current_page_position: Array


def initialize_page_state(
    num_pages: int,
    num_slots: int,
    max_pages_per_slot: int,
) -> PageState:
  return PageState(
      page_status=jnp.zeros((num_pages,), dtype=jnp.int32),
      page_map=jnp.zeros((num_slots, max_pages_per_slot), dtype=jnp.int32),
      sequence_lengths=jnp.zeros((num_slots,), dtype=jnp.int32),
      num_pages_used=jnp.zeros((num_slots,), dtype=jnp.int32),
      current_page=jnp.zeros((num_slots,), dtype=jnp.int32),
      current_page_position=jnp.zeros((num_slots,), dtype=jnp.int32),
  )


class PageManager:
  """Manages paged attention mechanism for efficient sequence processing.

  The PageManager implements a virtual memory-like system for attention, where the
  attention context is divided into fixed-size pages. This allows efficient handling
  of variable-length sequences and helps manage memory usage during inference.

  Attributes:
    num_pages: Total number of available pages in the system
    tokens_per_page: Number of tokens that can be stored in each page
    slots: Number of sequence slots available for parallel processing
    max_target_length: Maximum length of target sequences
    max_prefill_predict_length: Maximum length for prefill prediction
    max_pages_per_slot: Maximum number of pages that can be assigned to a slot
  """

  num_pages: int
  tokens_per_page: int
  slots: int
  max_target_length: int
  max_prefill_predict_length: int
  max_pages_per_slot: int

  def __init__(self, num_pages: int, tokens_per_page: int, max_target_length: int, max_prefill_length: int, batch_size: int):
    self.num_pages = num_pages
    self.tokens_per_page = tokens_per_page
    self.max_target_length = max_target_length
    self.max_pages_per_slot = max_target_length // tokens_per_page
    self.slots = int(batch_size * jax.device_count())
    self.max_prefill_predict_length = max_prefill_length

  def release_slot_pages(
      self,
      slot: int,
      page_state: PageState,
  ) -> PageState:
    page_status = page_state.page_status
    page_map = page_state.page_map
    sequence_lengths = page_state.sequence_lengths
    current_page = page_state.current_page
    current_page_position = page_state.current_page_position
    num_pages_used = page_state.num_pages_used

    def _release_page(i, state):
      page_map, page_status = state
      page_idx = page_map[slot][i]
      page_status = page_status.at[page_idx].set(0)
      page_map = page_map.at[slot, i].set(0)
      return page_map, page_status

    page_map, page_status = jax.lax.fori_loop(0, num_pages_used[slot], _release_page, (page_map, page_status))

    sequence_lengths = sequence_lengths.at[slot].set(0)
    num_pages_used = num_pages_used.at[slot].set(0)
    current_page = current_page.at[slot].set(0)
    current_page_position = current_page_position.at[slot].set(0)

    return PageState(
        page_status=page_status,
        page_map=page_map,
        sequence_lengths=sequence_lengths,
        num_pages_used=num_pages_used,
        current_page=current_page,
        current_page_position=current_page_position,
    )

  def reserve_prefix_slot_pages(
      self,
      slot: int,
      true_length: int,
      page_state: PageState,
  ) -> PageState:
    page_state = self.release_slot_pages(slot, page_state)
    page_status = page_state.page_status
    page_map = page_state.page_map
    sequence_lengths = page_state.sequence_lengths
    num_pages_used = page_state.num_pages_used
    current_page = page_state.current_page
    current_page_position = page_state.current_page_position

    prefill_slot_num_pages = jnp.ceil(true_length / self.tokens_per_page).astype(jnp.int32)
    prefill_slot_page_slice_idx = jnp.where(true_length == 0, 0, (true_length - 1) % self.tokens_per_page)

    def _reserve_page(i, state):
      slot, page_map, page_status, current_page = state
      page_idx = jnp.where((page_status[1:] == 0), size=1)[0][0] + 1
      page_status = page_status.at[page_idx].set(1)
      page_map = page_map.at[slot, i].set(page_idx)
      current_page = current_page.at[slot].set(page_idx)
      return slot, page_map, page_status, current_page

    _, page_map, page_status, current_page = jax.lax.fori_loop(
        0, prefill_slot_num_pages, _reserve_page, (slot, page_map, page_status, current_page)
    )
    sequence_lengths = sequence_lengths.at[slot].set(true_length)
    num_pages_used = num_pages_used.at[slot].set(prefill_slot_num_pages)
    current_page_position = current_page_position.at[slot].set(prefill_slot_page_slice_idx)

    return PageState(
        page_status=page_status,
        page_map=page_map,
        sequence_lengths=sequence_lengths,
        num_pages_used=num_pages_used,
        current_page=current_page,
        current_page_position=current_page_position,
    )

  def reserve_decode_step_pages(
      self,
      page_state: PageState,
  ) -> PageState:
    page_status = page_state.page_status
    page_map = page_state.page_map
    sequence_lengths = page_state.sequence_lengths
    num_pages_used = page_state.num_pages_used
    current_page = page_state.current_page
    current_page_position = page_state.current_page_position

    sequence_lengths_step = jnp.logical_and(jnp.ones(sequence_lengths.shape, dtype=jnp.int32), sequence_lengths).astype(
        jnp.int32
    )

    sequence_lengths += sequence_lengths_step

    current_num_pages_used = num_pages_used
    num_pages_used = jnp.ceil(sequence_lengths / self.tokens_per_page).astype(jnp.int32)

    current_page_position = jnp.where(sequence_lengths == 0, 0, (sequence_lengths - 1) % self.tokens_per_page)
    seq_new_page = num_pages_used - current_num_pages_used

    updating_slots = jnp.where((seq_new_page > 0), size=self.slots)[0]

    def _reserve_page(i, state):
      page_map, page_status, current_page, updating_slots = state
      slot = jax.lax.dynamic_index_in_dim(updating_slots, i, axis=0, keepdims=False)
      page_idx = jnp.where((page_status[1:] == 0), size=1)[0][0] + 1
      page_status = page_status.at[page_idx].set(1)
      page_map = page_map.at[slot, num_pages_used[slot] - 1].set(page_idx)
      current_page = current_page.at[slot].set(page_idx)
      return page_map, page_status, current_page, updating_slots

    page_map, page_status, current_page, _ = jax.lax.fori_loop(
        0, jnp.count_nonzero(seq_new_page), _reserve_page, (page_map, page_status, current_page, updating_slots)
    )

    return PageState(
        page_status=page_status,
        page_map=page_map,
        sequence_lengths=sequence_lengths,
        num_pages_used=num_pages_used,
        current_page=current_page,
        current_page_position=current_page_position,
    )

  def get_initial_page_state(self) -> PageState:
    return initialize_page_state(num_pages=self.num_pages, num_slots=self.slots, max_pages_per_slot=self.max_pages_per_slot)
