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

from typing import Optional, Tuple

import common_types
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

Array = common_types.Array
DType = common_types.DType
AxisNames = common_types.AxisNames


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


class PageManager(nn.Module):
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

  def init_or_get_vars(self):
    """Initializes or retrieves the state variables for the paging system.

    Returns:
      Tuple of nn.Variable objects representing:
        - page_status: Status of each page (free/used)
        - page_map: Mapping between slots and their assigned pages
        - sequence_lengths: Length of sequence in each slot
        - num_pages_used: Number of pages used by each slot
        - current_page: Current active page for each slot
        - current_page_position: Position within current pages
    """
    page_status_var = self.variable(
        "cache", "page_status", nn.with_logical_partitioning(jnp.zeros, ("num_pages",)), (self.num_pages,), jnp.int32
    )
    page_map_var = self.variable(
        "cache",
        "page_map",
        nn.with_logical_partitioning(jnp.zeros, ("slots", "max_pages_per_slot")),
        (self.slots, self.max_pages_per_slot),
        jnp.int32,
    )
    sequence_lengths_var = self.variable(
        "cache", "sequence_lengths", nn.with_logical_partitioning(jnp.zeros, ("slots",)), (self.slots,), jnp.int32
    )
    num_pages_used_var = self.variable(
        "cache", "num_pages_used", nn.with_logical_partitioning(jnp.zeros, ("slots",)), (self.slots,), jnp.int32
    )
    current_page_var = self.variable(
        "cache", "current_page", nn.with_logical_partitioning(jnp.zeros, ("slots",)), (self.slots,), jnp.int32
    )
    current_page_position_var = self.variable(
        "cache", "current_page_position", nn.with_logical_partitioning(jnp.zeros, ("slots",)), (self.slots,), jnp.int32
    )

    return (
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var,
    )

  def release_slot_pages(
      self,
      slot: int,
      page_status_var: nn.Variable,
      page_map_var: nn.Variable,
      sequence_lengths_var: nn.Variable,
      num_pages_used_var: nn.Variable,
      current_page_var: nn.Variable,
      current_page_position_var: nn.Variable,
  ) -> Tuple:
    """Releases all pages assigned to a specific slot.

    This method frees up all pages currently assigned to the given slot,
    resetting their status and updating the page mapping accordingly.

    Args:
      slot: Integer identifying the slot to be released
      page_status_var: Variable tracking page usage status
      page_map_var: Variable mapping slots to pages
      sequence_lengths_var: Variable tracking sequence lengths
      num_pages_used_var: Variable tracking page usage counts
      current_page_var: Variable tracking current active pages
      current_page_position_var: Variable tracking positions in current pages

    Returns:
      Tuple of updated variables after releasing the slot's pages
    """
    page_status = page_status_var.value
    page_map = page_map_var.value
    sequence_lengths = sequence_lengths_var.value
    num_pages_used = num_pages_used_var.value
    current_page = current_page_var.value
    current_page_position = current_page_position_var.value

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

    page_status_var.value = page_status
    page_map_var.value = page_map
    sequence_lengths_var.value = sequence_lengths
    num_pages_used_var.value = num_pages_used
    current_page_var.value = current_page
    current_page_position_var.value = current_page_position

    return (
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var,
    )

  def reserve_prefix_slot_pages(
      self,
      slot: int,
      true_length: int,
      page_status_var: nn.Variable,
      page_map_var: nn.Variable,
      sequence_lengths_var: nn.Variable,
      num_pages_used_var: nn.Variable,
      current_page_var: nn.Variable,
      current_page_position_var: nn.Variable,
  ) -> Tuple:
    """Reserves pages for a prefix sequence in the specified slot.

    This method allocates the necessary pages for a prefix sequence of given length,
    first releasing any existing pages assigned to the slot.

    Args:
      slot: Integer identifying the target slot
      true_length: Actual length of the prefix sequence
      page_status_var: Variable tracking page usage status
      page_map_var: Variable mapping slots to pages
      sequence_lengths_var: Variable tracking sequence lengths
      num_pages_used_var: Variable tracking page usage counts
      current_page_var: Variable tracking current active pages
      current_page_position_var: Variable tracking positions in current pages

    Returns:
      Tuple of updated variables after reserving pages for the prefix
    """
    (
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var,
    ) = self.release_slot_pages(
        slot,
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var,
    )

    page_status = page_status_var.value
    page_map = page_map_var.value
    sequence_lengths = sequence_lengths_var.value
    num_pages_used = num_pages_used_var.value
    current_page = current_page_var.value
    current_page_position = current_page_position_var.value

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

    page_status_var.value = page_status
    page_map_var.value = page_map
    sequence_lengths_var.value = sequence_lengths
    num_pages_used_var.value = num_pages_used
    current_page_var.value = current_page
    current_page_position_var.value = current_page_position

    return (
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var,
    )

  def reserve_decode_step_pages(
      self,
      page_status_var: nn.Variable,
      page_map_var: nn.Variable,
      sequence_lengths_var: nn.Variable,
      num_pages_used_var: nn.Variable,
      current_page_var: nn.Variable,
      current_page_position_var: nn.Variable,
  ) -> Tuple:
    """Reserves additional pages needed for a decoding step.

    This method allocates new pages as needed when sequences grow during
    autoregressive decoding, ensuring each active slot has sufficient pages
    for its sequence.

    Args:
      page_status_var: Variable tracking page usage status
      page_map_var: Variable mapping slots to pages
      sequence_lengths_var: Variable tracking sequence lengths
      num_pages_used_var: Variable tracking page usage counts
      current_page_var: Variable tracking current active pages
      current_page_position_var: Variable tracking positions in current pages

    Returns:
      Tuple of updated variables after reserving pages for the decode step
    """
    page_status = page_status_var.value
    page_map = page_map_var.value
    sequence_lengths = sequence_lengths_var.value
    num_pages_used = num_pages_used_var.value
    current_page = current_page_var.value
    current_page_position = current_page_position_var.value

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

    page_status_var.value = page_status
    page_map_var.value = page_map
    sequence_lengths_var.value = sequence_lengths
    num_pages_used_var.value = num_pages_used
    current_page_var.value = current_page
    current_page_position_var.value = current_page_position

    return (
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var,
    )

  @nn.compact
  def __call__(
      self, model_mode: Optional[str] = None, slot: Optional[int] = None, true_length: Optional[int] = None
  ) -> PageState:

    (
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var,
    ) = self.init_or_get_vars()

    if model_mode == common_types.MODEL_MODE_PREFILL and self.is_mutable_collection("params"):
      return PageState(
          page_status_var.value,
          page_map_var.value,
          sequence_lengths_var.value,
          num_pages_used_var.value,
          current_page_var.value,
          current_page_position_var.value,
      )
    elif model_mode == common_types.MODEL_MODE_PREFILL and slot is None and true_length is None:
      return PageState(
          page_status_var.value,
          page_map_var.value,
          sequence_lengths_var.value,
          num_pages_used_var.value,
          current_page_var.value,
          current_page_position_var.value,
      )
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      self.reserve_prefix_slot_pages(
          slot,
          true_length,
          page_status_var,
          page_map_var,
          sequence_lengths_var,
          num_pages_used_var,
          current_page_var,
          current_page_position_var,
      )
    elif model_mode == common_types.MODEL_MODE_INSERT:
      self.reserve_prefix_slot_pages(
          slot,
          true_length,
          page_status_var,
          page_map_var,
          sequence_lengths_var,
          num_pages_used_var,
          current_page_var,
          current_page_position_var,
      )
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      self.reserve_decode_step_pages(
          page_status_var,
          page_map_var,
          sequence_lengths_var,
          num_pages_used_var,
          current_page_var,
          current_page_position_var,
      )

    return PageState(
        page_status_var.value,
        page_map_var.value,
        sequence_lengths_var.value,
        num_pages_used_var.value,
        current_page_var.value,
        current_page_position_var.value,
    )
