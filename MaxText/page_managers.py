#  Copyright 2024 Google LLC
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

"""Page Managers."""

from typing import Optional

from flax import linen as nn
from flax import struct
import jax.numpy as jnp

import common_types
import jax

Array = common_types.Array
DType = common_types.DType

AxisNames = common_types.AxisNames


@struct.dataclass
class PageState:
  page_status: Array
  seq_page_idx_mappings: Array
  seq_lengths: Array
  seq_num_pages: Array
  seq_page_indices: Array
  seq_page_slice_indices: Array


class PageManager(nn.Module):
  """Page Manager"""
  num_pages: int
  page_size: int
  slots: int
  max_target_length: int
  max_prefill_predict_length: int
  max_pages_per_slot: int

  def init_or_get_vars(self):
    page_status_var = self.variable(
        "cache",
        "page_status",
        nn.with_logical_partitioning(jnp.zeros, ("num_pages",)),
        (self.num_pages,),
        jnp.int32)
    seq_page_idx_mappings_var = self.variable(
        "cache",
        "seq_page_idx_mappings",
        nn.with_logical_partitioning(jnp.zeros, ("slots", "max_pages_per_slot")),
        (self.slots, self.max_pages_per_slot),
        jnp.int32)
    seq_lengths_var = self.variable(
        "cache",
        "seq_lengths",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)
    seq_num_pages_var = self.variable(
        "cache",
        "seq_num_pages",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)
    seq_page_indices_var = self.variable(
        "cache",
        "seq_page_indices",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)
    seq_page_slice_indices_var = self.variable(
        "cache",
        "seq_page_slice_indices",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)

    return (
      page_status_var,
      seq_page_idx_mappings_var,
      seq_lengths_var,
      seq_num_pages_var,
      seq_page_indices_var,
      seq_page_slice_indices_var
    )

  def release_slot_pages(
      self,
      slot: int,
      page_status_var: nn.Variable,
      seq_page_idx_mappings_var: nn.Variable,
      seq_lengths_var: nn.Variable,
      seq_num_pages_var: nn.Variable,
      seq_page_indices_var: nn.Variable,
      seq_page_slice_indices_var: nn.Variable
  ) -> None:
    """Release sequence slot and the pages assigned to the slot."""    
    page_status = page_status_var.value
    seq_page_idx_mappings = seq_page_idx_mappings_var.value
    seq_lengths = seq_lengths_var.value
    seq_num_pages = seq_num_pages_var.value
    seq_page_indices = seq_page_indices_var.value
    seq_page_slice_indices = seq_page_slice_indices_var.value

    def _release_page(i, state):
      seq_page_idx_mappings, page_status = state
      page_idx = seq_page_idx_mappings[slot][i]
      page_status.at[page_idx].set(0)
      seq_page_idx_mappings.at[slot,i].set(0)
      return seq_page_idx_mappings, page_status

    seq_page_idx_mappings, page_status = jax.lax.fori_loop(
      0,
      seq_num_pages[slot],
      _release_page,
      (seq_page_idx_mappings, page_status)
    )

    seq_lengths.at[slot].set(0)
    seq_num_pages.at[slot].set(0)
    seq_page_indices.at[slot].set(0)
    seq_page_slice_indices.at[slot].set(0)

    page_status_var.value = page_status
    seq_page_idx_mappings_var.value = seq_page_idx_mappings
    seq_lengths_var.value = seq_lengths
    seq_num_pages_var.value = seq_num_pages
    seq_page_indices_var.value = seq_page_indices
    seq_page_slice_indices_var.value = seq_page_slice_indices

    return (
      page_status_var,
      seq_page_idx_mappings_var,
      seq_lengths_var,
      seq_num_pages_var,
      seq_page_indices_var,
      seq_page_slice_indices_var
    )

  def reserve_prefix_slot_pages(
      self,
      slot: int,
      true_length: int,
      page_status_var: nn.Variable,
      seq_page_idx_mappings_var: nn.Variable,
      seq_lengths_var: nn.Variable,
      seq_num_pages_var: nn.Variable,
      seq_page_indices_var: nn.Variable,
      seq_page_slice_indices_var: nn.Variable
  ) -> None:
    """Reserve pages for prefix slot."""
    (
      page_status_var,
      seq_page_idx_mappings_var,
      seq_lengths_var,
      seq_num_pages_var,
      seq_page_indices_var,
      seq_page_slice_indices_var
    ) = self.release_slot_pages(
      slot,
      page_status_var,
      seq_page_idx_mappings_var,
      seq_lengths_var,
      seq_num_pages_var,
      seq_page_indices_var,
      seq_page_slice_indices_var
    )

    page_status = page_status_var.value
    seq_page_idx_mappings = seq_page_idx_mappings_var.value
    seq_lengths = seq_lengths_var.value
    seq_num_pages = seq_num_pages_var.value
    seq_page_indices = seq_page_indices_var.value
    seq_page_slice_indices = seq_page_slice_indices_var.value

    jax.debug.print("reserve_prefix_slot_pages - seq_page_indices: {}", seq_page_indices)
    # reserve_prefix_slot_pages - seq_page_indices: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    prefill_slot_num_pages = jnp.ceil(true_length / self.page_size).astype(jnp.int32)
    jax.debug.print("reserve_prefix_slot_pages - prefill_slot_num_pages: {}", prefill_slot_num_pages)
    # reserve_prefix_slot_pages - prefill_slot_num_pages: 1

    prefill_slot_page_slice_idx = jnp.where(true_length == 0, 0, (true_length - 1) % self.page_size)
    jax.debug.print("reserve_prefix_slot_pages - prefill_slot_page_slice_idx: {}", prefill_slot_page_slice_idx)
    # reserve_prefix_slot_pages - prefill_slot_page_slice_idx: 3


    def _reserve_page(i, state):
      slot, seq_page_idx_mappings, page_status, seq_page_indices = state
      # assert jnp.count_nonzero(page_status[1:]) != self.num_pages-1, "All pages are in use."
      page_idx = jnp.where((page_status[1:]==0), size=1)[0][0]
      page_status.at[page_idx].set(1)
      seq_page_idx_mappings.at[slot, i].set(page_idx)
      seq_page_indices.at[slot].set(page_idx)
      return slot, seq_page_idx_mappings, page_status, seq_page_indices

    _, seq_page_idx_mappings, page_status, seq_page_indices = jax.lax.fori_loop(
      0,
      prefill_slot_num_pages,
      _reserve_page,
      (slot, seq_page_idx_mappings, page_status, seq_page_indices)
    )

    seq_lengths.at[slot].set(true_length)
    seq_num_pages.at[slot].set(prefill_slot_num_pages)
    seq_page_slice_indices.at[slot].set(prefill_slot_page_slice_idx)

    page_status_var.value = page_status
    seq_page_idx_mappings_var.value = seq_page_idx_mappings
    seq_lengths_var.value = seq_lengths
    seq_num_pages_var.value = seq_num_pages
    seq_page_indices_var.value = seq_page_indices
    seq_page_slice_indices_var.value = seq_page_slice_indices

    return (
      page_status_var,
      seq_page_idx_mappings_var,
      seq_lengths_var,
      seq_num_pages_var,
      seq_page_indices_var,
      seq_page_slice_indices_var
    )

  def reserve_decode_step_pages(
      self,
      page_status_var: nn.Variable,
      seq_page_idx_mappings_var: nn.Variable,
      seq_lengths_var: nn.Variable,
      seq_num_pages_var: nn.Variable,
      seq_page_indices_var: nn.Variable,
      seq_page_slice_indices_var: nn.Variable
  ) -> None:
    """Reserve pages for decode step."""
    page_status = page_status_var.value
    jax.debug.print("\nreserve_decode_step_pages - page_status: {}", page_status)

    seq_page_idx_mappings = seq_page_idx_mappings_var.value
    jax.debug.print("reserve_decode_step_pages - seq_page_idx_mappings: {}", seq_page_idx_mappings)

    seq_lengths = seq_lengths_var.value
    jax.debug.print("reserve_decode_step_pages - seq_lengths: {}", seq_lengths)

    seq_num_pages = seq_num_pages_var.value
    jax.debug.print("reserve_decode_step_pages - seq_num_pages: {}", seq_num_pages)

    seq_page_indices = seq_page_indices_var.value
    jax.debug.print("reserve_decode_step_pages - seq_page_indices: {}", seq_page_indices)

    seq_page_slice_indices = seq_page_slice_indices_var.value
    jax.debug.print("reserve_decode_step_pages - seq_page_slice_indices: {}", seq_page_slice_indices)

    seq_lengths_step = jnp.logical_and(jnp.ones(seq_lengths.shape, dtype = jnp.int32), seq_lengths).astype(jnp.int32)
    jax.debug.print("reserve_decode_step_pages - seq_lengths_step: {}", seq_lengths_step)

    seq_lengths += seq_lengths_step
    jax.debug.print("reserve_decode_step_pages - seq_lengths: {}", seq_lengths)

    current_seq_num_pages = seq_num_pages
    jax.debug.print("reserve_decode_step_pages - current_seq_num_pages: {}", current_seq_num_pages)

    seq_num_pages = jnp.ceil(seq_lengths / self.page_size).astype(jnp.int32)
    jax.debug.print("reserve_decode_step_pages - seq_num_pages: {}", seq_num_pages)

    seq_page_slice_indices = jnp.where(seq_lengths == 0, 0, (seq_lengths - 1) % self.page_size)
    jax.debug.print("reserve_decode_step_pages - seq_page_slice_indices: {}", seq_page_slice_indices)

    seq_new_page = seq_num_pages - current_seq_num_pages
    jax.debug.print("reserve_decode_step_pages - seq_new_page: {}", seq_new_page)

    updating_slots = jnp.where((seq_new_page>0), size=self.slots)[0]
    jax.debug.print("reserve_decode_step_pages - updating_slots: {}\n", updating_slots)

    def _reserve_page(i, state):
      seq_page_idx_mappings, page_status, seq_page_indices, updating_slots = state
      # assert jnp.count_nonzero(page_status[1:]) != self.num_pages-1, "All pages are in use."
      slot = jax.lax.dynamic_index_in_dim(updating_slots, i, axis=0, keepdims=False)
      page_idx = jnp.where((page_status[1:]==0), size=1)[0][0]
      page_status.at[page_idx].set(1)
      seq_page_idx_mappings.at[slot,seq_num_pages[slot]-1].set(page_idx)
      seq_page_indices.at[slot].set(page_idx)
      return seq_page_idx_mappings, page_status, seq_page_indices, updating_slots

    seq_page_idx_mappings, page_status, seq_page_indices, _ = jax.lax.fori_loop(
      0,
      jnp.count_nonzero(seq_new_page),
      _reserve_page,
      (seq_page_idx_mappings, page_status, seq_page_indices, updating_slots)
    )

    page_status_var.value = page_status
    seq_page_idx_mappings_var.value = seq_page_idx_mappings
    seq_lengths_var.value = seq_lengths
    seq_num_pages_var.value = seq_num_pages
    seq_page_indices_var.value = seq_page_indices
    seq_page_slice_indices_var.value = seq_page_slice_indices

    return (
      page_status_var,
      seq_page_idx_mappings_var,
      seq_lengths_var,
      seq_num_pages_var,
      seq_page_indices_var,
      seq_page_slice_indices_var
    )

  @nn.compact
  def __call__(
      self,
      model_mode: Optional[str] = None,
      slot: Optional[int] = None,
      true_length: Optional[int] = None
    ) -> PageState:

    (
      page_status_var,
      seq_page_idx_mappings_var,
      seq_lengths_var,
      seq_num_pages_var,
      seq_page_indices_var,
      seq_page_slice_indices_var
    ) = self.init_or_get_vars()

    if model_mode == common_types.MODEL_MODE_PREFILL:
      assert slot is not None and true_length is not None, f"but get {slot=} and {true_length=} instead"
      self.reserve_prefix_slot_pages(
        slot,
        true_length,
        page_status_var,
        seq_page_idx_mappings_var,
        seq_lengths_var,
        seq_num_pages_var,
        seq_page_indices_var,
        seq_page_slice_indices_var
      )
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      self.reserve_decode_step_pages(
        page_status_var,
        seq_page_idx_mappings_var,
        seq_lengths_var,
        seq_num_pages_var,
        seq_page_indices_var,
        seq_page_slice_indices_var
      )

    return PageState(
      page_status_var.value,
      seq_page_idx_mappings_var.value,
      seq_lengths_var.value,
      seq_num_pages_var.value,
      seq_page_indices_var.value,
      seq_page_slice_indices_var.value
    )
