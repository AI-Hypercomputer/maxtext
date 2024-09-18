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
  page_map: Array               
  sequence_lengths: Array       
  num_pages_used: Array         
  current_page: Array           
  current_page_position: Array  


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
    page_map_var = self.variable(
        "cache",
        "page_map",
        nn.with_logical_partitioning(jnp.zeros, ("slots", "max_pages_per_slot")),
        (self.slots, self.max_pages_per_slot),
        jnp.int32)
    sequence_lengths_var = self.variable(
        "cache",
        "sequence_lengths",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)
    num_pages_used_var = self.variable(
        "cache",
        "num_pages_used",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)
    current_page_var = self.variable(
        "cache",
        "current_page",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)
    current_page_position_var = self.variable(
        "cache",
        "current_page_position",
        nn.with_logical_partitioning(jnp.zeros, ("slots",)),
        (self.slots,),
        jnp.int32)

    return (
      page_status_var,
      page_map_var,
      sequence_lengths_var,
      num_pages_used_var,
      current_page_var,
      current_page_position_var
    )

  def release_slot_pages(
      self,
      slot: int,
      page_status_var: nn.Variable,
      page_map_var: nn.Variable,
      sequence_lengths_var: nn.Variable,
      num_pages_used_var: nn.Variable,
      current_page_var: nn.Variable,
      current_page_position_var: nn.Variable
  ) -> None:
    """Release sequence slot and the pages assigned to the slot."""    
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
      page_map = page_map.at[slot,i].set(0)
      return page_map, page_status

    page_map, page_status = jax.lax.fori_loop(
      0,
      num_pages_used[slot],
      _release_page,
      (page_map, page_status)
    )

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
      current_page_position_var
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
      current_page_position_var: nn.Variable
  ) -> None:
    """Reserve pages for prefix slot."""
    (
      page_status_var,
      page_map_var,
      sequence_lengths_var,
      num_pages_used_var,
      current_page_var,
      current_page_position_var
    ) = self.release_slot_pages(
      slot,
      page_status_var,
      page_map_var,
      sequence_lengths_var,
      num_pages_used_var,
      current_page_var,
      current_page_position_var
    )

    page_status = page_status_var.value
    page_map = page_map_var.value
    sequence_lengths = sequence_lengths_var.value
    num_pages_used = num_pages_used_var.value
    current_page = current_page_var.value
    current_page_position = current_page_position_var.value

    prefill_slot_num_pages = jnp.ceil(true_length / self.page_size).astype(jnp.int32)

    prefill_slot_page_slice_idx = jnp.where(true_length == 0, 0, (true_length - 1) % self.page_size)


    def _reserve_page(i, state):
      slot, page_map, page_status, current_page = state
      # assert jnp.count_nonzero(page_status[1:]) != self.num_pages-1, "All pages are in use."
      page_idx = jnp.where((page_status[1:]==0), size=1)[0][0] + 1
      page_status = page_status.at[page_idx].set(1)
      page_map = page_map.at[slot, i].set(page_idx)
      current_page = current_page.at[slot].set(page_idx)
      return slot, page_map, page_status, current_page

    _, page_map, page_status, current_page = jax.lax.fori_loop(
      0,
      prefill_slot_num_pages,
      _reserve_page,
      (slot, page_map, page_status, current_page)
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
      current_page_position_var
    )

  def reserve_decode_step_pages(
      self,
      page_status_var: nn.Variable,
      page_map_var: nn.Variable,
      sequence_lengths_var: nn.Variable,
      num_pages_used_var: nn.Variable,
      current_page_var: nn.Variable,
      current_page_position_var: nn.Variable
  ) -> None:
    """Reserve pages for decode step."""
    page_status = page_status_var.value
    page_map = page_map_var.value
    sequence_lengths = sequence_lengths_var.value
    num_pages_used = num_pages_used_var.value
    current_page = current_page_var.value
    current_page_position = current_page_position_var.value

    sequence_lengths_step = jnp.logical_and(jnp.ones(sequence_lengths.shape, dtype = jnp.int32), sequence_lengths).astype(jnp.int32)

    sequence_lengths += sequence_lengths_step

    current_num_pages_used = num_pages_used
    num_pages_used = jnp.ceil(sequence_lengths / self.page_size).astype(jnp.int32)

    current_page_position = jnp.where(sequence_lengths == 0, 0, (sequence_lengths - 1) % self.page_size)
    seq_new_page = num_pages_used - current_num_pages_used

    updating_slots = jnp.where((seq_new_page > 0), size=self.slots)[0]

    def _reserve_page(i, state):
      page_map, page_status, current_page, updating_slots = state
      slot = jax.lax.dynamic_index_in_dim(updating_slots, i, axis=0, keepdims=False)
      page_idx = jnp.where((page_status[1:]==0), size=1)[0][0] + 1
      page_status = page_status.at[page_idx].set(1)
      page_map = page_map.at[slot, num_pages_used[slot]-1].set(page_idx)
      current_page = current_page.at[slot].set(page_idx)
      return page_map, page_status, current_page, updating_slots

    print(f"{page_status.shape=}")
    page_map, page_status, current_page, _ = jax.lax.fori_loop(
      0,
      jnp.count_nonzero(seq_new_page),
      _reserve_page,
      (page_map, page_status, current_page, updating_slots)
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
      current_page_position_var
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
      page_map_var,
      sequence_lengths_var,
      num_pages_used_var,
      current_page_var,
      current_page_position_var
    ) = self.init_or_get_vars()

    if model_mode == common_types.MODEL_MODE_PREFILL:
      assert slot is not None and true_length is not None, f"but get {slot=} and {true_length=} instead"
      self.reserve_prefix_slot_pages(
        slot,
        true_length,
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var
      )
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      self.reserve_decode_step_pages(
        page_status_var,
        page_map_var,
        sequence_lengths_var,
        num_pages_used_var,
        current_page_var,
        current_page_position_var
      )

    return PageState(
      page_status_var.value,
      page_map_var.value,
      sequence_lengths_var.value,
      num_pages_used_var.value,
      current_page_var.value,
      current_page_position_var.value
    )
