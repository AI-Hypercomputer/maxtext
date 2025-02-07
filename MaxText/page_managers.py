from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
from typing import Optional

import common_types

Array = common_types.Array
DType = common_types.DType
AxisNames = common_types.AxisNames

# pylint: disable=too-many-positional-arguments


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

  Attributes:
    num_pages: Total number of available pages.
    tokens_per_page: Number of tokens per page.
    slots: Number of sequence slots.
    max_target_length: Maximum target sequence length.
    max_prefill_predict_length: Maximum prefill prediction length.
    max_pages_per_slot: Maximum pages per slot.
  """

  num_pages: int
  tokens_per_page: int
  slots: int
  max_target_length: int
  max_prefill_predict_length: int
  max_pages_per_slot: int

  def setup(self):
      """Initializes individual state variables, not a combined PageState."""
      self.page_status = self.variable(
          "cache", "page_status",
          lambda: jnp.zeros((self.num_pages,), dtype=jnp.int32)
      )
      self.page_map = self.variable(
          "cache", "page_map",
          lambda: jnp.full((self.slots, self.max_pages_per_slot), -1, dtype=jnp.int32)
      )
      self.sequence_lengths = self.variable(
          "cache", "sequence_lengths",
          lambda: jnp.zeros((self.slots,), dtype=jnp.int32)
      )
      self.num_pages_used = self.variable(
          "cache", "num_pages_used",
          lambda: jnp.zeros((self.slots,), dtype=jnp.int32)
      )
      self.current_page = self.variable(
          "cache", "current_page",
          lambda: jnp.full((self.slots,), -1, dtype=jnp.int32)
      )
      self.current_page_position = self.variable(
          "cache", "current_page_position",
          lambda: jnp.zeros((self.slots,), dtype=jnp.int32)
      )


  def release_slot_pages(
      self,
      slot: int,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
  ) -> None: # Changed return type
    """Releases all pages assigned to a specific slot, operating on individual vars."""

    # Find pages to release.  Use page_map.value directly.
    mapped_pages = self.page_map.value[slot]
    used_pages = mapped_pages[mapped_pages >= 0]


    for page_idx in used_pages:
      self.page_status.value = self.page_status.value.at[page_idx].set(0)  # Mark as free
      # Zero out the KV cache for the released page
      key_pages_var.value = key_pages_var.value.at[:, page_idx, :, :].set(
          jnp.zeros_like(key_pages_var.value[:, page_idx, :, :])
      )
      value_pages_var.value = value_pages_var.value.at[:, page_idx, :, :].set(
          jnp.zeros_like(value_pages_var.value[:, page_idx, :, :])
      )

    # Clear the slot's entries in the page map
    self.page_map.value = self.page_map.value.at[slot, :].set(-1)
    # Reset other state variables for the slot
    self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(0)
    self.num_pages_used.value = self.num_pages_used.value.at[slot].set(0)
    self.current_page.value = self.current_page.value.at[slot].set(-1)
    self.current_page_position.value = self.current_page_position.value.at[slot].set(0)


  def reserve_prefix_slot_pages(
      self,
      slot: int,
      true_length: int,
      key_pages_var: nn.Variable,
      value_pages_var: nn.Variable,
  ) -> None: # Changed return type
    """Reserves pages for a prefix sequence in the specified slot."""

    # First, release any existing pages for the slot.
    self.release_slot_pages(slot, key_pages_var, value_pages_var)

    # Calculate the number of pages needed.
    num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    last_page_position = (true_length - 1) % self.tokens_per_page if true_length > 0 else 0

    # Allocate pages using jax.lax.fori_loop.
    def _reserve_single_page(i, carry):
      page_status, page_map, current_page = carry
      # Find the next free page.
      next_free_page = jnp.where(page_status == 0, size=1, fill_value=-1)[0][0]
      page_status = page_status.at[next_free_page].set(1)      # Mark as used
      page_map = page_map.at[slot, i].set(next_free_page)    # Update page_map
      current_page = current_page.at[slot].set(next_free_page)  # Update current_page

      return page_status, page_map, current_page


    self.page_status.value, self.page_map.value, self.current_page.value = jax.lax.fori_loop(
        0, num_pages_needed, _reserve_single_page,
        (self.page_status.value, self.page_map.value, self.current_page.value)
    )
    # Update other state variables:
    self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(true_length)
    self.num_pages_used.value = self.num_pages_used.value.at[slot].set(num_pages_needed)
    self.current_page_position.value = self.current_page_position.value.at[slot].set(last_page_position)



  def reserve_decode_step_pages(
      self
  ) -> None:  # Changed return type
        """Reserves any needed pages for a decoding step."""

        new_sequence_lengths = self.sequence_lengths.value + jnp.where(self.current_page.value == -1, 0, 1)
        new_num_pages_used = (new_sequence_lengths + self.tokens_per_page - 1) // self.tokens_per_page
        new_current_page_position = jnp.where(new_sequence_lengths == 0, 0, (new_sequence_lengths - 1) % self.tokens_per_page)

        slots_to_update = jnp.where(new_num_pages_used > self.num_pages_used.value)[0]

        def _reserve_single_page(i, carry):
            page_status, page_map, current_page, num_pages_used = carry
            slot = slots_to_update[i]
            next_free_page = jnp.where(page_status == 0, size=1, fill_value=-1)[0][0]
            page_status = page_status.at[next_free_page].set(1)  # Mark as used
            page_map = page_map.at[slot, num_pages_used[slot]].set(next_free_page)  # Update page map
            current_page = current_page.at[slot].set(next_free_page) # Update current_page

            return page_status, page_map, current_page, num_pages_used


        self.page_status.value, self.page_map.value, self.current_page.value, new_num_pages_used = jax.lax.fori_loop(
          0, slots_to_update.shape[0], _reserve_single_page,
          (self.page_status.value, self.page_map.value, self.current_page.value, self.num_pages_used.value)
        )
        self.num_pages_used.value = new_num_pages_used
        self.sequence_lengths.value = new_sequence_lengths
        self.current_page_position.value = new_current_page_position


  def __call__(
      self, model_mode: Optional[str] = None, slot: Optional[int] = None, true_length: Optional[int] = None,
      key_pages_var: Optional[nn.Variable] = None, value_pages_var: Optional[nn.Variable] = None
  ) -> None: # Changed return type

    if model_mode == common_types.MODEL_MODE_PREFILL and self.is_mutable_collection("params"):
      # Initialization happens in setup().  No need to do anything extra here.
      return

    elif model_mode == common_types.MODEL_MODE_PREFILL:
        # During prefill, reserve pages for the given slot and length.
        self.reserve_prefix_slot_pages(slot, true_length, key_pages_var, value_pages_var)

    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      # During autoregressive decoding, reserve any needed pages.
      self.reserve_decode_step_pages()
