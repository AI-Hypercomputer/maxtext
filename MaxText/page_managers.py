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


  def find_next_free_page(self, page_status: Array) -> Array:
    """Finds the index of the next free page.

    Args:
        page_status: The current page status array.

    Returns:
      The index of the next free page.
    """
    # Efficiently find the *first* free page.  Skip index 0.
    free_pages = jnp.where(page_status[1:] == 0, size=1, fill_value=-1)[0] + 1
    return free_pages[0]

  def release_slot_pages(
    self,
    slot: int
  ) -> None:
    """Releases all pages assigned to a specific slot.
    
    Args:
        slot: The slot index to release
    """
    # Get indices of used pages
    used_pages = jnp.where(self.page_map.value[slot] > -1, 
                          self.page_map.value[slot], 
                          0)
    
    # Reset page status for used pages
    self.page_status.value = self.page_status.value.at[used_pages].set(0)
    
    # Reset slot state
    self.page_map.value = self.page_map.value.at[slot, :].set(-1)
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
    self.release_slot_pages(slot)

    # Calculate the number of pages needed.
    num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    last_page_position = (true_length - 1) % self.tokens_per_page if true_length > 0 else 0

    # Allocate pages using jax.lax.fori_loop.
    def _reserve_single_page(i, carry):
      page_status, page_map = carry
      # Find the next free page.
      next_free_page = self.find_next_free_page(page_status)
      page_status = page_status.at[next_free_page].set(1)      # Mark as used
      page_map = page_map.at[slot, i].set(next_free_page)    # Update page_map

      return page_status, page_map


    self.page_status.value, self.page_map.value = jax.lax.fori_loop(
        0, num_pages_needed, _reserve_single_page,
        (self.page_status.value, self.page_map.value)
    )
    # Update other state variables:
    self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(true_length)
    self.num_pages_used.value = self.num_pages_used.value.at[slot].set(num_pages_needed)
    # Correctly set the current page to the LAST allocated page.
    if num_pages_needed > 0:
        self.current_page.value = self.current_page.value.at[slot].set(
            self.page_map.value[slot, num_pages_needed - 1]
        )
    else:
        self.current_page.value = self.current_page.value.at[slot].set(-1)
    self.current_page_position.value = self.current_page_position.value.at[slot].set(last_page_position)



  def reserve_decode_step_pages(self) -> None:
    """Reserves any needed pages for a decoding step."""
    new_sequence_lengths = self.sequence_lengths.value + jnp.where(
        self.current_page.value == -1, 0, 1
    )
    new_num_pages_used = (new_sequence_lengths + self.tokens_per_page - 1) // self.tokens_per_page
    new_current_page_position = jnp.where(
        new_sequence_lengths == 0, 
        0, 
        (new_sequence_lengths - 1) % self.tokens_per_page
    )

    # Instead of using dynamic where, we'll iterate through all slots
    def _reserve_single_slot(i, carry):
        page_status, page_map, current_page, num_pages_used = carry
        
        # Check if this slot needs a new page
        needs_new_page = new_num_pages_used[i] > self.num_pages_used.value[i]
        
        # Find next free page only if needed
        next_free_page = jax.lax.cond(
            needs_new_page,
            lambda _: self.find_next_free_page(page_status),
            lambda _: -1,
            operand=None
        )
        
        # Update state conditionally
        page_status = jax.lax.cond(
            needs_new_page,
            lambda x: x.at[next_free_page].set(1),
            lambda x: x,
            page_status
        )
        
        page_map = jax.lax.cond(
            needs_new_page,
            lambda x: x.at[i, self.num_pages_used.value[i]].set(next_free_page),
            lambda x: x,
            page_map
        )
        
        current_page = jax.lax.cond(
            needs_new_page,
            lambda x: x.at[i].set(next_free_page),
            lambda x: x,
            current_page
        )

        return page_status, page_map, current_page, num_pages_used

    # Process all slots
    self.page_status.value, self.page_map.value, self.current_page.value, _ = jax.lax.fori_loop(
        0, self.slots,  # Process all slots
        _reserve_single_slot,
        (self.page_status.value, self.page_map.value, 
         self.current_page.value, self.num_pages_used.value)
    )

    # Update the remaining state variables
    self.num_pages_used.value = new_num_pages_used
    self.sequence_lengths.value = new_sequence_lengths
    self.current_page_position.value = new_current_page_position


  def __call__(
    self, 
    model_mode: Optional[str] = None, 
    slot: Optional[int] = None,
    true_length: Optional[int] = None
) -> None | PageState:
    """Manages page allocation and returns current PageState.
    
    Args:
        model_mode: Current model mode (prefill/autoregressive)
        slot: Slot index for prefill mode
        true_length: Sequence length for prefill mode
        
    Returns:
        PageState object containing current state, or None during initialization
    """
    # Early return during initialization
    if model_mode == common_types.MODEL_MODE_PREFILL and self.is_mutable_collection("params"):
        return None

    # Handle prefill mode - actively manages page allocation
    if model_mode == common_types.MODEL_MODE_PREFILL:
        # Release any existing pages for this slot
        self.release_slot_pages(slot)
        
        # Calculate number of pages needed
        num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
        
        # Update sequence length tracking
        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(true_length)
        self.num_pages_used.value = self.num_pages_used.value.at[slot].set(num_pages_needed)

    # Handle autoregressive mode - manages page updates during decoding
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        self.reserve_decode_step_pages()

    # Return current state after updates
    return PageState(
        page_status=self.page_status.value,
        page_map=self.page_map.value,
        sequence_lengths=self.sequence_lengths.value,
        num_pages_used=self.num_pages_used.value,
        current_page=self.current_page.value,
        current_page_position=self.current_page_position.value
    )