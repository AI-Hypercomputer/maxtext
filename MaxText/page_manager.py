import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional, Any, Tuple


@struct.dataclass
class PageState:
  """
  Dataclass that holds the state of pages managed by PageManager.

  This class represents the current state of memory pages in the system, tracking both
  global and per-layer page allocations, mappings, and usage information.

  Attributes:
      page_status: 2D array tracking whether each page is free (0) or allocated (1) across layers
      page_map: 3D array mapping page groups to their allocated pages for each layer
      sequence_lengths: 2D array storing sequence lengths for each page group in each layer
      num_pages_used: 2D array tracking number of pages used by each group in each layer
      current_page: 2D array indicating the current active page for each group in each layer
      current_page_position: 2D array tracking position within current page for each group
  """

  page_status: jnp.ndarray  # [num_layers, num_pages] | 0: free, 1: allocated
  page_map: jnp.ndarray  # [num_layers, max_page_groups, max_pages_per_group]
  sequence_lengths: jnp.ndarray  # [num_layers, max_page_groups]
  num_pages_used: jnp.ndarray  # [num_layers, max_page_groups]
  current_page: jnp.ndarray  # [num_layers, max_page_groups]
  current_page_position: jnp.ndarray  # [num_layers, max_page_groups]


def validate_page_group(page_group_id: int, max_page_groups: int) -> bool:
  """
  Validates if a page group ID is within the valid range.

  Args:
      page_group_id: The ID of the page group to validate
      max_page_groups: Maximum number of allowed page groups

  Returns:
      bool: True if the page group ID is valid, False otherwise
  """
  return jnp.logical_and(page_group_id >= 0, page_group_id < max_page_groups)


def validate_length(length: int, max_target_length: int) -> bool:
  """
  Validates if a sequence length is within the valid range.

  Args:
      length: The sequence length to validate
      max_target_length: Maximum allowed sequence length

  Returns:
      bool: True if the length is valid, False otherwise
  """
  return jnp.logical_and(length >= 0, length <= max_target_length)


class PageManager:
  """
  Manages memory page allocation for prefill and autoregressive decoding operations.

  This class handles the allocation, tracking, and deallocation of memory pages
  across multiple layers and page groups. It supports both prefill operations
  (initial allocation) and autoregressive decoding (incremental allocation).
  """

  def __init__(
      self,
      num_pages: int,
      tokens_per_page: int,
      max_page_groups: int,
      max_target_length: int,
      max_prefill_predict_length: int,
      max_pages_per_group: int,
      num_layers: int,
      config: Any,
  ):
    """
    Initialize the page manager with the given configuration.

    Args:
        num_pages: Total number of memory pages available
        tokens_per_page: Number of tokens that can be stored in each page
        max_page_groups: Maximum number of page groups allowed
        max_target_length: Maximum length of target sequences
        max_prefill_predict_length: Maximum length for prefill predictions
        max_pages_per_group: Maximum pages that can be allocated to a group
        num_layers: Number of model layers
        config: Model configuration object containing num_kv_heads and head_dim
    """
    self.num_pages = num_pages
    self.tokens_per_page = tokens_per_page
    self.max_page_groups = max_page_groups
    self.max_target_length = max_target_length
    self.max_prefill_predict_length = max_prefill_predict_length
    self.max_pages_per_group = max_pages_per_group
    self.num_layers = num_layers
    self.config = config

    # Initialize page states
    self.page_status = jnp.zeros((self.num_layers, self.num_pages), dtype=jnp.int32)
    self.page_map = jnp.full((self.num_layers, self.max_page_groups, self.max_pages_per_group), -1, dtype=jnp.int32)
    self.sequence_lengths = jnp.zeros((self.num_layers, self.max_page_groups), dtype=jnp.int32)
    self.num_pages_used = jnp.zeros((self.num_layers, self.max_page_groups), dtype=jnp.int32)
    self.current_page = jnp.full((self.num_layers, self.max_page_groups), -1, dtype=jnp.int32)
    self.current_page_position = jnp.zeros((self.num_layers, self.max_page_groups), dtype=jnp.int32)

    self.key_pages = jnp.zeros(
        (self.num_layers, self.num_pages, self.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
        dtype=self.config.dtype,
    )
    self.value_pages = jnp.zeros(
        (self.num_layers, self.num_pages, self.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
        dtype=self.config.dtype,
    )

    self._validate_init_params()

  def _validate_init_params(self):
    """
    Validates initialization parameters to ensure they meet requirements.

    Raises:
        ValueError: If any parameters are invalid or incompatible with each other
    """
    if self.num_pages <= 0:
      raise ValueError(f"Invalid num_pages: {self.num_pages}")
    if self.tokens_per_page <= 0:
      raise ValueError(f"Invalid tokens_per_page: {self.tokens_per_page}")
    if self.max_page_groups <= 0:
      raise ValueError(f"Invalid max_page_groups: {self.max_page_groups}")
    if self.max_pages_per_group <= 0:
      raise ValueError(f"Invalid max_pages_per_group: {self.max_pages_per_group}")

    pages_needed_for_max_target = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    if pages_needed_for_max_target > self.max_pages_per_group:
      raise ValueError(
          f"max_target_length of {self.max_target_length} would require {pages_needed_for_max_target} "
          f"pages but max_pages_per_group is {self.max_pages_per_group}"
      )

    pages_needed_for_max_prefill = (self.max_prefill_predict_length + self.tokens_per_page - 1) // self.tokens_per_page
    if pages_needed_for_max_prefill > self.max_pages_per_group:
      raise ValueError(
          f"max_prefill_predict_length of {self.max_prefill_predict_length} would require "
          f"{pages_needed_for_max_prefill} pages but max_pages_per_group is {self.max_pages_per_group}"
      )

  def __call__(
      self,
      model_mode: str,
      page_group_id: Optional[int] = None,
      true_length: Optional[int] = None,
      layer_id: Optional[int] = None,
  ) -> PageState:
    """
    Updates internal state based on the requested operation mode and parameters.

    Args:
        model_mode: Operation mode ('prefill' or 'autoregressive')
        page_group_id: ID of the page group to operate on (optional)
        true_length: Target sequence length (optional)
        layer_id: ID of the layer to operate on (optional)

    Returns:
        PageState: Updated state after the requested operation

    Raises:
        ValueError: If model_mode is invalid
    """
    if model_mode not in ["prefill", "autoregressive"]:
      raise ValueError(f"Invalid model_mode: {model_mode}")

    # Handle global state request
    if layer_id is None:
      return PageState(
          page_status=self.page_status,
          page_map=self.page_map,
          sequence_lengths=self.sequence_lengths,
          num_pages_used=self.num_pages_used,
          current_page=self.current_page,
          current_page_position=self.current_page_position,
      )

    # Get layer-specific states
    layer_page_status = self.page_status[layer_id]
    layer_page_map = self.page_map[layer_id]
    layer_sequence_lengths = self.sequence_lengths[layer_id]
    layer_pages_used = self.num_pages_used[layer_id]
    layer_current_page = self.current_page[layer_id]
    layer_current_position = self.current_page_position[layer_id]

    if model_mode == "prefill":
      if page_group_id is not None and true_length is not None:
        # Validate inputs using JAX-compatible validation
        is_valid_group = validate_page_group(page_group_id, self.max_page_groups)
        is_valid_length = validate_length(true_length, self.max_target_length)

        # Only proceed if inputs are valid
        def do_prefill(args):
          return self.reserve_prefill_page_group_pages(
              page_group_id,
              true_length,
              layer_page_status,
              layer_page_map,
              layer_sequence_lengths,
              layer_pages_used,
              layer_current_page,
              layer_current_position,
          )

        def keep_current_state(args):
          return (
              layer_page_status,
              layer_page_map,
              layer_sequence_lengths,
              layer_pages_used,
              layer_current_page,
              layer_current_position,
          )

        states = jax.lax.cond(jnp.logical_and(is_valid_group, is_valid_length), do_prefill, keep_current_state, None)

        (
            layer_page_status,
            layer_page_map,
            layer_sequence_lengths,
            layer_pages_used,
            layer_current_page,
            layer_current_position,
        ) = states

    elif model_mode == "autoregressive":
      states = self.reserve_decode_step_pages(
          layer_page_status,
          layer_page_map,
          layer_sequence_lengths,
          layer_pages_used,
          layer_current_page,
          layer_current_position,
      )
      (
          layer_page_status,
          layer_page_map,
          layer_sequence_lengths,
          layer_pages_used,
          layer_current_page,
          layer_current_position,
      ) = states

    # Update layer states
    self.page_status = self.page_status.at[layer_id].set(layer_page_status)
    self.page_map = self.page_map.at[layer_id].set(layer_page_map)
    self.sequence_lengths = self.sequence_lengths.at[layer_id].set(layer_sequence_lengths)
    self.num_pages_used = self.num_pages_used.at[layer_id].set(layer_pages_used)
    self.current_page = self.current_page.at[layer_id].set(layer_current_page)
    self.current_page_position = self.current_page_position.at[layer_id].set(layer_current_position)

    return PageState(
        page_status=layer_page_status,
        page_map=layer_page_map,
        sequence_lengths=layer_sequence_lengths,
        num_pages_used=layer_pages_used,
        current_page=layer_current_page,
        current_page_position=layer_current_position,
    )

  def find_next_free_page(self, page_status: jnp.ndarray) -> int:
    """
    Finds the index of the next available free page.

    Args:
        page_status: Array indicating which pages are allocated/free

    Returns:
        int: Index of the next free page, or -1 if no free pages are available
    """
    free_mask = page_status == 0
    next_free = jnp.argmax(free_mask)
    has_free = jnp.any(free_mask)
    return jnp.where(has_free, next_free, -1)

  def reserve_prefill_page_group_pages(
      self,
      page_group_id: int,
      true_length: int,
      page_status: jnp.ndarray,
      page_map: jnp.ndarray,
      sequence_lengths: jnp.ndarray,
      num_pages_used: jnp.ndarray,
      current_page: jnp.ndarray,
      current_page_position: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, ...]:
    """
    Reserves pages for prefill operations for a specific page group.

    Handles initial allocation of pages for a sequence, including releasing
    any previously allocated pages for the group and allocating new ones.

    Args:
        page_group_id: ID of the page group to allocate pages for
        true_length: Target sequence length
        page_status: Current page allocation status
        page_map: Current page group to page mappings
        sequence_lengths: Current sequence lengths
        num_pages_used: Current page usage counts
        current_page: Current active pages
        current_page_position: Current positions within active pages

    Returns:
        Tuple containing updated versions of all input state arrays
    """
    num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    last_page_position = (true_length - 1) % self.tokens_per_page

    # Check if we have enough free pages
    num_free_pages = jnp.sum(page_status == 0)
    has_enough_pages = num_free_pages >= num_pages_needed

    # Release existing pages
    def release_existing_pages(index, state):
      current_status, current_map = state
      page_index = current_map[page_group_id, index]
      updated_status = jnp.where(page_index >= 0, current_status.at[page_index].set(0), current_status)
      return (updated_status, current_map)

    page_status, page_map = jax.lax.fori_loop(0, self.max_pages_per_group, release_existing_pages, (page_status, page_map))

    # Reset page map for this group
    page_map = page_map.at[page_group_id].set(jnp.full(self.max_pages_per_group, -1, dtype=jnp.int32))

    # Only allocate if we have enough pages
    def do_allocation(_):
      def allocate_new_page(index, state):
        current_status, current_map = state
        next_free_page = self.find_next_free_page(current_status)
        should_allocate = jnp.logical_and(index < num_pages_needed, next_free_page >= 0)

        # Update page status and mapping if allocation should proceed
        updated_status = jnp.where(should_allocate, current_status.at[next_free_page].set(1), current_status)
        updated_map = jnp.where(should_allocate, current_map.at[page_group_id, index].set(next_free_page), current_map)
        return (updated_status, updated_map)

      new_page_status, new_page_map = jax.lax.fori_loop(
          0, self.max_pages_per_group, allocate_new_page, (page_status, page_map)
      )

      # Update tracking information
      new_sequence_lengths = sequence_lengths.at[page_group_id].set(true_length)
      new_pages_used = num_pages_used.at[page_group_id].set(num_pages_needed)

      last_page_index = jnp.where(num_pages_needed > 0, new_page_map[page_group_id, num_pages_needed - 1], -1)
      new_current_page = current_page.at[page_group_id].set(last_page_index)
      new_current_position = current_page_position.at[page_group_id].set(last_page_position)

      return (new_page_status, new_page_map, new_sequence_lengths, new_pages_used, new_current_page, new_current_position)

    def keep_current_state(_):
      return (page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position)

    # Allocate new pages only if we have enough free pages
    return jax.lax.cond(has_enough_pages, do_allocation, keep_current_state, None)

  def reserve_decode_step_pages(
      self,
      page_status: jnp.ndarray,
      page_map: jnp.ndarray,
      sequence_lengths: jnp.ndarray,
      num_pages_used: jnp.ndarray,
      current_page: jnp.ndarray,
      current_page_position: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, ...]:
      """
      Reserves pages for autoregressive decoding steps.
      
      This method handles incrementing sequence lengths and allocating new pages
      when needed for autoregressive token generation.
      
      Args:
          page_status: Current page allocation status
          page_map: Current page group to page mappings
          sequence_lengths: Current sequence lengths
          num_pages_used: Current page usage counts
          current_page: Current active pages
          current_page_position: Current positions within active pages
          
      Returns:
          Tuple containing updated versions of all input state arrays
      """
      # CRITICAL FIX: Instead of incrementing all sequence lengths, we need to be selective.
      # Start with the existing sequence lengths and only modify active groups
      new_sequence_lengths = sequence_lengths.copy()
      
      def update_page_group(group_index, state):
          current_status, current_map, current_pages, pages_used, seq_lengths, current_positions = state
          
          # Only increment and allocate pages for active groups
          # A group is active if it has already been used (pages_used > 0)
          is_active_group = pages_used[group_index] > 0
          
          # Increment sequence length only for active groups
          updated_seq_length = jnp.where(
              is_active_group,
              seq_lengths[group_index] + 1,
              seq_lengths[group_index]
          )
          seq_lengths = seq_lengths.at[group_index].set(updated_seq_length)
          
          # Calculate new position within page
          new_position = (updated_seq_length - 1) % self.tokens_per_page
          current_positions = current_positions.at[group_index].set(new_position)
          
          # Calculate pages needed for this potentially incremented length
          pages_needed = (updated_seq_length + self.tokens_per_page - 1) // self.tokens_per_page
          
          # Check if we need a new page
          needs_new_page = jnp.logical_and(
              is_active_group,
              pages_needed > pages_used[group_index]
          )
          
          # Find next free page if needed
          next_free_page = jnp.where(needs_new_page, self.find_next_free_page(current_status), -1)
          
          # Update page status (mark as allocated if needed)
          updated_status = jnp.where(
              next_free_page >= 0,
              current_status.at[next_free_page].set(1),
              current_status
          )
          
          # Update page map with new page
          updated_map = jnp.where(
              next_free_page >= 0,
              current_map.at[group_index, pages_used[group_index]].set(next_free_page),
              current_map
          )
          
          # Update current page to point to the new page
          updated_pages = jnp.where(
              next_free_page >= 0,
              current_pages.at[group_index].set(next_free_page),
              current_pages
          )
          
          # Update pages used count
          updated_used = jnp.where(
              next_free_page >= 0,
              pages_used.at[group_index].set(pages_used[group_index] + 1),
              pages_used
          )
          
          return (updated_status, updated_map, updated_pages, updated_used, seq_lengths, current_positions)

      # Process all page groups
      initial_state = (page_status, page_map, current_page, num_pages_used, new_sequence_lengths, current_page_position)
      page_status, page_map, current_page, num_pages_used, new_sequence_lengths, new_current_position = jax.lax.fori_loop(
          0, self.max_page_groups, update_page_group, initial_state
      )

      return (page_status, page_map, new_sequence_lengths, num_pages_used, current_page, new_current_position)

  def release_page_group(self, page_group_id: int):
    """
    Releases all pages associated with a given page group.

    This method deallocates all pages assigned to a specific page group across
    all layers, resetting their states to initial values.

    Args:
        page_group_id: ID of the page group whose pages should be released
    """
    is_valid = validate_page_group(page_group_id, self.max_page_groups)

    for layer_id in range(self.num_layers):
      pages_used = self.num_pages_used[layer_id, page_group_id]

      def release_page(index, state):
        current_status, current_map = state
        page_index = current_map[page_group_id, index]
        updated_status = jnp.where(page_index >= 0, current_status.at[page_index].set(0), current_status)
        return (updated_status, current_map)

      new_status, new_map = jax.lax.fori_loop(
          0, pages_used, release_page, (self.page_status[layer_id], self.page_map[layer_id])
      )

      # Reset states for this group
      new_map = new_map.at[page_group_id].set(jnp.full(self.max_pages_per_group, -1, dtype=jnp.int32))

      # Update the PageManager's state
      self.page_status = self.page_status.at[layer_id].set(new_status)
      self.page_map = self.page_map.at[layer_id].set(new_map)
      self.sequence_lengths = self.sequence_lengths.at[layer_id, page_group_id].set(0)
      self.num_pages_used = self.num_pages_used.at[layer_id, page_group_id].set(0)
      self.current_page = self.current_page.at[layer_id, page_group_id].set(-1)
      self.current_page_position = self.current_page_position.at[layer_id, page_group_id].set(0)

  def get_page_state(self):
    """Get the current page state without any updates.
    
    Returns:
        PageState: Current state of the page manager
    """
    # Simply return the current state
    return PageState(
        page_status=self.page_status,
        page_map=self.page_map,
        sequence_lengths=self.sequence_lengths,
        num_pages_used=self.num_pages_used,
        current_page=self.current_page,
        current_page_position=self.current_page_position,
    )

  def get_page_state_for_layer(self, model_mode, page_group_id, true_length, layer_id):
    """Get page state for a layer in a JAX-compatible way.
    
    This method provides a JAX-compatible alternative to the __call__ method.
    Instead of dynamically updating state, it returns the current state for a layer.
    
    Args:
        model_mode: Operation mode ('prefill' or 'autoregressive')
        page_group_id: ID of the page group to operate on
        true_length: Target sequence length 
        layer_id: ID of the layer to operate on
        
    Returns:
        PageState: Current state for the layer
    """
    # Get the page state directly without dynamic updates
    return PageState(
        page_status=self.page_status[layer_id],
        page_map=self.page_map[layer_id],
        sequence_lengths=self.sequence_lengths[layer_id],
        num_pages_used=self.num_pages_used[layer_id],
        current_page=self.current_page[layer_id],
        current_page_position=self.current_page_position[layer_id],
    )