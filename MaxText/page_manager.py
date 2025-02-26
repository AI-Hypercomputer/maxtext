import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional, Any, Tuple


@struct.dataclass
class PageState:
  """JAX-compatible immutable state container for paged attention."""

  # Core page allocation state
  page_status: jnp.ndarray  # [num_layers, num_pages] | 0: free, 1: allocated
  page_map: jnp.ndarray  # [num_layers, max_page_groups, max_pages_per_group]
  sequence_lengths: jnp.ndarray  # [num_layers, max_page_groups]
  num_pages_used: jnp.ndarray  # [num_layers, max_page_groups]
  current_page: jnp.ndarray  # [num_layers, max_page_groups]
  current_page_position: jnp.ndarray  # [num_layers, max_page_groups]

  # Key-value pages for all layers as arrays (not dictionaries)
  # Shape: [num_layers, num_pages, tokens_per_page, num_kv_heads, head_dim]
  key_pages: jnp.ndarray
  value_pages: jnp.ndarray


class PageManager:
  """JAX-compatible page manager for attention mechanisms.

  This class manages the allocation and use of pages for key-value caching
  in transformer attentions.
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
    """Initialize the page manager with the given configuration."""
    self.num_pages = num_pages
    self.tokens_per_page = tokens_per_page
    self.max_page_groups = max_page_groups
    self.max_target_length = max_target_length
    self.max_prefill_predict_length = max_prefill_predict_length
    self.max_pages_per_group = max_pages_per_group
    self.num_layers = num_layers
    self.config = config

    # Initialize arrays with appropriate shapes
    self._initial_page_status = jnp.zeros((self.num_layers, self.num_pages), dtype=jnp.int32)
    self._initial_page_map = jnp.full((self.num_layers, self.max_page_groups, self.max_pages_per_group), -1, dtype=jnp.int32)
    self._initial_sequence_lengths = jnp.zeros((self.num_layers, self.max_page_groups), dtype=jnp.int32)
    self._initial_num_pages_used = jnp.zeros((self.num_layers, self.max_page_groups), dtype=jnp.int32)
    self._initial_current_page = jnp.full((self.num_layers, self.max_page_groups), -1, dtype=jnp.int32)
    self._initial_current_page_position = jnp.zeros((self.num_layers, self.max_page_groups), dtype=jnp.int32)

    # Initialize key and value pages as arrays
    # Shape: [num_layers, num_pages, tokens_per_page, num_kv_heads, head_dim]
    self._initial_key_pages = jnp.zeros(
        (self.num_layers, self.num_pages, self.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
        dtype=self.config.dtype,
    )
    self._initial_value_pages = jnp.zeros(
        (self.num_layers, self.num_pages, self.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
        dtype=self.config.dtype,
    )

    # Create initial state
    self._initial_state = PageState(
        page_status=self._initial_page_status,
        page_map=self._initial_page_map,
        sequence_lengths=self._initial_sequence_lengths,
        num_pages_used=self._initial_num_pages_used,
        current_page=self._initial_current_page,
        current_page_position=self._initial_current_page_position,
        key_pages=self._initial_key_pages,
        value_pages=self._initial_value_pages,
    )

    self._validate_init_params()

  def _validate_init_params(self):
    """Validate initialization parameters."""
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

  def get_initial_state(self) -> PageState:
    """Returns the initial page state."""
    return self._initial_state

  def find_next_free_page(self, page_status: jnp.ndarray) -> int:
    """Find the index of the next free page."""
    free_mask = page_status == 0
    next_free = jnp.argmax(free_mask)
    has_free = jnp.any(free_mask)
    return jnp.where(has_free, next_free, -1)

  def reserve_prefill_pages(
      self,
      state: PageState,
      page_group_id: int,
      true_length: int,
      layer_id: int,
  ) -> PageState:
    """Reserve pages for prefill operations."""

    # Extract layer-specific states
    layer_page_status = state.page_status[layer_id]
    layer_page_map = state.page_map[layer_id]
    layer_sequence_lengths = state.sequence_lengths[layer_id]
    layer_pages_used = state.num_pages_used[layer_id]
    layer_current_page = state.current_page[layer_id]
    layer_current_position = state.current_page_position[layer_id]

    # Calculate needed pages
    num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    last_page_position = (true_length - 1) % self.tokens_per_page

    # Check if we have enough free pages
    num_free_pages = jnp.sum(layer_page_status == 0)
    has_enough_pages = num_free_pages >= num_pages_needed

    # Release previously allocated pages using JAX loop
    def release_existing_pages(i, status):
      page_idx = layer_page_map[page_group_id, i]
      valid_page = page_idx >= 0
      return jnp.where(valid_page, status.at[page_idx].set(0), status)

    # Use JAX's fori_loop for page release
    new_page_status = jax.lax.fori_loop(0, self.max_pages_per_group, release_existing_pages, layer_page_status)

    # Reset page map for this group
    new_page_map = layer_page_map.at[page_group_id].set(jnp.full(self.max_pages_per_group, -1, dtype=jnp.int32))

    def allocate_pages(_):
      # Initialize counters outside of the loop
      loop_status = new_page_status
      loop_map = new_page_map

      # Define a function for a single loop iteration
      def allocate_page(i, state):
        curr_status, curr_map = state
        # Find next free page
        free_mask = curr_status == 0
        next_free = jnp.argmax(free_mask)
        has_free = jnp.any(free_mask)
        next_free = jnp.where(has_free, next_free, -1)

        # Only allocate if within needed pages and a free page exists
        should_allocate = jnp.logical_and(i < num_pages_needed, next_free >= 0)

        # Update status and map when allocation is valid
        new_status = jnp.where(should_allocate, curr_status.at[next_free].set(1), curr_status)
        new_map = jnp.where(should_allocate, curr_map.at[page_group_id, i].set(next_free), curr_map)

        return new_status, new_map

      # Run the loop
      final_status, final_map = jax.lax.fori_loop(0, self.max_pages_per_group, allocate_page, (loop_status, loop_map))

      # Update tracking information
      new_seq_length = layer_sequence_lengths.at[page_group_id].set(true_length)
      new_pages_used = layer_pages_used.at[page_group_id].set(num_pages_needed)

      # Get last allocated page index
      last_page_idx_pos = jnp.maximum(0, num_pages_needed - 1)
      last_page_idx = final_map[page_group_id, last_page_idx_pos]

      # Handle the case where no pages were allocated
      page_allocated = num_pages_needed > 0
      new_current_page = jnp.where(
          page_allocated, layer_current_page.at[page_group_id].set(last_page_idx), layer_current_page
      )
      new_current_pos = jnp.where(
          page_allocated, layer_current_position.at[page_group_id].set(last_page_position), layer_current_position
      )

      return final_status, final_map, new_seq_length, new_pages_used, new_current_page, new_current_pos

    # Define function to keep current state
    def keep_current_state(_):
      return (
          new_page_status,
          new_page_map,
          layer_sequence_lengths,
          layer_pages_used,
          layer_current_page,
          layer_current_position,
      )

    # Conditionally allocate pages if we have enough
    updated_values = jax.lax.cond(has_enough_pages, allocate_pages, keep_current_state, operand=None)

    # Unpack updated values
    (updated_status, updated_map, updated_seq_lengths, updated_pages_used, updated_current_page, updated_current_pos) = (
        updated_values
    )

    # Create new state with updates
    new_page_status = state.page_status.at[layer_id].set(updated_status)
    new_page_map = state.page_map.at[layer_id].set(updated_map)
    new_sequence_lengths = state.sequence_lengths.at[layer_id].set(updated_seq_lengths)
    new_num_pages_used = state.num_pages_used.at[layer_id].set(updated_pages_used)
    new_current_page = state.current_page.at[layer_id].set(updated_current_page)
    new_current_page_position = state.current_page_position.at[layer_id].set(updated_current_pos)

    # Return updated state
    return PageState(
        page_status=new_page_status,
        page_map=new_page_map,
        sequence_lengths=new_sequence_lengths,
        num_pages_used=new_num_pages_used,
        current_page=new_current_page,
        current_page_position=new_current_page_position,
        key_pages=state.key_pages,
        value_pages=state.value_pages,
    )

  def reserve_autoregressive_pages(
      self,
      state: PageState,
      page_group_id: int,
      layer_id: int,
  ) -> PageState:
    """Reserve pages for autoregressive decoding steps."""

    # Extract layer-specific states
    layer_page_status = state.page_status[layer_id]
    layer_page_map = state.page_map[layer_id]
    layer_sequence_lengths = state.sequence_lengths[layer_id]
    layer_pages_used = state.num_pages_used[layer_id]
    layer_current_page = state.current_page[layer_id]
    layer_current_position = state.current_page_position[layer_id]

    # Increment sequence length if current page is valid
    valid_current_page = layer_current_page[page_group_id] >= 0
    new_sequence_length = jnp.where(
        valid_current_page, layer_sequence_lengths[page_group_id] + 1, layer_sequence_lengths[page_group_id]
    )

    # Calculate new position within page
    new_position = (new_sequence_length - 1) % self.tokens_per_page

    # Calculate pages needed for the new sequence length
    new_pages_needed = (new_sequence_length + self.tokens_per_page - 1) // self.tokens_per_page

    # Check if we need a new page
    needs_new_page = jnp.logical_and(new_pages_needed > layer_pages_used[page_group_id], valid_current_page)

    # Find next free page if needed
    free_mask = layer_page_status == 0
    next_free_page = jnp.argmax(free_mask)
    has_free = jnp.any(free_mask)
    next_free_page = jnp.where(has_free, next_free_page, -1)

    # Only proceed with allocation if we found a free page
    valid_allocation = next_free_page >= 0
    should_allocate = jnp.logical_and(needs_new_page, valid_allocation)

    # Update page status
    new_page_status = jnp.where(should_allocate, layer_page_status.at[next_free_page].set(1), layer_page_status)

    # Update page map
    current_pages_used = layer_pages_used[page_group_id]
    new_page_map = jnp.where(
        should_allocate, layer_page_map.at[page_group_id, current_pages_used].set(next_free_page), layer_page_map
    )

    # Update current page
    new_current_page = jnp.where(
        should_allocate, layer_current_page.at[page_group_id].set(next_free_page), layer_current_page
    )

    # Update pages used
    new_pages_used = jnp.where(
        should_allocate, layer_pages_used.at[page_group_id].set(current_pages_used + 1), layer_pages_used
    )

    # Update current page position for all cases
    new_current_position = layer_current_position.at[page_group_id].set(new_position)

    # Create new state with updates
    new_page_status_all = state.page_status.at[layer_id].set(new_page_status)
    new_page_map_all = state.page_map.at[layer_id].set(new_page_map)
    new_sequence_lengths_all = state.sequence_lengths.at[layer_id, page_group_id].set(new_sequence_length)
    new_num_pages_used_all = state.num_pages_used.at[layer_id].set(new_pages_used)
    new_current_page_all = state.current_page.at[layer_id].set(new_current_page)
    new_current_page_position_all = state.current_page_position.at[layer_id, page_group_id].set(new_position)

    # Return updated state
    return PageState(
        page_status=new_page_status_all,
        page_map=new_page_map_all,
        sequence_lengths=new_sequence_lengths_all,
        num_pages_used=new_num_pages_used_all,
        current_page=new_current_page_all,
        current_page_position=new_current_page_position_all,
        key_pages=state.key_pages,
        value_pages=state.value_pages,
    )

  def update_token_key_value(
      self,
      state: PageState,
      page_group_id: int,
      layer_id: int,
      key_proj: jnp.ndarray,
      value_proj: jnp.ndarray,
  ) -> PageState:
    """Update KV pages with token projectionss."""

    # Get current state
    current_page = state.current_page[layer_id, page_group_id]
    current_position = state.current_page_position[layer_id, page_group_id]

    # Check if we have a valid page
    valid_page = current_page >= 0

    def do_update(_):
      # Update key and value pages
      new_key_pages = state.key_pages.at[layer_id, current_page, current_position].set(key_proj)
      new_value_pages = state.value_pages.at[layer_id, current_page, current_position].set(value_proj)
      return new_key_pages, new_value_pages

    def keep_current(_):
      return state.key_pages, state.value_pages

    new_key_pages, new_value_pages = jax.lax.cond(valid_page, do_update, keep_current, operand=None)

    # Create new state with updates
    return PageState(
        page_status=state.page_status,
        page_map=state.page_map,
        sequence_lengths=state.sequence_lengths,
        num_pages_used=state.num_pages_used,
        current_page=state.current_page,
        current_page_position=state.current_page_position,
        key_pages=new_key_pages,
        value_pages=new_value_pages,
    )

  def release_page_group(
      self,
      state: PageState,
      page_group_id: int,
  ) -> PageState:
    """Release all pages associated with a group using JAX-compatible operations."""

    # Check if page group ID is valid
    is_valid = jnp.logical_and(page_group_id >= 0, page_group_id < self.max_page_groups)

    # Initialize the result as the current state
    result_state = state

    # Release pages for one layer
    def process_layer(layer_id, curr_state):
      # Get current values
      layer_page_status = curr_state.page_status[layer_id]
      layer_page_map = curr_state.page_map[layer_id]
      layer_pages_used = curr_state.num_pages_used[layer_id, page_group_id]

      # Release a single page
      def release_page(i, status):
        page_idx = layer_page_map[page_group_id, i]
        valid_page = page_idx >= 0
        return jnp.where(valid_page, status.at[page_idx].set(0), status)

      # Release all pages in a loop
      new_status = jax.lax.fori_loop(0, layer_pages_used, release_page, layer_page_status)

      # Create reset map for this group
      new_map = layer_page_map.at[page_group_id].set(jnp.full(self.max_pages_per_group, -1, dtype=jnp.int32))

      # Update all state components
      updated_status = curr_state.page_status.at[layer_id].set(new_status)
      updated_map = curr_state.page_map.at[layer_id].set(new_map)
      updated_seq_lengths = curr_state.sequence_lengths.at[layer_id, page_group_id].set(0)
      updated_pages_used = curr_state.num_pages_used.at[layer_id, page_group_id].set(0)
      updated_current_page = curr_state.current_page.at[layer_id, page_group_id].set(-1)
      updated_current_pos = curr_state.current_page_position.at[layer_id, page_group_id].set(0)

      # Create new state with updates
      return PageState(
          page_status=updated_status,
          page_map=updated_map,
          sequence_lengths=updated_seq_lengths,
          num_pages_used=updated_pages_used,
          current_page=updated_current_page,
          current_page_position=updated_current_pos,
          key_pages=curr_state.key_pages,
          value_pages=curr_state.value_pages,
      )

    # Process each layer
    def scan_process_layers(init_state, _):
      state = init_state
      for i in range(self.num_layers):
        state = process_layer(i, state)
      return state, None

    # Only release if page group is valid
    def do_release(_):
      return scan_process_layers(result_state, None)[0]

    def skip_release(_):
      return result_state

    final_state = jax.lax.cond(is_valid, do_release, skip_release, operand=None)

    return final_state

  def __call__(self, model_mode="prefill", page_group_id=0, true_length=0, layer_id=0):
    """Return the page state for a specific layer and group ID (trace-friendly access).
    This is a trace-friendly way to access page states that plays nicely with JAX.
    """
    return self._initial_state
