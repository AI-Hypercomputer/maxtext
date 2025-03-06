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


def initialize_page_state(
    num_layers: int,
    num_pages: int,
    max_page_groups: int,
    max_pages_per_group: int,
) -> PageState:
    """Initializes a PageState object with default values."""
    return PageState(
        page_status=jnp.zeros((num_layers, num_pages), dtype=jnp.int32),
        page_map=jnp.full(
            (num_layers, max_page_groups, max_pages_per_group), -1, dtype=jnp.int32
        ),
        sequence_lengths=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
        num_pages_used=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
        current_page=jnp.full((num_layers, max_page_groups), -1, dtype=jnp.int32),
        current_page_position=jnp.zeros((num_layers, max_page_groups), dtype=jnp.int32),
    )


def find_next_free_page(page_status: jnp.ndarray) -> int:
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
    page_state: PageState,
    page_group_id: int,
    true_length: int,
    layer_id: int,
    tokens_per_page: int,
    max_pages_per_group: int,
    num_pages: int,
) -> PageState:
    """
    Reserves pages for prefill operations for a specific page group.

    Args:
        page_state:  The current PageState.
        page_group_id: ID of the page group to allocate pages for.
        true_length: Target sequence length.
        layer_id: The layer ID.
        tokens_per_page:  Tokens per page.
        max_pages_per_group: Maximum pages per group.

    Returns:
        PageState: Updated PageState.
    """
    layer_page_status = page_state.page_status[layer_id]
    layer_page_map = page_state.page_map[layer_id]
    layer_sequence_lengths = page_state.sequence_lengths[layer_id]
    layer_pages_used = page_state.num_pages_used[layer_id]
    layer_current_page = page_state.current_page[layer_id]
    layer_current_position = page_state.current_page_position[layer_id]

    num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
    last_page_position = (true_length - 1) % tokens_per_page

    # Check if we have enough free pages.  This is now done on the *layer* status.
    num_free_pages = jnp.sum(layer_page_status == 0)
    has_enough_pages = num_free_pages >= num_pages_needed

    # Release existing pages (functional style).
    def release_existing_pages(index, state):
      current_status, current_map = state
      page_index = current_map[page_group_id, index]
      # Only clear the status if the page_index is valid.
      updated_status = jnp.where(page_index >= 0, current_status.at[page_index].set(0), current_status)
      return (updated_status, current_map)

    layer_page_status, layer_page_map = jax.lax.fori_loop(
        0, max_pages_per_group, release_existing_pages, (layer_page_status, layer_page_map)
    )
    # Reset the entire page map for this group.  No traced values involved.
    layer_page_map = layer_page_map.at[page_group_id].set(jnp.full(max_pages_per_group, -1, dtype=jnp.int32))


    def do_allocation(_):  # Helper for lax.cond
        def allocate_new_page(index, state):
            current_status, current_map = state
            next_free_page = find_next_free_page(current_status)
            # Only allocate if needed and a page is free.
            should_allocate = jnp.logical_and(index < num_pages_needed, next_free_page >= 0)

            updated_status = jnp.where(should_allocate, current_status.at[next_free_page].set(1), current_status)
            updated_map = jnp.where(
                should_allocate, current_map.at[page_group_id, index].set(next_free_page), current_map
            )
            return (updated_status, updated_map)

        new_page_status, new_page_map = jax.lax.fori_loop(
            0, max_pages_per_group, allocate_new_page, (layer_page_status, layer_page_map)
        )

        new_sequence_lengths = layer_sequence_lengths.at[page_group_id].set(true_length)
        new_pages_used = layer_pages_used.at[page_group_id].set(num_pages_needed)

        # Determine the last page index, handling the case where no pages are needed.
        last_page_index = jnp.where(num_pages_needed > 0, new_page_map[page_group_id, num_pages_needed - 1], -1)
        new_current_page = layer_current_page.at[page_group_id].set(last_page_index)
        new_current_position = layer_current_position.at[page_group_id].set(last_page_position)

        return (new_page_status, new_page_map, new_sequence_lengths, new_pages_used, new_current_page, new_current_position)

    def keep_current_state(_):  # Helper for lax.cond
        return (
            layer_page_status,
            layer_page_map,
            layer_sequence_lengths,
            layer_pages_used,
            layer_current_page,
            layer_current_position,
        )
    
    # Allocate new pages *only* if we have enough.
    (
        layer_page_status,
        layer_page_map,
        layer_sequence_lengths,
        layer_pages_used,
        layer_current_page,
        layer_current_position,
    ) = jax.lax.cond(has_enough_pages, do_allocation, keep_current_state, None)


    # Return updated *layer* state.
    return page_state.replace(
        page_status=page_state.page_status.at[layer_id].set(layer_page_status),
        page_map=page_state.page_map.at[layer_id].set(layer_page_map),
        sequence_lengths=page_state.sequence_lengths.at[layer_id].set(layer_sequence_lengths),
        num_pages_used=page_state.num_pages_used.at[layer_id].set(layer_pages_used),
        current_page=page_state.current_page.at[layer_id].set(layer_current_page),
        current_page_position=page_state.current_page_position.at[layer_id].set(layer_current_position),
    )



def reserve_decode_step_pages(
    page_state: PageState,
    layer_id: int,
    tokens_per_page: int,
    max_page_groups: int,
) -> PageState:
    """
    Reserves pages for autoregressive decoding steps.

    Args:
        page_state: The current PageState.
        layer_id: The layer ID.
        tokens_per_page: Tokens per page.
        max_page_groups:  Maximum number of page groups.

    Returns:
        PageState: Updated PageState.
    """
    layer_page_status = page_state.page_status[layer_id]
    layer_page_map = page_state.page_map[layer_id]
    layer_sequence_lengths = page_state.sequence_lengths[layer_id]
    layer_pages_used = page_state.num_pages_used[layer_id]
    layer_current_page = page_state.current_page[layer_id]
    layer_current_position = page_state.current_page_position[layer_id]

    # Update sequence lengths.  Only increment where a page is currently allocated.
    new_sequence_lengths = layer_sequence_lengths + jnp.where(layer_current_page >= 0, 1, 0)
    new_current_position = (new_sequence_lengths - 1) % tokens_per_page
    new_pages_needed = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page


    def update_page_group(group_index, state):
        current_status, current_map, current_pages, pages_used = state
        # Check if a new page is needed, only if the group already has a page.
        needs_new_page = jnp.logical_and(
            new_pages_needed[group_index] > pages_used[group_index], current_pages[group_index] >= 0
        )
        next_free_page = jnp.where(needs_new_page, find_next_free_page(current_status), -1)

        # Use lax.cond for conditional updates (important for JAX tracing).
        def allocate_page(args):
          next_free_page, current_status, current_map, current_pages, pages_used, group_index = args
          updated_status = current_status.at[next_free_page].set(1)
          updated_map = current_map.at[group_index, pages_used[group_index]].set(next_free_page)
          updated_pages = current_pages.at[group_index].set(next_free_page)
          updated_used = pages_used.at[group_index].set(pages_used[group_index] + 1)
          return (updated_status, updated_map, updated_pages, updated_used)
        
        def no_allocation(args):
          next_free_page, current_status, current_map, current_pages, pages_used, group_index = args
          return (current_status, current_map, current_pages, pages_used)

        updated_state = jax.lax.cond(
            next_free_page >= 0,
            allocate_page,
            no_allocation,
            (next_free_page, current_status, current_map, current_pages, pages_used, group_index)
        )

        return updated_state


    layer_page_status, layer_page_map, layer_current_page, layer_pages_used = jax.lax.fori_loop(
        0, max_page_groups, update_page_group, (layer_page_status, layer_page_map, layer_current_page, layer_pages_used)
    )

    return page_state.replace(
        page_status=page_state.page_status.at[layer_id].set(layer_page_status),
        page_map=page_state.page_map.at[layer_id].set(layer_page_map),
        sequence_lengths=page_state.sequence_lengths.at[layer_id].set(new_sequence_lengths),
        num_pages_used=page_state.num_pages_used.at[layer_id].set(layer_pages_used),
        current_page=page_state.current_page.at[layer_id].set(layer_current_page),
        current_page_position=page_state.current_page_position.at[layer_id].set(new_current_position),
    )



def release_page_group(
    page_state: PageState, page_group_id: int, max_page_groups: int, max_pages_per_group: int
) -> PageState:
    """
    Releases all pages associated with a given page group.

    Args:
        page_state: The current PageState.
        page_group_id: ID of the page group to release.
        max_page_groups: Max page groups.
        max_pages_per_group:  Max pages per group.

    Returns:
        PageState: Updated PageState.
    """
    is_valid = validate_page_group(page_group_id, max_page_groups)

    def do_release(page_state):  # Helper for lax.cond
        new_page_status = page_state.page_status
        new_page_map = page_state.page_map
        new_sequence_lengths = page_state.sequence_lengths
        new_num_pages_used = page_state.num_pages_used
        new_current_page = page_state.current_page
        new_current_page_position = page_state.current_page_position

        for layer_id in range(page_state.page_status.shape[0]):  # Iterate over layers
            layer_pages_used = page_state.num_pages_used[layer_id, page_group_id]

            def release_page(index, state):
                current_status, current_map = state
                page_index = current_map[page_group_id, index]
                # Only modify if page_index is valid (>= 0)
                updated_status = jnp.where(page_index >= 0, current_status.at[page_index].set(0), current_status)
                return (updated_status, current_map)

            # Release all pages for the group in the current layer.
            new_layer_status, new_layer_map = jax.lax.fori_loop(
                0, layer_pages_used, release_page, (new_page_status[layer_id], new_page_map[layer_id])
            )

            # Reset states for this group and layer.
            new_layer_map = new_layer_map.at[page_group_id].set(
                jnp.full(max_pages_per_group, -1, dtype=jnp.int32)
            )

            new_page_status = new_page_status.at[layer_id].set(new_layer_status)
            new_page_map = new_page_map.at[layer_id].set(new_layer_map)
            new_sequence_lengths = new_sequence_lengths.at[layer_id, page_group_id].set(0)
            new_num_pages_used = new_num_pages_used.at[layer_id, page_group_id].set(0)
            new_current_page = new_current_page.at[layer_id, page_group_id].set(-1)
            new_current_page_position = new_current_page_position.at[layer_id, page_group_id].set(0)

        return page_state.replace(
            page_status=new_page_status,
            page_map=new_page_map,
            sequence_lengths=new_sequence_lengths,
            num_pages_used=new_num_pages_used,
            current_page=new_current_page,
            current_page_position=new_current_page_position,
        )

    def keep_current_state(page_state): # Helper for lax.cond.
      return page_state

    # Only release if the page_group_id is valid.
    return jax.lax.cond(is_valid, do_release, keep_current_state, page_state)


class PageManager:  # Keep this for init and config, but no state.
  """
  Manages memory page allocation, but in a stateless, functional way.  All
  state is passed in and returned.
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
      config: Any,  # Still need config for dtype, etc.
  ):
    """Initialize the PageManager (stateless)."""
    self.num_pages = num_pages
    self.tokens_per_page = tokens_per_page
    self.max_page_groups = max_page_groups
    self.max_target_length = max_target_length
    self.max_prefill_predict_length = max_prefill_predict_length
    self.max_pages_per_group = max_pages_per_group
    self.num_layers = num_layers
    self.config = config

    self._validate_init_params()
    # No state initialization here.

  def _validate_init_params(self):
    """
    Validates initialization parameters (unchanged, still checks config).
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


  def prefill(
        self,
        page_state: PageState,
        page_group_id: int,
        true_length: int,
        layer_id: int,
    ) -> PageState:
        """Wrapper for prefill, now stateless."""
        return reserve_prefill_page_group_pages(
            page_state,
            page_group_id,
            true_length,
            layer_id,
            self.tokens_per_page,
            self.max_pages_per_group,
            self.num_pages
        )

  def decode_step(
        self,
        page_state: PageState,
        layer_id: int,
    ) -> PageState:
        """Wrapper for decode_step, now stateless."""
        return reserve_decode_step_pages(
            page_state,
            layer_id,
            self.tokens_per_page,
            self.max_page_groups,
        )


  def release(
        self,
        page_state: PageState,
        page_group_id: int,
    ) -> PageState:
      """Wrapper for release, now stateless."""
      return release_page_group(
          page_state, page_group_id, self.max_page_groups, self.max_pages_per_group
      )


  def get_page_state(self) -> PageState:
    """
    Creates and returns an *initial* PageState.  This replaces the old
    `get_page_state` method, as the PageManager is no longer stateful.
    """
    return initialize_page_state(
        self.num_layers, self.num_pages, self.max_page_groups, self.max_pages_per_group
    )