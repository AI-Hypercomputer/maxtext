import jax
import jax.numpy as jnp
from flax import struct
from typing import Optional, Any, Tuple


@struct.dataclass
class PageState:
  """
  Dataclass that holds the state of pages managed by PageManager.
  """

  page_status: jnp.ndarray  # [num_layers, num_pages] | 0: free, 1: allocated
  page_map: jnp.ndarray  # [num_layers, max_page_groups, max_pages_per_group]
  sequence_lengths: jnp.ndarray  # [max_page_groups] | Current length of each sequence.
  num_pages_used: jnp.ndarray  # [num_layers, max_page_groups]
  current_page: jnp.ndarray  # [num_layers, max_page_groups]
  current_page_position: jnp.ndarray  # [num_layers, max_page_groups]


class PageManager:
  """
  Class that manages page allocation for prefill and autoregressive decoding.
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
    """Initialize the page manager."""
    self.num_pages = num_pages
    self.tokens_per_page = tokens_per_page
    self.max_page_groups = max_page_groups
    self.max_target_length = max_target_length
    self.max_prefill_predict_length = max_prefill_predict_length
    self.max_pages_per_group = max_pages_per_group
    self.num_layers = num_layers  # Store number of layers
    self.config = config

    # Initialize page states
    self.page_status = jnp.zeros(
        (
            num_layers,
            self.num_pages,
        ),
        jnp.int32,
    )
    self.page_map = jnp.full((num_layers, self.max_page_groups, self.max_pages_per_group), -1, jnp.int32)
    self.sequence_lengths = jnp.zeros((self.max_page_groups,), jnp.int32)
    self.num_pages_used = jnp.zeros(
        (
            num_layers,
            self.max_page_groups,
        ),
        jnp.int32,
    )
    self.current_page = jnp.full(
        (
            num_layers,
            self.max_page_groups,
        ),
        -1,
        jnp.int32,
    )
    self.current_page_position = jnp.zeros(
        (
            num_layers,
            self.max_page_groups,
        ),
        jnp.int32,
    )

    # Initialize key and value pages with layer dimension
    self.key_pages = jnp.zeros(
        (num_layers, self.num_pages, self.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
        dtype=self.config.dtype,
    )
    self.value_pages = jnp.zeros(
        (num_layers, self.num_pages, self.tokens_per_page, self.config.num_kv_heads, self.config.head_dim),
        dtype=self.config.dtype,
    )

  def _validate_init_params(self):
    """Validates the initialization parameters."""
    if self.num_pages <= 0:
      raise ValueError(f"Invalid num_pages: {self.num_pages}")
    if self.tokens_per_page <= 0:
      raise ValueError(f"Invalid tokens_per_page: {self.tokens_per_page}")
    if self.max_page_groups <= 0:
      raise ValueError(f"Invalid max_page_groups: {self.max_page_groups}")
    if self.max_pages_per_group <= 0:
      raise ValueError(f"Invalid max_pages_per_page_group: {self.max_pages_per_group}")

    # Ensure max_pages_per_group is large enough for both max_target_length
    # and max_prefill_predict_length
    pages_needed_for_max_target = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    if pages_needed_for_max_target > self.max_pages_per_group:
      raise ValueError(
          f"max_target_length of {self.max_target_length} would require "
          f"{pages_needed_for_max_target} pages but max_pages_per_group is {self.max_pages_per_group}"
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
    """Updates internal state and returns PageState."""
    jax.debug.print("\nPageManager call:")
    jax.debug.print("  mode: {}, group: {}, length: {}, layer: {}", model_mode, page_group_id, true_length, layer_id)
    if layer_id is not None:
      page_status = self.page_status[layer_id]
      page_map = self.page_map[layer_id]
      num_pages_used = self.num_pages_used[layer_id]
      current_page = self.current_page[layer_id]
      current_page_position = self.current_page_position[layer_id]
    else:
      return PageState(
          page_status=self.page_status,
          page_map=self.page_map,
          sequence_lengths=self.sequence_lengths,
          num_pages_used=self.num_pages_used,
          current_page=self.current_page,
          current_page_position=self.current_page_position,
      )

    if model_mode == "prefill":
      jax.debug.print("Pre-allocation status:")
      jax.debug.print("  Free pages: {}", jnp.sum(self.page_status == 0))
      jax.debug.print("  Current mapping: {}", self.page_map[page_group_id])
      if page_group_id is None or true_length is None:
        raise ValueError("Prefill mode requires both page_group_id and true_length")

      # Calculate pages needed for this sequence
      num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
      last_page_position = jnp.where(true_length > 0, (true_length - 1) % self.tokens_per_page, 0)

      # Release any existing pages for this page_group_id
      def release_via_scan(carry, page_idx_in_group):
        ps, pm = carry
        old_page = pm[page_group_id, page_idx_in_group]
        new_ps = jnp.where(old_page >= 0, ps.at[old_page].set(0), ps)
        return (new_ps, pm), None

      (page_status, page_map), _ = jax.lax.scan(
          release_via_scan, (page_status, page_map), jnp.arange(self.max_pages_per_group)
      )

      # Verify we have enough free pages
      # def raise_error(dummy):
      #     raise ValueError("No free pages available")
      # def no_op(dummy):
      #    return
      # num_free_pages = jnp.sum(page_status == 0)
      # jax.lax.cond(num_free_pages < num_pages_needed, raise_error, no_op, operand=None)

      # Invalidate entire row for this page group
      page_map = page_map.at[page_group_id].set(jnp.full((self.max_pages_per_group,), -1, jnp.int32))

      # Allocate new pages
      def allocate_loop_body(idx, carry):
        ps, pm = carry
        next_free = self.find_next_free_page(ps)

        def do_allocate(args):
          ps, pm = args
          new_ps = ps.at[next_free].set(1)  # Mark as used
          new_pm = pm.at[page_group_id, idx].set(next_free)
          return new_ps, new_pm

        ps, pm = jax.lax.cond((idx < num_pages_needed) & (next_free >= 0), do_allocate, lambda x: x, (ps, pm))
        return ps, pm

      page_status, page_map = jax.lax.fori_loop(0, self.max_pages_per_group, allocate_loop_body, (page_status, page_map))

      # Update sequence tracking
      self.sequence_lengths = self.sequence_lengths.at[page_group_id].set(true_length)
      num_pages_used = num_pages_used.at[page_group_id].set(num_pages_needed)

      # Update current page tracking
      cur_page = jnp.where(num_pages_needed > 0, page_map[page_group_id, num_pages_needed - 1], -1)
      current_page = current_page.at[page_group_id].set(cur_page)
      current_page_position = current_page_position.at[page_group_id].set(last_page_position)

      # Update layer-specific state in the PageManager
      self.page_status = self.page_status.at[layer_id].set(page_status)
      self.page_map = self.page_map.at[layer_id].set(page_map)
      self.num_pages_used = self.num_pages_used.at[layer_id].set(num_pages_used)
      self.current_page = self.current_page.at[layer_id].set(current_page)
      self.current_page_position = self.current_page_position.at[layer_id].set(current_page_position)

    elif model_mode == "autoregressive":
      # Get new state for autoregressive generation
      page_state = self.page_manager(model_mode="autoregressive", page_group_id=page_group_id)
      # Update layer-specific autoregressive state
      self.sequence_lengths = self.sequence_lengths.at[page_group_id].add(1)
      current_page_position = current_page_position.at[page_group_id].add(1)

      # If we've filled the current page, allocate a new one
      needs_new_page = current_page_position[page_group_id] >= self.tokens_per_page
      if needs_new_page:
        # Find next free page and update state
        next_free = self.find_next_free_page(page_status)
        if next_free >= 0:
          page_status = page_status.at[next_free].set(1)
          num_pages = num_pages_used[page_group_id]
          page_map = page_map.at[page_group_id, num_pages].set(next_free)
          current_page = current_page.at[page_group_id].set(next_free)
          current_page_position = current_page_position.at[page_group_id].set(0)
          num_pages_used = num_pages_used.at[page_group_id].add(1)

          # Update layer state
          self.page_status = self.page_status.at[layer_id].set(page_status)
          self.page_map = self.page_map.at[layer_id].set(page_map)
          self.current_page = self.current_page.at[layer_id].set(current_page)
          self.current_page_position = self.current_page_position.at[layer_id].set(current_page_position)
          self.num_pages_used = self.num_pages_used.at[layer_id].set(num_pages_used)

    # Return current state for this layer
    return PageState(
        page_status=page_status,
        page_map=page_map,
        sequence_lengths=self.sequence_lengths,
        num_pages_used=num_pages_used,
        current_page=current_page,
        current_page_position=current_page_position,
    )

  def get_page_state(self) -> PageState:
    """Returns a PageState object representing the current state."""
    return PageState(
        page_status=self.page_status,
        page_map=self.page_map,
        sequence_lengths=self.sequence_lengths,
        num_pages_used=self.num_pages_used,
        current_page=self.current_page,
        current_page_position=self.current_page_position,
    )

  def _validate_init_params(self):
    """Validates the initialization parameters."""
    if self.num_pages <= 0:
      raise ValueError(f"Invalid num_pages: {self.num_pages}")
    if self.tokens_per_page <= 0:
      raise ValueError(f"Invalid tokens_per_page: {self.tokens_per_page}")
    if self.max_page_groups <= 0:
      raise ValueError(f"Invalid max_page_groups: {self.max_page_groups}")
    if self.max_pages_per_group <= 0:
      raise ValueError(f"Invalid max_pages_per_page_group: {self.max_pages_per_group}")

    # Ensure max_pages_per_group is large enough for BOTH max_target_length
    # and max_prefill_predict_length.
    pages_needed_for_max_target = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    if pages_needed_for_max_target > self.max_pages_per_group:
      raise ValueError(
          f"max_target_length of {self.max_target_length} would require "
          f"{pages_needed_for_max_target} pages but max_pages_per_page_group is {self.max_pages_per_group}"
      )
    pages_needed_for_max_prefill = (self.max_prefill_predict_length + self.tokens_per_page - 1) // self.tokens_per_page
    if pages_needed_for_max_prefill > self.max_pages_per_group:
      raise ValueError(
          f"max_prefill_predict_length of {self.max_prefill_predict_length} would require "
          f"{pages_needed_for_max_prefill} pages but max_pages_per_page_group is {self.max_pages_per_group}"
      )

  def _validate_page_group(self, page_group_id: int):
    """Validates that a page_group_id is within the allowed range."""

    def raise_page_group_error(page_group_val, max_page_groups_val):
      raise ValueError(f"Invalid page_group_id: {page_group_val}. Must be in [0, {max_page_groups_val})")

    is_invalid_page_group = jnp.logical_or(page_group_id < 0, page_group_id >= self.max_page_groups)  # Use jnp comparison
    _ = jax.lax.cond(
        is_invalid_page_group,
        lambda _: jax.debug.callback(
            raise_page_group_error, page_group_id, self.max_page_groups
        ),  # Use callback to raise error
        lambda _: None,  # No-op if page_group_id is valid
        operand=None,  # Dummy operand as callback doesn't need input from cond
    )

  def _validate_length(self, length: int):
    """Validates that a sequence length is non-negative and doesn't exceed max_target_length."""

    def raise_length_error(length_val, max_length_val, reason):
      raise ValueError(f"Sequence length {length_val} {reason} {max_length_val}")

    is_negative = length < 0
    _ = jax.lax.cond(
        is_negative,
        lambda _: jax.debug.callback(
            raise_length_error, length, self.max_target_length, "is negative and must be non-negative, but got"
        ),
        lambda _: None,
        operand=None,
    )

    is_too_long = length > self.max_target_length
    _ = jax.lax.cond(
        is_too_long,
        lambda _: jax.debug.callback(raise_length_error, length, self.max_target_length, "exceeds max_target_length"),
        lambda _: None,
        operand=None,
    )

  def find_next_free_page(self, page_status: jnp.ndarray) -> int:
    """Finds the index of the next free page."""
    free_mask = page_status == 0
    next_free = jnp.argmax(free_mask)
    has_free = jnp.any(free_mask)
    return jnp.where(has_free, next_free, -1)

  def _reserve_single_page(
      self, page_group_id: int, i: int, page_status: jnp.ndarray, page_map: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reserves a single page for a given page group."""
    next_free_page = self.find_next_free_page(page_status)

    # Conditionally update page_status and page_map if a free page was found.
    page_status = jax.lax.cond(
        next_free_page >= 0,
        lambda: page_status.at[next_free_page].set(1),  # Mark page as used
        lambda: page_status,  # No-op if no free page
    )
    page_map = jax.lax.cond(next_free_page >= 0, lambda: page_map.at[page_group_id, i].set(next_free_page), lambda: page_map)
    return page_status, page_map

  def release_page_group(self, page_group_id: int):
    """Releases all pages associated with a given page group ID."""

    num_used = self.num_pages_used.value[page_group_id]

    def release_single_page(page_idx_in_group, state):
      page_status, page_map = state
      page_idx = page_map[page_group_id, page_idx_in_group]

      def clear_page():
        return page_status.at[page_idx].set(0)

      def keep_page():
        return page_status

      new_status = jax.lax.cond(page_idx >= 0, lambda: clear_page(), lambda: keep_page())
      return (new_status, page_map)

    init_state = (self.page_status.value, self.page_map.value)
    # Use jax.lax.fori_loop to iterate and release pages
    final_status, final_map = jax.lax.fori_loop(0, num_used, release_single_page, init_state)

    self.page_status.value = final_status  # Update page_status
    self.page_map.value = final_map.at[page_group_id].set(
        jnp.full(self.max_pages_per_group, -1, dtype=jnp.int32)
    )  # Reset the entire row
    self.sequence_lengths.value = self.sequence_lengths.value.at[page_group_id].set(0)
    self.num_pages_used.value = self.num_pages_used.value.at[page_group_id].set(0)
    self.current_page.value = self.current_page.value.at[page_group_id].set(-1)
    self.current_page_position.value = self.current_page_position.value.at[page_group_id].set(0)

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
    """Reserves pages for a given page group during prefill."""

    # Calculate how many pages we need for this sequence.
    num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    # Where we are within the final page.
    last_page_position = jnp.where(true_length > 0, (true_length - 1) % self.tokens_per_page, 0)

    # Release existing pages for this page_group_id.
    def release_via_scan(carry, page_idx_in_group):
      ps, pm = carry
      old_page = pm[page_group_id, page_idx_in_group]
      new_ps = jnp.where(old_page >= 0, ps.at[old_page].set(0), ps)  # set to free
      return (new_ps, pm), None

    (page_status, page_map), _ = jax.lax.scan(
        release_via_scan, (page_status, page_map), jnp.arange(self.max_pages_per_group)
    )

    # Compute the free pages count *after* releasing.
    num_free_pages = jnp.sum(page_status == 0)
    jax.debug.print("num_free_pages: {}, num_pages_needed: {}", num_free_pages, num_pages_needed)

    # TODO:  This should probably be a panic/abort, not just a ValueError.  We
    # need to signal to the caller that the request cannot be fulfilled.
    def raise_value_error():
      raise ValueError("No free pages available")

    _ = jax.lax.cond(
        num_free_pages < num_pages_needed,
        lambda: jax.debug.callback(raise_value_error),  # Use jax.debug.callback for errors
        lambda: None,  # No-op if enough pages
    )

    page_map = page_map.at[page_group_id].set(
        jnp.full((self.max_pages_per_group,), -1, jnp.int32)
    )  # Invalidate the entire row

    # Allocate new pages using a fori_loop.
    def allocate_loop_body(idx, carry):
      ps, pm = carry
      next_free = self.find_next_free_page(ps)

      def do_allocate(args):
        ps, pm = args
        new_ps = ps.at[next_free].set(1)  # Mark as used
        new_pm = pm.at[page_group_id, idx].set(next_free)
        return new_ps, new_pm

      ps, pm = jax.lax.cond(
          (idx < num_pages_needed) & (next_free >= 0),  # Only allocate if needed AND a page is free.
          do_allocate,
          lambda x: x,  # No-op
          (ps, pm),
      )
      return ps, pm

    page_status, page_map = jax.lax.fori_loop(0, self.max_pages_per_group, allocate_loop_body, (page_status, page_map))

    # Update sequence tracking.
    sequence_lengths = sequence_lengths.at[page_group_id].set(true_length)
    num_pages_used = num_pages_used.at[page_group_id].set(num_pages_needed)

    # Update current page tracking.
    cur_page = jnp.where(num_pages_needed > 0, page_map[page_group_id, num_pages_needed - 1], -1)
    current_page = current_page.at[page_group_id].set(cur_page)
    current_page_position = current_page_position.at[page_group_id].set(last_page_position)

    return (page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position)

  def _reserve_single_page_group(
      self,
      page_group_id: int,
      page_status: jnp.ndarray,
      page_map: jnp.ndarray,
      current_page: jnp.ndarray,
      num_pages_used: jnp.ndarray,
      new_num_pages_used: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Reserves a single page during autoregressive decoding. It returns
    updated states.
    """

    def update_page(args):
      # Updates if a free page is available
      page_status, page_map, current_page, num_pages_used, next_free_page, page_group_id, num_pages_in_group = args
      page_status = page_status.at[next_free_page].set(1)  # Mark the page as used
      page_map = page_map.at[page_group_id, num_pages_in_group].set(next_free_page)  # Update map
      current_page = current_page.at[page_group_id].set(next_free_page)  # update current page
      num_pages_used = num_pages_used.at[page_group_id].set(num_pages_in_group + 1)  # increase number of pages
      return page_status, page_map, current_page, num_pages_used

    # Get the next free page
    next_free_page = self.find_next_free_page(page_status)

    # How many pages does this group currently have allocated?
    num_pages_in_group = num_pages_used[page_group_id]

    # Conditionally update:
    # 1.  We need a new page (new_num_pages_used > num_pages_used)
    # 2.  A free page is available (next_free_page >= 0)
    page_status, page_map, current_page, num_pages_used = jax.lax.cond(
        jnp.logical_and(new_num_pages_used[page_group_id] > num_pages_used[page_group_id], next_free_page >= 0),
        update_page,  # Apply if true
        lambda args: args[:4],  # Return same values if false
        (page_status, page_map, current_page, num_pages_used, next_free_page, page_group_id, num_pages_in_group),
    )
    return page_status, page_map, current_page, num_pages_used

  def reserve_decode_step_pages(
      self, page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reserves pages for autoregressive decoding for all sequences."""

    # Increment sequence lengths.  We only do this for sequences that
    # have a valid current_page (>= 0).  Sequences that are finished
    # or haven't started yet have current_page = -1.
    new_sequence_lengths = sequence_lengths + jnp.where(current_page >= 0, 1, 0)  # Add to length

    # Calculate how many pages would be used based on new length
    base_new_num_pages_used = (new_sequence_lengths + self.tokens_per_page - 1) // self.tokens_per_page
    # If our current position is at the end of the page, we require a new page
    new_num_pages_used = jnp.where(
        (new_sequence_lengths > 0) & (new_sequence_lengths % self.tokens_per_page == 0),
        base_new_num_pages_used,
        base_new_num_pages_used,
    )

    # Calculate where we are within the page.
    new_current_page_position = jnp.where(new_sequence_lengths == 0, 0, (new_sequence_lengths - 1) % self.tokens_per_page)

    def scan_body(carry, page_group_id):
      page_status, page_map, current_page, num_pages_used = carry
      page_status, page_map, current_page, num_pages_used = self._reserve_single_page_group(
          page_group_id, page_status, page_map, current_page, num_pages_used, new_num_pages_used
      )
      return (page_status, page_map, current_page, num_pages_used), None

    # Use jax.lax.scan to efficiently loop over all page groups.
    (page_status, page_map, current_page, num_pages_used), _ = jax.lax.scan(
        scan_body,
        (page_status, page_map, current_page, num_pages_used),  # Initial carry
        jnp.arange(self.max_page_groups),  # Loop from 0 to max_page_groups - 1
    )

    return page_status, page_map, new_sequence_lengths, num_pages_used, current_page, new_current_page_position

  def get_page_state(self) -> PageState:
    """Returns a PageState object representing the current state."""

    # Use jax.eval_shape to get the *shapes* of the initial arrays,
    # even if we're in a JIT context where we can't directly access
    # the values. This is important for AOT compilation.

    def _get_page_state_shape(name, init_fn, *args):
      return jax.eval_shape(lambda: init_fn(*args))

    return PageState(
        page_status=_get_page_state_shape("page_status", jnp.zeros, (self.num_pages,), jnp.int32),
        page_map=_get_page_state_shape(
            "page_map", jnp.full, (self.max_page_groups, self.max_pages_per_group), -1, jnp.int32
        ),
        sequence_lengths=_get_page_state_shape("sequence_lengths", jnp.zeros, (self.max_page_groups,), jnp.int32),
        num_pages_used=_get_page_state_shape("num_pages_used", jnp.zeros, (self.max_page_groups,), jnp.int32),
        current_page=_get_page_state_shape("current_page", jnp.full, (self.max_page_groups,), -1, jnp.int32),
        current_page_position=_get_page_state_shape("current_page_position", jnp.zeros, (self.max_page_groups,), jnp.int32),
    )
