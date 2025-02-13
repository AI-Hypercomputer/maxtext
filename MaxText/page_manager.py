from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any


@struct.dataclass
class PageState:
    """
    Dataclass that holds the state of pages managed by PageManager.
    """
    page_status: jnp.ndarray
    page_map: jnp.ndarray
    sequence_lengths: jnp.ndarray
    num_pages_used: jnp.ndarray
    current_page: jnp.ndarray
    current_page_position: jnp.ndarray

class PageManager(nn.Module):
    """
    Module that manages page allocation for prefix and autoregressive decoding.
    """
    num_pages: int
    tokens_per_page: int
    slots: int
    max_target_length: int
    max_prefill_predict_length: int
    max_pages_per_slot: int

    def setup(self):
        self._validate_init_params()
        # Use plain arrays (no logical partitioning for simplicity).
        self.page_status = self.variable("cache", "page_status",
                                          lambda: jnp.zeros((self.num_pages,), jnp.int32))
        self.page_map = self.variable("cache", "page_map",
                                       lambda: jnp.full((self.slots, self.max_pages_per_slot), -1, jnp.int32))
        self.sequence_lengths = self.variable("cache", "sequence_lengths",
                                               lambda: jnp.zeros((self.slots,), jnp.int32))
        self.num_pages_used = self.variable("cache", "num_pages_used",
                                             lambda: jnp.zeros((self.slots,), jnp.int32))
        self.current_page = self.variable("cache", "current_page",
                                           lambda: jnp.full((self.slots,), -1, jnp.int32))
        self.current_page_position = self.variable("cache", "current_page_position",
                                                    lambda: jnp.zeros((self.slots,), jnp.int32))

    def _validate_init_params(self):
        if self.num_pages <= 0:
            raise ValueError(f"Invalid num_pages: {self.num_pages}")
        if self.tokens_per_page <= 0:
            raise ValueError(f"Invalid tokens_per_page: {self.tokens_per_page}")
        if self.slots <= 0:
            raise ValueError(f"Invalid slots: {self.slots}")
        if self.max_pages_per_slot <= 0:
            raise ValueError(f"Invalid max_pages_per_slot: {self.max_pages_per_slot}")

        pages_needed_for_max_target = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
        if pages_needed_for_max_target > self.max_pages_per_slot:
            raise ValueError(
                f"max_target_length of {self.max_target_length} would require "
                f"{pages_needed_for_max_target} pages but max_pages_per_slot is {self.max_pages_per_slot}"
            )
        pages_needed_for_max_prefill = (self.max_prefill_predict_length + self.tokens_per_page - 1) // self.tokens_per_page
        if pages_needed_for_max_prefill > self.max_pages_per_slot:
            raise ValueError(
                f"max_prefill_predict_length of {self.max_prefill_predict_length} would require "
                f"{pages_needed_for_max_prefill} pages but max_pages_per_slot is {self.max_pages_per_slot}"
            )

    def _validate_slot(self, slot: int):
        def raise_slot_error(slot_val, slots_val):
            raise ValueError(f"Invalid slot: {slot_val}. Must be in [0, {slots_val})")

        is_invalid_slot = jnp.logical_or(slot < 0, slot >= self.slots) # Use jnp comparison
        _ = jax.lax.cond(
            is_invalid_slot,
            lambda _: jax.debug.callback(raise_slot_error, slot, self.slots), # Use callback to raise error
            lambda _: None, # No-op if slot is valid
            operand=None # Dummy operand as callback doesn't need input from cond
        )

    def _validate_length(self, length: int):
        def raise_length_error(length_val, max_length_val, reason):
            raise ValueError(f"Sequence length {length_val} {reason} {max_length_val}")

        is_negative = length < 0 # Use jnp comparison (though it will be promoted)
        _ = jax.lax.cond(
            is_negative,
            lambda _: jax.debug.callback(raise_length_error, length, self.max_target_length, "is negative and must be non-negative, but got"), # Use callback
            lambda _: None,
            operand=None
        )

        is_too_long = length > self.max_target_length # Use jnp comparison (though it will be promoted)
        _ = jax.lax.cond(
            is_too_long,
            lambda _: jax.debug.callback(raise_length_error, length, self.max_target_length, "exceeds max_target_length"), # Use callback
            lambda _: None,
            operand=None
        )

    def find_next_free_page(self, page_status: jnp.ndarray) -> int:
        """Find next free page using pure JAX operations"""
        # Create mask of free pages (where status is 0)
        free_mask = page_status == 0

        # Find first free page using argmax on the mask
        # Note: argmax on boolean mask will give first True position
        next_free = jnp.argmax(free_mask)

        # Check if we actually found a free page
        has_free = jnp.any(free_mask)

        return jnp.where(has_free, next_free, -1)

    def _reserve_single_page(self, slot: int, i: int,
                             page_status: jnp.ndarray, page_map: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_free_page = self.find_next_free_page(page_status)
        page_status = jax.lax.cond(
            next_free_page >= 0,
            lambda: page_status.at[next_free_page].set(1),
            lambda: page_status
        )
        page_map = jax.lax.cond(
            next_free_page >= 0,
            lambda: page_map.at[slot, i].set(next_free_page),
            lambda: page_map
        )
        return page_status, page_map

    def release_slot_pages(self, slot: int):
        """Completely revised release_slot_pages using pure JAX operations"""
        num_used = self.num_pages_used.value[slot]

        def release_single_page(i, state):
            page_status, page_map = state
            page_idx = page_map[slot, i]

            def clear_page():
                return page_status.at[page_idx].set(0)

            def keep_page():
                return page_status

            new_status = jax.lax.cond(
                page_idx >= 0,
                lambda: clear_page(),
                lambda: keep_page()
            )
            return (new_status, page_map)

        # Release pages using fori_loop
        init_state = (self.page_status.value, self.page_map.value)
        final_status, final_map = jax.lax.fori_loop(
            0,
            num_used,
            release_single_page,
            init_state
        )

        # Update all state variables atomically
        self.page_status.value = final_status
        self.page_map.value = final_map.at[slot].set(
            jnp.full(self.max_pages_per_slot, -1, dtype=jnp.int32)
        )
        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(0)
        self.num_pages_used.value = self.num_pages_used.value.at[slot].set(0)
        self.current_page.value = self.current_page.value.at[slot].set(-1)
        self.current_page_position.value = self.current_page_position.value.at[slot].set(0)


    def reserve_prefix_slot_pages(self, slot: int, true_length: int, page_status: jnp.ndarray,
                                page_map: jnp.ndarray, sequence_lengths: jnp.ndarray,
                                num_pages_used: jnp.ndarray, current_page: jnp.ndarray,
                                current_page_position: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """Revised prefix slot pages reservation WITHOUT host callback."""
        # Calculate needed pages.
        num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
        last_page_position = jnp.where(
            true_length > 0,
            (true_length - 1) % self.tokens_per_page,
            0
        )

        # Release existing pages for this slot.
        def release_scan(carry, idx):
            ps, pm = carry
            old_page = pm[slot, idx]
            new_ps = jnp.where(old_page >= 0,
                            ps.at[old_page].set(0),
                            ps)
            return (new_ps, pm), None

        (page_status, page_map), _ = jax.lax.scan(
            release_scan,
            (page_status, page_map),
            jnp.arange(self.max_pages_per_slot)
        )

        # Compute the free pages count after releasing.
        num_free_pages = jnp.sum(page_status == 0)
        jax.debug.print("num_free_pages: {}, num_pages_needed: {}", num_free_pages, num_pages_needed)

        # TODO: we may not want to error out in the future
        def raise_value_error():
            raise ValueError("No free pages available")
        _ = jax.lax.cond(
            num_free_pages < num_pages_needed,
            lambda: jax.debug.callback(raise_value_error),
            lambda: None,
         )

        page_map = page_map.at[slot].set(jnp.full((self.max_pages_per_slot,), -1, jnp.int32))

        # Allocate new pages using a fori_loop.
        def allocate_loop_body(idx, carry):
            ps, pm = carry
            next_free = self.find_next_free_page(ps)

            def do_allocate(args):
                ps, pm = args
                new_ps = ps.at[next_free].set(1)
                new_pm = pm.at[slot, idx].set(next_free)
                return new_ps, new_pm

            ps, pm = jax.lax.cond(
                (idx < num_pages_needed) & (next_free >= 0),
                do_allocate,
                lambda x: x,
                (ps, pm)
            )
            return ps, pm

        page_status, page_map = jax.lax.fori_loop(
            0, self.max_pages_per_slot, allocate_loop_body, (page_status, page_map)
        )

        # Update sequence tracking.
        sequence_lengths = sequence_lengths.at[slot].set(true_length)
        num_pages_used = num_pages_used.at[slot].set(num_pages_needed)

        # Update current page tracking.
        cur_page = jnp.where(num_pages_needed > 0,
                            page_map[slot, num_pages_needed - 1],
                            -1)
        current_page = current_page.at[slot].set(cur_page)
        current_page_position = current_page_position.at[slot].set(last_page_position)

        return (page_status, page_map, sequence_lengths, num_pages_used,
                current_page, current_page_position)


    def _reserve_single_slot(self, i: int,
                             page_status: jnp.ndarray, page_map: jnp.ndarray,
                             current_page: jnp.ndarray, num_pages_used: jnp.ndarray,
                             new_num_pages_used: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        def update_page(args):
          page_status, page_map, current_page, num_pages_used, next_free_page, i, num_pages_used_i = args
          page_status = page_status.at[next_free_page].set(1)
          page_map = page_map.at[i, num_pages_used_i].set(next_free_page)
          current_page = current_page.at[i].set(next_free_page)
          num_pages_used = num_pages_used.at[i].set(num_pages_used_i + 1)
          return page_status, page_map, current_page, num_pages_used

        next_free_page = self.find_next_free_page(page_status)
        page_status, page_map, current_page, num_pages_used = jax.lax.cond(
          jnp.logical_and(new_num_pages_used[i] > num_pages_used[i], next_free_page >= 0),
          update_page,
          lambda args: args[:4],
          (page_status, page_map, current_page, num_pages_used, next_free_page, i, num_pages_used[i])
        )
        return page_status, page_map, current_page, num_pages_used

    def reserve_decode_step_pages(self, page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        new_sequence_lengths = sequence_lengths + jnp.where(current_page >= 0, 1, 0)
        base_new_num_pages_used = (new_sequence_lengths + self.tokens_per_page - 1) // self.tokens_per_page
        new_num_pages_used = jnp.where(
            (new_sequence_lengths > 0) & (new_sequence_lengths % self.tokens_per_page == 0),
            base_new_num_pages_used,
            base_new_num_pages_used
        )

        new_current_page_position = jnp.where(new_sequence_lengths == 0, 0, (new_sequence_lengths - 1) % self.tokens_per_page)

        def scan_body(carry, i):
            page_status, page_map, current_page, num_pages_used = carry
            page_status, page_map, current_page, num_pages_used = self._reserve_single_slot(
                i, page_status, page_map, current_page, num_pages_used, new_num_pages_used
            )
            return (page_status, page_map, current_page, num_pages_used), None

        (page_status, page_map, current_page, num_pages_used), _ = jax.lax.scan(
            scan_body,
            (page_status, page_map, current_page, num_pages_used),
            jnp.arange(self.slots)
        )

        return page_status, page_map, new_sequence_lengths, num_pages_used, current_page, new_current_page_position

    def get_page_state(self) -> PageState:
      return PageState(
          page_status=self.page_status.value,
          page_map=self.page_map.value,
          sequence_lengths=self.sequence_lengths.value,
          num_pages_used=self.num_pages_used.value,
          current_page=self.current_page.value,
          current_page_position=self.current_page_position.value
      )

    def __call__(self, model_mode: Optional[str] = None, slot: Optional[int] = None,
                 true_length: Optional[int] = None) -> Optional[PageState]:

        if self.is_mutable_collection("params"):
            return None

        if model_mode is None:
          return self.get_page_state()

        if model_mode not in ["prefill", "autoregressive"]:
            raise ValueError(f"Invalid model_mode: {model_mode}")

        # Get the current state *before* any modifications.
        page_status = self.page_status.value
        page_map = self.page_map.value
        sequence_lengths = self.sequence_lengths.value
        num_pages_used = self.num_pages_used.value
        current_page = self.current_page.value
        current_page_position = self.current_page_position.value

        if model_mode == "prefill":
            if slot is None or true_length is None:
                raise ValueError("Prefill mode requires both slot and true_length")
            self._validate_slot(slot)
            self._validate_length(true_length)

            # Now perform allocation.
            page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position = (
                self.reserve_prefix_slot_pages(
                    slot, true_length, page_status, page_map,
                    sequence_lengths, num_pages_used, current_page, current_page_position
                )
            )
        elif model_mode == "autoregressive":
            if slot is not None:
                self._validate_slot(slot)
            page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position = (
                self.reserve_decode_step_pages(page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position)
            )

        # Update the state *after* all modifications.
        self.page_status.value = page_status
        self.page_map.value = page_map
        self.sequence_lengths.value = sequence_lengths
        self.num_pages_used.value = num_pages_used
        self.current_page.value = current_page
        self.current_page_position.value = current_page_position

        # Return the updated state.
        return self.get_page_state()