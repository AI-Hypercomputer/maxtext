"""
Module for managing page allocation for prefix and autoregressive decoding.
"""

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any

import common_types

Array = common_types.Array
DType = common_types.DType
AxisNames = common_types.AxisNames


@struct.dataclass
class PageState:
    """
    Dataclass that holds the state of pages managed by PageManager.

    Attributes:
        page_status: An array of shape (num_pages,) indicating whether each page is free (0) or used (1).
        page_map: An array of shape (slots, max_pages_per_slot) mapping slot indices to page indices.
        sequence_lengths: An array of shape (slots,) representing the token length per slot.
        num_pages_used: An array of shape (slots,) indicating the number of pages allocated for each slot.
        current_page: An array of shape (slots,) with the index of the current page for each slot.
        current_page_position: An array of shape (slots,) with the current position within the current page.
    """
    page_status: Array
    page_map: Array
    sequence_lengths: Array
    num_pages_used: Array
    current_page: Array
    current_page_position: Array


class PageManager(nn.Module):
    """
    Module that manages page allocation and release for both prefix and autoregressive decoding modes.

    Attributes:
        num_pages: Total number of pages available.
        tokens_per_page: Maximum number of tokens per page.
        slots: Number of slots (e.g. batch elements) to manage.
        max_target_length: Maximum target sequence length.
        max_prefill_predict_length: Maximum prefill/predict sequence length.
        max_pages_per_slot: Maximum number of pages that can be allocated per slot.
    """
    num_pages: int
    tokens_per_page: int
    slots: int
    max_target_length: int
    max_prefill_predict_length: int
    max_pages_per_slot: int

    def setup(self) -> None:
        """
        Validates initialization parameters and sets up the cache variables.
        """
        if self.num_pages <= 0:
            raise ValueError(f"Invalid num_pages: {self.num_pages}")
        if self.tokens_per_page <= 0:
            raise ValueError(f"Invalid tokens_per_page: {self.tokens_per_page}")
        if self.slots <= 0:
            raise ValueError(f"Invalid slots: {self.slots}")
        if self.max_pages_per_slot <= 0:
            raise ValueError(f"Invalid max_pages_per_slot: {self.max_pages_per_slot}")

        # Create the cache variables as plain arrays and attach metadata manually.
        self.page_status = self.variable(
            "cache", "page_status",
            lambda: jnp.zeros((self.num_pages,), jnp.int32)
        )
        self.page_status.meta = ("num_pages",)

        self.page_map = self.variable(
            "cache", "page_map",
            lambda: jnp.full((self.slots, self.max_pages_per_slot), -1, jnp.int32)
        )
        self.page_map.meta = ("cache_batch", "cache_sequence")

        self.sequence_lengths = self.variable(
            "cache", "sequence_lengths",
            lambda: jnp.zeros((self.slots,), jnp.int32)
        )
        self.sequence_lengths.meta = ("cache_batch",)

        self.num_pages_used = self.variable(
            "cache", "num_pages_used",
            lambda: jnp.zeros((self.slots,), jnp.int32)
        )
        self.num_pages_used.meta = ("cache_batch",)

        self.current_page = self.variable(
            "cache", "current_page",
            lambda: jnp.full((self.slots,), -1, jnp.int32)
        )
        self.current_page.meta = ("cache_batch",)

        self.current_page_position = self.variable(
            "cache", "current_page_position",
            lambda: jnp.zeros((self.slots,), jnp.int32)
        )
        self.current_page_position.meta = ("cache_batch",)

    def find_next_free_page(self, page_status: Array) -> Array:
        """
        Finds the index of the next free page from the page_status array.

        Args:
            page_status: Array indicating the current status of pages (0 for free, 1 for used).

        Returns:
            An integer array with the index of the free page, or a special value:
              - Returns -2 if no free page is available.
        """
        free_pages = jnp.where(page_status == 0, size=1, fill_value=-1)[0]
        return jnp.where(free_pages[0] == -1, -2, free_pages[0])

    def _reserve_single_page(
        self, slot: int, i: int, page_status: Array, page_map: Array
    ) -> Tuple[Array, Array]:
        """
        Reserves a single page for the given slot and position index in the page mapping.

        Args:
            slot: The slot index where the page should be reserved.
            i: The index (within the slot) to reserve.
            page_status: The current page_status array.
            page_map: The current page_map array.

        Returns:
            A tuple (new_page_status, new_page_map) after reserving the page.
        """
        next_free_page = self.find_next_free_page(page_status)
        page_status = jax.lax.cond(
            next_free_page == -2,
            lambda ps: ps,  # No operation if no free page is available.
            lambda ps: ps.at[next_free_page].set(1),
            page_status
        )
        page_map = jax.lax.cond(
            next_free_page == -2,
            lambda pm: pm,
            lambda pm: pm.at[slot, i].set(next_free_page),
            page_map
        )
        return page_status, page_map

    def release_slot_pages(self, slot: int) -> None:
        """
        Releases all pages allocated to a given slot, resetting its state.

        Args:
            slot: The slot index for which pages should be released.
        """
        used_pages = jnp.where(self.page_map.value[slot] > -1, self.page_map.value[slot], -1)
        valid_used = used_pages[used_pages > -1]
        self.page_status.value = self.page_status.value.at[valid_used].set(0)
        self.page_map.value = self.page_map.value.at[slot, :].set(-1)
        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(0)
        self.num_pages_used.value = self.num_pages_used.value.at[slot].set(0)
        self.current_page.value = self.current_page.value.at[slot].set(-1)
        self.current_page_position.value = self.current_page_position.value.at[slot].set(0)

    def reserve_prefix_slot_pages(self, slot: int, true_length: Array) -> None:
        """
        Reserves pages for a prefix sequence in the specified slot.

        The function releases any previously allocated pages in the slot, then
        reserves new pages based on the true_length of the sequence.

        Args:
            slot: The slot index to reserve pages for.
            true_length: The true sequence length as an integer array.
        """
        self.release_slot_pages(slot)
        num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
        last_page_position = jnp.where(true_length > 0, (true_length - 1) % self.tokens_per_page, 0)

        num_free_pages = jnp.sum(self.page_status.value == 0)
        num_free_pages_int = int(num_free_pages)
        if num_pages_needed > num_free_pages_int:
            if num_free_pages_int == 0:
                raise ValueError("No free pages available.")
            else:
                raise ValueError(
                    f"Not enough pages available to reserve. Requested: {num_pages_needed}, Available: {num_free_pages_int}"
                )

        page_status, page_map = jax.lax.fori_loop(
            0, num_pages_needed,
            lambda i, carry: self._reserve_single_page(slot, i, carry[0], carry[1]),
            (self.page_status.value, self.page_map.value)
        )
        self.page_status.value = page_status
        self.page_map.value = page_map

        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(true_length)
        self.num_pages_used.value = self.num_pages_used.value.at[slot].set(num_pages_needed)
        self.current_page.value = self.current_page.value.at[slot].set(
            jax.lax.cond(
                jnp.greater(num_pages_needed, 0),
                lambda _: self.page_map.value[slot, num_pages_needed - 1],
                lambda _: -1,
                operand=None
            )
        )
        self.current_page_position.value = self.current_page_position.value.at[slot].set(last_page_position)

    def _reserve_single_slot(
        self,
        i: int,
        page_status: Array,
        page_map: Array,
        current_page: Array,
        num_pages_used: Array,
        new_num_pages_used: Array
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Helper function for reserving a page for a single slot during autoregressive decoding.

        This function checks if a new page is needed (by comparing new_num_pages_used with
        num_pages_used for the slot) and, if so, reserves a page.

        Args:
            i: The slot index.
            page_status: Current page status array.
            page_map: Current page map array.
            current_page: Current array of current page indices.
            num_pages_used: Current number of pages used per slot.
            new_num_pages_used: Desired number of pages for each slot after adding the new token.

        Returns:
            A tuple of updated (page_status, page_map, current_page, num_pages_used).
        """
        needs_new_page = jnp.greater(new_num_pages_used[i], num_pages_used[i])
        next_free_page = jax.lax.cond(
            needs_new_page,
            lambda ps: self.find_next_free_page(ps),
            lambda ps: jnp.array(-1, dtype=jnp.int32),
            page_status
        )
        next_free_page = jnp.where((needs_new_page & (next_free_page == -2)), -1, next_free_page)

        page_status = jax.lax.cond(
            needs_new_page & (next_free_page >= 0),
            lambda ps: ps.at[next_free_page].set(1),
            lambda ps: ps,
            page_status
        )
        page_map = jax.lax.cond(
            needs_new_page & (next_free_page >= 0),
            lambda pm: pm.at[i, num_pages_used[i]].set(next_free_page),
            lambda pm: pm,
            page_map
        )
        current_page = jax.lax.cond(
            needs_new_page & (next_free_page >= 0),
            lambda cp: cp.at[i].set(next_free_page),
            lambda cp: cp,
            current_page
        )
        num_pages_used = jax.lax.cond(
            needs_new_page & (next_free_page >= 0),
            lambda npu: npu.at[i].set(num_pages_used[i] + 1),
            lambda npu: npu,
            num_pages_used
        )
        return page_status, page_map, current_page, num_pages_used

    def reserve_decode_step_pages(self) -> None:
        """
        Reserves any needed pages for a decoding step in autoregressive mode.

        The function increments the sequence length for each slot based on whether the
        current page is active and allocates a new page if the current page fills up.
        """
        # Increase sequence length by 1 if there is an active current page.
        new_sequence_lengths = self.sequence_lengths.value + jnp.where(self.current_page.value == -1, 0, 1)
        # Base number of pages required (ceiling division)
        base_new_num_pages_used = (new_sequence_lengths + self.tokens_per_page - 1) // self.tokens_per_page
        # If the sequence exactly fills the page, allocate one extra page.
        new_num_pages_used = jnp.where(
            (new_sequence_lengths > 0) & (new_sequence_lengths % self.tokens_per_page == 0),
            base_new_num_pages_used + 1,
            base_new_num_pages_used
        )
        new_current_page_position = jnp.where(
            new_sequence_lengths == 0, 0, (new_sequence_lengths - 1) % self.tokens_per_page
        )

        page_status, page_map, current_page, num_pages_used = jax.lax.fori_loop(
            0, self.slots,
            lambda i, carry: self._reserve_single_slot(
                i, carry[0], carry[1], carry[2], carry[3], new_num_pages_used
            ),
            (self.page_status.value, self.page_map.value, self.current_page.value, self.num_pages_used.value)
        )

        self.page_status.value = page_status
        self.page_map.value = page_map
        self.current_page.value = current_page
        self.num_pages_used.value = num_pages_used
        self.sequence_lengths.value = new_sequence_lengths
        self.current_page_position.value = new_current_page_position

    def get_cache_metadata(self) -> Dict[str, Optional[Tuple[str, ...]]]:
        """
        Returns the metadata (partitioning info) for each cache variable.

        Returns:
            A dictionary mapping variable names to their metadata tuples (if available).
        """
        return {
            "page_status": self.page_status.meta if hasattr(self.page_status, "meta") else None,
            "page_map": self.page_map.meta if hasattr(self.page_map, "meta") else None,
            "sequence_lengths": self.sequence_lengths.meta if hasattr(self.sequence_lengths, "meta") else None,
            "num_pages_used": self.num_pages_used.meta if hasattr(self.num_pages_used, "meta") else None,
            "current_page": self.current_page.meta if hasattr(self.current_page, "meta") else None,
            "current_page_position": self.current_page_position.meta if hasattr(self.current_page_position, "meta") else None,
        }

    def __call__(
        self,
        model_mode: Optional[str] = None,
        slot: Optional[int] = None,
        true_length: Optional[Array] = None
    ) -> Optional[PageState]:
        """
        Applies the appropriate reservation/release logic based on the model mode.

        Args:
            model_mode: A string indicating the mode ("prefill" or "autoregressive").
            slot: The slot index to use (only for "prefill" mode).
            true_length: The true sequence length (only for "prefill" mode).

        Returns:
            A PageState instance representing the current state of the page manager,
            or None if in a specific "params" initialization context.
        """
        if model_mode == common_types.MODEL_MODE_PREFILL and self.is_mutable_collection("params"):
            return None

        if model_mode == common_types.MODEL_MODE_PREFILL:
            assert slot is not None and true_length is not None, "slot and true_length must be provided for prefill mode."
            self.reserve_prefix_slot_pages(slot, true_length)
        elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
            self.reserve_decode_step_pages()

        return PageState(
            page_status=self.page_status.value,
            page_map=self.page_map.value,
            sequence_lengths=self.sequence_lengths.value,
            num_pages_used=self.num_pages_used.value,
            current_page=self.current_page.value,
            current_page_position=self.current_page_position.value
        )
