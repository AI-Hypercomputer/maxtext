# page_manager.py
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

        max_possible_pages = self.slots * self.max_pages_per_slot
        if max_possible_pages > self.num_pages:
            raise ValueError(
                f"Configuration would require up to {max_possible_pages} pages "
                f"but only {self.num_pages} are available"
            )
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
        if not (0 <= slot < self.slots):
            raise ValueError(f"Invalid slot index: {slot}")

    def _validate_length(self, length: int):
        if length < 0:
            raise ValueError(f"Negative sequence length: {length}")
        if length > self.max_target_length:
            raise ValueError(f"Sequence length {length} exceeds max_target_length {self.max_target_length}")

    def find_next_free_page(self, page_status: jnp.ndarray) -> int:
        free_indices = jnp.where(page_status == 0, size=1, fill_value=-1)[0]
        if free_indices[0] == -1:
            return -2
        return int(free_indices[0])
    
    def _reserve_single_page(self, slot: int, i: int,
                             page_status: jnp.ndarray, page_map: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_free_page = self.find_next_free_page(page_status)
        if next_free_page == -2:
            return page_status, page_map
        page_status = page_status.at[next_free_page].set(1)
        page_map = page_map.at[slot, i].set(next_free_page)
        return page_status, page_map

    def _check_page_capacity(self, slot: int, needed_pages: int):
        current_pages = int(self.num_pages_used.value[slot])
        total_pages = current_pages + needed_pages
        if total_pages > self.max_pages_per_slot:
            raise ValueError(
                f"Operation would require {total_pages} pages for slot {slot} "
                f"but max_pages_per_slot is {self.max_pages_per_slot}"
            )

    def release_slot_pages(self, slot: int) -> None:
        used_pages = self.page_map.value[slot]
        valid_used = used_pages[used_pages > -1]
        self.page_status.value = self.page_status.value.at[valid_used].set(0)
        self.page_map.value = self.page_map.value.at[slot, :].set(-1)
        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(0)
        self.num_pages_used.value = self.num_pages_used.value.at[slot].set(0)
        self.current_page.value = self.current_page.value.at[slot].set(-1)
        self.current_page_position.value = self.current_page_position.value.at[slot].set(0)

    def reserve_prefix_slot_pages(self, slot: int, true_length: int) -> None:
        self.release_slot_pages(slot)
        num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
        last_page_position = (true_length - 1) % self.tokens_per_page if true_length > 0 else 0

        num_free_pages = int(jnp.sum(self.page_status.value == 0))
        if num_pages_needed > num_free_pages:
            if num_free_pages == 0:
                raise ValueError("No free pages available.")
            else:
                raise ValueError(
                    f"Not enough pages available to reserve. Requested: {num_pages_needed}, Available: {num_free_pages}"
                )

        self._check_page_capacity(slot, num_pages_needed)
        page_status = self.page_status.value
        page_map = self.page_map.value
        for i in range(num_pages_needed):
            page_status, page_map = self._reserve_single_page(slot, i, page_status, page_map)
        self.page_status.value = page_status
        self.page_map.value = page_map
        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(true_length)
        self.num_pages_used.value = self.num_pages_used.value.at[slot].set(num_pages_needed)
        cur_page = page_map[slot, num_pages_needed - 1] if num_pages_needed > 0 else -1
        self.current_page.value = self.current_page.value.at[slot].set(cur_page)
        self.current_page_position.value = self.current_page_position.value.at[slot].set(last_page_position)

    def _reserve_single_slot(self, i: int,
                             page_status: jnp.ndarray, page_map: jnp.ndarray,
                             current_page: jnp.ndarray, num_pages_used: jnp.ndarray,
                             new_num_pages_used: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        needs_new_page = new_num_pages_used[i] > num_pages_used[i]
        if needs_new_page:
            next_free_page = self.find_next_free_page(page_status)
            if next_free_page == -2 or next_free_page < 0:
                next_free_page = -1
            if next_free_page >= 0:
                page_status = page_status.at[next_free_page].set(1)
                page_map = page_map.at[i, num_pages_used[i]].set(next_free_page)
                current_page = current_page.at[i].set(next_free_page)
                num_pages_used = num_pages_used.at[i].set(num_pages_used[i] + 1)
        return page_status, page_map, current_page, num_pages_used

    def reserve_decode_step_pages(self) -> None:
        new_sequence_lengths = self.sequence_lengths.value + jnp.where(self.current_page.value == -1, 0, 1)
        base_new_num_pages_used = (new_sequence_lengths + self.tokens_per_page - 1) // self.tokens_per_page
        new_num_pages_used = jnp.where(
            (new_sequence_lengths > 0) & (new_sequence_lengths % self.tokens_per_page == 0),
            base_new_num_pages_used + 1,
            base_new_num_pages_used
        )
        new_current_page_position = jnp.where(new_sequence_lengths == 0, 0, (new_sequence_lengths - 1) % self.tokens_per_page)
        for i in range(self.slots):
            self._check_page_capacity(i, int(new_num_pages_used[i] - self.num_pages_used.value[i]))
        page_status = self.page_status.value
        page_map = self.page_map.value
        current_page = self.current_page.value
        num_pages_used = self.num_pages_used.value
        for i in range(self.slots):
            ps, pm, cp, npu = self._reserve_single_slot(i, page_status, page_map, current_page, num_pages_used, new_num_pages_used)
            page_status, page_map, current_page, num_pages_used = ps, pm, cp, npu
        self.page_status.value = page_status
        self.page_map.value = page_map
        self.current_page.value = current_page
        self.num_pages_used.value = num_pages_used
        self.sequence_lengths.value = new_sequence_lengths
        self.current_page_position.value = new_current_page_position

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

        if model_mode == "prefill":
            if slot is None or true_length is None:
                raise ValueError("Prefill mode requires both slot and true_length")
            self._validate_slot(slot)
            self._validate_length(true_length)
            self.reserve_prefix_slot_pages(slot, true_length)
        else:
            if slot is not None:
                self._validate_slot(slot)
            self.reserve_decode_step_pages()

        return self.get_page_state()
