"""
Unit tests for the PageManager module.
"""

import unittest
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from page_managers import PageManager, PageState  # Assuming page_managers.py is in the same directory

class TestPageManager(unittest.TestCase):
    """
    Unit tests for verifying the behavior of PageManager.
    """
    def setUp(self) -> None:
        """
        Initialize test parameters and a PRNG key.
        """
        self.num_pages = 16
        self.tokens_per_page = 8
        self.slots = 4
        self.max_target_length = 32
        self.max_prefill_predict_length = 128

        # Validate parameter relationships.
        assert self.max_target_length % self.tokens_per_page == 0, (
            f"max_target_length ({self.max_target_length}) must be divisible by "
            f"tokens_per_page ({self.tokens_per_page})"
        )

        self.max_pages_per_slot = self.max_target_length // self.tokens_per_page

        # Validate page capacity.
        total_required_pages = self.slots * self.max_pages_per_slot
        assert self.num_pages >= total_required_pages, (
            f"Insufficient pages ({self.num_pages}) for slots*max_pages_per_slot "
            f"({total_required_pages})"
        )

        self.key = jax.random.PRNGKey(0)

    def _validate_state_shapes(self, state_dict: dict) -> None:
        """
        Helper function to validate the shapes of state components.

        Args:
            state_dict: A dictionary of state variables to validate.
        """
        def unwrap(x: Any) -> Any:
            return x.value if hasattr(x, "value") else x

        shapes = {k: np.asarray(unwrap(v)).shape for k, v in state_dict.items()}
        expected_shapes = {
            "page_status": (self.num_pages,),
            "page_map": (self.slots, self.max_pages_per_slot),
            "sequence_lengths": (self.slots,),
            "num_pages_used": (self.slots,),
            "current_page": (self.slots,),
            "current_page_position": (self.slots,),
        }

        for k, expected in expected_shapes.items():
            self.assertEqual(
                shapes[k],
                expected,
                f"Shape mismatch for {k}: expected {expected}, got {shapes[k]}"
            )

    def test_initialization(self) -> None:
        """
        Test that initialization produces the correct variable shapes and initial values.
        """
        page_manager = PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            slots=self.slots,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_slot=self.max_pages_per_slot
        )

        state = page_manager.init(self.key, mutable=["cache"])
        cache_vars = state["cache"]

        self._validate_state_shapes(cache_vars)

        np.testing.assert_array_equal(
            cache_vars["page_status"],
            np.zeros(self.num_pages, dtype=np.int32),
            "Initial page status should be zeros"
        )
        np.testing.assert_array_equal(
            cache_vars["page_map"],
            np.full((self.slots, self.max_pages_per_slot), -1, dtype=np.int32),
            "Initial page map should be -1"
        )
        np.testing.assert_array_equal(
            cache_vars["sequence_lengths"],
            np.zeros(self.slots, dtype=np.int32),
            "Initial sequence lengths should be zero"
        )
        np.testing.assert_array_equal(
            cache_vars["current_page"],
            np.full(self.slots, -1, dtype=np.int32),
            "Initial current page should be -1"
        )

    def test_reserve_prefix_slot_pages(self) -> None:
        """
        Test that reserving prefix slot pages updates state as expected.
        """
        page_manager = PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            slots=self.slots,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_slot=self.max_pages_per_slot
        )

        state = page_manager.init(self.key, mutable=["cache"])
        slot_index = 0
        true_length = 12  # Will require 2 pages.
        pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page

        updated_state = page_manager.apply(
            {"cache": state["cache"]},
            model_mode="prefill",
            slot=slot_index,
            true_length=jnp.array(true_length),
            mutable=["cache"]
        )[1]["cache"]

        np.testing.assert_array_equal(
            updated_state["sequence_lengths"][slot_index],
            true_length,
            "Incorrect sequence length update"
        )
        np.testing.assert_array_equal(
            updated_state["num_pages_used"][slot_index],
            pages_needed,
            "Incorrect number of pages allocated"
        )

        page_map = updated_state["page_map"][slot_index]
        used_page_indices = page_map[page_map >= 0]

        np.testing.assert_array_equal(
            len(used_page_indices),
            pages_needed,
            "Wrong number of pages allocated"
        )
        np.testing.assert_array_equal(
            len(np.unique(used_page_indices)),
            pages_needed,
            "Duplicate pages allocated"
        )
        for page_idx in used_page_indices:
            np.testing.assert_array_equal(
                updated_state["page_status"][page_idx],
                1,
                f"Page {page_idx} not marked as used"
            )

    def test_reserve_prefix_edge_cases(self) -> None:
        """
        Test edge cases for prefix slot reservation, including zero length and exact multiples.
        """
        page_manager = PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            slots=self.slots,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_slot=self.max_pages_per_slot
        )
        state = page_manager.init(self.key, mutable=["cache"])

        # Test with true_length = 0.
        updated_state = page_manager.apply(
            {"cache": state["cache"]},
            model_mode="prefill",
            slot=0,
            true_length=jnp.array(0),
            mutable=["cache"]
        )[1]["cache"]
        np.testing.assert_array_equal(updated_state["sequence_lengths"][0], 0)
        np.testing.assert_array_equal(updated_state["num_pages_used"][0], 0)
        np.testing.assert_array_equal(updated_state["current_page"][0], -1)

        # Test with true_length as a multiple of tokens_per_page.
        updated_state = page_manager.apply(
            {"cache": state["cache"]},
            model_mode="prefill",
            slot=1,
            true_length=jnp.array(self.tokens_per_page * 2),
            mutable=["cache"]
        )[1]["cache"]
        np.testing.assert_array_equal(updated_state["sequence_lengths"][1], self.tokens_per_page * 2)
        np.testing.assert_array_equal(updated_state["num_pages_used"][1], 2)

        # Test with a different slot.
        updated_state = page_manager.apply(
            {"cache": state["cache"]},
            model_mode="prefill",
            slot=3,
            true_length=jnp.array(5),
            mutable=["cache"]
        )[1]["cache"]
        np.testing.assert_array_equal(updated_state["sequence_lengths"][3], 5)
        np.testing.assert_array_equal(updated_state["num_pages_used"][3], 1)

    def test_release_slot_pages(self) -> None:
        """
        Test that releasing pages for a slot resets the state properly.
        """
        page_manager = PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            slots=self.slots,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_slot=self.max_pages_per_slot
        )

        state = page_manager.init(self.key, mutable=["cache"])
        initial_cache = state["cache"]

        slot_index = jnp.array(1, dtype=jnp.int32)
        initial_length = jnp.array(10, dtype=jnp.int32)

        updated_state = page_manager.apply(
            {"cache": initial_cache},
            model_mode="prefill",
            slot=slot_index,
            true_length=initial_length,
            mutable=["cache"]
        )[1]["cache"]

        page_map = updated_state["page_map"]
        slot_map = page_map[slot_index.item()]
        allocated_pages = slot_map[slot_map >= 0]

        released_state = page_manager.apply(
            {"cache": updated_state},
            slot=slot_index,
            method=page_manager.release_slot_pages,
            mutable=["cache"]
        )[1]["cache"]

        np.testing.assert_array_equal(
            released_state["sequence_lengths"][slot_index.item()],
            0,
            "Sequence length not reset"
        )
        np.testing.assert_array_equal(
            released_state["num_pages_used"][slot_index.item()],
            0,
            "Pages used count not reset"
        )
        for page_idx in allocated_pages:
            np.testing.assert_array_equal(
                released_state["page_status"][page_idx],
                0,
                f"Page {page_idx} not freed"
            )

    def test_reserve_decode_step_pages(self) -> None:
        """
        Test that the autoregressive decode step page reservation updates state correctly.
        """
        page_manager = PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            slots=self.slots,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_slot=self.max_pages_per_slot
        )
        state = page_manager.init(self.key, mutable=["cache"])

        # Case 1: current_page is -1; no new token added.
        updated_state = page_manager.apply(
            {"cache": state["cache"]},
            model_mode="autoregressive",
            mutable=["cache"]
        )[1]["cache"]
        np.testing.assert_array_equal(
            updated_state["sequence_lengths"][0],
            0,
            "Sequence length should not change when no page is active."
        )

        # Case 2: New page needed.
        new_state = page_manager.apply(
            {"cache": state["cache"]},
            slot=0,
            method=page_manager.reserve_prefix_slot_pages,
            true_length=jnp.array(self.tokens_per_page - 1),
            mutable=["cache"]
        )[1]["cache"]

        updated_state = page_manager.apply(
            {"cache": new_state},
            model_mode="autoregressive",
            mutable=["cache"]
        )[1]["cache"]

        np.testing.assert_array_equal(updated_state["sequence_lengths"][0], self.tokens_per_page)
        np.testing.assert_array_equal(updated_state["num_pages_used"][0], 2)  # A new page allocated.
        self.assertGreater(updated_state["current_page"][0], -1, "A page should have been allocated.")

        # Case 3: New page not needed.
        new_state = page_manager.apply(
            {"cache": state["cache"]},
            slot=2,
            method=page_manager.reserve_prefix_slot_pages,
            true_length=jnp.array(5),
            mutable=["cache"]
        )[1]["cache"]

        updated_state = page_manager.apply(
            {"cache": new_state},
            model_mode="autoregressive",
            mutable=["cache"]
        )[1]["cache"]
        np.testing.assert_array_equal(updated_state["sequence_lengths"][2], 6)
        np.testing.assert_array_equal(updated_state["num_pages_used"][2], 1)

    def test_page_exhaustion(self) -> None:
        """
        Test that attempting to reserve pages when none are free raises an error.
        """
        page_manager = PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            slots=self.slots,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_slot=self.max_pages_per_slot
        )
        state = page_manager.init(self.key, mutable=["cache"])

        state["cache"]["page_status"] = jnp.ones(self.num_pages, dtype=jnp.int32)

        with self.assertRaises(ValueError) as context:
            page_manager.apply(
                {"cache": state["cache"]},
                model_mode="prefill",
                slot=0,
                true_length=jnp.array(self.tokens_per_page),
                mutable=["cache"]
            )
        self.assertTrue("No free pages available." in str(context.exception))


if __name__ == "__main__":
    unittest.main()
