import unittest
import jax
import jax.numpy as jnp
import numpy as np
from page_manager import PageManager

class TestPageManager(unittest.TestCase):
    """
    Unit tests for verifying the behavior of PageManager.
    """
    def setUp(self) -> None:
        self.num_pages = 128
        self.tokens_per_page = 8
        self.slots = 4
        self.max_target_length = 256
        self.max_prefill_predict_length = 128

        self.max_pages_per_slot = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page

        total_required_pages = self.slots * self.max_pages_per_slot
        assert self.num_pages >= total_required_pages, (
            f"Insufficient pages ({self.num_pages}) for slots*max_pages_per_slot ({total_required_pages})"
        )

        self.key = jax.random.PRNGKey(0)

    def _init_page_manager(self) -> PageManager:
        return PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            slots=self.slots,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_slot=self.max_pages_per_slot
        )

    def _validate_state_shapes(self, state_dict: dict) -> None:
        """
        Helper to validate that the shapes of state variables match expectations.
        """
        shapes = {k: np.array(v).shape for k, v in state_dict.items()}
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
        Test that initialization produces the correct shapes and initial values.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        self._validate_state_shapes(state)
        np.testing.assert_array_equal(np.array(state["page_status"]), np.zeros(self.num_pages, dtype=np.int32))
        np.testing.assert_array_equal(
            np.array(state["page_map"]),
            np.full((self.slots, self.max_pages_per_slot), -1, dtype=np.int32)
        )
        np.testing.assert_array_equal(np.array(state["sequence_lengths"]), np.zeros(self.slots, dtype=np.int32))
        np.testing.assert_array_equal(
            np.array(state["current_page"]),
            np.full(self.slots, -1, dtype=np.int32)
        )
        np.testing.assert_array_equal(np.array(state["current_page_position"]), np.zeros(self.slots, dtype=np.int32))

    def test_reserve_prefix_slot_pages(self) -> None:
        """
        Test that reserving prefix slot pages updates state as expected.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        slot = 0
        true_length = 12  # Will require 2 pages.
        pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page

        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            slot=slot,
            true_length=true_length,
            mutable=["cache"]
        )[1]["cache"]

        self.assertEqual(int(updated_state["sequence_lengths"][slot]), true_length, "Incorrect sequence length update")
        self.assertEqual(int(updated_state["num_pages_used"][slot]), pages_needed, "Incorrect number of pages allocated")

        page_map = updated_state["page_map"][slot]
        used_page_indices = page_map[page_map >= 0]
        self.assertEqual(len(np.array(used_page_indices)), pages_needed, "Wrong number of pages allocated")
        self.assertEqual(len(np.unique(np.array(used_page_indices))), pages_needed, "Duplicate pages allocated")
        for page_idx in np.array(used_page_indices):
            self.assertEqual(int(updated_state["page_status"][page_idx]), 1, f"Page {page_idx} not marked as used")

    def test_reserve_prefix_edge_cases(self) -> None:
        """
        Test edge cases for prefix reservation:
          - true_length = 0
          - true_length is an exact multiple of tokens_per_page
          - Reservation in different slots.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]

        # Case 1: true_length = 0.
        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            slot=0,
            true_length=0,
            mutable=["cache"]
        )[1]["cache"]
        self.assertEqual(int(updated_state["sequence_lengths"][0]), 0)
        self.assertEqual(int(updated_state["num_pages_used"][0]), 0)
        self.assertEqual(int(updated_state["current_page"][0]), -1)

        # Case 2: true_length exactly a multiple of tokens_per_page.
        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            slot=1,
            true_length=self.tokens_per_page * 2,
            mutable=["cache"]
        )[1]["cache"]
        self.assertEqual(int(updated_state["sequence_lengths"][1]), self.tokens_per_page * 2)
        self.assertEqual(int(updated_state["num_pages_used"][1]), 2)

        # Case 3: Different slot.
        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            slot=3,
            true_length=5,
            mutable=["cache"]
        )[1]["cache"]
        self.assertEqual(int(updated_state["sequence_lengths"][3]), 5)
        self.assertEqual(int(updated_state["num_pages_used"][3]), 1)

    def test_release_slot_pages(self) -> None:
        """
        Test that releasing pages for a slot resets its state.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        slot = 1
        initial_length = 10

        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            slot=slot,
            true_length=initial_length,
            mutable=["cache"]
        )[1]["cache"]

        page_map = updated_state["page_map"]
        slot_map = np.array(page_map[slot])
        allocated_pages = slot_map[slot_map >= 0]

        released_state = pm.apply(
            {"cache": updated_state},
            slot=slot,
            method=pm.release_slot_pages,
            mutable=["cache"]
        )[1]["cache"]

        self.assertEqual(int(released_state["sequence_lengths"][slot]), 0, "Sequence length not reset")
        self.assertEqual(int(released_state["num_pages_used"][slot]), 0, "Pages used count not reset")
        for page_idx in allocated_pages:
            self.assertEqual(int(released_state["page_status"][page_idx]), 0, f"Page {page_idx} not freed")

    def test_reserve_decode_step_pages(self) -> None:
        """
        Test that autoregressive decode step updates the state correctly.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]

        # Case 1: When no page is active, sequence length remains unchanged.
        updated_state = pm.apply(
            {"cache": state},
            model_mode="autoregressive",
            mutable=["cache"]
        )[1]["cache"]
        self.assertEqual(int(updated_state["sequence_lengths"][0]), 0, "Sequence length should not change when no page is active.")

        # Case 2: When a new page is needed.
        new_state = pm.apply(
            {"cache": state},
            slot=0,
            method=pm.reserve_prefix_slot_pages,
            true_length=self.tokens_per_page - 1,
            mutable=["cache"]
        )[1]["cache"]
        updated_state = pm.apply(
            {"cache": new_state},
            model_mode="autoregressive",
            mutable=["cache"]
        )[1]["cache"]

        self.assertEqual(int(updated_state["sequence_lengths"][0]), self.tokens_per_page)
        self.assertEqual(int(updated_state["num_pages_used"][0]), 2, "A new page should have been allocated.")
        self.assertGreater(int(updated_state["current_page"][0]), -1, "A page should have been allocated.")

        # Case 3: When no new page is needed.
        new_state = pm.apply(
            {"cache": state},
            slot=2,
            method=pm.reserve_prefix_slot_pages,
            true_length=5,
            mutable=["cache"]
        )[1]["cache"]
        updated_state = pm.apply(
            {"cache": new_state},
            model_mode="autoregressive",
            mutable=["cache"]
        )[1]["cache"]

        self.assertEqual(int(updated_state["sequence_lengths"][2]), 6)
        self.assertEqual(int(updated_state["num_pages_used"][2]), 1)

    def test_page_exhaustion(self) -> None:
        """
        Test that attempting to reserve pages when none are free raises an error.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        # Mark all pages as used.
        state["page_status"] = jnp.ones((self.num_pages,), dtype=jnp.int32)

        with self.assertRaises(ValueError) as context:
            pm.apply(
                {"cache": state},
                model_mode="prefill",
                slot=0,
                true_length=self.tokens_per_page,
                mutable=["cache"]
            )
        self.assertTrue("No free pages available." in str(context.exception))

    def test_invalid_init_params(self) -> None:
        """
        Test that invalid initialization parameters raise a ValueError.
        """
        with self.assertRaises(ValueError):
            PageManager(
                num_pages=0,
                tokens_per_page=self.tokens_per_page,
                slots=self.slots,
                max_target_length=self.max_target_length,
                max_prefill_predict_length=self.max_prefill_predict_length,
                max_pages_per_slot=self.max_pages_per_slot
            ).init(self.key)
        with self.assertRaises(ValueError):
            PageManager(
                num_pages=self.num_pages,
                tokens_per_page=-1,
                slots=self.slots,
                max_target_length=self.max_target_length,
                max_prefill_predict_length=self.max_prefill_predict_length,
                max_pages_per_slot=self.max_pages_per_slot
            ).init(self.key)

    def test_state_consistency(self) -> None:
        """
        Test that the overall page state remains consistent.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        self.assertEqual(int(jnp.sum(state["page_status"])), 0, "Initial page_status should have no allocated pages")
        self.assertEqual(int(jnp.sum(state["page_map"] != -1)), 0, "Initial page_map should have no mappings")

        slot = 0
        true_length = 12
        state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            slot=slot,
            true_length=true_length,
            mutable=["cache"]
        )[1]["cache"]

        allocated_pages = int(jnp.sum(state["page_status"]))
        mapped_pages = int(jnp.sum(state["page_map"] != -1))
        self.assertEqual(allocated_pages, mapped_pages, "Allocated pages should match mapped pages")

        page_assignments = np.array(state["page_map"][state["page_map"] != -1]).flatten()
        self.assertEqual(len(page_assignments), len(np.unique(page_assignments)), "Found duplicate page assignments")

    def test_slot_boundaries(self) -> None:
        """
        Test that invalid slot indices and exceeding sequence length are rejected.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]

        with self.assertRaises(ValueError):
            pm.apply(
                {"cache": state},
                model_mode="prefill",
                slot=self.slots,  # Invalid slot (slots are 0-indexed)
                true_length=1,
                mutable=["cache"]
            )

        slot = 0
        max_length = self.tokens_per_page * self.max_pages_per_slot
        state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            slot=slot,
            true_length=max_length,
            mutable=["cache"]
        )[1]["cache"]

        with self.assertRaises(ValueError):
            pm.apply(
                {"cache": state},
                model_mode="prefill",
                slot=slot,
                true_length=max_length + 1,
                mutable=["cache"]
            )

if __name__ == "__main__":
    unittest.main()
