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
        self.max_page_groups = 4
        self.max_target_length = 256
        self.max_prefill_predict_length = 128
        self.max_pages_per_group = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
        self.key = jax.random.PRNGKey(0)

    def _init_page_manager(self) -> PageManager:
        return PageManager(
            num_pages=self.num_pages,
            tokens_per_page=self.tokens_per_page,
            max_page_groups=self.max_page_groups,
            max_target_length=self.max_target_length,
            max_prefill_predict_length=self.max_prefill_predict_length,
            max_pages_per_group=self.max_pages_per_group
        )

    def _validate_state_shapes(self, state_dict: dict) -> None:
        """
        Helper to validate that the shapes of state variables match expectations.
        """
        shapes = {k: np.array(v).shape for k, v in state_dict.items()}
        expected_shapes = {
            "page_status": (self.num_pages,),
            "page_map": (self.max_page_groups, self.max_pages_per_group),
            "sequence_lengths": (self.max_page_groups,),
            "num_pages_used": (self.max_page_groups,),
            "current_page": (self.max_page_groups,),
            "current_page_position": (self.max_page_groups,),
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
            np.full((self.max_page_groups, self.max_pages_per_group), -1, dtype=np.int32)
        )
        np.testing.assert_array_equal(np.array(state["sequence_lengths"]), np.zeros(self.max_page_groups, dtype=np.int32))
        np.testing.assert_array_equal(
            np.array(state["current_page"]),
            np.full(self.max_page_groups, -1, dtype=np.int32)
        )
        np.testing.assert_array_equal(np.array(state["current_page_position"]), np.zeros(self.max_page_groups, dtype=np.int32))

    def test_reserve_prefill_page_group(self) -> None:
        """
        Test that reserving prefill page group updates state as expected.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        page_group_id = 0
        true_length = 12  # Will require 2 pages.
        pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page

        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            page_group_id=page_group_id,
            true_length=true_length,
            mutable=["cache"]
        )[1]["cache"]

        self.assertEqual(int(updated_state["sequence_lengths"][page_group_id]), true_length, "Incorrect sequence length update")
        self.assertEqual(int(updated_state["num_pages_used"][page_group_id]), pages_needed, "Incorrect number of pages allocated")

        page_map = updated_state["page_map"][page_group_id]
        used_page_indices = page_map[page_map >= 0]
        self.assertEqual(len(np.array(used_page_indices)), pages_needed, "Wrong number of pages allocated")
        self.assertEqual(len(np.unique(np.array(used_page_indices))), pages_needed, "Duplicate pages allocated")
        for page_idx in np.array(used_page_indices):
            self.assertEqual(int(updated_state["page_status"][page_idx]), 1, f"Page {page_idx} not marked as used")

    def test_reserve_prefill_edge_cases(self) -> None:
        """
        Test edge cases for prefill reservation:
          - true_length = 0
          - true_length is an exact multiple of tokens_per_page
          - Reservation in different page_groups.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]

        # Case 1: true_length = 0.
        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            page_group_id=0,
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
            page_group_id=1,
            true_length=self.tokens_per_page * 2,
            mutable=["cache"]
        )[1]["cache"]
        self.assertEqual(int(updated_state["sequence_lengths"][1]), self.tokens_per_page * 2)
        self.assertEqual(int(updated_state["num_pages_used"][1]), 2)

        # Case 3: Different page_group_id.
        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            page_group_id=3,
            true_length=5,
            mutable=["cache"]
        )[1]["cache"]
        self.assertEqual(int(updated_state["sequence_lengths"][3]), 5)
        self.assertEqual(int(updated_state["num_pages_used"][3]), 1)

    def test_release_page_group(self) -> None:
        """
        Test that releasing pages for a page_group_id resets its state.
        """
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        page_group_id = 1
        initial_length = 10

        # First, allocate some pages.
        updated_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            page_group_id=page_group_id,
            true_length=initial_length,
            mutable=["cache"]
        )[1]["cache"]

        page_map = updated_state["page_map"]
        page_group_map = np.array(page_map[page_group_id])
        allocated_pages = page_group_map[page_group_map >= 0]

        _, released_vars = pm.apply(
            {"cache": updated_state},
            method=pm.release_page_group,
            page_group_id=page_group_id,
            mutable=["cache"]
        )
        released_state = released_vars["cache"]

        self.assertEqual(int(released_state["sequence_lengths"][page_group_id]), 0, "Sequence length not reset")
        self.assertEqual(int(released_state["num_pages_used"][page_group_id]), 0, "Pages used count not reset")
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
            model_mode="prefill",
            page_group_id=0,
            true_length=self.tokens_per_page -1,
            mutable=["cache"]
        )[1]["cache"]
        updated_state = pm.apply(
            {"cache": new_state},
            model_mode="autoregressive",
            mutable=["cache"]
        )[1]["cache"]

        self.assertEqual(int(updated_state["sequence_lengths"][0]), self.tokens_per_page)
        self.assertEqual(int(updated_state["num_pages_used"][0]), 1, "A new page should have been allocated.")
        self.assertGreater(int(updated_state["current_page"][0]), -1, "A page should have been allocated.")

        # Case 3: When no new page is needed.
        new_state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            page_group_id=2,
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

    # def test_page_exhaustion(self) -> None:
    #     """
    #     Test that attempting to reserve pages when none are free raises an error.
    #     """
    #     pm = self._init_page_manager()
    #     state = pm.init(self.key, mutable=["cache"])["cache"]
    #     # Mark all pages as used
    #     state = {
    #         **state,
    #         "page_status": jnp.ones((self.num_pages,), dtype=jnp.int32)
    #     }

    #     exception_raised = False  # Flag to track if the exception is raised
    #     caught_exception = None   # Variable to store the caught exception
    #     try:
    #         pm.apply(
    #             {"cache": state},
    #             model_mode="prefill",
    #             page_group_id=0,
    #             true_length=self.tokens_per_page,
    #             mutable=["cache"]
    #         )
    #     except ValueError as e:
    #         exception_raised = True
    #         caught_exception = e
    #     except Exception as other_exception:
    #         self.fail(f"Unexpected exception raised: {other_exception}")

    #     self.assertTrue(exception_raised, "Expected ValueError (or ArithmeticError) was not raised")
    #     if caught_exception: # Only check message if an exception was actually caught
    #         self.assertTrue("No free pages available" in str(caught_exception))


    def test_invalid_init_params(self) -> None:
        """
        Test that invalid initialization parameters raise a ValueError.
        """
        with self.assertRaises(ValueError):
            PageManager(
                num_pages=0,
                tokens_per_page=self.tokens_per_page,
                max_page_groups=self.max_page_groups,
                max_target_length=self.max_target_length,
                max_prefill_predict_length=self.max_prefill_predict_length,
                max_pages_per_group=self.max_pages_per_group
            ).init(self.key)
        with self.assertRaises(ValueError):
            PageManager(
                num_pages=self.num_pages,
                tokens_per_page=-1,
                max_page_groups=self.max_page_groups,
                max_target_length=self.max_target_length,
                max_prefill_predict_length=self.max_prefill_predict_length,
                max_pages_per_group=self.max_pages_per_group
            ).init(self.key)

    def test_state_consistency(self) -> None:
        pm = self._init_page_manager()
        state = pm.init(self.key, mutable=["cache"])["cache"]
        self.assertEqual(int(jnp.sum(state["page_status"])), 0)
        self.assertEqual(int(jnp.sum(state["page_map"] != -1)), 0)
        
        page_group_id = 0
        true_length = 12
        
        state = pm.apply(
            {"cache": state},
            model_mode="prefill",
            page_group_id=page_group_id,
            true_length=true_length,
            mutable=["cache"]
        )[1]["cache"]

        allocated_pages = int(jnp.sum(state["page_status"]))
        mapped_pages = int(jnp.sum(state["page_map"] != -1))
        self.assertEqual(allocated_pages, mapped_pages, "Allocated pages should match mapped pages")

        page_assignments = np.array(state["page_map"][state["page_map"] != -1]).flatten()
        self.assertEqual(len(page_assignments), len(np.unique(page_assignments)), "Found duplicate page assignments")

    # def test_page_group_boundaries(self) -> None:
    #     """
    #     Test that invalid page_group_id indices and exceeding sequence length are rejected.
    #     """
    #     pm = self._init_page_manager()
    #     state = pm.init(self.key, mutable=["cache"])["cache"]

    #     with self.assertRaises(ValueError):
    #         pm.apply(
    #             {"cache": state},
    #             model_mode="prefill",
    #             page_group_id=self.max_page_groups,
    #             true_length=1,
    #             mutable=["cache"]
    #         )

    #     page_group_id = 0
    #     max_length = self.tokens_per_page * self.max_pages_per_group
    #     state = pm.apply(
    #         {"cache": state},
    #         model_mode="prefill",
    #         page_group_id=page_group_id,
    #         true_length=max_length,
    #         mutable=["cache"]
    #     )[1]["cache"]

    #     with self.assertRaises(ValueError):
    #         pm.apply(
    #             {"cache": state},
    #             model_mode="prefill",
    #             page_group_id=page_group_id,
    #             true_length=max_length + 1,
    #             mutable=["cache"]
    #         )
    # def test_jit_compatibility(self) -> None:
    #     """JIT compatibility test with proper JAX operations"""
    #     pm = self._init_page_manager()
    #     state = pm.init(self.key, mutable=["cache"])["cache"]

    #     # Define static arguments
    #     page_group_id = 0
    #     true_length = 12

    #     # JIT the apply function
    #     @jax.jit
    #     def jitted_prefill(state_dict):
    #         vars = {"cache": state_dict}
    #         return pm.apply(vars, 
    #                         model_mode="prefill", 
    #                         page_group_id=page_group_id, 
    #                         true_length=true_length,
    #                         mutable=["cache"])

    #     # Run jitted version
    #     _, new_vars = jitted_prefill(state)
    #     updated_state = new_vars["cache"]
        
    #     # Get concrete values for testing
    #     seq_len = jax.device_get(updated_state["sequence_lengths"][page_group_id])
    #     self.assertEqual(seq_len, true_length)

    #     @jax.jit
    #     def jitted_autoregressive(state_dict):
    #         vars = {"cache": state_dict}
    #         return pm.apply(vars, model_mode="autoregressive",
    #                     mutable=["cache"])
        
    #     _, final_vars = jitted_autoregressive(updated_state)

if __name__ == "__main__":
    unittest.main()