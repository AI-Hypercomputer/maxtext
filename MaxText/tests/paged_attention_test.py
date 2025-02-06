import unittest
import pytest
import jax
import numpy as np
import math
import jax.numpy as jnp
from flax.core import freeze
from flax import linen as nn
from layers.attentions import PagedAttentionOp, Attention
from page_managers import PageManager, PageState
import common_types


def reference_attention(query, key, value):
  attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)
  attn_weights = jax.nn.softmax(attn_weights, axis=-1)

  return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


class PagedAttentionTest(unittest.TestCase):

  def setUp(self):
    self.cfg = {
        "per_device_batch_size": 1,
        "num_query_heads": 8,
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_prefill_predict_length": 512,
        "max_target_length": 1024,
        "num_pages": 64,
        "tokens_per_page": 32,
        "pages_per_compute_block": 16,
        "dtype": jnp.float32,
    }
    self.rng = jax.random.PRNGKey(42)
    devices = jax.devices()
    if len(devices) > 1:
      self.mesh = jax.sharding.Mesh(devices, axis_names=("data",))
    else:
      # Fallback for single-device testing
      self.mesh = jax.sharding.Mesh(devices, axis_names=())
    self.attention_op = PagedAttentionOp(
        mesh=self.mesh,
        num_pages=self.cfg["num_pages"],
        tokens_per_page=self.cfg["tokens_per_page"],
        max_pages_per_slot=self.cfg["max_target_length"] // self.cfg["tokens_per_page"],
        max_pages_per_prefill=self.cfg["max_prefill_predict_length"] // self.cfg["tokens_per_page"],
        pages_per_compute_block=self.cfg["pages_per_compute_block"],
        num_kv_heads=self.cfg["num_kv_heads"],
        kv_head_dim_size=self.cfg["head_dim"],
        dtype=self.cfg["dtype"],
    )


  def test_update_prefill_step_pages_state_management(self):
    """Test page state updates during prefill with state tracking."""
    batch_size = 4
    seq_len = 64  # Must be divisible by tokens_per_page
    num_heads = 8
    head_dim = 128

    # Create test input
    key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    value = jnp.ones((batch_size, seq_len, num_heads, head_dim))
    
    # Create initial page state
    page_state = PageState(
        page_status=jnp.zeros(self.cfg["num_pages"], dtype=jnp.int32),
        page_map=jnp.full((batch_size, self.cfg["num_pages"]), -1, dtype=jnp.int32),
        sequence_lengths=jnp.zeros(batch_size, dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
    )

    # Initialize
    variables = self.attention_op.init(
        self.rng, key, key, value, None,
        common_types.MODEL_MODE_PREFILL, page_state
    )

    # First apply - should initialize state
    output_tuple, vars = self.attention_op.apply(
        variables, key, key, value, None,
        common_types.MODEL_MODE_PREFILL, page_state,
        mutable=["cache"]
    )

    # Verify state was stored and updated
    self.assertIn("page_state", vars["cache"], "page_state not stored in cache")
    new_page_state = vars["cache"]["page_state"]  # Remove .value access
    self.assertIsNotNone(new_page_state, "page_state is None after update")

    # Add type verification
    self.assertIsInstance(new_page_state, PageState, 
                         f"Expected PageState, got {type(new_page_state)}")

    # Print debug info
    print(f"\nPage State after update:")
    print(f"  sequence_lengths: {new_page_state.sequence_lengths}")
    print(f"  num_pages_used: {new_page_state.num_pages_used}")
    print(f"  page_map shape: {new_page_state.page_map.shape}")

    # Verify page assignments
    expected_pages_per_seq = seq_len // self.cfg["tokens_per_page"]
    for b in range(batch_size):
        # Check sequence tracking
        print(f"{new_page_state.sequence_lengths[b]=}") # [64]
        print(f"{seq_len=}")                            # 64
        print(f"{new_page_state.num_pages_used[b]=}")   # [2]
        print(f"{expected_pages_per_seq=}")             # 2

        self.assertEqual(
            new_page_state.sequence_lengths[b], seq_len,
            f"Wrong sequence length for batch {b}"
        )
        self.assertEqual(
            new_page_state.num_pages_used[b], expected_pages_per_seq,
            f"Wrong number of pages used for batch {b}"
        )

        # Verify used pages are marked
        used_pages = new_page_state.page_map[b][new_page_state.page_map[b] >= 0]
        print(f"{new_page_state.page_map[b]=}")         # [0, 1, 0, ..., 0]
        print(f"{used_pages=}")                         # [1]
        self.assertEqual(
            len(used_pages), expected_pages_per_seq,
            f"Wrong number of pages mapped for batch {b}"
        )
        for page in used_pages:
            self.assertEqual(
                new_page_state.page_status[page], 1,
                f"Page {page} not marked as used"
            )

    # Verify page contents
    key_pages = vars["cache"]["key_pages"].value
    value_pages = vars["cache"]["value_pages"].value

    for b in range(batch_size):
        for p in range(expected_pages_per_seq):
            page_idx = new_page_state.page_map[b][p]
            start_pos = p * self.cfg["tokens_per_page"]
            end_pos = start_pos + self.cfg["tokens_per_page"]

            # Check key storage
            np.testing.assert_array_equal(
                key_pages[:, page_idx],
                key[b, start_pos:end_pos].transpose(1, 0, 2),
                f"Key mismatch batch {b} page {p}"
            )

            # Check value storage 
            np.testing.assert_array_equal(
                value_pages[:, page_idx],
                value[b, start_pos:end_pos].transpose(1, 0, 2),
                f"Value mismatch batch {b} page {p}"
            )
  
  @pytest.mark.tpu_only
  def test_decode_step_pages_state_management(self):
    """Test page state management during decode steps with proper boundary handling.
    
    Critical test case: Step 31 (page boundary)
    - Data is written to position 31 of old pages [1,2,3,4]
    - State advances to position 0 of new pages [5,6,7,8]
    - Must verify data in position 31 of old pages [1,2,3,4]
    """
    batch_size = 4
    num_heads = 8
    head_dim = 128
    tokens_per_page = self.cfg["tokens_per_page"]
    
    def determine_read_location(step: int, state: PageState) -> tuple[jnp.ndarray, int]:
        """Calculate correct page and position for reading stored data.
        
        Args:
            step: Current generation step
            state: Current page state with updated page map
            
        Returns:
            Tuple of (page_numbers, position) where page_numbers is array of
            pages to read from and position is where in page to read.
        """
        if step == 0:
            return jnp.ones(batch_size, dtype=jnp.int32), 0
            
        # At page boundary (e.g. step 31), data is in old pages at last position
        if step % tokens_per_page == tokens_per_page - 1:
            # Get first (original) page from each batch's page map
            read_pages = state.page_map[:, 0]
            return read_pages, tokens_per_page - 1
        
        # Normal case - read from current page at current position
        return state.current_page, step % tokens_per_page

    def get_stored_key(
        vars: dict, 
        read_page: jnp.ndarray, 
        read_pos: int, 
        batch_idx: int
    ) -> jnp.ndarray:
        """Retrieve stored key data for verification."""
        return vars["cache"]["key_pages"].value[
            :,  # heads
            read_page[batch_idx],  # Page for this batch
            read_pos,  # Position in page
            :   # dims
        ]

    # Initialize empty page state
    page_state = PageState(
        page_status=jnp.zeros(self.cfg["num_pages"], dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self.cfg["num_pages"]), dtype=jnp.int32),
        sequence_lengths=jnp.zeros(batch_size, dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
    )
    
    # Initialize attention op
    variables = self.attention_op.init(
        self.rng,
        jnp.ones((batch_size, 1, num_heads, head_dim)),
        jnp.ones((batch_size, 1, num_heads, head_dim)),
        jnp.ones((batch_size, 1, num_heads, head_dim)),
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        page_state
    )

    # Test through one page boundary
    num_steps = tokens_per_page + 2

    print(f"\nTesting {num_steps} decode steps:")
    print(f"  tokens_per_page: {tokens_per_page}")
    print(f"  num_pages: {self.cfg['num_pages']}")
    
    for step in range(num_steps):
        # Create distinct input for this step
        step_value = float(step + 1)
        step_key = jnp.ones((batch_size, 1, num_heads, head_dim)) * step_value
        step_value = jnp.ones((batch_size, 1, num_heads, head_dim)) * step_value * 2

        print(f"\nStep {step}:")
        print(f"  Pre-step sequence lengths: {page_state.sequence_lengths}")
        print(f"  Pre-step current pages: {page_state.current_page}")
        print(f"  Pre-step page positions: {page_state.current_page_position}")
        print(f"  Pre-step page map:")
        for b in range(batch_size):
            used_pages = page_state.page_map[b][page_state.page_map[b] > 0]
            print(f"    Batch {b}: {used_pages}")
        
        # Process step
        output_tuple, vars = self.attention_op.apply(
            variables,
            step_key, step_key, step_value,
            None,
            common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state,
            mutable=["cache"]
        )
        
        # Get updated state
        new_page_state = vars["cache"]["page_state"]
        
        # Determine correct read location
        read_pages, read_pos = determine_read_location(step, new_page_state)
        
        print(f"  Post-process state:")
        print(f"    sequence_lengths: {new_page_state.sequence_lengths}")
        print(f"    current_pages: {new_page_state.current_page}")
        print(f"    current_positions: {new_page_state.current_page_position}")
        print(f"  Verification:")
        print(f"    reading from pages: {read_pages}")
        print(f"    reading at position: {read_pos}")
        print(f"    updated page map:")
        for b in range(batch_size):
            used_pages = new_page_state.page_map[b][new_page_state.page_map[b] > 0]
            print(f"      Batch {b}: {used_pages}")
        
        # Verify state and data for each batch
        for b in range(batch_size):
            # Verify sequence length increased
            self.assertEqual(
                new_page_state.sequence_lengths[b],
                step + 1,
                f"Wrong sequence length batch {b} step {step}"
            )
            
            # Get stored data
            stored_key = get_stored_key(vars, read_pages, read_pos, b)
            
            # Print first few values for debugging
            if b == 0:
                print(f"    batch 0 stored key values: {stored_key[0, 0:5]}")
                print(f"    batch 0 expected values: {step_key[0, 0, 0, 0:5]}")
            
            # Verify stored key matches input
            np.testing.assert_array_almost_equal(
                stored_key,
                step_key[b, 0],
                decimal=5,
                err_msg=f"Key mismatch batch {b} step {step}"
            )
            
            # After page boundary, verify both old and new pages tracked
            if step >= tokens_per_page:
                used_pages = new_page_state.page_map[b][new_page_state.page_map[b] > 0]
                self.assertEqual(
                    len(used_pages),
                    2,  # Should have both old and new page
                    f"Should have 2 pages tracked for batch {b} step {step}"
                )
        
        # Update state for next iteration
        page_state = new_page_state
        variables = vars
    
    # Final state verification
    print("\nFinal state verification:")
    for b in range(batch_size):
        used_pages = page_state.page_map[b][page_state.page_map[b] > 0]
        print(f"  Batch {b}:")
        print(f"    Used pages: {used_pages}")
        print(f"    Sequence length: {page_state.sequence_lengths[b]}")
        
        # Verify both pages still tracked
        self.assertEqual(
            len(used_pages),
            2,
            f"Wrong number of pages tracked for batch {b}"
        )
        self.assertTrue(
            used_pages[1] > used_pages[0],  # Second page should be higher numbered
            f"Second page {used_pages[1]} not higher than first {used_pages[0]}"
        )

if __name__ == "__main__":
  unittest.main()
