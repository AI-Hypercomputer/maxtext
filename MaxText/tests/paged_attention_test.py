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
import pyconfig
import os


def reference_attention(query, key, value):
    """Reference implementation of attention for validation."""
    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


class PagedAttentionTest(unittest.TestCase):
    def setUp(self):
      # Initialize pyconfig with minimal overrides
      base_config = {
          "per_device_batch_size": 1,  # Use integer instead of float
          "max_target_length": 2048,
          "max_prefill_predict_length": 64,
          "base_emb_dim": 2048,
          "head_dim": 128,
          "base_num_kv_heads": 32,
          "dtype": "float32",
          "enable_checkpointing": False,
      }
      
      # Initialize config first
      config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yml")
      pyconfig.initialize([None, config_path], **base_config)
      self.config = pyconfig.config
      
      # Create mesh separately
      devices = jax.devices()
      if len(devices) > 1:
          self.mesh = jax.sharding.Mesh(devices, axis_names=("data",))
      else:
          self.mesh = jax.sharding.Mesh(devices, axis_names=())
      
      # Initialize test rng
      self.rng = jax.random.PRNGKey(42)

      # Initialize attention operator
      self.attention_op = PagedAttentionOp(
          mesh=self.mesh,
          num_pages=self.config.num_pages,
          tokens_per_page=self.config.tokens_per_page,
          max_pages_per_slot=self.config.max_target_length // self.config.tokens_per_page,
          max_pages_per_prefill=self.config.max_prefill_predict_length // self.config.tokens_per_page,
          pages_per_compute_block=self.config.pages_per_compute_block,
          num_kv_heads=self.config.base_num_kv_heads,
          kv_head_dim_size=self.config.head_dim,
          dtype=self.config.dtype,
          config=self.config
      )

    def test_update_prefill_step_pages_state_management(self):
        """Test page state updates during prefill with state tracking."""
        batch_size = int(self.config.per_device_batch_size)  # Convert float to int for array shapes
        seq_len = 64  # Must be divisible by tokens_per_page
        num_heads = self.config.base_num_kv_heads
        head_dim = self.config.head_dim

        # Create test input tensors
        key = jnp.ones((batch_size, seq_len, num_heads, head_dim))
        value = jnp.ones((batch_size, seq_len, num_heads, head_dim))

        # Create initial page state
        page_state = PageState(
            page_status=jnp.zeros(self.config.num_pages, dtype=jnp.int32),
            page_map=jnp.full(
                (batch_size, self.config.max_target_length // self.config.tokens_per_page),
                -1,
                dtype=jnp.int32
            ),
            sequence_lengths=jnp.zeros(batch_size, dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.full(batch_size, -1, dtype=jnp.int32),
            current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
        )

        # Initialize variables
        variables = self.attention_op.init(
            self.rng,
            key, key, value,
            None,  # decoder_segment_ids
            "prefill",  # model_mode
            page_state
        )

        # First apply - should initialize state
        output_tuple, vars = self.attention_op.apply(
            variables,
            key, key, value,
            None,  # decoder_segment_ids
            "prefill",  # model_mode
            page_state,
            mutable=["cache"]
        )

        # Verify individual state components
        cache_vars = vars["cache"]
        self.assertIn("page_status", cache_vars)
        self.assertIn("page_map", cache_vars)
        self.assertIn("sequence_lengths", cache_vars)
        self.assertIn("num_pages_used", cache_vars)
        self.assertIn("current_page", cache_vars)
        self.assertIn("current_page_position", cache_vars)

        new_page_status = cache_vars["page_status"].value
        new_page_map = cache_vars["page_map"].value
        new_sequence_lengths = cache_vars["sequence_lengths"].value
        new_num_pages_used = cache_vars["num_pages_used"].value

        # Validate sequence lengths and page usage
        expected_pages_per_seq = seq_len // self.config.tokens_per_page
        for b in range(batch_size):
            self.assertEqual(new_sequence_lengths[b], seq_len)
            self.assertEqual(new_num_pages_used[b], expected_pages_per_seq)

            # Count actual used pages
            used_pages_count = 0
            for i in range(new_page_map.shape[1]):
                if new_page_map[b, i] >= 0:
                    used_pages_count += 1
            self.assertEqual(used_pages_count, expected_pages_per_seq)

            # Validate page status consistency
            for p in range(new_page_map.shape[1]):
                page_val = new_page_map[b, p]
                if page_val >= 0:
                    self.assertEqual(new_page_status[page_val], 1)

        # Verify page contents
        key_pages = vars["cache"]["key_pages"].value
        value_pages = vars["cache"]["value_pages"].value
        current_page = vars["cache"]["current_page"].value
        current_page_position = vars["cache"]["current_page_position"].value

        for b in range(batch_size):
            for p in range(expected_pages_per_seq):
                page_idx = new_page_map[b, p]
                start_pos = p * self.config.tokens_per_page
                end_pos = start_pos + self.config.tokens_per_page

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
        """Test page state management during decode steps."""
        batch_size = int(self.config.per_device_batch_size)
        num_heads = self.config.base_num_kv_heads
        head_dim = self.config.head_dim
        tokens_per_page = self.config.tokens_per_page

        def determine_read_location(step: int, page_map, current_page, current_page_position, sequence_lengths) -> tuple[jnp.ndarray, int]:
            """Calculate correct page and position for reading stored data."""
            is_first_step = step == 0
            is_end_of_page = (current_page_position + 1) % tokens_per_page == 0

            def _first_step_fn():
                return current_page, 0

            def _end_of_page_fn():
                return current_page, tokens_per_page - 1

            def _within_page_fn():
                return current_page, current_page_position

            page_indices, position = jax.lax.cond(
                is_first_step,
                _first_step_fn,
                lambda: jax.lax.cond(is_end_of_page, _end_of_page_fn, _within_page_fn)
            )

            return page_indices, position

        def get_stored_key(vars: dict, read_page: jnp.ndarray, read_pos: int, batch_idx: int) -> jnp.ndarray:
            """Retrieve stored key data for verification."""
            key_pages = vars["cache"]["key_pages"].value
            page_idx = jax.lax.dynamic_index_in_dim(read_page, batch_idx, keepdims=False)
            return jax.lax.dynamic_slice_in_dim(key_pages, page_idx, 1, axis=1)[:, 0, read_pos, :]

        # Initialize empty page state
        page_state = PageState(
            page_status=jnp.zeros(self.config.num_pages, dtype=jnp.int32),
            page_map=jnp.full(
                (batch_size, self.config.max_target_length // self.config.tokens_per_page),
                -1,
                dtype=jnp.int32
            ),
            sequence_lengths=jnp.zeros(batch_size, dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.full(batch_size, -1, dtype=jnp.int32),
            current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
        )

        # Initialize attention op
        variables = self.attention_op.init(
            self.rng,
            jnp.ones((batch_size, 1, num_heads, head_dim)),
            jnp.ones((batch_size, 1, num_heads, head_dim)),
            jnp.ones((batch_size, 1, num_heads, head_dim)),
            None,
            "autoregressive",
            page_state
        )

        # Test through one page boundary
        num_steps = tokens_per_page + 2

        # Accumulate values for later assertions
        all_sequence_lengths = []
        all_stored_keys = []
        all_step_keys = []
        all_used_pages_counts = []

        for step in range(num_steps):
            step_value = float(step + 1)
            step_key = jnp.ones((batch_size, 1, num_heads, head_dim)) * step_value
            step_value_v = jnp.ones((batch_size, 1, num_heads, head_dim)) * step_value * 2

            output_tuple, vars = self.attention_op.apply(
                variables,
                step_key, step_key, step_value_v,
                None,
                "autoregressive",
                page_state,
                mutable=["cache"]
            )

            cache_vars = vars["cache"]
            new_page_map = cache_vars["page_map"].value
            new_current_page = cache_vars["current_page"].value
            new_current_page_position = cache_vars["current_page_position"].value
            new_sequence_lengths = cache_vars["sequence_lengths"].value
            new_num_pages_used = cache_vars["num_pages_used"].value

            read_pages, read_pos = determine_read_location(
                step,
                new_page_map,
                new_current_page,
                new_current_page_position,
                new_sequence_lengths
            )

            step_lengths = []
            step_keys = []
            step_used_pages = []

            for b in range(batch_size):
                step_lengths.append(new_sequence_lengths[b])
                stored_key = get_stored_key(vars, read_pages, read_pos, b)
                step_keys.append(np.array(stored_key))
                expected_key = np.array(step_key[b, 0])

                used_pages_count = 0
                for i in range(new_page_map.shape[1]):
                    if new_page_map[b, i] >= 0:
                        used_pages_count += 1
                step_used_pages.append(used_pages_count)

            all_sequence_lengths.append(step_lengths)
            all_stored_keys.append(step_keys)
            all_step_keys.append(expected_key)
            all_used_pages_counts.append(step_used_pages)
            variables = vars

        # Perform assertions after the loop
        for step in range(num_steps):
            for b in range(batch_size):
                self.assertEqual(all_sequence_lengths[step][b], step + 1)
                np.testing.assert_array_almost_equal(
                    all_stored_keys[step][b],
                    all_step_keys[step],
                    decimal=5
                )
                if step >= tokens_per_page:
                    self.assertEqual(all_used_pages_counts[step][b], 2)

        # Final state verification
        new_page_map = cache_vars["page_map"].value
        for b in range(batch_size):
            used_pages = []
            for i in range(new_page_map.shape[1]):
                page_val = new_page_map[b, i]
                if page_val >= 0:
                    used_pages.append(page_val)

            self.assertEqual(len(used_pages), 2 if num_steps > tokens_per_page else 1)
            if len(used_pages) == 2:
                self.assertTrue(used_pages[1] > used_pages[0])


if __name__ == "__main__":
    unittest.main()