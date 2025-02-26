import common_types
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers.attentions import PagedAttentionOp
from page_manager import PageManager, PageState
import pyconfig
import sys
from typing import Any


def debug_print(name, tensor):
  """Helper function for printing tensor shapes with a descriptive name."""
  if isinstance(tensor, jnp.ndarray):
    print(f"{name}.shape: {tensor.shape}")
  elif isinstance(tensor, np.ndarray):
    print(f"{name}.shape: {tensor.shape} (numpy)")
  elif isinstance(tensor, tuple):
    print(f"{name} is a tuple of length {len(tensor)=}")
  elif isinstance(tensor, int):
    print(f"{name}: {tensor} (int)")
  elif isinstance(tensor, list):
    print(f"{name} is a list of length {len(tensor)=}")
  else:
    print(f"{name}: {type(tensor)}")


class PagedAttentionTest(unittest.TestCase):

  def setUp(self):
    pyconfig.initialize(
        [sys.argv[0], "configs/base.yml"],
        per_device_batch_size=1,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=16,
        max_prefill_predict_length=8,
        tokens_per_page=4,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=4,
        base_emb_dim=8,
        base_mlp_dim=16,
        base_num_decoder_layers=2,
        dtype="float32",
        attention="paged",
        model_name="default",
        rope_min_timescale=1,
        rope_max_timescale=10000,
        matmul_precision="highest",
        num_pages=32,
        max_pages_per_group=16,
    )

    self.config = pyconfig.config
    self.mesh = None
    self.quant = None
    self.output_dim = self.config.emb_dim

    # Create a page manager for testing
    self.page_manager = PageManager(
        num_pages=self.config.num_pages,
        tokens_per_page=self.config.tokens_per_page,
        max_page_groups=int(self.config.per_device_batch_size * jax.device_count()),
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        max_pages_per_group=(self.config.max_target_length + self.config.tokens_per_page - 1) // self.config.tokens_per_page,
        num_layers=self.config.num_decoder_layers,
        config=self.config,
    )

    # Initialize the attention module in a parent context
    class DummyParent(nn.Module):
      config: Any
      mesh: Any
      quant: Any
      output_dim: int

      @nn.compact
      def __call__(
          self,
          query,
          key_pages,
          value_pages,
          page_map,
          sequence_lengths,
          page_group_id,
          layer_id,
          is_initialized,
          model_mode,
      ):
        paged_attention = PagedAttentionOp(
            mesh=self.mesh,
            num_kv_heads=self.config.base_num_kv_heads,
            kv_head_dim_size=self.config.head_dim,
            config=self.config,
            output_dim=self.output_dim,
            dtype=self.config.dtype,
            quant=self.quant,
        )
        return paged_attention(
            query=query,
            key=None,
            value=None,
            decoder_segment_ids=None,
            model_mode=model_mode,
            key_pages=key_pages,
            value_pages=value_pages,
            page_map=page_map,
            sequence_lengths=sequence_lengths,
            page_group_id=page_group_id,
            layer_id=layer_id,
            is_initialized=is_initialized,
        )

    self.parent_module = DummyParent(config=self.config, mesh=self.mesh, quant=self.quant, output_dim=self.output_dim)

    self.initial_state = self.page_manager.get_initial_state()

  def _get_dummy_inputs(self, batch_size, seq_len, num_heads, head_dim):
    """Helper to create dummy input arrays with fixed values for predictability."""
    key = jax.random.PRNGKey(42)
    query_key, key_key, value_key = jax.random.split(key, 3)

    query = jax.random.normal(query_key, (batch_size, seq_len, num_heads, head_dim))
    key = jax.random.normal(key_key, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(value_key, (batch_size, seq_len, num_heads, head_dim))
    return query, key, value

  def _create_initialized_page_state(self, batch_size, true_length, layer_id, page_group_id):
    """Create a page state with initialized pages for testing."""
    state = self.initial_state

    # Reserve pages for the sequence
    state = self.page_manager.reserve_prefill_pages(
        state=state,
        page_group_id=page_group_id,
        true_length=true_length,
        layer_id=layer_id,
    )

    return state

  def test_prefill_single_page(self):
    """Test prefill with a sequence that fits in a single page."""
    batch_size = 1
    seq_len = 3  # Less than tokens_per_page (4)
    num_heads = self.config.base_num_kv_heads
    head_dim = self.config.head_dim
    layer_id = 0
    page_group_id = 0
    true_length = seq_len

    # 1. Create query and dummy KV data
    query, key, value = self._get_dummy_inputs(batch_size, seq_len, num_heads, head_dim)
    print("\n======= test_prefill_single_page ======")
    print("0. Original parameters:")
    debug_print("batch_size", batch_size)
    debug_print("seq_len", seq_len)
    debug_print("num_kv_heads", num_heads)
    debug_print("head_dim", head_dim)
    print("1. Input Shapes:")
    debug_print("query", query)
    debug_print("key", key)
    debug_print("value", value)

    # 2. Initialize page state with allocated pages
    page_state = self._create_initialized_page_state(batch_size, true_length, layer_id, page_group_id)
    print("\n2. Page State:")
    debug_print("page_state.page_map", page_state.page_map)
    debug_print("page_state.page_status", page_state.page_status)
    debug_print("page_state.sequence_lengths", page_state.sequence_lengths)
    debug_print("page_state.sequence_lengths.shape", page_state.sequence_lengths.shape)

    # 3. Get the physical page assigned to this sequence
    physical_page = page_state.page_map[layer_id, page_group_id, 0]
    self.assertGreaterEqual(physical_page, 0, "No physical page allocated")
    print(f"\n3. {physical_page=}")

    # 4. Create dummy key_pages and value_pages
    key_pages = jnp.zeros((self.config.num_pages, self.config.tokens_per_page, num_heads, head_dim))
    value_pages = jnp.zeros((self.config.num_pages, self.config.tokens_per_page, num_heads, head_dim))
    print("\n4. Initialized KV Pages:")
    debug_print("key_pages", key_pages)
    debug_print("value_pages", value_pages)

    # 5. Fill the pages with our test data
    for i in range(seq_len):
      key_pages = key_pages.at[physical_page, i].set(key[0, i])
      value_pages = value_pages.at[physical_page, i].set(value[0, i])
    print("\n5. Filled KV Pages:")
    debug_print("key_pages", key_pages)
    debug_print("value_pages", value_pages)

    # 6. Initialize the module
    variables = self.parent_module.init(
        jax.random.PRNGKey(0),
        query,
        key_pages,
        value_pages,
        page_state.page_map[layer_id],
        page_state.sequence_lengths[layer_id],
        page_group_id,
        layer_id,
        True,
        common_types.MODEL_MODE_PREFILL,
    )
    print("\n6. Initialized Variables:")
    debug_print("variables", variables)  # Flax variables are dictionaries.

    # 7. Apply the module
    result = self.parent_module.apply(
        variables,
        query,
        key_pages,
        value_pages,
        page_state.page_map[layer_id],
        page_state.sequence_lengths[layer_id],
        page_group_id,
        layer_id,
        True,
        common_types.MODEL_MODE_PREFILL,
    )
    print("\n7. Result:")
    debug_print("result", result)

    # Verify output shape is correct
    self.assertEqual(
        result.shape,
        (batch_size, seq_len, self.output_dim),
        f"Result shape mismatch: {result.shape} vs expected {(batch_size, seq_len, self.output_dim)}",
    )

    # Verify output is not all zeros or NaNs
    self.assertFalse(jnp.allclose(result, 0.0), "Result is all zeros")
    self.assertFalse(jnp.any(jnp.isnan(result)), "Result contains NaN values")

    # 9. Compare result shape with expected output shape (without out_projection applied)
    # Since our actual implementation will have applied the out_projection (DenseGeneral),
    # we only check that our expected shape is correct
    self.assertEqual(
        result.shape,
        (batch_size, seq_len, self.output_dim),
        f"Result shape mismatch: {result.shape} vs expected {(batch_size, seq_len, self.output_dim)}",
    )

  def test_prefill_multi_page(self):
    """Test prefill with a sequence spanning multiple pages."""
    batch_size = 1
    seq_len = 6  # Spans two pages (tokens_per_page=4)
    num_heads = self.config.base_num_kv_heads
    head_dim = self.config.head_dim
    layer_id = 0
    page_group_id = 0
    true_length = seq_len

    print("\n======= test_prefill_multi_page ======")

    # 1. Create query and dummy KV data
    query, key, value = self._get_dummy_inputs(batch_size, seq_len, num_heads, head_dim)
    print("1. Input Shapes:")
    debug_print("query", query)
    debug_print("key", key)
    debug_print("value", value)

    # 2. Initialize page state with allocated pages
    page_state = self._create_initialized_page_state(batch_size, true_length, layer_id, page_group_id)
    print("\n2. Page State:")
    debug_print("page_state.page_map", page_state.page_map)
    debug_print("page_state.page_status", page_state.page_status)
    debug_print("page_state.sequence_lengths", page_state.sequence_lengths)

    # 3. Get the physical pages assigned to this sequence
    physical_page1 = page_state.page_map[layer_id, page_group_id, 0]
    physical_page2 = page_state.page_map[layer_id, page_group_id, 1]
    self.assertGreaterEqual(physical_page1, 0, "First physical page not allocated")
    self.assertGreaterEqual(physical_page2, 0, "Second physical page not allocated")
    self.assertNotEqual(physical_page1, physical_page2, "Both pages are the same physical page")
    print(f"\n3. {physical_page1=}, {physical_page2=}")

    # 4. Create dummy key_pages and value_pages
    key_pages = jnp.zeros((self.config.num_pages, self.config.tokens_per_page, num_heads, head_dim))
    value_pages = jnp.zeros((self.config.num_pages, self.config.tokens_per_page, num_heads, head_dim))
    print("\n4. Initialized KV Pages:")
    debug_print("key_pages", key_pages)
    debug_print("value_pages", value_pages)

    # 5. Fill the pages with our test data
    tokens_per_page = self.config.tokens_per_page

    # Fill first page (tokens 0-3)
    for i in range(min(tokens_per_page, seq_len)):
      key_pages = key_pages.at[physical_page1, i].set(key[0, i])
      value_pages = value_pages.at[physical_page1, i].set(value[0, i])

    # Fill second page (tokens 4-5) if needed
    for i in range(tokens_per_page, seq_len):
      pos_in_page = i - tokens_per_page
      key_pages = key_pages.at[physical_page2, pos_in_page].set(key[0, i])
      value_pages = value_pages.at[physical_page2, pos_in_page].set(value[0, i])
    print("\n5. Filled KV Pages:")
    debug_print("key_pages", key_pages)
    debug_print("value_pages", value_pages)

    # 6. Initialize and apply the module
    variables = self.parent_module.init(
        jax.random.PRNGKey(0),
        query,
        key_pages,
        value_pages,
        page_state.page_map[layer_id],
        page_state.sequence_lengths[layer_id],
        page_group_id,
        layer_id,
        True,
        common_types.MODEL_MODE_PREFILL,
    )
    print("\n6. Initialized Variables:")
    debug_print("variables", variables)

    result = self.parent_module.apply(
        variables,
        query,
        key_pages,
        value_pages,
        page_state.page_map[layer_id],
        page_state.sequence_lengths[layer_id],
        page_group_id,
        layer_id,
        True,
        common_types.MODEL_MODE_PREFILL,
    )
    print("\n7. Result:")
    debug_print("result", result)

    # 8. Verify basic properties of the result
    self.assertEqual(
        result.shape,
        (batch_size, seq_len, self.output_dim),
        f"Result shape mismatch: {result.shape} vs expected {(batch_size, seq_len, self.output_dim)}",
    )

    # Verify the result is not all zeros or NaNs
    self.assertFalse(jnp.allclose(result, 0.0), "Result is all zeros")
    self.assertFalse(jnp.any(jnp.isnan(result)), "Result contains NaN values")

  def test_decode_single_token(self):
    """Test autoregressive generation of a single token."""
    batch_size = 1
    num_heads = self.config.base_num_kv_heads
    head_dim = self.config.head_dim
    layer_id = 0
    page_group_id = 0

    print("\n======= test_decode_single_token ======")

    # 1. Set up a context with a few tokens already processed
    context_length = 3
    page_state = self._create_initialized_page_state(batch_size, context_length, layer_id, page_group_id)
    print("1. Page State:")
    debug_print("page_state.page_map", page_state.page_map)
    debug_print("page_state.sequence_lengths", page_state.sequence_lengths)

    # Get physical page for the context
    physical_page = page_state.page_map[layer_id, page_group_id, 0]
    print(f"\n2. {physical_page=}")

    # 2. Fill the context with some dummy values
    key_pages = jnp.zeros((self.config.num_pages, self.config.tokens_per_page, num_heads, head_dim))
    value_pages = jnp.zeros((self.config.num_pages, self.config.tokens_per_page, num_heads, head_dim))
    print("\n3. Initialized KV Pages:")
    debug_print("key_pages", key_pages)
    debug_print("value_pages", value_pages)

    # Create some context embeddings
    _, context_keys, context_values = self._get_dummy_inputs(batch_size, context_length, num_heads, head_dim)
    print("\n4. Context Embeddings:")
    debug_print("context_keys", context_keys)
    debug_print("context_values", context_values)

    # Fill the page with context
    for i in range(context_length):
      key_pages = key_pages.at[physical_page, i].set(context_keys[0, i])
      value_pages = value_pages.at[physical_page, i].set(context_values[0, i])
    print("\n5. Filled KV Pages:")
    debug_print("key_pages", key_pages)
    debug_print("value_pages", value_pages)

    # 3. Create query for autoregressive step
    query, _, _ = self._get_dummy_inputs(batch_size, 1, num_heads, head_dim)
    print("\n6. Query for Autoregressive Step:")
    debug_print("query", query)

    # 4. Set sequence_lengths to context_length to simulate generating the next token
    sequence_lengths = page_state.sequence_lengths.at[layer_id, page_group_id].set(context_length)
    page_state = page_state.replace(sequence_lengths=sequence_lengths)
    print("\n7. Updated Sequence Lengths:")
    debug_print("page_state.sequence_lengths", page_state.sequence_lengths)

    # 5. Initialize and apply the module in decode mode
    variables = self.parent_module.init(
        jax.random.PRNGKey(0),
        query,
        key_pages,
        value_pages,
        page_state.page_map[layer_id],
        page_state.sequence_lengths[layer_id],
        page_group_id,
        layer_id,
        True,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
    )
    print("\n8. Initialized Variables:")
    debug_print("variables", variables)

    result = self.parent_module.apply(
        variables,
        query,
        key_pages,
        value_pages,
        page_state.page_map[layer_id],
        page_state.sequence_lengths[layer_id],
        page_group_id,
        layer_id,
        True,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
    )
    print("\n9. Result:")
    debug_print("result", result)

    # 6. For decode mode, check basic properties
    self.assertEqual(
        result.shape,
        (batch_size, 1, self.output_dim),
        f"Result shape mismatch: {result.shape} vs expected {(batch_size, 1, self.output_dim)}",
    )

    self.assertFalse(jnp.allclose(result, 0.0), "Result is all zeros")
    self.assertFalse(jnp.any(jnp.isnan(result)), "Result contains NaN values")

  def test_interaction_with_page_manager(self):
    """Test the interaction between PageManager and PagedAttentionOp."""
    batch_size = 1
    seq_len = 3
    num_heads = self.config.base_num_kv_heads
    head_dim = self.config.head_dim
    layer_id = 0
    page_group_id = 0

    print("\n======= test_interaction_with_page_manager ======")

    # 1. Create initial state and reserve pages
    page_state = self.page_manager.get_initial_state()
    print("1. Initial Page State:")
    debug_print("page_state.page_map", page_state.page_map)
    debug_print("page_state.page_status", page_state.page_status)
    debug_print("page_state.sequence_lengths", page_state.sequence_lengths)

    page_state = self.page_manager.reserve_prefill_pages(
        state=page_state,
        page_group_id=page_group_id,
        true_length=seq_len,
        layer_id=layer_id,
    )
    print("\n2. Page State After Reserve:")
    debug_print("page_state.page_map", page_state.page_map)
    debug_print("page_state.page_status", page_state.page_status)
    debug_print("page_state.sequence_lengths", page_state.sequence_lengths)

    # 2. Get information about allocated pages
    physical_page = page_state.page_map[layer_id, page_group_id, 0]
    self.assertGreaterEqual(physical_page, 0, "No physical page allocated")
    print(f"\n3. {physical_page=}")

    # Verify page status shows allocated
    self.assertEqual(page_state.page_status[layer_id, physical_page], 1, "Page status not updated correctly")

    # Verify sequence length is set correctly
    self.assertEqual(page_state.sequence_lengths[layer_id, page_group_id], seq_len, "Sequence length not set correctly")

    # 3. Test releasing the pages
    released_state = self.page_manager.release_page_group(
        state=page_state,
        page_group_id=page_group_id,
    )
    print("\n4. Page State After Release:")
    debug_print("released_state.page_map", released_state.page_map)
    debug_print("released_state.page_status", released_state.page_status)
    debug_print("released_state.sequence_lengths", released_state.sequence_lengths)

    # Verify pages were released
    self.assertEqual(released_state.page_status[layer_id, physical_page], 0, "Page not released correctly")

    # Verify sequence length reset
    self.assertEqual(released_state.sequence_lengths[layer_id, page_group_id], 0, "Sequence length not reset on release")

    # 4. Test reserving pages for autoregressive mode
    # First, set up a context in prefill mode
    page_state = self.page_manager.reserve_prefill_pages(
        state=released_state,
        page_group_id=page_group_id,
        true_length=1,  # Start with 1 token
        layer_id=layer_id,
    )
    print("\n5. Page State After Prefill (1 token):")
    debug_print("page_state.page_map", page_state.page_map)
    debug_print("page_state.page_status", page_state.page_status)
    debug_print("page_state.sequence_lengths", page_state.sequence_lengths)

    # Now reserve for autoregressive step
    ar_state = self.page_manager.reserve_autoregressive_pages(
        state=page_state,
        page_group_id=page_group_id,
        layer_id=layer_id,
    )
    print("\n6. Page State After Autoregressive Reserve:")
    debug_print("ar_state.page_map", ar_state.page_map)
    debug_print("ar_state.page_status", ar_state.page_status)
    debug_print("ar_state.sequence_lengths", ar_state.sequence_lengths)

    # Verify sequence length incremented
    self.assertEqual(
        ar_state.sequence_lengths[layer_id, page_group_id], 2, "Sequence length not incremented in autoregressive mode"
    )


if __name__ == "__main__":
  unittest.main()
