import unittest
import jax
import jax.numpy as jnp
import numpy as np
from page_manager import PageManager, PageState, validate_page_group, validate_length


class TestPageManager(unittest.TestCase):

  def setUp(self):
    self.num_pages = 128
    self.tokens_per_page = 8
    self.max_page_groups = 4
    self.max_target_length = 256
    self.max_prefill_predict_length = 128
    self.max_pages_per_group = (self.max_target_length + self.tokens_per_page - 1) // self.tokens_per_page
    self.num_layers = 2
    self.config = type("Config", (object,), {"num_kv_heads": 2, "head_dim": 64, "dtype": jnp.float32})()
    self.key = jax.random.PRNGKey(0)

  def _init_page_manager(self):
    return PageManager(
        self.num_pages,
        self.tokens_per_page,
        self.max_page_groups,
        self.max_target_length,
        self.max_prefill_predict_length,
        self.max_pages_per_group,
        self.num_layers,
        self.config,
    )

  def _validate_state_shapes(self, state, layer_id=None):
    if layer_id is None:
      expected_shapes = {
          "page_status": (self.num_layers, self.num_pages),
          "page_map": (self.num_layers, self.max_page_groups, self.max_pages_per_group),
          "sequence_lengths": (
              self.num_layers,
              self.max_page_groups,
          ),
          "num_pages_used": (self.num_layers, self.max_page_groups),
          "current_page": (self.num_layers, self.max_page_groups),
          "current_page_position": (self.num_layers, self.max_page_groups),
      }
    else:
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
          getattr(state, k).shape, expected, f"Shape mismatch for {k}: expected {expected}, got {getattr(state, k).shape}"
      )

  def test_initialization(self):
    pm = self._init_page_manager()
    state = pm.get_page_state()
    self._validate_state_shapes(state)

    np.testing.assert_array_equal(np.array(state.page_status), np.zeros((self.num_layers, self.num_pages), dtype=np.int32))
    np.testing.assert_array_equal(
        np.array(state.page_map),
        np.full((self.num_layers, self.max_page_groups, self.max_pages_per_group), -1, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(state.sequence_lengths), np.zeros((self.num_layers, self.max_page_groups), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.array(state.current_page), np.full((self.num_layers, self.max_page_groups), -1, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.array(state.current_page_position), np.zeros((self.num_layers, self.max_page_groups), dtype=np.int32)
    )

  def test_jit_compatibility(self):
    pm = self._init_page_manager()
    layer_id = 0
    page_group_id = 0
    true_length = 12

    @jax.jit
    def jitted_prefill(
        page_status,
        page_map,
        sequence_lengths,
        num_pages_used,
        current_page,
        current_page_position,
        page_group_id,
        true_length,
    ):
      return pm.reserve_prefill_page_group_pages(
          page_group_id,
          true_length,
          page_status,
          page_map,
          sequence_lengths,
          num_pages_used,
          current_page,
          current_page_position,
      )

    initial_state = pm.get_page_state()
    updated_states = jitted_prefill(
        initial_state.page_status[layer_id],
        initial_state.page_map[layer_id],
        initial_state.sequence_lengths[layer_id],
        initial_state.num_pages_used[layer_id],
        initial_state.current_page[layer_id],
        initial_state.current_page_position[layer_id],
        page_group_id,
        true_length,
    )

    page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position = updated_states
    self.assertEqual(int(sequence_lengths[page_group_id]), true_length)

    @jax.jit
    def jitted_autoregressive(page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position):
      return pm.reserve_decode_step_pages(
          page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position
      )

    ar_states = jitted_autoregressive(
        page_status, page_map, sequence_lengths, num_pages_used, current_page, current_page_position
    )
    _, _, new_sequence_lengths, _, _, _ = ar_states
    self.assertEqual(int(new_sequence_lengths[page_group_id]), true_length + 1)

  def test_validation_functions(self):
    # Test the JAX-compatible validation functions
    is_valid = validate_page_group(1, self.max_page_groups)
    self.assertTrue(jnp.bool_(is_valid))

    is_invalid = validate_page_group(-1, self.max_page_groups)
    self.assertFalse(jnp.bool_(is_invalid))

    is_valid = validate_length(100, self.max_target_length)
    self.assertTrue(jnp.bool_(is_valid))

    is_invalid = validate_length(-1, self.max_target_length)
    self.assertFalse(jnp.bool_(is_invalid))

  def test_reserve_prefill_page_group(self):
    pm = self._init_page_manager()
    layer_id = 0
    page_group_id = 0
    true_length = 12

    updated_state = pm(model_mode="prefill", page_group_id=page_group_id, true_length=true_length, layer_id=layer_id)

    pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
    self.assertEqual(int(updated_state.sequence_lengths[page_group_id]), true_length)
    self.assertEqual(int(updated_state.num_pages_used[page_group_id]), pages_needed)

    page_map = updated_state.page_map[page_group_id]
    used_page_indices = page_map[page_map >= 0]
    self.assertEqual(len(np.array(used_page_indices)), pages_needed)
    self.assertEqual(len(np.unique(np.array(used_page_indices))), pages_needed)

    for page_idx in np.array(used_page_indices):
      self.assertEqual(int(updated_state.page_status[page_idx]), 1)

  def test_reserve_prefill_edge_cases(self):
    pm = self._init_page_manager()
    layer_id = 1

    # Test zero length
    updated_state = pm(model_mode="prefill", page_group_id=0, true_length=0, layer_id=layer_id)
    self.assertEqual(int(updated_state.sequence_lengths[0]), 0)
    self.assertEqual(int(updated_state.num_pages_used[0]), 0)
    self.assertEqual(int(updated_state.current_page[0]), -1)

    # Test exact page multiple
    updated_state = pm(model_mode="prefill", page_group_id=1, true_length=self.tokens_per_page * 2, layer_id=layer_id)
    self.assertEqual(int(updated_state.sequence_lengths[1]), self.tokens_per_page * 2)
    self.assertEqual(int(updated_state.num_pages_used[1]), 2)

    # Test partial page
    updated_state = pm(model_mode="prefill", page_group_id=3, true_length=5, layer_id=layer_id)
    self.assertEqual(int(updated_state.sequence_lengths[3]), 5)
    self.assertEqual(int(updated_state.num_pages_used[3]), 1)

  def test_release_page_group(self):
    pm = self._init_page_manager()
    page_group_id = 1
    initial_length = 10
    layer_id = 0

    # First allocate some pages
    updated_state = pm(model_mode="prefill", page_group_id=page_group_id, true_length=initial_length, layer_id=layer_id)

    page_map = updated_state.page_map
    page_group_map = np.array(page_map[page_group_id])
    allocated_pages = page_group_map[page_group_map >= 0]

    # Now release them
    pm.release_page_group(page_group_id)
    released_state = pm.get_page_state()

    self.assertEqual(int(released_state.sequence_lengths[layer_id, page_group_id]), 0)
    self.assertEqual(int(released_state.num_pages_used[layer_id, page_group_id]), 0)

    for page_idx in allocated_pages:
      self.assertEqual(int(released_state.page_status[layer_id, page_idx]), 0)

  def test_reserve_decode_step_pages(self):
    pm = self._init_page_manager()
    layer_id = 0

    # Test initial autoregressive step
    updated_state = pm(model_mode="autoregressive", layer_id=layer_id)
    self.assertEqual(int(updated_state.sequence_lengths[0]), 0)

    # Test after prefill
    new_state = pm(model_mode="prefill", page_group_id=0, true_length=self.tokens_per_page, layer_id=layer_id)
    updated_state = pm(model_mode="autoregressive", layer_id=layer_id)
    self.assertEqual(int(updated_state.sequence_lengths[0]), self.tokens_per_page + 1)
    self.assertEqual(int(updated_state.num_pages_used[0]), 2)
    self.assertGreater(int(updated_state.current_page[0]), -1)

    # Test with partial page
    new_state = pm(model_mode="prefill", page_group_id=2, true_length=5, layer_id=layer_id)
    updated_state = pm(model_mode="autoregressive", layer_id=layer_id)
    self.assertEqual(int(updated_state.sequence_lengths[2]), 6)
    self.assertEqual(int(updated_state.num_pages_used[2]), 1)

  def test_state_consistency(self):
    pm = self._init_page_manager()
    state = pm.get_page_state()
    self.assertEqual(int(jnp.sum(state.page_status)), 0)
    self.assertEqual(int(jnp.sum(state.page_map != -1)), 0)

    page_group_id = 0
    true_length = 12
    layer_id = 0
    state = pm(model_mode="prefill", page_group_id=page_group_id, true_length=true_length, layer_id=layer_id)

    allocated_pages = int(jnp.sum(state.page_status))
    mapped_pages = int(jnp.sum(state.page_map != -1))
    self.assertEqual(allocated_pages, mapped_pages)

    page_assignments = np.array(state.page_map[state.page_map != -1]).flatten()
    self.assertEqual(len(page_assignments), len(np.unique(page_assignments)))

  def test_page_group_boundaries(self):
    pm = self._init_page_manager()
    layer_id = 0

    # Test max page group boundary
    state = pm(model_mode="prefill", page_group_id=self.max_page_groups - 1, true_length=1, layer_id=layer_id)
    self.assertEqual(int(state.sequence_lengths[self.max_page_groups - 1]), 1)

    # Test max length boundary
    page_group_id = 0
    max_length = self.tokens_per_page * self.max_pages_per_group
    state = pm(model_mode="prefill", page_group_id=page_group_id, true_length=max_length, layer_id=layer_id)
    self.assertEqual(int(state.sequence_lengths[page_group_id]), max_length)

  def test_multi_layer_consistency(self):
    pm = self._init_page_manager()

    # Allocate pages in different layers
    for layer_id in range(self.num_layers):
      true_length = 10 + layer_id * 5
      page_group_id = layer_id % self.max_page_groups

      updated_state = pm(model_mode="prefill", page_group_id=page_group_id, true_length=true_length, layer_id=layer_id)

      self.assertEqual(
          updated_state.sequence_lengths[page_group_id], true_length, f"Layer {layer_id}: incorrect sequence length"
      )

      pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
      self.assertEqual(updated_state.num_pages_used[page_group_id], pages_needed, f"Layer {layer_id}: incorrect page count")

      page_map = updated_state.page_map[page_group_id]
      used_page_indices = page_map[page_map >= 0]
      for page_idx in np.array(used_page_indices):
        self.assertEqual(
            int(updated_state.page_status[page_idx]), 1, f"Layer {layer_id}, Page {page_idx} not marked as used"
        )

    # Test autoregressive steps for each layer
    for layer_id in range(self.num_layers):
      page_group_id = layer_id % self.max_page_groups
      updated_state = pm(model_mode="autoregressive", layer_id=layer_id)

      self.assertEqual(
          updated_state.sequence_lengths[page_group_id],
          10 + layer_id * 5 + 1,
          f"Layer {layer_id}: incorrect sequence length after autoregressive step",
      )

      if (10 + layer_id * 5) % self.tokens_per_page == 0:
        self.assertEqual(
            updated_state.num_pages_used[page_group_id],
            (10 + layer_id * 5 + self.tokens_per_page - 1) // self.tokens_per_page + 1,
            f"Layer {layer_id} page not incremented",
        )

  def test_call_with_no_layer(self):
    pm = self._init_page_manager()
    global_state = pm(model_mode="prefill", page_group_id=0, true_length=10, layer_id=None)
    self._validate_state_shapes(global_state)

  def test_invalid_model_mode(self):
    pm = self._init_page_manager()
    with self.assertRaises(ValueError) as context:
      pm(model_mode="invalid", page_group_id=0, true_length=10, layer_id=0)
    self.assertTrue("Invalid model_mode" in str(context.exception))


if __name__ == "__main__":
  unittest.main()
