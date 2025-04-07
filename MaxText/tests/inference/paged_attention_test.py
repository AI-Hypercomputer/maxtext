#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import unittest

from MaxText import common_types
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from MaxText.inference.page_manager import PageState
from MaxText.inference.paged_attention import PagedAttentionOp


def reference_attention(query, key, value):
  attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)
  attn_weights = jax.nn.softmax(attn_weights, axis=-1)

  return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


class PagedAttentionTest(unittest.TestCase):

  def setUp(self):
    self._per_device_batch_size = 1
    self._num_query_heads = 8
    self._num_kv_heads = 8
    self._head_dim = 128
    self._max_prefill_predict_length = 512
    self._max_target_length = 1024
    self._dtype = jnp.float32
    # PagedAttention settings
    self._num_pages = 64
    self._tokens_per_page = 32
    self._pages_per_compute_block = 16
    self.rng = jax.random.PRNGKey(42)
    devices = jax.devices()
    if len(devices) > 1:
      self.mesh = jax.sharding.Mesh(devices, axis_names=("data",))
    else:
      # Fallback for single-device testing
      self.mesh = jax.sharding.Mesh(devices, axis_names=())
    self.attention_op = PagedAttentionOp(
        mesh=self.mesh,
        num_pages=self._num_pages,
        tokens_per_page=self._tokens_per_page,
        max_pages_per_slot=self._max_target_length // self._tokens_per_page,
        max_pages_per_prefill=self._max_prefill_predict_length // self._tokens_per_page,
        pages_per_compute_block=self._pages_per_compute_block,
        num_kv_heads=self._num_kv_heads,
        kv_head_dim_size=self._head_dim,
        dtype=self._dtype,
    )

  @pytest.mark.tpu_only
  def test_paged_attention_output_shape(self):
    attention_op = PagedAttentionOp(
        mesh=self.mesh,
        num_pages=self._num_pages,
        tokens_per_page=self._tokens_per_page,
        max_pages_per_slot=self._max_target_length // self._tokens_per_page,
        max_pages_per_prefill=self._max_prefill_predict_length // self._tokens_per_page,
        pages_per_compute_block=self._pages_per_compute_block,
        num_kv_heads=self._num_kv_heads,
        kv_head_dim_size=self._head_dim,
        dtype=self._dtype,
    )

    query = jnp.ones((1, self._max_prefill_predict_length, self._num_query_heads, self._head_dim))
    key = jnp.ones((1, self._max_prefill_predict_length, self._num_kv_heads, self._head_dim))
    value = jnp.ones((1, self._max_prefill_predict_length, self._num_kv_heads, self._head_dim))

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((1, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.zeros(1, dtype=jnp.int32),
        num_pages_used=jnp.zeros(1, dtype=jnp.int32),
        current_page=jnp.zeros(1, dtype=jnp.int32),
        current_page_position=jnp.zeros(1, dtype=jnp.int32),
    )

    variables = attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )
    output_tuple, mutated_variables = attention_op.apply(
        variables, query, key, value, None, common_types.MODEL_MODE_PREFILL, slot=0, page_state=page_state, mutable=["cache"]
    )

    output, _, _ = output_tuple  # Unpack the tuple
    self.assertEqual(output.shape, (1, self._max_prefill_predict_length, self._num_query_heads, self._head_dim))

  @pytest.mark.tpu_only
  def test_paged_dot_product_attention_with_max_and_sum(self):
    query = jnp.ones((1, self._max_prefill_predict_length, self._num_query_heads, self._head_dim))
    key = jnp.ones((1, self._max_prefill_predict_length, self._num_kv_heads, self._head_dim))
    value = jnp.ones((1, self._max_prefill_predict_length, self._num_kv_heads, self._head_dim))

    output, max_vals, sum_vals = self.attention_op.paged_dot_product_attention_with_max_and_sum(query, key, value)
    self.assertEqual(output.shape, (1, self._max_prefill_predict_length, self._num_query_heads, self._head_dim))
    self.assertEqual(max_vals.shape[-1], 1)
    self.assertEqual(sum_vals.shape[-1], 1)

  @pytest.mark.tpu_only
  def test_update_prefill_step_pages(self):
    key = jnp.ones((1, self._max_prefill_predict_length, self._num_kv_heads, self._head_dim))
    value = jnp.ones((1, self._max_prefill_predict_length, self._num_kv_heads, self._head_dim))
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((1, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.zeros(1, dtype=jnp.int32),
        num_pages_used=jnp.zeros(1, dtype=jnp.int32),
        current_page=jnp.zeros(1, dtype=jnp.int32),
        current_page_position=jnp.zeros(1, dtype=jnp.int32),
    )
    variables = self.attention_op.init(
        self.rng,
        jnp.ones((1, 1, self._num_query_heads, self._head_dim)),
        key,
        value,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
    )

    # Use apply() to update the variable's value
    _, mutated_variables = self.attention_op.apply(
        variables,
        query=jnp.ones((1, 1, self._num_query_heads, self._head_dim)),
        key=key,  # Provide the key
        value=value,  # Provide the value
        decoder_segment_ids=None,
        # Provide a None or a suitable Array for decoder_segment_ids
        model_mode=common_types.MODEL_MODE_PREFILL,  # Provide the model mode
        previous_chunk=None,
        slot=0,
        page_state=page_state,  # Provide the page_state
        mutable=["cache"],
    )

    # Access the updated values from mutated_variables
    updated_key_pages_var = mutated_variables["cache"]["key_pages"]
    updated_value_pages_var = mutated_variables["cache"]["value_pages"]

    # Assertions using the updated variables
    self.assertEqual(
        updated_key_pages_var.value.shape,
        (
            self._num_kv_heads,
            self._max_prefill_predict_length // self._tokens_per_page,
            self._tokens_per_page,
            self._head_dim,
        ),
    )
    self.assertEqual(
        updated_value_pages_var.value.shape,
        (
            self._num_kv_heads,
            self._max_prefill_predict_length // self._tokens_per_page,
            self._tokens_per_page,
            self._head_dim,
        ),
    )

  @pytest.mark.tpu_only
  def test_update_decode_step_pages(self):
    """Test cache update during autoregressive generation."""
    batch_size = 1
    # Create distinctive key/value patterns
    rng1, rng2 = jax.random.split(self.rng)
    key = jax.random.normal(rng1, (batch_size, 1, self._num_kv_heads, self._head_dim))
    value = jax.random.normal(rng2, (batch_size, 1, self._num_kv_heads, self._head_dim))

    # Initialize page state at specific position
    test_page = 2
    test_position = 3
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.array([test_page], dtype=jnp.int32),
        current_page_position=jnp.array([test_position], dtype=jnp.int32),
    )

    # Initialize attention op and run update
    variables = self.attention_op.init(
        self.rng,
        query=key,  # Use key as query for initialization
        key=key,
        value=value,
        decoder_segment_ids=None,
        model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        previous_chunk=None,
        slot=0,
        page_state=page_state,
    )

    _, mutated_variables = self.attention_op.apply(
        variables,
        query=key,
        key=key,
        value=value,
        decoder_segment_ids=None,
        model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        previous_chunk=None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # Extract updated cache
    updated_key_pages = mutated_variables["cache"]["key_pages"]
    updated_value_pages = mutated_variables["cache"]["value_pages"]

    # Verify shapes
    self.assertEqual(
        updated_key_pages.value.shape,
        (self._num_kv_heads, self._num_pages, self._tokens_per_page, self._head_dim),
    )

    # Instead of trying to extract logical axes from the value,
    # verify against the attention op's configured axis names
    self.assertEqual(
        self.attention_op.kv_pages_axis_names, ("paged_kv_heads", "num_pages", "tokens_per_page", "paged_kv_head_dim_size")
    )

    # Verify key placement
    zeros = jnp.zeros_like(updated_key_pages.value)
    expected_key_pages = zeros.at[:, test_page, test_position, :].set(jnp.squeeze(key))
    np.testing.assert_allclose(updated_key_pages.value, expected_key_pages, rtol=1e-5, atol=1e-5)

    # Verify surrounding positions are unchanged
    np.testing.assert_allclose(
        updated_key_pages.value[:, test_page, test_position + 1 :, :],
        zeros[:, test_page, test_position + 1 :, :],
        rtol=1e-5,
        atol=1e-5,
    )

  @pytest.mark.tpu_only
  def test_prefill_attention(self):
    batch_size, seq_len, num_heads, head_dim = 1, self._max_prefill_predict_length, 8, 128
    query = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
    key = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((1, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(1, dtype=jnp.int32),
        current_page=jnp.zeros(1, dtype=jnp.int32),
        current_page_position=jnp.zeros(1, dtype=jnp.int32),
    )

    variables = self.attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )
    output_tuple, _ = self.attention_op.apply(
        variables,
        query,
        key,
        value,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )
    paged_output, max_vals, sum_vals = output_tuple

    # Normalize the output using the returned max and sum values
    paged_output = paged_output / (sum_vals + 1e-9)  # Add epsilon for numerical stability
    reference_output = reference_attention(query, key, value)

    np.testing.assert_allclose(paged_output, reference_output, rtol=1e-5, atol=1e-5)

  @pytest.mark.tpu_only
  def test_autoregressive_attention(self):
    batch_size, seq_len, num_heads, head_dim = 1, 1, 8, 128
    query = jax.random.normal(self.rng, (batch_size, 1, num_heads, head_dim))
    key = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((1, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(1, dtype=jnp.int32),
        current_page=jnp.zeros(1, dtype=jnp.int32),
        current_page_position=jnp.zeros(1, dtype=jnp.int32),
    )

    variables = self.attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_AUTOREGRESSIVE, None, slot=0, page_state=page_state
    )
    output_tuple, _ = self.attention_op.apply(
        variables,
        query,
        key,
        value,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # In autoregressive mode, normalization is handled internally
    paged_output, _, _ = output_tuple
    reference_output = reference_attention(query, key, value)
    np.testing.assert_allclose(paged_output, reference_output, rtol=1e-2, atol=1e-2)

  @pytest.mark.tpu_only
  def test_basic_prefill(self):
    """Test just the prefill operation without any AR steps."""
    batch_size = 1  # Prefill requires batch_size=1
    seq_len = self._max_prefill_predict_length
    num_heads = 8
    head_dim = 128

    # Create input sequence
    prefill_tokens = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    # Initialize page state
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Initialize attention op
    variables = self.attention_op.init(
        self.rng,
        prefill_tokens,
        prefill_tokens,
        prefill_tokens,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
    )

    output_tuple, _ = self.attention_op.apply(
        variables,
        prefill_tokens,
        prefill_tokens,
        prefill_tokens,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    paged_prefill_output, max_vals, sum_vals = output_tuple
    paged_prefill_output = paged_prefill_output / (sum_vals + 1e-9)

    # Compare with reference implementation
    reference_output = reference_attention(prefill_tokens, prefill_tokens, prefill_tokens)
    np.testing.assert_allclose(
        paged_prefill_output, reference_output, rtol=1e-2, atol=1e-2, err_msg="Prefill outputs don't match reference"
    )

  @pytest.mark.tpu_only
  def test_prefill_then_single_ar(self):
    """Test basic prefill followed by single AR step matches reference impl."""
    batch_size = 1
    prefill_len = self._max_prefill_predict_length
    num_heads = 8
    head_dim = 128

    # Create input sequence
    rng1, rng2 = jax.random.split(self.rng)
    prefill_tokens = jax.random.normal(rng1, (batch_size, prefill_len, num_heads, head_dim))

    # Initialize page state
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([prefill_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Initialize attention ops
    variables = self.attention_op.init(
        self.rng,
        prefill_tokens,
        prefill_tokens,
        prefill_tokens,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
    )

    output_tuple, mutated_vars = self.attention_op.apply(
        variables,
        prefill_tokens,
        prefill_tokens,
        prefill_tokens,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    prefill_output, max_vals, sum_vals = output_tuple
    prefill_output = prefill_output / (sum_vals + 1e-9)

    # Use updated variables for AR step
    variables = mutated_vars
    ar_token = jax.random.normal(rng2, (batch_size, 1, num_heads, head_dim))

    ar_output_tuple, _ = self.attention_op.apply(
        variables,
        ar_token,
        ar_token,
        ar_token,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    ar_output, _, _ = ar_output_tuple

    # Compare with reference implementation including prefill context
    full_sequence = jnp.concatenate([prefill_tokens, ar_token], axis=1)
    reference_ar_output = reference_attention(ar_token, full_sequence, full_sequence)

    np.testing.assert_allclose(ar_output, reference_ar_output, rtol=1e-2, atol=1e-2, err_msg="AR outputs don't match")

  @pytest.mark.tpu_only
  def test_basic_ar(self):
    """Test just the autoregressive operation with a single step."""
    batch_size = 1
    num_heads = 8
    head_dim = 128

    # Create separate random values for query/key/value
    rng1, rng2, rng3 = jax.random.split(self.rng, 3)
    query = jax.random.normal(rng1, (batch_size, 1, num_heads, head_dim))
    key = jax.random.normal(rng2, (batch_size, 1, num_heads, head_dim))
    value = jax.random.normal(rng3, (batch_size, 1, num_heads, head_dim))

    # Initialize page state with sequence length of 1
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
        # Start with length 1
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Initialize and apply attention
    variables = self.attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_AUTOREGRESSIVE, None, slot=0, page_state=page_state
    )

    output_tuple, _ = self.attention_op.apply(
        variables,
        query,
        key,
        value,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # AR mode returns (output, None, None)
    ar_output, _, _ = output_tuple

    # Compare with reference implementation
    reference_output = reference_attention(query, key, value)
    np.testing.assert_allclose(ar_output, reference_output, rtol=1e-2, atol=1e-2, err_msg="AR outputs don't match reference")

  @pytest.mark.tpu_only
  def test_paged_attention_single_token_batch(self):
    """Test attention with batch_size=1, seq_len=1 - smallest possible input."""
    batch_size = 1
    seq_len = self._tokens_per_page * 16
    query = jax.random.normal(self.rng, (batch_size, seq_len, self._num_query_heads, self._head_dim))
    key = jax.random.normal(self.rng, (batch_size, seq_len, self._num_kv_heads, self._head_dim))
    value = jax.random.normal(self.rng, (batch_size, seq_len, self._num_kv_heads, self._head_dim))

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    variables = self.attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )

    output_tuple, _ = self.attention_op.apply(
        variables,
        query,
        key,
        value,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    paged_output, max_vals, sum_vals = output_tuple
    # Normalize using returned values
    paged_output = paged_output / (sum_vals + 1e-9)

    reference_output = reference_attention(query, key, value)
    np.testing.assert_allclose(
        paged_output, reference_output, rtol=1e-5, atol=1e-5, err_msg="Single token attention outputs don't match reference"
    )

  @pytest.mark.tpu_only
  def test_attention_pattern_consistency(self):
    """Test attention pattern maintains consistency across prefill and autoregressive steps."""
    batch_size = 1
    seq_len = self._max_prefill_predict_length

    query = jax.random.normal(self.rng, (batch_size, seq_len, self._num_query_heads, self._head_dim))
    key = jax.random.normal(self.rng, (batch_size, seq_len, self._num_kv_heads, self._head_dim))
    value = jax.random.normal(self.rng, (batch_size, seq_len, self._num_kv_heads, self._head_dim))

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Run prefill
    variables = self.attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )
    output_tuple, mutated_vars = self.attention_op.apply(
        variables,
        query,
        key,
        value,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    prefill_output, _, _ = output_tuple
    reference_output = reference_attention(query, key, value)
    np.testing.assert_allclose(prefill_output, reference_output, rtol=1e-5, atol=1e-5)

    # Test single autoregressive step
    ar_query = jax.random.normal(self.rng, (batch_size, 1, self._num_query_heads, self._head_dim))
    ar_key = jax.random.normal(self.rng, (batch_size, 1, self._num_kv_heads, self._head_dim))
    ar_value = jax.random.normal(self.rng, (batch_size, 1, self._num_kv_heads, self._head_dim))

    ar_output_tuple, _ = self.attention_op.apply(
        mutated_vars,
        ar_query,
        ar_key,
        ar_value,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    ar_output, _, _ = ar_output_tuple

    # Compare against reference
    full_key = jnp.concatenate([key, ar_key], axis=1)
    full_value = jnp.concatenate([value, ar_value], axis=1)
    ar_reference = reference_attention(ar_query, full_key, full_value)
    assert ar_output.shape == ar_reference.shape
    np.testing.assert_allclose(ar_output, ar_reference, rtol=1e-2, atol=1e-2)

  @pytest.mark.tpu_only
  def test_sequential_page_updates(self):
    """Test multiple sequential page updates to verify cache consistency."""
    batch_size = 1
    seq_len = 1
    num_heads = 8
    head_dim = 128

    # Create initial key/value
    rng1, rng2 = jax.random.split(self.rng)
    key = jax.random.normal(rng1, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(rng2, (batch_size, seq_len, num_heads, head_dim))

    # Initialize page state for first position
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.array([0], dtype=jnp.int32),
        current_page_position=jnp.array([0], dtype=jnp.int32),
    )

    # Initialize attention op
    variables = self.attention_op.init(
        self.rng,
        key,
        key,
        value,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,  # Use as query too
    )

    # Perform multiple sequential updates
    num_updates = 3
    expected_values = []

    for i in range(num_updates):
      # Generate new key/value
      rng1, rng2 = jax.random.split(rng1)
      new_key = jax.random.normal(rng1, (batch_size, seq_len, num_heads, head_dim))
      new_value = jax.random.normal(rng2, (batch_size, seq_len, num_heads, head_dim))
      expected_values.append((new_key, new_value))

      # Update cache
      _, variables = self.attention_op.apply(
          variables,
          new_key,
          new_key,
          new_value,
          None,
          common_types.MODEL_MODE_AUTOREGRESSIVE,
          None,
          slot=0,
          page_state=page_state,
          mutable=["cache"],
      )

      # Update page state
      page_state = PageState(
          page_status=page_state.page_status,
          page_map=page_state.page_map,
          sequence_lengths=page_state.sequence_lengths + 1,
          num_pages_used=page_state.num_pages_used,
          current_page=page_state.current_page,
          current_page_position=jnp.array([i + 1], dtype=jnp.int32),
      )

    # Verify cache contents
    cache = variables["cache"]
    key_pages = cache["key_pages"]
    value_pages = cache["value_pages"]

    # Check each position
    for i, (expected_key, expected_value) in enumerate(expected_values):
      for head in range(num_heads):
        np.testing.assert_allclose(
            key_pages.value[head, 0, i],
            expected_key[0, 0, head],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Mismatch in key cache at position {i}, head {head}",
        )
        np.testing.assert_allclose(
            value_pages.value[head, 0, i],
            expected_value[0, 0, head],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Mismatch in value cache at position {i}, head {head}",
        )

  @pytest.mark.tpu_only
  def test_page_boundary_conditions(self):
    """Test attention computation across page boundaries."""
    batch_size = 1
    seq_len = self._tokens_per_page * 2  # Two pages exactly
    num_heads = 8
    head_dim = 128

    # Create attention op with exactly 2 pages for prefill
    attention_op = PagedAttentionOp(
        mesh=self.mesh,
        num_pages=self._num_pages,
        tokens_per_page=self._tokens_per_page,
        max_pages_per_slot=self._max_target_length // self._tokens_per_page,
        max_pages_per_prefill=2,  # Override to exactly what we need
        pages_per_compute_block=self._pages_per_compute_block,
        num_kv_heads=self._num_kv_heads,
        kv_head_dim_size=self._head_dim,
        dtype=self._dtype,
    )

    # Create distinct patterns for each page
    rng1, rng2 = jax.random.split(self.rng)
    query_page1 = jax.random.normal(rng1, (batch_size, self._tokens_per_page, num_heads, head_dim))
    query_page2 = jax.random.normal(rng2, (batch_size, self._tokens_per_page, num_heads, head_dim))
    query = jnp.concatenate([query_page1, query_page2], axis=1)

    key = query  # Use same patterns for key and value for simplicity
    value = query

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    variables = attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )

    output_tuple, _ = attention_op.apply(
        variables, query, key, value, None, common_types.MODEL_MODE_PREFILL, slot=0, page_state=page_state, mutable=["cache"]
    )

    output, max_vals, sum_vals = output_tuple
    output = output / (sum_vals + 1e-9)
    reference_output = reference_attention(query, key, value)

    # Test boundary attention patterns

    # 1. Check last token of first page
    boundary_idx = self._tokens_per_page
    np.testing.assert_allclose(
        output[:, boundary_idx - 1 : boundary_idx],
        reference_output[:, boundary_idx - 1 : boundary_idx],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Last token of first page doesn't match reference",
    )

    # 2. Check first token of second page
    np.testing.assert_allclose(
        output[:, boundary_idx : boundary_idx + 1],
        reference_output[:, boundary_idx : boundary_idx + 1],
        rtol=1e-5,
        atol=1e-5,
        err_msg="First token of second page doesn't match reference",
    )

    # 3. Check boundary transition
    window_size = 4  # Check 2 tokens on each side of boundary
    boundary_window = slice(boundary_idx - window_size // 2, boundary_idx + window_size // 2)
    np.testing.assert_allclose(
        output[:, boundary_window],
        reference_output[:, boundary_window],
        rtol=1e-5,
        atol=1e-5,
        err_msg="Attention pattern at page boundary doesn't match reference",
    )

    # 4. Verify no discontinuities at boundary
    attention_diff = jnp.abs(output[:, boundary_idx] - output[:, boundary_idx - 1])
    self.assertTrue(jnp.all(attention_diff < 1e3), "Detected unexpected discontinuity at page boundary")

    # 5. Verify overall output
    np.testing.assert_allclose(
        output, reference_output, rtol=1e-5, atol=1e-5, err_msg="Complete attention output doesn't match reference"
    )

  @pytest.mark.tpu_only
  def test_page_reuse(self):
    """Test page reuse after releasing pages."""
    batch_size = 1
    seq_len = 1
    num_heads = 8
    head_dim = 128

    # Initialize with one sequence
    key1 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
    value1 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    # Initialize page state for first sequence
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    variables = self.attention_op.init(
        self.rng, key1, key1, value1, None, common_types.MODEL_MODE_AUTOREGRESSIVE, None, slot=0, page_state=page_state
    )

    # Store first sequence
    _, mutated_vars = self.attention_op.apply(
        variables,
        key1,
        key1,
        value1,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # Create new sequence with different values
    key2 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
    value2 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    # Reset page state (simulating page release)
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Store second sequence in same location
    output_tuple, final_vars = self.attention_op.apply(
        mutated_vars,
        key2,
        key2,
        value2,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    output, _, _ = output_tuple
    reference_output = reference_attention(key2, key2, value2)

    # Verify second sequence is stored correctly
    np.testing.assert_allclose(
        output, reference_output, rtol=1e-2, atol=1e-2, err_msg="Page reuse produced incorrect attention output"
    )

  @pytest.mark.tpu_only
  def test_multi_head_consistency(self):
    """Test consistency across different attention heads."""
    batch_size = 1
    seq_len = self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Create input where each head gets different patterns
    query = jnp.stack(
        [jax.random.normal(self.rng, (batch_size, seq_len, head_dim)) * (i + 1) for i in range(num_heads)], axis=2
    )

    key = jnp.stack(
        [jax.random.normal(self.rng, (batch_size, seq_len, head_dim)) * (i + 1) for i in range(self._num_kv_heads)],
        axis=2,
    )

    value = jnp.stack(
        [jax.random.normal(self.rng, (batch_size, seq_len, head_dim)) * (i + 1) for i in range(self._num_kv_heads)],
        axis=2,
    )

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    variables = self.attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )

    output_tuple, _ = self.attention_op.apply(
        variables,
        query,
        key,
        value,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    output, max_vals, sum_vals = output_tuple
    output = output / (sum_vals + 1e-9)
    reference_output = reference_attention(query, key, value)

    # Check each head separately
    for head in range(num_heads):
      np.testing.assert_allclose(
          output[:, :, head, :],
          reference_output[:, :, head, :],
          rtol=1e-5,
          atol=1e-5,
          err_msg=f"Head {head} attention output doesn't match reference",
      )

  @pytest.mark.tpu_only
  def test_long_sequence_stability(self):
    """Test numerical stability with long sequences."""
    batch_size = 1
    seq_len = self._max_prefill_predict_length
    num_heads = 8
    head_dim = 128

    # Create sequence with large magnitude differences
    query = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim)) * 10
    key = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim)) * 10
    value = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    variables = self.attention_op.init(
        self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )

    output_tuple, _ = self.attention_op.apply(
        variables,
        query,
        key,
        value,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    output, max_vals, sum_vals = output_tuple
    output = output / (sum_vals + 1e-9)
    reference_output = reference_attention(query, key, value)

    # Check numerical stability
    np.testing.assert_allclose(
        output, reference_output, rtol=1e-5, atol=1e-5, err_msg="Long sequence attention is numerically unstable"
    )

    # Verify that max values aren't too large (check for overflow)
    self.assertTrue(jnp.all(jnp.abs(max_vals) < 1e5), "Attention weights may be experiencing numerical overflow")

  @pytest.mark.tpu_only
  def test_multi_slot_cache_isolation(self):
    """Test that cache entries remain isolated between slots."""
    num_slots = 4
    seq_len = self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Track outputs for comparison
    slot_outputs = []

    # Initialize single page state to handle all slots
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((num_slots, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(num_slots, dtype=jnp.int32) * seq_len,
        num_pages_used=jnp.zeros(num_slots, dtype=jnp.int32),
        current_page=jnp.arange(num_slots, dtype=jnp.int32),
        # Each slot gets its own initial page
        current_page_position=jnp.zeros(num_slots, dtype=jnp.int32),
    )

    # Create queries with distinct patterns for each slot
    queries = []
    for slot in range(num_slots):
      rng_slot = jax.random.fold_in(self.rng, slot)
      queries.append(jax.random.normal(rng_slot, (1, seq_len, num_heads, head_dim)) * (slot + 1))
    queries = jnp.concatenate(queries, axis=0)  # Combine into batch

    # Initialize and run attention
    variables = self.attention_op.init(
        self.rng, queries, queries, queries, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )

    output_tuple, _ = self.attention_op.apply(
        variables,
        queries,
        queries,
        queries,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # Verify outputs are different for each slot
    output = output_tuple[0]
    for i in range(num_slots - 1):
      for j in range(i + 1, num_slots):
        self.assertFalse(
            jnp.allclose(output[i], output[j], rtol=1e-5, atol=1e-5), f"Slots {i} and {j} produced identical outputs"
        )

  @pytest.mark.tpu_only
  def test_prefill_to_ar_transition_multi_slot(self):
    """Test transition from prefill to autoregressive generation across multiple slots."""
    num_slots = 4
    seq_len = self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Initialize single page state for all slots
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((num_slots, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(num_slots, dtype=jnp.int32) * seq_len,
        num_pages_used=jnp.zeros(num_slots, dtype=jnp.int32),
        current_page=jnp.arange(num_slots, dtype=jnp.int32),
        current_page_position=jnp.zeros(num_slots, dtype=jnp.int32),
    )

    # Create prefill inputs for all slots
    prefill_inputs = []
    for slot in range(num_slots):
      rng_slot = jax.random.fold_in(self.rng, slot)
      prefill_inputs.append(jax.random.normal(rng_slot, (1, seq_len, num_heads, head_dim)) * (slot + 1))
    prefill_inputs = jnp.concatenate(prefill_inputs, axis=0)

    # Do prefill for all slots
    variables = self.attention_op.init(
        self.rng,
        prefill_inputs,
        prefill_inputs,
        prefill_inputs,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
    )

    _, mutated_vars = self.attention_op.apply(
        variables,
        prefill_inputs,
        prefill_inputs,
        prefill_inputs,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # Create AR inputs for all slots
    ar_inputs = []
    for slot in range(num_slots):
      rng_slot = jax.random.fold_in(self.rng, slot + num_slots)
      ar_inputs.append(jax.random.normal(rng_slot, (1, 1, num_heads, head_dim)) * (slot + 1))
    ar_inputs = jnp.concatenate(ar_inputs, axis=0)

    # Update page state for AR step
    ar_page_state = PageState(
        page_status=page_state.page_status,
        page_map=page_state.page_map,
        sequence_lengths=page_state.sequence_lengths + 1,
        # Increment all lengths
        num_pages_used=page_state.num_pages_used,
        current_page=page_state.current_page,
        current_page_position=page_state.current_page_position + 1,
        # Increment all positions
    )

    ar_output_tuple, _ = self.attention_op.apply(
        mutated_vars,
        ar_inputs,
        ar_inputs,
        ar_inputs,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=ar_page_state,
        mutable=["cache"],
    )

    ar_outputs = ar_output_tuple[0]

    # Verify AR outputs differ between slots
    for i in range(num_slots - 1):
      for j in range(i + 1, num_slots):
        self.assertFalse(
            jnp.allclose(ar_outputs[i], ar_outputs[j], rtol=1e-5, atol=1e-5),
            f"AR outputs for slots {i} and {j} are identical",
        )

  @pytest.mark.tpu_only
  def test_cache_state_preservation(self):
    """Test cache state preservation across multiple AR steps."""
    num_slots = 4
    seq_len = self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim
    num_ar_steps = 3
    tokens_per_page = self._tokens_per_page

    # Initialize single page state for all slots
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((num_slots, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(num_slots, dtype=jnp.int32) * seq_len,
        num_pages_used=jnp.zeros(num_slots, dtype=jnp.int32),
        current_page=jnp.arange(num_slots, dtype=jnp.int32),
        # Each slot starts on a different page
        current_page_position=jnp.zeros(num_slots, dtype=jnp.int32),
    )

    # Create prefill inputs for all slots
    prefill_inputs = []
    for slot in range(num_slots):
      rng_slot = jax.random.fold_in(self.rng, slot)
      prefill_inputs.append(jax.random.normal(rng_slot, (1, seq_len, num_heads, head_dim)) * (slot + 1))
    prefill_inputs = jnp.concatenate(prefill_inputs, axis=0)

    # Do prefill
    variables = self.attention_op.init(
        self.rng,
        prefill_inputs,
        prefill_inputs,
        prefill_inputs,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
    )

    _, current_vars = self.attention_op.apply(
        variables,
        prefill_inputs,
        prefill_inputs,
        prefill_inputs,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # Track AR outputs per slot
    ar_outputs = [[] for _ in range(num_slots)]
    current_page_state = page_state

    # Track per-slot sequence lengths
    sequence_lengths = jnp.ones(num_slots, dtype=jnp.int32) * seq_len

    # Do AR steps
    for step in range(num_ar_steps):
      # Create AR inputs for all slots
      ar_inputs = []
      for slot in range(num_slots):
        rng_slot = jax.random.fold_in(self.rng, step * num_slots + slot)
        ar_inputs.append(
            jax.random.normal(rng_slot, (1, 1, num_heads, head_dim))
            * (slot + 1)
            * ((step + 1) * 100)  # Make patterns very distinct
        )
      ar_inputs = jnp.concatenate(ar_inputs, axis=0)

      # Calculate indices for each slot
      sequence_lengths = sequence_lengths + 1
      token_indices = sequence_lengths - 1
      page_indices = token_indices // tokens_per_page
      page_positions = token_indices % tokens_per_page

      # Update page mapping for new positions
      new_page_map = current_page_state.page_map
      for slot in range(num_slots):
        new_page_map = new_page_map.at[slot, page_indices[slot]].set(page_indices[slot])

        # Update pages used count
      new_pages_used = current_page_state.num_pages_used.at[:].set(
          jnp.maximum(current_page_state.num_pages_used, page_indices + 1)
      )

      # Create updated page state
      current_page_state = PageState(
          page_status=current_page_state.page_status,
          page_map=new_page_map,
          sequence_lengths=sequence_lengths,
          num_pages_used=new_pages_used,
          current_page=page_indices,
          current_page_position=page_positions,
      )

      # Verify page state is updating correctly
      for slot in range(num_slots):
        self.assertEqual(
            current_page_state.sequence_lengths[slot],
            seq_len + step + 1,
            f"Incorrect sequence length for slot {slot} at step {step}",
        )
        self.assertEqual(
            current_page_state.current_page_position[slot],
            page_positions[slot],
            f"Incorrect page position for slot {slot} at step {step}",
        )

      ar_output_tuple, current_vars = self.attention_op.apply(
          current_vars,
          ar_inputs,
          ar_inputs,
          ar_inputs,
          None,
          common_types.MODEL_MODE_AUTOREGRESSIVE,
          None,
          slot=0,
          page_state=current_page_state,
          mutable=["cache"],
      )

      # Store outputs per slot
      ar_output = ar_output_tuple[0]
      for slot in range(num_slots):
        ar_outputs[slot].append(ar_output[slot])

    # Verify outputs differ across steps for each slot
    for slot in range(num_slots):
      for step1 in range(num_ar_steps - 1):
        for step2 in range(step1 + 1, num_ar_steps):
          logits1 = ar_outputs[slot][step1]
          logits2 = ar_outputs[slot][step2]
          self.assertFalse(
              jnp.allclose(logits1, logits2, rtol=1e-5, atol=1e-5),
              f"AR outputs identical for slot {slot} at steps {step1} and {step2}",
          )

  @pytest.mark.tpu_only
  def test_multi_slot_token_generation(self):
    """Test token generation with proper sharding to verify cache consistency."""
    num_slots = self._per_device_batch_size * jax.device_count()
    seq_len = self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Create page state matching expected dimensions
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((num_slots, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(num_slots, dtype=jnp.int32) * seq_len,
        num_pages_used=jnp.zeros(num_slots, dtype=jnp.int32),
        current_page=jnp.arange(num_slots, dtype=jnp.int32),
        current_page_position=jnp.zeros(num_slots, dtype=jnp.int32),
    )

    # Create distinct prefill inputs per slot
    prefill_inputs = []
    for slot in range(num_slots):
      rng_slot = jax.random.fold_in(self.rng, slot)
      prefill_inputs.append(jax.random.normal(rng_slot, (1, seq_len, num_heads, head_dim)) * (slot + 1))
    prefill_input = jnp.concatenate(prefill_inputs, axis=0)

    # Initialize attention op
    variables = self.attention_op.init(
        self.rng,
        prefill_input,
        prefill_input,
        prefill_input,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
    )

    output_tuple, variables = self.attention_op.apply(
        variables,
        prefill_input,
        prefill_input,
        prefill_input,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    prefill_output = output_tuple[0]

    # Verify distinct outputs per slot after prefill
    for slot in range(num_slots - 1):
      self.assertFalse(
          jnp.allclose(prefill_output[slot], prefill_output[slot + 1], rtol=1e-5, atol=1e-5),
          f"Prefill outputs identical for slots {slot} and {slot + 1}",
      )

    # Test autoregressive generation
    num_generations = 3
    for i in range(num_generations):
      ar_input = jax.random.normal(jax.random.fold_in(self.rng, i), (num_slots, 1, num_heads, head_dim))

      ar_output_tuple, variables = self.attention_op.apply(
          variables,
          ar_input,
          ar_input,
          ar_input,
          None,
          common_types.MODEL_MODE_AUTOREGRESSIVE,
          None,
          slot=0,
          page_state=page_state,
          mutable=["cache"],
      )

      ar_output = ar_output_tuple[0]

      # Verify outputs remain distinct between slots
      for slot in range(num_slots - 1):
        self.assertFalse(
            jnp.allclose(ar_output[slot], ar_output[slot + 1], rtol=1e-5, atol=1e-5),
            f"Generation {i} produced identical outputs for slots {slot} and {slot + 1}",
        )

  @pytest.mark.tpu_only
  def test_slot_reuse_after_free(self):
    """Test that a freed slot can be reused without contamination from previous contents."""
    batch_size, seq_len = 1, self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Initialize with one sequence
    key1 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
    value1 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    # Initialize page state
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Store first sequence
    variables = self.attention_op.init(
        self.rng, key1, key1, value1, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )
    output_tuple1, vars1 = self.attention_op.apply(
        variables,
        key1,
        key1,
        value1,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )
    first_output = output_tuple1[0]

    # Capture cache state before release
    pre_release_cache = vars1["cache"]
    print("Pre-release cache keys:", pre_release_cache.keys())
    print("Key pages shape:", pre_release_cache["key_pages"].value.shape)

    # Release slot
    page_state = self.attention_op.release_slot(0, page_state)

    # Verify page state is cleaned
    print("\nPage state after release:")
    print("sequence_lengths:", page_state.sequence_lengths)
    print("num_pages_used:", page_state.num_pages_used)
    print("page_map:", page_state.page_map)

    # Create new sequence with very different values
    key2 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim)) * 100.0  # Scale up significantly
    value2 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim)) * 100.0

    # Store second sequence in same slot
    output_tuple2, vars2 = self.attention_op.apply(
        vars1,
        key2,
        key2,
        value2,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )
    second_output = output_tuple2[0]

    # Print cache states
    print("\nPost-second sequence cache keys:", vars2["cache"].keys())
    print("Input scale difference:", jnp.mean(jnp.abs(key2 - key1)))
    print("Output scale difference:", jnp.mean(jnp.abs(second_output - first_output)))

    # Compare outputs element-wise
    output_diff = second_output - first_output
    print("\nOutput statistics:")
    print("Min diff:", jnp.min(output_diff))
    print("Max diff:", jnp.max(output_diff))
    print("Mean diff:", jnp.mean(output_diff))

    # Verify outputs are different
    self.assertFalse(
        jnp.allclose(first_output, second_output, rtol=1e-5, atol=1e-5),
        "Second sequence output shows contamination from first sequence",
    )

  @pytest.mark.tpu_only
  def test_basic_slot_cleanup(self):
    """Test that releasing a slot properly cleans up its pages in KV cache."""
    batch_size = 1
    seq_len = self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Initialize page state
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Initial sequence
    x1 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    # Initialize and store first sequence
    variables = self.attention_op.init(
        self.rng, x1, x1, x1, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )

    # Get initial KV cache state
    initial_cache = variables["cache"]
    initial_k_pages = initial_cache["key_pages"].value.copy()
    initial_v_pages = initial_cache["value_pages"].value.copy()

    # Store initial page map state
    initial_page_map = page_state.page_map.copy()

    # Release the slot
    page_state = self.attention_op.release_slot(0, page_state)

    # Check page state was reset
    self.assertEqual(page_state.sequence_lengths[0], 0)
    self.assertEqual(page_state.num_pages_used[0], 0)
    used_pages = page_state.page_map[0][page_state.page_map[0] > 0]
    self.assertEqual(len(used_pages), 0)

    # Check KV cache pages marked as free are actually zeroed/cleaned
    current_k_pages = variables["cache"]["key_pages"].value
    current_v_pages = variables["cache"]["value_pages"].value

    # For pages that were in use and then freed
    prev_used_pages = jnp.where(initial_page_map[0] > 0)[0]
    for page in prev_used_pages:
      zeros = jnp.zeros_like(current_k_pages[:, page])
      np.testing.assert_array_equal(current_k_pages[:, page], zeros, "Key cache not cleaned for freed page")
      np.testing.assert_array_equal(current_v_pages[:, page], zeros, "Value cache not cleaned for freed page")

  @pytest.mark.tpu_only
  def test_single_slot_content_preservation(self):
    """Test that content generated in a slot remains consistent and unique."""
    batch_size = 1
    seq_len = self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Initialize page state
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Create initial input
    x1 = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

    # Initialize
    variables = self.attention_op.init(
        self.rng, x1, x1, x1, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )

    # Store sequence of outputs for examination
    outputs = []

    # Do prefill
    output_tuple, vars = self.attention_op.apply(
        variables, x1, x1, x1, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state, mutable=["cache"]
    )
    outputs.append(output_tuple[0])
    variables = vars

    # Generate sequence of 16 tokens
    # In real runs we see 16-token sequences repeating
    num_generations = 16
    for i in range(num_generations):
      # Generate with slightly different input each time
      gen_input = jax.random.normal(jax.random.fold_in(self.rng, i), (batch_size, 1, num_heads, head_dim)) * (
          i + 1
      )  # Scale to make patterns distinct

      output_tuple, vars = self.attention_op.apply(
          variables,
          gen_input,
          gen_input,
          gen_input,
          None,
          common_types.MODEL_MODE_AUTOREGRESSIVE,
          None,
          slot=0,
          page_state=page_state,
          mutable=["cache"],
      )
      outputs.append(output_tuple[0])
      variables = vars

      # Verify the new output is different from all previous outputs
      for j in range(len(outputs) - 1):
        self.assertFalse(
            jnp.allclose(outputs[-1], outputs[j], rtol=1e-5, atol=1e-5),
            f"Generation {i} produced output identical to generation {j}",
        )

      # Also verify attention pattern is using full context
      # Get logits portion of output and verify non-zero attention weights
      # across multiple positions
      logits = output_tuple[0]
      attention_weights = jnp.sum(jnp.abs(logits), axis=-1)
      num_attended = jnp.sum(attention_weights > 1e-6)
      self.assertGreater(num_attended, 1, f"Generation {i} only attending to single position")

  @pytest.mark.tpu_only
  def test_kv_cache_persistence(self):
    """Test that KV cache contents persist correctly between generations."""
    batch_size = 1
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Start with single token
    x = jax.random.normal(self.rng, (batch_size, 1, num_heads, head_dim))

    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    variables = self.attention_op.init(
        self.rng, x, x, x, None, common_types.MODEL_MODE_AUTOREGRESSIVE, None, slot=0, page_state=page_state
    )

    # Store initial KV cache state
    initial_k_pages = variables["cache"]["key_pages"].value.copy()

    # Do generation
    output_tuple, variables = self.attention_op.apply(
        variables,
        x,
        x,
        x,
        None,
        common_types.MODEL_MODE_AUTOREGRESSIVE,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )

    # Verify initial content preserved in unused pages
    current_k_pages = variables["cache"]["key_pages"].value
    # Check that pages not in page_map are unchanged
    unused_pages = jnp.where(page_state.page_map[0] == 0)[0]
    for page in unused_pages:
      np.testing.assert_array_equal(initial_k_pages[:, page], current_k_pages[:, page], "KV cache modified in unused pages")

  # TODO(wyzhang): Fix the following broken test
  @pytest.mark.tpu_only
  @unittest.skip("This test is temporarily disabled. To be fixed")
  def test_slot_reuse_attention_computation(self):
    """Test that reused slot properly computes attention with new content."""
    batch_size, seq_len = 1, self._max_prefill_predict_length
    num_heads = self._num_query_heads
    head_dim = self._head_dim

    # Create structured input where we can verify attention is working
    key1 = jnp.ones((batch_size, seq_len, num_heads, head_dim))  # All ones
    value1 = jnp.ones((batch_size, seq_len, num_heads, head_dim))

    # Initialize page state
    page_state = PageState(
        page_status=jnp.zeros(self._num_pages, dtype=jnp.int32),
        page_map=jnp.zeros((batch_size, self._num_pages), dtype=jnp.int32),
        sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
        num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page=jnp.zeros(batch_size, dtype=jnp.int32),
        current_page_position=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    # Store first sequence
    variables = self.attention_op.init(
        self.rng, key1, key1, value1, None, common_types.MODEL_MODE_PREFILL, None, slot=0, page_state=page_state
    )
    output_tuple1, vars1 = self.attention_op.apply(
        variables,
        key1,
        key1,
        value1,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )
    first_output = output_tuple1[0]

    # Release slot
    page_state = self.attention_op.release_slot(0, page_state)

    # Create new sequence with specific pattern to detect attention
    key2 = jnp.zeros((batch_size, seq_len, num_heads, head_dim))  # All zeros except...
    key2 = key2.at[:, 0, :, :].set(1.0)  # First position is ones
    value2 = jnp.ones((batch_size, seq_len, num_heads, head_dim)) * 2.0  # All twos

    # Store second sequence in same slot
    output_tuple2, vars2 = self.attention_op.apply(
        vars1,
        key2,
        key2,
        value2,
        None,
        common_types.MODEL_MODE_PREFILL,
        None,
        slot=0,
        page_state=page_state,
        mutable=["cache"],
    )
    second_output = output_tuple2[0]

    # If attention is working:
    # - First sequence should attend equally to all positions (all ones)
    # - Second sequence should attend mainly to first position (only non-zero)

    print("First output mean per position:", jnp.mean(first_output, axis=(0, 2, 3)))
    print("Second output mean per position:", jnp.mean(second_output, axis=(0, 2, 3)))

    # We expect second output to have larger values at early positions
    # if attention is working properly
    second_means = jnp.mean(second_output, axis=(0, 2, 3))
    self.assertGreater(second_means[0], second_means[-1], "Second sequence not showing position-dependent attention")


if __name__ == "__main__":
  unittest.main()
