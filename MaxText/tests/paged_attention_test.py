
import unittest
import jax
import numpy as np
import jax.numpy as jnp
from flax.core import freeze
from flax import linen as nn
from layers.attentions import PagedAttentionOp, Attention
from page_managers import PageManager, PageState
import common_types


def reference_attention(query, key, value):
    attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    
    return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)

class PagedAttentionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'num_query_heads': 8,
            'num_kv_heads': 8,
            'head_dim': 128,
            'max_prefill_predict_length': 512,
            'max_target_length': 1024,
            'block_size': 256,
            'num_pages': 64,
            'page_size': 32,
            'dtype': jnp.float32,
        }
        self.rng = jax.random.PRNGKey(42)
        devices = jax.devices()
        if len(devices) > 1:
            self.mesh = jax.sharding.Mesh(devices, axis_names=('data',))
        else:
            # Fallback for single-device testing
            self.mesh = jax.sharding.Mesh(devices, axis_names=()) 
        self.attention_op = PagedAttentionOp(
            mesh=self.mesh,
            num_pages=self.cfg['num_pages'],
            page_size=self.cfg['page_size'],
            max_pages_per_slot=self.cfg['max_target_length'] // self.cfg['page_size'],
            max_pages_per_prefill=self.cfg['max_prefill_predict_length'] // self.cfg['page_size'],
            block_size=self.cfg['block_size'],
            pages_per_compute_block=self.cfg['block_size'] // self.cfg['page_size'],
            num_kv_heads=self.cfg['num_kv_heads'],
            kv_head_dim_size=self.cfg['head_dim'],
            dtype=self.cfg['dtype'],
        )

    def test_paged_attention_output_shape(self):
        attention_op = PagedAttentionOp(
            mesh=self.mesh,
            num_pages=self.cfg['num_pages'],
            page_size=self.cfg['page_size'],
            max_pages_per_slot=self.cfg['max_target_length'] // self.cfg['page_size'],
            max_pages_per_prefill=self.cfg['max_prefill_predict_length'] // self.cfg['page_size'],
            block_size=self.cfg['block_size'],
            pages_per_compute_block=self.cfg['block_size'] // self.cfg['page_size'],
            num_kv_heads=self.cfg['num_kv_heads'],
            kv_head_dim_size=self.cfg['head_dim'],
            dtype=self.cfg['dtype'],
        )

        query = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))
        key = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))
        value = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))

        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((1, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.zeros(1, dtype=jnp.int32),
            num_pages_used=jnp.zeros(1, dtype=jnp.int32),
            current_page=jnp.zeros(1, dtype=jnp.int32),
            current_page_position=jnp.zeros(1, dtype=jnp.int32)
        )

        variables = attention_op.init(self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, page_state)
        output_tuple, mutated_variables = attention_op.apply(
            variables,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]
        )
        
        output, _, _ = output_tuple  # Unpack the tuple
        self.assertEqual(output.shape, (1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))


    def test_paged_dot_product_attention_with_max_and_sum(self):
        query = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))
        key = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))
        value = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))

        output, max_vals, sum_vals = self.attention_op.paged_dot_product_attention_with_max_and_sum(query, key, value)
        self.assertEqual(output.shape, (1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))
        self.assertEqual(max_vals.shape[-1], 1)  # Check max values shape
        self.assertEqual(sum_vals.shape[-1], 1)  # Check sum values shape

    def test_update_prefill_step_pages(self):
        key = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))
        value = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))
        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((1, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.zeros(1, dtype=jnp.int32),
            num_pages_used=jnp.zeros(1, dtype=jnp.int32),
            current_page=jnp.zeros(1, dtype=jnp.int32),
            current_page_position=jnp.zeros(1, dtype=jnp.int32)
        )
        variables = self.attention_op.init(self.rng, jnp.ones((1, 1, self.cfg['num_query_heads'], self.cfg['head_dim'])), key, value, None, common_types.MODEL_MODE_PREFILL, page_state)

        # Use apply() to update the variable's value
        _, mutated_variables = self.attention_op.apply(
            variables,
            query=jnp.ones((1, 1, self.cfg['num_query_heads'], self.cfg['head_dim'])),
            key=key,  # Provide the key
            value=value,  # Provide the value
            decoder_segment_ids=None,  # Provide a None or a suitable Array for decoder_segment_ids
            model_mode=common_types.MODEL_MODE_PREFILL,  # Provide the model mode
            page_state=page_state,  # Provide the page_state
            mutable=["cache"]
        )

        # Access the updated values from mutated_variables
        updated_key_pages_var = mutated_variables['cache']['key_pages']
        updated_value_pages_var = mutated_variables['cache']['value_pages']

        # Assertions using the updated variables
        self.assertEqual(updated_key_pages_var.value.shape, (self.cfg['num_kv_heads'], self.cfg['max_prefill_predict_length'] // self.cfg['page_size'], self.cfg['page_size'], self.cfg['head_dim']))
        self.assertEqual(updated_value_pages_var.value.shape, (self.cfg['num_kv_heads'], self.cfg['max_prefill_predict_length'] // self.cfg['page_size'], self.cfg['page_size'], self.cfg['head_dim']))
    

    def test_update_decode_step_pages(self):
        """Test cache update during autoregressive generation."""
        batch_size = 1
        # Create distinctive key/value patterns
        rng1, rng2 = jax.random.split(self.rng)
        key = jax.random.normal(rng1, 
                            (batch_size, 1, self.cfg['num_kv_heads'], 
                                self.cfg['head_dim']))
        value = jax.random.normal(rng2,
                                (batch_size, 1, self.cfg['num_kv_heads'], 
                                self.cfg['head_dim']))

        # Initialize page state at specific position
        test_page = 2
        test_position = 3
        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((batch_size, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.array([test_page], dtype=jnp.int32),
            current_page_position=jnp.array([test_position], dtype=jnp.int32)
        )

        # Initialize attention op and run update
        variables = self.attention_op.init(
            self.rng,
            query=key,  # Use key as query for initialization
            key=key,
            value=value,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state=page_state
        )

        _, mutated_variables = self.attention_op.apply(
            variables,
            query=key,
            key=key,
            value=value,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state=page_state,
            mutable=["cache"]
        )

        # Extract updated cache
        updated_key_pages = mutated_variables['cache']['key_pages']
        updated_value_pages = mutated_variables['cache']['value_pages']

        # Verify shapes
        self.assertEqual(updated_key_pages.value.shape,
                        (self.cfg['num_kv_heads'], self.cfg['num_pages'], 
                        self.cfg['page_size'], self.cfg['head_dim']))

        # Instead of trying to extract logical axes from the value,
        # verify against the attention op's configured axis names
        self.assertEqual(
            self.attention_op.kv_pages_axis_names,
            ("paged_kv_heads", "num_pages", "page_size", "paged_kv_head_dim_size")
        )

        # Verify key placement
        zeros = jnp.zeros_like(updated_key_pages.value)
        expected_key_pages = zeros.at[:,test_page,test_position,:].set(
            jnp.squeeze(key))
        np.testing.assert_allclose(
            updated_key_pages.value,
            expected_key_pages,
            rtol=1e-5, atol=1e-5
        )

        # Verify surrounding positions are unchanged
        np.testing.assert_allclose(
            updated_key_pages.value[:,test_page,test_position+1:,:],
            zeros[:,test_page,test_position+1:,:],
            rtol=1e-5, atol=1e-5
        )
    
    
    def test_prefill_attention(self):
        batch_size, seq_len, num_heads, head_dim = 1, self.cfg['max_prefill_predict_length'], 8, 128
        query = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
        key = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
        value = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((1, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
            num_pages_used=jnp.zeros(1, dtype=jnp.int32),
            current_page=jnp.zeros(1, dtype=jnp.int32),
            current_page_position=jnp.zeros(1, dtype=jnp.int32)
        )

        variables = self.attention_op.init(self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, page_state)
        output_tuple, _ = self.attention_op.apply(
            variables, query, key, value, None, common_types.MODEL_MODE_PREFILL, page_state, mutable=["cache"]
        )
        paged_output, max_vals, sum_vals = output_tuple

        # Normalize the output using the returned max and sum values
        paged_output = paged_output / (sum_vals + 1e-9)  # Add epsilon for numerical stability
        reference_output = reference_attention(query, key, value)
        
        np.testing.assert_allclose(paged_output, reference_output, rtol=1e-5, atol=1e-5)

    def test_autoregressive_attention(self):
        batch_size, seq_len, num_heads, head_dim = 1, 1, 8, 128
        query = jax.random.normal(self.rng, (batch_size, 1, num_heads, head_dim))
        key = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
        value = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))

        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((1, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
            num_pages_used=jnp.zeros(1, dtype=jnp.int32),
            current_page=jnp.zeros(1, dtype=jnp.int32),
            current_page_position=jnp.zeros(1, dtype=jnp.int32)
        )

        variables = self.attention_op.init(self.rng, query, key, value, None, common_types.MODEL_MODE_AUTOREGRESSIVE, page_state)
        output_tuple, _ = self.attention_op.apply(
            variables, query, key, value, None, common_types.MODEL_MODE_AUTOREGRESSIVE, page_state, mutable=["cache"]
        )
        
        # In autoregressive mode, normalization is handled internally
        paged_output, _, _ = output_tuple
        reference_output = reference_attention(query, key, value)
        np.testing.assert_allclose(paged_output, reference_output, rtol=1e-2, atol=1e-2)

    def test_basic_prefill(self):
        """Test just the prefill operation without any AR steps."""
        batch_size = 1  # Prefill requires batch_size=1
        seq_len = self.cfg['max_prefill_predict_length']
        num_heads = 8
        head_dim = 128
        
        # Create input sequence
        prefill_tokens = jax.random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
        
        # Initialize page state
        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((batch_size, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
        )
        
        # Initialize attention op
        variables = self.attention_op.init(
            self.rng,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state
        )

        output_tuple, _ = self.attention_op.apply(
            variables,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]
        )
        
        paged_prefill_output, max_vals, sum_vals = output_tuple
        paged_prefill_output = paged_prefill_output / (sum_vals + 1e-9)
        
        # Compare with reference implementation
        reference_output = reference_attention(prefill_tokens, prefill_tokens, prefill_tokens)
        np.testing.assert_allclose(
            paged_prefill_output,
            reference_output,
            rtol=1e-2, 
            atol=1e-2,
            err_msg="Prefill outputs don't match reference"
        )
    
    def test_prefill_then_single_ar(self):
        """Test basic prefill followed by single AR step matches reference impl."""
        batch_size = 1
        prefill_len = self.cfg['max_prefill_predict_length'] 
        num_heads = 8
        head_dim = 128
        
        # Create input sequence
        rng1, rng2 = jax.random.split(self.rng)
        prefill_tokens = jax.random.normal(rng1, (batch_size, prefill_len, num_heads, head_dim))
        
        # Initialize page state
        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((batch_size, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.array([prefill_len], dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
        )
        
        # Initialize attention ops
        variables = self.attention_op.init(
            self.rng,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state
        )

        output_tuple, mutated_vars = self.attention_op.apply(
            variables,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]
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
            page_state,
            mutable=["cache"]
        )

        ar_output, _, _ = ar_output_tuple

        # Compare with reference implementation including prefill context
        full_sequence = jnp.concatenate([prefill_tokens, ar_token], axis=1)
        reference_ar_output = reference_attention(
            ar_token,
            full_sequence,
            full_sequence
        )
        
        np.testing.assert_allclose(
            ar_output,
            reference_ar_output,
            rtol=1e-2,
            atol=1e-2,
            err_msg="AR outputs don't match"
        )

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
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((batch_size, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),  # Start with length 1
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
        )
        
        # Initialize and apply attention
        variables = self.attention_op.init(
            self.rng,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state
        )

        output_tuple, _ = self.attention_op.apply(
            variables,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state,
            mutable=["cache"]
        )
        
        # AR mode returns (output, None, None)
        ar_output, _, _ = output_tuple
        
        # Compare with reference implementation 
        reference_output = reference_attention(query, key, value)
        np.testing.assert_allclose(
            ar_output,
            reference_output,
            rtol=1e-2,
            atol=1e-2,
            err_msg="AR outputs don't match reference"
        )

    def test_paged_attention_single_token_batch(self):
        """Test attention with batch_size=1, seq_len=1 - smallest possible input."""
        batch_size = 1
        seq_len = self.cfg['page_size'] * 16
        query = jax.random.normal(self.rng, (batch_size, seq_len, self.cfg['num_query_heads'], self.cfg['head_dim']))
        key = jax.random.normal(self.rng, (batch_size, seq_len, self.cfg['num_kv_heads'], self.cfg['head_dim']))
        value = jax.random.normal(self.rng, (batch_size, seq_len, self.cfg['num_kv_heads'], self.cfg['head_dim']))

        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((batch_size, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
        )

        variables = self.attention_op.init(
            self.rng,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state
        )

        output_tuple, _ = self.attention_op.apply(
            variables,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]
        )
        
        paged_output, max_vals, sum_vals = output_tuple
        # Normalize using returned values
        paged_output = paged_output / (sum_vals + 1e-9)

        reference_output = reference_attention(query, key, value)
        np.testing.assert_allclose(
            paged_output,
            reference_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Single token attention outputs don't match reference"
        )
    
    def test_attention_pattern_consistency(self):
        """Test attention pattern maintains consistency across prefill and autoregressive steps."""
        batch_size = 1
        seq_len = self.cfg['max_prefill_predict_length']
        
        query = jax.random.normal(self.rng, (batch_size, seq_len, self.cfg['num_query_heads'], self.cfg['head_dim']))
        key = jax.random.normal(self.rng, (batch_size, seq_len, self.cfg['num_kv_heads'], self.cfg['head_dim']))
        value = jax.random.normal(self.rng, (batch_size, seq_len, self.cfg['num_kv_heads'], self.cfg['head_dim']))

        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((batch_size, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.array([seq_len], dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page_position=jnp.zeros(batch_size, dtype=jnp.int32)
        )

        # Run prefill
        variables = self.attention_op.init(self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, page_state)
        output_tuple, mutated_vars = self.attention_op.apply(
            variables, query, key, value, None, common_types.MODEL_MODE_PREFILL, page_state, mutable=["cache"]
        )
        
        prefill_output, _, _ = output_tuple
        reference_output = reference_attention(query, key, value)
        np.testing.assert_allclose(prefill_output, reference_output, rtol=1e-5, atol=1e-5)

        # Test single autoregressive step
        ar_query = jax.random.normal(self.rng, (batch_size, 1, self.cfg['num_query_heads'], self.cfg['head_dim']))
        ar_key = jax.random.normal(self.rng, (batch_size, 1, self.cfg['num_kv_heads'], self.cfg['head_dim']))
        ar_value = jax.random.normal(self.rng, (batch_size, 1, self.cfg['num_kv_heads'], self.cfg['head_dim']))

        ar_output_tuple, _ = self.attention_op.apply(
            mutated_vars, ar_query, ar_key, ar_value, None, common_types.MODEL_MODE_AUTOREGRESSIVE, page_state, mutable=["cache"]
        )
        
        ar_output, _, _ = ar_output_tuple

        # Compare against reference
        full_key = jnp.concatenate([key, ar_key], axis=1)
        full_value = jnp.concatenate([value, ar_value], axis=1)
        ar_reference = reference_attention(ar_query, full_key, full_value)
        assert ar_output.shape == ar_reference.shape
        np.testing.assert_allclose(ar_output, ar_reference, rtol=1e-2, atol=1e-2)
    
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
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((batch_size, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.ones(batch_size, dtype=jnp.int32),
            num_pages_used=jnp.zeros(batch_size, dtype=jnp.int32),
            current_page=jnp.array([0], dtype=jnp.int32),
            current_page_position=jnp.array([0], dtype=jnp.int32)
        )

        # Initialize attention op
        variables = self.attention_op.init(
            self.rng,
            key,  # Use as query too
            key,
            value,
            None,
            common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state
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
                page_state,
                mutable=["cache"]
            )
            
            # Update page state
            page_state = PageState(
                page_status=page_state.page_status,
                page_map=page_state.page_map,
                sequence_lengths=page_state.sequence_lengths + 1,
                num_pages_used=page_state.num_pages_used,
                current_page=page_state.current_page,
                current_page_position=jnp.array([i + 1], dtype=jnp.int32)
            )
        
        # Verify cache contents
        cache = variables['cache']
        key_pages = cache['key_pages']
        value_pages = cache['value_pages']
        
        # Check each position
        for i, (expected_key, expected_value) in enumerate(expected_values):
            for head in range(num_heads):
                np.testing.assert_allclose(
                    key_pages.value[head, 0, i],
                    expected_key[0, 0, head],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Mismatch in key cache at position {i}, head {head}"
                )
                np.testing.assert_allclose(
                    value_pages.value[head, 0, i],
                    expected_value[0, 0, head],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Mismatch in value cache at position {i}, head {head}"
                )

if __name__ == "__main__":
    unittest.main()
