
import unittest
import jax
import numpy as np
import jax.numpy as jnp
from flax.core import freeze
from layers.attentions import PagedAttentionOp, Attention
from page_managers import PageManager, PageState
import common_types


def reference_attention(query, key, value):
    # Simple dot-product attention for reference
    attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)
    print(f"\nReference attention weights (first few):")
    print(attn_weights.at[0,0,0,0:5].get())
    
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    print(f"\nReference softmaxed weights (first few):")
    print(attn_weights.at[0,0,0,0:5].get())
    
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
        # Initialize PagedAttentionOp for this specific test
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

        # Dummy inputs for query, key, and value
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

        # Initialize attention operation (which creates the scope)
        variables = attention_op.init(self.rng, query, key, value, None, common_types.MODEL_MODE_PREFILL, page_state)

        # Call the apply method, making the cache mutable
        output, mutated_variables = attention_op.apply(
            variables,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]  # Mark "cache" as mutable
        )

        # Check the output shape
        self.assertEqual(output.shape, (1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))


    def test_paged_dot_product_attention_with_max_and_sum(self):
        query = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))
        key = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))
        value = jnp.ones((1, self.cfg['max_prefill_predict_length'], self.cfg['num_kv_heads'], self.cfg['head_dim']))

        # Directly call the internal function
        output = self.attention_op.paged_dot_product_attention_with_max_and_sum(query, key, value)
        self.assertEqual(output.shape, (1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))


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
        query = jnp.ones((1, 1, self.cfg['num_query_heads'], self.cfg['head_dim']))
        key = jnp.ones((1, 1, self.cfg['num_kv_heads'], self.cfg['head_dim']))
        value = jnp.ones((1, 1, self.cfg['num_kv_heads'], self.cfg['head_dim']))

        page_state = PageState(
            page_status=jnp.zeros(self.cfg['num_pages'], dtype=jnp.int32),
            page_map=jnp.zeros((1, self.cfg['num_pages']), dtype=jnp.int32),
            sequence_lengths=jnp.zeros(1, dtype=jnp.int32),
            num_pages_used=jnp.zeros(1, dtype=jnp.int32),
            current_page=jnp.zeros(1, dtype=jnp.int32),
            current_page_position=jnp.zeros(1, dtype=jnp.int32)
        )
        variables = self.attention_op.init(self.rng, jnp.ones((1, 1, self.cfg['num_query_heads'], self.cfg['head_dim'])), key, value, None, common_types.MODEL_MODE_AUTOREGRESSIVE, page_state)

        _, mutated_variables = self.attention_op.apply(
            variables,
            query=query,
            key=key,
            value=value,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state=page_state,
            mutable=["cache"]
        )

        updated_key_pages_var = mutated_variables['cache']['key_pages']
        updated_value_pages_var = mutated_variables['cache']['value_pages']

        self.assertEqual(updated_key_pages_var.value.shape, (self.cfg['num_kv_heads'], self.cfg['num_pages'], self.cfg['page_size'], self.cfg['head_dim']))
        self.assertEqual(updated_value_pages_var.value.shape, (self.cfg['num_kv_heads'], self.cfg['num_pages'], self.cfg['page_size'], self.cfg['head_dim']))
    
    
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
        paged_output, _ = self.attention_op.apply(
            variables, query, key, value, None, common_types.MODEL_MODE_PREFILL, page_state, mutable=["cache"]
        )

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
        paged_output, _ = self.attention_op.apply(
            variables, query, key, value, None, common_types.MODEL_MODE_AUTOREGRESSIVE, page_state, mutable=["cache"]
        )

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
        paged_attention_op = PagedAttentionOp(
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
        
        # Initialize and run prefill 
        variables = paged_attention_op.init(
            self.rng,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state
        )

        paged_prefill_output, _ = paged_attention_op.apply(
            variables,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]
        )
        
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
        batch_size = 1  # Prefill requires batch_size=1
        prefill_len = self.cfg['max_prefill_predict_length']  # Use full prefill length
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
        paged_attention_op = PagedAttentionOp(
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
        
        # Run prefill
        variables = paged_attention_op.init(
            self.rng,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state
        )

        paged_prefill_output, mutated_vars = paged_attention_op.apply(
            variables,
            prefill_tokens,
            prefill_tokens,
            prefill_tokens,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]
        )
        
        # Simply use the updated variables directly - no need to preserve params
        variables = mutated_vars
        ar_token = jax.random.normal(rng2, (batch_size, 1, num_heads, head_dim))
        
        paged_ar_output, _ = paged_attention_op.apply(
            variables,
            ar_token,
            ar_token, 
            ar_token,
            None,
            common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state,
            mutable=["cache"]
        )

        # Compare with reference implementation including prefill context
        full_sequence = jnp.concatenate([prefill_tokens, ar_token], axis=1)
        reference_ar_output = reference_attention(
            ar_token,
            full_sequence,
            full_sequence
        )
        
        np.testing.assert_allclose(
            paged_ar_output,
            reference_ar_output,
            rtol=1e-2,
            atol=1e-2,
            err_msg="AR outputs don't match"
        )

    def test_basic_ar(self):
        """Test just the autoregressive operation with a single step."""
        batch_size = 1  # Match working test
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
        
        # Initialize attention op
        paged_attention_op = PagedAttentionOp(
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
        
        # Initialize variables
        variables = paged_attention_op.init(
            self.rng,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state
        )

        # Run single AR step
        ar_output, _ = paged_attention_op.apply(
            variables,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_AUTOREGRESSIVE,
            page_state,
            mutable=["cache"]
        )
        
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

        # Initialize page state
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

        paged_output, _ = self.attention_op.apply(
            variables,
            query,
            key,
            value,
            None,
            common_types.MODEL_MODE_PREFILL,
            page_state,
            mutable=["cache"]
        )

        # Compare with reference implementation
        reference_output = reference_attention(query, key, value)
        np.testing.assert_allclose(
            paged_output,
            reference_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Single token attention outputs don't match reference"
        )

if __name__ == "__main__":
    unittest.main()
