
import unittest
import jax
import jax.numpy as jnp
from flax.core import freeze
from layers.attentions import PagedAttentionOp, Attention
from page_managers import PageManager, PageState
import common_types


class PagedAttentionTest(unittest.TestCase):
    def setUp(self):
        # Initializing configuration, PRNG key, and mesh setup
        self.cfg = {
            'num_query_heads': 8,
            'num_kv_heads': 8,
            'head_dim': 128,
            'max_prefill_predict_length': 512,
            'max_target_length': 1024,
            'block_size': 32,
            'num_pages': 1024,
            'page_size': 16,
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
        key = jnp.ones((1, 1, self.cfg['num_kv_heads'], self.cfg['head_dim']))  # Shape for decode step
        value = jnp.ones((1, 1, self.cfg['num_kv_heads'], self.cfg['head_dim']))  # Shape for decode step
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
            query=jnp.ones((1, 1, self.cfg['num_query_heads'], self.cfg['head_dim'])),
            key=key,
            value=value,
            decoder_segment_ids=None,
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,  # Use AUTOREGRESSIVE mode
            page_state=page_state,
            mutable=["cache"]
        )

        updated_key_pages_var = mutated_variables['cache']['key_pages']
        updated_value_pages_var = mutated_variables['cache']['value_pages']

        self.assertEqual(updated_key_pages_var.value.shape, (self.cfg['num_kv_heads'], self.cfg['num_pages'], self.cfg['page_size'], self.cfg['head_dim']))
        self.assertEqual(updated_value_pages_var.value.shape, (self.cfg['num_kv_heads'], self.cfg['num_pages'], self.cfg['page_size'], self.cfg['head_dim']))



if __name__ == "__main__":
    unittest.main()
