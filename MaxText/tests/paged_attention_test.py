
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
            'head_dim': 64,
            'max_prefill_predict_length': 512,
            'max_target_length': 1024,
            'block_size': 32,
            'num_pages': 1024,
            'page_size': 16,
            'dtype': jnp.float32,
        }
        self.rng = jax.random.PRNGKey(42)

    def test_paged_attention_output_shape(self):
        # Initialize PagedAttentionOp
        attention_op = PagedAttentionOp(
            mesh=None,  # For this test, we're not focusing on mesh-specific details
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

        # Debugging: Print initialized variable shapes
        print(f"Initialized Variables: {variables['cache']}")

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

        # Debugging: Print mutated variable shapes after apply
        print(f"Mutated Variables after Apply: {mutated_variables['cache']}")

        # Check the output shape
        self.assertEqual(output.shape, (1, self.cfg['max_prefill_predict_length'], self.cfg['num_query_heads'], self.cfg['head_dim']))


if __name__ == "__main__":
    unittest.main()