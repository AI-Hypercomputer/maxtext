import unittest
import jax
import jax.numpy as jnp
import numpy as np
from MaxText import maxengine
from MaxText.configs.types import MaxTextConfig as Config

class Qwen3EngineIntegrationTest(unittest.TestCase):
  
  def setUp(self):
    # Initialize a mutable config directly
    self.config = Config(
        model_name="qwen3-next-80b-a3b",
        decoder_block="qwen3_next",
        
        # Dimensions scaled down for testing
        emb_dim=128,
        head_dim=32,
        mlp_dim=256,
        num_decoder_layers=2,
        
        # Qwen3 Next Specifics
        gdn_num_value_heads=4,
        gdn_num_key_heads=4,
        gdn_key_head_dim=32,
        gdn_value_head_dim=32,
        gdn_conv_kernel_dim=4,
        gdn_chunk_size=16,
        inhomogeneous_layer_cycle_interval=2, # Layer 0=GDN, Layer 1=Attention
        
        # Standard Attention Params (needed for FullAttention layers)
        num_query_heads=4,
        num_kv_heads=4,
        
        # Engine Params
        max_prefill_predict_length=32,
        max_target_length=48,
        per_device_batch_size=1,
        scan_layers=False,
        
        # Required boilerplate
        dtype="float32",
        weight_dtype="float32",
        attention="dot_product",
        vocab_size=1024,
        run_name="test_run",
        base_output_directory="/tmp/maxtext_test",
        enable_checkpointing=False,
        skip_jax_distributed_system=True,
        megablox=False,
        sparse_matmul=False,
    )
    
    # Initialize Engine
    self.engine = maxengine.MaxEngine(self.config)
    self.rng = jax.random.PRNGKey(0)


  def find_leaf(self, d, target_key):
    """Recursively searches for a key in a nested dictionary."""
    if not isinstance(d, dict):
        return None
    if target_key in d:
        return d[target_key]
    for k, v in d.items():
        result = self.find_leaf(v, target_key)
        if result is not None:
            return result
    return None


  def test_cache_insertion(self):
    """Verifies that MaxEngine correctly moves GDN states from prefill to decode."""
    
    # 1. Initialize Params
    params = self.engine.load_params(rng=self.rng)
    
    # 2. Create Dummy Input
    seq_len = 16
    dummy_tokens = jnp.ones((1, seq_len), dtype=jnp.int32)
    true_length = seq_len
    
    # 3. Run Prefill
    prefill_result, first_token = self.engine.prefill(
        params=params,
        padded_tokens=dummy_tokens[0], 
        true_length=true_length,
        rng=self.rng,
        slot=0
    )
    
    # 4. Initialize Decode State (Zeros)
    decode_state = self.engine.init_decode_state()
    
    # 5. Perform Insertion (Tests maxengine.py changes)
    new_decode_state = self.engine.insert(prefill_result, decode_state, slot=0)
    
    # 6. Verification
    p_recurrent_var = self.find_leaf(prefill_result['cache'], 'recurrent_state')
    d_recurrent_var = self.find_leaf(new_decode_state['cache'], 'recurrent_state')
    
    if p_recurrent_var is None or d_recurrent_var is None:
        # Debugging info
        print("\nPrefill Cache Keys:", prefill_result['cache'].keys())
        if 'decoder' in prefill_result['cache']:
             print("Decoder Keys:", prefill_result['cache']['decoder'].keys())
        self.fail("Could not find 'recurrent_state' in one of the caches.")

    # Check Recurrent State Values
    p_state = p_recurrent_var.value
    d_state = d_recurrent_var.value
    
    diff = jnp.max(jnp.abs(p_state - d_state))
    print(f"Recurrent State Insertion Diff: {diff}")
    self.assertTrue(diff < 1e-6, "Recurrent state was not inserted correctly!")

    # Check Conv State
    p_conv_var = self.find_leaf(prefill_result['cache'], 'conv_state')
    d_conv_var = self.find_leaf(new_decode_state['cache'], 'conv_state')
    
    if p_conv_var is None or d_conv_var is None:
        self.fail("Could not find 'conv_state' in one of the caches.")

    p_conv = p_conv_var.value
    d_conv = d_conv_var.value
    
    diff_conv = jnp.max(jnp.abs(p_conv - d_conv))
    print(f"Conv State Insertion Diff: {diff_conv}")
    self.assertTrue(diff_conv < 1e-6, "Conv state was not inserted correctly!")

  def test_end_to_end_step(self):
    """Verifies that engine.generate() runs without crashing."""
    params = self.engine.load_params(rng=self.rng)
    dummy_tokens = jnp.ones((1, 16), dtype=jnp.int32)
    
    # Prefill
    prefill_result, _ = self.engine.prefill(
        params=params, padded_tokens=dummy_tokens[0], true_length=16, rng=self.rng, slot=0
    )
    
    # Insert
    decode_state = self.engine.init_decode_state()
    decode_state = self.engine.insert(prefill_result, decode_state, slot=0)
    
    # Generate 1 step
    new_decode_state, output_tokens = self.engine.generate(
        params, decode_state, rng=self.rng
    )
    
    print("Generation Step Successful!")
    # Check that we got a token back
    self.assertTrue(output_tokens.data.shape[1] >= 1)

if __name__ == '__main__':
  unittest.main()