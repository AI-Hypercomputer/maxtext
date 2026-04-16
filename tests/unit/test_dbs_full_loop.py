import unittest
import jax
import jax.numpy as jnp
from maxtext.inference.maxengine import maxengine
from maxtext.configs import pyconfig
import os
import sys

class DBSFullLoopTest(unittest.TestCase):
  def test_dbs_full_loop_cpu(self):
    """Test that full prefill + generate loop works with diverse_beam_search on CPU."""
    
    # Resolve config path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "../../src/maxtext/configs/base.yml")
    
    # Setup for a tiny model that fits on CPU.
    argv = [
        "tests/unit/test_dbs_full_loop",
        config_path,
        "model_name=gemma-2b", # Using Gemma-2b config but overriding dimensions
        "base_num_decoder_layers=2",
        "base_emb_dim=128",
        "base_num_query_heads=2",
        "base_num_kv_heads=2",
        "head_dim=64",
        "attention=dot_product", 
        "decode_sampling_strategy=diverse_beam_search",
        "decode_num_beams=4",
        "decode_num_beam_groups=2",
        "decode_diversity_penalty=0.5",
        "max_prefill_predict_length=8",
        "max_target_length=16",
        "per_device_batch_size=1",
        "ici_fsdp_parallelism=4",
        "ici_tensor_parallelism=1",
        "override_model_config=True",
        "skip_jax_distributed_system=True"
    ]
    config = pyconfig.initialize(argv)
    
    # Initialize engine
    # Note: On CPU, we don't need TPU sharding
    engine = maxengine.MaxEngine(config)
    
    # Load dummy params
    params = engine.load_params(jax.random.PRNGKey(0))
    
    # 1. PREFILL, fill real, unique tokens, avoid 0 since it's padding or EOS
    true_length = config.max_prefill_predict_length
    tokens = jnp.zeros((config.max_prefill_predict_length,), dtype=jnp.int32)
    tokens = tokens.at[:true_length].set(jnp.arange(true_length) + 1)
    
    print("\n--- Running Prefill ---")
    prefill_res, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    
    # 2. INITIALIZE DECODE STATE
    print("--- Initializing Decode State ---")
    rng = jax.random.PRNGKey(42)
    decode_state = engine.init_decode_state(rng)
    
    # In MaxEngine, we need to bulk_insert the prefill results into the slots
    # For a single user with 4 beams, we fill slots 0, 1, 2, 3
    slots = list(range(config.decode_num_beams))
    decode_state = engine.bulk_insert(prefill_res, decode_state, slots=slots)
    
    # CRITICAL: Also update next_pos and tokens for all beams
    for slot in slots:
        decode_state["next_pos"] = decode_state["next_pos"].at[slot, 0].set(true_length)
        # first_token is a ResultTokens object, tokens are in .data
        # Based on tokens_idx=(0, 1), tokens are at the beginning
        token_val = first_token.data[0, 0]
        decode_state["tokens"] = decode_state["tokens"].at[slot, 0].set(token_val)
    
    # 3. GENERATION LOOP (5 steps)
    print("--- Running Generation Loop ---")
    for step in range(5):
        decode_state, sampled_tokens = engine.generate(params, decode_state)
        
        # Verify tokens are being produced for all beams
        expected_total_batch_size = int(config.per_device_batch_size * engine.mesh.size) * config.decode_num_beams
        self.assertEqual(decode_state["tokens"].shape[0], expected_total_batch_size)
        print(f"Step {step} completed. Sampled tokens: {decode_state['tokens'].flatten()}")

    # 4. FINAL VERIFICATION
    self.assertTrue("is_dbs" in decode_state)
    self.assertTrue(decode_state["is_dbs"].any())
    # cumulative_logprobs should have been updated
    self.assertFalse(jnp.all(decode_state["cumulative_logprobs"] == 0))
    
    print("\n✅ SUCCESS: Full DBS loop completed on CPU!")

if __name__ == "__main__":
  unittest.main()
