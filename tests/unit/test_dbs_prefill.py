import unittest
import jax
import jax.numpy as jnp
from maxtext.inference.maxengine import maxengine
from maxtext.configs import pyconfig
import os

class DBSPrefillTest(unittest.TestCase):
  def test_dbs_prefill_no_crash(self):
    """Test that prefill doesn't crash when diverse_beam_search is enabled."""
    # Set DECOUPLE_GCLOUD=TRUE to use stubs
    os.environ["DECOUPLE_GCLOUD"] = "TRUE"
    
    base_dir = "/mnt/mac/Users/kevinwang/Projects/maxtext"
    config_path = f"{base_dir}/src/maxtext/configs/base.yml"
    
    argv = [
        "tests/unit/test_dbs_prefill_no_crash",
        config_path,
        "model_name=gemma-2b",
        "attention=dot_product", # Fallback to standard JAX attention for CPU tests
        "decode_sampling_strategy=diverse_beam_search",
        "decode_num_beams=4",
        "max_prefill_predict_length=128",
        "max_target_length=256",
        "ici_fsdp_parallelism=1",
        "ici_tensor_parallelism=1",
        "skip_jax_distributed_system=True"
    ]
    config = pyconfig.initialize(argv)
    
    # Initialize engine (with stubs)
    engine = maxengine.MaxEngine(config)
    
    # Load dummy params
    params = engine.load_params(jax.random.PRNGKey(0))
    
    # Create dummy tokens for prefill
    true_length = 10
    tokens = jnp.zeros((config.max_prefill_predict_length,), dtype=jnp.int32)
    tokens = tokens.at[:true_length].set(jnp.arange(true_length) + 1)
    
    print("Attempting prefill with diverse_beam_search...")
    # This should trigger the new 'diverse_beam_search' override in _prefill_jit
    prefill_res, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    
    print("✅ SUCCESS: Prefill completed!")
    self.assertIsNotNone(prefill_res)
    self.assertIsNotNone(first_token)

if __name__ == "__main__":
  unittest.main()
