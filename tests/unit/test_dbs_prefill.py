import sys
import unittest
import jax
import jax.numpy as jnp
from maxtext.inference.maxengine import maxengine
from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path
class DBSPrefillTest(unittest.TestCase):
  def test_dbs_prefill_no_crash(self):
    # Set up config
    init_kwargs = {
        "decode_sampling_strategy": "diverse_beam_search",
        "max_prefill_predict_length": 8,
        "max_target_length": 16,
        "run_name": "test_dbs",
        "enable_checkpointing": False,
        "attention": "dot_product",
        "base_num_decoder_layers": 2,
        "base_emb_dim": 128,
        "base_num_query_heads": 2,
        "base_num_kv_heads": 2,
        "skip_jax_distributed_system": True,
    }
    
    config = pyconfig.initialize([sys.argv[0], get_test_config_path()], **init_kwargs)
    object.__setattr__(config, 'decode_num_beams', 4)
    engine = maxengine.MaxEngine(config, jax.devices()[:1])
    params = engine.load_params(jax.random.PRNGKey(0))
    
    # tokens 1D (seq_len)
    tokens = jnp.zeros((config.max_prefill_predict_length,), dtype=jnp.int32)
    # true_length MUST be a 0D scalar for jax.lax.dynamic_slice
    true_length = jnp.array(4, dtype=jnp.int32)
    
    print("Attempting prefill with scalar true_length...")
    prefill_res, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    print("✅ SUCCESS: Prefill completed!")
if __name__ == "__main__":
  unittest.main()
