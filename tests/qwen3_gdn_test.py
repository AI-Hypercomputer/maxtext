import unittest
import jax
import jax.numpy as jnp
from flax import nnx
from MaxText.layers import qwen3
from MaxText.inference import kvcache
from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText.configs.types import MaxTextConfig as Config

class Qwen3GatedDeltaNetTest(unittest.TestCase):

  def setUp(self):
    # Create a dummy config
    self.config = Config(
        emb_dim=128,
        gdn_num_value_heads=4,
        gdn_num_key_heads=4,
        gdn_key_head_dim=32,
        gdn_value_head_dim=32,
        gdn_conv_kernel_dim=4,
        gdn_chunk_size=16, # Chunk size for prefill
        dtype="float32",
        normalization_layer_epsilon=1e-6,
        matmul_precision="default",
        use_qk_norm_in_gdn=False,
        # Dummy values for required config fields
        num_query_heads=4, num_kv_heads=4, head_dim=32,
        max_target_length=128, max_prefill_predict_length=64,
        dropout_rate=0.0, attention="dot_product",
        scan_layers=False, param_scan_axis=0,
        enable_dropout=False,
        weight_dtype="float32",
        mlp_activations=["silu"], mlp_dim=128,
        quantize_kvcache=False,
    )
    self.rngs = nnx.Rngs(0)
    self.batch_size = 1
    self.seq_len = 32

  def test_prefill_decode_equivalence(self):
    """Verifies that processing a sequence all at once (prefill) yields
    the same results as processing it token-by-token (decode)."""
    
    # Initialize Layer
    model = qwen3.Qwen3NextGatedDeltaNet(self.config, rngs=self.rngs)
    
    # Initialize Inputs
    dummy_input = jax.random.normal(
        self.rngs.params(), (self.batch_size, self.seq_len, self.config.emb_dim)
    )

    # --- Run 1: Prefill (Parallel) ---
    # We pass a fresh cache just to capture the final state, though prefill doesn't read from it usually
    prefill_cache = kvcache.GatedDeltaNetCache(
        batch=self.batch_size,
        num_heads=self.config.gdn_num_value_heads,
        k_head_dim=self.config.gdn_key_head_dim,
        v_head_dim=self.config.gdn_value_head_dim,
        conv_kernel_size=self.config.gdn_conv_kernel_dim,
        conv_dim=2 * model.key_dim + model.value_dim,
        dtype="float32",
    )
    
    prefill_out = model(dummy_input, model_mode=MODEL_MODE_PREFILL, kv_cache=prefill_cache)

    # --- Run 2: Autoregressive (Recurrent) ---
    # Initialize an empty cache
    ar_cache = kvcache.GatedDeltaNetCache(
        batch=self.batch_size,
        num_heads=self.config.gdn_num_value_heads,
        k_head_dim=self.config.gdn_key_head_dim,
        v_head_dim=self.config.gdn_value_head_dim,
        conv_kernel_size=self.config.gdn_conv_kernel_dim,
        conv_dim=2 * model.key_dim + model.value_dim,
        dtype="float32",
    )

    ar_outputs = []
    
    # Feed tokens one by one
    for t in range(self.seq_len):
        # Shape: [Batch, 1, Dim]
        token_input = dummy_input[:, t:t+1, :]
        
        # Run decode step
        token_out = model(token_input, model_mode=MODEL_MODE_AUTOREGRESSIVE, kv_cache=ar_cache)
        ar_outputs.append(token_out)

    # Concatenate all decode steps: [Batch, Seq, Dim]
    ar_out_stacked = jnp.concatenate(ar_outputs, axis=1)

    # --- Verification ---
    # 1. Compare Outputs
    # Note: Convolution boundary conditions might cause slight diffs at the very start (t < kernel_size)
    # depending on how padding works, but generally they should match closely.
    max_diff = jnp.max(jnp.abs(prefill_out - ar_out_stacked))
    print(f"Max difference between Prefill and Autoregressive: {max_diff}")
    
    # 2. Compare Final States
    # The cache state at the end of decode should match the cache state returned by prefill
    recurrent_diff = jnp.max(jnp.abs(prefill_cache.recurrent_state.value - ar_cache.recurrent_state.value))
    print(f"Max difference in Recurrent State: {recurrent_diff}")

    # Use a reasonable tolerance for float32 (accumulation errors happen)
    self.assertTrue(max_diff < 5e-3, f"Output mismatch! Max diff: {max_diff}")
    self.assertTrue(recurrent_diff < 5e-3, f"State mismatch! Max diff: {recurrent_diff}")

if __name__ == '__main__':
  unittest.main()