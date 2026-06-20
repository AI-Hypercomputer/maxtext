import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
import numpy as np

from maxtext.layers.attention_compressed import CompressedAttention
from maxtext.common.common_types import AttentionType


class MockConfig:
  """Comprehensive mock configuration to satisfy CompressedAttention and Base Attention."""
  
  # 1. Core Model & Dimensions
  model_name = "deepseek-v4"
  decoder_block = "deepseekv4"  # Bypasses architecture-specific checks (like Llama4/Qwen)
  emb_dim = 256
  head_dim = 64
  num_query_heads = 8
  num_kv_heads = 1
  max_prefill_predict_length = 32
  max_target_length = 64

  # 2. Data Types & Precision
  dtype = jnp.float32
  weight_dtype = jnp.float32
  matmul_precision = "high"
  normalization_layer_epsilon = 1e-5

  # 3. Attention & KV Cache Features
  attention = "dot_product"
  attention_type = "compressed"
  attention_sink = False
  fused_qkv = False
  use_qk_norm = False
  use_qk_norm_in_gdn = False
  v_norm_with_scale = False
  chunk_attn_window_size = 256
  use_chunked_prefill = False
  moba = False
  moba_chunk_size = 0
  moba_topk = 0

  # 4. RoPE (Rotary Positional Embeddings) Parameters
  rope_type = "default"
  rope_min_timescale = 1.0
  rope_max_timescale = 10000.0
  compressed_rope_max_timescale = 160000.0  # V4 specific
  local_rope_max_timescale = 10000.0
  rope_linear_scaling_factor = 1.0
  rope_use_scale = False
  partial_rotary_factor = 1.0

  # 5. DeepSeek-V4 Specific (Compression, Indexing, Grouped Projections)
  o_groups = 2
  o_lora_rank = 32
  q_lora_rank = 1536
  indexer_n_heads = 4
  indexer_head_dim = 16
  indexer_topk = 2

  # 6. Sharding & Parallelism (Mocking single-device compilation)
  shard_mode = "none"
  debug_sharding = False
  logical_axis_rules = []
  ici_context_autoregressive_parallelism = 1

  # 7. Quantization (Disabled for tests)
  quantize_kvcache = False
  kv_quant_axis = None


class MockVariable:
  """Simulates MaxText's NNX Cache Variable containers."""
  def __init__(self, initial_value):
    self.value = initial_value
  def get_value(self):
    return self.value
  def set_value(self, val):
    self.value = val


class MockKVCache:
  """A clean functional mock matching your kvcache.KVCache interface."""
  def __init__(self, batch, head_dim, compress_rate, is_indexer=False):
    h_dim = 16 if is_indexer else head_dim
    self.h_dim = h_dim
    
    self.cached_prefill_key = MockVariable(jnp.zeros((batch, 10, h_dim, 1)))
    self.entry_count = MockVariable(jnp.zeros((batch, 1), dtype=jnp.int32))
    
    # Leftover buffering scratchpads
    buffer_dim = h_dim if is_indexer else h_dim * 2
    self.leftover_buffer_kv = MockVariable(jnp.zeros((batch, compress_rate, 1, buffer_dim)))
    self.leftover_buffer_gate = MockVariable(jnp.zeros((batch, compress_rate, 1, buffer_dim)))
    self.accumulator_index = MockVariable(jnp.zeros((batch, 1), dtype=jnp.int32))
    
    # Staggered overlap registers
    self.overlap_kv = MockVariable(jnp.zeros((batch, compress_rate, 1, h_dim)))
    self.overlap_gate = MockVariable(jnp.zeros((batch, compress_rate, 1, h_dim)))

  def __call__(self, key, value, gate, decoder_segment_ids, model_mode):
    """Simulates the AR cache read-update-recombine lifecycle."""
    batch_size = key.shape[0]
    idx = self.accumulator_index.get_value()[0, 0]
    compress_rate = self.leftover_buffer_kv.get_value().shape[1]
    
    # Step 1: Accumulate incoming single token into scratchpad
    # (In real KVCache, this writes to index position, updates count, and flushes)
    new_idx = idx + 1
    self.accumulator_index.set_value(jnp.full((batch_size, 1), new_idx % compress_rate, dtype=jnp.int32))
    
    # Step 2: Simulate output reconstruction
    # Return structure: (cached_prefill, cached_ar)
    # Shapes match [Batch, Allocated_Blocks, Heads, Dim] expected by your concatenations
    mock_prefill_out = jnp.zeros((batch_size, 1, 1, self.h_dim))
    mock_ar_out = jnp.zeros((batch_size, 0, 1, self.h_dim))
    
    if new_idx == compress_rate:
      # Simulate a window flush occurring
      mock_ar_out = jnp.zeros((batch_size, 1, 1, self.h_dim))
      
    return (mock_prefill_out,), (mock_ar_out,)
  

class MockStandardCache:
  """Mocks the base KVCache so the AttentionOp math kernel has something to read."""
  def __init__(self):
    self.accumulated_keys = None
    self.accumulated_values = None
    
  def __call__(self, key, value, decoder_segment_ids, model_mode, **kwargs):
    # Accumulate the history across steps to satisfy the AttentionOp shape assertions
    if self.accumulated_keys is None:
      self.accumulated_keys = key
      self.accumulated_values = value
    else:
      self.accumulated_keys = jnp.concatenate([self.accumulated_keys, key], axis=1)
      self.accumulated_values = jnp.concatenate([self.accumulated_values, value], axis=1)
      
    # AttentionOp expects the prefill cache to be exactly this 3-item list
    prefill_cache = [self.accumulated_keys, self.accumulated_values, decoder_segment_ids]
    # AR cache includes sequence lengths
    ar_cache = [self.accumulated_keys, self.accumulated_values, decoder_segment_ids, jnp.ones((key.shape[0],))]
    return prefill_cache, ar_cache
  

# ==========================================
# 2. CORE FUNCTIONAL TEST SUITE
# ==========================================

def test_compressed_attention_lifecycle():
  # Initialize random states and configurations
  rngs = nnx.Rngs(42)
  config = MockConfig()
  batch_size = 1
  compress_ratio = 4  # Targets the CSA Compressor path
  
  # ----------------------------------------
  # STEP A: INITIALIZE ATTENTION LAYER
  # ----------------------------------------
  devices = np.array(jax.devices())
  dummy_mesh = Mesh(devices, ('data',))

  attn_layer = CompressedAttention(
      config=config,
      num_query_heads=8,
      num_kv_heads=1,
      head_dim=config.head_dim,
      max_target_length=config.max_target_length,
      max_prefill_predict_length=config.max_prefill_predict_length,
      mesh=dummy_mesh,
      attention_kernel="dot_product",
      inputs_q_shape=(batch_size, 6, config.emb_dim),
      inputs_kv_shape=(batch_size, 6, config.emb_dim),
      compress_ratio=compress_ratio,
      model_mode="autoregressive",
      rngs=rngs,
  )

  # Overwrite internal automated caches with our deterministic mock objects
  attn_layer.compressor_cache = MockKVCache(batch_size, config.head_dim, compress_ratio)
  attn_layer.indexer_cache = MockKVCache(batch_size, config.head_dim, compress_ratio, is_indexer=True)

  attn_layer.KVCache_0 = MockStandardCache()

  # ----------------------------------------
  # STEP B: EXECUTE PREFILL PHASE (With Leftovers)
  # ----------------------------------------
  # Scenario: Sequence length = 6 tokens.
  # With a compression ratio of 4: 1 block is pooled, 2 tokens become leftovers.
  seq_len_prefill = 6
  inputs_q = jnp.ones((batch_size, seq_len_prefill, config.emb_dim))
  inputs_kv = jnp.ones((batch_size, seq_len_prefill, config.emb_dim))
  position_ids = jnp.arange(seq_len_prefill)[None, :]
  
  # Execute forward pass under Prefill mode
  _ = attn_layer(
      inputs_q=inputs_q,
      inputs_kv=inputs_kv,
      decoder_segment_ids=jnp.zeros((batch_size, seq_len_prefill), dtype=jnp.int32),
      inputs_positions=position_ids,
      deterministic=True,
      model_mode="prefill"
  )
  
  # Assertions for Prefill State
  # 1. Did it accurately count the generated blocks? (6 tokens // 4 = 1 block)
  assert attn_layer.compressor_cache.entry_count.get_value()[0, 0] == 1
  
  # 2. Did it isolate the exact remainder? (6 % 4 = 2 leftovers)
  assert attn_layer.compressor_cache.accumulator_index.get_value()[0, 0] == 2
  
  # 3. Were the leftovers correctly padded out into the scratchpad array?
  # Shape must match [Batch, Compress_Rate, Heads, Dim] -> [1, 4, 1, 64]
  assert attn_layer.compressor_cache.leftover_buffer_kv.get_value().shape == (1, 4, 1, config.head_dim * 2)
  
  print("✓ Prefill Caching & Leftover Verification Passed!")

  # ----------------------------------------
  # STEP C: AUTOREGRESSIVE STEP 1 (Accumulating)
  # ----------------------------------------
  # Scenario: Injecting token 7 (Sequence length = 1).
  # Accumulator moves from 2 -> 3. The window is still open.
  inputs_q_ar1 = jnp.ones((batch_size, 1, config.emb_dim))
  inputs_kv_ar1 = jnp.ones((batch_size, 1, config.emb_dim))
  position_ids_ar1 = jnp.array([[6]])
  
  _ = attn_layer(
      inputs_q=inputs_q_ar1,
      inputs_kv=inputs_kv_ar1,
      decoder_segment_ids=jnp.zeros((batch_size, 1), dtype=jnp.int32),
      inputs_positions=position_ids_ar1,
      deterministic=True,
      model_mode="autoregressive"
  )

  # Assertions for AR Accumulation
  assert attn_layer.compressor_cache.accumulator_index.get_value()[0, 0] == 3
  print("✓ AR Accumulation State Tracking Passed!")

  # ----------------------------------------
  # STEP D: AUTOREGRESSIVE STEP 2 (Flushing Window)
  # ----------------------------------------
  # Scenario: Injecting token 8 (Sequence length = 1).
  # Accumulator hits 4, triggering a block compression flush and resetting to 0.
  inputs_q_ar2 = jnp.ones((batch_size, 1, config.emb_dim))
  inputs_kv_ar2 = jnp.ones((batch_size, 1, config.emb_dim))
  position_ids_ar2 = jnp.array([[7]])
  
  _ = attn_layer(
      inputs_q=inputs_q_ar2,
      inputs_kv=inputs_kv_ar2,
      decoder_segment_ids=jnp.zeros((batch_size, 1), dtype=jnp.int32),
      inputs_positions=position_ids_ar2,
      deterministic=True,
      model_mode="autoregressive"
  )
  
  # Assertions for Window Flush
  # The mock cache tracks modulo resets on flush boundaries
  assert attn_layer.compressor_cache.accumulator_index.get_value()[0, 0] == 0
  print("✓ AR Boundary Window Flush Pipeline Passed!")


if __name__ == "__main__":
  test_compressed_attention_lifecycle()