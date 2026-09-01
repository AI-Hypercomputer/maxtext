import os
import time
import jax
import pathwaysutils
import jax.numpy as jnp
from flax import nnx

# Adjust import path if needed based on your MaxText directory structure
from maxtext.models.qwen3 import Qwen3NextGatedDeltaNet 
from maxtext.common.common_types import MODEL_MODE_TRAIN

class MockQwen35Config:
    """Mock Config strictly mirroring the Qwen3.5 YAML parameters"""
    # Core Architectural
    emb_dim = 2048
    normalization_layer_epsilon = 1.0e-6
    
    # GDN Specific
    gdn_conv_kernel_dim = 4
    gdn_key_head_dim = 128
    gdn_value_head_dim = 128
    gdn_num_key_heads = 16
    gdn_num_value_heads = 32
    gdn_chunk_size = 64  # Base config size, but we will mutate this in the loop
    
    # Execution & Memory
    dtype = jnp.bfloat16
    weight_dtype = jnp.bfloat16
    matmul_precision = jax.lax.Precision.DEFAULT
    use_qk_norm_in_gdn = False
    using_pipeline_parallelism = False
    logical_axis_rules = None
    
    # FORCED FALSE for Benchmarking: 
    # If True, it bypasses our JAX hybrid solver and calls the Pallas kernel.
    use_gdn_kernel = False 
    
    # Cache parameters (required by init even if not used in train mode)
    max_prefill_predict_length = 4096
    max_target_length = 4096

def benchmark_qwen35_layer():
    pathwaysutils.initialize()
    print(f"Hardware: {jax.devices()[0].device_kind}")
    batch_size = 4
    seq_len = 4096  # Standard long context
    cfg = MockQwen35Config()
    
    print(f"Hardware: {jax.devices()[0].device_kind}")
    print(f"Layer: Qwen3.5 GDN | Batch: {batch_size} | Seq: {seq_len} | Embed: {cfg.emb_dim}")
    print(f"Heads: {cfg.gdn_num_key_heads} KV / {cfg.gdn_num_value_heads} V | Head Dim: {cfg.gdn_key_head_dim}")
    print(f"{'Chunk Size':<12} | {'Time per forward pass (ms)':<25}")
    print("-" * 55)

    key = jax.random.PRNGKey(42)
    
    # Test across scaling chunk sizes (64 is the baseline in your config)
    for chunk_size in [64, 128, 256, 512, 1024]:
        cfg.gdn_chunk_size = chunk_size
        
        # 1. Initialize the full NNX layer using Qwen3.5 specs
        layer = Qwen3NextGatedDeltaNet(
            config=cfg, 
            mesh=None, 
            rngs=nnx.Rngs(0)
        )
        
        # 2. Create dummy hidden states (Input from previous MoE layer)
        hidden_states = jax.random.normal(
            key, 
            (batch_size, seq_len, cfg.emb_dim), 
            dtype=cfg.dtype
        )
        
        # 3. JIT compile the layer
        @jax.jit
        def forward_fn(x):
            out, _ = layer(x, model_mode=MODEL_MODE_TRAIN)
            return out
            
        # 4. Warmup (Triggers XLA compilation)
        out = forward_fn(hidden_states)
        out.block_until_ready()
        
        # 5. Timing Loop
        iters = 20
        start_time = time.time()
        for _ in range(iters):
            out = forward_fn(hidden_states)
        out.block_until_ready()
        
        avg_time_ms = ((time.time() - start_time) / iters) * 1000
        print(f"{chunk_size:<12} | {avg_time_ms:<25.2f}")

if __name__ == "__main__":
    benchmark_qwen35_layer()