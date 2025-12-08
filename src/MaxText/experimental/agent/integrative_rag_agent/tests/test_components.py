import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from MaxText.layers import linears

# --- CRITICAL: TRY "HIGHEST" TO FORCE FLOAT32 ON TPU ---
# "default" often maps to bfloat16 on TPU even if config is float32
jax.config.update("jax_default_matmul_precision", "float32")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
sys.path.append(project_root)

def test_components():
    print("--- DEBUGGING JAX COMPONENTS ON DEVICE ---")
    EMBED = 32
    HEADS = 4
    SEQ = 10
    BATCH = 2
    
    # Common Inputs
    np_in = np.random.randn(BATCH, SEQ, EMBED).astype(np.float32)
    pt_in = torch.from_numpy(np_in)
    jax_in = jnp.array(np_in)

    # --- TEST 1: DenseGeneral vs Linear (Projections) ---
    print("\n[Test 1] DenseGeneral Layer Precision")
    # Init JAX Layer
    dense = linears.dense_general(
        in_features_shape=(EMBED,), out_features_shape=(EMBED,), 
        use_bias=False, name="test_dense", matmul_precision="highest",
    )
    key = jax.random.PRNGKey(0)
    vars = dense.init(key, jax_in)
    
    # Init PyTorch Layer
    linear = torch.nn.Linear(EMBED, EMBED, bias=False)
    
    # Copy Weights
    # JAX: (In, Out), PT: (Out, In)
    vars['params']['kernel'] = linear.weight.detach().numpy().T
    
    # Run
    jax_out = dense.apply(vars, jax_in)
    with torch.no_grad():
        pt_out = linear(pt_in).numpy()
        
    diff = np.abs(pt_out - np.array(jax_out)).max()
    print(f"DenseGeneral Diff: {diff:.2e}")
    if diff > 1e-4:
        print("  -> FAIL: DenseGeneral is running in low precision (BF16).")
        print("     FIX: Pass matmul_precision='highest' to dense_general.")

    # --- TEST 2: Raw Matmul (Context Aggregation) ---
    print("\n[Test 2] jnp.matmul Precision (Attention @ Values)")
    # Shapes: (N, H, L, L) @ (N, H, L, D)
    H_DIM = EMBED // HEADS
    np_attn = np.random.rand(BATCH, HEADS, SEQ, SEQ).astype(np.float32)
    np_vals = np.random.randn(BATCH, HEADS, SEQ, H_DIM).astype(np.float32)
    
    jax_res = jnp.matmul(jnp.array(np_attn), jnp.array(np_vals))
    pt_res = torch.matmul(torch.from_numpy(np_attn), torch.from_numpy(np_vals))
    
    diff = np.abs(pt_res.numpy() - np.array(jax_res)).max()
    print(f"jnp.matmul Diff:   {diff:.2e}")
    if diff > 1e-4:
        print("  -> FAIL: jnp.matmul is running in low precision (BF16).")
        print("     FIX: Use jax.lax.dot_general with precision=lax.Precision.HIGHEST")

if __name__ == "__main__":
    test_components()
