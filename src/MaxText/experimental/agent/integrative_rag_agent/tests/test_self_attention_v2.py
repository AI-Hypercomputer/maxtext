import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch

# --- CONFIGURATION ---
# Force JAX to use High Precision (float32) for matrix multiplications.
# This is critical for comparing against PyTorch on TPU/GPU.
jax.config.update("jax_default_matmul_precision", "float32")

# 1. Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
sys.path.append(project_root)

from generated_code.SelfAttention.layers import SelfAttention as SelfAttentionJAX
from examples.level2_pytorch_module.self_attention import SelfAttention as SelfAttentionPT



def debug_test():
    print("--- DEBUGGING LAYER BY LAYER ---")
    EMBED_SIZE = 32
    HEADS = 4
    BATCH = 2
    SEQ_LEN = 10

    # Init Models
    pt_model = SelfAttentionPT(EMBED_SIZE, HEADS).eval()
    jax_model = SelfAttentionJAX(EMBED_SIZE, HEADS)
    
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((BATCH, SEQ_LEN, EMBED_SIZE))
    variables = jax_model.init(key, dummy_input, dummy_input, dummy_input, None)
    params = variables['params']

    # Copy Weights
    print("Copying weights...")
    def copy_layer(pt_layer, jax_name):
        params[jax_name]['kernel'] = pt_layer.weight.detach().numpy().T
        if pt_layer.bias is not None and 'bias' in params[jax_name]:
            params[jax_name]['bias'] = pt_layer.bias.detach().numpy()
    
    copy_layer(pt_model.values, 'values_projection')
    copy_layer(pt_model.keys, 'keys_projection')
    copy_layer(pt_model.queries, 'queries_projection')
    copy_layer(pt_model.fc_out, 'output_projection')

    # Generate Input
    np_input = np.random.randn(BATCH, SEQ_LEN, EMBED_SIZE).astype(np.float32)
    pt_in = torch.from_numpy(np_input)

    # --- STEP 1: PROJECTIONS ---
    print("\n[Step 1] Projections")
    with torch.no_grad():
        pt_q = pt_model.queries(pt_in).numpy()
    
    # Manual JAX calc
    w_q = params['queries_projection']['kernel']
    jax_q = np.dot(np_input, w_q)
    
    diff = np.abs(pt_q - jax_q).max()
    print(f"Queries Projection Diff: {diff:.2e}")

    # --- STEP 2: RESHAPE & TRANSPOSE ---
    print("\n[Step 2] Reshape & Transpose")
    with torch.no_grad():
        # PyTorch: (N, L, E) -> (N, L, H, D) -> (N, H, L, D)
        pt_q_heads = pt_model.queries(pt_in).reshape(BATCH, SEQ_LEN, HEADS, -1).transpose(1, 2).numpy()
        pt_k_heads = pt_model.keys(pt_in).reshape(BATCH, SEQ_LEN, HEADS, -1).transpose(1, 2).numpy()

    # Manual JAX calc
    # JAX: (N, L, E) -> (N, L, H, D) -> (N, H, L, D) (transpose 0,2,1,3)
    jax_q_heads = jax_q.reshape(BATCH, SEQ_LEN, HEADS, -1).transpose(0, 2, 1, 3)
    
    w_k = params['keys_projection']['kernel']
    jax_k = np.dot(np_input, w_k)
    jax_k_heads = jax_k.reshape(BATCH, SEQ_LEN, HEADS, -1).transpose(0, 2, 1, 3)

    print(f"Queries Head Shape Diff: {np.abs(pt_q_heads - jax_q_heads).max():.2e}")

    # --- STEP 3: ATTENTION SCORES (The likely culprit for 1e-3 diffs) ---
    print("\n[Step 3] Attention Scores (Energy)")
    with torch.no_grad():
        # PyTorch: matmul(q, k.transpose(-2, -1))
        pt_energy = torch.matmul(torch.from_numpy(pt_q_heads), torch.from_numpy(pt_k_heads).transpose(-2, -1)).numpy()
    
    # Manual JAX calc
    # JAX: matmul(q, swapaxes(k, -2, -1))
    jax_energy = np.matmul(jax_q_heads, jax_k_heads.swapaxes(-2, -1))
    
    print(f"Energy Diff:             {np.abs(pt_energy - jax_energy).max():.2e}")

    # --- STEP 4: SOFTMAX ---
    print("\n[Step 4] Softmax")
    scale = (EMBED_SIZE // HEADS) ** 0.5
    pt_attn = torch.softmax(torch.from_numpy(pt_energy) / scale, dim=-1).numpy()
    jax_attn = jax.nn.softmax(jax_energy / scale, axis=-1)
    
    print(f"Softmax Diff:            {np.abs(pt_attn - jax_attn).max():.2e}")

    # --- FINAL CHECK ---
    print("\n[Step 5] Final Output")
    with torch.no_grad():
        pt_out = pt_model(pt_in, pt_in, pt_in, None).numpy()
    
    jax_out = jax_model.apply({'params': params}, np_input, np_input, np_input, None)
    
    print(f"Final Model Diff:        {np.abs(pt_out - np.array(jax_out)).max():.2e}")

    if np.abs(pt_out - np.array(jax_out)).max() < 1e-5:
        print("\nSUCCESS: Models match with float32 precision!")
    else:
        print("\nFAILURE: Models still diverge.")

if __name__ == "__main__":
    debug_test()
