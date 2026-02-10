# src/MaxText/kernels/gdn_pallas.py
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gdn_scan_kernel_tpu(
    w_ref,      # [NumChunks, ChunkSize, KeyDim]
    u_ref,      # [NumChunks, ChunkSize, ValDim]
    q_ref,      # [NumChunks, ChunkSize, KeyDim]
    k_ref,      # [NumChunks, ChunkSize, KeyDim]
    v_ref,      # [NumChunks, ChunkSize, ValDim]
    g_ref,      # [NumChunks, ChunkSize]
    beta_ref,   # [NumChunks, ChunkSize]
    o_ref,      # [NumChunks, ChunkSize, ValDim] (Output)
    # Hyperparameters captured by closure
    num_chunks: int,
    chunk_size: int,
    key_dim: int,
    val_dim: int,
    dtype: jnp.dtype = jnp.bfloat16
):
    # Initialize State h in VMEM (SRAM) - Shape: (KeyDim, ValDim)
    h = jnp.zeros((key_dim, val_dim), dtype=jnp.float32)

    # Loop over chunks (Sequential Dependency)
    for i in range(num_chunks):
        # 1. Load Inputs from HBM to VMEM
        w = w_ref[i] 
        u = u_ref[i] 
        q = q_ref[i] 
        k = k_ref[i] 
        v = v_ref[i] 
        g = g_ref[i] 
        beta = beta_ref[i] 

        # 2. Compute Outputs
        # Re-compute exp(g) here to save HBM IO (fusion)
        g_exp = jnp.exp(g.astype(jnp.float32))
        q_g = q.astype(jnp.float32) * g_exp[:, None]
        term1 = jnp.dot(q_g, h) 

        # Intra-chunk attention
        attn = jnp.dot(q