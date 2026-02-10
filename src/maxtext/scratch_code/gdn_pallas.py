# src/MaxText/kernels/gdn_pallas.py
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gdn_scan_kernel_tpu(
    w_ref,      # Shape: [1, 1, NumChunks, ChunkSize, KeyDim]
    u_ref,      # Shape: [1, 1, NumChunks, ChunkSize, ValDim]
    q_ref,      # Shape: [1, 1, NumChunks, ChunkSize, KeyDim]
    k_ref,      # Shape: [1, 1, NumChunks, ChunkSize, KeyDim]
    v_ref,      # Shape: [1, 1, NumChunks, ChunkSize, ValDim]
    g_ref,      # Shape: [1, 1, NumChunks, ChunkSize]
    beta_ref,   # Shape: [1, 1, NumChunks, ChunkSize]
    o_ref,      # Shape: [1, 1, NumChunks, ChunkSize, ValDim]
    # Hyperparameters
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
        # FIX: Explicitly index [0, 0, i] to access the i-th chunk in the block
        w = w_ref[0, 0, i] 
        u = u_ref[0, 0, i] 
        q = q_ref[0, 0, i] 
        k = k_ref[0, 0, i] 
        v = v_ref[0, 0, i] 
        g = g_ref[0, 0, i] 
        beta = beta_ref[0, 0, i] 

        # 2. Compute Outputs
        # g is (ChunkSize,), g_exp is (ChunkSize,)
        g_exp = jnp.exp(g.astype(jnp.float32))
        
        # q is (ChunkSize, KeyDim), g_exp[:, None] is (ChunkSize, 1)
        # q_g becomes (ChunkSize, KeyDim)
        q_g = q.astype(jnp.float32) * g_exp[:, None]
        
        # term1: (C, Dk) @ (Dk, Dv) -> (C, Dv)
        term1 = jnp.dot(q_g, h) 

        # Intra-chunk attention: (C, Dk) @ (Dk, C) -> (C, C)
        attn = jnp.dot(q.astype(jnp.float32), k.astype(jnp.float32).T)
        
        mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
        attn = jnp.where(mask, attn, 0.0) 
        
        # Apply beta gate: (C, C) * (C, 1) -> (C, C)
        attn = attn * beta.astype(jnp.float32)[:, None] 
        
        # term2: (C, C) @ (C, Dv) -> (C, Dv)
        term2 = jnp.dot(attn, v.astype(jnp.float32))
        
        o_chunk = term1 + term2
        
        # Store Output: Index [0, 0, i]
        o_ref[0, 0, i] = o_chunk.astype(dtype)

        # 3. State Update
        chunk_decay = jnp.exp(g[..., -1]) 
        
        # update: (Dk, C) @ (C, Dv) -> (Dk, Dv)
        update = jnp.dot(w.astype(jnp.float32).T, u.astype(jnp.float32))
        
        h = h * chunk_decay + update

def gdn_pallas_layer(w, u, q, k, v, g, beta):
    """
    Launcher for the Pallas Kernel.
    Inputs must be shaped: (Batch, NumHeads, NumChunks, ChunkSize, Dim)
    """
    B, H, N_chunks, C, Dk = k.shape
    _, _, _, _, Dv = v.shape
    
    # Map grid (Batch, Head) -> Parallel Execution
    grid = (B, H)
    
    # We map grid indices (i, j) to the input block (i, j, :, :, :)
    # This means the Kernel receives a block of shape (1, 1, N_chunks, C, D)
    in_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dk))
    val_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))
    scalar_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, N_chunks, C))
    out_spec = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))

    kernel_fn = functools.partial(
        gdn_scan_kernel_tpu,
        num_chunks=N_chunks,
        chunk_size=C,
        key_dim=Dk,
        val_dim=Dv,
        dtype=v.dtype
    )

    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((B, H, N_chunks, C, Dv), v.dtype),
        grid=grid,
        in_specs=[in_specs, val_specs, in_specs, in_specs, val_specs, scalar_specs, scalar_specs],
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    )(w, u, q, k, v, g, beta)