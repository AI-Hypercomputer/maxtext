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
        attn = jnp.dot(q.astype(jnp.float32), k.astype(jnp.float32).T)
        mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
        attn = jnp.where(mask, attn, 0.0) 
        attn = attn * beta.astype(jnp.float32)[:, None] 
        
        term2 = jnp.dot(attn, v.astype(jnp.float32))
        
        # Store Output
        o_chunk = term1 + term2
        o_ref[i] = o_chunk.astype(dtype)

        # 3. State Update
        chunk_decay = jnp.exp(g[..., -1]) 
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
    
    # Use Keyword Arguments for BlockSpec to avoid TypeError
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
        # Ensure CompilerParams is used (fixed from previous turn)
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    )(w, u, q, k, v, g, beta)