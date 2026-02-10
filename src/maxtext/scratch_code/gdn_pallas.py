# src/MaxText/kernels/gdn_pallas.py
import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ==============================================================================
# 1. Pallas Kernel Implementation (Forward Pass Logic)
# ==============================================================================
def gdn_scan_kernel_tpu(
    w_ref, u_ref, q_ref, k_ref, v_ref, g_ref, beta_ref, o_ref,
    # Hyperparameters
    num_chunks: int, chunk_size: int, key_dim: int, val_dim: int,
    dtype: jnp.dtype = jnp.bfloat16
):
    # Initialize State h in VMEM (SRAM)
    h = jnp.zeros((key_dim, val_dim), dtype=jnp.float32)

    # Standard loop over chunks
    for i in range(num_chunks):
        # Load Inputs (Indexing into the chunk dimension [0,0,i])
        w = w_ref[0, 0, i] 
        u = u_ref[0, 0, i] 
        q = q_ref[0, 0, i] 
        k = k_ref[0, 0, i] 
        v = v_ref[0, 0, i] 
        g = g_ref[0, 0, i] 
        beta = beta_ref[0, 0, i] 

        # --- Output Computation ---
        g_exp = jnp.exp(g.astype(jnp.float32))
        q_g = q.astype(jnp.float32) * g_exp[:, None]
        
        # Term 1: Recurrent State
        term1 = jnp.dot(q_g, h) 

        # Term 2: Intra-chunk Attention
        attn = jnp.dot(q.astype(jnp.float32), k.astype(jnp.float32).T)
        
        # FIX: Use float32 arithmetic mask instead of bool to avoid Mosaic compilation error
        # ("Unsupported target bitwidth for truncation" i8->i1)
        # jnp.tri returns 1.0 on/below diagonal, 0.0 above.
        mask_val = jnp.tri(chunk_size, dtype=jnp.float32)
        attn = attn * mask_val
        
        attn = attn * beta.astype(jnp.float32)[:, None] 
        term2 = jnp.dot(attn, v.astype(jnp.float32))
        
        o_chunk = term1 + term2
        o_ref[0, 0, i] = o_chunk.astype(dtype)

        # --- State Update ---
        # Explicitly use static indexing for the last element
        chunk_decay = jnp.exp(g[chunk_size - 1]) 
        update = jnp.dot(w.astype(jnp.float32).T, u.astype(jnp.float32))
        h = h * chunk_decay + update

# ==============================================================================
# 2. JAX Reference Implementation (For Backward Pass / Autodiff)
# ==============================================================================
def _gdn_reference(w, u, q, k, v, g, beta):
    """Pure JAX equivalent of the kernel for autodiff."""
    # Inputs: (B, H, N, C, D)
    # Transpose for Scan: (N, B, H, C, D)
    perm_vec = (2, 0, 1, 3, 4)
    perm_scl = (2, 0, 1, 3)
    
    w_s = w.transpose(perm_vec)
    u_s = u.transpose(perm_vec)
    q_s = q.transpose(perm_vec)
    k_s = k.transpose(perm_vec)
    v_s = v.transpose(perm_vec)
    g_s = g.transpose(perm_scl)
    beta_s = beta.transpose(perm_scl)
    
    B, H, N, C, Dk = k.shape
    Dv = v.shape[-1]
    h_init = jnp.zeros((B, H, Dk, Dv), dtype=jnp.float32)

    def scan_body(h, args):
        wt, ut, qt, kt, vt, gt, betat = args
        
        # Match Pallas Math Exactly
        gt_exp = jnp.exp(gt.astype(jnp.float32))
        q_g = qt.astype(jnp.float32) * gt_exp[..., None]
        
        # Term 1
        term1 = jnp.matmul(q_g, h)
        
        # Term 2
        attn = jnp.matmul(qt.astype(jnp.float32), kt.astype(jnp.float32).swapaxes(-1, -2))
        
        # Reference masking (Logic matches jnp.tri)
        mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
        attn = attn * mask
        
        attn = attn * betat.astype(jnp.float32)[..., None]
        term2 = jnp.matmul(attn, vt.astype(jnp.float32))
        
        out = (term1 + term2).astype(v.dtype)
        
        # Update
        chunk_decay = jnp.exp(gt[..., -1])[..., None, None]
        update = jnp.matmul(wt.astype(jnp.float32).swapaxes(-1, -2), ut.astype(jnp.float32))
        h_new = h * chunk_decay + update
        
        return h_new, out

    _, o_scan = lax.scan(
        scan_body, 
        h_init, 
        (w_s, u_s, q_s, k_s, v_s, g_s, beta_s)
    )
    
    # Transpose back: (N, B, H, C, D) -> (B, H, N, C, D)
    return o_scan.transpose(1, 2, 0, 3, 4)

# ==============================================================================
# 3. Custom VJP Registration (The Glue)
# ==============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=())
def gdn_pallas_layer(w, u, q, k, v, g, beta):
    """
    Public entry point. 
    Forward: Uses Pallas Kernel.
    Backward: Uses JAX Reference VJP.
    """
    return _gdn_pallas_forward(w, u, q, k, v, g, beta)

def _gdn_pallas_forward(w, u, q, k, v, g, beta):
    """Invokes the Pallas kernel."""
    B, H, N_chunks, C, Dk = k.shape
    _, _, _, _, Dv = v.shape
    
    in_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dk))
    val_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))
    scalar_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, N_chunks, C))
    out_spec = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))

    kernel_fn = functools.partial(
        gdn_scan_kernel_tpu,
        num_chunks=N_chunks, chunk_size=C, key_dim=Dk, val_dim=Dv, dtype=v.dtype
    )

    out = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((B, H, N_chunks, C, Dv), v.dtype),
        grid=(B, H),
        in_specs=[in_specs, val_specs, in_specs, in_specs, val_specs, scalar_specs, scalar_specs],
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    )(w, u, q, k, v, g, beta)
    
    # Return output and residuals for backward pass
    return out, (w, u, q, k, v, g, beta)

def _gdn_pallas_backward(residuals, grad_out):
    """Uses the JAX reference implementation to calculate gradients."""
    w, u, q, k, v, g, beta = residuals
    
    # We use jax.vjp on the reference function to get gradients
    # This runs the JAX version of the forward pass to setup the backward pass
    _, vjp_fn = jax.vjp(_gdn_reference, w, u, q, k, v, g, beta)
    
    # Compute gradients
    grads = vjp_fn(grad_out)
    return grads

# Register the forward and backward functions
gdn_pallas_layer.defvjp(_gdn_pallas_forward, _gdn_pallas_backward)