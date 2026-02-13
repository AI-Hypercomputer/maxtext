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
    w_ref, u_ref, q_ref, k_ref, v_ref, g_ref, beta_ref, h_init_ref, 
    o_ref, h_final_ref,
    # Hyperparameters
    num_chunks: int, chunk_size: int, key_dim: int, val_dim: int,
    dtype: jnp.dtype = jnp.bfloat16
):
    # 1. Load Initial State
    h = h_init_ref[0, 0].astype(jnp.float32)

    for i in range(num_chunks):
        # 2. Load Inputs
        w = w_ref[0, 0, i] 
        u = u_ref[0, 0, i] 
        q = q_ref[0, 0, i] 
        k = k_ref[0, 0, i] 
        v = v_ref[0, 0, i] 
        g = g_ref[0, 0, i] 
        beta = beta_ref[0, 0, i] 

        # 3. Output Computation
        # Inter-chunk: q * exp(g) @ h
        g_exp = jnp.exp(g.astype(jnp.float32))
        q_g = q.astype(jnp.float32) * g_exp[:, None]
        term1 = jnp.dot(q_g, h) 

        # Intra-chunk: (q @ k.T * decay) @ v
        attn = jnp.dot(q.astype(jnp.float32), k.astype(jnp.float32).T)
        
        # Decay: exp(g[i] - g[j])
        g_diff = g.astype(jnp.float32)[:, None] - g.astype(jnp.float32)[None, :]
        
        # FIX: Apply mask BEFORE exp to prevent Inf * 0 = NaN
        # Use float32 arithmetic for masking to avoid boolean type issues in Mosaic
        mask_val = jnp.tri(chunk_size, dtype=jnp.float32)
        
        # For upper triangle (mask=0), set g_diff to -1e30 so exp() becomes 0
        # g_diff_masked = g_diff * 1.0 + (1.0 - 0.0) * -1e30 = -1e30
        large_neg = -1e30
        g_diff = g_diff * mask_val + (1.0 - mask_val) * large_neg
        
        attn_decay = jnp.exp(g_diff)
        attn = attn * attn_decay

        # Apply Beta gates
        attn = attn * beta.astype(jnp.float32)[:, None] 
        
        # V projection
        term2 = jnp.dot(attn, v.astype(jnp.float32))
        
        o_chunk = term1 + term2
        o_ref[0, 0, i] = o_chunk.astype(dtype)

        # 4. State Update
        chunk_decay = jnp.exp(g[chunk_size - 1]) 
        update = jnp.dot(w.astype(jnp.float32).T, u.astype(jnp.float32))
        h = h * chunk_decay + update

    # 5. Store Final State
    h_final_ref[0, 0] = h.astype(dtype)

# ==============================================================================
# 2. JAX Reference Implementation (For Autodiff)
# ==============================================================================
def _gdn_reference(w, u, q, k, v, g, beta, h_init):
    """Pure JAX equivalent for autodiff."""
    perm_vec = (2, 0, 1, 3, 4)
    perm_scl = (2, 0, 1, 3)
    
    w_s = w.transpose(perm_vec)
    u_s = u.transpose(perm_vec)
    q_s = q.transpose(perm_vec)
    k_s = k.transpose(perm_vec)
    v_s = v.transpose(perm_vec)
    g_s = g.transpose(perm_scl)
    beta_s = beta.transpose(perm_scl)
    
    h_curr = h_init.astype(jnp.float32)
    B, H, N, C, Dk = k.shape
    
    def scan_body(h, args):
        wt, ut, qt, kt, vt, gt, betat = args
        
        # Inter-chunk
        gt_exp = jnp.exp(gt.astype(jnp.float32))
        q_g = qt.astype(jnp.float32) * gt_exp[..., None]
        term1 = jnp.matmul(q_g, h)
        
        # Intra-chunk
        attn = jnp.matmul(qt.astype(jnp.float32), kt.astype(jnp.float32).swapaxes(-1, -2))
        
        # Decay (g[i] - g[j])
        g_diff = gt[..., :, None] - gt[..., None, :]
        
        # Mask before exp (match Pallas logic)
        mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
        g_diff = g_diff * mask + (1.0 - mask) * -1e30
        
        attn = attn * jnp.exp(g_diff)
        attn = attn * betat.astype(jnp.float32)[..., None]
        term2 = jnp.matmul(attn, vt.astype(jnp.float32))
        
        out = (term1 + term2).astype(v.dtype)
        
        # Update
        chunk_decay = jnp.exp(gt[..., -1])[..., None, None]
        update = jnp.matmul(wt.astype(jnp.float32).swapaxes(-1, -2), ut.astype(jnp.float32))
        h_new = h * chunk_decay + update
        
        return h_new, out

    h_final, o_scan = lax.scan(
        scan_body, 
        h_curr, 
        (w_s, u_s, q_s, k_s, v_s, g_s, beta_s)
    )
    
    return o_scan.transpose(1, 2, 0, 3, 4), h_final.astype(v.dtype)

# ==============================================================================
# 3. Custom VJP Registration
# ==============================================================================

def _gdn_pallas_forward(w, u, q, k, v, g, beta, h_init):
    B, H, N_chunks, C, Dk = k.shape
    _, _, _, _, Dv = v.shape
    
    # Specs
    in_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dk))
    val_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))
    scalar_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, N_chunks, C))
    out_spec = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))
    state_spec = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, Dk, Dv))

    kernel_fn = functools.partial(
        gdn_scan_kernel_tpu,
        num_chunks=N_chunks, chunk_size=C, key_dim=Dk, val_dim=Dv, dtype=v.dtype
    )

    out, h_final = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct((B, H, N_chunks, C, Dv), v.dtype), 
            jax.ShapeDtypeStruct((B, H, Dk, Dv), v.dtype)
        ],
        grid=(B, H),
        in_specs=[in_specs, val_specs, in_specs, in_specs, val_specs, scalar_specs, scalar_specs, state_spec],
        out_specs=[out_spec, state_spec],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    )(w, u, q, k, v, g, beta, h_init)
    
    return (out, h_final), (w, u, q, k, v, g, beta, h_init)

def _gdn_pallas_backward(residuals, grad_out_tuple):
    grad_out, _ = grad_out_tuple 
    w, u, q, k, v, g, beta, h_init = residuals
    
    _, vjp_fn = jax.vjp(_gdn_reference, w, u, q, k, v, g, beta, h_init)
    
    grad_h_final = jnp.zeros_like(h_init)
    grads = vjp_fn((grad_out, grad_h_final))
    return grads

@functools.partial(jax.custom_vjp, nondiff_argnums=())
def gdn_pallas_layer(w, u, q, k, v, g, beta, h_init):
    # Returns (output, final_state)
    res, _ = _gdn_pallas_forward(w, u, q, k, v, g, beta, h_init)
    return res

gdn_pallas_layer.defvjp(_gdn_pallas_forward, _gdn_pallas_backward)