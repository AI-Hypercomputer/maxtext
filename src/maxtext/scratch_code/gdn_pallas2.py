# src/MaxText/kernels/gdn_pallas_optimized.py
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
    """Forward kernel for GDN layer.
    
    Args:
        w_ref: (N_chunks, C, Dk) # Memory: VMEM
        u_ref: (N_chunks, C, Dv) # Memory: VMEM
        q_ref: (N_chunks, C, Dk) # Memory: VMEM
        k_ref: (N_chunks, C, Dk) # Memory: VMEM
        v_ref: (N_chunks, C, Dv) # Memory: VMEM
        g_ref: (N_chunks, C)     # Memory: VMEM
        beta_ref: (N_chunks, C)  # Memory: VMEM
        h_init_ref: (Dk, Dv)     # Memory: VMEM
        o_ref: (N_chunks, C, Dv) # Memory: VMEM (Output)
        h_final_ref: (Dk, Dv)    # Memory: VMEM (Output)
    """

    # 1. Load Initial State
    h = h_init_ref[0, 0].astype(jnp.float32)

    for i in range(num_chunks):

        # 2. Load Inputs
        w = w_ref[0, 0, i] 
        u = u_ref[0, 0, i] 
        q = q_ref[0, 0, i] 
        k = k_ref[0, 0, i] 
        v = v_ref[0, 0, i] 
        # Cast g to float32 immediately to ensure scalar indexing works (Mosaic restriction)
        g = g_ref[0, 0, i].astype(jnp.float32)
        beta = beta_ref[0, 0, i] 

        # 3. Output Computation
        # Inter-chunk: q * exp(g) @ h
        g_exp = jnp.exp(g)
        q_g = q.astype(jnp.float32) * g_exp[:, None]
        term1 = jnp.dot(q_g, h) 

        # Intra-chunk: (q @ k.T * decay) @ v
        attn = jnp.dot(q.astype(jnp.float32), k.astype(jnp.float32).T)
        
        # Decay: exp(g[i] - g[j])
        g_diff = g[:, None] - g[None, :]
        
        # FIX: Apply mask BEFORE exp to prevent Inf * 0 = NaN
        # Use float32 arithmetic for masking to avoid boolean type issues in Mosaic
        mask_val = jnp.tri(chunk_size, dtype=jnp.float32)
        
        # For upper triangle (mask=0), set g_diff to -1e30 so exp() becomes 0
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
        # g is already float32, so g[idx] yields a float32 scalar, which is valid
        chunk_decay = jnp.exp(g[chunk_size - 1]) 
        update = jnp.dot(w.astype(jnp.float32).T, u.astype(jnp.float32))
        h = h * chunk_decay + update
        

    # 5. Store Final State
    h_final_ref[0, 0] = h.astype(dtype)
    

# ==============================================================================
# 2. Pallas Kernel Implementation (Backward Pass Logic)
# ==============================================================================
def gdn_backward_kernel_tpu(
    w_ref, u_ref, q_ref, k_ref, v_ref, g_ref, beta_ref, h_init_ref, 
    grad_o_ref, grad_h_final_ref,
    grad_w_ref, grad_u_ref, grad_q_ref, grad_k_ref, grad_v_ref, grad_g_ref, grad_beta_ref, grad_h_init_ref,
    h_buffer_ref, # Scratch buffer for state recomputation
    # Hyperparameters
    num_chunks: int, chunk_size: int, key_dim: int, val_dim: int,
    dtype: jnp.dtype = jnp.bfloat16
):
    """Backward kernel for GDN layer.
    
    Strategy: Recompute-and-Backprop
    1. Re-run forward pass to store intermediate states `h` in VMEM buffer.
    2. Iterate backwards to compute gradients.
    
    Args:
        w_ref, u_ref, ...: Inputs (N_chunks, C, ...) # Memory: VMEM
        grad_o_ref: Output gradients (N_chunks, C, Dv) # Memory: VMEM
        grad_h_final_ref: Final state gradient (Dk, Dv) # Memory: VMEM
        grad_w_ref, ...: Input gradients (N_chunks, C, ...) # Memory: VMEM (Accumulators)
        h_buffer_ref: Scratch memory (N_chunks, Dk, Dv) # Memory: VMEM
    """

    # --------------------------------------------------------------------------
    # Phase 1: Forward Recompute (Fill h_buffer)
    # --------------------------------------------------------------------------
    h = h_init_ref[0, 0].astype(jnp.float32) # Load initial state # Load: VMEM -> Registers

    for i in range(num_chunks):

        # Store current h to buffer (state before update)
        # Use Ref assignment instead of at[].set to avoid scatter on local arrays
        h_buffer_ref[i] = h 
        
        # Load Inputs
        w = w_ref[0, 0, i] 
        u = u_ref[0, 0, i] 
        # Cast g to float32 immediately to ensure scalar indexing works
        g = g_ref[0, 0, i].astype(jnp.float32)
        
        # State Update Logic Only (don't need full output logic here)
        chunk_decay = jnp.exp(g[chunk_size - 1])
        update = jnp.dot(w.astype(jnp.float32).T, u.astype(jnp.float32))
        h = h * chunk_decay + update

    # --------------------------------------------------------------------------
    # Phase 2: Backward Scan
    # --------------------------------------------------------------------------
    grad_h = grad_h_final_ref[0, 0].astype(jnp.float32) # Load: VMEM -> Registers
    
    # Iterate backwards from last chunk
    for i in range(num_chunks - 1, -1, -1):

        # Load Inputs for this chunk
        w = w_ref[0, 0, i].astype(jnp.float32)
        u = u_ref[0, 0, i].astype(jnp.float32)
        q = q_ref[0, 0, i].astype(jnp.float32)
        k = k_ref[0, 0, i].astype(jnp.float32)
        v = v_ref[0, 0, i].astype(jnp.float32)
        g = g_ref[0, 0, i].astype(jnp.float32)
        beta = beta_ref[0, 0, i].astype(jnp.float32)
        grad_o = grad_o_ref[0, 0, i].astype(jnp.float32)
        
        # Load state h from buffer (state at start of this chunk)
        h = h_buffer_ref[i] # Load: VMEM -> Registers
        
        # --- Gradients from State Update (h_new = h * decay + update) ---
        # grad_h is dL/dh_new coming from future
        
        # grad_update = grad_h
        grad_update = grad_h
        
        # grad_w = u @ grad_update.T
        grad_w = jnp.dot(u, grad_update.T)
        
        # grad_u = w @ grad_update
        grad_u = jnp.dot(w, grad_update)
        
        # grad_chunk_decay = sum(grad_h * h)
        chunk_decay = jnp.exp(g[chunk_size - 1])
        grad_chunk_decay = jnp.sum(grad_h * h)
        
        # grad_h_prev (part 1) = grad_h * chunk_decay
        grad_h_prev = grad_h * chunk_decay
        
        # Contribution to grad_g from chunk_decay
        # FIX: Avoid scatter (at[].set) on local array. Use mask instead.
        mask = (jnp.arange(chunk_size) == (chunk_size - 1)).astype(jnp.float32)
        grad_g_from_decay = mask * (grad_chunk_decay * chunk_decay)

        # --- Gradients from Output (o = term1 + term2) ---
        grad_term1 = grad_o
        grad_term2 = grad_o
        
        # --- Gradients from Term 1 (Inter-chunk: q * exp(g) @ h) ---
        # term1 = (q * exp(g)) @ h
        g_exp = jnp.exp(g)
        q_g = q * g_exp[:, None]
        
        # grad_q_g = grad_term1 @ h.T
        grad_q_g = jnp.dot(grad_term1, h.T)
        
        # grad_h_prev (part 2) += q_g.T @ grad_term1
        grad_h_prev += jnp.dot(q_g.T, grad_term1)
        
        # grad_q = grad_q_g * exp(g)
        grad_q = grad_q_g * g_exp[:, None]
        
        # grad_g (part 1) = sum(grad_q_g * q * exp(g), axis=1)
        grad_g_term1 = jnp.sum(grad_q_g * q_g, axis=1)
        
        # --- Gradients from Term 2 (Intra-chunk) ---
        # term2 = attn @ v
        # attn = (q @ k.T) * exp(g_diff) * beta
        
        attn_logits = jnp.dot(q, k.T)
        
        # Recompute g_diff and mask
        g_diff = g[:, None] - g[None, :]
        mask_val = jnp.tri(chunk_size, dtype=jnp.float32)
        large_neg = -1e30
        g_diff_masked = g_diff * mask_val + (1.0 - mask_val) * large_neg
        attn_decay = jnp.exp(g_diff_masked)
        
        # attn = attn_logits * attn_decay * beta
        
        # grad_v = attn.T @ grad_term2
        # Reconstruct full attn for grad_v
        attn = attn_logits * attn_decay * beta[:, None]
        grad_v = jnp.dot(attn.T, grad_term2)
        
        # grad_attn = grad_term2 @ v.T
        grad_attn = jnp.dot(grad_term2, v.T)
        
        # grad_beta
        # attn = pre_beta_attn * beta
        pre_beta_attn = attn_logits * attn_decay
        grad_beta = jnp.sum(grad_attn * pre_beta_attn, axis=1)
        
        # grad_pre_beta_attn = grad_attn * beta
        grad_pre_beta_attn = grad_attn * beta[:, None]
        
        # grad_attn_decay = grad_pre_beta_attn * attn_logits
        grad_attn_decay = grad_pre_beta_attn * attn_logits
        
        # grad_g_diff_masked = grad_attn_decay * attn_decay
        grad_g_diff_masked = grad_attn_decay * attn_decay
        
        # grad_g (part 2) from g_diff
        # Mask handling: gradients only flow where mask=1
        grad_g_diff = grad_g_diff_masked * mask_val
        grad_g_term2 = jnp.sum(grad_g_diff, axis=1) - jnp.sum(grad_g_diff, axis=0)
        
        # grad_attn_logits = grad_pre_beta_attn * attn_decay
        grad_attn_logits = grad_pre_beta_attn * attn_decay
        
        # grad_q (part 2) = grad_attn_logits @ k
        grad_q += jnp.dot(grad_attn_logits, k)
        
        # grad_k = grad_attn_logits.T @ q
        grad_k = jnp.dot(grad_attn_logits.T, q)
        
        # Total grad_g
        grad_g = grad_g_from_decay + grad_g_term1 + grad_g_term2
        
        # --- Store Gradients ---
        grad_w_ref[0, 0, i] = grad_w.astype(dtype)
        grad_u_ref[0, 0, i] = grad_u.astype(dtype)
        grad_q_ref[0, 0, i] = grad_q.astype(dtype)
        grad_k_ref[0, 0, i] = grad_k.astype(dtype)
        grad_v_ref[0, 0, i] = grad_v.astype(dtype)
        grad_g_ref[0, 0, i] = grad_g.astype(dtype)
        grad_beta_ref[0, 0, i] = grad_beta.astype(dtype)
        
        # Update grad_h for next iteration (previous chunk)
        grad_h = grad_h_prev
        
        
    # Store final grad_h_init
    grad_h_init_ref[0, 0] = grad_h.astype(dtype)
    


# ==============================================================================
# 3. JAX Reference Implementation (For Autodiff)
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
# 4. Custom VJP Registration & Wrappers
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
    grad_out, grad_h_final = grad_out_tuple 
    w, u, q, k, v, g, beta, h_init = residuals
    
    B, H, N_chunks, C, Dk = k.shape
    _, _, _, _, Dv = v.shape
    
    # BlockSpecs for inputs (same as forward)
    in_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dk))
    val_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))
    scalar_specs = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, N_chunks, C))
    state_spec = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, Dk, Dv))
    
    # BlockSpecs for gradients (match inputs)
    grad_out_spec = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0, 0), block_shape=(1, 1, N_chunks, C, Dv))
    grad_state_spec = pl.BlockSpec(index_map=lambda i, j: (i, j, 0, 0), block_shape=(1, 1, Dk, Dv))
    
    # Scratch spec for h_buffer
    scratch_spec = pltpu.VMEM((N_chunks, Dk, Dv), jnp.float32)
    
    kernel_fn = functools.partial(
        gdn_backward_kernel_tpu,
        num_chunks=N_chunks, chunk_size=C, key_dim=Dk, val_dim=Dv, dtype=v.dtype
    )
    
    # Grid spec with scratch
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(B, H),
        in_specs=[
            in_specs, val_specs, in_specs, in_specs, val_specs, scalar_specs, scalar_specs, state_spec, # Inputs
            grad_out_spec, grad_state_spec # Gradients
        ],
        out_specs=[
            in_specs, val_specs, in_specs, in_specs, val_specs, scalar_specs, scalar_specs, state_spec
        ],
        scratch_shapes=[scratch_spec]
    )
    
    # Input order: w, u, q, k, v, g, beta, h_init, grad_o, grad_h_final
    # Output order: grad_w, grad_u, grad_q, grad_k, grad_v, grad_g, grad_beta, grad_h_init
    
    grads = pl.pallas_call(
        kernel_fn,
        out_shape=[
            jax.ShapeDtypeStruct(w.shape, w.dtype), # grad_w
            jax.ShapeDtypeStruct(u.shape, u.dtype), # grad_u
            jax.ShapeDtypeStruct(q.shape, q.dtype), # grad_q
            jax.ShapeDtypeStruct(k.shape, k.dtype), # grad_k
            jax.ShapeDtypeStruct(v.shape, v.dtype), # grad_v
            jax.ShapeDtypeStruct(g.shape, g.dtype), # grad_g
            jax.ShapeDtypeStruct(beta.shape, beta.dtype), # grad_beta
            jax.ShapeDtypeStruct(h_init.shape, h_init.dtype), # grad_h_init
        ],
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
    )(w, u, q, k, v, g, beta, h_init, grad_out, grad_h_final)
    
    return grads

@functools.partial(jax.custom_vjp, nondiff_argnums=())
def gdn_pallas_layer(w, u, q, k, v, g, beta, h_init):
    # Returns (output, final_state)
    res, _ = _gdn_pallas_forward(w, u, q, k, v, g, beta, h_init)
    return res

gdn_pallas_layer.defvjp(_gdn_pallas_forward, _gdn_pallas_backward)

# ==============================================================================
# 5. Main Function (Testing)
# ==============================================================================
def main():
    print("Initializing inputs...")
    B, H, N_chunks, C, Dk, Dv = 1, 2, 4, 16, 32, 32
    key = jax.random.PRNGKey(0)
    
    w = jax.random.normal(key, (B, H, N_chunks, C, Dk), dtype=jnp.bfloat16)
    u = jax.random.normal(key, (B, H, N_chunks, C, Dv), dtype=jnp.bfloat16)
    q = jax.random.normal(key, (B, H, N_chunks, C, Dk), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (B, H, N_chunks, C, Dk), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (B, H, N_chunks, C, Dv), dtype=jnp.bfloat16)
    g = jax.random.normal(key, (B, H, N_chunks, C), dtype=jnp.bfloat16)
    beta = jax.random.normal(key, (B, H, N_chunks, C), dtype=jnp.bfloat16)
    h_init = jax.random.normal(key, (B, H, Dk, Dv), dtype=jnp.bfloat16)
    
    print("Testing Forward Pass (No JIT)...")
    out, h_final = gdn_pallas_layer(w, u, q, k, v, g, beta, h_init)
    print(f"Output shape: {out.shape}")
    print(f"Final state shape: {h_final.shape}")
    
    print("\nTesting Backward Pass (No JIT)...")
    def loss_fn(w, u, q, k, v, g, beta, h_init):
        out, _ = gdn_pallas_layer(w, u, q, k, v, g, beta, h_init)
        return jnp.sum(out)
    
    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(w, u, q, k, v, g, beta, h_init)
    print("Gradients computed successfully.")
    print(f"Grad w shape: {grads[0].shape}")
    
    print("\nTesting with JIT...")
    # FIX: JIT the gradient function, not just the loss function
    jit_grad = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6, 7)))
    grads_jit = jit_grad(w, u, q, k, v, g, beta, h_init)
    print("JIT Gradients computed successfully.")
    print(f"JIT Grad w shape: {grads_jit[0].shape}")

if __name__ == "__main__":
    main()