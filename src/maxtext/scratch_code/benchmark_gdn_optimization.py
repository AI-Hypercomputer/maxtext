import time
import functools
import types
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
from flax import nnx
from typing import Any, cast

# Import common dependencies
from MaxText import common_types
from MaxText.layers import normalizations
from MaxText.layers.linears import DenseGeneral
from MaxText.layers import qwen3 

# ==============================================================================
# SECTION 1: THE MIRROR (BASELINE IMPLEMENTATION)
# ==============================================================================
def baseline_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None, use_qk_norm_in_gdn=False
):
    """The ORIGINAL implementation (Do not edit this function)."""
    initial_dtype = query.dtype
    if use_qk_norm_in_gdn:
        query = normalizations.l2norm(query, dim=-1, eps=1e-6)
        key = normalizations.l2norm(key, dim=-1, eps=1e-6)

    query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
    key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
    value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
    beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
    g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))

    total_sequence_length = sequence_length + pad_size
    scale = jax.lax.rsqrt(jnp.array(query.shape[-1]).astype(jnp.float32))
    query = query * scale

    v_beta = value * jnp.expand_dims(beta, -1)
    k_beta = key * jnp.expand_dims(beta, -1)

    num_chunks = total_sequence_length // chunk_size
    query_c = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    key_c = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    k_beta_c = k_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    v_beta_c = v_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    g_c = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

    mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

    g_cumsum = jnp.cumsum(g_c, axis=-1)
    g_diff = jnp.expand_dims(g_cumsum, -1) - jnp.expand_dims(g_cumsum, -2)
    g_diff_tril = jnp.tril(g_diff)
    g_diff_exp = jnp.exp(g_diff_tril).astype(jnp.float32)
    decay_mask = g_diff_exp

    prec = jax.lax.Precision.HIGHEST
    attn = -jnp.matmul(k_beta_c, jnp.swapaxes(key_c, -1, -2), precision=prec) * decay_mask
    attn = jnp.where(mask, 0.0, attn)

    def inner_attn_body(i, attn_val):
        indices = jnp.arange(chunk_size)
        col_mask = indices < i
        row = attn_val[..., i, :] * col_mask
        sub_mask = jnp.expand_dims(indices < i, -1) & (indices < i)
        sub = attn_val * sub_mask
        row_exp = jnp.expand_dims(row, -1)
        term = row_exp * sub
        summed = jnp.sum(term, axis=-2)
        update_val = row + summed
        original_row = attn_val[..., i, :]
        new_row = jnp.where(col_mask, update_val, original_row)
        return attn_val.at[..., i, :].set(new_row)

    attn = jax.lax.fori_loop(1, chunk_size, inner_attn_body, attn)
    attn = attn + jnp.eye(chunk_size, dtype=attn.dtype)
    value_intra = jnp.matmul(attn, v_beta_c, precision=prec)
    k_cumdecay = jnp.matmul(attn, (k_beta_c * jnp.expand_dims(jnp.exp(g_cumsum), -1)), precision=prec)

    output_final_state = initial_state is not None
    if initial_state is None:
        last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=value_intra.dtype)
    else:
        last_recurrent_state = initial_state.astype(value_intra.dtype)

    mask_inter = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

    query_scan = jnp.transpose(query_c, (2, 0, 1, 3, 4))
    key_scan = jnp.transpose(key_c, (2, 0, 1, 3, 4))
    value_scan = jnp.transpose(value_intra, (2, 0, 1, 3, 4))
    k_cumdecay_scan = jnp.transpose(k_cumdecay, (2, 0, 1, 3, 4))
    g_scan = jnp.transpose(g_cumsum, (2, 0, 1, 3))
    decay_mask_scan = jnp.transpose(decay_mask, (2, 0, 1, 3, 4))

    xs = (query_scan, key_scan, value_scan, k_cumdecay_scan, g_scan, decay_mask_scan)

    def scan_body(prev_state, x):
        q_i, k_i, v_i, k_cumdecay_i, g_i, decay_mask_i = x
        last_recurrent_state = prev_state
        prec = jax.lax.Precision.HIGHEST

        attn_i = jnp.matmul(q_i, jnp.swapaxes(k_i, -1, -2), precision=prec) * decay_mask_i
        attn_i = jnp.where(mask_inter, 0.0, attn_i)

        v_prime = jnp.matmul(k_cumdecay_i, last_recurrent_state, precision=prec)
        v_new = v_i - v_prime

        g_i_exp = jnp.exp(g_i)
        attn_inter = jnp.matmul(q_i * jnp.expand_dims(g_i_exp, -1), last_recurrent_state, precision=prec)

        core_attn_out_i = attn_inter + jnp.matmul(attn_i, v_new, precision=prec)

        g_i_last_exp = jnp.exp(g_i[..., -1, None, None])
        new_last_recurrent_state = last_recurrent_state * g_i_last_exp

        g_diff_exp = jnp.expand_dims(jnp.exp(jnp.expand_dims(g_i[..., -1], -1) - g_i), -1)
        k_i_g_diff = k_i * g_diff_exp

        update_term = jnp.matmul(jnp.swapaxes(k_i_g_diff, -1, -2), v_new, precision=prec)
        new_last_recurrent_state = new_last_recurrent_state + update_term

        return new_last_recurrent_state, core_attn_out_i

    final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

    core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length, :]
    core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

    return core_attn_out, final_state if output_final_state else None


class BaselineGatedDeltaNet(nnx.Module):
    """The Mirror/Baseline Wrapper."""
    def __init__(self, config, *, rngs):
        self.config = config
        cfg = self.config
        
        in_features = cfg.emb_dim
        self.num_v_heads = cfg.gdn_num_value_heads
        self.num_k_heads = cfg.gdn_num_key_heads
        self.head_k_dim = cfg.gdn_key_head_dim
        self.head_v_dim = cfg.gdn_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        conv_dim = self.key_dim * 2 + self.value_dim
        conv_kernel_size = cfg.gdn_conv_kernel_dim
        self.v_heads_per_k_head = self.num_v_heads // self.num_k_heads

        self.in_proj_qkvz = DenseGeneral(in_features, (self.key_dim * 2 + self.value_dim * 2), dtype=cfg.dtype, kernel_axes=("embed", "mlp"), matmul_precision=cfg.matmul_precision, rngs=rngs)
        self.in_proj_ba = DenseGeneral(in_features, (self.num_v_heads * 2), dtype=cfg.dtype, kernel_axes=("embed", "mlp"), matmul_precision=cfg.matmul_precision, rngs=rngs)
        
        self.conv1d = nnx.Conv(conv_dim, conv_dim, kernel_size=(conv_kernel_size,), feature_group_count=conv_dim, padding="CAUSAL", use_bias=False, dtype=cfg.dtype, precision=cfg.matmul_precision, rngs=rngs)

        def a_log_init(key, shape, dtype=jnp.float32):
            a_vals = jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0)
            return jnp.log(a_vals)

        self.A_log = nnx.Param(a_log_init(rngs.params(), (self.num_v_heads,)))
        self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (self.num_v_heads,)))

        self.norm = normalizations.Qwen3NextRMSNormGated(self.head_v_dim, eps=cfg.normalization_layer_epsilon, dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, rngs=rngs)
        self.out_proj = DenseGeneral(self.value_dim, (in_features,), dtype=cfg.dtype, kernel_axes=("mlp", "embed"), matmul_precision=cfg.matmul_precision, rngs=rngs)

    def __call__(self, hidden_states):
        cfg = self.config
        batch, seq_len, _ = hidden_states.shape
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)

        new_shape_qkvz = (batch, seq_len, self.num_k_heads, 2 * self.head_k_dim + 2 * self.head_v_dim * self.v_heads_per_k_head)
        mixed_qkvz = qkvz.reshape(new_shape_qkvz)
        split_indices_qkvz = [self.head_k_dim, 2 * self.head_k_dim, 2 * self.head_k_dim + (self.v_heads_per_k_head * self.head_v_dim)]
        query, key, value_raw, z_raw = jnp.split(mixed_qkvz, split_indices_qkvz, axis=3)
        value = value_raw.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        z = z_raw.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        new_shape_ba = (batch, seq_len, self.num_k_heads, 2 * self.v_heads_per_k_head)
        mixed_ba = ba.reshape(new_shape_ba)
        b_raw, a_raw = jnp.split(mixed_ba, [self.v_heads_per_k_head], axis=3)
        b = b_raw.reshape(batch, seq_len, self.num_v_heads)
        a = a_raw.reshape(batch, seq_len, self.num_v_heads)

        q = query.reshape(batch, seq_len, -1)
        k = key.reshape(batch, seq_len, -1)
        v = value.reshape(batch, seq_len, -1)
        qkv = jnp.concatenate([q, k, v], axis=-1)
        
        conv_out = self.conv1d(qkv)
        qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(cfg.dtype)
        q_conv, k_conv, v_conv = jnp.split(qkv_conv, [self.key_dim, 2 * self.key_dim], axis=-1)

        query = q_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = k_conv.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = v_conv.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        # FIXED .value DEPRECATION
        A_log = self.A_log[...]
        dt_bias = self.dt_bias[...]
        beta = jax.nn.sigmoid(b)
        g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias.astype(jnp.float32))
        g = g.astype(cfg.dtype)

        if self.num_v_heads > self.num_k_heads and self.num_v_heads % self.num_k_heads == 0:
            repeats = self.num_v_heads // self.num_k_heads
            query = jnp.repeat(query, repeats, axis=2)
            key = jnp.repeat(key, repeats, axis=2)

        # USING BASELINE KERNEL
        core_attn_out, _ = baseline_chunk_gated_delta_rule(
            query, key, value, g, beta, chunk_size=cfg.gdn_chunk_size, use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn
        )

        gated_output_reshaped = self.norm(core_attn_out, z)
        gated_output = gated_output_reshaped.reshape(batch, seq_len, -1)
        output = self.out_proj(gated_output)
        return output


# ==============================================================================
# SECTION 2: BENCHMARK HARNESS
# ==============================================================================

def run_comparison():
    backend = jax.extend.backend.get_backend().platform
    print(f"\nDevice: {jax.devices()[0]} ({backend})")
    
    # --- CONFIGURATION ---
    if backend == 'tpu':
        # REAL BENCHMARK SETTINGS (Heavy)
        DTYPE = jnp.bfloat16
        BATCH = 2
        SEQ_LEN = 4096      
        ITERS = 20          
        WARMUP = 5
    else:
        # CPU DEBUG SETTINGS (Fast)
        print("⚠️  Running on CPU: Using reduced dimensions for speed.")
        DTYPE = jnp.float32 
        BATCH = 1
        SEQ_LEN = 128       
        ITERS = 5           
        WARMUP = 1

    HIDDEN_SIZE = 2048
    NUM_KEY_HEADS = 16
    NUM_VALUE_HEADS = 32
    HEAD_DIM = 128
    CONV_KERNEL_DIM = 4
    CHUNK_SIZE = 64
    PROFILE_DIR = "/tmp/maxtext_gdn_profile"
    
    print(f"Config: Batch={BATCH}, SeqLen={SEQ_LEN}, Dtype={DTYPE}")
    print(f"Model: H={HIDDEN_SIZE}, K_Heads={NUM_KEY_HEADS}, V_Heads={NUM_VALUE_HEADS}, HeadDim={HEAD_DIM}")

    dummy_config = types.SimpleNamespace(
        emb_dim=HIDDEN_SIZE,
        gdn_num_value_heads=NUM_VALUE_HEADS,
        gdn_num_key_heads=NUM_KEY_HEADS,
        gdn_key_head_dim=HEAD_DIM,
        gdn_value_head_dim=HEAD_DIM,
        gdn_conv_kernel_dim=CONV_KERNEL_DIM,
        dtype=DTYPE,
        matmul_precision='default', 
        normalization_layer_epsilon=1e-6,
        weight_dtype=DTYPE,
        gdn_chunk_size=CHUNK_SIZE,
        use_qk_norm_in_gdn=True,
        load_balance_loss_weight=0.0,
        scan_layers=False
    )

    # 1. INSTANTIATE MODELS
    print("Initializing models...")
    rngs_base = nnx.Rngs(0)
    baseline_model = BaselineGatedDeltaNet(config=dummy_config, rngs=rngs_base)
    
    rngs_opt = nnx.Rngs(0)
    optimized_model = qwen3.Qwen3NextGatedDeltaNet(config=dummy_config, rngs=rngs_opt)

    # 2. WEIGHT SYNCHRONIZATION
    _, params_state = nnx.split(optimized_model)
    nnx.update(baseline_model, params_state)
    print("✅ Models synchronized with identical weights.")

    # 3. INPUTS
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (BATCH, SEQ_LEN, HIDDEN_SIZE), dtype=DTYPE)

    # -------------------------------------------------------------------------
    # Helper: Pure Functional wrappers to avoid Flax/JAX Version mismatch issues
    # -------------------------------------------------------------------------
    def create_jitted_train_step(model):
        graphdef, params = nnx.split(model)
        
        @jax.jit
        def pure_train_step(params, x):
            m = nnx.merge(graphdef, params)
            def loss_fn(m_inner):
                y = m_inner(x)
                return jnp.mean(y)
            loss, grads = nnx.value_and_grad(loss_fn)(m)
            return loss, grads
            
        return pure_train_step, params

    def create_jitted_forward(model):
        graphdef, params = nnx.split(model)
        
        @jax.jit
        def pure_forward(params, x):
            m = nnx.merge(graphdef, params)
            return m(x)
            
        return pure_forward, params

    # ==============================================================================
    # PART A: LOGICAL CORRECTNESS
    # ==============================================================================
    print("\n--- Checking Logical Correctness ---")

    # Create safe functional wrappers
    jit_train_base, params_base = create_jitted_train_step(baseline_model)
    jit_train_opt, params_opt = create_jitted_train_step(optimized_model)

    loss_base, grads_base = jit_train_base(params_base, inputs)
    jax.block_until_ready((loss_base, grads_base))

    loss_opt, grads_opt = jit_train_opt(params_opt, inputs)
    jax.block_until_ready((loss_opt, grads_opt))

    diff_loss = jnp.abs(loss_base - loss_opt)
    print(f"Forward Pass Loss Diff: {float(diff_loss):.2e}")

    flat_grads_base, _ = jax.tree_util.tree_flatten(grads_base)
    flat_grads_opt, _ = jax.tree_util.tree_flatten(grads_opt)
    
    max_grad_diff = 0.0
    for g1, g2 in zip(flat_grads_base, flat_grads_opt):
        if hasattr(g1, 'shape'):
            d = jnp.max(jnp.abs(g1 - g2))
            max_grad_diff = max(max_grad_diff, float(d))

    print(f"Backward Pass Grad Diff: {max_grad_diff:.2e}")

    if max_grad_diff > 1e-2:
        print("WARNING: Significant divergence in gradients!")
    else:
        print("✅ Outputs & Gradients match within tolerance.")

    # ==============================================================================
    # PART B: SPEED BENCHMARKING
    # ==============================================================================
    print("\n--- Performance Benchmark ---")

    def benchmark_func(name, func, *args):
        print(f"Benchmarking {name}...")
        # Warmup
        for _ in range(WARMUP):
            out = func(*args)
            jax.block_until_ready(out)
        
        # Time it
        t0 = time.time()
        for _ in range(ITERS):
            out = func(*args)
            jax.block_until_ready(out)
        t_avg = (time.time() - t0) / ITERS * 1000
        print(f"  -> {t_avg:.2f} ms")
        return t_avg
    
    # Create forward-only wrappers
    jit_fwd_base, _ = create_jitted_forward(baseline_model)
    jit_fwd_opt, _ = create_jitted_forward(optimized_model)
    
    t_fwd_base = benchmark_func("Baseline Forward", jit_fwd_base, params_base, inputs)
    t_fwd_opt = benchmark_func("Optimized Forward", jit_fwd_opt, params_opt, inputs)
    
    t_train_base = benchmark_func("Baseline Train Step", jit_train_base, params_base, inputs)
    t_train_opt = benchmark_func("Optimized Train Step", jit_train_opt, params_opt, inputs)

    print(f"\n--- Results ---")
    print(f"Forward Speedup:       {t_fwd_base/t_fwd_opt:.2f}x  ({t_fwd_base:.2f}ms -> {t_fwd_opt:.2f}ms)")
    print(f"Training Step Speedup: {t_train_base/t_train_opt:.2f}x  ({t_train_base:.2f}ms -> {t_train_opt:.2f}ms)")

    # ==============================================================================
    # PART C: STATIC MEMORY ANALYSIS
    # ==============================================================================
    print("\n--- Static Memory Analysis (Compiler Estimate) ---")
    
    def analyze_memory(name, func, *args):
        print(f"Analyzing {name}...")
        try:
            compiled = func.lower(*args).compile()
            mem_analysis = compiled.memory_analysis()
            
            if mem_analysis is None:
                print("  Memory analysis not supported on this backend/version.")
                return 0
            
            # JAX 0.8.0+: The string representation is the most reliable way to read stats
            print(f"  {mem_analysis}")
            
            # Try to grab bytes if available, otherwise return 0 (skipping reduction calc)
            if hasattr(mem_analysis, 'temp_size_in_bytes'):
                return mem_analysis.temp_size_in_bytes
            return 0
            
        except Exception as e:
            print(f"  Memory analysis failed: {e}")
            return 0

    mem_base = analyze_memory("Baseline Train Step", jit_train_base, params_base, inputs)
    mem_opt = analyze_memory("Optimized Train Step", jit_train_opt, params_opt, inputs)
    
    if mem_base > 0 and mem_opt > 0:
        reduction = (mem_base - mem_opt) / mem_base * 100
        # Positive reduction = Good (Saved memory)
        # Negative reduction = Bad (Regression)
        print(f"\nMemory Reduction: {reduction:.2f}% (Higher is better)")

    # ==============================================================================
    # PART D: PROFILING
    # ==============================================================================
    print(f"\n--- Profiling Optimized Implementation ---")
    print(f"Saving trace to: {PROFILE_DIR}")
    
    try:
        jax.profiler.start_trace(PROFILE_DIR)
        for _ in range(WARMUP): # Use warmup count just to get a few samples
            out = jit_train_opt(params_opt, inputs)
            jax.block_until_ready(out)
        jax.profiler.stop_trace()
        print("Profiling complete.")
    except Exception as e:
        print(f"Profiling failed (possibly already active): {e}")
        
    print(f"Path: {PROFILE_DIR}")

if __name__ == "__main__":
    run_comparison()