# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmarking script comparing Hybrid GDN (Tokamax v3 Fwd + Custom VJP Bwd) vs Pure JAX GDN in MaxText."""

import argparse
import functools
import os
import sys
import time
import types
from typing import Any, Tuple

from flax import nnx
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np

# Import MaxText dependencies
from maxtext.common import common_types
from maxtext.layers import normalizations
from maxtext.layers.linears import DenseGeneral
from maxtext.models import hybrid_gdn
from maxtext.models import qwen3


# ==============================================================================
# SECTION 1: CONFIGURATION & MODEL HELPERS
# ==============================================================================

def create_model_configs(
    hidden_size: int = 2048,
    num_key_heads: int = 16,
    num_value_heads: int = 32,
    head_dim: int = 128,
    conv_kernel_dim: int = 4,
    chunk_size: int = 64,
    dtype: jnp.dtype = jnp.bfloat16,
    use_qk_norm: bool = True,
) -> Tuple[types.SimpleNamespace, types.SimpleNamespace]:
    """Creates configurations for both Pure JAX GDN and Hybrid GDN."""
    base_dict = dict(
        emb_dim=hidden_size,
        gdn_num_value_heads=num_value_heads,
        gdn_num_key_heads=num_key_heads,
        gdn_key_head_dim=head_dim,
        gdn_value_head_dim=head_dim,
        gdn_conv_kernel_dim=conv_kernel_dim,
        dtype=dtype,
        weight_dtype=dtype,
        matmul_precision="default",
        normalization_layer_epsilon=1e-6,
        gdn_chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm,
        load_balance_loss_weight=0.0,
        scan_layers=False,
        using_pipeline_parallelism=False,
        logical_axis_rules=(),
    )

    # 1. Pure JAX GDN config
    pure_jax_config = types.SimpleNamespace(
        **base_dict,
        use_gdn_kernel=False,
        use_hybrid_gdn=False,
    )

    # 2. Hybrid GDN config (Tokamax GDN v3 forward + Custom VJP backward)
    hybrid_gdn_config = types.SimpleNamespace(
        **base_dict,
        use_gdn_kernel=True,
        use_hybrid_gdn=True,
    )

    return pure_jax_config, hybrid_gdn_config


def create_jitted_train_step(model: nnx.Module, input_shape: Tuple[int, ...]):
    """Creates a pure functional, JIT-compiled training step with position-aware loss."""
    graphdef, params = nnx.split(model)

    proj_key = jax.random.PRNGKey(99)
    projection = jax.random.normal(proj_key, input_shape)

    @jax.jit
    def pure_train_step(params, x):
        m = nnx.merge(graphdef, params)

        def loss_fn(m_inner):
            out = m_inner(x)
            y = out[0] if isinstance(out, tuple) else out
            loss = jnp.mean(y * projection.astype(y.dtype))
            return loss, y

        (loss, y), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
        return loss, y, grads

    return pure_train_step, params


def create_jitted_forward(model: nnx.Module):
    """Creates a pure functional, JIT-compiled forward pass."""
    graphdef, params = nnx.split(model)

    @jax.jit
    def pure_forward(params, x):
        m = nnx.merge(graphdef, params)
        out = m(x)
        y = out[0] if isinstance(out, tuple) else out
        return y

    return pure_forward, params


# ==============================================================================
# SECTION 2: BENCHMARK HARNESS
# ==============================================================================

def run_comparison(
    batch_size: int | None = None,
    seq_len: int | None = None,
    iters: int | None = None,
    warmup: int | None = None,
    dtype_str: str | None = None,
    profile_dir: str = "/tmp/maxtext_gdn_profile",
):
    backend = jax.extend.backend.get_backend().platform
    print(f"\nDevice: {jax.devices()[0]} ({backend})")

    # --- DEFAULT CONFIGURATION BASED ON HARDWARE ---
    if backend == "tpu":
        # REAL TPU BENCHMARK SETTINGS (Heavy)
        DTYPE = jnp.bfloat16 if dtype_str is None else getattr(jnp, dtype_str)
        BATCH = 2 if batch_size is None else batch_size
        SEQ_LEN = 4096 if seq_len is None else seq_len
        ITERS = 20 if iters is None else iters
        WARMUP = 5 if warmup is None else warmup
    else:
        # CPU DEBUG SETTINGS (Fast)
        print("⚠️  Running on CPU: Using reduced dimensions for speed.")
        DTYPE = jnp.float32 if dtype_str is None else getattr(jnp, dtype_str)
        BATCH = 1 if batch_size is None else batch_size
        SEQ_LEN = 128 if seq_len is None else seq_len
        ITERS = 5 if iters is None else iters
        WARMUP = 1 if warmup is None else warmup

    HIDDEN_SIZE = 2048
    NUM_KEY_HEADS = 16
    NUM_VALUE_HEADS = 32
    HEAD_DIM = 128
    CONV_KERNEL_DIM = 4
    CHUNK_SIZE = 64

    print(f"Config: Batch={BATCH}, SeqLen={SEQ_LEN}, Dtype={DTYPE}")
    print(f"Model: H={HIDDEN_SIZE}, K_Heads={NUM_KEY_HEADS}, V_Heads={NUM_VALUE_HEADS}, HeadDim={HEAD_DIM}")

    pure_jax_cfg, hybrid_gdn_cfg = create_model_configs(
        hidden_size=HIDDEN_SIZE,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_dim=CONV_KERNEL_DIM,
        chunk_size=CHUNK_SIZE,
        dtype=DTYPE,
        use_qk_norm=True,
    )

    # 1. INSTANTIATE MODELS
    print("\nInitializing Pure JAX and Hybrid GDN models...")
    rngs_pure = nnx.Rngs(0)
    pure_jax_model = qwen3.Qwen3NextGatedDeltaNet(config=pure_jax_cfg, rngs=rngs_pure)

    rngs_hybrid = nnx.Rngs(0)
    hybrid_model = qwen3.Qwen3NextGatedDeltaNet(config=hybrid_gdn_cfg, rngs=rngs_hybrid)

    # 2. WEIGHT SYNCHRONIZATION
    _, params_state = nnx.split(hybrid_model)
    nnx.update(pure_jax_model, params_state)
    print("✅ Models synchronized with identical weights.")

    # 3. INPUTS
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (BATCH, SEQ_LEN, HIDDEN_SIZE), dtype=DTYPE)

    # ==============================================================================
    # PART A: LOGICAL CORRECTNESS
    # ==============================================================================
    print("\n--- Checking Logical Correctness (Pure JAX vs Hybrid GDN) ---")

    jit_train_pure, params_pure = create_jitted_train_step(pure_jax_model, inputs.shape)
    jit_train_hybrid, params_hybrid = create_jitted_train_step(hybrid_model, inputs.shape)

    loss_pure, out_pure, grads_pure = jit_train_pure(params_pure, inputs)
    jax.block_until_ready((loss_pure, out_pure, grads_pure))

    loss_hybrid, out_hybrid, grads_hybrid = jit_train_hybrid(params_hybrid, inputs)
    jax.block_until_ready((loss_hybrid, out_hybrid, grads_hybrid))

    # 1. Compare Forward Output Tensors (Element-by-Element)
    max_out_diff = float(jnp.max(jnp.abs(out_pure - out_hybrid)))
    print(f"Forward Pass Max Output Diff: {max_out_diff:.2e}")

    # 2. Compare Loss
    diff_loss = float(jnp.abs(loss_pure - loss_hybrid))
    print(f"Loss Scalar Diff:             {diff_loss:.2e}")

    # 3. Compare Gradients (Element-by-Element)
    flat_grads_pure, _ = jax.tree_util.tree_flatten(grads_pure)
    flat_grads_hybrid, _ = jax.tree_util.tree_flatten(grads_hybrid)

    max_grad_diff = 0.0
    for g1, g2 in zip(flat_grads_pure, flat_grads_hybrid):
        if hasattr(g1, "shape"):
            d = jnp.max(jnp.abs(g1 - g2))
            max_grad_diff = max(max_grad_diff, float(d))

    print(f"Backward Pass Max Grad Diff:  {max_grad_diff:.2e}")

    TOLERANCE = 1e-2 if DTYPE == jnp.bfloat16 else 1e-5
    if max_out_diff > TOLERANCE or max_grad_diff > TOLERANCE:
        print("❌ WARNING: Significant divergence detected between Pure JAX and Hybrid GDN!")
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
    jit_fwd_pure, _ = create_jitted_forward(pure_jax_model)
    jit_fwd_hybrid, _ = create_jitted_forward(hybrid_model)

    t_fwd_pure = benchmark_func("Pure JAX Forward", jit_fwd_pure, params_pure, inputs)
    t_fwd_hybrid = benchmark_func("Hybrid GDN Forward", jit_fwd_hybrid, params_hybrid, inputs)

    t_train_pure = benchmark_func("Pure JAX Train Step", jit_train_pure, params_pure, inputs)
    t_train_hybrid = benchmark_func("Hybrid GDN Train Step", jit_train_hybrid, params_hybrid, inputs)

    print(f"\n--- Results Summary ---")
    fwd_speedup = t_fwd_pure / t_fwd_hybrid if t_fwd_hybrid > 0 else 0.0
    train_speedup = t_train_pure / t_train_hybrid if t_train_hybrid > 0 else 0.0
    print(f"Forward Pass Speedup:       {fwd_speedup:.2f}x  ({t_fwd_pure:.2f}ms -> {t_fwd_hybrid:.2f}ms)")
    print(f"Training Step Speedup:      {train_speedup:.2f}x  ({t_train_pure:.2f}ms -> {t_train_hybrid:.2f}ms)")

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

            print(f"  {mem_analysis}")
            if hasattr(mem_analysis, "temp_size_in_bytes"):
                return mem_analysis.temp_size_in_bytes
            return 0
        except Exception as e:
            print(f"  Memory analysis failed: {e}")
            return 0

    mem_pure = analyze_memory("Pure JAX Train Step", jit_train_pure, params_pure, inputs)
    mem_hybrid = analyze_memory("Hybrid GDN Train Step", jit_train_hybrid, params_hybrid, inputs)

    if mem_pure > 0 and mem_hybrid > 0:
        reduction = (mem_pure - mem_hybrid) / mem_pure * 100
        print(f"\nMemory Reduction: {reduction:.2f}% (Higher is better)")

    # ==============================================================================
    # PART D: PROFILING
    # ==============================================================================
    print(f"\n--- Profiling Hybrid GDN Implementation ---")
    print(f"Saving trace to: {profile_dir}")

    try:
        jax.profiler.start_trace(profile_dir)
        for _ in range(WARMUP):
            out = jit_train_hybrid(params_hybrid, inputs)
            jax.block_until_ready(out)
        jax.profiler.stop_trace()
        print("Profiling complete.")
    except Exception as e:
        print(f"Profiling failed (possibly already active): {e}")

    print(f"Profile trace directory: {profile_dir}")

    # ==============================================================================
    # PART E: STABILITY STRESS TEST
    # ==============================================================================
    print("\n--- Stability Stress Test (Hybrid GDN) ---")

    def create_jitted_update_step(model: nnx.Module, learning_rate: float = 1e-4):
        graphdef, params = nnx.split(model)

        @jax.jit
        def train_update(params, x):
            m = nnx.merge(graphdef, params)

            def loss_fn(m_inner):
                out = m_inner(x)
                y = out[0] if isinstance(out, tuple) else out
                return jnp.mean(jnp.square(y))

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            new_params = jax.tree_util.tree_map(
                lambda p, g: p - learning_rate * g if g is not None else p,
                params, grads,
            )
            return loss, grads, new_params

        return train_update, params

    jit_update_hybrid, current_params = create_jitted_update_step(hybrid_model)

    TEST_STEPS = 15
    print(f"Running {TEST_STEPS} simulated training steps with FRESH inputs...")

    for step in range(1, TEST_STEPS + 1):
        step_key = jax.random.fold_in(key, step * 100)
        step_input = jax.random.normal(step_key, (BATCH, SEQ_LEN, HIDDEN_SIZE), dtype=DTYPE)

        loss_val, grads_val, current_params = jit_update_hybrid(current_params, step_input)
        jax.block_until_ready(loss_val)

        # 1. Check Loss
        if jnp.isnan(loss_val) or jnp.isinf(loss_val):
            print(f"\n❌ CRITICAL FAIL at Step {step}: Loss is {loss_val}!")
            return

        # 2. Check Gradients
        grad_any_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc or jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)),
            grads_val,
            False,
        )
        if grad_any_nan:
            print(f"\n❌ CRITICAL FAIL at Step {step}: Gradients contain NaN or Inf!")
            return

        # 3. Check Parameters
        param_any_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc or jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)),
            current_params,
            False,
        )
        if param_any_nan:
            print(f"\n❌ CRITICAL FAIL at Step {step}: Parameters contain NaN or Inf after update!")
            return

        print(f"  Step {step}: Loss = {float(loss_val):.6f} | Stability: OK")

    print(f"✅ Stability Stress Test Passed: No NaNs encountered in {TEST_STEPS} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Hybrid GDN vs Pure JAX GDN in MaxText")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length")
    parser.add_argument("--iters", type=int, default=None, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warmup iterations")
    parser.add_argument("--dtype", type=str, default=None, help="Compute dtype (bfloat16, float32, etc.)")
    parser.add_argument("--profile_dir", type=str, default="/tmp/maxtext_gdn_profile", help="Directory for trace output")
    args = parser.parse_args()

    run_comparison(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        iters=args.iters,
        warmup=args.warmup,
        dtype_str=args.dtype,
        profile_dir=args.profile_dir,
    )