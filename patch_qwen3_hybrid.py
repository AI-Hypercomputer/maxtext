import re

qwen3_path = '/usr/local/google/home/rbierneni/maxtext/src/maxtext/models/qwen3.py'
with open(qwen3_path, 'r') as f:
    qwen3_code = f.read()

qwen3_code = qwen3_code.replace(
    'core_attn_out, (next_conv_state, next_recurrent_state) = hybrid_fused_conv1d_gdn(',
    'core_attn_out, (next_conv_state, next_recurrent_state), pure_jax_tap = hybrid_fused_conv1d_gdn('
)

qwen3_code = qwen3_code.replace(
    'return hidden_states, next_recurrent_state, next_conv_state, attn_weights',
    'return hidden_states, next_recurrent_state, next_conv_state, pure_jax_tap'
)
qwen3_code = qwen3_code.replace(
    'return hidden_states, None, None, None',
    'return hidden_states, None, None, pure_jax_tap'
)

with open(qwen3_path, 'w') as f:
    f.write(qwen3_code)

bench_path = '/usr/local/google/home/rbierneni/maxtext/src/maxtext/scratch_code/benchmark_gdn_optimizations.py'
with open(bench_path, 'r') as f:
    bench_code = f.read()

bench_code = bench_code.replace(
    'out = m_inner(x)\n            y = out[0] if isinstance(out, tuple) else out\n            loss = jnp.mean(y * projection.astype(y.dtype))\n            return loss, y',
    'out = m_inner(x)\n            y = out[0] if isinstance(out, tuple) else out\n            loss = jnp.mean(y * projection.astype(y.dtype))\n            # Qwen3NextGatedDeltaNet outputs: hidden_states, recurrent, conv, tap. It is a tuple if it has layers. If it is the full model, we need to extract from layers.\n            return loss, out'
)

bench_code = bench_code.replace(
    'out = m(x)\n        y = out[0] if isinstance(out, tuple) else out\n        return y',
    'out = m(x)\n        return out'
)

bench_code = bench_code.replace(
    'def pure_forward(params, x):',
    'def pure_forward(params, x):\n        jax.debug.print("Compiling pure_forward...")'
)

bench_code = bench_code.replace(
    'def pure_train_step(params, x):',
    'def pure_train_step(params, x):\n        jax.debug.print("Compiling pure_train_step...")'
)

bench_code = bench_code.replace(
    'loss_hybrid, out_hybrid, grads_hybrid = jit_train_hybrid(params_hybrid, inputs)\n    jax.block_until_ready((loss_hybrid, out_hybrid, grads_hybrid))',
    'loss_hybrid, out_hybrid, grads_hybrid = jit_train_hybrid(params_hybrid, inputs)\n    jax.block_until_ready((loss_hybrid, out_hybrid, grads_hybrid))\n\n    print("Extracting Tap Outputs...")\n    tap_pure = out_pure[-1] if isinstance(out_pure, tuple) else None\n    tap_hybrid = out_hybrid[-1] if isinstance(out_hybrid, tuple) else None\n\n    if tap_pure is not None and tap_hybrid is not None:\n        # The model returns a tuple of outputs for each layer. For a single layer model like Qwen3NextGatedDeltaNet layer directly:\n        if isinstance(tap_pure, tuple):\n             tap_pure = tap_pure[-1]\n             tap_hybrid = tap_hybrid[-1]\n        \n        max_tap_diff = float(jnp.max(jnp.abs(tap_pure - tap_hybrid)))\n        print(f"Tap Variable Max Diff (e.g. t_inv or A): {max_tap_diff:.2e}")\n    else:\n        print("Tap variables not found in output.")\n'
)

with open(bench_path, 'w') as f:
    f.write(bench_code)
