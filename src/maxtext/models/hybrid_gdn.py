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

"""Hybrid Gated Delta Net (GDN) implementations for MaxText using Tokamax GDN v3 forward + Custom VJP backward."""

import functools
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp


def pure_jax_fused_conv1d_gdn(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array], jax.Array]:
    """Pure-JAX composite of Conv1D + GDN used during backward pass autodiff."""
    from maxtext.models.qwen3 import jax_chunk_gated_delta_rule
    batch, seq_len, _ = qkv.shape
    key_dim = num_k_heads * head_k_dim

    # --- Step B: Pure JAX 1D Convolution ---
    conv_input = jnp.pad(qkv, ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))
    conv_weight_cast = conv_weight.astype(qkv.dtype)
    conv_out = jax.lax.conv_general_dilated(
        lhs=conv_input,
        rhs=conv_weight_cast,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=qkv.shape[-1],
    )
    if conv_bias is not None:
        conv_out = conv_out + conv_bias.astype(qkv.dtype)
    conv_out = conv_out[:, -seq_len:, :]
    qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(compute_dtype)

    q_conv, k_conv, v_conv = jnp.split(qkv_conv, [key_dim, 2 * key_dim], axis=-1)

    # Reshape for GDN
    query = q_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
    key = k_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
    value = v_conv.reshape(batch, seq_len, num_v_heads, head_v_dim)

    A_log_cast = jnp.asarray(a_log, dtype=compute_dtype)
    dt_bias_cast = jnp.asarray(dt_bias, dtype=compute_dtype)
    beta = jax.nn.sigmoid(b)
    g = -jnp.exp(A_log_cast) * jax.nn.softplus(a + dt_bias_cast)

    if num_v_heads > num_k_heads and num_v_heads % num_k_heads == 0:
        repeats = num_v_heads // num_k_heads
        query = jnp.repeat(query, repeats, axis=2)
        key = jnp.repeat(key, repeats, axis=2)

    core_attn_out, next_recurrent_state, pure_jax_tap = jax_chunk_gated_delta_rule(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=recurrent_state,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
    )

    next_conv_state = qkv[:, -(conv_kernel_size - 1):, :] if seq_len >= conv_kernel_size - 1 else jnp.zeros((batch, conv_kernel_size - 1, qkv.shape[-1]), dtype=qkv.dtype)
    if next_recurrent_state is None:
        next_recurrent_state = jnp.zeros((batch, num_v_heads, head_k_dim, head_v_dim), dtype=compute_dtype)

    return core_attn_out.astype(qkv.dtype), (next_conv_state.astype(qkv.dtype), next_recurrent_state.astype(qkv.dtype)), pure_jax_tap


def _run_tokamax_fused_fwd(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
):
    if jax.default_backend() != "tpu":
        return pure_jax_fused_conv1d_gdn(
            qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state,
            num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size, chunk_size=chunk_size, use_qk_norm_in_gdn=use_qk_norm_in_gdn, compute_dtype=compute_dtype,
        )

    # When on TPU, invoke local MaxText GDN v3 fused_conv1d_gdn kernel
    from maxtext.kernels.causal_conv1d_gated_delta_rule import wrapper as tokamax_gdn_wrapper
    batch_size, seq_len, dim_size = qkv.shape
    num_seqs = batch_size

    qkv_flat = qkv.reshape(-1, dim_size)
    b_flat = b.reshape(-1, b.shape[-1])
    a_flat = a.reshape(-1, a.shape[-1])
    tokamax_conv_weight = jnp.swapaxes(conv_weight, 0, 2)

    query_start_loc = jnp.arange(0, (num_seqs + 1) * seq_len, seq_len, dtype=jnp.int32)
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
    seq_lens = jnp.full((num_seqs,), seq_len, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    if conv_state is None:
        tokamax_conv_state = jnp.zeros((num_seqs + 1, conv_kernel_size - 1, dim_size), dtype=qkv.dtype)
    elif conv_state.shape[0] == num_seqs:
        tokamax_conv_state = jnp.pad(conv_state, ((1, 0), (0, 0), (0, 0)))
    else:
        tokamax_conv_state = conv_state

    if recurrent_state is None:
        tokamax_recurrent_state = jnp.zeros((num_seqs + 1, num_v_heads, head_k_dim, head_v_dim), dtype=qkv.dtype)
    elif recurrent_state.shape[0] == num_seqs:
        tokamax_recurrent_state = jnp.pad(recurrent_state, ((1, 0), (0, 0), (0, 0), (0, 0)))
    else:
        tokamax_recurrent_state = recurrent_state

    (new_conv_state, new_recurrent_state), core_attn_out_flat, tap_out = tokamax_gdn_wrapper.fused_conv1d_gdn(
        qkv=qkv_flat,
        b=b_flat,
        a=a_flat,
        conv_state=tokamax_conv_state,
        recurrent_state=tokamax_recurrent_state,
        conv_weight=tokamax_conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        n_kq=num_k_heads,
        n_v=num_v_heads,
        d_k=head_k_dim,
        d_v=head_v_dim,
        kernel_size=conv_kernel_size,
        compute_precision=jnp.dtype(jnp.float32),
    )

    core_attn_out = core_attn_out_flat.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
    num_chunks = seq_len // chunk_size
    tap_out = tap_out.reshape(batch_size, -1, num_v_heads, chunk_size, chunk_size)[:, :num_chunks].astype(jnp.float32)
    return core_attn_out.astype(qkv.dtype), (new_conv_state[1:].astype(qkv.dtype), new_recurrent_state[1:].astype(qkv.dtype)), tap_out


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16))
def hybrid_fused_conv1d_gdn(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array], jax.Array]:
    """Hybrid Fused Conv1D + GDN: Tokamax GDN v3 forward + Custom VJP backward."""
    return _run_tokamax_fused_fwd(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state,
        num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size, chunk_size=chunk_size, use_qk_norm_in_gdn=use_qk_norm_in_gdn, compute_dtype=compute_dtype,
    )


def _hybrid_fused_conv1d_gdn_fwd(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
):
    out, states, tap_out = _run_tokamax_fused_fwd(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state,
        num_k_heads=num_k_heads, num_v_heads=num_v_heads, head_k_dim=head_k_dim, head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size, chunk_size=chunk_size, use_qk_norm_in_gdn=use_qk_norm_in_gdn, compute_dtype=compute_dtype,
    )
    residuals = (
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state
    )
    return (out, states, tap_out), residuals


def _hybrid_fused_conv1d_gdn_bwd(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    residuals: tuple,
    cotangents: tuple,
):
    (
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state
    ) = residuals

    def target_fn(qkv_, b_, a_, cw_, cb_, al_, dt_, cs_, rs_):
        return pure_jax_fused_conv1d_gdn(
            qkv_, b_, a_, cw_, cb_, al_, dt_, cs_, rs_,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            compute_dtype=compute_dtype,
        )

    _, vjp_fn = jax.vjp(
        target_fn,
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state,
    )
    d_out, d_states, d_tap = cotangents
    d_conv_state, d_recurrent_state = d_states
    return vjp_fn((d_out, (d_conv_state, d_recurrent_state), d_tap))


hybrid_fused_conv1d_gdn.defvjp(_hybrid_fused_conv1d_gdn_fwd, _hybrid_fused_conv1d_gdn_bwd)