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

"""Hybrid Gated Delta Net (GDN) implementations for MaxText using Tokamax GDN v3 forward + Pallas Custom VJP backward."""

import functools
from typing import Any, Optional, Tuple

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _pallas_gdn_bwd_kernel(
    padded_pre_conv_qkv_ref,
    qkv_ref,
    b_ref,
    a_ref,
    a_log_ref,
    dt_bias_ref,
    do_ref,
    chunk_states_ref,
    conv_weight_ref,
    seq_lens_ref,
    d_qkv_ref,
    d_b_ref,
    d_a_ref,
    d_conv_weight_ref,
    d_a_log_ref,
    d_dt_bias_ref,
    padded_pre_conv_qkv_vmem,
    qkv_vmem,
    b_vmem,
    a_vmem,
    a_log_vmem,
    dt_bias_vmem,
    do_vmem,
    chunk_states_vmem,
    d_qkv_vmem,
    d_b_vmem,
    d_a_vmem,
    d_conv_weight_scratch,
    d_a_log_scratch,
    d_dt_bias_scratch,
    sem_pre_conv_qkv,
    sem_qkv,
    sem_b,
    sem_a,
    sem_a_log,
    sem_dt_bias,
    sem_do,
    sem_chunk_states,
    sem_d_qkv,
    sem_d_b,
    sem_d_a,
    sem_d_conv_weight,
    sem_d_a_log,
    sem_d_dt_bias,
    *,
    batch_size: int,
    num_chunks: int,
    chunk_size: int,
    dim_size: int,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    kernel_size: int,
    pad_len: int = 8,
    use_qk_norm_in_gdn: bool = False,
):
  seq_idx = pl.program_id(0)

  d_conv_weight_scratch[...] = jnp.zeros_like(d_conv_weight_scratch)

  num_kq_heads = (dim_size - num_v_heads * v_head_dim) // (kq_head_dim * 2)
  q_size = num_kq_heads * kq_head_dim
  k_size = num_kq_heads * kq_head_dim
  v_size = num_v_heads * v_head_dim
  repeats = num_v_heads // num_kq_heads

  def chunk_forward(q, k, v, b_val, a_val, a_log_val, dt_bias_val, state_prev):
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    if use_qk_norm_in_gdn:
      q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
      k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    scale = 1.0 / jnp.sqrt(kq_head_dim)
    q = q * scale
    b_val = b_val.astype(jnp.float32)
    a_val = a_val.astype(jnp.float32)
    a_log_val = a_log_val.astype(jnp.float32)
    dt_bias_val = dt_bias_val.astype(jnp.float32)
    state_prev = state_prev.astype(jnp.float32)
    q_rep = jnp.repeat(q, repeats, axis=1)
    k_rep = jnp.repeat(k, repeats, axis=1)

    beta = jax.nn.sigmoid(b_val)

    # EXACT GDN v3 gating formula
    log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)
    v_beta = v * beta[:, :, None]

    # Fast MXU cumsum replacement
    mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
    cumsum_log_g = jnp.dot(mask_cumsum, log_g)

    diff = cumsum_log_g[:, None, :] - cumsum_log_g[None, :, :]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=diff.dtype))
    safe_diff = jnp.where(mask[:, :, None] == 1.0, diff, -1e4)
    G = jnp.exp(safe_diff) * mask[:, :, None]
    G = jnp.transpose(G, (2, 0, 1))

    q_h = jnp.transpose(q_rep, (1, 0, 2))
    k_h = jnp.transpose(k_rep, (1, 0, 2))
    v_h = jnp.transpose(v_beta, (1, 0, 2))

    # Dual Causal Attention
    attn = jnp.einsum("hck,hdk->hcd", q_h, k_h)
    attn_causal = attn * G

    out_intra = jnp.einsum("hcd,hdv->hcv", attn_causal, v_h)
    out_intra = jnp.transpose(out_intra, (1, 0, 2))

    cross_decay = jnp.exp(cumsum_log_g)
    q_scaled = q_rep * cross_decay[:, :, None]
    out_cross = jnp.einsum("chk,hkv->chv", q_scaled, state_prev)

    out = out_intra + out_cross

    state_decay_end = G[:, chunk_size - 1, :]
    state_prev_decayed = state_prev * cross_decay[-1, :, None, None]

    k_scaled = k_h * state_decay_end[:, :, None]
    state_new_intra = jnp.einsum("hck,hcv->hkv", k_scaled, v_h)

    state_new = state_prev_decayed + state_new_intra

    return out, state_new

  def fetch_inputs(chunk_idx, slot):
    start_idx = chunk_idx * chunk_size
    copy_len = chunk_size + pad_len
    pltpu.make_async_copy(
        padded_pre_conv_qkv_ref.at[seq_idx, pl.ds(start_idx, copy_len)],
        padded_pre_conv_qkv_vmem.at[slot],
        sem_pre_conv_qkv.at[slot],
    ).start()
    pltpu.make_async_copy(qkv_ref.at[seq_idx, chunk_idx], qkv_vmem.at[slot], sem_qkv.at[slot]).start()
    pltpu.make_async_copy(b_ref.at[seq_idx, chunk_idx], b_vmem.at[slot], sem_b.at[slot]).start()
    pltpu.make_async_copy(a_ref.at[seq_idx, chunk_idx], a_vmem.at[slot], sem_a.at[slot]).start()
    pltpu.make_async_copy(do_ref.at[seq_idx, chunk_idx], do_vmem.at[slot], sem_do.at[slot]).start()
    pltpu.make_async_copy(
        chunk_states_ref.at[seq_idx, chunk_idx], chunk_states_vmem.at[slot], sem_chunk_states.at[slot]
    ).start()

  def wait_inputs(slot):
    pltpu.make_async_copy(
        padded_pre_conv_qkv_vmem.at[slot], padded_pre_conv_qkv_vmem.at[slot], sem_pre_conv_qkv.at[slot]
    ).wait()
    pltpu.make_async_copy(qkv_vmem.at[slot], qkv_vmem.at[slot], sem_qkv.at[slot]).wait()
    pltpu.make_async_copy(b_vmem.at[slot], b_vmem.at[slot], sem_b.at[slot]).wait()
    pltpu.make_async_copy(a_vmem.at[slot], a_vmem.at[slot], sem_a.at[slot]).wait()
    pltpu.make_async_copy(do_vmem.at[slot], do_vmem.at[slot], sem_do.at[slot]).wait()
    pltpu.make_async_copy(chunk_states_vmem.at[slot], chunk_states_vmem.at[slot], sem_chunk_states.at[slot]).wait()

  def store_outputs(chunk_idx, slot):
    pltpu.make_async_copy(d_qkv_vmem.at[slot], d_qkv_ref.at[seq_idx, chunk_idx], sem_d_qkv.at[slot]).start()
    pltpu.make_async_copy(d_b_vmem.at[slot], d_b_ref.at[seq_idx, chunk_idx], sem_d_b.at[slot]).start()
    pltpu.make_async_copy(d_a_vmem.at[slot], d_a_ref.at[seq_idx, chunk_idx], sem_d_a.at[slot]).start()

  def wait_outputs(slot):
    pltpu.make_async_copy(d_qkv_vmem.at[slot], d_qkv_vmem.at[slot], sem_d_qkv.at[slot]).wait()
    pltpu.make_async_copy(d_b_vmem.at[slot], d_b_vmem.at[slot], sem_d_b.at[slot]).wait()
    pltpu.make_async_copy(d_a_vmem.at[slot], d_a_vmem.at[slot], sem_d_a.at[slot]).wait()

  # Prologue
  fetch_inputs(num_chunks - 1, 0)

  # Fetch static inputs (a_log and dt_bias)
  pltpu.make_async_copy(a_log_ref.at[seq_idx], a_log_vmem, sem_a_log.at[0]).start()
  pltpu.make_async_copy(dt_bias_ref.at[seq_idx], dt_bias_vmem, sem_dt_bias.at[0]).start()
  pltpu.make_async_copy(a_log_vmem, a_log_vmem, sem_a_log.at[0]).wait()
  pltpu.make_async_copy(dt_bias_vmem, dt_bias_vmem, sem_dt_bias.at[0]).wait()

  d_state = jnp.zeros((num_v_heads, kq_head_dim, v_head_dim), dtype=jnp.float32)
  d_conv_weight_acc = jnp.zeros((kernel_size, dim_size), dtype=jnp.float32)
  d_a_log_acc = jnp.zeros((num_v_heads,), dtype=jnp.float32)
  d_dt_bias_acc = jnp.zeros((num_v_heads,), dtype=jnp.float32)

  def loop_body(i, carry):
    d_state, d_conv_weight_acc, d_a_log_acc, d_dt_bias_acc = carry

    chunk_idx = num_chunks - 1 - i
    slot = i % 2
    next_slot = (i + 1) % 2

    wait_inputs(slot)

    def fetch_next():
      fetch_inputs(chunk_idx - 1, next_slot)

    jax.lax.cond(i < num_chunks - 1, fetch_next, lambda: None)

    def wait_prev_out():
      wait_outputs(slot)

    jax.lax.cond(i > 1, wait_prev_out, lambda: None)

    padded_pre_conv_qkv_val = padded_pre_conv_qkv_vmem[slot, ...]
    qkv_val = qkv_vmem[slot, ...]
    b_val = b_vmem[slot, ...]
    a_val = a_vmem[slot, ...]
    do_val = do_vmem[slot, ...]
    state_prev_val = chunk_states_vmem[slot, ...]
    a_log_val = a_log_vmem[...]
    dt_bias_val = dt_bias_vmem[...]

    q = qkv_val[:, :q_size].reshape((chunk_size, num_kq_heads, kq_head_dim))
    k = qkv_val[:, q_size : q_size + k_size].reshape((chunk_size, num_kq_heads, kq_head_dim))
    v = qkv_val[:, q_size + k_size :].reshape((chunk_size, num_v_heads, v_head_dim))

    _, vjp_fn = jax.vjp(chunk_forward, q, k, v, b_val, a_val, a_log_val, dt_bias_val, state_prev_val)
    d_q, d_k, d_v, d_b_val, d_a_val, d_a_log_val, d_dt_bias_val, d_state_prev = vjp_fn(
        (do_val.astype(jnp.float32), d_state)
    )

    d_qkv = jnp.concatenate(
        [
            d_q.reshape(chunk_size, q_size),
            d_k.reshape(chunk_size, k_size),
            d_v.reshape(chunk_size, v_size),
        ],
        axis=-1,
    )

    d_cw_rows = []
    for k_idx in range(kernel_size):
      start_slice = pad_len - k_idx
      shifted = padded_pre_conv_qkv_val[start_slice : start_slice + chunk_size]
      d_cw_rows.append(jnp.sum(d_qkv * shifted.astype(jnp.float32), axis=0))
    d_cw = jnp.stack(d_cw_rows, axis=0)

    d_conv_weight_acc += d_cw.astype(jnp.float32)
    d_a_log_acc += d_a_log_val.astype(jnp.float32)
    d_dt_bias_acc += d_dt_bias_val.astype(jnp.float32)
    d_state = d_state_prev.astype(jnp.float32)

    d_qkv_vmem[slot, ...] = d_qkv.astype(d_qkv_vmem.dtype)
    d_b_vmem[slot, ...] = d_b_val.astype(d_b_vmem.dtype)
    d_a_vmem[slot, ...] = d_a_val.astype(d_a_vmem.dtype)

    store_outputs(chunk_idx, slot)

    return d_state, d_conv_weight_acc, d_a_log_acc, d_dt_bias_acc

  d_state, d_conv_weight_acc, d_a_log_acc, d_dt_bias_acc = jax.lax.fori_loop(
      0, num_chunks, loop_body, (d_state, d_conv_weight_acc, d_a_log_acc, d_dt_bias_acc)
  )

  def wait_last_out():
    wait_outputs((num_chunks - 1) % 2)

  jax.lax.cond(num_chunks > 0, wait_last_out, lambda: None)

  def wait_prev_last_out():
    wait_outputs((num_chunks - 2) % 2)

  jax.lax.cond(num_chunks > 1, wait_prev_last_out, lambda: None)

  d_conv_weight_scratch[...] = d_conv_weight_acc.astype(d_conv_weight_scratch.dtype)
  d_a_log_scratch[...] = d_a_log_acc.astype(d_a_log_scratch.dtype)
  d_dt_bias_scratch[...] = d_dt_bias_acc.astype(d_dt_bias_scratch.dtype)

  pltpu.make_async_copy(d_conv_weight_scratch, d_conv_weight_ref.at[seq_idx, ...], sem_d_conv_weight.at[0]).start()
  pltpu.make_async_copy(d_a_log_scratch, d_a_log_ref.at[seq_idx, ...], sem_d_a_log.at[0]).start()
  pltpu.make_async_copy(d_dt_bias_scratch, d_dt_bias_ref.at[seq_idx, ...], sem_d_dt_bias.at[0]).start()

  pltpu.make_async_copy(d_conv_weight_scratch, d_conv_weight_scratch, sem_d_conv_weight.at[0]).wait()
  pltpu.make_async_copy(d_a_log_scratch, d_a_log_scratch, sem_d_a_log.at[0]).wait()
  pltpu.make_async_copy(d_dt_bias_scratch, d_dt_bias_scratch, sem_d_dt_bias.at[0]).wait()


def pallas_fused_conv1d_gdn_bwd_computation(
    pre_conv_qkv: jax.Array,
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    chunk_states: jax.Array,
    conv_weight: jax.Array,
    seq_lens: Optional[jax.Array] = None,
    *,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    kernel_size: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Executes the Pallas reverse-chunk GDNv3 backward kernel."""
  batch_size, seq_len, dim_size = pre_conv_qkv.shape
  num_chunks = seq_len // chunk_size

  pre_conv_qkv_4d = pre_conv_qkv.reshape(batch_size, num_chunks, chunk_size, dim_size)
  qkv_4d = qkv.reshape(batch_size, num_chunks, chunk_size, dim_size)
  b_4d = b.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  a_4d = a.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  do_4d = do.reshape(batch_size, num_chunks, chunk_size, num_v_heads, v_head_dim)

  if a_log.ndim == 1:
    a_log_2d = jnp.broadcast_to(a_log[None, :], (batch_size, num_v_heads))
  else:
    a_log_2d = a_log
  if dt_bias.ndim == 1:
    dt_bias_2d = jnp.broadcast_to(dt_bias[None, :], (batch_size, num_v_heads))
  else:
    dt_bias_2d = dt_bias

  if conv_weight.ndim == 3:
    conv_weight_2d = conv_weight.squeeze(1)
  else:
    conv_weight_2d = conv_weight

  if seq_lens is None:
    seq_lens = jnp.full((batch_size,), seq_len, dtype=jnp.int32)

  pad_len = ((kernel_size - 1 + 7) // 8) * 8
  pre_conv_pad = jnp.zeros((batch_size, pad_len, dim_size), dtype=pre_conv_qkv.dtype)
  padded_pre_conv_qkv = jnp.concatenate([pre_conv_pad, pre_conv_qkv], axis=1)

  grid = (batch_size,)
  hbm_spec = pl.BlockSpec(memory_space=pl.ANY)

  d_qkv_shape = jax.ShapeDtypeStruct(qkv_4d.shape, qkv_4d.dtype)
  d_b_shape = jax.ShapeDtypeStruct(b_4d.shape, b_4d.dtype)
  d_a_shape = jax.ShapeDtypeStruct(a_4d.shape, a_4d.dtype)
  d_conv_weight_shape = jax.ShapeDtypeStruct((batch_size, kernel_size, dim_size), conv_weight_2d.dtype)
  d_a_log_shape = jax.ShapeDtypeStruct((batch_size, num_v_heads), a_log_2d.dtype)
  d_dt_bias_shape = jax.ShapeDtypeStruct((batch_size, num_v_heads), dt_bias_2d.dtype)

  out_shapes = (d_qkv_shape, d_b_shape, d_a_shape, d_conv_weight_shape, d_a_log_shape, d_dt_bias_shape)

  scratch_shapes = (
      pltpu.VMEM((2, chunk_size + pad_len, dim_size), padded_pre_conv_qkv.dtype),
      pltpu.VMEM((2, chunk_size, dim_size), qkv_4d.dtype),
      pltpu.VMEM((2, chunk_size, num_v_heads), b_4d.dtype),
      pltpu.VMEM((2, chunk_size, num_v_heads), a_4d.dtype),
      pltpu.VMEM((num_v_heads,), a_log_2d.dtype),
      pltpu.VMEM((num_v_heads,), dt_bias_2d.dtype),
      pltpu.VMEM((2, chunk_size, num_v_heads, v_head_dim), do_4d.dtype),
      pltpu.VMEM((2, num_v_heads, kq_head_dim, v_head_dim), chunk_states.dtype),
      pltpu.VMEM((2, chunk_size, dim_size), d_qkv_shape.dtype),
      pltpu.VMEM((2, chunk_size, num_v_heads), d_b_shape.dtype),
      pltpu.VMEM((2, chunk_size, num_v_heads), d_a_shape.dtype),
      pltpu.VMEM((kernel_size, dim_size), d_conv_weight_shape.dtype),
      pltpu.VMEM((num_v_heads,), d_a_log_shape.dtype),
      pltpu.VMEM((num_v_heads,), d_dt_bias_shape.dtype),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((1,)),
      pltpu.SemaphoreType.DMA((1,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((1,)),
      pltpu.SemaphoreType.DMA((1,)),
      pltpu.SemaphoreType.DMA((1,)),
  )

  d_qkv, d_b, d_a, d_conv_weight, d_a_log, d_dt_bias = pl.pallas_call(
      functools.partial(
          _pallas_gdn_bwd_kernel,
          batch_size=batch_size,
          num_chunks=num_chunks,
          chunk_size=chunk_size,
          dim_size=dim_size,
          num_v_heads=num_v_heads,
          kq_head_dim=kq_head_dim,
          v_head_dim=v_head_dim,
          kernel_size=kernel_size,
          pad_len=pad_len,
          use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      ),
      out_shape=out_shapes,
      grid=grid,
      in_specs=[hbm_spec] * 10,
      out_specs=[hbm_spec] * 6,
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          disable_bounds_checks=True,
      ),
  )(padded_pre_conv_qkv, qkv_4d, b_4d, a_4d, a_log_2d, dt_bias_2d, do_4d, chunk_states, conv_weight_2d, seq_lens)

  d_conv_weight_reduced = jnp.sum(d_conv_weight, axis=0)
  d_a_log_reduced = jnp.sum(d_a_log, axis=0)
  d_dt_bias_reduced = jnp.sum(d_dt_bias, axis=0)

  d_qkv_flat = d_qkv.reshape(batch_size, seq_len, dim_size)
  d_b_flat = d_b.reshape(batch_size, seq_len, num_v_heads)
  d_a_flat = d_a.reshape(batch_size, seq_len, num_v_heads)

  if conv_weight.ndim == 3:
    d_conv_weight_out = d_conv_weight_reduced[:, None, :]
  else:
    d_conv_weight_out = d_conv_weight_reduced

  return d_qkv_flat, d_b_flat, d_a_flat, d_conv_weight_out, d_a_log_reduced, d_dt_bias_reduced


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
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
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

  core_attn_out, next_recurrent_state = jax_chunk_gated_delta_rule(
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

  next_conv_state = (
      qkv[:, -(conv_kernel_size - 1) :, :]
      if seq_len >= conv_kernel_size - 1
      else jnp.zeros((batch, conv_kernel_size - 1, qkv.shape[-1]), dtype=qkv.dtype)
  )
  if next_recurrent_state is None:
    next_recurrent_state = jnp.zeros((batch, num_v_heads, head_k_dim, head_v_dim), dtype=compute_dtype)

  return core_attn_out.astype(qkv.dtype), (next_conv_state.astype(qkv.dtype), next_recurrent_state.astype(qkv.dtype))


def _compute_forward_conv_and_states(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype,
) -> Tuple[jax.Array, jax.Array]:
  """Computes convolved QKV and inter-chunk states during forward pass."""
  batch_size, seq_len, dim_size = qkv.shape
  num_chunks = seq_len // chunk_size

  # Conv1D
  conv_input = jnp.pad(qkv, ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))
  conv_out = jax.lax.conv_general_dilated(
      lhs=conv_input,
      rhs=conv_weight.astype(qkv.dtype),
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NWC", "WIO", "NWC"),
      feature_group_count=dim_size,
  )
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(qkv.dtype)
  conv_out = conv_out[:, -seq_len:, :]
  qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(compute_dtype)

  # Chunk states progression
  num_kq_heads = num_k_heads
  q_size = num_kq_heads * head_k_dim
  k_size = num_kq_heads * head_k_dim
  repeats = num_v_heads // num_kq_heads

  q = qkv_conv[:, :, :q_size].reshape(batch_size, num_chunks, chunk_size, num_kq_heads, head_k_dim)
  k = qkv_conv[:, :, q_size : q_size + k_size].reshape(batch_size, num_chunks, chunk_size, num_kq_heads, head_k_dim)
  v = qkv_conv[:, :, q_size + k_size :].reshape(batch_size, num_chunks, chunk_size, num_v_heads, head_v_dim)

  if use_qk_norm_in_gdn:
    from maxtext.layers.normalizations import l2norm

    q = l2norm(q, dim=-1, eps=1e-6)
    k = l2norm(k, dim=-1, eps=1e-6)

  scale = jax.lax.rsqrt(jnp.array(head_k_dim, dtype=jnp.float32)).astype(compute_dtype)
  q = q * scale

  b_4d = b.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  a_4d = a.reshape(batch_size, num_chunks, chunk_size, num_v_heads)

  if recurrent_state is None:
    init_state = jnp.zeros((batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
  else:
    init_state = recurrent_state.astype(jnp.float32)

  def scan_fn(carry_state, chunk_inputs):
    q_i, k_i, v_i, b_i, a_i = chunk_inputs
    q_rep = jnp.repeat(q_i, repeats, axis=2)
    k_rep = jnp.repeat(k_i, repeats, axis=2)

    beta = jax.nn.sigmoid(b_i)
    log_g = -jnp.exp(a_log) * jax.nn.softplus(a_i + dt_bias)
    v_beta = v_i * beta[:, :, :, None]

    mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
    cumsum_log_g = jnp.einsum("cd,bhd->bhc", mask_cumsum, log_g.swapaxes(1, 2)).swapaxes(1, 2)

    diff = cumsum_log_g[:, :, None, :] - cumsum_log_g[:, None, :, :]
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=diff.dtype))
    safe_diff = jnp.where(mask[None, :, :, None] == 1.0, diff, -1e4)
    G = jnp.exp(safe_diff) * mask[None, :, :, None]

    cross_decay = jnp.exp(cumsum_log_g)
    state_decay_end = G[:, chunk_size - 1, :, :]
    state_prev_decayed = carry_state * cross_decay[:, -1, :, None, None]

    k_scaled = k_rep * state_decay_end[:, :, :, None]
    state_new_intra = jnp.einsum("bchk,bchv->bhkv", k_scaled, v_beta)

    new_state = state_prev_decayed + state_new_intra
    return new_state, carry_state

  q_chunks = q.swapaxes(0, 1)
  k_chunks = k.swapaxes(0, 1)
  v_chunks = v.swapaxes(0, 1)
  b_chunks = b_4d.swapaxes(0, 1)
  a_chunks = a_4d.swapaxes(0, 1)

  _, chunk_states = jax.lax.scan(scan_fn, init_state, (q_chunks, k_chunks, v_chunks, b_chunks, a_chunks))
  chunk_states = chunk_states.swapaxes(0, 1)

  return qkv_conv, chunk_states


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
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
    )

  from tokamax._src.ops.experimental.causal_conv1d_gated_delta_rule import wrapper as tokamax_gdn_wrapper

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

  (new_conv_state, new_recurrent_state), core_attn_out_flat = tokamax_gdn_wrapper.fused_conv1d_gdn(
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
  return core_attn_out.astype(qkv.dtype), (
      new_conv_state[1:].astype(qkv.dtype),
      new_recurrent_state[1:].astype(qkv.dtype),
  )


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
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
  """Hybrid Fused Conv1D + GDN: Tokamax GDN v3 forward + Pallas Custom VJP backward."""
  return _run_tokamax_fused_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
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
  out, states = _run_tokamax_fused_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
  )
  qkv_conv, chunk_states = _compute_forward_conv_and_states(
      qkv=qkv,
      b=b,
      a=a,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      a_log=a_log,
      dt_bias=dt_bias,
      recurrent_state=recurrent_state,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
  )
  residuals = (
      qkv,
      qkv_conv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      chunk_states,
      conv_state,
      recurrent_state,
  )
  return (out, states), residuals


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
      pre_conv_qkv,
      qkv_conv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      chunk_states,
      conv_state,
      recurrent_state,
  ) = residuals
  d_out, d_states = cotangents
  d_conv_state, d_recurrent_state = d_states

  if jax.default_backend() == "tpu":
    d_qkv, d_b, d_a, d_conv_weight, d_a_log, d_dt_bias = pallas_fused_conv1d_gdn_bwd_computation(
        pre_conv_qkv=pre_conv_qkv,
        qkv=qkv_conv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=d_out,
        chunk_states=chunk_states,
        conv_weight=conv_weight,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
    )
    d_conv_bias = None if conv_bias is None else jnp.zeros_like(conv_bias)
    d_conv_state = None if conv_state is None else jnp.zeros_like(conv_state)
    d_recurrent_state = None if recurrent_state is None else jnp.zeros_like(recurrent_state)
    return (d_qkv, d_b, d_a, d_conv_weight, d_conv_bias, d_a_log, d_dt_bias, d_conv_state, d_recurrent_state)

  # Fallback to JAX Newton-Schulz VJP on non-TPU / CPU
  def target_fn(qkv_, b_, a_, cw_, cb_, al_, dt_, cs_, rs_):
    return pure_jax_fused_conv1d_gdn(
        qkv_,
        b_,
        a_,
        cw_,
        cb_,
        al_,
        dt_,
        cs_,
        rs_,
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
      pre_conv_qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
  )
  return vjp_fn((d_out, (d_conv_state, d_recurrent_state)))


hybrid_fused_conv1d_gdn.defvjp(_hybrid_fused_conv1d_gdn_fwd, _hybrid_fused_conv1d_gdn_bwd)
