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

"""Chunked and recurrent Gated DeltaNet attention kernels in JAX."""

import jax
import jax.numpy as jnp
# pylint: disable=no-name-in-module,too-many-positional-arguments
from maxtext.m3.models.qwen3_8_flash_next import hyper_connection

l2norm = hyper_connection.l2norm


def _init_state(shape: tuple[int, ...], state: jax.Array | None) -> jax.Array:
  """Returns zero-initialized or float32-cast initial state."""
  return jnp.zeros(shape, jnp.float32) if state is None else state.astype(jnp.float32)


def jax_chunk_gated_delta_rule(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = 64,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jax.Array, jax.Array | None]:
  """Chunked Gated DeltaNet forward pass in float32."""
  init_dtype = value.dtype
  if use_qk_l2norm_in_kernel:
    query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)

  query, key, value, beta, g = (jnp.swapaxes(x.astype(jnp.float32), 1, 2) for x in (query, key, value, beta, g))
  b, h, s, d_k = key.shape
  d_v = value.shape[-1]
  pad_size = (chunk_size - s % chunk_size) % chunk_size
  if pad_size > 0:
    query, key, value = (jnp.pad(x, ((0, 0), (0, 0), (0, pad_size), (0, 0))) for x in (query, key, value))
    beta, g = (jnp.pad(x, ((0, 0), (0, 0), (0, pad_size))) for x in (beta, g))

  total_s = s + pad_size
  num_chunks = total_s // chunk_size
  beta_c = jnp.expand_dims(beta.reshape(b, h, num_chunks, chunk_size), -1)
  query = (query * (1.0 / jnp.sqrt(d_k))).reshape(b, h, num_chunks, chunk_size, d_k)
  key = key.reshape(b, h, num_chunks, chunk_size, d_k)
  k_beta, v_beta = key * beta_c, value.reshape(b, h, num_chunks, chunk_size, d_v) * beta_c
  g = g.reshape(b, h, num_chunks, chunk_size)

  prec = jax.lax.Precision.HIGHEST
  g_cumsum = jnp.cumsum(g, axis=-1)
  decay_mask = jnp.tril(jnp.exp(jnp.tril(jnp.expand_dims(g_cumsum, -1) - jnp.expand_dims(g_cumsum, -2))))
  s_mat = (jnp.matmul(k_beta, jnp.swapaxes(key, -1, -2), precision=prec)) * jnp.tril(decay_mask, k=-1)
  i_plus_s = jnp.eye(chunk_size, dtype=jnp.float32) + s_mat
  rhs = jnp.concatenate([v_beta, k_beta * jnp.expand_dims(jnp.exp(g_cumsum), -1)], axis=-1)
  v_inv, k_cumdecay = jnp.split(
      jax.scipy.linalg.solve_triangular(i_plus_s, rhs, lower=True, unit_diagonal=True), [d_v], axis=-1
  )

  init_state = _init_state((b, h, d_k, d_v), initial_state)

  def _chunk_step(carry, xs):
    q_i, k_i, v_i, k_cd_i, dm_i, g_i = xs
    v_new = v_i - jnp.matmul(k_cd_i, carry, precision=prec)
    attn_inter = jnp.matmul(q_i * jnp.expand_dims(jnp.exp(g_i), -1), carry, precision=prec)
    core_out = attn_inter + jnp.matmul(
        (jnp.matmul(q_i, jnp.swapaxes(k_i, -1, -2), precision=prec)) * dm_i, v_new, precision=prec
    )
    decay_last = jnp.expand_dims(jnp.exp(g_i[..., -1:]), -1)
    k_decay = k_i * jnp.expand_dims(jnp.exp(g_i[..., -1:] - g_i), -1)
    return carry * decay_last + jnp.matmul(jnp.swapaxes(k_decay, -1, -2), v_new, precision=prec), core_out

  xs = tuple(jnp.moveaxis(x, 2, 0) for x in (query, key, v_inv, k_cumdecay, decay_mask, g_cumsum))
  final_state, core_chunks = jax.lax.scan(_chunk_step, init_state, xs)
  core_out = jnp.moveaxis(core_chunks, 0, 2).reshape(b, h, total_s, d_v)
  core_out = jnp.swapaxes(core_out[:, :, :s, :], 1, 2).astype(init_dtype)
  return core_out, (final_state if output_final_state else None)


def jax_recurrent_gated_delta_rule(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[jax.Array, jax.Array | None]:
  """Recurrent Gated DeltaNet forward pass in float32."""
  prec = jax.lax.Precision.HIGHEST
  init_dtype = value.dtype
  if use_qk_l2norm_in_kernel:
    query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)

  b, _, h, d_k = key.shape
  d_v = value.shape[-1]
  query = jnp.swapaxes(query.astype(jnp.float32), 1, 2) * (1.0 / jnp.sqrt(d_k))
  key, value, beta, g = (jnp.swapaxes(x.astype(jnp.float32), 1, 2) for x in (key, value, beta, g))

  init_state = _init_state((b, h, d_k, d_v), initial_state)

  def _step(state, inputs):
    q_t, k_t, v_t, beta_t, g_t = inputs
    state = state * jnp.expand_dims(jnp.exp(g_t), (-1, -2))
    kv_mem = jnp.squeeze(jnp.matmul(jnp.expand_dims(k_t, -2), state, precision=prec), -2)
    delta = (v_t - kv_mem) * jnp.expand_dims(beta_t, -1)
    state = state + jnp.matmul(jnp.expand_dims(k_t, -1), jnp.expand_dims(delta, -2), precision=prec)
    return state, jnp.squeeze(jnp.matmul(jnp.expand_dims(q_t, -2), state, precision=prec), -2)

  xs = tuple(jnp.moveaxis(x, 2, 0) for x in (query, key, value, beta, g))
  final_state, outs = jax.lax.scan(_step, init_state, xs)
  core_out = jnp.swapaxes(jnp.moveaxis(outs, 0, 2), 1, 2).astype(init_dtype)
  return core_out, (final_state if output_final_state else None)
