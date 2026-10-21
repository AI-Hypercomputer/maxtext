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

"""Qwen3.8-Flash-Next model in m3 format.

Self-contained single-file implementation following the m3 design
(go/maxtext-m3-design): the model owns every component of its architecture,
composed from Flax NNX primitives, with no shared decoder or attention class.

Sections (in dependency order):
  * Normalization layers and 4-stream gated residual (hyper-connection)
  * Gated DeltaNet recurrence (chunked and recurrent delta rules)
  * Gated DeltaNet (GDN) linear attention layer
"""

from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


# =================================================================================================
# Normalization layers and 4-stream gated residual (hyper-connection)
# =================================================================================================


class Qwen3_8RMSNorm(nnx.Module):
  """RMSNorm with zero-centered scale (1.0 + scale) and optional group size."""

  def __init__(
      self,
      dim: int,
      group_size: int | None = None,
      eps: float = 1e-6,
      *,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.dim, self.group_size, self.eps = dim, group_size, eps
    if group_size is not None and dim % group_size != 0:
      raise ValueError(f"dim ({dim}) not divisible by group_size ({group_size}).")
    self.scale = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))

  def _norm(self, x: jax.Array) -> jax.Array:
    if self.group_size is not None:
      shape = x.shape
      x = x.reshape(*shape[:-1], -1, self.group_size)
      var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
      return (x * jax.lax.rsqrt(var + self.eps)).reshape(shape)
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)

  def __call__(self, x: jax.Array) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    scale_f32 = 1.0 + self.scale[...].astype(jnp.float32)
    return (self._norm(x_f32) * scale_f32).astype(x.dtype)


class Qwen3_8RMSNormGated(nnx.Module):
  """Gated RMSNorm with standard scale (init ones) and configurable gate activation."""

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      activation: str = "silu",
      *,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.dim, self.eps = dim, eps
    self.activation = activation
    if activation == "silu":
      self._act_fn = jax.nn.silu
    elif activation == "sigmoid":
      self._act_fn = jax.nn.sigmoid
    else:
      raise ValueError(f"Unsupported activation: {activation}. Expected 'silu' or 'sigmoid'.")
    self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

  def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)
    hidden = self.scale[...] * x_norm.astype(orig_dtype)
    gate_act = self._act_fn(gate.astype(jnp.float32))
    return (hidden * gate_act).astype(orig_dtype)


def l2norm(x: jax.Array, dim: int = -1, eps: float = 1e-6) -> jax.Array:
  """L2 normalization along a dimension aligning with FLA library."""
  x_f32 = x.astype(jnp.float32)
  return (x_f32 * jax.lax.rsqrt(jnp.sum(jnp.square(x_f32), axis=dim, keepdims=True) + eps)).astype(x.dtype)


def init_hyper_connection_streams(token_embeddings: jax.Array, hc_count: int = 4) -> jax.Array:
  """Initializes residual stream by replicating token embeddings across streams."""
  return jnp.concatenate([token_embeddings] * hc_count, axis=-1)


class Qwen3_8GatedResidual(nnx.Module):
  """4-stream Gated Residual (Hyper-Connection) module."""

  def __init__(
      self,
      emb_dim: int,
      hc_count: int = 4,
      hc_lowrank: int = 320,
      *,
      use_combine: bool = True,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      rngs: nnx.Rngs,
  ):
    self.emb_dim, self.hc_count = emb_dim, hc_count
    self.hc_lowrank, self.use_combine = hc_lowrank, use_combine
    self.dtype, self.weight_dtype = dtype, weight_dtype

    hc_dim = hc_count * emb_dim
    self.hc_norm = Qwen3_8RMSNorm(hc_dim, group_size=emb_dim, eps=eps, rngs=rngs)
    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.input_mix_weight_down = nnx.Linear(hc_dim, hc_lowrank, **kw)
    self.input_mix_weight_up = nnx.Linear(hc_lowrank, hc_dim, **kw)
    self.block_inject_weight = nnx.Linear(hc_dim, hc_count, **kw) if use_combine else None

  def _mix(self, normed: jax.Array, out_dtype: Any) -> jax.Array:
    w_down = jax.nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
    w_up = jax.nn.sigmoid(self.input_mix_weight_up(w_down))
    shape = w_up.shape[:-1]
    w_up_streams = w_up.reshape(*shape, self.hc_count, self.emb_dim)
    normed_streams = normed.reshape(*shape, self.hc_count, self.emb_dim)
    return jnp.mean(w_up_streams * normed_streams, axis=-2).astype(out_dtype)

  def mix_input(self, hyper_input: jax.Array) -> jax.Array:
    """Mixes multiple residual streams into a single representation."""
    if hyper_input.shape[-1] != self.hc_count * self.emb_dim:
      raise ValueError(f"Expected {self.hc_count * self.emb_dim} features, got {hyper_input.shape[-1]}.")
    return self._mix(self.hc_norm(hyper_input), hyper_input.dtype)

  def before_block(self, hyper_input: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Pre-block transformation computing mixed input and injection weights."""
    if hyper_input.shape[-1] != self.hc_count * self.emb_dim:
      raise ValueError(f"Expected {self.hc_count * self.emb_dim} features, got {hyper_input.shape[-1]}.")
    if self.block_inject_weight is None:
      raise ValueError("block_inject_weight is None (use_combine=False).")
    normed = self.hc_norm(hyper_input)
    inject = (2.0 * jax.nn.sigmoid(self.block_inject_weight(normed) / self.hc_count)).astype(hyper_input.dtype)
    return self._mix(normed, hyper_input.dtype), inject

  def combine(
      self,
      hyper_input: jax.Array,
      block_output: jax.Array,
      injection_weights: jax.Array,
  ) -> jax.Array:
    """Injects block output into hyper-connection residual streams."""
    inj = jnp.expand_dims(block_output, axis=-2) * jnp.expand_dims(injection_weights, axis=-1)
    return (hyper_input + inj.reshape(*hyper_input.shape)).astype(hyper_input.dtype)

  def apply_block(
      self,
      hyper_input: jax.Array,
      block_fn: Callable[[jax.Array], jax.Array],
  ) -> jax.Array:
    """Executes before_block, applies block_fn, and combines.

    Args:
      hyper_input: Multi-stream residual representations with shape `(..., hc_count * emb_dim)`.
      block_fn: Callable mapping the mixed pre-block representation `(..., emb_dim)` to the
        single-stream block output `(..., emb_dim)`.

    Returns:
      The combined output `combine(hyper_input, block_fn(mixed), injection_weights)` with shape
      `(..., hc_count * emb_dim)`.
    """
    mixed, inject_gate = self.before_block(hyper_input)
    return self.combine(hyper_input, block_fn(mixed), inject_gate)


# =================================================================================================
# Gated DeltaNet recurrence (chunked and recurrent delta rules)
# =================================================================================================


# pylint: disable=no-name-in-module,too-many-positional-arguments


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


# =================================================================================================
# Gated DeltaNet (GDN) linear attention layer
# =================================================================================================


# pylint: disable=no-name-in-module,too-many-positional-arguments


class Qwen3_8GatedDeltaNet(nnx.Module):
  """Gated DeltaNet (GDN) linear attention layer for Qwen3.8-Flash-Next."""

  def __init__(
      self,
      emb_dim: int = 2560,
      num_k_heads: int = 16,
      num_v_heads: int = 48,
      head_k_dim: int = 128,
      head_v_dim: int = 128,
      conv_kernel_size: int = 4,
      chunk_size: int = 64,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      output_gate_activation: str,
      rngs: nnx.Rngs,
  ):
    del sharding
    self.emb_dim, self.chunk_size = emb_dim, chunk_size
    self.num_k_heads, self.num_v_heads = num_k_heads, num_v_heads
    if num_v_heads % num_k_heads != 0:
      raise ValueError(f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads}).")
    self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim
    self.output_gate_activation = output_gate_activation
    self.dtype = dtype

    self.key_dim = num_k_heads * head_k_dim
    self.value_dim = num_v_heads * head_v_dim
    conv_dim = self.key_dim * 2 + self.value_dim

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.in_proj_qkv = nnx.Linear(emb_dim, conv_dim, **kw)
    self.in_proj_z = nnx.Linear(emb_dim, self.value_dim, **kw)
    self.in_proj_b = nnx.Linear(emb_dim, num_v_heads, **kw)
    self.in_proj_a = nnx.Linear(emb_dim, num_v_heads, **kw)

    self.conv1d = nnx.Conv(
        in_features=conv_dim,
        out_features=conv_dim,
        kernel_size=(conv_kernel_size,),
        feature_group_count=conv_dim,
        padding="CAUSAL",
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )
    u = jax.random.uniform(rngs.params(), (num_v_heads,), minval=0.01, maxval=16.0)
    self.A_log = nnx.Param(jnp.log(u.astype(jnp.float32)))  # pylint: disable=invalid-name
    self.dt_bias = nnx.Param(jnp.ones((num_v_heads,), dtype=jnp.float32))
    self.norm = Qwen3_8RMSNormGated(head_v_dim, eps=eps, activation=output_gate_activation, rngs=rngs)
    self.out_proj = nnx.Linear(self.value_dim, emb_dim, **kw)

  def __call__(
      self,
      hidden_states: jax.Array,
      attention_mask: jax.Array | None = None,
      initial_state: jax.Array | None = None,
      output_final_state: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    if attention_mask is not None:
      hidden_states = (hidden_states * jnp.expand_dims(attention_mask, -1)).astype(hidden_states.dtype)

    b, s, _ = hidden_states.shape
    mixed_qkv = jax.nn.silu(self.conv1d(self.in_proj_qkv(hidden_states)))
    z = self.in_proj_z(hidden_states).reshape(b, s, self.num_v_heads, self.head_v_dim)

    beta = jax.nn.sigmoid(self.in_proj_b(hidden_states))
    g = -jnp.exp(self.A_log[...].astype(jnp.float32)) * jax.nn.softplus(
        self.in_proj_a(hidden_states).astype(jnp.float32) + self.dt_bias[...].astype(jnp.float32)
    )

    kd = self.key_dim
    query = mixed_qkv[..., :kd].reshape(b, s, self.num_k_heads, self.head_k_dim)
    key = mixed_qkv[..., kd : kd * 2].reshape(b, s, self.num_k_heads, self.head_k_dim)
    value = mixed_qkv[..., kd * 2 :].reshape(b, s, self.num_v_heads, self.head_v_dim)

    if attention_mask is not None:
      mask = attention_mask != 0
      key = jnp.where(mask[:, :, None, None], key, 0.0)
      value = jnp.where(mask[:, :, None, None], value, 0.0)
      g = jnp.where(mask[:, :, None], g, 0.0)

    if self.num_v_heads // self.num_k_heads > 1:
      rep = self.num_v_heads // self.num_k_heads
      query, key = jnp.repeat(query, rep, axis=2), jnp.repeat(key, rep, axis=2)

    if s == 1 and initial_state is not None:
      core_attn_out, last_state = jax_recurrent_gated_delta_rule(
          query,
          key,
          value,
          g=g,
          beta=beta,
          initial_state=initial_state,
          output_final_state=output_final_state,
          use_qk_l2norm_in_kernel=True,
      )
    else:
      core_attn_out, last_state = jax_chunk_gated_delta_rule(
          query,
          key,
          value,
          g=g,
          beta=beta,
          chunk_size=self.chunk_size,
          initial_state=initial_state,
          output_final_state=output_final_state,
          use_qk_l2norm_in_kernel=True,
      )
    core_normed = self.norm(core_attn_out, z).reshape(b, s, self.value_dim)
    out = self.out_proj(core_normed)
    return (out, last_state) if output_final_state else out
