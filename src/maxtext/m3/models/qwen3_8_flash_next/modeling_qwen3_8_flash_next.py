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
  * Qwen Sparse Attention (QSA), indexer, and partial interleaved mRoPE
  * Sparse MoE block (routed experts, top-k router, gated shared expert)
"""

from collections.abc import Callable
from collections.abc import Sequence
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


# =================================================================================================
# Qwen Sparse Attention (QSA), indexer, and partial interleaved mRoPE
# =================================================================================================


# pylint: disable=no-name-in-module,too-many-positional-arguments


def rotate_half(x: jax.Array) -> jax.Array:
  """Rotates half the hidden dimensions of the input."""
  return jnp.concatenate((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), axis=-1)


def apply_rotary_pos_emb(
    q: jax.Array,
    k: jax.Array | None = None,
    cos: jax.Array | None = None,
    sin: jax.Array | None = None,
    unsqueeze_dim: int = 1,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Applies Rotary Position Embedding to query and optionally key tensors."""
  if cos is None or sin is None:
    raise ValueError("cos and sin must be provided to apply_rotary_pos_emb.")
  cos, sin = jnp.expand_dims(cos, axis=unsqueeze_dim), jnp.expand_dims(sin, axis=unsqueeze_dim)
  d = cos.shape[-1]

  def _rot(x: jax.Array) -> jax.Array:
    return jnp.concatenate([(x[..., :d] * cos) + (rotate_half(x[..., :d]) * sin), x[..., d:]], axis=-1)

  return _rot(q) if k is None else (_rot(q), _rot(k))


class Qwen3_8RotaryEmbedding(nnx.Module):
  """Partial interleaved mRoPE for Qwen3.8-Flash-Next."""

  def __init__(
      self,
      head_dim: int,
      partial_rotary_factor: float = 0.25,
      rope_theta: float = 10000000.0,
      mrope_section: Sequence[int] = (11, 11, 10),
  ):
    self.mrope_section = tuple(mrope_section)
    dim = int(head_dim * partial_rotary_factor)
    if len(self.mrope_section) != 3 or sum(self.mrope_section) != dim // 2:
      raise ValueError(
          f"Expected 3 mrope_section values summing to {dim // 2} (rotary dim {dim} / 2), got {self.mrope_section}."
      )
    self.rotary_dim = dim
    self.rope_theta = float(rope_theta)

  @property
  def inv_freq(self) -> jax.Array:
    """Returns inverse frequencies for rotary position embedding."""
    return 1.0 / (self.rope_theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim))

  def apply_interleaved_mrope(self, freqs: jax.Array, mrope_section: Sequence[int]) -> jax.Array:
    """Interleaves multimodal frequency components."""
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
      idx = slice(offset, mrope_section[dim] * 3, 3)
      freqs_t = freqs_t.at[..., idx].set(freqs[dim, ..., idx])
    return freqs_t

  def __call__(self, x: jax.Array, position_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Generates cos and sin embeddings for input positions."""
    if position_ids.ndim == 2:
      position_ids = jnp.broadcast_to(position_ids[None, ...], (3, *position_ids.shape))
    freqs = position_ids[:, :, :, None].astype(jnp.float32) * self.inv_freq[None, None, None, :]
    freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(x.dtype), jnp.sin(emb).astype(x.dtype)


class Qwen3_8QSAIndexer(nnx.Module):
  """MQA micro-block indexer selecting top-k complete blocks for QSA."""

  def __init__(
      self,
      emb_dim: int = 2560,
      index_n_heads: int = 4,
      index_kv_heads: int = 1,
      index_head_dim: int = 128,
      token_budget: int = 2048,
      compress_ratio: int = 4,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    if index_kv_heads != 1:
      raise ValueError(f"Expected index_kv_heads == 1 (MQA indexer keys), got {index_kv_heads}.")
    if compress_ratio <= 0 or token_budget < compress_ratio:
      raise ValueError(
          "Expected compress_ratio > 0 and token_budget >= compress_ratio,"
          f" got compress_ratio={compress_ratio}, token_budget={token_budget}."
      )
    self.index_head_dim, self.index_n_heads = index_head_dim, index_n_heads
    self.compress_ratio, self.block_topk = compress_ratio, token_budget // compress_ratio

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.index_qk_proj = nnx.Linear(emb_dim, (index_n_heads + index_kv_heads) * index_head_dim, **kw)
    self.q_layernorm = Qwen3_8RMSNorm(index_head_dim, eps=eps, rngs=rngs)
    self.k_layernorm = Qwen3_8RMSNorm(index_head_dim, eps=eps, rngs=rngs)

  def __call__(
      self,
      hidden_states: jax.Array,
      position_embeddings: tuple[jax.Array, jax.Array],
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """Computes sparse token mask (B, 1, S, S) and document-local block indices (B, S, topk).

    Args:
      hidden_states: Input tensor of shape (batch, seq_len, emb_dim).
      position_embeddings: Tuple of (cos, sin) arrays for RoPE.
      attention_mask: Optional (B, S) mask; 0/False = padding, non-zero/True = active (NOT a PyTorch additive mask).
      segment_ids: Optional (B, S) 1-indexed document IDs (0 = padding); when both are given, padding from
        attention_mask is folded into segment_ids.

    Returns:
      Tuple of (token_mask, selected_block_indices).
    """
    if attention_mask is not None and attention_mask.ndim != 2:
      raise ValueError(f"Expected 2D (B, S) attention_mask or None, got shape {attention_mask.shape}.")

    b, s, _ = hidden_states.shape
    c = self.compress_ratio
    full_cos, full_sin = position_embeddings
    curr_cos, curr_sin = full_cos[:, -s:, :], full_sin[:, -s:, :]

    qk = self.index_qk_proj(hidden_states)
    q_dim = self.index_n_heads * self.index_head_dim
    q = self.q_layernorm(qk[..., :q_dim].reshape(b, s, self.index_n_heads, self.index_head_dim))
    q_rot = apply_rotary_pos_emb(q, cos=curr_cos, sin=curr_sin, unsqueeze_dim=2)
    assert isinstance(q_rot, jax.Array)
    raw_keys = qk[..., q_dim:].reshape(b, s, self.index_head_dim)

    if segment_ids is None:
      segment_ids = jnp.ones((b, s), dtype=jnp.int32)
    if attention_mask is not None:
      segment_ids = jnp.where(attention_mask != 0, segment_ids, 0)

    same_doc = (segment_ids[:, :, None] == segment_ids[:, None, :]) & (segment_ids[:, :, None] != 0)
    doc_start = jnp.argmax(same_doc, axis=-1)
    num_blocks = max(1, s // c)
    local_pos, end_t = jnp.arange(s)[None, :] - doc_start, jnp.clip(jnp.arange(s) + c - 1, 0, s - 1)
    is_block_start = (
        (segment_ids != 0)
        & ((local_pos % c) == 0)
        & ((jnp.arange(s) + c - 1 < s)[None, :])
        & (segment_ids[:, end_t] == segment_ids)
    )

    group_starts = jax.vmap(lambda m: jnp.nonzero(m, size=num_blocks, fill_value=0)[0])(is_block_start)
    block_valid = jnp.arange(num_blocks)[None, :] < jnp.sum(is_block_start.astype(jnp.int32), axis=-1, keepdims=True)
    block_seg_ids = jnp.take_along_axis(segment_ids, group_starts, axis=1)

    block_token_idx = jnp.clip(group_starts[:, :, None] + jnp.arange(c)[None, None, :], 0, s - 1)
    block_keys_raw = jax.vmap(lambda k, idx: k[idx])(raw_keys, block_token_idx)
    pooled_keys = self.k_layernorm(jnp.mean(block_keys_raw.astype(jnp.float32), axis=2).astype(raw_keys.dtype))

    cos_blk = jax.vmap(lambda cb, idx: cb[idx])(full_cos, group_starts)
    sin_blk = jax.vmap(lambda sb, idx: sb[idx])(full_sin, group_starts)
    res = apply_rotary_pos_emb(pooled_keys[:, :, None, :], cos=cos_blk, sin=sin_blk, unsqueeze_dim=2)
    assert isinstance(res, jax.Array)
    block_keys = res.squeeze(2)

    raw_scores = jnp.einsum("bthd,bmd->bthm", q_rot.astype(jnp.float32), block_keys.astype(jnp.float32))
    scores = jnp.sum(jax.nn.relu(raw_scores), axis=2) / jnp.sqrt(float(self.index_head_dim))

    valid_q_block = (
        block_valid[:, None, :]
        & (block_seg_ids[:, None, :] == segment_ids[:, :, None])
        & ((group_starts + c - 1)[:, None, :] <= jnp.arange(s)[None, :, None])
        & (segment_ids[:, :, None] != 0)
    )

    k_top = min(self.block_topk, num_blocks)
    top_vals, top_indices = jax.lax.top_k(jnp.where(valid_q_block, scores, -1e9), k_top)
    doc_local_idx = (jnp.take_along_axis(group_starts[:, None, :], top_indices, axis=2) - doc_start[:, :, None]) // c
    selected_block_indices = jnp.where(top_vals > -1e8, doc_local_idx, -1).astype(jnp.int32)
    if self.block_topk > k_top:
      selected_block_indices = jnp.pad(
          selected_block_indices, ((0, 0), (0, 0), (0, self.block_topk - k_top)), constant_values=-1
      )

    m_local = jnp.arange(num_blocks)[None, None, :]
    in_topk = jnp.any(
        (selected_block_indices[:, :, :, None] == m_local) & (selected_block_indices[:, :, :, None] >= 0), axis=2
    )
    ar_k, ar_q = jnp.arange(s)[None, None, :], jnp.arange(s)[None, :, None]
    is_selected = jnp.take_along_axis(in_topk, jnp.clip((ar_k - doc_start[:, :, None]) // c, 0, num_blocks - 1), axis=-1)
    in_tail = ar_k >= (doc_start + ((jnp.arange(s)[None, :] - doc_start + 1) // c) * c)[:, :, None]
    token_mask = (same_doc & (ar_k <= ar_q) & (is_selected | in_tail))[:, None, :, :]
    return token_mask, selected_block_indices


class Qwen3_8SparseAttention(nnx.Module):
  """Qwen Sparse Attention (QSA) with RMSNorm, gated output, and micro-block indexer."""

  def __init__(
      self,
      emb_dim: int = 2560,
      num_query_heads: int = 24,
      num_kv_heads: int = 2,
      head_dim: int = 256,
      partial_rotary_factor: float = 0.25,
      rope_theta: float = 10000000.0,
      mrope_section: Sequence[int] = (11, 11, 10),
      index_n_heads: int = 4,
      index_kv_heads: int = 1,
      index_head_dim: int = 128,
      token_budget: int = 2048,
      compress_ratio: int = 4,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    del sharding
    if num_query_heads % num_kv_heads != 0:
      raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads}).")
    self.head_dim, self.num_query_heads, self.num_kv_heads = head_dim, num_query_heads, num_kv_heads

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.q_proj = nnx.Linear(emb_dim, num_query_heads * head_dim * 2, **kw)
    self.k_proj = nnx.Linear(emb_dim, num_kv_heads * head_dim, **kw)
    self.v_proj = nnx.Linear(emb_dim, num_kv_heads * head_dim, **kw)
    self.o_proj = nnx.Linear(num_query_heads * head_dim, emb_dim, **kw)

    self.q_norm = Qwen3_8RMSNorm(head_dim, eps=eps, rngs=rngs)
    self.k_norm = Qwen3_8RMSNorm(head_dim, eps=eps, rngs=rngs)

    self.rotary_emb = Qwen3_8RotaryEmbedding(head_dim, partial_rotary_factor, rope_theta, mrope_section)
    self.indexer = Qwen3_8QSAIndexer(
        emb_dim,
        index_n_heads,
        index_kv_heads,
        index_head_dim,
        token_budget,
        compress_ratio,
        eps,
        dtype,
        weight_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      position_ids: jax.Array | None = None,
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      return_selected_blocks: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Forward pass computing sparse attention with document boundary support.

    Args:
      hidden_states: Input tensor of shape (batch, seq_len, emb_dim).
      position_ids: Optional (B, S) or (3, B, S) (T/H/W rows for interleaved mRoPE).
      attention_mask: Optional (B, S) mask; 0/False = padding, non-zero/True = active (NOT a PyTorch additive mask).
      segment_ids: Optional (B, S) 1-indexed document IDs (0 = padding); when both are given, padding from
        attention_mask is folded into segment_ids.
      return_selected_blocks: Whether to return selected block indices.

    Returns:
      Attention output tensor, or tuple of (output, selected_block_indices) if return_selected_blocks is True.
    """
    b, s, _ = hidden_states.shape
    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))

    cos, sin = self.rotary_emb(hidden_states, position_ids)
    selected_token_mask, selected_block_indices = self.indexer(
        hidden_states, (cos, sin), attention_mask=attention_mask, segment_ids=segment_ids
    )

    q_and_gate = self.q_proj(hidden_states).reshape(b, s, self.num_query_heads, 2 * self.head_dim)
    query_states = self.q_norm(q_and_gate[..., : self.head_dim])
    gate = q_and_gate[..., self.head_dim :].reshape(b, s, self.num_query_heads * self.head_dim)
    key_states = self.k_norm(self.k_proj(hidden_states).reshape(b, s, self.num_kv_heads, self.head_dim))
    value_states = self.v_proj(hidden_states).reshape(b, s, self.num_kv_heads, self.head_dim)

    res = apply_rotary_pos_emb(query_states, key_states, cos=cos, sin=sin, unsqueeze_dim=2)
    assert isinstance(res, tuple)
    query_states, key_states = res

    if (rep := self.num_query_heads // self.num_kv_heads) > 1:
      key_states, value_states = jnp.repeat(key_states, rep, axis=2), jnp.repeat(value_states, rep, axis=2)

    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query_states, key_states) * (1.0 / jnp.sqrt(float(self.head_dim)))
    min_val = jnp.finfo(jnp.float32).min
    attn_weights = jnp.where(selected_token_mask, attn_weights, min_val)

    probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(value_states.dtype)
    attn_output = jnp.einsum("bhqk,bkhd->bqhd", probs, value_states).reshape(b, s, self.num_query_heads * self.head_dim)
    attn_output = (attn_output * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(attn_output.dtype)

    out = self.o_proj(attn_output)
    return (out, selected_block_indices) if return_selected_blocks else out


# =================================================================================================
# Sparse MoE block (routed experts, top-k router, gated shared expert)
# =================================================================================================


class Qwen3_8MLP(nnx.Module):
  """SwiGLU MLP used as shared expert in Qwen3.8-Flash-Next MoE."""

  def __init__(
      self,
      emb_dim: int = 2560,
      intermediate_dim: int = 640,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.gate_proj = nnx.Linear(emb_dim, intermediate_dim, **kw)
    self.up_proj = nnx.Linear(emb_dim, intermediate_dim, **kw)
    self.down_proj = nnx.Linear(intermediate_dim, emb_dim, **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass computing down_proj(silu(gate_proj(x)) * up_proj(x))."""
    return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_8Experts(nnx.Module):
  """Collection of routed expert weights stored as 3D tensors.

  Weight Layout Contract (for CL 9 checkpoint conversion):
    - gate_up_proj: (num_experts, emb_dim, 2 * moe_mlp_dim) -> (E, H, 2I).
      HuggingFace stores gate_up_proj as (E, 2I, H); transpose last two axes.
    - down_proj: (num_experts, moe_mlp_dim, emb_dim) -> (E, I, H).
      HuggingFace stores down_proj as (E, H, I); transpose last two axes.
  """

  def __init__(
      self,
      num_experts: int = 512,
      emb_dim: int = 2560,
      moe_mlp_dim: int = 640,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.emb_dim = emb_dim
    self.moe_mlp_dim = moe_mlp_dim
    self.dtype = dtype

    self.gate_up_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (num_experts, emb_dim, 2 * moe_mlp_dim), weight_dtype)
    )
    self.down_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (num_experts, moe_mlp_dim, emb_dim), weight_dtype)
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      top_k_index: jax.Array,
      top_k_weights: jax.Array,
  ) -> jax.Array:
    """Dispatches tokens via sorted token-expert grouping and jax.lax.ragged_dot."""
    orig_shape = hidden_states.shape
    if hidden_states.ndim == 3:
      hidden_states = hidden_states.reshape(-1, orig_shape[-1])
      top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])
      top_k_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])

    num_tokens, top_k = top_k_index.shape
    flat_expert_ids = top_k_index.reshape(-1)
    perm = jnp.argsort(flat_expert_ids)
    group_sizes = jnp.bincount(flat_expert_ids, length=self.num_experts).astype(jnp.int32)
    flat_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
    sorted_x = hidden_states[flat_token_ids[perm]]

    w_gate_up = jnp.asarray(self.gate_up_proj.value, dtype=self.dtype)
    gate_up = jax.lax.ragged_dot(sorted_x.astype(self.dtype), w_gate_up, group_sizes)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    act = jax.nn.silu(gate) * up

    w_down = jnp.asarray(self.down_proj.value, dtype=self.dtype)
    sorted_out = jax.lax.ragged_dot(act, w_down, group_sizes)

    inv_perm = jnp.argsort(perm)
    expert_outs = sorted_out[inv_perm].reshape(num_tokens, top_k, self.emb_dim)
    out = jnp.sum(expert_outs * top_k_weights[:, :, None].astype(expert_outs.dtype), axis=1)
    out = out.astype(hidden_states.dtype)

    if len(orig_shape) == 3:
      out = out.reshape(orig_shape)
    return out

  def reference_forward(
      self,
      hidden_states: jax.Array,
      top_k_index: jax.Array,
      top_k_weights: jax.Array,
  ) -> jax.Array:
    """Per-token weight gather reference forward used in parity tests."""
    orig_shape = hidden_states.shape
    if hidden_states.ndim == 3:
      hidden_states = hidden_states.reshape(-1, orig_shape[-1])
      top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])
      top_k_weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])

    w_gate_up = jnp.asarray(self.gate_up_proj.value, dtype=self.dtype)[top_k_index]
    gate_up = jnp.einsum("nd,nkdf->nkf", hidden_states.astype(self.dtype), w_gate_up)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    act = jax.nn.silu(gate) * up
    w_down = jnp.asarray(self.down_proj.value, dtype=self.dtype)[top_k_index]
    expert_out = jnp.einsum("nkm,nkmd->nkd", act, w_down)
    out = jnp.einsum("nkd,nk->nd", expert_out, top_k_weights.astype(expert_out.dtype))
    out = out.astype(hidden_states.dtype)

    if len(orig_shape) == 3:
      out = out.reshape(orig_shape)
    return out


class Qwen3_8TopKRouter(nnx.Module):
  """Top-k router computing routing weights, expert indices, and full probabilities."""

  def __init__(
      self,
      emb_dim: int = 2560,
      num_experts: int = 512,
      num_experts_per_tok: int = 10,
      norm_topk_prob: bool = True,
      routed_scaling_factor: float = 1.0,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor
    self.dtype = dtype
    self.linear = nnx.Linear(
        emb_dim,
        num_experts,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes routing weights, selected experts, and full float32 router probabilities."""
    router_logits = self.linear(hidden_states)
    router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1)
    router_top_value, router_indices = jax.lax.top_k(router_probs, self.num_experts_per_tok)
    if self.norm_topk_prob:
      router_top_value = router_top_value / jnp.sum(router_top_value, axis=-1, keepdims=True)
    routing_weights = (router_top_value * self.routed_scaling_factor).astype(router_logits.dtype)
    return routing_weights, router_indices, router_probs


class Qwen3_8SparseMoeBlock(nnx.Module):
  """Sparse MoE block with routed experts and gated shared expert."""

  def __init__(
      self,
      emb_dim: int = 2560,
      moe_mlp_dim: int = 640,
      num_experts: int = 512,
      num_experts_per_tok: int = 10,
      norm_topk_prob: bool = True,
      routed_scaling_factor: float = 1.0,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      shared_expert_intermediate_size: int,
      rngs: nnx.Rngs,
  ):
    del sharding
    if shared_expert_intermediate_size <= 0:
      raise ValueError(f"shared_expert_intermediate_size must be > 0, got {shared_expert_intermediate_size}.")
    self.emb_dim = emb_dim
    self.moe_mlp_dim = moe_mlp_dim
    self.shared_expert_intermediate_size = shared_expert_intermediate_size
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor
    self.dtype = dtype

    self.gate = Qwen3_8TopKRouter(
        emb_dim=emb_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )
    self.experts = Qwen3_8Experts(
        num_experts=num_experts,
        emb_dim=emb_dim,
        moe_mlp_dim=moe_mlp_dim,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )
    self.shared_expert = Qwen3_8MLP(
        emb_dim=emb_dim,
        intermediate_dim=shared_expert_intermediate_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )
    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.shared_expert_gate = nnx.Linear(emb_dim, 1, **kw)

  def __call__(
      self,
      hidden_states: jax.Array,
      return_router_topk: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward pass through routed experts and gated shared expert."""
    b, s, d = hidden_states.shape
    flat_states = hidden_states.reshape(-1, d)

    routing_weights, router_indices, router_probs = self.gate(flat_states)
    routed_out = self.experts(flat_states, router_indices, routing_weights)
    shared_out = self.shared_expert(flat_states)
    shared_gate = jax.nn.sigmoid(self.shared_expert_gate(flat_states).astype(jnp.float32)).astype(shared_out.dtype)

    combined = (routed_out + shared_gate * shared_out).reshape(b, s, d)
    if return_router_topk:
      return (
          combined,
          routing_weights.reshape(b, s, self.num_experts_per_tok),
          router_indices.reshape(b, s, self.num_experts_per_tok),
          router_probs.reshape(b, s, self.num_experts),
      )
    return combined
