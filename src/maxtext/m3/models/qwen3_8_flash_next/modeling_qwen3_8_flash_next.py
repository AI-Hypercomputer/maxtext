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
  * SplitMix64 n-gram hashing and per-layer embedding (PLE)
  * Decoder layer, scannable 4-layer cycle, and scan helpers
  * Multi-token prediction (MTP) draft block
  * Cross-entropy, router load-balancing, and total training loss
  * Decoder backbone, top-level model, and factory
"""

from collections.abc import Callable
from collections.abc import Sequence
import math
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


_BATCH_MESH_AXES = ("data", "stage", "fsdp", "fsdp_transpose", "expert")


def get_batch_mesh_axes() -> tuple[str, ...]:
  """Returns active batch mesh axis names when a multi-device mesh is set."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty or mesh.size <= 1:
    return ()
  return tuple(ax for ax in _BATCH_MESH_AXES if mesh.shape.get(ax, 1) > 1)


def shard_batch(x: jax.Array) -> jax.Array:
  """Constrains axis 0 of x to active batch mesh axes when a multi-device mesh is set."""
  if x.ndim == 0:
    return x
  batch_axes = get_batch_mesh_axes()
  if not batch_axes:
    return x
  spec = jax.sharding.PartitionSpec(batch_axes, *([None] * (x.ndim - 1)))
  return jax.lax.with_sharding_constraint(x, spec)


def init_hyper_connection_streams(token_embeddings: jax.Array, hc_count: int = 4) -> jax.Array:
  """Initializes residual stream by replicating token embeddings across streams."""
  return shard_batch(jnp.concatenate([token_embeddings] * hc_count, axis=-1))


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


# pylint: disable=no-name-in-module


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

    w_gate_up = jnp.asarray(self.gate_up_proj.value, dtype=self.dtype)
    w_down = jnp.asarray(self.down_proj.value, dtype=self.dtype)

    def _dispatch_2d(
        x_2d: jax.Array,
        idx_2d: jax.Array,
        w_2d: jax.Array,
        w_gu: jax.Array,
        w_dn: jax.Array,
    ) -> jax.Array:
      num_tokens, top_k = idx_2d.shape
      flat_expert_ids = idx_2d.reshape(-1)
      perm = jnp.argsort(flat_expert_ids)
      group_sizes = jnp.bincount(flat_expert_ids, length=self.num_experts).astype(jnp.int32)
      flat_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
      sorted_x = x_2d[flat_token_ids[perm]]

      gate_up = jax.lax.ragged_dot(sorted_x.astype(self.dtype), w_gu, group_sizes)
      gate, up = jnp.split(gate_up, 2, axis=-1)
      act = jax.nn.silu(gate) * up
      sorted_out = jax.lax.ragged_dot(act, w_dn, group_sizes)

      inv_perm = jnp.argsort(perm)
      expert_outs = sorted_out[inv_perm].reshape(num_tokens, top_k, self.emb_dim)
      out_2d = jnp.sum(expert_outs * w_2d[:, :, None].astype(expert_outs.dtype), axis=1)
      return out_2d.astype(x_2d.dtype)

    mesh = jax.sharding.get_abstract_mesh()
    batch_axes = get_batch_mesh_axes()
    batch_shards = math.prod(mesh.shape[ax] for ax in batch_axes) if batch_axes else 1
    if batch_axes and hidden_states.shape[0] % batch_shards == 0:
      token_spec = jax.sharding.PartitionSpec(batch_axes, None)
      weight_spec = jax.sharding.PartitionSpec()
      out = jax.shard_map(
          _dispatch_2d,
          mesh=mesh,
          in_specs=(
              token_spec,
              token_spec,
              token_spec,
              weight_spec,
              weight_spec,
          ),
          out_specs=token_spec,
          check_vma=False,
      )(hidden_states, top_k_index, top_k_weights, w_gate_up, w_down)
    else:
      out = _dispatch_2d(hidden_states, top_k_index, top_k_weights, w_gate_up, w_down)

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


# =================================================================================================
# SplitMix64 n-gram hashing and per-layer embedding (PLE)
# =================================================================================================


# pylint: disable=no-name-in-module,too-many-positional-arguments

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
  """Exact 64-bit integer SplitMix64 mixing."""
  value = (value + _SPLITMIX_GAMMA) & _MASK64
  value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
  value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
  return (value ^ (value >> 31)) & _MASK64


splitmix64 = _splitmix64


def _is_prime(value: int) -> bool:
  """Returns True if value is prime."""
  if value < 2 or value % 2 == 0:
    return value == 2
  return all(value % d != 0 for d in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
  """Finds the count-th prime strictly greater than start."""
  prime = start
  for _ in range(count):
    prime += 1
    while not _is_prime(prime):
      prime += 1
  return prime


def compute_ple_buffers(
    ngram_size: int = 3,
    heads_per_ngram: int = 8,
    ngram_vocab_size_base: int = 20_000_000,
    make_ngram_vocab_size_divisible_by: int = 128,
    vocab_size: int = 248320,
    ple_layer_index: int = 0,
    seed: int = 1234,
) -> tuple[list[int], list[int], list[int], int]:
  """Computes non-trainable PLE buffer arrays (multipliers, moduli, offsets, padded_vocab_size)."""
  half_bound = max(1, (((1 << 63) - 1) // max(vocab_size, 1)) // 2)
  base_seed = seed + _PRIME_1 * ple_layer_index
  multipliers = [
      2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1 for i in range(ngram_size)
  ]
  ngram_heads = (ngram_size - 1) * heads_per_ngram
  moduli, table_offsets, tot = [], [], 0
  for h in range(ngram_heads):
    sz = _find_nth_prime_after(ngram_vocab_size_base - 1, ple_layer_index * ngram_heads + h + 1)
    moduli.append(sz)
    table_offsets.append(tot)
    tot += sz
  padded_vocab_size = math.ceil(tot / make_ngram_vocab_size_divisible_by) * make_ngram_vocab_size_divisible_by
  return multipliers, moduli, table_offsets, padded_vocab_size


def _shift_right_ignore_eos(token_history: jax.Array, shift: int, eos_token_id: int) -> jax.Array:
  """Shifts tokens right by shift positions, resetting context on EOS."""
  if shift == 0:
    return token_history
  b, s = token_history.shape
  pos = jnp.arange(s, dtype=jnp.int32)
  eos_pos = jnp.where(token_history == eos_token_id, pos[None, :], -1)
  prev_eos = jnp.concatenate([jnp.full((b, 1), -1, dtype=jnp.int32), jax.lax.cummax(eos_pos, axis=1)[:, :-1]], axis=1)
  src = pos - shift
  shifted = jnp.take_along_axis(token_history, jnp.maximum(src, 0)[None, :], axis=1)
  valid = (pos[None, :] - (prev_eos + 1) >= shift) & (src[None, :] >= 0)
  return jnp.where(valid, shifted, jnp.full_like(shifted, eos_token_id))


def _mul_u64_bytes(tokens: jax.Array, mult_bytes: jax.Array) -> jax.Array:
  """Multiplies tokens by 8-byte multiplier in uint32 with carry propagation."""
  t, carry, out = tokens.astype(jnp.uint32), jnp.zeros_like(tokens, dtype=jnp.uint32), []
  for b in range(8):
    prod = t * mult_bytes[b] + carry
    out.append(prod & jnp.uint32(0xFF))
    carry = prod >> jnp.uint32(8)
  return jnp.stack(out, axis=-1)


class Qwen3_8NGramEmbedding(nnx.Module):
  """Hashed n-gram embedding table with SplitMix64 multipliers."""

  def __init__(
      self,
      eos_token_id: int,
      vocab_size: int = 248320,
      ple_embed_dim: int = 2560,
      ngram_size: int = 3,
      heads_per_ngram: int = 8,
      ngram_vocab_size_base: int = 20_000_000,
      make_ngram_vocab_size_divisible_by: int = 128,
      seed: int = 1234,
      ple_layer_index: int = 0,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.ple_embed_dim = ple_embed_dim
    self.ngram_size = ngram_size
    self.heads_per_ngram = heads_per_ngram
    self.ngram_heads = (ngram_size - 1) * heads_per_ngram
    self.eos_token_id = eos_token_id

    if ple_embed_dim % self.ngram_heads != 0:
      raise ValueError(f"ple_embed_dim ({ple_embed_dim}) not divisible by ngram_heads ({self.ngram_heads}).")
    head_dim_per_ngram = ple_embed_dim // self.ngram_heads

    (
        self.layer_multipliers,
        self.head_vocab_sizes,
        self.head_offsets,
        self.padded_vocab_size,
    ) = compute_ple_buffers(
        ngram_size,
        heads_per_ngram,
        ngram_vocab_size_base,
        make_ngram_vocab_size_divisible_by,
        vocab_size,
        ple_layer_index,
        seed,
    )

    if (
        max(self.head_vocab_sizes) > (1 << 28)
        or self.padded_vocab_size >= (1 << 31)
        or vocab_size > (1 << 24)
        or eos_token_id >= (1 << 24)
    ):
      raise ValueError(
          "PLE hashing uint32 bounds exceeded: head vocab must be <= 2**28, padded vocab < 2**31, "
          f"vocab_size/eos_token_id < 2**24; got max head {max(self.head_vocab_sizes)}, "
          f"padded {self.padded_vocab_size}, vocab_size {vocab_size}, eos_token_id {eos_token_id}."
      )

    self._mult_bytes_tuple = tuple(tuple((m >> (8 * b)) & 0xFF for b in range(8)) for m in self.layer_multipliers)

    self.ngram_embedding = nnx.Embed(
        self.padded_vocab_size, head_dim_per_ngram, dtype=dtype, param_dtype=weight_dtype, rngs=rngs
    )

  @property
  def head_vocab_sizes_arr(self) -> jax.Array:
    """Returns head vocabulary sizes as a uint32 JAX array."""
    return jnp.array(self.head_vocab_sizes, dtype=jnp.uint32)

  @property
  def head_offsets_arr(self) -> jax.Array:
    """Returns head vocabulary offsets as a uint32 JAX array."""
    return jnp.array(self.head_offsets, dtype=jnp.uint32)

  @property
  def mult_bytes(self) -> jax.Array:
    """Returns SplitMix64 multiplier bytes as a uint32 JAX array."""
    return jnp.array(self._mult_bytes_tuple, dtype=jnp.uint32)

  def compute_ngram_ids(self, input_ids: jax.Array) -> jax.Array:
    """Computes exact 64-bit n-gram hash IDs for each token position."""
    b, s = input_ids.shape
    mult_bytes = self.mult_bytes
    head_vocab_sizes_arr = self.head_vocab_sizes_arr
    head_offsets_arr = self.head_offsets_arr
    context = jnp.full((b, self.ngram_size - 1), self.eos_token_id, dtype=input_ids.dtype)
    history = jnp.concatenate([context, input_ids], axis=-1)
    shifted_tokens = [_shift_right_ignore_eos(history, sh, self.eos_token_id) for sh in range(self.ngram_size)]
    shifted_bytes = [_mul_u64_bytes(shifted_tokens[p], mult_bytes[p]) for p in range(self.ngram_size)]

    blocks = []
    for ngram in range(2, self.ngram_size + 1):
      s_idx = (ngram - 2) * self.heads_per_ngram
      e_idx = s_idx + self.heads_per_ngram
      mixed = shifted_bytes[0]
      for pos in range(1, ngram):
        mixed = jnp.bitwise_xor(mixed, shifted_bytes[pos])

      v = head_vocab_sizes_arr[s_idx:e_idx][None, None, :]
      rem = jnp.zeros((b, history.shape[1], v.shape[-1]), dtype=jnp.uint32)
      for b_idx in range(7, -1, -1):
        byte = mixed[:, :, b_idx : b_idx + 1]
        rem = (rem * jnp.uint32(16) + ((byte >> jnp.uint32(4)) & jnp.uint32(0xF))) % v
        rem = (rem * jnp.uint32(16) + (byte & jnp.uint32(0xF))) % v

      blocks.append((rem + head_offsets_arr[s_idx:e_idx][None, None, :]).astype(jnp.int32))

    return jnp.concatenate(blocks, axis=-1)[:, -s:, :]

  def __call__(self, input_ids: jax.Array) -> jax.Array:
    ngram_ids = self.compute_ngram_ids(input_ids)
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and mesh.size > 1:
      ngram_ids = jax.lax.with_sharding_constraint(ngram_ids, jax.sharding.PartitionSpec())
    embeds = shard_batch(self.ngram_embedding(ngram_ids))
    return embeds.reshape(input_ids.shape[0], -1, self.ple_embed_dim)


class Qwen3_8PLELayer(nnx.Module):
  """PLE layer injecting hashed n-gram features into hyper-connection streams."""

  def __init__(
      self,
      eos_token_id: int,
      emb_dim: int = 2560,
      hc_count: int = 4,
      vocab_size: int = 248320,
      ple_embed_dim: int = 2560,
      ngram_size: int = 3,
      heads_per_ngram: int = 8,
      ngram_vocab_size_base: int = 20_000_000,
      make_ngram_vocab_size_divisible_by: int = 128,
      ple_conv_kernel_size: int = 4,
      seed: int = 1234,
      ple_layer_index: int = 0,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    del sharding
    super().__init__()
    self.emb_dim, self.hc_count = emb_dim, hc_count
    self.hc_hidden_size = emb_dim * hc_count
    self.short_conv_state_len = (ple_conv_kernel_size - 1) * ngram_size
    self.gate_scale = 1.0 / math.sqrt(emb_dim)

    self.ple_embedding = Qwen3_8NGramEmbedding(
        eos_token_id,
        vocab_size,
        ple_embed_dim,
        ngram_size,
        heads_per_ngram,
        ngram_vocab_size_base,
        make_ngram_vocab_size_divisible_by,
        seed,
        ple_layer_index,
        dtype,
        weight_dtype,
        rngs=rngs,
    )

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.key_proj = nnx.Linear(ple_embed_dim, self.hc_hidden_size, **kw)
    self.value_proj = nnx.Linear(ple_embed_dim, emb_dim, **kw)

    norm_kw = {"group_size": emb_dim, "eps": eps, "rngs": rngs}
    self.norm_key = Qwen3_8RMSNorm(self.hc_hidden_size, **norm_kw)
    self.norm_query = Qwen3_8RMSNorm(self.hc_hidden_size, **norm_kw)
    self.norm_conv = Qwen3_8RMSNorm(self.hc_hidden_size, **norm_kw)

    self.conv1d = nnx.Conv(
        in_features=self.hc_hidden_size,
        out_features=self.hc_hidden_size,
        kernel_size=(ple_conv_kernel_size,),
        feature_group_count=self.hc_hidden_size,
        kernel_dilation=(ngram_size,),
        padding="VALID",
        use_bias=False,
        kernel_init=nnx.initializers.zeros_init(),
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def _short_conv(self, hidden_states: jax.Array) -> jax.Array:
    padded = jnp.pad(hidden_states, ((0, 0), (self.short_conv_state_len, 0), (0, 0)))
    return jax.nn.silu(self.conv1d(padded))

  def __call__(self, hidden_states: jax.Array, input_ids: jax.Array, conv_mask: jax.Array | None = None) -> jax.Array:
    """Forward pass injecting hashed n-gram features into hidden streams."""
    is_4d = hidden_states.ndim == 4
    b, s = hidden_states.shape[:2]
    hidden_flat = hidden_states.reshape(b, s, self.hc_hidden_size) if is_4d else hidden_states

    embeds = self.ple_embedding(input_ids)
    key_normed = self.norm_key(self.key_proj(embeds)).reshape(b, s, self.hc_count, self.emb_dim)
    value = self.value_proj(embeds)
    query_normed = self.norm_query(hidden_flat).reshape(b, s, self.hc_count, self.emb_dim)

    gate = jnp.sum(key_normed.astype(jnp.float32) * query_normed.astype(jnp.float32), axis=-1, keepdims=True)
    gate = jnp.sqrt(jnp.maximum(jnp.abs(gate * self.gate_scale), 1e-6)) * jnp.sign(gate)
    gated_val = (jax.nn.sigmoid(gate) * value[:, :, None, :].astype(jnp.float32)).astype(hidden_states.dtype)

    gated_flat = gated_val.reshape(b, s, self.hc_hidden_size)
    gated_norm = self.norm_conv(gated_flat)

    if conv_mask is not None:
      m = conv_mask[:, :, None].astype(gated_flat.dtype)
      gated_flat, gated_norm = gated_flat * m, gated_norm * m

    out_flat = gated_flat + self._short_conv(gated_norm)
    return out_flat.reshape(b, s, self.hc_count, self.emb_dim) if is_4d else out_flat


# =================================================================================================
# Decoder layer, scannable 4-layer cycle, and scan helpers
# =================================================================================================


def build_ple_layer(config: Any, layer_idx: int = 1, *, rngs: nnx.Rngs) -> Qwen3_8PLELayer | None:
  """Instantiates a Qwen3_8PLELayer if configured at layer_idx, else None."""
  if not (bool(config.use_ple_layer) and layer_idx == int(config.ple_layer_id)):
    return None
  c, dt, wdt = config, jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
  return Qwen3_8PLELayer(
      eos_token_id=int(c.ple_eos_token_id),
      emb_dim=int(c.emb_dim),
      hc_count=int(c.hc_count),
      vocab_size=int(c.vocab_size),
      ple_embed_dim=int(c.ple_embed_dim),
      ngram_size=int(c.ngram_size),
      heads_per_ngram=int(c.heads_per_ngram),
      ngram_vocab_size_base=int(c.ngram_vocab_size_base),
      make_ngram_vocab_size_divisible_by=int(c.make_ngram_vocab_size_divisible_by),
      seed=int(c.ple_seed),
      ple_conv_kernel_size=int(c.ple_conv_kernel_size),
      eps=float(c.normalization_layer_epsilon),
      dtype=dt,
      weight_dtype=wdt,
      rngs=rngs,
  )


class Qwen3_8DecoderLayer(nnx.Module):
  """Single decoder layer with Hyper-Connection residuals and MoE."""

  def __init__(self, config: Any, layer_idx: int, *, include_ple: bool = True, rngs: nnx.Rngs):
    c, emb, dt, wdt = config, int(config.emb_dim), jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
    ckw = {"eps": float(c.normalization_layer_epsilon), "dtype": dt, "weight_dtype": wdt, "rngs": rngs}
    hc_kw = {"emb_dim": emb, "hc_count": int(c.hc_count), "hc_lowrank": int(c.hc_lowrank), **ckw}
    cyc = int(c.inhomogeneous_layer_cycle_interval)
    self.layer_type = "linear_attention" if (layer_idx % cyc) != (cyc - 1) else "full_attention"
    self.attn_hyper_connection = Qwen3_8GatedResidual(use_combine=True, **hc_kw)
    if self.layer_type == "linear_attention":
      self.linear_attn = Qwen3_8GatedDeltaNet(
          emb_dim=emb,
          num_k_heads=int(c.gdn_num_key_heads),
          num_v_heads=int(c.gdn_num_value_heads),
          **ckw,
          head_k_dim=int(c.gdn_key_head_dim),
          head_v_dim=int(c.gdn_value_head_dim),
          chunk_size=int(c.gdn_chunk_size),
          conv_kernel_size=int(c.gdn_conv_kernel_dim),
          output_gate_activation=str(c.gdn_output_gate_activation),
      )
    else:
      self.self_attn = Qwen3_8SparseAttention(
          emb_dim=emb,
          num_query_heads=int(c.num_query_heads),
          num_kv_heads=int(c.num_kv_heads),
          head_dim=int(c.head_dim),
          partial_rotary_factor=float(c.partial_rotary_factor),
          rope_theta=float(c.rope_max_timescale),
          **ckw,
          mrope_section=tuple(c.mrope_section),
          index_n_heads=int(c.index_n_heads),
          index_kv_heads=int(c.index_kv_heads),
          index_head_dim=int(c.index_head_dim),
          token_budget=int(c.index_select_budget),
          compress_ratio=int(c.index_select_block_size),
      )
    self.mlp_hyper_connection = Qwen3_8GatedResidual(use_combine=True, **hc_kw)
    self.mlp = Qwen3_8SparseMoeBlock(
        emb_dim=emb,
        shared_expert_intermediate_size=int(c.mlp_dim),
        moe_mlp_dim=int(c.moe_mlp_dim),
        routed_scaling_factor=float(c.routed_scaling_factor),
        num_experts=int(c.num_experts),
        rngs=rngs,
        num_experts_per_tok=int(c.num_experts_per_tok),
        norm_topk_prob=bool(c.norm_topk_prob),
        dtype=dt,
        weight_dtype=wdt,
    )
    self.ple = build_ple_layer(c, layer_idx, rngs=rngs) if include_ple else None

  def __call__(
      self,
      hidden_states: jax.Array,
      *,
      positions: jax.Array | None = None,
      conv_mask: jax.Array | None = None,
      ple_input_ids: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      return_aux: bool = False,
  ) -> Any:
    """Executes single decoder layer forward pass."""
    if self.ple is not None and ple_input_ids is not None:
      hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, conv_mask=conv_mask)
    mixed, inject = self.attn_hyper_connection.before_block(hidden_states)
    if self.layer_type == "linear_attention":
      attn_out, s_idx = self.linear_attn(mixed, attention_mask=conv_mask), None
    else:
      attn_out, s_idx = self.self_attn(
          mixed, position_ids=positions, attention_mask=conv_mask, segment_ids=segment_ids, return_selected_blocks=True
      )
    hidden_states = self.attn_hyper_connection.combine(hidden_states, attn_out, inject)
    mixed, inject = self.mlp_hyper_connection.before_block(hidden_states)
    mlp_out, r_w, r_idx, r_probs = self.mlp(mixed, return_router_topk=True)
    hidden_states = self.mlp_hyper_connection.combine(hidden_states, mlp_out, inject)
    aux = {
        "router_indices": r_idx,
        "routing_weights": r_w,
        "router_probs": r_probs,
        **({"selected_block_indices": s_idx} if s_idx is not None else {}),
    }
    return (hidden_states, aux) if return_aux else hidden_states


class Qwen3_8CycleBlock(nnx.Module):
  """4-layer cycle block composed of 3 GDN layers and 1 QSA layer."""

  def __init__(self, config: Any, *, rngs: nnx.Rngs):
    self.gdn_0, self.gdn_1, self.gdn_2, self.qsa_3 = (
        Qwen3_8DecoderLayer(config, i, include_ple=False, rngs=rngs) for i in range(4)
    )


def create_scanned_cycles(config: Any, num_cycles: int, *, rngs: nnx.Rngs) -> Qwen3_8CycleBlock:
  """Creates scanned cycle block with stacked parameters along cycle axis."""
  if int(config.inhomogeneous_layer_cycle_interval) != 4:
    raise ValueError(
        f"Scanned cycles require inhomogeneous_layer_cycle_interval == 4, got "
        f"{config.inhomogeneous_layer_cycle_interval}."
    )
  if int(config.num_decoder_layers) % 4 != 0:
    raise ValueError(f"Scanned cycles require num_decoder_layers to be a multiple of 4, got {config.num_decoder_layers}.")
  if bool(config.use_ple_layer) and int(config.ple_layer_id) != 1:
    raise ValueError(f"Scanned cycles only support ple_layer_id == 1, got {config.ple_layer_id}.")
  sp = [nnx.split(Qwen3_8CycleBlock(config, rngs=rngs), nnx.Param, ...) for _ in range(num_cycles)]
  return nnx.merge(sp[0][0], *(jax.tree.map(lambda *x: jnp.stack(x, 0), *(s[i] for s in sp)) for i in (1, 2)))


def apply_scanned_cycles(
    cycles: Qwen3_8CycleBlock,
    ple_layer: Qwen3_8PLELayer | None,
    hidden_states: jax.Array,
    *,
    num_cycles: int,
    positions: jax.Array | None = None,
    conv_mask: jax.Array | None = None,
    ple_input_ids: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    return_aux: bool = False,
    remat: bool = False,
) -> Any:
  """Executes scan over stacked cycle weights with conditional PLE at cycle 0."""
  c_def, c_params, c_rest = nnx.split(cycles, nnx.Param, ...)
  p_def, p_state = nnx.split(ple_layer) if ple_layer is not None else (None, None)

  def ple_step(h, cyc_idx):
    if p_def is not None and ple_input_ids is not None:
      return jax.lax.cond(
          cyc_idx == 0, lambda x: x + nnx.merge(p_def, p_state)(x, ple_input_ids, conv_mask=conv_mask), lambda x: x, h
      )
    return h

  def scan_fn(carry, xs):
    carry = shard_batch(carry)
    blk, kw = nnx.merge(c_def, xs[0], xs[1]), {"positions": positions, "conv_mask": conv_mask, "segment_ids": segment_ids}
    if return_aux:
      h, a0 = blk.gdn_0(carry, return_aux=True, **kw)
      h, a1 = blk.gdn_1(ple_step(h, xs[2]), return_aux=True, **kw)
      h, a2 = blk.gdn_2(h, return_aux=True, **kw)
      h, a3 = blk.qsa_3(h, return_aux=True, **kw)
      return shard_batch(h), (a0, a1, a2, a3)
    h = ple_step(blk.gdn_0(carry, **kw), xs[2])
    return (
        shard_batch(blk.qsa_3(blk.gdn_2(blk.gdn_1(h, **kw), **kw), **kw)),
        None,
    )

  step_fn = jax.checkpoint(scan_fn, prevent_cse=False) if remat else scan_fn
  res = jax.lax.scan(step_fn, hidden_states, (c_params, c_rest, jnp.arange(num_cycles)))
  if return_aux:
    return res[0], [jax.tree.map(lambda x, c=c: x[c], res[1][k]) for c in range(num_cycles) for k in range(4)]
  return res[0]


def copy_unscanned_to_scanned(unscanned_decoder: Any, scanned_decoder: Any) -> None:
  """Copies and stacks weights from unscanned decoder layers into scanned cycles."""
  u, s = getattr(unscanned_decoder, "decoder", unscanned_decoder), getattr(scanned_decoder, "decoder", scanned_decoder)
  if hasattr(unscanned_decoder, "lm_head") and hasattr(scanned_decoder, "lm_head"):
    nnx.update(scanned_decoder.lm_head, nnx.state(unscanned_decoder.lm_head))

  def get_sub(l, a):
    return (l.attn_hyper_connection, getattr(l, a), l.mlp_hyper_connection, l.mlp)

  for k, name in enumerate(["gdn_0", "gdn_1", "gdn_2", "qsa_3"]):
    a = "linear_attn" if k < 3 else "self_attn"
    st = [nnx.state(get_sub(getattr(u, f"layers_{4 * c + k}"), a)) for c in range(s.num_cycles)]
    nnx.update(get_sub(getattr(s.cycles, name), a), jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *st))
  for m in ("token_embedder", "hyper_connection_mixer"):
    nnx.update(getattr(s, m), nnx.state(getattr(u, m)))
  u_ple, s_ple = getattr(getattr(u, "layers_1", None), "ple", None), getattr(s, "ple", None)
  if (u_ple is not None) != (s_ple is not None):
    raise ValueError(f"PLE presence mismatch: unscanned={u_ple is not None}, scanned={s_ple is not None}.")
  if u_ple is not None:
    nnx.update(s_ple, nnx.state(u_ple))


# =================================================================================================
# Multi-token prediction (MTP) draft block
# =================================================================================================


class Qwen3_8MTPBlock(nnx.Module):
  """Single-layer MTP draft block fusing next-token embeddings and HC streams."""

  def __init__(self, config: Any, *, rngs: nnx.Rngs):
    self.config, self.emb_dim = config, int(config.emb_dim)
    self.hc_count, self.hc_lowrank = int(config.hc_count), int(config.hc_lowrank)
    dt, wdt, eps = jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype), float(config.normalization_layer_epsilon)
    self.pre_fc_norm_embedding = Qwen3_8RMSNorm(self.emb_dim, eps=eps, rngs=rngs)
    self.pre_fc_norm_hidden = Qwen3_8RMSNorm(self.hc_count * self.emb_dim, eps=eps, rngs=rngs)
    l_kw = {"use_bias": False, "dtype": dt, "param_dtype": wdt, "rngs": rngs}
    self.fc_embedding = nnx.Linear(self.emb_dim, self.emb_dim, **l_kw)
    self.fc_hidden = nnx.Linear(self.emb_dim, self.emb_dim, **l_kw)
    cyc = int(config.inhomogeneous_layer_cycle_interval)
    self.layers_0 = Qwen3_8DecoderLayer(config, layer_idx=cyc - 1, include_ple=False, rngs=rngs)
    self.hyper_connection_mixer = Qwen3_8GatedResidual(
        emb_dim=self.emb_dim,
        hc_count=self.hc_count,
        hc_lowrank=self.hc_lowrank,
        use_combine=False,
        eps=eps,
        dtype=dt,
        weight_dtype=wdt,
        rngs=rngs,
    )

  def __call__(
      self,
      hc_hidden_states: jax.Array,
      next_token_embeds: jax.Array,
      *,
      positions: jax.Array | None = None,
      conv_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      return_aux: bool = False,
  ) -> jax.Array | tuple[jax.Array, dict[str, Any]]:
    """Forward pass fusing next-token embeddings with HC hidden states."""
    expected_dim = self.hc_count * self.emb_dim
    if hc_hidden_states.shape[-1] != expected_dim:
      raise ValueError(
          f"hc_hidden_states last dimension {hc_hidden_states.shape[-1]} "
          f"must equal hc_count * emb_dim ({expected_dim})."
      )
    expected_embed_shape = (*hc_hidden_states.shape[:-1], self.emb_dim)
    if next_token_embeds.shape != expected_embed_shape:
      raise ValueError(f"next_token_embeds shape {next_token_embeds.shape} must match {expected_embed_shape}.")
    e_proj = self.fc_embedding(self.pre_fc_norm_embedding(next_token_embeds))
    h_norm = self.pre_fc_norm_hidden(hc_hidden_states).reshape(*hc_hidden_states.shape[:-1], self.hc_count, self.emb_dim)
    h_proj = self.fc_hidden(h_norm)
    h_mtp_in = (h_proj + jnp.expand_dims(e_proj, axis=-2)).reshape(hc_hidden_states.shape)
    out = self.layers_0(
        h_mtp_in, positions=positions, conv_mask=conv_mask, segment_ids=segment_ids, return_aux=return_aux
    )
    h_layer, aux = out if return_aux else (out, None)
    h_out = self.hyper_connection_mixer.mix_input(h_layer)
    return (h_out, aux) if return_aux else h_out


# =================================================================================================
# Cross-entropy, router load-balancing, and total training loss
# =================================================================================================


def cross_entropy_loss(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
  """Computes cross-entropy loss between logits and target token IDs in float32."""
  if logits.size == 0 or targets.size == 0:
    raise ValueError("logits and targets must be non-empty.")
  if logits.shape[:-1] != targets.shape:
    raise ValueError(f"logits prefix shape {logits.shape[:-1]} must match targets shape {targets.shape}.")
  if mask is not None and mask.shape != targets.shape:
    raise ValueError(f"mask shape {mask.shape} must match targets shape {targets.shape}.")
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  loss = -jnp.take_along_axis(log_probs, jnp.expand_dims(targets, -1), axis=-1).squeeze(-1)
  if mask is not None:
    m = mask.astype(jnp.float32)
    return jnp.sum(loss * m) / jnp.maximum(jnp.sum(m), 1.0)
  return jnp.mean(loss)


def router_load_balancing_loss(
    aux_list: Any,
    num_experts: int,
    mask: jax.Array | None = None,
    mtp_mask: jax.Array | None = None,
) -> jax.Array:
  """Computes MoE router load balancing auxiliary loss across decoder layers."""
  if aux_list is None:
    raise ValueError("aux_list cannot be None.")
  if isinstance(aux_list, dict):
    if "layer_aux" not in aux_list:
      raise ValueError("aux_list dict must contain 'layer_aux'.")
    layer_items = [(layer, mask) for layer in aux_list["layer_aux"]]
    if aux_list.get("mtp_aux") is not None:
      layer_items.append((aux_list["mtp_aux"], mtp_mask if mtp_mask is not None else mask))
  elif isinstance(aux_list, (list, tuple)):
    layer_items = [(layer, mask) for layer in aux_list]
  else:
    raise ValueError(f"Unsupported aux_list type: {type(aux_list)}.")
  if not layer_items:
    raise ValueError("aux_list contains no layers.")
  for i, (layer_aux, _) in enumerate(layer_items):
    if not isinstance(layer_aux, dict) or "router_probs" not in layer_aux or "router_indices" not in layer_aux:
      raise ValueError(f"Layer {i} aux must be a dict with 'router_probs' and 'router_indices'.")

  tokens_per_expert_sum = jnp.zeros(num_experts, dtype=jnp.float32)
  router_prob_sum = jnp.zeros(num_experts, dtype=jnp.float32)
  total_rows = jnp.array(0.0, dtype=jnp.float32)

  for layer_aux, layer_mask in layer_items:
    r_probs = layer_aux["router_probs"].astype(jnp.float32)
    r_idx = layer_aux["router_indices"]
    oh = jax.nn.one_hot(r_idx, num_experts, dtype=jnp.float32)
    oh_reduce_axes = tuple(range(oh.ndim - 1))
    prob_reduce_axes = tuple(range(r_probs.ndim - 1))
    if layer_mask is not None:
      m = layer_mask.astype(jnp.float32)
      tokens_per_expert_sum = tokens_per_expert_sum + jnp.sum(oh * jnp.expand_dims(m, (-1, -2)), axis=oh_reduce_axes)
      router_prob_sum = router_prob_sum + jnp.sum(r_probs * jnp.expand_dims(m, -1), axis=prob_reduce_axes)
      total_rows = total_rows + jnp.sum(m)
    else:
      tokens_per_expert_sum = tokens_per_expert_sum + jnp.sum(oh, axis=oh_reduce_axes)
      router_prob_sum = router_prob_sum + jnp.sum(r_probs, axis=prob_reduce_axes)
      total_rows = total_rows + float(r_probs.size // num_experts)

  denom = jnp.maximum(total_rows, 1.0)
  tokens_per_expert = tokens_per_expert_sum / denom
  router_prob_per_expert = router_prob_sum / denom
  return (num_experts * jnp.sum(tokens_per_expert * router_prob_per_expert)).astype(jnp.float32)


def compute_total_loss(
    model: Any,
    input_tokens: jax.Array,
    target_tokens: jax.Array,
    *,
    target_mask: jax.Array | None = None,
    positions: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    router_aux_coeff: float | None = None,
    mtp_loss_coeff: float | None = None,
    use_mtp: bool = True,
    **kwargs: Any,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes primary CE loss, router load balancing loss, and optional MTP loss."""
  if router_aux_coeff is None:
    router_aux_coeff = getattr(model.config, "load_balance_loss_weight", None)
  if router_aux_coeff is None:
    raise ValueError("router_aux_coeff must be specified or present in model.config.load_balance_loss_weight.")
  router_aux_coeff = float(router_aux_coeff)

  run_mtp, need_aux = use_mtp and (model.mtp is not None), router_aux_coeff > 0.0
  out = model(input_tokens, positions, segment_ids, return_aux=need_aux, return_mtp=run_mtp, **kwargs)
  logits = out[0] if (run_mtp or need_aux) else out
  aux = out[1] if (run_mtp or need_aux) else None
  mtp_logits = aux["mtp_logits"] if (run_mtp and need_aux) else (out[1] if run_mtp else None)

  primary_loss = cross_entropy_loss(logits, target_tokens, mask=target_mask)

  mtp_aux_mask = None
  mtp_loss = None
  if run_mtp and mtp_logits is not None:
    if mtp_logits.shape[1] <= 1:
      mtp_loss = jnp.array(0.0, dtype=jnp.float32)
      mtp_aux_mask = jnp.zeros_like(target_tokens, dtype=jnp.float32)
    else:
      m = (
          target_mask[:, 1:].astype(jnp.float32)
          if target_mask is not None
          else jnp.ones_like(target_tokens[:, 1:], dtype=jnp.float32)
      )
      if segment_ids is not None:
        m = m * ((segment_ids[:, :-1] == segment_ids[:, 1:]) & (segment_ids[:, :-1] != 0)).astype(jnp.float32)
      mtp_loss = cross_entropy_loss(mtp_logits[:, :-1, :], target_tokens[:, 1:], mask=m)
      mtp_aux_mask = jnp.pad(m, ((0, 0), (0, 1)))

  router_loss = (
      router_load_balancing_loss(
          aux,
          int(model.config.num_experts),
          mask=target_mask,
          mtp_mask=mtp_aux_mask,
      )
      if need_aux
      else jnp.array(0.0, dtype=jnp.float32)
  )
  total_loss = primary_loss + router_aux_coeff * router_loss
  metrics = {"total_loss": total_loss, "primary_loss": primary_loss, "router_loss": router_loss}

  if mtp_loss is not None:
    if mtp_loss_coeff is None:
      mtp_loss_coeff_val = getattr(model.config, "mtp_loss_scaling_factor", None)
    else:
      mtp_loss_coeff_val = mtp_loss_coeff
    if mtp_loss_coeff_val is None:
      raise ValueError(
          "mtp_loss_coeff must be specified or present in model.config.mtp_loss_scaling_factor when MTP is enabled."
      )
    alpha = float(mtp_loss_coeff_val)
    total_loss = total_loss + alpha * mtp_loss
    metrics["total_loss"], metrics["mtp_loss"] = total_loss, mtp_loss

  return total_loss, metrics


# =================================================================================================
# Decoder backbone, top-level model, and factory
# =================================================================================================


def _positions_and_conv_mask(
    tokens: jax.Array,
    positions: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
  """Computes default positions and convolution mask if not provided."""
  b, s = tokens.shape
  pos = jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32), (b, s)) if positions is None else positions
  cmask = (segment_ids != 0).astype(jnp.int32) if segment_ids is not None else None
  return pos, cmask


class Qwen3_8Decoder(nnx.Module):
  """Decoder stack with token embedding, 4-stream residual, and final mixer."""

  def __init__(self, config: Any, *, rngs: nnx.Rngs):
    c, dt, wdt = config, jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
    self.emb_dim, self.vocab_size, self.eos_token_id = int(c.emb_dim), int(c.vocab_size), int(c.ple_eos_token_id)
    self.num_layers, self.hc_count, self.scan_layers = int(c.num_decoder_layers), int(c.hc_count), bool(c.scan_layers)
    self.remat = str(getattr(c, "remat_policy", "none")) not in (
        "none",
        "",
        "None",
    )
    self.dtype, self.weight_dtype = dt, wdt
    self.token_embedder = nnx.Embed(self.vocab_size, self.emb_dim, dtype=dt, param_dtype=wdt, rngs=rngs)
    if self.scan_layers:
      if int(config.inhomogeneous_layer_cycle_interval) != 4:
        raise ValueError(
            f"Scanned cycles require inhomogeneous_layer_cycle_interval == 4, got "
            f"{config.inhomogeneous_layer_cycle_interval}."
        )
      if self.num_layers % 4 != 0:
        raise ValueError(f"Scanned cycles require num_decoder_layers to be a multiple of 4, got {self.num_layers}.")
      if bool(config.use_ple_layer) and int(config.ple_layer_id) != 1:
        raise ValueError(f"Scanned cycles only support ple_layer_id == 1, got {config.ple_layer_id}.")
      self.num_cycles = self.num_layers // 4
      self.cycles = create_scanned_cycles(config, self.num_cycles, rngs=rngs)
      self.ple = build_ple_layer(config, rngs=rngs)
    else:
      for i in range(self.num_layers):
        setattr(self, f"layers_{i}", Qwen3_8DecoderLayer(config, i, include_ple=True, rngs=rngs))
    self.hyper_connection_mixer = Qwen3_8GatedResidual(
        emb_dim=self.emb_dim,
        hc_count=self.hc_count,
        hc_lowrank=int(config.hc_lowrank),
        use_combine=False,
        eps=float(config.normalization_layer_epsilon),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      *,
      return_aux: bool = False,
      return_pre_mixer: bool = False,
  ) -> Any:
    """Forward pass computing full decoder representations."""
    tok = decoder_input_tokens
    pos, cmask = _positions_and_conv_mask(tok, decoder_positions, decoder_segment_ids)
    pids = tok if cmask is None else jnp.where(cmask.astype(jnp.bool_), tok, self.eos_token_id)
    tok_emb = shard_batch(self.token_embedder(tok))
    h, aux_list = (
        init_hyper_connection_streams(tok_emb, hc_count=self.hc_count),
        [],
    )
    if self.scan_layers:
      res = apply_scanned_cycles(
          self.cycles,
          self.ple,
          h,
          num_cycles=self.num_cycles,
          positions=pos,
          conv_mask=cmask,
          ple_input_ids=pids,
          segment_ids=decoder_segment_ids,
          return_aux=return_aux,
          remat=self.remat,
      )
      h, aux_list = res if return_aux else (res, [])
    else:
      for i in range(self.num_layers):
        out = getattr(self, f"layers_{i}")(
            h, positions=pos, conv_mask=cmask, ple_input_ids=pids, segment_ids=decoder_segment_ids, return_aux=return_aux
        )
        h, _ = (out[0], aux_list.append(out[1])) if return_aux else (out, None)
    h_4stream, h_mixed = h, shard_batch(self.hyper_connection_mixer.mix_input(h))
    if return_pre_mixer:
      return (h_mixed, h_4stream, aux_list) if return_aux else (h_mixed, h_4stream)
    return (h_mixed, aux_list) if return_aux else h_mixed


class Qwen3_8FlashNextModel(nnx.Module):
  """Top-level Qwen3.8-Flash-Next model with decoder and lm_head."""

  def __init__(self, config: Any, mesh: Any = None, *, quant_mode: str = "train", rngs: nnx.Rngs):
    self.config, self.mesh, self.quant_mode, self.decoder = config, mesh, quant_mode, Qwen3_8Decoder(config, rngs=rngs)
    self.lm_head = nnx.Linear(
        self.decoder.emb_dim,
        self.decoder.vocab_size,
        use_bias=False,
        dtype=self.decoder.dtype,
        param_dtype=self.decoder.weight_dtype,
        rngs=rngs,
    )
    mtp_layers = int(getattr(config, "mtp_num_layers", 0))
    if mtp_layers not in (0, 1):
      raise ValueError(f"mtp_num_layers must be 0 or 1, got {mtp_layers}.")
    self.mtp = Qwen3_8MTPBlock(config, rngs=rngs) if mtp_layers > 0 else None

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      *,
      enable_dropout: bool = False,
      return_hidden_states: bool = False,
      return_aux: bool = False,
      return_mtp: bool = False,
      next_tokens: jax.Array | None = None,
      **kwargs: Any,
  ) -> Any:
    """Forward pass computing language model logits and optional aux outputs."""
    del enable_dropout, kwargs
    if return_mtp and return_hidden_states:
      raise ValueError("Simultaneous return_mtp=True and return_hidden_states=True is not supported.")
    if return_mtp:
      if self.mtp is None:
        raise ValueError("MTP module not initialized.")
      dec_out = self.decoder(
          decoder_input_tokens, decoder_positions, decoder_segment_ids, return_aux=return_aux, return_pre_mixer=True
      )
      h_mixed, h_4stream = dec_out[0], dec_out[1]
      aux_list = dec_out[2] if return_aux else []
      if next_tokens is not None:
        next_tok = next_tokens
      else:
        next_tok = jnp.pad(decoder_input_tokens[:, 1:], ((0, 0), (0, 1)))
        if decoder_segment_ids is not None:
          same_seg = (decoder_segment_ids != 0) & (
              decoder_segment_ids == jnp.pad(decoder_segment_ids[:, 1:], ((0, 0), (0, 1)))
          )
          next_tok = jnp.where(same_seg, next_tok, 0)
      pos, cmask = _positions_and_conv_mask(decoder_input_tokens, decoder_positions, decoder_segment_ids)
      mtp_out = self.mtp(
          h_4stream,
          shard_batch(self.decoder.token_embedder(next_tok)),
          positions=pos,
          conv_mask=cmask,
          segment_ids=decoder_segment_ids,
          return_aux=return_aux,
      )
      mtp_h, mtp_aux = mtp_out if return_aux else (mtp_out, None)
      mtp_logits = shard_batch(self.lm_head(mtp_h))
      logits = shard_batch(self.lm_head(h_mixed))
      if return_aux:
        aux_dict = {"layer_aux": aux_list, "mtp_aux": mtp_aux, "mtp_logits": mtp_logits}
        return (logits, aux_dict)
      return (logits, mtp_logits)

    out = self.decoder(
        decoder_input_tokens,
        decoder_positions,
        decoder_segment_ids,
        return_aux=return_aux,
        return_pre_mixer=return_hidden_states,
    )
    if return_hidden_states or return_aux:
      return (shard_batch(self.lm_head(out[0])), *out[1:])
    return shard_batch(self.lm_head(out))


def create_qwen3_8_flash_next_model(
    config: Any, mesh: Any = None, *, rngs: nnx.Rngs | None = None, quant_mode_str: str = "train", **_: Any
) -> Qwen3_8FlashNextModel:
  """Factory function creating a Qwen3_8FlashNextModel instance."""
  return Qwen3_8FlashNextModel(config=config, mesh=mesh, quant_mode=quant_mode_str, rngs=rngs or nnx.Rngs(params=0))
