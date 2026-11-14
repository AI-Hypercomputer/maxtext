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
import dataclasses
import functools
import math
from typing import Any

from flax import nnx
import jax
from jax import checkpoint_policies as cp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import xla_metadata
import jax.numpy as jnp
import numpy as np
from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as tokamax_splash_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as tokamax_splash_mask
from maxtext.m3.core import kernels as m3_kernels
from maxtext.m3.core import moe as m3_moe
from maxtext.m3.core.sharding import DefaultSharding, Sharding, TensorType, param_metadata, sharding_from_config
from maxtext.m3.infra import remat as m3_remat


# =================================================================================================
# Sharding helpers
# =================================================================================================


def _mesh_axis_names(entry: Any) -> tuple[str, ...]:
  """Mesh axis names carried by one `PartitionSpec` entry."""
  if entry is None:
    return ()
  return (entry,) if isinstance(entry, str) else tuple(entry)


def _param_kwargs(
    sharding: Sharding | None,
    tensor_name: str,
    axes: Sequence[str | None],
    shape: Sequence[int],
) -> dict[str, Any]:
  """Returns `param_metadata` for a parameter of `shape` whose dimensions carry logical `axes`.

  A dimension the mesh in scope cannot divide evenly is left replicated instead of failing construction, so small
  test and decode configs build on any mesh. With no mesh in scope the logical spec is recorded but not applied.
  """
  if len(axes) != len(shape):
    raise ValueError(f"{tensor_name}: got {len(axes)} logical axes {tuple(axes)} for shape {tuple(shape)}.")
  meta = param_metadata(sharding or DefaultSharding(), tensor_name, axes)
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty:
    return meta
  entries = []
  for dim, entry in zip(shape, meta["out_sharding"]):
    shards = math.prod(int(mesh.shape[name]) for name in _mesh_axis_names(entry))
    entries.append(entry if dim % shards == 0 else None)
  meta["out_sharding"] = jax.sharding.PartitionSpec(*entries)
  return meta


def _linear(
    in_features: int,
    out_features: int,
    sharding: Sharding | None,
    tensor_name: str,
    axes: tuple[str | None, str | None],
    **kwargs: Any,
) -> nnx.Linear:
  """Builds a bias-free `nnx.Linear` whose `(in_features, out_features)` kernel carries logical `axes`."""
  meta = _param_kwargs(sharding, tensor_name, axes, (in_features, out_features))
  return nnx.Linear(in_features, out_features, use_bias=False, kernel_metadata=meta, **kwargs)


def _depthwise_conv1d(
    features: int,
    kernel_size: int,
    sharding: Sharding | None,
    tensor_name: str,
    **kwargs: Any,
) -> nnx.Conv:
  """Builds a bias-free depthwise 1D `nnx.Conv` whose `(kernel_size, 1, features)` kernel shards over its channels."""
  meta = _param_kwargs(sharding, tensor_name, (None, None, "conv_channels"), (kernel_size, 1, features))
  return nnx.Conv(
      in_features=features,
      out_features=features,
      kernel_size=(kernel_size,),
      feature_group_count=features,
      use_bias=False,
      kernel_metadata=meta,
      **kwargs,
  )


def get_batch_mesh_axes(sharding: Sharding | None = None) -> tuple[str, ...]:
  """Returns the active mesh axes `sharding` maps the `batch` axis onto when a multi-device mesh is set."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty or mesh.size <= 1:
    return ()
  spec = (sharding or DefaultSharding())("batch", ("batch",), TensorType.ACTIVATION)
  return tuple(ax for ax in _mesh_axis_names(spec[0]) if mesh.shape.get(ax, 1) > 1)


def get_length_mesh_axes(sharding: Sharding | None = None) -> tuple[str, ...]:
  """Returns the active mesh axes `sharding` maps the `length` axis onto when a multi-device mesh is set."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty or mesh.size <= 1:
    return ()
  spec = (sharding or DefaultSharding())("length", ("length",), TensorType.ACTIVATION)
  return tuple(ax for ax in _mesh_axis_names(spec[0]) if mesh.shape.get(ax, 1) > 1)


def get_norm_length_mesh_axes(sharding: Sharding | None = None) -> tuple[str, ...]:
  """Returns the active mesh axes `sharding` maps the `norm_length` axis onto when a multi-device mesh is set."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty or mesh.size <= 1:
    return ()
  spec = (sharding or DefaultSharding())("norm_length", ("norm_length",), TensorType.ACTIVATION)
  return tuple(ax for ax in _mesh_axis_names(spec[0]) if mesh.shape.get(ax, 1) > 1)


def get_heads_mesh_axes(sharding: Sharding | None = None) -> tuple[str, ...]:
  """Returns the active mesh axes `sharding` maps the `heads` axis onto when a multi-device mesh is set."""
  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty or mesh.size <= 1:
    return ()
  spec = (sharding or DefaultSharding())("q", ("heads",), TensorType.ACTIVATION)
  return tuple(ax for ax in _mesh_axis_names(spec[0]) if mesh.shape.get(ax, 1) > 1)


def shard_batch(x: jax.Array, sharding: Sharding | None = None, tensor_name: str = "hidden_states") -> jax.Array:
  """Constrains `(batch, norm_length, ...)` activations to the `batch` and `norm_length` axes of `sharding`.

  Axis 0 is the batch and, for tensors of rank 2 and up, axis 1 is the sequence. Nothing is constrained without a
  multi-device mesh. A batch or a sequence that doesn't divide evenly over its shards (e.g. batch-1 decode) is left
  unconstrained on that axis, matching the unsharded fallback in `Qwen3_8Experts`.
  """
  if x.ndim == 0:
    return x
  mesh = jax.sharding.get_abstract_mesh()
  batch_axes = get_batch_mesh_axes(sharding)
  length_axes = get_norm_length_mesh_axes(sharding) if x.ndim >= 2 else ()
  if batch_axes and x.shape[0] % math.prod(mesh.shape[ax] for ax in batch_axes) != 0:
    batch_axes = ()
  if length_axes and x.shape[1] % math.prod(mesh.shape[ax] for ax in length_axes) != 0:
    length_axes = ()
  if not batch_axes and not length_axes:
    return x
  axes = ["batch" if batch_axes else None, *([None] * (x.ndim - 1))]
  if length_axes:
    axes[1] = "norm_length"
  return (sharding or DefaultSharding()).constrain(x, tensor_name, tuple(axes))


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
      sharding: Sharding | None = None,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.dim, self.group_size, self.eps = dim, group_size, eps
    if group_size is not None and dim % group_size != 0:
      raise ValueError(f"dim ({dim}) not divisible by group_size ({group_size}).")
    self.scale = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32), **_param_kwargs(sharding, "scale", ("norm",), (dim,)))

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
      sharding: Sharding | None = None,
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
    self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32), **_param_kwargs(sharding, "scale", ("norm",), (dim,)))

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


def init_hyper_connection_streams(
    token_embeddings: jax.Array,
    hc_count: int = 4,
    sharding: Sharding | None = None,
) -> jax.Array:
  """Initializes residual stream by replicating token embeddings across streams."""
  return shard_batch(jnp.concatenate([token_embeddings] * hc_count, axis=-1), sharding, "hyper_connection_streams")


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
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.emb_dim, self.hc_count = emb_dim, hc_count
    self.hc_lowrank, self.use_combine = hc_lowrank, use_combine
    self.dtype, self.weight_dtype = dtype, weight_dtype

    # The concatenated `hc_count * emb_dim` stream dimension is the model dimension of these weights.
    hc_dim = hc_count * emb_dim
    self.hc_norm = Qwen3_8RMSNorm(hc_dim, group_size=emb_dim, eps=eps, sharding=sharding, rngs=rngs)
    kw = {"dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.input_mix_weight_down = _linear(hc_dim, hc_lowrank, sharding, "input_mix_weight_down", ("embed", None), **kw)
    self.input_mix_weight_up = _linear(hc_lowrank, hc_dim, sharding, "input_mix_weight_up", (None, "embed"), **kw)
    self.block_inject_weight = (
        _linear(hc_dim, hc_count, sharding, "block_inject_weight", ("embed", None), **kw) if use_combine else None
    )

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


def _block_doubling_unit_lower_inverse(s_mat: jax.Array, chunk_size: int) -> jax.Array:
  """Computes the exact inverse of (I + s_mat) for strictly lower-triangular s_mat with MXU-friendly matmuls.

  Bottom-up block doubling: starting from 1x1 blocks (inverse = I), each level merges adjacent b x b diagonal
  blocks whose inverses A11^-1 and A22^-1 are already known, using
  [[A11, 0], [A21, A22]]^-1 = [[A11^-1, 0], [-A22^-1 A21 A11^-1, A22^-1]].
  Every intermediate is a block of the true inverse, so unlike the Neumann product (I+N)(I+N^2)(I+N^4)... it
  does not suffer catastrophic cancellation in float32 when keys within a chunk are correlated and decay is weak
  (N^(2^j) can reach 1e7+ while the inverse stays O(1)). Costs 2 * ceil(log2(chunk_size)) matmuls per chunk and
  works for non-power-of-two chunk sizes.
  """
  prec = jax.lax.Precision.HIGHEST
  ar = jnp.arange(chunk_size)
  inv = jnp.broadcast_to(jnp.eye(chunk_size, dtype=jnp.float32), s_mat.shape)
  b = 1
  while b < chunk_size:
    # Lower-left b x b sub-block of each 2b x 2b diagonal block: the only part of s_mat not yet folded into inv.
    cross = ((ar[:, None] // (2 * b)) == (ar[None, :] // (2 * b))) & ((ar[:, None] // b) == (ar[None, :] // b) + 1)
    off = jnp.matmul(jnp.matmul(inv, jnp.where(cross, s_mat, 0.0), precision=prec), inv, precision=prec)
    inv = inv - jnp.where(cross, off, 0.0)
    b *= 2
  return inv


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
    chunk_inverse: str = "block_doubling",
) -> tuple[jax.Array, jax.Array | None]:
  """Chunked Gated DeltaNet forward pass in float32."""
  init_dtype = value.dtype
  s = key.shape[1]
  xs, state_shape = _gdn_chunk_inputs(query, key, value, g, beta, chunk_size, use_qk_l2norm_in_kernel, chunk_inverse)
  final_state, core_out = _gdn_chunk_scan(xs, _init_state(state_shape, initial_state))
  core_out = jnp.swapaxes(core_out[:, :, :s, :], 1, 2).astype(init_dtype)
  return core_out, (final_state if output_final_state else None)


def _gdn_chunk_inputs(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool,
    chunk_inverse: str,
) -> tuple[tuple[jax.Array, ...], tuple[int, int, int, int]]:
  """Per-chunk terms of the chunked delta rule, chunk axis leading, and the `(B, H, D_k, D_v)` state shape.

  Nothing here crosses a chunk boundary: the only coupling between chunks is the recurrent state that
  `_gdn_chunk_scan` carries.
  """
  if chunk_inverse not in ("solve", "block_doubling"):
    raise ValueError(f"Expected chunk_inverse in ('solve', 'block_doubling'), got {chunk_inverse!r}.")
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
  rhs = jnp.concatenate([v_beta, k_beta * jnp.expand_dims(jnp.exp(g_cumsum), -1)], axis=-1)
  if chunk_inverse == "block_doubling":
    solved = jnp.matmul(_block_doubling_unit_lower_inverse(s_mat, chunk_size), rhs, precision=prec)
  else:
    i_plus_s = jnp.eye(chunk_size, dtype=jnp.float32) + s_mat
    solved = jax.scipy.linalg.solve_triangular(i_plus_s, rhs, lower=True, unit_diagonal=True)
  v_inv, k_cumdecay = jnp.split(solved, [d_v], axis=-1)
  xs = tuple(jnp.moveaxis(x, 2, 0) for x in (query, key, v_inv, k_cumdecay, decay_mask, g_cumsum))
  return xs, (b, h, d_k, d_v)


def _gdn_chunk_scan(xs: tuple[jax.Array, ...], init_state: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Carries the recurrent state over chunks; returns the final state and the `(B, H, S_padded, D_v)` output."""
  prec = jax.lax.Precision.HIGHEST

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

  final_state, core_chunks = jax.lax.scan(_chunk_step, init_state, xs)
  core_out = jnp.moveaxis(core_chunks, 0, 2)
  b, h, num_chunks, chunk_size, d_v = core_out.shape
  return final_state, core_out.reshape(b, h, num_chunks * chunk_size, d_v)


@functools.partial(jax.checkpoint, static_argnums=(1,))
def _gdn_chunk_transfer(xs: tuple[jax.Array, ...], state_shape: tuple[int, int, int, int]) -> tuple[jax.Array, jax.Array]:
  """The affine map `S_out = A @ S_in + B` that a run of chunks applies to the recurrent state.

  Each chunk updates the state as `S * decay + k_decay^T @ (v - k_cumdecay @ S)`, which is affine in `S`. Carrying
  `[A | B]` through the same update, from `[I | 0]`, composes those maps without knowing `S_in`.
  """
  _, k_all, v_all, k_cd_all, _, g_all = xs
  b, h, d_k, d_v = state_shape
  prec = jax.lax.Precision.HIGHEST

  def _step(carry, step_xs):
    k_i, v_i, k_cd_i, g_i = step_xs
    v_ext = jnp.concatenate([jnp.zeros((*v_i.shape[:-1], d_k), v_i.dtype), v_i], axis=-1)
    v_new = v_ext - jnp.matmul(k_cd_i, carry, precision=prec)
    decay_last = jnp.expand_dims(jnp.exp(g_i[..., -1:]), -1)
    k_decay = k_i * jnp.expand_dims(jnp.exp(g_i[..., -1:] - g_i), -1)
    return carry * decay_last + jnp.matmul(jnp.swapaxes(k_decay, -1, -2), v_new, precision=prec), None

  eye = jnp.broadcast_to(jnp.eye(d_k, dtype=jnp.float32), (b, h, d_k, d_k))
  init = jnp.concatenate([eye, jnp.zeros((b, h, d_k, d_v), jnp.float32)], axis=-1)
  final, _ = jax.lax.scan(_step, init, (k_all, v_all, k_cd_all, g_all))
  return final[..., :d_k], final[..., d_k:]


def context_parallel_chunk_gated_delta_rule(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    batch_axes: tuple[str, ...],
    length_axes: tuple[str, ...],
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_inverse: str = "block_doubling",
) -> jax.Array:
  """Chunked Gated DeltaNet over a sequence that stays sharded along `length_axes` (zero initial state).

  The recurrence only couples sequence shards through the `(B, H, D_k, D_v)` state. Each shard first computes the
  affine map its own chunks apply to that state, and the maps are composed across shards via a Hillis-Steele prefix
  scan (`ppermute` over a single length axis, or `all_gather` fold over multi-axis length specs) to obtain each
  shard's incoming state. It then runs the ordinary chunk scan from that state without gathering the sequence.

  Each shard's sequence must be a multiple of `chunk_size`, so that chunk boundaries match the unsharded rule.
  """
  mesh = jax.sharding.get_abstract_mesh()
  init_dtype = value.dtype
  num_shards = math.prod(mesh.shape[ax] for ax in length_axes)
  if key.shape[1] % (num_shards * chunk_size) != 0:
    raise ValueError(
        f"Sequence length {key.shape[1]} must be a multiple of {num_shards} sequence shards x chunk_size {chunk_size}."
    )
  batch_entry = batch_axes if batch_axes else None
  spec4 = jax.sharding.PartitionSpec(batch_entry, length_axes, None, None)
  spec3 = jax.sharding.PartitionSpec(batch_entry, length_axes, None)
  prec = jax.lax.Precision.HIGHEST

  def _per_shard(q_l, k_l, v_l, g_l, beta_l):
    xs, state_shape = _gdn_chunk_inputs(q_l, k_l, v_l, g_l, beta_l, chunk_size, use_qk_l2norm_in_kernel, chunk_inverse)
    a_loc, b_loc = _gdn_chunk_transfer(xs, state_shape)
    if len(length_axes) == 1:
      cp_ax = length_axes[0]
      shard_id = jax.lax.axis_index(cp_ax)
      a_run, b_run = a_loc, b_loc
      step = 1
      while step < num_shards:
        fwd = [(i, (i + step) % num_shards) for i in range(num_shards)]
        a_recv = jax.lax.ppermute(a_run, cp_ax, fwd)
        b_recv = jax.lax.ppermute(b_run, cp_ax, fwd)
        a_cmp = jnp.matmul(a_run, a_recv, precision=prec)
        b_cmp = jnp.matmul(a_run, b_recv, precision=prec) + b_run
        live = shard_id >= step
        a_run = jnp.where(live, a_cmp, a_run)
        b_run = jnp.where(live, b_cmp, b_run)
        step *= 2
      shift1 = [(i, (i + 1) % num_shards) for i in range(num_shards)]
      b_ex = jax.lax.ppermute(b_run, cp_ax, shift1)
      state_in = jnp.where(shard_id == 0, jnp.zeros(state_shape, jnp.float32), b_ex)
    else:
      a_all = jax.lax.all_gather(a_loc, length_axes, axis=0, tiled=False)
      b_all = jax.lax.all_gather(b_loc, length_axes, axis=0, tiled=False)
      shard_id = jax.lax.axis_index(length_axes)

      def _fold(state, step_xs):
        a_i, b_i, i = step_xs
        nxt = jnp.matmul(a_i, state, precision=prec) + b_i
        return jnp.where(i < shard_id, nxt, state), None

      state_in, _ = jax.lax.scan(_fold, jnp.zeros(state_shape, jnp.float32), (a_all, b_all, jnp.arange(num_shards)))
    _, core_out = _gdn_chunk_scan(xs, state_in)
    return jnp.swapaxes(core_out, 1, 2).astype(init_dtype)

  return jax.shard_map(
      _per_shard,
      mesh=mesh,
      in_specs=(spec4, spec4, spec4, spec3, spec3),
      out_specs=spec4,
      check_vma=False,
  )(query, key, value, g, beta)


def _context_parallel_causal_conv1d(
    x: jax.Array,
    kernel: jax.Array,
    *,
    batch_axes: tuple[str, ...],
    length_axes: tuple[str, ...],
    sharding: Sharding | None = None,
    dtype: Any = jnp.bfloat16,
) -> jax.Array:
  """Causal depthwise 1D convolution over a sequence sharded along `length_axes` via `ppermute` halo exchange."""
  mesh = jax.sharding.get_abstract_mesh()
  num_shards = math.prod(int(mesh.shape[ax]) for ax in length_axes)
  kernel_size = kernel.shape[0]
  halo_len = kernel_size - 1
  chan_spec = _param_kwargs(sharding, "conv1d", (None, None, "conv_channels"), kernel.shape)["out_sharding"]
  channel_axes = tuple(
      ax
      for ax in _mesh_axis_names(chan_spec[2])
      if mesh.shape.get(ax, 1) > 1 and ax not in batch_axes and ax not in length_axes
  )
  if channel_axes and x.shape[-1] % math.prod(int(mesh.shape[ax]) for ax in channel_axes) != 0:
    channel_axes = ()
  batch_entry = batch_axes if batch_axes else None
  chan_entry = channel_axes if channel_axes else None
  x_spec = jax.sharding.PartitionSpec(batch_entry, length_axes, chan_entry)
  k_spec = jax.sharding.PartitionSpec(None, None, chan_entry)

  def _per_shard(x_l: jax.Array, k_l: jax.Array) -> jax.Array:
    x_l = x_l.astype(dtype)
    k_l = k_l.astype(dtype)
    if halo_len > 0:
      tail = x_l[:, -halo_len:, :]
      if len(length_axes) == 1:
        cp_ax = length_axes[0]
        perm = [(i, (i + 1) % num_shards) for i in range(num_shards)]
        recv_halo = jax.lax.ppermute(tail, cp_ax, perm)
        shard_id = jax.lax.axis_index(cp_ax)
      else:
        all_tails = jax.lax.all_gather(tail, length_axes, axis=0, tiled=False)
        shard_id = jax.lax.axis_index(length_axes)
        recv_halo = all_tails[jnp.maximum(shard_id - 1, 0)]
      left_halo = jnp.where(shard_id == 0, jnp.zeros_like(recv_halo), recv_halo)
      x_padded = jnp.concatenate([left_halo, x_l], axis=1)
    else:
      x_padded = x_l
    c_loc = x_l.shape[-1]
    return jax.lax.conv_general_dilated(
        lhs=x_padded,
        rhs=k_l,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=c_loc,
    )

  return jax.shard_map(
      _per_shard,
      mesh=mesh,
      in_specs=(x_spec, k_spec),
      out_specs=x_spec,
      check_vma=False,
  )(x, kernel)


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
      sharding: Sharding | None = None,
      *,
      output_gate_activation: str,
      chunk_inverse: str = "block_doubling",
      rngs: nnx.Rngs,
  ):
    if chunk_inverse not in ("solve", "block_doubling"):
      raise ValueError(f"Expected chunk_inverse in ('solve', 'block_doubling'), got {chunk_inverse!r}.")
    self.emb_dim, self.chunk_size, self.chunk_inverse = emb_dim, chunk_size, chunk_inverse
    self.num_k_heads, self.num_v_heads = num_k_heads, num_v_heads
    if num_v_heads % num_k_heads != 0:
      raise ValueError(f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads}).")
    self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim
    self.output_gate_activation = output_gate_activation
    self.dtype = dtype
    self.sharding = sharding

    self.key_dim = num_k_heads * head_k_dim
    self.value_dim = num_v_heads * head_v_dim
    conv_dim = self.key_dim * 2 + self.value_dim

    # `in_proj_qkv` emits the query, key, and value head groups one after another, so a contiguous split of its
    # features is not a split over heads: only the model dimension is sharded (no tensor parallelism).
    kw = {"dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.in_proj_qkv = _linear(emb_dim, conv_dim, sharding, "in_proj_qkv", ("embed", None), **kw)
    self.in_proj_z = _linear(emb_dim, self.value_dim, sharding, "in_proj_z", ("embed", None), **kw)
    self.in_proj_b = _linear(emb_dim, num_v_heads, sharding, "in_proj_b", ("embed", None), **kw)
    self.in_proj_a = _linear(emb_dim, num_v_heads, sharding, "in_proj_a", ("embed", None), **kw)

    self.conv1d = _depthwise_conv1d(conv_dim, conv_kernel_size, sharding, "conv1d", padding="CAUSAL", **kw)
    u = jax.random.uniform(rngs.params(), (num_v_heads,), minval=0.01, maxval=16.0)
    head_meta = _param_kwargs(sharding, "A_log", (None,), (num_v_heads,))
    self.A_log = nnx.Param(jnp.log(u.astype(jnp.float32)), **head_meta)  # pylint: disable=invalid-name
    self.dt_bias = nnx.Param(
        jnp.ones((num_v_heads,), dtype=jnp.float32), **_param_kwargs(sharding, "dt_bias", (None,), (num_v_heads,))
    )
    self.norm = Qwen3_8RMSNormGated(head_v_dim, eps=eps, activation=output_gate_activation, sharding=sharding, rngs=rngs)
    self.out_proj = _linear(self.value_dim, emb_dim, sharding, "out_proj", (None, "embed"), **kw)

  def _context_parallel_axes(
      self,
      batch: int,
      seq_len: int,
      initial_state: jax.Array | None,
      output_final_state: bool,
  ) -> tuple[str, ...]:
    """Mesh axes to keep the sequence sharded on through the recurrence, or `()` for the unsharded rule.

    The context-parallel rule covers training: no carried-in state, no state returned, and a per-shard sequence that
    is a whole number of chunks. Everything else (decode, ragged lengths) runs the unsharded rule, where the
    partitioner gathers the sequence.
    """
    del batch
    if self.sharding is None or initial_state is not None or output_final_state:
      return ()
    length_axes = get_length_mesh_axes(self.sharding)
    if not length_axes:
      return ()
    mesh = jax.sharding.get_abstract_mesh()
    if seq_len % (math.prod(mesh.shape[ax] for ax in length_axes) * self.chunk_size) != 0:
      return ()
    return length_axes

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
    qkv = checkpoint_name(self.in_proj_qkv(hidden_states), "qkv_proj")
    length_axes = self._context_parallel_axes(b, s, initial_state, output_final_state)
    batch_axes = ()
    if length_axes:
      batch_axes = get_batch_mesh_axes(self.sharding)
      mesh = jax.sharding.get_abstract_mesh()
      if b % math.prod(mesh.shape[ax] for ax in batch_axes) != 0:
        batch_axes = ()
      s_local = s // math.prod(mesh.shape[ax] for ax in length_axes)
      if s_local >= self.conv1d.kernel[...].shape[0] - 1:
        conv_out = _context_parallel_causal_conv1d(
            qkv,
            self.conv1d.kernel[...],
            batch_axes=batch_axes,
            length_axes=length_axes,
            sharding=self.sharding,
            dtype=self.dtype,
        )
      else:
        conv_out = self.conv1d(qkv)
    else:
      conv_out = self.conv1d(qkv)
    mixed_qkv = jax.nn.silu(conv_out)
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
    elif length_axes:
      core_attn_out, last_state = (
          context_parallel_chunk_gated_delta_rule(
              query,
              key,
              value,
              g=g,
              beta=beta,
              batch_axes=batch_axes,
              length_axes=length_axes,
              chunk_size=self.chunk_size,
              use_qk_l2norm_in_kernel=True,
              chunk_inverse=self.chunk_inverse,
          ),
          None,
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
          chunk_inverse=self.chunk_inverse,
      )
    core_attn_out = checkpoint_name(core_attn_out, "context")
    core_normed = self.norm(core_attn_out, z).reshape(b, s, self.value_dim)
    out = checkpoint_name(self.out_proj(core_normed), "out_proj")
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
    inv_freq_np = 1.0 / (
        np.float64(self.rope_theta) ** (np.arange(0, self.rotary_dim, 2, dtype=np.float64) / np.float64(self.rotary_dim))
    )
    self._inv_freq = tuple(np.asarray(inv_freq_np, dtype=np.float32).tolist())

  @property
  def inv_freq(self) -> jax.Array:
    """Returns inverse frequencies precomputed in host `float64` and cast once to `float32`."""
    return jnp.asarray(self._inv_freq, dtype=jnp.float32)

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
      mask_mode: str = "gather",
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if index_kv_heads != 1:
      raise ValueError(f"Expected index_kv_heads == 1 (MQA indexer keys), got {index_kv_heads}.")
    if compress_ratio <= 0 or token_budget < compress_ratio:
      raise ValueError(
          "Expected compress_ratio > 0 and token_budget >= compress_ratio,"
          f" got compress_ratio={compress_ratio}, token_budget={token_budget}."
      )
    if mask_mode not in ("gather", "reshape"):
      raise ValueError(f"Expected mask_mode in ('gather', 'reshape'), got {mask_mode!r}.")
    self.index_head_dim, self.index_n_heads = index_head_dim, index_n_heads
    self.token_budget = token_budget
    self.compress_ratio, self.block_topk = compress_ratio, token_budget // compress_ratio
    self.mask_mode = mask_mode

    # Query heads and the single key head share the output dimension, so only the model dimension is sharded.
    kw = {"dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    index_dim = (index_n_heads + index_kv_heads) * index_head_dim
    self.index_qk_proj = _linear(emb_dim, index_dim, sharding, "index_qk_proj", ("embed", None), **kw)
    self.q_layernorm = Qwen3_8RMSNorm(index_head_dim, eps=eps, sharding=sharding, rngs=rngs)
    self.k_layernorm = Qwen3_8RMSNorm(index_head_dim, eps=eps, sharding=sharding, rngs=rngs)

  def __call__(
      self,
      hidden_states: jax.Array,
      position_embeddings: tuple[jax.Array, jax.Array],
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
      return_token_mask: bool = True,
  ) -> tuple[jax.Array | None, jax.Array]:
    """Computes sparse token mask (B, 1, S, S) and document-local block indices (B, S, topk).

    Args:
      hidden_states: Input tensor of shape (batch, seq_len, emb_dim).
      position_embeddings: Tuple of (cos, sin) arrays for RoPE.
      attention_mask: Optional (B, S) mask; 0/False = padding, non-zero/True = active (NOT a PyTorch additive mask).
      segment_ids: Optional (B, S) 1-indexed document IDs (0 = padding); when both are given, padding from
        attention_mask is folded into segment_ids.
      return_token_mask: Whether to construct and return the (B, 1, S, S) boolean token mask.

    Returns:
      Tuple of (token_mask_or_none, selected_block_indices).
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

    unpacked = segment_ids is None and attention_mask is None
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

    if not return_token_mask:
      return None, selected_block_indices
    if self.mask_mode == "reshape" and s <= self.token_budget:
      ar_k, ar_q = jnp.arange(s)[None, None, :], jnp.arange(s)[None, :, None]
      token_mask = (same_doc & (ar_k <= ar_q))[:, None, :, :]
      return token_mask, selected_block_indices
    m_local = jnp.arange(num_blocks)[None, None, :]
    in_topk = jnp.any(
        (selected_block_indices[:, :, :, None] == m_local) & (selected_block_indices[:, :, :, None] >= 0), axis=2
    )
    ar_k, ar_q = jnp.arange(s)[None, None, :], jnp.arange(s)[None, :, None]
    if self.mask_mode == "reshape" and unpacked:
      expanded = jnp.broadcast_to(in_topk[:, :, :, None], (b, s, num_blocks, c)).reshape(b, s, num_blocks * c)
      if num_blocks * c < s:
        pad_cols = jnp.broadcast_to(in_topk[:, :, -1:], (b, s, s - num_blocks * c))
        is_selected = jnp.concatenate([expanded, pad_cols], axis=-1)
      else:
        is_selected = expanded[..., :s]
      in_tail = ar_k >= (((ar_q + 1) // c) * c)
      token_mask = ((ar_k <= ar_q) & (is_selected | in_tail))[:, None, :, :]
      return token_mask, selected_block_indices
    if self.mask_mode == "reshape":
      lblk = jnp.clip(local_pos // c, 0, num_blocks - 1)
      one_hot_lblk = jax.nn.one_hot(lblk, num_blocks, dtype=jnp.bfloat16)
      is_selected = jnp.einsum("bqm,bkm->bqk", in_topk.astype(jnp.bfloat16), one_hot_lblk) > 0.5
      in_tail = local_pos[:, None, :] >= (((local_pos + 1) // c) * c)[:, :, None]
      token_mask = (same_doc & (ar_k <= ar_q) & (is_selected | in_tail))[:, None, :, :]
      return token_mask, selected_block_indices

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
      sharding: Sharding | None = None,
      *,
      indexer_mask_mode: str = "gather",
      attention_kernel: str = "dense",
      rngs: nnx.Rngs,
  ):
    self.sharding = sharding or DefaultSharding()
    if num_query_heads % num_kv_heads != 0:
      raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads}).")
    if attention_kernel not in ("dense", "flash"):
      raise ValueError(f"Expected attention_kernel in ('dense', 'flash'), got {attention_kernel!r}.")
    self.head_dim, self.num_query_heads, self.num_kv_heads = head_dim, num_query_heads, num_kv_heads
    self.attention_kernel = attention_kernel

    # Projection features are head-major (`q_proj` packs each head's query and gate together), so a contiguous split
    # of the flattened feature dimension is a split over heads.
    kw = {"dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.q_proj = _linear(emb_dim, num_query_heads * head_dim * 2, sharding, "q_proj", ("embed", "heads"), **kw)
    self.k_proj = _linear(emb_dim, num_kv_heads * head_dim, sharding, "k_proj", ("embed", "kv_heads"), **kw)
    self.v_proj = _linear(emb_dim, num_kv_heads * head_dim, sharding, "v_proj", ("embed", "kv_heads"), **kw)
    self.o_proj = _linear(num_query_heads * head_dim, emb_dim, sharding, "o_proj", ("heads", "embed"), **kw)

    self.q_norm = Qwen3_8RMSNorm(head_dim, eps=eps, sharding=sharding, rngs=rngs)
    self.k_norm = Qwen3_8RMSNorm(head_dim, eps=eps, sharding=sharding, rngs=rngs)

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
        mask_mode=indexer_mask_mode,
        sharding=sharding,
        rngs=rngs,
    )

  def _flash_attention(
      self,
      query_states: jax.Array,
      key_states: jax.Array,
      value_states: jax.Array,
      *,
      attention_mask: jax.Array | None = None,
      segment_ids: jax.Array | None = None,
  ) -> jax.Array:
    """Runs causal Tokamax Splash Attention when seq_len <= token_budget."""
    b, s, _, _ = query_states.shape
    block_size = 128
    pad_len = (block_size - s % block_size) % block_size
    padded_s = s + pad_len

    q = jnp.swapaxes(query_states, 1, 2)
    q = (q * (1.0 / jnp.sqrt(float(self.head_dim)))).astype(q.dtype)
    k = jnp.swapaxes(key_states, 1, 2)
    v = jnp.swapaxes(value_states, 1, 2)
    if pad_len > 0:
      pad_cfg = ((0, 0), (0, 0), (0, pad_len), (0, 0))
      q, k, v = jnp.pad(q, pad_cfg), jnp.pad(k, pad_cfg), jnp.pad(v, pad_cfg)

    seg_q: jax.Array | None = None
    seg_kv: jax.Array | None = None
    if segment_ids is not None or attention_mask is not None or pad_len > 0:
      seg = jnp.ones((b, s), dtype=jnp.int32) if segment_ids is None else segment_ids.astype(jnp.int32)
      if attention_mask is not None:
        seg = jnp.where(attention_mask != 0, seg, 0)
      seg_q = jnp.where(seg == 0, -2, seg)
      seg_kv = jnp.where(seg == 0, -1, seg)
      if pad_len > 0:
        seg_q = jnp.pad(seg_q, ((0, 0), (0, pad_len)), constant_values=-2)
        seg_kv = jnp.pad(seg_kv, ((0, 0), (0, pad_len)), constant_values=-1)

    sa_config = tokamax_splash_kernel.SplashConfig(
        block_q=block_size,
        block_kv=block_size,
        block_kv_compute=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        block_q_dq=block_size,
        block_kv_dq=block_size,
        fuse_reciprocal=True,
        use_base2_exp=True,
        interpret=(jax.default_backend() != "tpu"),
    )
    mesh = jax.sharding.get_abstract_mesh()
    # Queries stay on the sequence shard that holds them; keys and values are gathered along the sequence, which a
    # causal query needs anyway. The kernel's block tables are split the same way as the queries.
    length_axes = get_length_mesh_axes(self.sharding)
    q_shards = math.prod(mesh.shape[ax] for ax in length_axes) if length_axes else 1
    if padded_s % (q_shards * block_size) != 0 or padded_s % (q_shards * q_shards) != 0:
      length_axes, q_shards = (), 1
    causal_mask = tokamax_splash_mask.CausalMask(shape=(padded_s, padded_s), shard_count=q_shards)

    def _run_batch(
        q_b: jax.Array,
        k_b: jax.Array,
        v_b: jax.Array,
        sq_b: jax.Array | None,
        skv_b: jax.Array | None,
        kernel: Any = None,
    ) -> jax.Array:
      if kernel is None:
        kernel = tokamax_splash_kernel.make_splash_mha_single_device(causal_mask, config=sa_config)
      if sq_b is None:
        return jax.vmap(kernel)(q_b, k_b, v_b)
      return jax.vmap(
          lambda q_i, k_i, v_i, sq_i, skv_i: kernel(
              q_i, k_i, v_i, segment_ids=tokamax_splash_kernel.SegmentIds(q=sq_i, kv=skv_i)
          )
      )(q_b, k_b, v_b, sq_b, skv_b)

    batch_axes = get_batch_mesh_axes(self.sharding)
    batch_shards = math.prod(mesh.shape[ax] for ax in batch_axes) if batch_axes else 1
    batch_entry = batch_axes if batch_axes and b % batch_shards == 0 else None
    head_axes = get_heads_mesh_axes(self.sharding)
    head_shards = math.prod(mesh.shape[ax] for ax in head_axes) if head_axes else 1
    head_entry = None
    if head_axes and q.shape[1] % head_shards == 0:
      if k.shape[1] % head_shards == 0:
        head_entry = head_axes
      elif head_shards % k.shape[1] == 0:
        rep = head_shards // k.shape[1]
        k, v = jnp.repeat(k, rep, axis=1), jnp.repeat(v, rep, axis=1)
        head_entry = head_axes
    if length_axes:
      q_spec = jax.sharding.PartitionSpec(batch_entry, head_entry, length_axes, None)
      kv_spec = jax.sharding.PartitionSpec(batch_entry, head_entry, None, None)
      seg_q_spec = jax.sharding.PartitionSpec(batch_entry, length_axes) if seg_q is not None else None
      seg_kv_spec = jax.sharding.PartitionSpec(batch_entry, None) if seg_q is not None else None
      kernel = tokamax_splash_kernel.make_splash_mha(mask=causal_mask, config=sa_config, q_seq_shards=q_shards)
      kernel_spec = kernel.manual_sharding_spec(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(length_axes)))
      out = jax.shard_map(
          _run_batch,
          mesh=mesh,
          in_specs=(q_spec, kv_spec, kv_spec, seg_q_spec, seg_kv_spec, kernel_spec),
          out_specs=q_spec,
          check_vma=False,
      )(q, k, v, seg_q, seg_kv, kernel)
    elif not mesh.empty and mesh.size > 1:
      qkv_spec = jax.sharding.PartitionSpec(batch_entry, head_entry, None, None)
      seg_spec = jax.sharding.PartitionSpec(batch_entry, None) if seg_q is not None else None
      out = jax.shard_map(
          _run_batch,
          mesh=mesh,
          in_specs=(qkv_spec, qkv_spec, qkv_spec, seg_spec, seg_spec),
          out_specs=qkv_spec,
          check_vma=False,
      )(q, k, v, seg_q, seg_kv)
    else:
      out = _run_batch(q, k, v, seg_q, seg_kv)

    out = jnp.swapaxes(out[:, :, :s, :], 1, 2)
    return out.reshape(b, s, self.num_query_heads * self.head_dim)

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
    if self.attention_kernel == "flash" and s > self.indexer.token_budget:
      raise ValueError(
          f"qsa_attention_kernel='flash' requires seq_len <= token_budget ({self.indexer.token_budget}), got seq_len={s}."
      )
    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(s)[None, :], (b, s))

    cos, sin = self.rotary_emb(hidden_states, position_ids)
    selected_token_mask, selected_block_indices = self.indexer(
        hidden_states,
        (cos, sin),
        attention_mask=attention_mask,
        segment_ids=segment_ids,
        return_token_mask=(self.attention_kernel == "dense"),
    )

    q_and_gate = checkpoint_name(self.q_proj(hidden_states), "query_proj").reshape(
        b, s, self.num_query_heads, 2 * self.head_dim
    )
    query_states = self.q_norm(q_and_gate[..., : self.head_dim])
    gate = q_and_gate[..., self.head_dim :].reshape(b, s, self.num_query_heads * self.head_dim)
    key_states = self.k_norm(
        checkpoint_name(self.k_proj(hidden_states), "key_proj").reshape(b, s, self.num_kv_heads, self.head_dim)
    )
    value_states = checkpoint_name(self.v_proj(hidden_states), "value_proj").reshape(
        b, s, self.num_kv_heads, self.head_dim
    )

    res = apply_rotary_pos_emb(query_states, key_states, cos=cos, sin=sin, unsqueeze_dim=2)
    assert isinstance(res, tuple)
    query_states, key_states = res

    if self.attention_kernel == "flash":
      attn_output = self._flash_attention(
          query_states, key_states, value_states, attention_mask=attention_mask, segment_ids=segment_ids
      )
    else:
      assert selected_token_mask is not None
      if (rep := self.num_query_heads // self.num_kv_heads) > 1:
        key_states, value_states = jnp.repeat(key_states, rep, axis=2), jnp.repeat(value_states, rep, axis=2)

      attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query_states, key_states) * (1.0 / jnp.sqrt(float(self.head_dim)))
      min_val = jnp.finfo(jnp.float32).min
      attn_weights = jnp.where(selected_token_mask, attn_weights, min_val)

      probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(value_states.dtype)
      attn_output = jnp.einsum("bhqk,bkhd->bqhd", probs, value_states).reshape(b, s, self.num_query_heads * self.head_dim)
    attn_output = checkpoint_name(attn_output, "context")
    attn_output = (attn_output * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(attn_output.dtype)

    out = checkpoint_name(self.o_proj(attn_output), "out_proj")
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
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    kw = {"dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.gate_proj = _linear(emb_dim, intermediate_dim, sharding, "gate_proj", ("embed", "mlp"), **kw)
    self.up_proj = _linear(emb_dim, intermediate_dim, sharding, "up_proj", ("embed", "mlp"), **kw)
    self.down_proj = _linear(intermediate_dim, emb_dim, sharding, "down_proj", ("mlp", "embed"), **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass computing down_proj(silu(gate_proj(x)) * up_proj(x))."""
    gate = jax.nn.silu(checkpoint_name(self.gate_proj(x), "mlpwi_0"))
    up = checkpoint_name(self.up_proj(x), "mlpwi_1")
    return checkpoint_name(self.down_proj(gate * up), "mlpwo")


class Qwen3_8Experts(nnx.Module):
  """Collection of routed expert weights stored as 4D/3D tensors.

  Weight Layout Contract (for CL 9 checkpoint conversion):
    - gate_up_proj: (num_experts, emb_dim, 2, moe_mlp_dim) -> (E, H, 2, I), 0=gate, 1=up.
      HuggingFace stores gate_up_proj as (E, 2I, H); transpose last two axes and reshape to (E, H, 2, I).
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
      sharding: Sharding | None = None,
      *,
      executor: m3_moe.MoEExecutor | None = None,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.emb_dim = emb_dim
    self.moe_mlp_dim = moe_mlp_dim
    self.dtype = dtype
    sharding_obj = sharding or DefaultSharding()
    self.sharding = sharding_obj

    if executor is not None:
      self.executor = executor
    elif sharding is not None:
      tok_spec = sharding_obj("moe_tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION)
      wi_spec = sharding_obj("wi_0", ("experts", "expert_embed", "expert_mlp"), m3_moe.TensorType.WEIGHT)
      self.executor = m3_moe.MoEExecutor(
          kind="gmm_v2",
          batch=tok_spec[0],
          length=tok_spec[1],
          experts=wi_spec[0],
          mlp=wi_spec[2],
      )
    else:
      self.executor = m3_moe.RAGGED_DOT

    gu_shape = (num_experts, emb_dim, 2, moe_mlp_dim)
    dn_shape = (num_experts, moe_mlp_dim, emb_dim)
    self.gate_up_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), gu_shape, weight_dtype),
        **_param_kwargs(sharding_obj, "gate_up_proj", ("experts", "expert_embed", None, "expert_mlp"), gu_shape),
    )
    self.down_proj = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), dn_shape, weight_dtype),
        **_param_kwargs(sharding_obj, "down_proj", ("experts", "expert_mlp", "expert_embed"), dn_shape),
    )

  def __call__(
      self,
      hidden_states: jax.Array,
      top_k_index: jax.Array,
      top_k_weights: jax.Array,
  ) -> jax.Array:
    """Dispatches tokens through shared `MoEExecutor` (`ragged_all_to_all` EP + grouped matmul)."""
    orig_dtype = hidden_states.dtype
    orig_shape = hidden_states.shape
    if hidden_states.ndim == 2:
      x_3d = hidden_states[:, None, :].astype(self.dtype)
      w_3d = top_k_weights[:, None, :].astype(self.dtype)
      idx_3d = top_k_index[:, None, :]
    elif hidden_states.ndim == 3:
      x_3d = hidden_states.astype(self.dtype)
      w_3d = top_k_weights.astype(self.dtype)
      idx_3d = top_k_index
    else:
      raise ValueError(f"Expected 2D or 3D hidden_states; got {hidden_states.shape}.")

    w_gate_up = jnp.asarray(self.gate_up_proj[...], dtype=self.dtype)
    w_down = jnp.asarray(self.down_proj[...], dtype=self.dtype)

    def _run_with_mesh(active_mesh: Any) -> jax.Array:
      exec_cfg = self.executor
      x_in, w_in, idx_in = x_3d, w_3d, idx_3d
      reshaped_to_tokens = False
      if exec_cfg.kind in m3_moe.SHARDED_EXECUTOR_KINDS:
        batch_ax = m3_kernels.resolve_mesh_axes(active_mesh, exec_cfg.batch)
        batch_shards = m3_kernels.mesh_axis_size(active_mesh, batch_ax)
        ep_size = m3_kernels.mesh_axis_size(active_mesh, m3_kernels.resolve_mesh_axes(active_mesh, exec_cfg.experts))
        tp_size = m3_kernels.mesh_axis_size(active_mesh, m3_kernels.resolve_mesh_axes(active_mesh, exec_cfg.mlp))
        length_ax = m3_kernels.resolve_mesh_axes(active_mesh, exec_cfg.length)
        if x_in.shape[1] % m3_kernels.mesh_axis_size(active_mesh, length_ax) != 0:
          exec_cfg = dataclasses.replace(exec_cfg, length=None)
        if x_in.shape[0] % batch_shards != 0:
          total_tokens = x_in.shape[0] * x_in.shape[1]
          if total_tokens % batch_shards == 0:
            x_in = x_in.reshape(total_tokens, 1, x_in.shape[2])
            w_in = w_in.reshape(total_tokens, 1, w_in.shape[2])
            idx_in = idx_in.reshape(total_tokens, 1, idx_in.shape[2])
            exec_cfg = dataclasses.replace(exec_cfg, length=None)
            reshaped_to_tokens = True
          elif ep_size == 1 and tp_size == 1:
            exec_cfg = dataclasses.replace(exec_cfg, batch=None)
      y = exec_cfg.fused(x_in, w_in, idx_in, w_gate_up, w_down)
      if reshaped_to_tokens:
        y = y.reshape(x_3d.shape)
      return y

    amb_mesh = jax.sharding.get_abstract_mesh()
    if amb_mesh.empty:
      fallback_exec = (
          dataclasses.replace(self.executor, kind="ragged_dot")
          if self.executor.kind in m3_moe.SHARDED_EXECUTOR_KINDS
          else self.executor
      )
      out_3d = fallback_exec.fused(x_3d, w_3d, idx_3d, w_gate_up, w_down)
    else:
      out_3d = _run_with_mesh(amb_mesh)

    return out_3d.reshape(orig_shape).astype(orig_dtype)

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

    w_gate_up = jnp.asarray(self.gate_up_proj[...], dtype=self.dtype)[top_k_index]
    gate = jnp.einsum("td,tkdm->tkm", hidden_states.astype(self.dtype), w_gate_up[:, :, :, 0, :])
    up = jnp.einsum("td,tkdm->tkm", hidden_states.astype(self.dtype), w_gate_up[:, :, :, 1, :])
    act = jax.nn.silu(gate) * up
    w_down = jnp.asarray(self.down_proj[...], dtype=self.dtype)[top_k_index]
    expert_out = jnp.einsum("tkm,tkmd->tkd", act, w_down)
    out = jnp.einsum("tkd,tk->td", expert_out, top_k_weights.astype(expert_out.dtype))
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
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor
    self.dtype = dtype
    # The expert dimension stays replicated: `embed` and `experts` both land on the `expert` mesh axis.
    self.linear = _linear(
        emb_dim, num_experts, sharding, "router", ("embed", None), dtype=dtype, param_dtype=weight_dtype, rngs=rngs
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
      sharding: Sharding | None = None,
      *,
      shared_expert_intermediate_size: int,
      executor: m3_moe.MoEExecutor | None = None,
      rngs: nnx.Rngs,
  ):
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
        sharding=sharding,
        rngs=rngs,
    )
    self.experts = Qwen3_8Experts(
        num_experts=num_experts,
        emb_dim=emb_dim,
        moe_mlp_dim=moe_mlp_dim,
        dtype=dtype,
        weight_dtype=weight_dtype,
        sharding=sharding,
        executor=executor,
        rngs=rngs,
    )
    self.shared_expert = Qwen3_8MLP(
        emb_dim=emb_dim,
        intermediate_dim=shared_expert_intermediate_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        sharding=sharding,
        rngs=rngs,
    )
    kw = {"dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.shared_expert_gate = _linear(emb_dim, 1, sharding, "shared_expert_gate", ("embed", None), **kw)

  def __call__(
      self,
      hidden_states: jax.Array,
      return_router_topk: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward pass through routed experts and gated shared expert."""
    orig_shape = hidden_states.shape
    if hidden_states.ndim == 2:
      hidden_states = hidden_states[:, None, :]
    b, s, _ = hidden_states.shape

    routing_weights, router_indices, router_probs = self.gate(hidden_states)
    routed_out = self.experts(hidden_states, router_indices, routing_weights)
    shared_out = self.shared_expert(hidden_states)
    shared_gate = jax.nn.sigmoid(self.shared_expert_gate(hidden_states).astype(jnp.float32)).astype(shared_out.dtype)

    combined = (routed_out + shared_gate * shared_out).reshape(orig_shape)
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
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.sharding = sharding or DefaultSharding()
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

    # At full size the table is the largest single weight in the model, and its feature dimension is too small to
    # divide over a large mesh, so it is the rows that are sharded.
    table_shape = (self.padded_vocab_size, head_dim_per_ngram)
    self.ngram_embedding = nnx.Embed(
        *table_shape,
        dtype=dtype,
        param_dtype=weight_dtype,
        embedding_metadata=_param_kwargs(sharding, "ngram_embedding", ("ngram_vocab", None), table_shape),
        rngs=rngs,
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
    b = input_ids.shape[0]
    mult_bytes = self.mult_bytes
    head_vocab_sizes_arr = self.head_vocab_sizes_arr
    head_offsets_arr = self.head_offsets_arr
    context = jnp.full((b, self.ngram_size - 1), self.eos_token_id, dtype=input_ids.dtype)
    history = jnp.concatenate([context, input_ids], axis=-1)
    shifted_tokens = [_shift_right_ignore_eos(history, sh, self.eos_token_id) for sh in range(self.ngram_size)]
    shifted_bytes = [_mul_u64_bytes(shifted_tokens[p], mult_bytes[p]) for p in range(self.ngram_size)]

    blocks = []
    mixed = shifted_bytes[0]
    for ngram in range(2, self.ngram_size + 1):
      s_idx = (ngram - 2) * self.heads_per_ngram
      e_idx = s_idx + self.heads_per_ngram
      mixed = jnp.bitwise_xor(mixed, shifted_bytes[ngram - 1])  # Running XOR over the first `ngram` shifts.

      v = head_vocab_sizes_arr[s_idx:e_idx][None, None, :]
      rem = jnp.zeros((b, history.shape[1], v.shape[-1]), dtype=jnp.uint32)
      for b_idx in range(7, -1, -1):
        byte = mixed[:, :, b_idx : b_idx + 1]
        rem = (rem * jnp.uint32(16) + ((byte >> jnp.uint32(4)) & jnp.uint32(0xF))) % v
        rem = (rem * jnp.uint32(16) + (byte & jnp.uint32(0xF))) % v

      blocks.append((rem + head_offsets_arr[s_idx:e_idx][None, None, :]).astype(jnp.int32))

    return jnp.concatenate(blocks, axis=-1)[:, self.ngram_size - 1 :, :]

  def __call__(self, input_ids: jax.Array) -> jax.Array:
    ngram_ids = self.compute_ngram_ids(input_ids)
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and mesh.size > 1:
      ngram_ids = self.sharding.constrain(ngram_ids, "ngram_ids", (None,) * ngram_ids.ndim)
    embeds = shard_batch(self.ngram_embedding(ngram_ids), self.sharding, "ngram_embeds")
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
      sharding: Sharding | None = None,
      *,
      rngs: nnx.Rngs,
  ):
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
        sharding=sharding,
        rngs=rngs,
    )

    kw = {"dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.key_proj = _linear(ple_embed_dim, self.hc_hidden_size, sharding, "key_proj", ("embed", None), **kw)
    self.value_proj = _linear(ple_embed_dim, emb_dim, sharding, "value_proj", ("embed", None), **kw)

    norm_kw = {"group_size": emb_dim, "eps": eps, "sharding": sharding, "rngs": rngs}
    self.norm_key = Qwen3_8RMSNorm(self.hc_hidden_size, **norm_kw)
    self.norm_query = Qwen3_8RMSNorm(self.hc_hidden_size, **norm_kw)
    self.norm_conv = Qwen3_8RMSNorm(self.hc_hidden_size, **norm_kw)

    self.conv1d = _depthwise_conv1d(
        self.hc_hidden_size,
        ple_conv_kernel_size,
        sharding,
        "conv1d",
        kernel_dilation=(ngram_size,),
        padding="VALID",
        kernel_init=nnx.initializers.zeros_init(),
        **kw,
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
    if conv_mask is not None:
      # The dilated causal conv reads up to `short_conv_state_len` earlier positions, so without this the
      # right-padded positions after the last real token would pick up its activations.
      out_flat = out_flat * m
    return out_flat.reshape(b, s, self.hc_count, self.emb_dim) if is_4d else out_flat


# =================================================================================================
# Decoder layer, scannable 4-layer cycle, and scan helpers
# =================================================================================================


def build_ple_layer(
    config: Any,
    layer_idx: int = 1,
    *,
    sharding: Sharding | None = None,
    rngs: nnx.Rngs,
) -> Qwen3_8PLELayer | None:
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
      sharding=sharding,
      rngs=rngs,
  )


class Qwen3_8DecoderLayer(nnx.Module):
  """Single decoder layer with Hyper-Connection residuals and MoE."""

  def __init__(
      self,
      config: Any,
      layer_idx: int,
      *,
      include_ple: bool = True,
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    c, emb, dt, wdt = config, int(config.emb_dim), jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
    sharding_obj = sharding or sharding_from_config(config)
    self.sharding = sharding_obj
    ckw = {
        "eps": float(c.normalization_layer_epsilon),
        "dtype": dt,
        "weight_dtype": wdt,
        "sharding": sharding_obj,
        "rngs": rngs,
    }
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
          chunk_inverse=str(c.gdn_chunk_inverse),
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
          indexer_mask_mode=str(c.qsa_indexer_mask),
          attention_kernel=str(c.qsa_attention_kernel),
      )
    self.mlp_hyper_connection = Qwen3_8GatedResidual(use_combine=True, **hc_kw)
    moe_executor = m3_moe.moe_from_config(c, sharding_obj)
    self.mlp = Qwen3_8SparseMoeBlock(
        emb_dim=emb,
        shared_expert_intermediate_size=int(c.mlp_dim),
        moe_mlp_dim=int(c.moe_mlp_dim),
        routed_scaling_factor=float(c.routed_scaling_factor),
        num_experts=int(c.num_experts),
        sharding=sharding_obj,
        rngs=rngs,
        num_experts_per_tok=int(c.num_experts_per_tok),
        norm_topk_prob=bool(c.norm_topk_prob),
        dtype=dt,
        weight_dtype=wdt,
        executor=moe_executor,
    )
    self.ple = build_ple_layer(c, layer_idx, sharding=sharding_obj, rngs=rngs) if include_ple else None

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
    hidden_states = checkpoint_name(hidden_states, "hidden_states")
    if self.ple is not None and ple_input_ids is not None:
      hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, conv_mask=conv_mask)
    mixed, inject = self.attn_hyper_connection.before_block(hidden_states)
    if self.layer_type == "linear_attention":
      attn_out, s_idx = self.linear_attn(mixed, attention_mask=conv_mask), None
    else:
      attn_out, s_idx = self.self_attn(
          mixed, position_ids=positions, attention_mask=conv_mask, segment_ids=segment_ids, return_selected_blocks=True
      )
    attn_out = checkpoint_name(attn_out, "self_attn")
    hidden_states = shard_batch(self.attn_hyper_connection.combine(hidden_states, attn_out, inject), self.sharding)
    mixed, inject = self.mlp_hyper_connection.before_block(hidden_states)
    mlp_out, r_w, r_idx, r_probs = self.mlp(mixed, return_router_topk=True)
    hidden_states = shard_batch(self.mlp_hyper_connection.combine(hidden_states, mlp_out, inject), self.sharding)
    aux = {
        "router_indices": r_idx,
        "routing_weights": r_w,
        "router_probs": r_probs,
        **({"selected_block_indices": s_idx} if s_idx is not None else {}),
    }
    return (hidden_states, aux) if return_aux else hidden_states


class Qwen3_8CycleBlock(nnx.Module):
  """4-layer cycle block composed of 3 GDN layers and 1 QSA layer."""

  def __init__(
      self,
      config: Any,
      *,
      include_ple: bool = False,
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    sharding_obj = sharding or sharding_from_config(config)
    self.gdn_0, self.gdn_1, self.gdn_2, self.qsa_3 = (
        Qwen3_8DecoderLayer(config, i, include_ple=include_ple, sharding=sharding_obj, rngs=rngs) for i in range(4)
    )


def _is_nnx_var(node: Any) -> bool:
  """Returns whether `node` is an `nnx.Variable` leaf for `jax.tree.map`."""
  return isinstance(node, nnx.Variable)


def _stack_cycle_vars(first: Any, *others: Any) -> Any:
  """Stacks per-cycle `nnx.Variable` leaves along axis 0 while prepending `None` to `out_sharding`."""
  if not isinstance(first, nnx.Variable):
    return jnp.stack([first, *others], axis=0)
  stacked = jnp.stack([first.get_value(), *(o.get_value() for o in others)], axis=0)
  spec = first.get_metadata().get("out_sharding")
  if isinstance(spec, jax.sharding.PartitionSpec):
    new_spec = jax.sharding.PartitionSpec(None, *spec)
    cur_sharding = getattr(first.get_value(), "sharding", None)
    if isinstance(cur_sharding, jax.sharding.NamedSharding):
      stacked = jax.device_put(stacked, jax.sharding.NamedSharding(cur_sharding.mesh, new_spec))
    return first.replace(
        stacked, out_sharding=new_spec, eager_sharding=bool(first.get_metadata().get("eager_sharding", False))
    )
  return first.replace(stacked)


def _strip_cycle_axis(var: Any) -> Any:
  """Drops the leading `None` scan axis from `out_sharding` metadata inside `apply_scanned_cycles`."""
  if isinstance(var, nnx.Variable):
    spec = var.get_metadata().get("out_sharding")
    if isinstance(spec, jax.sharding.PartitionSpec) and len(spec) == var.get_value().ndim + 1:
      return var.replace(out_sharding=jax.sharding.PartitionSpec(*spec[1:]))
  return var


def _slice_cycle_var(var: Any, cycle_idx: int) -> Any:
  """Slices cycle `cycle_idx` along axis 0 while dropping the leading `None` from `out_sharding`."""
  if not isinstance(var, nnx.Variable):
    return var[cycle_idx]
  val = var.get_value()[cycle_idx]
  spec = var.get_metadata().get("out_sharding")
  if isinstance(spec, jax.sharding.PartitionSpec) and len(spec) == var.get_value().ndim:
    return var.replace(
        val,
        out_sharding=jax.sharding.PartitionSpec(*spec[1:]),
        eager_sharding=bool(var.get_metadata().get("eager_sharding", False)),
    )
  return var.replace(val)


def _slice_cycle_state(tree: Any, cycle_idx: int) -> Any:
  """Slices an NNX state tree along axis 0 for unrolled cycle execution (`scan_layers=False`)."""
  return jax.tree.map(lambda v: _slice_cycle_var(v, cycle_idx), tree, is_leaf=_is_nnx_var)


def create_scanned_cycles(
    config: Any,
    num_cycles: int,
    *,
    sharding: Sharding | None = None,
    rngs: nnx.Rngs,
) -> Qwen3_8CycleBlock:
  """Creates scanned cycle block with stacked parameters along cycle axis."""
  if int(config.inhomogeneous_layer_cycle_interval) != 4:
    raise ValueError(
        f"Scanned cycles require inhomogeneous_layer_cycle_interval == 4, got "
        f"{config.inhomogeneous_layer_cycle_interval}."
    )
  if int(config.num_decoder_layers) % 4 != 0:
    raise ValueError(f"Scanned cycles require num_decoder_layers to be a multiple of 4, got {config.num_decoder_layers}.")
  sharding_obj = sharding or sharding_from_config(config)
  sp = [nnx.split(Qwen3_8CycleBlock(config, sharding=sharding_obj, rngs=rngs), nnx.Param, ...) for _ in range(num_cycles)]
  stacked_trees = [jax.tree.map(_stack_cycle_vars, *(s[i] for s in sp), is_leaf=_is_nnx_var) for i in (1, 2)]
  return nnx.merge(sp[0][0], *stacked_trees)


def _coerce_remat_callable(remat: Any) -> Callable[..., Any] | None:
  """Extracts a `jax.checkpoint` policy callable from `remat` (`Remat`, callable, or `None`)."""
  if remat is None or isinstance(remat, m3_remat.Remat) or callable(remat):
    return m3_remat.policy_callable(remat)
  pol = getattr(remat, "policy", None)
  return cp.nothing_saveable if pol is None else pol


def _resolve_qwen3_8_remat_policy(config: Any) -> Callable[..., Any] | None:
  """Resolves `config.remat_policy` into a `jax.checkpoint` policy callable via `m3.infra.remat`."""
  raw = getattr(config, "remat_policy", "none")
  if callable(raw):
    return raw
  return m3_remat.policy_callable(m3_remat.resolve_policy(config))


def _run_qwen3_8_sublayer(
    layer: Qwen3_8DecoderLayer,
    hidden_states: jax.Array,
    *,
    eff_policy: Callable[..., Any] | None,
    return_aux: bool,
    positions: jax.Array | None = None,
    conv_mask: jax.Array | None = None,
    ple_input_ids: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
) -> Any:
  """Executes one `Qwen3_8DecoderLayer`, wrapping in `jax.checkpoint(..., prevent_cse=False)` if active."""
  dyn_kw = {
      k: v
      for k, v in (
          ("positions", positions),
          ("conv_mask", conv_mask),
          ("ple_input_ids", ple_input_ids),
          ("segment_ids", segment_ids),
      )
      if v is not None
  }
  if eff_policy is None:
    return layer(hidden_states, return_aux=return_aux, **dyn_kw)
  gd, lp, lr = nnx.split(layer, nnx.Param, ...)

  def _eval_layer(h_in, p_in, r_in, kw_in):
    mod = nnx.merge(gd, p_in, r_in)
    out = mod(h_in, return_aux=return_aux, **kw_in)
    _, _, updated_r = nnx.split(mod, nnx.Param, ...)
    return out, updated_r

  out, new_lr = jax.checkpoint(_eval_layer, policy=eff_policy, prevent_cse=False)(hidden_states, lp, lr, dyn_kw)
  if jax.tree.leaves(new_lr):
    nnx.update(layer, new_lr)
  return out


def run_single_cycle(
    cycle: Qwen3_8CycleBlock,
    hidden_states: jax.Array,
    *,
    eff_policy: Callable[..., Any] | None = None,
    return_aux: bool = False,
    positions: jax.Array | None = None,
    conv_mask: jax.Array | None = None,
    ple_input_ids: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    sharding: Sharding | None = None,
) -> Any:
  """Executes one unstacked 4-layer `Qwen3_8CycleBlock` (`first_cycle`)."""
  h = shard_batch(hidden_states, sharding)
  aux_list = []
  for name in ("gdn_0", "gdn_1", "gdn_2", "qsa_3"):
    out = _run_qwen3_8_sublayer(
        getattr(cycle, name),
        h,
        eff_policy=eff_policy,
        return_aux=return_aux,
        positions=positions,
        conv_mask=conv_mask,
        ple_input_ids=ple_input_ids,
        segment_ids=segment_ids,
    )
    if return_aux:
      h, a = out
      aux_list.append(a)
      h = shard_batch(h, sharding)
    else:
      h = shard_batch(out, sharding)
  return (h, aux_list) if return_aux else h


def apply_scanned_cycles(
    cycles: Qwen3_8CycleBlock,
    hidden_states: jax.Array,
    *,
    num_cycles: int,
    scan_layers: bool = True,
    positions: jax.Array | None = None,
    conv_mask: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    return_aux: bool = False,
    remat: bool = False,
    remat_policy: Any = None,
    sharding: Sharding | None = None,
) -> Any:
  """Executes `jax.lax.scan` (or unrolled Python loop when `scan_layers=False`) over stacked homogeneous cycles."""
  eff_policy = _coerce_remat_callable(remat_policy)
  if eff_policy is None and remat:
    eff_policy = cp.nothing_saveable
  use_remat = eff_policy is not None

  sub_names = ("gdn_0", "gdn_1", "gdn_2", "qsa_3")
  sub_defs: dict[str, Any] = {}
  sub_params: dict[str, Any] = {}
  sub_rests: dict[str, Any] = {}
  for name in sub_names:
    gd, lp, lr = nnx.split(getattr(cycles, name), nnx.Param, ...)
    sub_defs[name] = gd
    sub_params[name] = lp
    sub_rests[name] = lr

  dyn_kw = {
      k: v for k, v in (("positions", positions), ("conv_mask", conv_mask), ("segment_ids", segment_ids)) if v is not None
  }

  def _run_slot(name: str, h_in: jax.Array, p_slice: Any, r_slice: Any) -> Any:
    gd = sub_defs[name]
    lp = jax.tree.map(_strip_cycle_axis, p_slice, is_leaf=_is_nnx_var)
    lr = jax.tree.map(_strip_cycle_axis, r_slice, is_leaf=_is_nnx_var)

    def _eval_slot(x_in, p_in, r_in, kw_in):
      lyr = nnx.merge(gd, p_in, r_in)
      return lyr(x_in, return_aux=return_aux, **kw_in)

    args = (h_in, lp, lr, dyn_kw)
    if use_remat:
      return jax.checkpoint(_eval_slot, policy=eff_policy, prevent_cse=scan_layers)(*args)
    return _eval_slot(*args)

  def scan_fn(carry, xs):
    p_slices, r_slices = xs
    h = shard_batch(carry, sharding)
    if return_aux:
      h, a0 = _run_slot("gdn_0", h, p_slices["gdn_0"], r_slices["gdn_0"])
      h, a1 = _run_slot("gdn_1", shard_batch(h, sharding), p_slices["gdn_1"], r_slices["gdn_1"])
      h, a2 = _run_slot("gdn_2", shard_batch(h, sharding), p_slices["gdn_2"], r_slices["gdn_2"])
      h, a3 = _run_slot("qsa_3", shard_batch(h, sharding), p_slices["qsa_3"], r_slices["qsa_3"])
      return shard_batch(h, sharding), (a0, a1, a2, a3)
    h = shard_batch(_run_slot("gdn_0", h, p_slices["gdn_0"], r_slices["gdn_0"]), sharding)
    h = shard_batch(_run_slot("gdn_1", h, p_slices["gdn_1"], r_slices["gdn_1"]), sharding)
    h = shard_batch(_run_slot("gdn_2", h, p_slices["gdn_2"], r_slices["gdn_2"]), sharding)
    h = shard_batch(_run_slot("qsa_3", h, p_slices["qsa_3"], r_slices["qsa_3"]), sharding)
    return h, None

  if not scan_layers:
    h = hidden_states
    aux_out = []
    for c in range(num_cycles):
      p_c = _slice_cycle_state(sub_params, c)
      r_c = _slice_cycle_state(sub_rests, c)
      h, a_c = scan_fn(h, (p_c, r_c))
      if return_aux:
        aux_out.extend(a_c)
    return (h, aux_out) if return_aux else h

  if num_cycles == 1 and not (m3_remat.is_offload_policy(remat_policy) or m3_remat.is_offload_policy(eff_policy)):
    # Retain the trip-count-1 while loop as a memory barrier (see `ScannedLayers` in
    # `maxtext.m3.infra.scan`): prevents XLA from inlining the single cycle and overlapping
    # activation temporaries across the remat boundary. Skip when offloading residuals to
    # `pinned_host` (`S(5)`), as an un-simplified length-1 scan folds its backward
    # `dynamic_index_in_dim/squeeze` (`[1, B, S, 4*D]` -> `[B, S, 4*D]`) into a cross-memory-space
    # `bitcast` (`S(0)` -> `S(5)`) rejected by the XLA HLO verifier.
    with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
      res = jax.lax.scan(scan_fn, hidden_states, (sub_params, sub_rests))
  else:
    res = jax.lax.scan(scan_fn, hidden_states, (sub_params, sub_rests))
  if return_aux:
    return res[0], [jax.tree.map(lambda x, c=c: x[c], res[1][k]) for c in range(num_cycles) for k in range(4)]
  return res[0]


def copy_unscanned_to_scanned(unscanned_decoder: Any, scanned_decoder: Any) -> None:
  """Copies weights from an unscanned (`scan_layers=False`) model/decoder into a scanned (`scan_layers=True`) one."""
  u, s = getattr(unscanned_decoder, "decoder", unscanned_decoder), getattr(scanned_decoder, "decoder", scanned_decoder)
  if u.scan_layers or not s.scan_layers:
    raise ValueError(f"Expected an unscanned and a scanned decoder, got scan_layers={u.scan_layers}, {s.scan_layers}.")
  if u.num_layers != 4 * s.num_cycles:
    raise ValueError(f"Layer count mismatch: unscanned={u.num_layers} layers, scanned={s.num_cycles} cycles of 4.")
  if u.ple_layer_id != s.ple_layer_id:
    raise ValueError(f"PLE layer mismatch: unscanned ple_layer_id={u.ple_layer_id}, scanned={s.ple_layer_id}.")
  if hasattr(unscanned_decoder, "lm_head") and hasattr(scanned_decoder, "lm_head"):
    nnx.update(scanned_decoder.lm_head, nnx.state(unscanned_decoder.lm_head))
  u_mtp, s_mtp = getattr(unscanned_decoder, "mtp", None), getattr(scanned_decoder, "mtp", None)
  if (u_mtp is not None) != (s_mtp is not None):
    raise ValueError(f"MTP presence mismatch: unscanned={u_mtp is not None}, scanned={s_mtp is not None}.")
  if u_mtp is not None:
    nnx.update(s_mtp, nnx.state(u_mtp))
  nnx.update(s, nnx.state(u))


# =================================================================================================
# Multi-token prediction (MTP) draft block
# =================================================================================================


class Qwen3_8MTPBlock(nnx.Module):
  """Single-layer MTP draft block fusing next-token embeddings and HC streams."""

  def __init__(self, config: Any, *, sharding: Sharding | None = None, rngs: nnx.Rngs):
    self.config, self.emb_dim = config, int(config.emb_dim)
    self.hc_count, self.hc_lowrank = int(config.hc_count), int(config.hc_lowrank)
    self.remat_policy = _resolve_qwen3_8_remat_policy(config)
    self.remat = self.remat_policy is not None
    sharding_obj = sharding or sharding_from_config(config)
    self.sharding = sharding_obj
    dt, wdt, eps = jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype), float(config.normalization_layer_epsilon)
    self.pre_fc_norm_embedding = Qwen3_8RMSNorm(self.emb_dim, eps=eps, sharding=sharding_obj, rngs=rngs)
    self.pre_fc_norm_hidden = Qwen3_8RMSNorm(self.hc_count * self.emb_dim, eps=eps, sharding=sharding_obj, rngs=rngs)
    l_kw = {"dtype": dt, "param_dtype": wdt, "rngs": rngs}
    self.fc_embedding = _linear(self.emb_dim, self.emb_dim, sharding_obj, "fc_embedding", ("embed", None), **l_kw)
    self.fc_hidden = _linear(self.emb_dim, self.emb_dim, sharding_obj, "fc_hidden", ("embed", None), **l_kw)
    cyc = int(config.inhomogeneous_layer_cycle_interval)
    self.layers_0 = Qwen3_8DecoderLayer(config, layer_idx=cyc - 1, include_ple=False, sharding=sharding_obj, rngs=rngs)
    self.hyper_connection_mixer = Qwen3_8GatedResidual(
        emb_dim=self.emb_dim,
        hc_count=self.hc_count,
        hc_lowrank=self.hc_lowrank,
        use_combine=False,
        eps=eps,
        dtype=dt,
        weight_dtype=wdt,
        sharding=sharding_obj,
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
    # Applied with the streams folded into the sequence, `[B, S * hc_count, E]`: on the 4-D form the partitioner
    # returns a wrong `fc_hidden` kernel gradient when the batch and the sequence are sharded on different mesh axes
    # (`fsdp` x `context`; seen with JAX 0.11.1 and 0.12.1).
    h_proj = self.fc_hidden(h_norm.reshape(h_norm.shape[0], -1, self.emb_dim)).reshape(h_norm.shape)
    h_mtp_in = (h_proj + jnp.expand_dims(e_proj, axis=-2)).reshape(hc_hidden_states.shape)
    eff_policy = self.remat_policy if self.remat else None
    if self.remat and eff_policy is None:
      eff_policy = cp.nothing_saveable
    out = _run_qwen3_8_sublayer(
        self.layers_0,
        h_mtp_in,
        eff_policy=eff_policy,
        return_aux=return_aux,
        positions=positions,
        conv_mask=conv_mask,
        segment_ids=segment_ids,
    )
    h_layer, aux = out if return_aux else (out, None)
    h_out = self.hyper_connection_mixer.mix_input(h_layer)
    return (h_out, aux) if return_aux else h_out

  def set_remat(self, remat: Any) -> None:
    """Updates the MTP block's activation checkpointing policy in place."""
    self.remat_policy = _coerce_remat_callable(remat)
    self.remat = self.remat_policy is not None


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


def _compute_tiled_cross_entropy(
    model: Any,
    hidden_states: jax.Array,
    targets: jax.Array,
    mask: jax.Array | None = None,
    *,
    num_vocab_tiling: int = 1,
) -> jax.Array:
  """Computes cross-entropy loss tiled across tokens to avoid materializing full `[B, S, V]` logits."""
  if num_vocab_tiling <= 1:
    logits = model.logits_from_hidden_states_for_vocab_tiling(hidden_states)
    return cross_entropy_loss(logits, targets, mask=mask)
  if hidden_states.size == 0 or targets.size == 0:
    raise ValueError("hidden_states and targets must be non-empty.")
  if hidden_states.shape[:-1] != targets.shape:
    raise ValueError(f"hidden_states prefix shape {hidden_states.shape[:-1]} must match targets shape {targets.shape}.")
  if mask is not None and mask.shape != targets.shape:
    raise ValueError(f"mask shape {mask.shape} must match targets shape {targets.shape}.")

  b, s, emb_dim = hidden_states.shape
  mesh = jax.sharding.get_abstract_mesh()
  length_axes = get_norm_length_mesh_axes(model.sharding)
  length_shards = math.prod(int(mesh.shape[ax]) for ax in length_axes) if length_axes else 1
  if s % length_shards != 0:
    length_shards = 1
  s_local = s // length_shards
  m_full = jnp.ones_like(targets, dtype=jnp.float32) if mask is None else mask.astype(jnp.float32)
  if s_local % num_vocab_tiling == 0:
    tile_s_local = s_local // num_vocab_tiling
    tile_s = length_shards * tile_s_local
    h_tiles = jnp.moveaxis(
        hidden_states.reshape(b, length_shards, num_vocab_tiling, tile_s_local, emb_dim), 2, 0
    ).reshape(num_vocab_tiling, b, tile_s, emb_dim)
    t_tiles = jnp.moveaxis(targets.reshape(b, length_shards, num_vocab_tiling, tile_s_local), 2, 0).reshape(
        num_vocab_tiling, b, tile_s
    )
    m_tiles = jnp.moveaxis(m_full.reshape(b, length_shards, num_vocab_tiling, tile_s_local), 2, 0).reshape(
        num_vocab_tiling, b, tile_s
    )
  elif (b * s) % num_vocab_tiling == 0:
    tile_tokens = (b * s) // num_vocab_tiling
    h_tiles = hidden_states.reshape(num_vocab_tiling, tile_tokens, 1, emb_dim)
    t_tiles = targets.reshape(num_vocab_tiling, tile_tokens, 1)
    m_tiles = m_full.reshape(num_vocab_tiling, tile_tokens, 1)
  else:
    raise ValueError(
        f"Token count ({b} * {s} = {b * s}) or per-shard sequence ({s_local}) must be divisible by "
        f"num_vocab_tiling={num_vocab_tiling}."
    )

  total_weights = jnp.sum(m_tiles).astype(jnp.float32) if mask is not None else jnp.array(float(b * s), dtype=jnp.float32)

  @jax.checkpoint
  def _tile_loss_sum(h_tile: jax.Array, t_tile: jax.Array, m_tile: jax.Array) -> jax.Array:
    h_tile = shard_batch(h_tile, model.sharding, "hidden_states")
    tile_logits = model.logits_from_hidden_states_for_vocab_tiling(h_tile).astype(jnp.float32)
    log_probs = jax.nn.log_softmax(tile_logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, jnp.expand_dims(t_tile, -1), axis=-1).squeeze(-1)
    return jnp.sum(nll * m_tile).astype(jnp.float32)

  def _scan_step(acc: jax.Array, xs: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, None]:
    h_tile, t_tile, m_tile = xs
    return acc + _tile_loss_sum(h_tile, t_tile, m_tile), None

  loss_sum, _ = jax.lax.scan(_scan_step, jnp.zeros((), dtype=jnp.float32), (h_tiles, t_tiles, m_tiles))
  return (loss_sum / jnp.maximum(total_weights, 1.0)).astype(jnp.float32)


def router_load_balancing_loss(
    aux_list: Any,
    num_experts: int,
    mask: jax.Array | None = None,
    mtp_mask: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
  """Computes MoE router load balancing auxiliary loss across decoder layers."""
  if aux_list is None:
    raise ValueError("aux_list cannot be None.")
  eff_mask: jax.Array | None = None
  if mask is not None:
    eff_mask = mask != 0
  if segment_ids is not None:
    seg_valid = segment_ids != 0
    eff_mask = seg_valid if eff_mask is None else (eff_mask & seg_valid)
  eff_mtp_mask = (mtp_mask != 0) if mtp_mask is not None else eff_mask

  if isinstance(aux_list, dict):
    if "layer_aux" not in aux_list:
      raise ValueError("aux_list dict must contain 'layer_aux'.")
    layer_items = [(layer, eff_mask) for layer in aux_list["layer_aux"]]
    if aux_list.get("mtp_aux") is not None:
      layer_items.append((aux_list["mtp_aux"], eff_mtp_mask))
  elif isinstance(aux_list, (list, tuple)):
    layer_items = [(layer, eff_mask) for layer in aux_list]
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
      m = (layer_mask != 0).astype(jnp.float32)
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
  num_vocab_tiling = int(getattr(model.config, "num_vocab_tiling", 1))
  use_tiled_ce = num_vocab_tiling > 1
  out = model(
      input_tokens,
      positions,
      segment_ids,
      return_aux=need_aux,
      return_mtp=run_mtp,
      compute_logits=not use_tiled_ce,
      **kwargs,
  )
  primary_out = out[0] if (run_mtp or need_aux) else out
  aux = out[1] if (run_mtp or need_aux) else None
  mtp_out = aux["mtp_logits"] if (run_mtp and need_aux) else (out[1] if run_mtp else None)

  eff_target_mask = target_mask
  if segment_ids is not None:
    seg_valid = (segment_ids != 0).astype(jnp.float32)
    eff_target_mask = seg_valid if eff_target_mask is None else (eff_target_mask.astype(jnp.float32) * seg_valid)
  if use_tiled_ce:
    primary_loss = _compute_tiled_cross_entropy(
        model, primary_out, target_tokens, mask=eff_target_mask, num_vocab_tiling=num_vocab_tiling
    )
  else:
    primary_loss = cross_entropy_loss(primary_out, target_tokens, mask=eff_target_mask)

  mtp_aux_mask = None
  mtp_loss = None
  if run_mtp and mtp_out is not None:
    if mtp_out.shape[1] <= 1:
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
      mtp_aux_mask = jnp.pad(m, ((0, 0), (0, 1)))
      if use_tiled_ce:
        mtp_targets = jnp.pad(target_tokens[:, 1:], ((0, 0), (0, 1)))
        mtp_loss = _compute_tiled_cross_entropy(
            model, mtp_out, mtp_targets, mask=mtp_aux_mask, num_vocab_tiling=num_vocab_tiling
        )
      else:
        mtp_loss = cross_entropy_loss(mtp_out[:, :-1, :], target_tokens[:, 1:], mask=m)

  router_loss = (
      router_load_balancing_loss(
          aux,
          int(model.config.num_experts),
          mask=target_mask,
          mtp_mask=mtp_aux_mask,
          segment_ids=segment_ids,
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

  def __init__(self, config: Any, *, sharding: Sharding | None = None, rngs: nnx.Rngs):
    c, dt, wdt = config, jnp.dtype(config.dtype), jnp.dtype(config.weight_dtype)
    self.emb_dim, self.vocab_size, self.eos_token_id = int(c.emb_dim), int(c.vocab_size), int(c.ple_eos_token_id)
    self.num_layers, self.hc_count, self.scan_layers = int(c.num_decoder_layers), int(c.hc_count), bool(c.scan_layers)
    self.remat_policy = _resolve_qwen3_8_remat_policy(c)
    self.remat = self.remat_policy is not None
    self.dtype, self.weight_dtype = dt, wdt
    self.sharding = sharding or sharding_from_config(config)
    table_shape = (self.vocab_size, self.emb_dim)
    self.token_embedder = nnx.Embed(
        *table_shape,
        dtype=dt,
        param_dtype=wdt,
        embedding_metadata=_param_kwargs(self.sharding, "token_embedder", ("vocab", "embed"), table_shape),
        rngs=rngs,
    )
    self.ple_layer_id = int(c.ple_layer_id) if bool(c.use_ple_layer) else None
    if int(config.inhomogeneous_layer_cycle_interval) != 4:
      raise ValueError(
          f"Scanned cycles require inhomogeneous_layer_cycle_interval == 4, got "
          f"{config.inhomogeneous_layer_cycle_interval}."
      )
    if self.num_layers < 4 or self.num_layers % 4 != 0:
      raise ValueError(f"Scanned cycles require num_decoder_layers to be a multiple of 4, got {self.num_layers}.")
    if bool(config.use_ple_layer) and int(config.ple_layer_id) != 1:
      raise ValueError(f"Scanned cycles only support ple_layer_id == 1, got {config.ple_layer_id}.")
    self.num_cycles = self.num_layers // 4
    self.num_stacked_cycles = self.num_cycles - 1
    self.first_cycle = Qwen3_8CycleBlock(config, include_ple=True, sharding=self.sharding, rngs=rngs)
    self.cycles = (
        create_scanned_cycles(config, self.num_stacked_cycles, sharding=self.sharding, rngs=rngs)
        if self.num_stacked_cycles > 0
        else None
    )
    self.hyper_connection_mixer = Qwen3_8GatedResidual(
        emb_dim=self.emb_dim,
        hc_count=self.hc_count,
        hc_lowrank=int(config.hc_lowrank),
        use_combine=False,
        eps=float(config.normalization_layer_epsilon),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        sharding=self.sharding,
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
    tok_emb = shard_batch(self.token_embedder(tok), self.sharding, "token_embeds")
    h, aux_list = (
        init_hyper_connection_streams(tok_emb, hc_count=self.hc_count, sharding=self.sharding),
        [],
    )
    eff_policy = self.remat_policy if self.remat else None
    if self.remat and eff_policy is None:
      eff_policy = cp.nothing_saveable
    first_out = run_single_cycle(
        self.first_cycle,
        h,
        eff_policy=eff_policy,
        return_aux=return_aux,
        positions=pos,
        conv_mask=cmask,
        ple_input_ids=pids,
        segment_ids=decoder_segment_ids,
        sharding=self.sharding,
    )
    h, aux_list = first_out if return_aux else (first_out, [])
    if self.cycles is not None:
      res = apply_scanned_cycles(
          self.cycles,
          h,
          num_cycles=self.num_stacked_cycles,
          scan_layers=self.scan_layers,
          positions=pos,
          conv_mask=cmask,
          segment_ids=decoder_segment_ids,
          return_aux=return_aux,
          remat=self.remat,
          remat_policy=eff_policy,
          sharding=self.sharding,
      )
      if return_aux:
        h, cycle_aux = res
        aux_list = aux_list + cycle_aux
      else:
        h = res
    h_4stream, h_mixed = h, shard_batch(self.hyper_connection_mixer.mix_input(h), self.sharding, "mixed_streams")
    if return_pre_mixer:
      return (h_mixed, h_4stream, aux_list) if return_aux else (h_mixed, h_4stream)
    return (h_mixed, aux_list) if return_aux else h_mixed

  def set_remat(self, remat: Any) -> None:
    """Updates the decoder's per-sublayer activation checkpointing policy in place."""
    self.remat_policy = _coerce_remat_callable(remat)
    self.remat = self.remat_policy is not None


class Qwen3_8FlashNextModel(nnx.Module):
  """Top-level Qwen3.8-Flash-Next model with decoder and lm_head."""

  def __init__(
      self,
      config: Any,
      mesh: Any = None,
      *,
      quant_mode: str = "train",
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.config, self.mesh, self.quant_mode = config, mesh, quant_mode
    self.sharding = sharding or sharding_from_config(config)
    self.decoder = Qwen3_8Decoder(config, sharding=self.sharding, rngs=rngs)
    self.lm_head = _linear(
        self.decoder.emb_dim,
        self.decoder.vocab_size,
        self.sharding,
        "lm_head",
        ("embed", "vocab"),
        dtype=self.decoder.dtype,
        param_dtype=self.decoder.weight_dtype,
        rngs=rngs,
    )
    mtp_layers = int(getattr(config, "mtp_num_layers", 0))
    if mtp_layers not in (0, 1):
      raise ValueError(f"mtp_num_layers must be 0 or 1, got {mtp_layers}.")
    self.mtp = Qwen3_8MTPBlock(config, sharding=self.sharding, rngs=rngs) if mtp_layers > 0 else None

  def logits_from_hidden_states_for_vocab_tiling(
      self,
      hidden_states: jax.Array,
      deterministic: bool = True,
      model_mode: str = "train",
      **kwargs: Any,
  ) -> jax.Array:
    """Projects post-mixer hidden states to output logits for vocabulary tiling."""
    del deterministic, model_mode, kwargs
    logits = self.lm_head(hidden_states)
    return shard_batch(logits, self.sharding, "logits") if logits.ndim == 3 else logits

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
      compute_logits: bool = True,
      **kwargs: Any,
  ) -> Any:
    """Forward pass computing language model logits and optional aux outputs."""
    del enable_dropout
    has_target_tokens = "decoder_target_tokens" in kwargs
    del kwargs
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
          shard_batch(self.decoder.token_embedder(next_tok), self.sharding, "next_token_embeds"),
          positions=pos,
          conv_mask=cmask,
          segment_ids=decoder_segment_ids,
          return_aux=return_aux,
      )
      mtp_h, mtp_aux = mtp_out if return_aux else (mtp_out, None)
      mtp_logits = shard_batch(self.lm_head(mtp_h), self.sharding, "mtp_logits") if compute_logits else mtp_h
      logits = shard_batch(self.lm_head(h_mixed), self.sharding, "logits") if compute_logits else h_mixed
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
    if (
        int(getattr(self.config, "num_vocab_tiling", 1)) > 1
        and has_target_tokens
        and compute_logits
        and not return_hidden_states
        and not return_aux
    ):
      self.decoder.sow(nnx.Intermediate, "hidden_states", out)
      return None
    if return_hidden_states or return_aux:
      first = shard_batch(self.lm_head(out[0]), self.sharding, "logits") if compute_logits else out[0]
      return (first, *out[1:])
    return shard_batch(self.lm_head(out), self.sharding, "logits") if compute_logits else out


def create_qwen3_8_flash_next_model(
    config: Any,
    mesh: Any = None,
    *,
    rngs: nnx.Rngs | None = None,
    quant_mode_str: str = "train",
    sharding: Sharding | None = None,
    **_: Any,
) -> Qwen3_8FlashNextModel:
  """Factory function creating a Qwen3_8FlashNextModel instance."""
  return Qwen3_8FlashNextModel(
      config=config,
      mesh=mesh,
      quant_mode=quant_mode_str,
      sharding=sharding or sharding_from_config(config),
      rngs=rngs or nnx.Rngs(params=0),
  )
