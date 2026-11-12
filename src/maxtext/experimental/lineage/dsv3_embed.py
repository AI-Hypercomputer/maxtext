# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token embedding, final norm and LM head of DSv3.

Faithful ports of the MaxText decoder's non-layer compute
(`Embed.__call__`, `Embed.embed_single_token`, `attend_on_embedding`,
`RMSNorm` and the `logits_dense` `DenseGeneral`), expressed as pure functions
over explicit weight pytrees.
"""

from collections.abc import Mapping
import dataclasses
import math
from typing import Any, Generic, TypeVar

import jax
import jax.experimental.compute_on
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops

ArrayType = dsv3_types.ArrayType
ShardingType = dsv3_types.ShardingType
Initializer = dsv3_types.Initializer
T = TypeVar("T", ArrayType, ShardingType, dsv3_types.AxisNameType)

# MaxText's `Embed` default (`models.py`): normal(stddev=1.0).
default_embedding_init = jax.nn.initializers.normal(stddev=1.0)
# MaxText's `DenseGeneral` default: `nd_dense_init(1.0, "fan_in",
# "truncated_normal")`, which is `dsv3_types.default_kernel_init`.
default_head_kernel_init = dsv3_types.default_kernel_init

# Whether `embed_tokens_collected` / `collect_head_kernel` reduce their f32
# weight gradients on SparseCore 1 (see `set_grad_sparse_core`).
_EMBED_GRAD_SC = False
_HEAD_GRAD_SC = False
# Whether those reductions are `compute_on` SparseCore blocks (False runs the
# same collectives on the default device, e.g. in CPU tests).
_GRAD_SC_COMPUTE_ON = True
# SparseCore of the all-reduce stage, or None for the TensorCore.
_GRAD_SC_ALL_REDUCE_CORE: int | None = None
# Reduce with `ops.reduce_along_axis` instead of the local shard_maps.
_GRAD_SC_USE_OPS_REDUCE = True
# Reduce-scatter stacked `[emb / n, vocab]` shards (see `_collect_with_sc_grad`).
_GRAD_SC_STACKED = False


def set_grad_sparse_core(embed: bool, head: bool, *, compute_on: bool = True) -> None:
  """Reduces the f32 embedding table / LM head kernel gradients on SparseCore.

  By default the cotangent of the all-gathered (`reduced`) table or kernel is
  reduce-scattered (and all-reduced over the other mesh axes) by one
  synchronous TensorCore `all-reduce-scatter` fusion, which XLA schedules at
  the end of the step. With this, the reduction is an async SparseCore 1
  collective (`compute_on`), so the LM head kernel's, ready at the start of the
  backward pass, overlaps the decoder backward.

  Args:
    embed: Reduce the embedding table gradient on SparseCore.
    head: Reduce the LM head kernel gradient on SparseCore.
    compute_on: Place the reductions on SparseCore (False keeps the same
      collectives on the default device, for tests off TPU).
  """
  global _EMBED_GRAD_SC, _HEAD_GRAD_SC, _GRAD_SC_COMPUTE_ON
  _EMBED_GRAD_SC = embed
  _HEAD_GRAD_SC = head
  _GRAD_SC_COMPUTE_ON = compute_on


def _collect_with_sc_grad(
    weight: jax.Array,
    *,
    emb_dim: int,
    dtype: jax.typing.DTypeLike,
    grad_dtype: jax.typing.DTypeLike,
) -> jax.Array:
  """All-gathers `weight` along its embedding dim; reduces its grad on SC 1.

  The forward pass casts the local shard to `dtype`, all-gathers whole
  `[emb / shards, vocab]` shards and returns the full weight in `grad_dtype`
  (holding `dtype` values), `reduced` over every mesh axis. The backward pass
  reduce-scatters the (unreduced, full) `grad_dtype` cotangent onto the shards
  and all-reduces it over the other mesh axes, in `grad_dtype`, on SparseCore 1.

  Args:
    weight: `[vocab, emb]` (emb_dim=1) or `[emb, vocab]` (emb_dim=0) weight,
      sharded on its embedding dim only.
    emb_dim: The embedding dim of `weight`.
    dtype: Dtype of the gathered values.
    grad_dtype: Dtype of the returned weight and of the gradient reduction.

  Returns:
    The full weight in `grad_dtype`, `reduced` over every mesh axis.
  """
  spec = tuple(jax.typeof(weight).sharding.spec)
  spec += (None,) * (2 - len(spec))
  if spec[1 - emb_dim] is not None:
    raise ValueError(f"The vocab dim must be replicated: {weight}")
  emb_axes = spec[emb_dim]
  emb_axes = (emb_axes,) if isinstance(emb_axes, str) else tuple(emb_axes or ())
  mesh = jax.typeof(weight).sharding.mesh
  other_axes = tuple(ax for ax in mesh.axis_names if ax not in emb_axes)
  all_axes = frozenset(emb_axes + other_axes)
  full_spec = jax.sharding.PartitionSpec(None, None, reduced=all_axes)
  grad_spec = jax.sharding.PartitionSpec(*spec)
  num_shards = math.prod(mesh.shape[a] for a in emb_axes)

  def _gather(w_local):
    # [vocab, emb / n] -> [emb / n, vocab] -> [n, emb / n, vocab] -> [emb, vocab]
    w_local = w_local.astype(dtype)
    if emb_dim == 1:
      w_local = w_local.T
    for ax in other_axes:
      w_local = jax.lax.pcast(w_local, ax, to="reduced")
    if emb_axes:
      w_local = jax.lax.all_gather(w_local, emb_axes, axis=0, to="reduced")
      w_local = w_local.reshape(-1, w_local.shape[-1])
    w_local = w_local.astype(grad_dtype)
    return w_local.T if emb_dim == 1 else w_local

  def _scatter(ct):
    # [vocab, emb] / [emb, vocab] -> [n, emb / n, vocab] -> [emb / n, vocab]
    ct = ct.astype(grad_dtype)
    if emb_dim == 1:
      ct = ct.T
    if emb_axes:
      ct = ct.reshape(num_shards, -1, ct.shape[-1])
      ct = jax.lax.psum_scatter(ct, emb_axes, scatter_dimension=0, tiled=True)
      ct = ct.reshape(-1, ct.shape[-1])
    return ct.T if emb_dim == 1 else ct

  def _all_reduce(ct):
    for ax in reversed(other_axes):
      ct = jax.lax.psum(ct, ax)
    return ct

  def _on_sc(fn, core):
    if not _GRAD_SC_COMPUTE_ON or core is None:
      return fn
    return jax.experimental.compute_on.compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [core]}},
    )(fn)

  def _fwd_impl(w):
    return jax.shard_map(_gather, mesh=mesh, in_specs=(grad_spec,), out_specs=full_spec)(w)

  @jax.custom_vjp
  def _f(w):
    return _fwd_impl(w)

  def _f_fwd(w):
    # `w` is only read for its sharding (the `like` of the reduction).
    return _fwd_impl(w), w

  def _f_bwd(w_res, ct):
    if _GRAD_SC_USE_OPS_REDUCE:
      # The layer weights' proven path (`ops.reduce_along_axis`): the
      # reduce-scatter along the shard axes on SparseCore 1, then the
      # all-reduce along the other axes.
      axis_names = emb_axes + other_axes
      sc_kwargs = (
          dict(
              compute_type="tpu_sparsecore",
              compiler_options={"sparse_core_config": {"core_ids": [1]}},
          )
          if _GRAD_SC_COMPUTE_ON
          else {}
      )
      # Reduce-scatter whole `[emb / n, vocab]` shards stacked on a new
      # leading dim: scattering the (minor-laid-out) embedding dim in place is
      # legalized into an all-reduce of the full cotangent.
      g = ct.astype(grad_dtype)
      if not _GRAD_SC_STACKED:
        # TODO: the stacked reduce-scatter crashes the TPU latency hiding
        # scheduler (`start != nullptr`); this is legalized into a SparseCore
        # all-reduce of the full cotangent instead.
        if emb_axes:
          g = ops.reduce_along_axis(g, axis_names, {}, w_res, axes=frozenset(emb_axes), **sc_kwargs)
        if other_axes:
          g = ops.reduce_along_axis(g, axis_names, {}, w_res, axes=frozenset(other_axes))
        return (g.astype(weight.dtype),)
      like = w_res
      if emb_dim == 1:
        g, like = g.T, w_res.T
      stacked = (num_shards, g.shape[0] // num_shards, g.shape[1])
      g = g.reshape(stacked)
      like = like.reshape(
          stacked,
          out_sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(emb_axes or None, None, None)),
      )
      if emb_axes:
        g = ops.reduce_along_axis(g, axis_names, {}, like, axes=frozenset(emb_axes), **sc_kwargs)
      if other_axes:
        g = ops.reduce_along_axis(g, axis_names, {}, like, axes=frozenset(other_axes))
      g = g.reshape(stacked[0] * stacked[1], stacked[2])
      if emb_dim == 1:
        g = g.T
      return (jax.sharding.reshard(g, grad_spec).astype(weight.dtype),)
    # Like `ops.reduce_along_axis` for the layer weights: the reduce-scatter
    # along the shard axes on SparseCore 1, then the all-reduce along the other
    # axes (on 1 / n of the data) on SparseCore 0, as separate blocks.
    ct_spec = jax.sharding.PartitionSpec(None, None, unreduced=all_axes)
    mid_parts = [None, None]
    mid_parts[emb_dim] = emb_axes or None
    mid_spec = jax.sharding.PartitionSpec(*mid_parts, unreduced=frozenset(other_axes))
    g = ct
    if emb_axes:
      g = jax.shard_map(_on_sc(_scatter, 1), mesh=mesh, in_specs=(ct_spec,), out_specs=mid_spec)(g)
    else:
      mid_spec = ct_spec
    if other_axes:
      g = jax.shard_map(
          _on_sc(_all_reduce, _GRAD_SC_ALL_REDUCE_CORE),
          mesh=mesh,
          in_specs=(mid_spec,),
          out_specs=grad_spec,
      )(g)
    return (g.astype(weight.dtype),)

  _f.defvjp(_f_fwd, _f_bwd)
  return _f(weight)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3EmbedWeightsPytree(Generic[T]):
  """Pytree to hold the DSv3 token embedding weights or weight shardings."""

  table: T | None = None  # (vocab, emb)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DSv3HeadWeightsPytree(Generic[T]):
  """Pytree to hold the DSv3 final norm / LM head weights or shardings.

  Attributes:
    final_norm_scale: Final RMSNorm scale `(emb,)`.
    kernel: LM head kernel `(emb, vocab)`, or None when the head is tied to the
      embedding table (`logits_via_embedding`).
  """

  final_norm_scale: T | None = None
  kernel: T | None = None


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def embed_tokens(
    tokens: jt.Int[jax.Array, "*batch"],
    table: jt.Num[jax.Array, "V D"],
    *,
    dtype: jax.typing.DTypeLike,
    iota_embed: bool = False,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jt.Num[jax.Array, "*batch D"]:
  """Looks up token embeddings, as MaxText's `Embed.__call__`.

  Args:
    tokens: Integer token ids; every dim is a batch dim.
    table: Embedding table `[vocab, emb]`.
    dtype: Compute dtype; the whole table is cast to it before the lookup.
    iota_embed: Look the rows up with a one-hot matmul instead of a gather
      (`use_iota_embed`).
    out_sharding: Optional sharding of the output.

  Returns:
    The embeddings `[*batch, emb]` in `dtype`.
  """
  embedding = jnp.asarray(table, dtype)
  if iota_embed:
    iota = jax.lax.iota(jnp.int32, embedding.shape[0])
    one_hot = jnp.array(tokens[..., jnp.newaxis] == iota, dtype=dtype)
    return jnp.dot(one_hot, embedding, out_sharding=out_sharding)
  return embedding.at[tokens].get(out_sharding=out_sharding)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def embed_tokens_collected(
    tokens: jt.Int[jax.Array, "*batch"],
    table: jt.Num[jax.Array, "V D"],
    *,
    dtype: jax.typing.DTypeLike,
    iota_embed: bool = False,
    grad_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "*batch D"]:
  """`embed_tokens` that all-gathers the sharded table explicitly.

  The SPMD partitioner lowers a lookup into a table sharded on its embedding
  dim as an all-gather of the table, and its transpose as an all-reduce of the
  full `[vocab, emb]` cotangent of which each device keeps one slice. Here the
  table is cast to `dtype` and all-gathered inside a `shard_map`, so the
  backward pass reduce-scatters the cotangent onto the table shards instead.

  Args:
    tokens: Integer token ids; every dim is a batch dim.
    table: Embedding table `[vocab, emb]`, sharded on its embedding dim only.
    dtype: Compute dtype; the table is cast to it before the all-gather.
    iota_embed: Look the rows up with a one-hot matmul instead of a gather
      (`use_iota_embed`).
    grad_dtype: If set (f32), the table's gradient is accumulated and
      reduce-scattered in it (`ops.f32_grad`) instead of `dtype`.

  Returns:
    The embeddings `[*batch, emb]` in `dtype`, sharded like `tokens` with a
    replicated embedding dim.
  """
  table_spec = jax.typeof(table).sharding.spec
  if table_spec and table_spec[0] is not None:
    raise ValueError(f"The vocab dim of the table must be replicated: {table}")
  emb_axes = table_spec[1] if len(table_spec) > 1 else None
  tokens_spec = tuple(jax.typeof(tokens).sharding.spec)
  tokens_spec += (None,) * (tokens.ndim - len(tokens_spec))

  def _gathered_lookup(table_local, tokens_local):
    if emb_axes is not None:
      # Stack whole `[emb / shards, vocab]` shards on a new leading dim: the
      # transpose then reduce-scatters whole shards. A tiled gather along the
      # (minor, not tile-aligned) embedding dim is legalized into an all-reduce.
      shards = jax.lax.all_gather(table_local.T, emb_axes, axis=0)
      table_local = shards.reshape(-1, shards.shape[-1]).T
    return embed_tokens(
        tokens_local,
        table_local,
        dtype=table_local.dtype,
        iota_embed=iota_embed,
    )

  def _lookup(table, tokens):
    # The table is cast before the call (inside it, the all-gather's transpose
    # would reduce-scatter the gradient in `dtype`).
    return jax.shard_map(
        _gathered_lookup,
        mesh=jax.typeof(table).sharding.mesh,
        in_specs=(table_spec, jax.sharding.PartitionSpec(*tokens_spec)),
        out_specs=jax.sharding.PartitionSpec(*tokens_spec, None),
    )(table, tokens)

  if grad_dtype is not None:
    if _EMBED_GRAD_SC:
      # The full table in `grad_dtype`, so the lookup's transpose sums the
      # token cotangents in it; its reduction runs on SparseCore 1.
      full = _collect_with_sc_grad(table, emb_dim=1, dtype=dtype, grad_dtype=grad_dtype)
      out_sharding = jax.sharding.NamedSharding(
          jax.typeof(table).sharding.mesh,
          jax.sharding.PartitionSpec(*tokens_spec, None),
      )
      return embed_tokens(
          tokens,
          full,
          dtype=grad_dtype,
          iota_embed=iota_embed,
          out_sharding=out_sharding,
      ).astype(dtype)
    # Outside the `shard_map`: a `custom_vjp` inside it would make the output
    # varying over the table's axes.
    return ops.f32_grad(_lookup, table, tokens, compute_dtype=dtype)
  return _lookup(jnp.asarray(table, dtype), tokens)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def embed_single_token(
    table: jt.Num[jax.Array, "V D"],
    token_id: int,
    *,
    dtype: jax.typing.DTypeLike,
    iota_embed: bool = False,
) -> jt.Num[jax.Array, "D"]:
  """Returns one row of the embedding table, bitwise equal to `embed_tokens`.

  Args:
    table: Embedding table `[vocab, emb]`.
    token_id: Row to return.
    dtype: Compute dtype.
    iota_embed: Mirror the `use_iota_embed` lookup, which turns negative zeros
      into positive zeros.

  Returns:
    The embedding `[emb]` in `dtype`.
  """
  row = jnp.asarray(table, dtype)[token_id]
  if iota_embed:
    row = jnp.where(row == 0, jnp.zeros_like(row), row)
  return row


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def final_norm(
    x: jt.Num[jax.Array, "*batch D"],
    scale: jt.Num[jax.Array, "D"],
    *,
    epsilon: float,
    grad_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "*batch D"]:
  """Applies the final RMSNorm, as MaxText's `RMSNorm` without scale offset.

  `x` must already be in the compute dtype: like MaxText, the normalized
  activations and the scale are both rounded to that dtype before the scale
  multiply.

  Args:
    x: Hidden states `[*batch, emb]` in the compute dtype.
    scale: Norm scale `[emb]` in any dtype.
    epsilon: Norm epsilon.
    grad_dtype: If set (f32), the backward pass (including the scale gradient's
      sum over the tokens) runs in it (`ops.f32_grad`).

  Returns:
    The normalized hidden states in the compute dtype.
  """
  if grad_dtype is not None:
    return ops.f32_grad(
        lambda xs: ops.rms_norm(xs[0], jnp.asarray(xs[1], xs[0].dtype), epsilon=epsilon),
        (x, scale),
        compute_dtype=x.dtype,
    )
  return ops.rms_norm(x, jnp.asarray(scale, x.dtype), epsilon=epsilon)


def _cotangent_sharding(x: jax.Array) -> jax.sharding.NamedSharding:
  """The sharding of x's cotangent (unreduced where x is reduced)."""
  sharding = jax.typeof(x).sharding
  return sharding.update(spec=sharding.spec.to_ct_spec())


def _head_dot_with_wgrad_dtype(
    x: jax.Array,
    kernel: jax.Array,
    *,
    dot_dtype: jax.typing.DTypeLike,
    grad_dtype: jax.typing.DTypeLike,
    precision: jax.lax.Precision,
    out_sharding: jax.sharding.NamedSharding | None,
) -> jax.Array:
  """`x @ kernel` in `dot_dtype` whose kernel gradient is computed in f32.

  The forward pass and the input gradient are those of a `dot_dtype` matmul.
  The kernel gradient contracts the tokens with `preferred_element_type=
  grad_dtype`, so the per-device sum over the tokens is not rounded to
  `dot_dtype`, and is returned in the kernel's dtype.

  Args:
    x: Hidden states `[*batch, emb]`.
    kernel: Head kernel `[emb, vocab]`.
    dot_dtype: Dtype of the matmul operands.
    grad_dtype: Accumulation and output dtype of the kernel gradient matmul.
    precision: Matmul precision.
    out_sharding: Optional sharding of the logits.

  Returns:
    The logits `[*batch, vocab]` in `dot_dtype`.
  """
  batch_dims = tuple(range(x.ndim - 1))

  def _dot(x, kernel):
    return jax.lax.dot_general(
        x.astype(dot_dtype),
        kernel.astype(dot_dtype),
        (((x.ndim - 1,), (0,)), ((), ())),
        precision=precision,
        out_sharding=out_sharding,
    )

  @jax.custom_vjp
  def _f(x, kernel):
    return _dot(x, kernel)

  def _fwd(x, kernel):
    return _dot(x, kernel), (x, kernel)

  def _bwd(res, g):
    x, kernel = res
    g = g.astype(dot_dtype)
    x_c, kernel_c = x.astype(dot_dtype), kernel.astype(dot_dtype)
    grad_x = jax.lax.dot_general(
        g,
        kernel_c,
        (((g.ndim - 1,), (1,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
        out_sharding=_cotangent_sharding(x),
    ).astype(x.dtype)
    grad_kernel = jax.lax.dot_general(
        x_c,
        g,
        ((batch_dims, batch_dims), ((), ())),
        precision=precision,
        preferred_element_type=grad_dtype,
        out_sharding=_cotangent_sharding(kernel),
    ).astype(kernel.dtype)
    if _HEAD_GRAD_SC:
      # Computes the kernel gradient before the decoder backward pass (which
      # needs grad_x), so its SparseCore reduction overlaps it instead of
      # running after the step's last layer.
      grad_x, grad_kernel = jax.lax.optimization_barrier((grad_x, grad_kernel))
    return grad_x, grad_kernel

  _f.defvjp(_fwd, _bwd)
  return _f(x, kernel)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def lm_head(
    x: jt.Num[jax.Array, "*batch D"],
    kernel: jt.Num[jax.Array, "D V"],
    *,
    dtype: jax.typing.DTypeLike,
    dot_in_fp32: bool = False,
    cast_logits_to_fp32: bool = True,
    precision: str = "default",
    out_sharding: jax.sharding.NamedSharding | None = None,
    grad_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "*batch V"]:
  """Projects hidden states to logits, as MaxText's `logits_dense`.

  Args:
    x: Normalized hidden states `[*batch, emb]`.
    kernel: Head kernel `[emb, vocab]`.
    dtype: Compute dtype.
    dot_in_fp32: Run the matmul in fp32 (`logits_dot_in_fp32`).
    cast_logits_to_fp32: Cast the logits to fp32 (`cast_logits_to_fp32`).
    precision: `jax.lax.Precision` name of the matmul (`matmul_precision`).
    out_sharding: Optional sharding of the logits.
    grad_dtype: If set (f32), the kernel gradient matmul accumulates and outputs
      in it (from operands in the matmul dtype), and the kernel gradient is
      returned in the kernel's dtype (e.g. the f32 kernel of
      `collect_head_kernel(grad_dtype=...)`) without rounding to the matmul
      dtype.

  Returns:
    The logits `[*batch, vocab]`.
  """
  dot_dtype = jnp.float32 if dot_in_fp32 else dtype
  if grad_dtype is not None:
    logits = _head_dot_with_wgrad_dtype(
        x,
        kernel,
        dot_dtype=dot_dtype,
        grad_dtype=grad_dtype,
        precision=jax.lax.Precision(precision),
        out_sharding=out_sharding,
    )
  else:
    inputs = jnp.asarray(x, dot_dtype)
    weights = jnp.asarray(kernel, dot_dtype)
    logits = jax.lax.dot_general(
        inputs,
        weights,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=jax.lax.Precision(precision),
        out_sharding=out_sharding,
    )
  if cast_logits_to_fp32:
    logits = logits.astype(jnp.float32)
  return logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_head_kernel(
    kernel: jt.Num[jax.Array, "D V"],
    *,
    dtype: jax.typing.DTypeLike,
    grad_dtype: jax.typing.DTypeLike | None = None,
) -> jt.Num[jax.Array, "D V"]:
  """All-gathers the LM head kernel, sharded on its embedding dim, explicitly.

  Left to the SPMD partitioner, every use of the sharded kernel all-gathers it
  and all-reduces its full `[emb, vocab]` cotangent. Collecting it once here
  lets the main and MTP heads share one all-gather, and the backward pass sums
  their cotangents before a single reduce-scatter onto the kernel shards.

  Args:
    kernel: Head kernel `[emb, vocab]`, sharded on its embedding dim only.
    dtype: Dtype of the matmul; the kernel is cast to it before the gather.
    grad_dtype: If set (f32), the kernel is still gathered in `dtype` but
      returned in `grad_dtype` (holding `dtype` values), so that its gradient
      reaches the reduction unrounded (see `lm_head(grad_dtype=...)`), and the
      gradient is reduced in `grad_dtype` (`ops.f32_grad`).

  Returns:
    The kernel in `dtype` (or `grad_dtype`), replicated (`reduced`) over every
    mesh axis: the
    ones it was sharded on and the ones it was already replicated over (`dcn`,
    and any ICI axes its spec leaves out, e.g. `x` on 8x16x16).
  """
  spec = jax.typeof(kernel).sharding.spec
  if len(spec) > 1 and spec[1] is not None:
    raise ValueError(f"The vocab dim of the kernel must be replicated: {kernel}")
  emb_axes = spec[0] if spec else None
  emb_axes = (emb_axes,) if isinstance(emb_axes, str) else tuple(emb_axes or ())
  mesh = jax.typeof(kernel).sharding.mesh
  if not emb_axes and "dcn" not in mesh.axis_names:
    return jnp.asarray(kernel, dtype)
  # The logits contract the batch and sequence, which can be sharded over every
  # mesh axis, so the kernel must be `reduced` over all of them for its
  # cotangent to be summed over each.
  replicated_axes = tuple(ax for ax in mesh.axis_names if ax not in emb_axes)

  def _gather(kernel_local):
    for ax in replicated_axes:
      kernel_local = jax.lax.pcast(kernel_local, ax, to="reduced")
    if not emb_axes:
      return kernel_local
    # Stack whole `[emb / shards, vocab]` shards on a new leading dim, so the
    # transpose reduce-scatters whole shards: a tiled gather along the
    # embedding dim (not tile-aligned per shard) is legalized into an
    # all-reduce.
    return jax.lax.all_gather(kernel_local, emb_axes, axis=0, to="reduced")

  out_ndim = 2 if not emb_axes else 3

  def _collect(kernel):
    # The kernel is cast before the call (see `embed_tokens_collected`).
    collected = jax.shard_map(
        _gather,
        mesh=mesh,
        in_specs=(spec,),
        out_specs=jax.sharding.PartitionSpec(*(None,) * out_ndim, reduced=frozenset(emb_axes + replicated_axes)),
    )(kernel)
    return collected.reshape(kernel.shape)

  if grad_dtype is not None:
    if _HEAD_GRAD_SC:
      return _collect_with_sc_grad(kernel, emb_dim=0, dtype=dtype, grad_dtype=grad_dtype)
    return ops.f32_grad(lambda k: _collect(k).astype(grad_dtype), kernel, compute_dtype=dtype)
  return _collect(jnp.asarray(kernel, dtype))


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def tied_lm_head(
    x: jt.Num[jax.Array, "*batch D"],
    table: jt.Num[jax.Array, "V D"],
    *,
    dtype: jax.typing.DTypeLike,
    dot_in_fp32: bool = False,
    cast_logits_to_fp32: bool = True,
    normalize_logits: bool = True,
    soft_cap: float | None = None,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jt.Num[jax.Array, "*batch V"]:
  """Computes logits against the embedding table (`logits_via_embedding`).

  Mirrors MaxText's `attend_on_embedding` plus the `normalize_embedding_logits`
  / `final_logits_soft_cap` / `cast_logits_to_fp32` steps of
  `apply_output_head`.

  Args:
    x: Normalized hidden states `[*batch, emb]`.
    table: Embedding table `[vocab, emb]`.
    dtype: Compute dtype.
    dot_in_fp32: Accumulate the matmul in fp32 (`logits_dot_in_fp32`).
    cast_logits_to_fp32: Cast the logits to fp32 (`cast_logits_to_fp32`).
    normalize_logits: Divide the logits by `sqrt(emb)`
      (`normalize_embedding_logits`).
    soft_cap: Optional tanh soft cap of the logits (`final_logits_soft_cap`).
    out_sharding: Optional sharding of the logits.

  Returns:
    The logits `[*batch, vocab]`.
  """
  attend_dtype = jnp.float32 if dot_in_fp32 else dtype
  logits = jnp.dot(
      x,
      jnp.asarray(table, jnp.bfloat16).T,
      preferred_element_type=attend_dtype,
      out_sharding=out_sharding,
  )
  if normalize_logits:
    logits = logits / jnp.sqrt(x.shape[-1])
  if soft_cap:
    logits = jnp.tanh(logits / soft_cap) * soft_cap
  if cast_logits_to_fp32:
    logits = logits.astype(jnp.float32)
  return logits


def init_dsv3_embed_weights(
    rng: jax.Array,
    vocab_size: int,
    emb_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jax.typing.DTypeLike = jnp.float32,
    out_shardings: DSv3EmbedWeightsPytree[ShardingType] | None = None,
    embedding_init: Initializer = default_embedding_init,
) -> DSv3EmbedWeightsPytree[ArrayType]:
  """Initializes the token embedding table.

  Args:
    rng: A PRNG key.
    vocab_size: Vocabulary size.
    emb_dim: Embedding dimension.
    mesh: Physical mesh for logical sharding resolution.
    axis_mapping: Mapping from logical to physical mesh axes.
    weight_dtype: Dtype of the table.
    out_shardings: Optional output shardings.
    embedding_init: Table initializer.

  Returns:
    The initialized embedding weights.
  """
  shardings = _resolve(out_shardings, mesh, axis_mapping)
  shardings = shardings or DSv3EmbedWeightsPytree[ShardingType]()
  return DSv3EmbedWeightsPytree[ArrayType](
      table=embedding_init(
          rng,
          (vocab_size, emb_dim),
          weight_dtype,
          out_sharding=shardings.table,
      )
  )


def init_dsv3_head_weights(
    rng: jax.Array,
    vocab_size: int,
    emb_dim: int,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    weight_dtype: jax.typing.DTypeLike = jnp.float32,
    tied_head: bool = False,
    out_shardings: DSv3HeadWeightsPytree[ShardingType] | None = None,
    kernel_init: Initializer = default_head_kernel_init,
    scale_init: Initializer = dsv3_types.default_scale_init,
) -> DSv3HeadWeightsPytree[ArrayType]:
  """Initializes the final norm scale and (unless tied) the LM head kernel.

  Args:
    rng: A PRNG key.
    vocab_size: Vocabulary size.
    emb_dim: Embedding dimension.
    mesh: Physical mesh for logical sharding resolution.
    axis_mapping: Mapping from logical to physical mesh axes.
    weight_dtype: Dtype of the weights.
    tied_head: Skip the kernel (`logits_via_embedding`).
    out_shardings: Optional output shardings.
    kernel_init: Head kernel initializer.
    scale_init: Norm scale initializer.

  Returns:
    The initialized head weights.
  """
  shardings = _resolve(out_shardings, mesh, axis_mapping)
  shardings = shardings or DSv3HeadWeightsPytree[ShardingType]()
  scale_key, kernel_key = jax.random.split(rng)
  kernel = None
  if not tied_head:
    kernel = kernel_init(
        kernel_key,
        (emb_dim, vocab_size),
        weight_dtype,
        out_sharding=shardings.kernel,
    )
  return DSv3HeadWeightsPytree[ArrayType](
      final_norm_scale=scale_init(
          scale_key,
          (emb_dim,),
          weight_dtype,
          out_sharding=shardings.final_norm_scale,
      ),
      kernel=kernel,
  )


def _resolve(
    out_shardings: Any,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> Any:
  """Resolves logical PartitionSpec leaves to physical NamedShardings."""
  if out_shardings is None:
    return None

  def _named(s):
    if not isinstance(s, jax.sharding.PartitionSpec):
      return s
    return jax.sharding.NamedSharding(mesh, ops.physical_pspec(s, axis_mapping))

  return jax.tree.map(
      _named,
      out_shardings,
      is_leaf=lambda s: isinstance(s, jax.sharding.PartitionSpec),
  )
