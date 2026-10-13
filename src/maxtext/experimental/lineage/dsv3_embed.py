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
from typing import Any, Generic, TypeVar

import jax
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

  def _lookup(tokens_local, table_local):
    table_local = jnp.asarray(table_local, dtype)
    if emb_axes is not None:
      # Stack whole `[emb / shards, vocab]` shards on a new leading dim: the
      # transpose then reduce-scatters whole shards. A tiled gather along the
      # (minor, not tile-aligned) embedding dim is legalized into an all-reduce.
      shards = jax.lax.all_gather(table_local.T, emb_axes, axis=0)
      table_local = shards.reshape(-1, shards.shape[-1]).T
    return embed_tokens(tokens_local, table_local, dtype=dtype, iota_embed=iota_embed)

  return jax.shard_map(
      _lookup,
      mesh=jax.typeof(table).sharding.mesh,
      in_specs=(jax.sharding.PartitionSpec(*tokens_spec), table_spec),
      out_specs=jax.sharding.PartitionSpec(*tokens_spec, None),
  )(tokens, table)


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
) -> jt.Num[jax.Array, "*batch D"]:
  """Applies the final RMSNorm, as MaxText's `RMSNorm` without scale offset.

  `x` must already be in the compute dtype: like MaxText, the normalized
  activations and the scale are both rounded to that dtype before the scale
  multiply.

  Args:
    x: Hidden states `[*batch, emb]` in the compute dtype.
    scale: Norm scale `[emb]` in any dtype.
    epsilon: Norm epsilon.

  Returns:
    The normalized hidden states in the compute dtype.
  """
  return ops.rms_norm(x, jnp.asarray(scale, x.dtype), epsilon=epsilon)


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

  Returns:
    The logits `[*batch, vocab]`.
  """
  dot_dtype = jnp.float32 if dot_in_fp32 else dtype
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
