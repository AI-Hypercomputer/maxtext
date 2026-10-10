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

"""Per-layer expert shuffling for DSv3 routed experts.

An expert permutation `perm` of shape `(num_experts,)` maps logical expert `e`
to physical position `p = perm[e]`, i.e. local slot `p % L` of expert shard
`p // L`, where `L = num_experts // num_expert_shards` is the number of experts
per shard. Without a permutation, logical expert `e` lives at physical position
`e` (shard `e // L`, slot `e % L`).

The permutation is realized with a single `ragged_all_to_all` per routed expert
weight array along the expert axis, moving each local expert slot to its
physical destination. Each shard sends and receives exactly `L` slots, so the
transfer volume per shard is one shard's worth of routed expert weights. The
backward pass is the inverse ragged all-to-all (permuting by `argsort(perm)`),
which is where gradients are "unmapped" back to logical expert order.

The router is not permuted: its kernel and bias stay in logical expert order so
that group-limited routing and the auxiliary outputs (`selected_experts`,
`logits`, `group_sizes`) are unaffected. Only the routing *metadata*
(sort order, local group sizes) is computed on physical expert ids, see
`remap_selected_experts`.
"""

from collections.abc import Mapping
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import typeguard

from maxtext.experimental.lineage import dsv3_types

P = jax.sharding.PartitionSpec


def physical_expert_axis(
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[str, ...]:
  """Returns the physical mesh axes backing the logical expert axis."""
  axes = axis_mapping.get(expert_axis_name, expert_axis_name)
  return (axes,) if isinstance(axes, str) else tuple(axes)


def num_expert_shards(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> int:
  """Returns the number of shards along the (physical) expert axis."""
  return math.prod(mesh.shape[a] for a in physical_expert_axis(expert_axis_name, axis_mapping))


def inverse_permutation(
    perm: jt.Int[jax.Array, "*L E"],
) -> jt.Int[jax.Array, "*L E"]:
  """Returns the inverse of each (batched) permutation."""
  return jnp.argsort(perm, axis=-1).astype(perm.dtype)


def _pack_permutation(
    perm: jax.Array,
) -> tuple[jax.Array, int, int]:
  """Packs `perm` into uint32 words of `per_word` `bits`-wide entries."""
  num_experts = perm.shape[0]
  bits = max(1, (num_experts - 1).bit_length())
  per_word = 32 // bits
  num_words = -(-num_experts // per_word)
  table = jnp.pad(perm.astype(jnp.uint32), (0, num_words * per_word - num_experts)).reshape(num_words, per_word)
  shifts = (jnp.arange(per_word, dtype=jnp.uint32) * bits)[None, :]
  packed = jnp.bitwise_or.reduce(table << shifts, axis=1)
  return packed, bits, per_word


@jt.jaxtyped(typechecker=typeguard.typechecked)
def remap_selected_experts(
    selected_experts: jt.Int[jax.Array, "..."],
    perm: jt.Int[jax.Array, "E"],
) -> jt.Int[jax.Array, "..."]:
  """Maps logical expert ids to physical positions under `perm`.

  Equivalent to `perm[selected_experts]`, but implemented as an elementwise
  lookup in a bit-packed copy of `perm` (`ceil(E / (32 // ceil(log2 E)))`
  uint32 words, selected with an unrolled `where` chain) instead of a
  per-token gather, which is slow on TensorCore.
  """
  packed, bits, per_word = _pack_permutation(perm)
  sel = selected_experts.astype(jnp.int32)
  word_index = sel // per_word
  shift = ((sel % per_word) * bits).astype(jnp.uint32)
  word = jnp.zeros(sel.shape, jnp.uint32)
  for i in range(packed.shape[0]):
    word = jnp.where(word_index == i, packed[i], word)
  mask = jnp.uint32((1 << bits) - 1)
  return ((word >> shift) & mask).astype(selected_experts.dtype)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def remap_group_sizes(
    group_sizes: jt.Int[jax.Array, "E"],
    perm: jt.Int[jax.Array, "E"],
) -> jt.Int[jax.Array, "E"]:
  """Reorders per-logical-expert counts into physical position order."""
  # physical_counts[p] = group_sizes[e] where perm[e] == p.
  return jnp.zeros_like(group_sizes).at[perm].set(group_sizes)


def local_slot_routing(
    perm: jt.Int[jax.Array, "E"],
    *,
    shard_index: jt.Int[jax.Array, ""],
    num_shards: int,
) -> tuple[
    jt.Int[jax.Array, "E"],
    jt.Int[jax.Array, "E"],
    jt.Int[jax.Array, "E"],
    jt.Int[jax.Array, "E"],
]:
  """Computes ragged all-to-all routing that moves local slot `s` to `perm[..]`.

  On shard `d`, local slot `s` holds position `d * L + s` and must be moved to
  position `perm[d * L + s]`, i.e. slot `perm[..] % L` on shard `perm[..] // L`.
  The ragged all-to-all uses `E = num_shards * L` size-1 updates, `L` per
  destination shard: update `j * L + t` carries the `t`-th local slot (in
  increasing slot order) destined to shard `j`, and is empty if fewer than
  `t + 1` local slots go to shard `j`.

  Args:
    perm: The full permutation, replicated on every shard.
    shard_index: Index of this shard along the expert axis.
    num_shards: Number of shards along the expert axis.

  Returns:
    `(input_offsets, send_sizes, output_offsets, recv_sizes)` for
    `jax.lax.ragged_all_to_all`.
  """
  num_experts = perm.shape[0]
  local_experts = num_experts // num_shards
  dtype = jnp.int32
  perm = perm.astype(dtype)
  slots = jnp.arange(local_experts, dtype=dtype)
  dest = perm[shard_index * local_experts + slots]
  dest_shard = dest // local_experts
  dest_slot = dest % local_experts
  # Rank of each slot among the earlier local slots sharing its destination.
  same_dest = dest_shard[:, None] == dest_shard[None, :]
  earlier = slots[None, :] < slots[:, None]
  rank = jnp.sum(same_dest & earlier, axis=1, dtype=dtype)
  update = dest_shard * local_experts + rank
  zeros = jnp.zeros((num_experts,), dtype=dtype)
  input_offsets = zeros.at[update].set(slots)
  send_sizes = zeros.at[update].set(1)
  output_offsets = zeros.at[update].set(dest_slot)
  # Shard j sends us `counts[j]` slots, in updates j * L + [0, counts[j]).
  counts = jnp.sum(
      (perm // local_experts).reshape(num_shards, local_experts) == shard_index,
      axis=1,
      dtype=dtype,
  )
  recv_sizes = (slots[None, :] < counts[:, None]).reshape(-1).astype(dtype)
  return input_offsets, send_sizes, output_offsets, recv_sizes


Routing = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _zeros_like_typed(x: jax.Array) -> jax.Array:
  """`zeros_like(x)` with the same varying/reduced/unreduced axes as `x`.

  Inside `shard_map`, `zeros_like` keeps the varying axes of `x` but drops the
  `reduced` (e.g. `dcn` after the DCN weight collect) and `unreduced` (the
  corresponding cotangents) axes. The casts are no-ops on the data.

  Args:
    x: Array inside `shard_map`.

  Returns:
    Zeros with the shape, dtype and varying/reduced/unreduced axes of `x`.
  """
  zeros = jnp.zeros_like(x)
  want = jax.typeof(x).mat
  have = jax.typeof(zeros).mat
  for ax in sorted(want.reduced - have.reduced):
    zeros = jax.lax.pcast(zeros, ax, to="reduced")
  for ax in sorted(want.unreduced - have.unreduced):
    if ax not in jax.typeof(zeros).mat.varying:
      zeros = jax.lax.pcast(zeros, ax, to="varying")
    zeros = jax.lax.pcast(zeros, ax, to="unreduced")
  return zeros


def _local_ragged_all_to_all(x: jax.Array, routing: Routing, axis_name: tuple[str, ...]) -> jax.Array:
  input_offsets, send_sizes, output_offsets, recv_sizes = routing
  # Every slot of the output is overwritten by a received slot, so the
  # initial contents are irrelevant.
  return jax.lax.ragged_all_to_all(
      x,
      _zeros_like_typed(x),
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      axis_name=axis_name,
  )


def _ragged_permute_shard(
    x: jax.Array,
    perm: jax.Array,
    *,
    axis_name: tuple[str, ...],
) -> jax.Array:
  """Per-shard body: moves local expert slots (dim 0) to their `perm` targets.

  The ragged all-to-all is issued on the TensorCore. XLA auto-offloads ragged
  all-to-alls to SparseCore whenever the module has any `compute_on` region,
  where it queues behind the FSDP collects. Callers should keep it async on the
  TensorCore with the XLA flags
    --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false
    --xla_tpu_force_async_all_to_all=true

  Args:
    x: Local shard of an expert-sharded array, expert slots on dim 0.
    perm: Expert permutation, replicated on every shard.
    axis_name: Physical expert axis names.

  Returns:
    The permuted local shard.
  """
  num_shards = math.prod(jax.lax.axis_size(a) for a in axis_name)
  shard_index = jax.lax.axis_index(axis_name)
  routing = local_slot_routing(perm, shard_index=shard_index, num_shards=num_shards)
  return _local_ragged_all_to_all(x, routing, axis_name)


def _ragged_permute(
    x: jax.Array,
    perm: jax.Array,
    *,
    axis_name: tuple[str, ...],
) -> jax.Array:
  """Permutes the expert-sharded dim 0 of `x` across the expert axis."""
  x_spec = jax.typeof(x).sharding.spec
  mesh = jax.typeof(x).sharding.mesh
  partitions = x_spec.partitions
  dim0 = partitions[0] if partitions else None
  dim0 = (dim0,) if isinstance(dim0, str) else tuple(dim0 or ())
  assert dim0 == axis_name, (
      "routed expert weights must be sharded along the expert axis" f" {axis_name} on dim 0, got spec {x_spec}"
  )
  perm = jax.reshard(perm, jax.sharding.NamedSharding(mesh, P()))
  return jax.shard_map(
      functools.partial(_ragged_permute_shard, axis_name=axis_name),
      mesh=mesh,
      in_specs=(x_spec, P()),
      out_specs=x_spec,
      check_vma=False,
  )(x, perm)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _permute_experts(
    x: jax.Array,
    perm: jax.Array,
    inv_perm: jax.Array,
    axis_name: tuple[str, ...],
) -> jax.Array:
  """Permutes expert slots by `perm`; the VJP permutes back by `inv_perm`."""
  del inv_perm
  return _ragged_permute(x, perm, axis_name=axis_name)


def _permute_experts_fwd(x, perm, inv_perm, axis_name):
  return _ragged_permute(x, perm, axis_name=axis_name), inv_perm


def _permute_experts_bwd(axis_name, inv_perm, g):
  # The transpose of a permutation is its inverse, so the cotangent of each
  # physical slot is sent back to its logical slot.
  return _ragged_permute(g, inv_perm, axis_name=axis_name), None, None


_permute_experts.defvjp(_permute_experts_fwd, _permute_experts_bwd)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def shuffle_experts(
    xs: jt.PyTree[jt.Num[jax.Array, "..."]],
    perm: jt.Int[jax.Array, "E"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.PyTree[jt.Num[jax.Array, "..."]]:
  """Moves expert-sharded arrays from logical to physical expert positions.

  Each leaf of `xs` must have its expert dimension first and sharded along the
  physical expert axis (`axis_mapping[expert_axis_name]`); other dimensions may
  be sharded arbitrarily. Returns arrays with the same shape and sharding whose
  physical position `p` (shard `p // L`, slot `p % L`) holds logical expert
  `argsort(perm)[p]`.

  The operation is pure data movement (no arithmetic), and its VJP is the
  inverse movement, so it is bit-exact in both directions.

  Args:
    xs: Pytree of expert-sharded arrays (expert dim 0).
    perm: Expert permutation; logical expert `e` goes to position `perm[e]`.
    expert_axis_name: Logical expert axis name.
    axis_mapping: Mapping from logical to physical mesh axes.

  Returns:
    The shuffled pytree.
  """
  axis_name = physical_expert_axis(expert_axis_name, axis_mapping)
  inv_perm = inverse_permutation(perm)
  return jax.tree.map(lambda x: _permute_experts(x, perm, inv_perm, axis_name), xs)


def validate_expert_permutations(
    perm: Any,
    *,
    num_experts: int,
    num_layers: int | None = None,
) -> None:
  """Checks shape (and values, when concrete) of an expert permutation array."""
  if perm is None:
    return
  expected_rank = 1 if num_layers is None else 2
  if perm.ndim != expected_rank:
    raise ValueError(f"expert_permutations must have rank {expected_rank}, got {perm.shape}")
  if perm.shape[-1] != num_experts:
    raise ValueError("expert_permutations last dim must equal num_experts" f" ({num_experts}), got {perm.shape}")
  if num_layers is not None and perm.shape[0] < num_layers:
    raise ValueError(f"expert_permutations must have at least {num_layers} rows, got" f" {perm.shape}")
  if not jnp.issubdtype(perm.dtype, jnp.integer):
    raise ValueError(f"expert_permutations must be integer, got {perm.dtype}")
  if isinstance(perm, jax.core.Tracer):
    return
  rows = np.asarray(perm).reshape(-1, num_experts)
  expected = np.arange(num_experts)
  for i, row in enumerate(rows):
    if not np.array_equal(np.sort(row), expected):
      raise ValueError(f"expert_permutations row {i} is not a permutation of" f" range({num_experts}): {row}")


def _lpt_permutation(group_sizes: jax.Array, num_shards: int) -> jax.Array:
  """Single-layer LPT on `(num_token_shards, num_experts)` group sizes."""
  num_experts = group_sizes.shape[-1]
  local_experts = num_experts // num_shards
  load = jnp.sum(group_sizes, axis=0, dtype=jnp.int32)
  # Most popular first; the stable sort breaks ties toward the lower expert id.
  order = jnp.argsort(-load, stable=True)
  full = jnp.iinfo(jnp.int32).max

  def body(i, carry):
    perm, shard_load, shard_count = carry
    e = order[i]
    open_load = jnp.where(shard_count < local_experts, shard_load, full)
    d = jnp.argmin(open_load)  # Ties go to the lower shard index.
    perm = perm.at[e].set(d * local_experts + shard_count[d])
    return perm, shard_load.at[d].add(load[e]), shard_count.at[d].add(1)

  init = (
      jnp.zeros((num_experts,), jnp.int32),
      jnp.zeros((num_shards,), jnp.int32),
      jnp.zeros((num_shards,), jnp.int32),
  )
  return jax.lax.fori_loop(0, num_experts, body, init)[0]


@functools.partial(jax.jit, static_argnames=("num_shards",))
def lpt_expert_permutation(
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_shards: int,
) -> jt.Int[jax.Array, "*L E"]:
  """Returns a load-balancing expert permutation from router aux.

  Uses the LPT (Longest Processing Time first) heuristic with a per-shard
  cardinality constraint: experts are ordered from most to least popular
  (total tokens routed to them, summed over token shards), and each one is
  assigned to the expert shard with the smallest total load so far among
  shards that still have a free slot. Every shard gets exactly
  `L = num_experts // num_shards` experts. Ties go to the lower expert
  id / shard index, so the result is deterministic.

  Runs on device (jittable); the result is replicated.

  Args:
    aux: Router aux; only `aux.group_sizes` is read (other fields may be
      None). Shape `(num_token_shards, num_experts)` for one layer, or
      `(num_layers, num_token_shards, num_experts)` for scanned layers.
    num_shards: Number of shards along the physical expert axis, e.g.
      `num_expert_shards(mesh, expert_axis_name, axis_mapping)`.

  Returns:
    `(num_experts,)` (or `(num_layers, num_experts)`) int32 permutation:
    logical expert `e` goes to physical position `perm[e]` (shard
    `perm[e] // L`, slot `perm[e] % L`).
  """
  group_sizes = aux.group_sizes
  num_experts = group_sizes.shape[-1]
  if num_shards <= 0 or num_experts % num_shards:
    raise ValueError(f"num_experts ({num_experts}) must be divisible by num_shards" f" ({num_shards})")
  if group_sizes.ndim == 3:
    return jax.vmap(functools.partial(_lpt_permutation, num_shards=num_shards))(group_sizes)
  if group_sizes.ndim != 2:
    raise ValueError(
        "aux.group_sizes must have shape (num_token_shards, num_experts) or"
        f" (num_layers, num_token_shards, num_experts), got {group_sizes.shape}"
    )
  return _lpt_permutation(group_sizes, num_shards)


# Values of the `expert_permutation` flag. "none" disables expert shuffling
# entirely; every other value names an algorithm that computes the next step's
# per-layer permutation from the current step's router aux.
EXPERT_PERMUTATION_ALGORITHMS = ("none", "lpt")


def next_expert_permutation(
    algorithm: str,
    aux: dsv3_types.DSv3RouterAux[Any],
    *,
    num_shards: int,
) -> jax.Array | None:
  """Returns the next expert permutation from router aux under `algorithm`.

  Args:
    algorithm: One of `EXPERT_PERMUTATION_ALGORITHMS`.
    aux: Router aux of the current step (see `lpt_expert_permutation` for the
      accepted shapes).
    num_shards: Number of shards along the physical expert axis.

  Returns:
    The permutation for the next step, or None for "none".
  """
  if algorithm == "none":
    return None
  if algorithm == "lpt":
    return lpt_expert_permutation(aux, num_shards=num_shards)
  raise ValueError(
      f"Unknown expert_permutation algorithm {algorithm!r}; expected one of" f" {EXPERT_PERMUTATION_ALGORITHMS}"
  )
