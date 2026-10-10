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

"""DeepSeek MoE router implementation."""

from collections.abc import Mapping
import dataclasses
import functools
import math
from typing import Any, Generic, TypeVar

import jax
import jax.experimental.compute_on
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_expert_shuffle
from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops

compute_on = jax.experimental.compute_on.compute_on
ArrayType = jax.Array
ShardingType = jax.sharding.PartitionSpec
T = TypeVar("T", ArrayType, ShardingType, Any)
# MoE chunk capacities are rounded up to a multiple of this many slots, since
# the chunked MoE path produces garbage for unaligned capacities.
_CAPACITY_ALIGN = 256


def _get_axis_size(axis_name: str | tuple[str, ...]) -> int:
  """Returns the size of the given axis name or combination of axis names."""
  if isinstance(axis_name, str):
    return jax.lax.axis_size(axis_name)
  else:
    return math.prod(jax.lax.axis_size(name) for name in axis_name)


def _get_axis_index(axis_name: str | tuple[str, ...]) -> jax.Array:
  """Returns the index of the given axis name or combination of axis names."""
  if isinstance(axis_name, str):
    return jax.lax.axis_index(axis_name)
  else:
    idx = jnp.array(0, dtype=jnp.int32)
    for name in axis_name:
      idx = idx * jax.lax.axis_size(name) + jax.lax.axis_index(name)
    return idx


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_get_token_axis(
    x_pspec: jax.sharding.PartitionSpec,
    num_model_dims: int = 1,
) -> str | tuple[str, ...] | None:
  """Extracts the token axis from a PartitionSpec for leading token dimensions."""
  token_axes = []
  for ax in x_pspec[:-num_model_dims] if num_model_dims > 0 else x_pspec:
    if ax is not None:
      if isinstance(ax, tuple):
        token_axes.extend(ax)
      else:
        token_axes.append(ax)
  if not token_axes:
    return None
  elif len(token_axes) == 1:
    return token_axes[0]
  else:
    return tuple(token_axes)


def _all_gather_sc0(
    *operands: jax.Array,
    axis_name: str | tuple[str, ...],
) -> tuple[jax.Array, ...]:
  """Executes all_gather on SparseCore 0."""

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _ag(*xs: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True) for x in xs)

  return _ag(*operands)


def _psum_scatter_sc0(
    *operands: jax.Array,
    axis_name: str | tuple[str, ...],
) -> tuple[jax.Array, ...]:
  """Executes psum_scatter on SparseCore 0."""

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _rs(*xs: jax.Array) -> tuple[jax.Array, ...]:
    return tuple(jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=0, tiled=True) for x in xs)

  return _rs(*operands)


def _gather_sc0(
    x: jax.Array,
    indices: jax.Array,
) -> jax.Array:
  """Executes a gather on SparseCore 0."""

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _gather(
      flat_x: jax.Array,
      flat_indices: jax.Array,
  ) -> jax.Array:
    return flat_x.at[flat_indices].get(mode="promise_in_bounds", wrap_negative_indices=False)

  out = _gather(jnp.ravel(x), jnp.ravel(indices))
  return jnp.reshape(out, indices.shape + x.shape[1:])


def _scatter_sc0(
    x: jax.Array,
    indices: jax.Array,
) -> jax.Array:
  """Executes a 1D permutation scatter on SparseCore 0.

  SparseCore scatter only supports add, and a float32 scatter-add is far slower
  than an int32 one, so the 32-bit patterns of `x` are scattered as int32 with
  unique indices into zeros, which reproduces them exactly.

  Args:
    x: Values to scatter.
    indices: Destination indices, a permutation of `[0, x.size)` (not checked at
      runtime).

  Returns:
    Array of shape `x.shape` with `out.ravel()[indices.ravel()[i]] =
    x.ravel()[i]`.
  """

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _scatter(
      zeros: jax.Array,
      flat_x: jax.Array,
      flat_indices: jax.Array,
  ) -> jax.Array:
    return zeros.at[flat_indices].add(
        flat_x,
        mode="promise_in_bounds",
        wrap_negative_indices=False,
        unique_indices=True,
    )

  sc_dtype = x.dtype if x.dtype in (jnp.float32, jnp.int32) else jnp.float32
  flat_x = jax.lax.bitcast_convert_type(jnp.ravel(x).astype(sc_dtype), jnp.int32)
  zeros = jnp.zeros_like(flat_x)
  out = _scatter(zeros, flat_x, jnp.ravel(indices))
  out = jax.lax.bitcast_convert_type(out, sc_dtype).astype(x.dtype)
  return jnp.reshape(out, x.shape)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RouterMetadata(Generic[T]):
  sort_indices: T
  group_sizes: T
  sorted_coeffs: T
  # Routing of the ragged gather reduce of chunk 0 in unpermute fwd and permute
  # bwd. A single PartitionSpec (a pytree prefix) in the spec tree.
  reduce_metadata: ops.RaggedGatherReduceMetadata | T
  # Forward routing decision of the local tokens, shape (BT, k). The backward
  # pass replays it (`dsv3_routing_replay`) instead of recomputing top-k.
  selected_experts: T
  # Forward pre-bias router scores sigmoid(x @ kernel) of the local tokens in
  # the router compute dtype, shape (BT, E). The backward pass differentiates
  # through these saved scores instead of recomputing the router projection.
  probs: T
  # Number of sorted local-expert slots processed per chunk on each device.
  capacity: int = jax.tree.static()


def chunk_num_tokens(
    num_local_tokens: jt.Int[jax.Array, ""],
    start: jt.Int[jax.Array, ""] | int,
    capacity: int,
) -> jt.Int[jax.Array, ""]:
  """Returns the number of local-expert slots in the chunk starting at `start`."""
  remaining = num_local_tokens - start
  return jnp.where(remaining >= capacity, capacity, remaining)


def chunk_group_sizes(
    group_sizes: jt.Int[jax.Array, "E_local"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""],
) -> jt.Int[jax.Array, "E_local"]:
  """Returns the per-expert sizes of sorted slots [start, start + num_tokens)."""
  ends = jnp.cumsum(group_sizes)
  begins = ends - group_sizes
  stop = start + num_tokens
  return (jnp.clip(ends, start, stop) - jnp.clip(begins, start, stop)).astype(group_sizes.dtype)


def chunk_reduce_metadata(
    sort_indices: jt.Int[jax.Array, "kNBT"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""],
    *,
    capacity: int,
    num_experts_per_tok: int,
) -> ops.RaggedGatherReduceMetadata:
  """Derives the ragged gather reduce routing of one chunk of sorted slots."""
  return ops.ragged_gather_reduce_tc_metadata(
      sort_indices // num_experts_per_tok,
      start=start,
      num_tokens=num_tokens,
      top_k=num_experts_per_tok,
      max_num_tokens=capacity,
  )


def permute_chunk(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""],
    *,
    capacity: int,
    num_experts_per_tok: int,
    with_absmax: bool = False,
) -> Any:
  """Gathers sorted slots [start, start + num_tokens) into a capacity buffer."""
  return ops.ragged_gather_tc(
      x_ag,
      sort_indices // num_experts_per_tok,
      start=start,
      num_tokens=num_tokens,
      max_out_tokens=capacity,
      with_absmax=with_absmax,
  )


def unpermute_chunk(
    x: jt.Num[jax.Array, "C D0 D1"],
    reduce_metadata: ops.RaggedGatherReduceMetadata,
    *,
    num_out_tokens: int,
    num_experts_per_tok: int,
    out: jt.Num[jax.Array, "NBT D0 D1"] | None = None,
    zero_initialized: bool | None = None,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Sends a chunk's slots back to their tokens, summing over top-k."""
  return ops.ragged_gather_reduce_tc(
      x,
      reduce_metadata,
      num_out_tokens=num_out_tokens,
      top_k=num_experts_per_tok,
      out=out,
      zero_initialized=zero_initialized,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _sort_coeffs(
    coeffs: jt.Num[jax.Array, "BT k"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    num_experts_per_tok: int,
    axis_name: str | tuple[str, ...],
) -> jt.Num[jax.Array, "kNBT 1"]:
  """All-gathers coeffs and sorts them into expert order with local experts first."""
  return _sort_coeffs_fwd(
      coeffs,
      sort_indices,
      group_sizes,
      num_experts_per_tok,
      axis_name,
  )[0]


def _sort_coeffs_fwd(
    coeffs: jt.Num[jax.Array, "BT k"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    num_experts_per_tok: int,
    axis_name: str | tuple[str, ...],
) -> tuple[
    jt.Num[jax.Array, "kNBT 1"],
    tuple[jt.Int[jax.Array, "kNBT"], jt.Int[jax.Array, "E_local"]],
]:
  del num_experts_per_tok
  (coeffs_ag,) = _all_gather_sc0(coeffs, axis_name=axis_name)
  sorted_coeffs = _gather_sc0(jnp.reshape(coeffs_ag, (-1, 1)), sort_indices)
  return sorted_coeffs, (sort_indices, group_sizes)


def _sort_coeffs_bwd(
    num_experts_per_tok: int,
    axis_name: str | tuple[str, ...],
    res: tuple[jt.Int[jax.Array, "kNBT"], jt.Int[jax.Array, "E_local"]],
    grad_sorted_coeffs: jt.Num[jax.Array, "kNBT 1"],
) -> tuple[jt.Num[jax.Array, "BT k"], None, None]:
  """Masks non-local slots, unsorts via scatter, and reduce-scatters coeff gradients."""
  sort_indices, group_sizes = res
  num_local_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  valid = jnp.arange(grad_sorted_coeffs.shape[0]) < num_local_tokens
  masked_grad = jnp.where(valid[:, None], grad_sorted_coeffs, 0)
  unsorted = _scatter_sc0(masked_grad, sort_indices)
  grad_coeffs_ag = jnp.reshape(unsorted, (-1, num_experts_per_tok))
  (grad_coeffs,) = _psum_scatter_sc0(grad_coeffs_ag, axis_name=axis_name)
  return grad_coeffs, None, None


_sort_coeffs.defvjp(_sort_coeffs_fwd, _sort_coeffs_bwd)


def _cumsum_small_ints(
    x: jt.Shaped[jax.Array, "*B n"],
) -> jt.Int[jax.Array, "*B n"]:
  """Returns the inclusive cumsum over the last axis of small ints on the MXU.

  Exact, as bf16 represents the inputs and f32 the sums.

  Args:
    x: Bools or integers in [0, 256] whose sums are below 2**24.
  """
  n = x.shape[-1]
  upper = jnp.arange(n)[:, None] <= jnp.arange(n)[None, :]
  return jnp.matmul(
      x.astype(jnp.bfloat16),
      upper.astype(jnp.bfloat16),
      preferred_element_type=jnp.float32,
  ).astype(jnp.int32)


def _counting_argsort(keys: jt.Int[jax.Array, "N"], num_keys: int) -> jt.Int[jax.Array, "N"]:
  """Returns `jnp.argsort(keys, stable=True)` for keys in [0, num_keys).

  A counting sort, much faster on TPU than a sort for few distinct keys: a
  slot's destination is the number of slots with a smaller key plus the number
  of earlier slots with its key. The slots are laid out in blocks of 128 rows
  of 128 slots, the ranks are counted within rows and then within blocks on the
  MXU and across blocks by a short cumulative sum, and the destinations are
  inverted by a SparseCore scatter.

  Args:
    keys: Keys to sort, with values in [0, num_keys).
    num_keys: Number of distinct keys.
  """
  (n,) = keys.shape
  row = 128
  # Padded slots take the last key after every slot, so they sort last.
  rows = jnp.reshape(
      jnp.pad(keys, (0, -n % (row * row)), constant_values=num_keys - 1),
      (-1, row),
  )
  one_hot = rows[None] == jnp.arange(num_keys, dtype=keys.dtype)[:, None, None]
  # [c, r, i]: slots of key c in row r up to slot i.
  rank_in_row = _cumsum_small_ints(one_hot)
  row_counts = jnp.reshape(rank_in_row[..., -1], (num_keys, -1, row))
  # [c, b, r]: slots of key c in block b of 128 rows up to row r.
  rows_in_block = _cumsum_small_ints(row_counts)
  block_counts = rows_in_block[..., -1]
  key_counts = jnp.sum(block_counts, axis=1)
  # Slots with a smaller key, or with the same key in an earlier block or row.
  row_offsets = jnp.reshape(
      (jnp.cumsum(key_counts) - key_counts)[:, None, None]
      + (jnp.cumsum(block_counts, axis=1) - block_counts)[..., None]
      + rows_in_block
      - row_counts,
      (num_keys, -1),
  )
  dest = jnp.sum(jnp.where(one_hot, rank_in_row + row_offsets[..., None], 0), 0) - 1
  dest = jnp.ravel(dest)[:n]
  return _scatter_sc0(jnp.arange(n, dtype=jnp.int32), dest)


def _routing_metadata_impl(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    expert_permutations: jt.Int[jax.Array, "E"] | None = None,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    capacity: int,
    router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    RouterMetadata[jax.Array],
    jt.Num[jax.Array, "BT k"],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Computes routing metadata and coefficients for local tokens."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))

  (
      selected_experts,
      coeffs,
      local_group_sizes,
      logits,
      probs,
  ) = _expert_selection_impl(
      x_flat,
      w,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
      router_dtype=router_dtype,
  )

  # Routing metadata (sort order, local group sizes) is computed on physical
  # expert positions, which differ from logical expert ids when the routed
  # expert weights are shuffled. Aux outputs stay in logical expert order.
  if expert_permutations is not None:
    physical_selected_experts = dsv3_expert_shuffle.remap_selected_experts(selected_experts, expert_permutations)
    physical_group_sizes = dsv3_expert_shuffle.remap_group_sizes(local_group_sizes, expert_permutations)
  else:
    physical_selected_experts = selected_experts
    physical_group_sizes = local_group_sizes

  # Metadata computation across all devices along the expert axis.
  num_devices = _get_axis_size(physical_expert_axis_name)
  local_num_experts = num_experts // num_devices
  device_index = _get_axis_index(physical_expert_axis_name)

  selected_experts_flat = jnp.ravel(physical_selected_experts)
  (all_selected_experts,) = _all_gather_sc0(selected_experts_flat, axis_name=physical_expert_axis_name)

  # Place tokens routed to local experts at the start of the buffer in expert
  # order; non-local tokens are placed at the end.
  local_expert = all_selected_experts - device_index * local_num_experts
  is_local = (local_expert >= 0) & (local_expert < local_num_experts)
  sort_key = jnp.where(is_local, local_expert, local_num_experts)
  sort_indices = _counting_argsort(sort_key, local_num_experts + 1)

  # Sizes
  all_group_sizes = jax.lax.all_gather(physical_group_sizes, axis_name=physical_expert_axis_name, axis=0)
  all_send_sizes_3d = jnp.reshape(all_group_sizes, (num_devices, num_devices, local_num_experts))
  group_sizes = jnp.sum(all_send_sizes_3d, axis=0)[device_index]

  sorted_coeffs = _sort_coeffs(
      coeffs,
      sort_indices,
      group_sizes,
      num_experts_per_tok,
      physical_expert_axis_name,
  )

  # Permute fwd gathers chunk 0, the first min(sum(group_sizes), capacity)
  # local-expert slots of sort_indices // k; unpermute fwd and permute bwd
  # reduce them. Later chunks derive their routing in the MoE loop.
  reduce_metadata = chunk_reduce_metadata(
      sort_indices,
      0,
      chunk_num_tokens(jnp.sum(group_sizes, dtype=jnp.int32), 0, capacity),
      capacity=capacity,
      num_experts_per_tok=num_experts_per_tok,
  )

  metadata = RouterMetadata[jax.Array](
      sort_indices=sort_indices,
      group_sizes=group_sizes,
      sorted_coeffs=sorted_coeffs,
      reduce_metadata=reduce_metadata,
      selected_experts=selected_experts,
      probs=jax.lax.stop_gradient(probs),
      capacity=capacity,
  )

  selected_experts = jnp.reshape(selected_experts, x.shape[:-1] + (num_experts_per_tok,))
  logits = jnp.reshape(logits, x.shape[:-1] + (num_experts,))
  aux = dsv3_types.DSv3RouterAux[jax.Array](
      group_sizes=local_group_sizes[None, :],
      selected_experts=selected_experts,
      logits=logits,
  )
  return metadata, coeffs, aux


def chunk_capacity(num_local_slots: int, num_expert_shards: int, capacity_factor: float) -> int:
  """Returns the number of sorted local-expert slots in each MoE chunk.

  Args:
    num_local_slots: Number of local tokens times `num_experts_per_tok`.
    num_expert_shards: Number of devices the experts are sharded over.
    capacity_factor: A chunk holds `capacity_factor * num_local_slots` slots,
      rounded up to a multiple of `_CAPACITY_ALIGN`, but never more than all the
      slots that can be routed to the device.

  Returns:
    The chunk capacity.
  """
  return min(
      num_local_slots * num_expert_shards,
      math.ceil(capacity_factor * num_local_slots / _CAPACITY_ALIGN) * _CAPACITY_ALIGN,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_metadata(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    capacity_factor: float,
    router_dtype: jax.typing.DTypeLike | None = None,
    expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> tuple[
    RouterMetadata[jax.Array],
    jt.Num[jax.Array, "BT k"],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Computes routing metadata and routing coefficients for all tokens.

  Args:
    x: Input tokens of shape (B, T, D).
    w: MoE router weights.
    num_experts: Total number of routed experts.
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    capacity_factor: Each MoE chunk processes `capacity_factor` times the
      average number of slots routed to each device's local experts, i.e. the
      number of local tokens times `num_experts_per_tok`, rounded up to a
      multiple of `_CAPACITY_ALIGN` slots.
    router_dtype: Optional dtype for expert selection; see `expert_selection`.
    expert_permutations: Optional physical position of each logical expert (see
      `dsv3_expert_shuffle`). Routing decisions, coefficients, and aux outputs
      are unaffected; only the metadata (sort order and local group sizes) is
      expressed in physical expert positions so that it matches shuffled routed
      expert weights.

  Returns:
    A tuple of (RouterMetadata, coeffs, aux).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec)

  coeffs_spec = jax.sharding.PartitionSpec(token_axis, None)
  token_spec_1d = jax.sharding.PartitionSpec(token_axis)
  token_pspec = x_pspec[:-1]
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)

  def _num_shards(axis: str | tuple[str, ...] | None) -> int:
    if axis is None:
      return 1
    if isinstance(axis, str):
      return mesh.shape[axis]
    return math.prod(mesh.shape[a] for a in axis)

  num_local_slots = math.prod(x.shape[:-1]) // _num_shards(token_axis) * num_experts_per_tok
  capacity = chunk_capacity(num_local_slots, _num_shards(physical_expert_axis_name), capacity_factor)

  metadata_out_spec = RouterMetadata[jax.sharding.PartitionSpec](
      sort_indices=token_spec_1d,
      group_sizes=jax.sharding.PartitionSpec(physical_expert_axis_name),
      sorted_coeffs=coeffs_spec,
      reduce_metadata=token_spec_1d,
      selected_experts=coeffs_spec,
      probs=coeffs_spec,
      capacity=capacity,
  )
  aux_out_spec = dsv3_types.DSv3RouterAux[jax.sharding.PartitionSpec](
      group_sizes=jax.sharding.PartitionSpec(token_axis, None),
      selected_experts=jax.sharding.PartitionSpec(*token_pspec, None),
      logits=jax.sharding.PartitionSpec(*token_pspec, None),
  )
  out_specs = (metadata_out_spec, coeffs_spec, aux_out_spec)

  if expert_permutations is not None:
    expert_permutations = jax.reshard(
        expert_permutations,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
  return jax.shard_map(
      functools.partial(
          _routing_metadata_impl,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          topk_in_group=topk_in_group,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          capacity=capacity,
          router_dtype=router_dtype,
      ),
      mesh=mesh,
      out_specs=out_specs,
      check_vma=False,
  )(x, w, expert_permutations)


@jax.custom_vjp
def _saved_router_probs(
    x: jt.Num[jax.Array, "BT D"],
    kernel: jt.Num[jax.Array, "D E"],
    probs: jt.Num[jax.Array, "BT E"],
) -> jt.Num[jax.Array, "BT E"]:
  """Returns the saved `probs = sigmoid(x @ kernel)` without recomputing them.

  The VJP is that of `sigmoid(x @ kernel)` evaluated at the saved `probs`. The
  sigmoid derivative runs in fp32; the projection transposes take bf16 (the
  dtype of `x` and `kernel`) operands with fp32 accumulation and return the
  gradients in the dtypes of `x` and `kernel`.

  Args:
    x: Flattened local input tokens.
    kernel: Router kernel.
    probs: Saved forward `sigmoid(x @ kernel)`.

  Returns:
    `probs`.
  """
  del x, kernel
  return probs


def _saved_router_probs_fwd(
    x: jax.Array, kernel: jax.Array, probs: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  return probs, (x, kernel, probs)


def _saved_router_probs_bwd(
    res: tuple[jax.Array, jax.Array, jax.Array],
    grad_probs: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  x, kernel, probs = res
  input_dtype = jnp.result_type(x, kernel)
  probs_f32 = probs.astype(jnp.float32)
  grad_z = (grad_probs.astype(jnp.float32) * probs_f32 * (1.0 - probs_f32)).astype(input_dtype)
  grad_x = jnp.dot(
      grad_z,
      kernel.astype(input_dtype).T,
      preferred_element_type=jnp.float32,
  ).astype(x.dtype)
  grad_kernel = jnp.dot(
      x.astype(input_dtype).T,
      grad_z,
      preferred_element_type=jnp.float32,
  ).astype(kernel.dtype)
  return grad_x, grad_kernel, None


_saved_router_probs.defvjp(_saved_router_probs_fwd, _saved_router_probs_bwd)


def _routing_replay_impl(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"],
    probs: jt.Num[jax.Array, "BT E"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    *,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "kNBT 1"], jt.Num[jax.Array, "B T E"]]:
  """Rebuilds the local sorted coefficients and logits from saved routing."""
  assert w.kernel is not None
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))
  pre_bias_logits = _saved_router_probs(x_flat, w.kernel, probs)
  coeffs = selected_coeffs(
      pre_bias_logits,
      selected_experts,
      routed_scaling_factor=routed_scaling_factor,
  ).astype(x.dtype)
  sorted_coeffs = _sort_coeffs(
      coeffs,
      sort_indices,
      group_sizes,
      num_experts_per_tok,
      physical_expert_axis_name,
  )
  logits = jnp.reshape(pre_bias_logits.astype(x.dtype), x.shape[:-1] + (-1,))
  return sorted_coeffs, logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_replay(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "kNBT 1"], jt.Num[jax.Array, "B T E"]]:
  """Replays the forward routing of `dsv3_routing_metadata` for its backward.

  Returns the `(metadata.sorted_coeffs, aux.logits)` that
  `dsv3_routing_metadata` produced for `metadata`, as differentiable functions
  of `x` and `w`, without recomputing the routing: expert selection, the sort
  into expert order and the group sizes come from `metadata`, and the pre-bias
  router scores are the saved `metadata.probs`. Differentiating the replay
  therefore uses exactly the forward routing decisions and scores, even when a
  recomputation would round differently. The gradients match those of
  `dsv3_routing_metadata` with the same routing; they are zero for the bias,
  which only enters the routing.

  Args:
    x: Input tokens of shape (B, T, D), as passed to `dsv3_routing_metadata`.
    w: MoE router weights, as passed to `dsv3_routing_metadata`.
    metadata: Routing metadata that `dsv3_routing_metadata` returned for `x` and
      `w`.
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    A tuple of (sorted_coeffs, logits).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec)
  out_specs = (
      jax.sharding.PartitionSpec(token_axis, None),
      jax.sharding.PartitionSpec(*x_pspec[:-1], None),
  )
  return jax.shard_map(
      functools.partial(
          _routing_replay_impl,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
      check_vma=False,
  )(
      x,
      w,
      metadata.selected_experts,
      metadata.probs,
      metadata.sort_indices,
      metadata.group_sizes,
  )


def _unsort_coeff_grads_impl(
    grad_sorted_coeffs: jt.Num[jax.Array, "kNBT"],
    sort_indices: jt.Int[jax.Array, "kNBT"],
    group_sizes: jt.Int[jax.Array, "E_local"],
    *,
    num_shards: int,
) -> jt.Num[jax.Array, "R C"]:
  """Unsorts this device's coefficient gradients into all-gathered slot order.

  Slots past the local experts' tokens are zeroed, so every slot is nonzero on
  at most one device (the one holding its expert) and a psum_scatter over the
  expert axis sends each local token exactly its coefficient gradients. The
  result is laid out lane-dense, (slots / 128, 128) when that splits evenly
  across `num_shards`, with each shard's block contiguous along dim 0: a
  (slots, 1) or (tokens, k) layout is padded to 128 lanes and makes the
  SparseCore reduce-scatter more than an order of magnitude slower.

  Args:
    grad_sorted_coeffs: Cotangent of the sorted coefficients on this device.
    sort_indices: Sorted slot -> all-gathered slot (token * k + j) permutation.
    group_sizes: Local expert group sizes.
    num_shards: Size of the expert axis.

  Returns:
    The unsorted coefficient gradients of all all-gathered slots, reshaped for
    a tiled psum_scatter along dim 0.
  """
  num_local_tokens = jnp.sum(group_sizes, dtype=jnp.int32)
  valid = jnp.arange(grad_sorted_coeffs.shape[0]) < num_local_tokens
  masked = jnp.where(valid, grad_sorted_coeffs, 0).astype(grad_sorted_coeffs.dtype)
  unsorted = _scatter_sc0(masked, sort_indices)
  num_slots = unsorted.shape[0]
  if num_slots % (num_shards * 128) == 0:
    return jnp.reshape(unsorted, (num_slots // 128, 128))
  return jnp.reshape(unsorted, (num_shards, num_slots // num_shards))


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unsort_coeff_grads(
    grad_sorted_coeffs: jt.Num[jax.Array, "kNBT"],
    metadata: RouterMetadata[jax.Array],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "R C"]:
  """Locally unsorts coefficient gradients for `dsv3_combine_with_coeff_grads`.

  The unsort is a SparseCore scatter that the combine's reduce-scatter depends
  on. Callers should make sure it completes before the compute that the
  combine is meant to overlap (e.g. with an optimization barrier); otherwise
  the scheduler defers its completion behind that compute and the
  reduce-scatter starts late.

  Args:
    grad_sorted_coeffs: Cotangent of `metadata.sorted_coeffs`, flattened.
    metadata: Router metadata of the microbatch.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Per-device unsorted, lane-dense coefficient gradients of all all-gathered
    slots, sharded along dim 0 like `grad_sorted_coeffs`.
  """
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  token_axis = jax.typeof(grad_sorted_coeffs).sharding.spec[0]

  def _impl(g, sort_indices, group_sizes):
    return _unsort_coeff_grads_impl(
        g,
        sort_indices,
        group_sizes,
        num_shards=_get_axis_size(physical_expert_axis_name),
    )

  return jax.shard_map(
      _impl,
      mesh=mesh,
      out_specs=jax.sharding.PartitionSpec(token_axis, None),
      check_vma=False,
  )(grad_sorted_coeffs, metadata.sort_indices, metadata.group_sizes)


def _combine_with_coeff_grads_impl(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    *,
    local_seq_length: int,
    num_experts_per_tok: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"]]:
  """Per-shard body of `dsv3_combine_with_coeff_grads`."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  grad_x_local, grad_coeffs_local = _psum_scatter_sc0(grad_x_ag, grad_coeffs_ag, axis_name=physical_expert_axis_name)
  grad_x = jnp.reshape(
      grad_x_local,
      (-1, local_seq_length, grad_x_local.shape[-2] * grad_x_local.shape[-1]),
  )
  return grad_x, jnp.reshape(grad_coeffs_local, (-1, num_experts_per_tok))


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_with_coeff_grads(
    grad_x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    grad_coeffs_ag: jt.Num[jax.Array, "R C"],
    metadata: RouterMetadata[jax.Array],
    *,
    local_seq_length: int,
    num_experts_per_tok: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
) -> tuple[jt.Num[jax.Array, "B T D"], jt.Num[jax.Array, "BT k"]]:
  """Transposes the dispatch and the coefficient sort in one psum_scatter.

  The token gradients `grad_x_ag` go through the same psum_scatter as
  `dsv3_combine`. The coefficient gradients, already unsorted into
  all-gathered slot order by `dsv3_unsort_coeff_grads`, are reduce-scattered
  in the same SparseCore block, so the router backward needs no collective of
  its own. The coefficient gradients are exactly those of `_sort_coeffs`'s VJP.

  Args:
    grad_x_ag: Cotangent of the all-gathered tokens, shape (NBT, D0, D1).
    grad_coeffs_ag: Output of `dsv3_unsort_coeff_grads`.
    metadata: Router metadata of the microbatch.
    local_seq_length: Sequence length per shard.
    num_experts_per_tok: Number of experts selected per token.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    out_specs: Output PartitionSpec for the (B, T, D) token gradients.

  Returns:
    A tuple of the local token gradients (B, T, D) and the local token
    coefficient gradients (BT, k), sharded like `metadata.selected_experts`.
  """
  out_specs_phys = ops.physical_pspec(out_specs, axis_mapping)
  coeffs_spec = jax.typeof(metadata.selected_experts).sharding.spec
  return jax.shard_map(
      functools.partial(
          _combine_with_coeff_grads_impl,
          local_seq_length=local_seq_length,
          num_experts_per_tok=num_experts_per_tok,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=(out_specs_phys, coeffs_spec),
      check_vma=False,
  )(grad_x_ag, grad_coeffs_ag)


def _dense_selected_coeffs(
    pre_bias_logits: jt.Num[jax.Array, "BT E"],
    indices: jt.Int[jax.Array, "BT k"],
    *,
    routed_scaling_factor: float,
) -> jt.Num[jax.Array, "BT k"]:
  """`selected_coeffs` with a one-hot select instead of a SparseCore gather.

  Each selected score is the only nonzero term of its sum, so the gathered
  scores, and the scatter-add transpose over the k distinct experts of a
  token, are bitwise those of the gather. The (BT, k, E) select fuses into a
  TensorCore reduction, avoiding the SparseCore sort and scatter-add that the
  gather's transpose needs.

  Args:
    pre_bias_logits: Router scores of each token, shape (BT, E).
    indices: Selected experts of each token, shape (BT, k).
    routed_scaling_factor: Scaling applied to the normalized coefficients.

  Returns:
    The normalized, scaled coefficients of the selected experts, (BT, k).
  """
  num_experts = pre_bias_logits.shape[-1]
  one_hot = indices[..., None] == jnp.arange(num_experts, dtype=indices.dtype)
  coeffs = jnp.sum(
      jnp.where(one_hot, pre_bias_logits[:, None, :], 0),
      axis=-1,
      dtype=pre_bias_logits.dtype,
  )
  return _normalize_coeffs(coeffs, routed_scaling_factor=routed_scaling_factor)


def _local_routing_impl(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"],
    probs: jt.Num[jax.Array, "BT E"],
    *,
    routed_scaling_factor: float,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "B T E"]]:
  """Per-shard body of `dsv3_local_routing`."""
  assert w.kernel is not None
  x_flat = jnp.reshape(x, (-1, x.shape[-1]))
  pre_bias_logits = _saved_router_probs(x_flat, w.kernel, probs)
  coeffs = _dense_selected_coeffs(
      pre_bias_logits,
      selected_experts,
      routed_scaling_factor=routed_scaling_factor,
  ).astype(x.dtype)
  logits = jnp.reshape(pre_bias_logits.astype(x.dtype), x.shape[:-1] + (-1,))
  return coeffs, logits


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_local_routing(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    selected_experts: jt.Int[jax.Array, "BT k"],
    probs: jt.Num[jax.Array, "BT E"],
    *,
    routed_scaling_factor: float,
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "B T E"]]:
  """Computes local token coefficients and logits from saved router scores."""
  x_pspec = jax.typeof(x).sharding.spec
  out_specs = (
      jax.typeof(selected_experts).sharding.spec,
      jax.sharding.PartitionSpec(*x_pspec[:-1], None),
  )
  return jax.shard_map(
      functools.partial(
          _local_routing_impl,
          routed_scaling_factor=routed_scaling_factor,
      ),
      mesh=mesh,
      out_specs=out_specs,
      check_vma=False,
  )(x, w, selected_experts, probs)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_routing_bwd(
    grad_coeffs: jt.Num[jax.Array, "BT k"],
    grad_logits: jt.Num[jax.Array, "B T E"] | None,
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    metadata: RouterMetadata[jax.Array],
    *,
    routed_scaling_factor: float,
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3MoERouterWeightsPytree]:
  """Computes the router input and weight gradients without communication.

  Differentiates the local routing (coefficients and logits of the local
  tokens) at the saved routing decision and scores. `grad_coeffs` are the
  unsorted, local coefficient gradients from `dsv3_combine_with_coeff_grads`,
  which already did the cross-device part of `_sort_coeffs`'s transpose.

  Args:
    grad_coeffs: Coefficient gradients of the local tokens, shape (BT, k).
    grad_logits: Cotangent of the aux logits, or None.
    x: Router input tokens, shape (B, T, D).
    w: MoE router weights.
    metadata: Router metadata of the microbatch.
    routed_scaling_factor: Scaling factor for routed token scores.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    A tuple of (grad_x, grad_w).
  """

  def _fwd(x, w):
    return dsv3_local_routing(
        x,
        w,
        metadata.selected_experts,
        metadata.probs,
        routed_scaling_factor=routed_scaling_factor,
        mesh=mesh,
    )

  (_, logits), vjp_fn = jax.vjp(_fwd, x, w)
  if grad_logits is None:
    grad_logits = jnp.zeros_like(logits)
  grad_x, grad_w = vjp_fn((grad_coeffs, grad_logits))
  return grad_x, grad_w


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _permute_impl(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> jt.Num[jax.Array, "C D0 D1"]:
  return _permute_impl_fwd(x_ag, metadata, num_experts_per_tok)[0]


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_fwd(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> tuple[
    jt.Num[jax.Array, "C D0 D1"],
    RouterMetadata,
]:
  """Sorts chunk 0 of all-gathered tokens into local expert order."""
  num_local_tokens = jnp.sum(metadata.group_sizes, dtype=jnp.int32)
  out = permute_chunk(
      x_ag,
      metadata.sort_indices,
      0,
      chunk_num_tokens(num_local_tokens, 0, metadata.capacity),
      capacity=metadata.capacity,
      num_experts_per_tok=num_experts_per_tok,
  )
  return out, metadata


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_bwd(
    num_experts_per_tok: int,
    metadata: RouterMetadata,
    grads: jt.Num[jax.Array, "C D0 D1"],
) -> tuple[
    jt.Num[jax.Array, "NBT D0 D1"],
    None,
]:
  """Backward pass for permute: sums chunk 0 slots of each token over k."""
  grad_x_ag = unpermute_chunk(
      grads,
      metadata.reduce_metadata,
      num_out_tokens=metadata.sort_indices.shape[0] // num_experts_per_tok,
      num_experts_per_tok=num_experts_per_tok,
  )
  return grad_x_ag, None


_permute_impl.defvjp(_permute_impl_fwd, _permute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_permute(
    x_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata,
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "C D0 D1"]:
  """Permutes chunk 0 of all-gathered tokens into sorted expert order.

  Args:
    x_ag: All-gathered tokens of shape (NBT, D0, D1).
    metadata: Router metadata.
    num_experts_per_tok: Number of experts selected per token.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Sorted token buffer of shape (capacity, D0, D1) per device holding the first
    min(sum(group_sizes), capacity) local-expert slots.
  """
  x_pspec = jax.typeof(x_ag).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=2)

  return jax.shard_map(
      functools.partial(
          _permute_impl,
          num_experts_per_tok=num_experts_per_tok,
      ),
      mesh=mesh,
      out_specs=jax.sharding.PartitionSpec(token_axis, None, None),
  )(x_ag, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_permute_bwd(
    grad_x_sorted: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata,
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Backward pass for permute returning gradients wrt all-gathered tokens."""
  x_pspec = jax.typeof(grad_x_sorted).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=2)

  def _bwd(grads, md):
    grad_x_ag, _ = _permute_impl_bwd(num_experts_per_tok, md, grads)
    return grad_x_ag

  return jax.shard_map(
      _bwd,
      mesh=mesh,
      out_specs=jax.sharding.PartitionSpec(token_axis, None, None),
  )(grad_x_sorted, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch_impl(
    x: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Dispatches local tokens via all-gather on SC0."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  x_flat = jnp.reshape(x, (-1, x.shape[-1] // 128, 128))
  (x_ag,) = _all_gather_sc0(x_flat, axis_name=physical_expert_axis_name)
  return x_ag


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch(
    x: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Dispatches tokens across the expert axis via all-gather.

  Args:
    x: Input tokens of shape (B, T, D).
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    All-gathered tokens x_ag of shape (NBT, D0, D1).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=1)
  x_ag_spec = ops.physical_pspec(jax.sharding.PartitionSpec(token_axis, None, None), axis_mapping)
  return jax.shard_map(
      functools.partial(
          dsv3_dispatch_impl,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=x_ag_spec,
  )(x)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches_impl(
    moe_in: jt.Num[jax.Array, "B T D"],
    grad_out: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "NBT D0 D1"]]:
  """All-gathers moe_in and grad_out in a single SC0 compute_on block."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  moe_in_flat = jnp.reshape(moe_in, (-1, moe_in.shape[-1] // 128, 128))
  grad_out_flat = jnp.reshape(grad_out, (-1, grad_out.shape[-1] // 128, 128))
  x_ag, grad_y_ag = _all_gather_sc0(moe_in_flat, grad_out_flat, axis_name=physical_expert_axis_name)
  return x_ag, grad_y_ag


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches(
    moe_in: jt.Num[jax.Array, "B T D"],
    grad_out: jt.Num[jax.Array, "B T D"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], jt.Num[jax.Array, "NBT D0 D1"]]:
  """Rematerializes dispatch of moe_in and transposes combine of grad_out.

  Args:
    moe_in: Input tokens of shape (B, T, D).
    grad_out: Upstream output cotangent of shape (B, T, D).
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Tuple of (x_ag, grad_y_ag), each of shape (NBT, D0, D1).
  """
  x1_pspec = jax.typeof(moe_in).sharding.spec
  x2_pspec = jax.typeof(grad_out).sharding.spec
  token_axis = dsv3_get_token_axis(x1_pspec, num_model_dims=1) or dsv3_get_token_axis(x2_pspec, num_model_dims=1)
  token_spec = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec = ops.physical_pspec(token_spec, axis_mapping)
  out_specs = (token_spec, token_spec)
  return jax.shard_map(
      functools.partial(
          dsv3_aggregated_dispatches_impl,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(moe_in, grad_out)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_impl(
    y_ag: jt.Num[jax.Array, "NBT D0 D1"],
    *,
    local_seq_length: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "B T D"]:
  """Combines reduced expert output tokens across devices via psum_scatter."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  (y_local,) = _psum_scatter_sc0(y_ag, axis_name=physical_expert_axis_name)
  return jnp.reshape(
      y_local,
      (-1, local_seq_length, y_local.shape[-2] * y_local.shape[-1]),
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine(
    y_ag: jt.Num[jax.Array, "NBT D0 D1"],
    *,
    local_seq_length: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
) -> jt.Num[jax.Array, "B T D"]:
  """Combines reduced expert output tokens across devices via psum_scatter.

  Args:
    y_ag: Partial top-k reduced expert outputs of shape (NBT, D0, D1).
    local_seq_length: Sequence length per shard.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.
    out_specs: Output PartitionSpec for (B, T, D).

  Returns:
    Combined output tokens of shape (B, T, D).
  """
  out_specs_phys = ops.physical_pspec(out_specs, axis_mapping)
  return jax.shard_map(
      functools.partial(
          dsv3_combine_impl,
          local_seq_length=local_seq_length,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs_phys,
  )(y_ag)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _unpermute_impl(
    x: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Unsorts local expert outputs, masks non-local slots, and reduces over k."""
  return _unpermute_impl_fwd(x, metadata, num_experts_per_tok)[0]


def _unpermute_impl_fwd(
    x: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> tuple[jt.Num[jax.Array, "NBT D0 D1"], RouterMetadata]:
  """Forward pass for unpermute, which is the transpose of permute."""
  combined_x_ag, _ = _permute_impl_bwd(num_experts_per_tok, metadata, x)
  return combined_x_ag, metadata


def _unpermute_impl_bwd(
    num_experts_per_tok: int,
    metadata: RouterMetadata,
    cotangent_ag: jt.Num[jax.Array, "NBT D0 D1"],
) -> tuple[jt.Num[jax.Array, "C D0 D1"], None]:
  """Backward pass for unpermute, which is identical to permute."""
  grad_x, _ = _permute_impl_fwd(cotangent_ag, metadata, num_experts_per_tok)
  return grad_x, None


_unpermute_impl.defvjp(_unpermute_impl_fwd, _unpermute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute(
    x: jt.Num[jax.Array, "C D0 D1"],
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "NBT D0 D1"]:
  """Unsorts chunk 0 local expert tokens and reduces over top-k on each device.

  Args:
    x: Sorted, coefficient-weighted expert output tokens of chunk 0, of shape
      (capacity, D0, D1) per device.
    metadata: Router metadata.
    num_experts_per_tok: Number of experts selected per token.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Partial top-k reduced tokens of shape (NBT, D0, D1).
  """
  token_axis = dsv3_get_token_axis(jax.typeof(x).sharding.spec, num_model_dims=2)
  out_specs = jax.sharding.PartitionSpec(token_axis, None, None)
  return jax.shard_map(
      functools.partial(
          _unpermute_impl,
          num_experts_per_tok=num_experts_per_tok,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(x, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute_bwd(
    cotangent_ag: jt.Num[jax.Array, "NBT D0 D1"],
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "C D0 D1"]:
  """Computes gradient wrt sorted expert outputs; identical to permute."""
  return dsv3_permute(
      cotangent_ag,
      metadata,
      num_experts_per_tok=num_experts_per_tok,
      mesh=mesh,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_group_mask(
    gate_logits: jt.Num[jax.Array, "BT E"],
    *,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
) -> jt.Num[jax.Array, "BT E"]:
  """Computes expert group mask for node-limited routing.

  Args:
    gate_logits: Gate logits for all experts of shape (BT, E).
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.

  Returns:
    Mask of shape (BT, E) indicating which experts can be selected.
  """
  num_experts = gate_logits.shape[-1]
  # Find top groups based on each group's top-2 expert scores, where
  # `scores_grouped.shape =
  # (batch * seq, n_routing_groups, experts_per_group)`.
  scores_grouped = jnp.reshape(
      gate_logits,
      gate_logits.shape[:-1] + (n_routing_groups, -1),
  )
  topk_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=topk_in_group)
  group_scores = jnp.sum(jnp.astype(topk_in_group_vals, jnp.float32), axis=-1)
  _, group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)

  # Mask selected groups so that only those experts are considered.
  group_mask = jax.nn.one_hot(group_idx, num_classes=n_routing_groups, dtype=jnp.float32)
  group_mask = jnp.sum(group_mask, axis=-2)

  # Apply masks and get top-k indices.
  score_mask_expanded = jnp.broadcast_to(
      group_mask[..., None],
      group_mask.shape + (num_experts // n_routing_groups,),
  )
  return jnp.reshape(
      score_mask_expanded,
      score_mask_expanded.shape[:-2] + (num_experts,),
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_indices_and_coeffs(
    gate_logits: jt.Num[jax.Array, "BT E"],
    pre_bias_logits: jt.Num[jax.Array, "BT E"],
    *,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
) -> tuple[jt.Num[jax.Array, "BT k"], jt.Num[jax.Array, "BT k"]]:
  """Computes expert indices for each token and their corresponding scores.

  Args:
    gate_logits: Gate logits for all experts of shape (BT, E).
    pre_bias_logits: Pre-bias sigmoid logits of shape (BT, E).
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.

  Returns:
    A tuple of (selected_expert_indices, routing_coefficients).
  """
  expert_mask = expert_group_mask(
      gate_logits,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
  )
  gate_logits = jnp.where(expert_mask > 0, gate_logits, -jnp.inf)

  _, indices = jax.lax.top_k(
      gate_logits,
      k=num_experts_per_tok,
  )
  coeffs = selected_coeffs(pre_bias_logits, indices, routed_scaling_factor=routed_scaling_factor)
  return indices, coeffs


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def selected_coeffs(
    pre_bias_logits: jt.Num[jax.Array, "BT E"],
    indices: jt.Int[jax.Array, "BT k"],
    *,
    routed_scaling_factor: float,
) -> jt.Num[jax.Array, "BT k"]:
  """Gathers and normalizes the pre-bias scores of the selected experts.

  Args:
    pre_bias_logits: Pre-bias sigmoid logits of shape (BT, E).
    indices: Selected expert indices of shape (BT, k).
    routed_scaling_factor: Scaling factor for routed token scores.

  Returns:
    Routing coefficients of shape (BT, k).
  """
  num_experts_per_tok = indices.shape[-1]
  bt, num_e = pre_bias_logits.shape
  flat_logits = jnp.ravel(pre_bias_logits)
  flat_indices = jnp.ravel(indices + jnp.arange(bt, dtype=indices.dtype)[:, None] * num_e)

  # We don't pin to SC0 if padding is required because it results in
  # an XLA SC fusion error. This code path is only used for the tiny test.
  if flat_indices.shape[0] % 1024 != 0:
    coeffs = (
        flat_logits.at[flat_indices]
        .get(mode="promise_in_bounds", wrap_negative_indices=False)
        .reshape(bt, num_experts_per_tok)
    )
  else:

    @compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )
    def _sc_gather(flat_a: jax.Array, flat_idx: jax.Array) -> jax.Array:
      return flat_a.at[flat_idx].get(mode="promise_in_bounds", wrap_negative_indices=False)

    @compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )
    def _sc_scatter_add_f32(zeros_f32: jax.Array, flat_idx: jax.Array, updates_f32: jax.Array) -> jax.Array:
      return zeros_f32.at[flat_idx].add(updates_f32, mode="promise_in_bounds", wrap_negative_indices=False)

    @jax.custom_vjp
    def _gather(flat_a: jax.Array, flat_idx: jax.Array) -> jax.Array:
      return _sc_gather(flat_a, flat_idx)

    def _gather_fwd(flat_a: jax.Array, flat_idx: jax.Array):
      return _sc_gather(flat_a, flat_idx), flat_idx

    def _gather_bwd(flat_idx: jax.Array, g: jax.Array):
      zeros_f32 = jnp.zeros(flat_logits.shape, dtype=jnp.float32)
      grad_a = _sc_scatter_add_f32(zeros_f32, flat_idx, g.astype(jnp.float32)).astype(g.dtype)
      return grad_a, None

    _gather.defvjp(_gather_fwd, _gather_bwd)

    coeffs = _gather(flat_logits, flat_indices).reshape(bt, num_experts_per_tok)
  return _normalize_coeffs(coeffs, routed_scaling_factor=routed_scaling_factor)


def _normalize_coeffs(
    coeffs: jt.Num[jax.Array, "BT k"],
    *,
    routed_scaling_factor: float,
) -> jt.Num[jax.Array, "BT k"]:
  """Normalizes the selected scores of each token and applies the scaling."""
  # Normalize in fp32 and round once, so the forward and the backward
  # (`dsv3_routing_bwd`) produce identical coefficients from identical scores
  # regardless of how XLA fuses a low-precision normalization.
  coeffs_f32 = coeffs.astype(jnp.float32)
  coeffs_f32 = routed_scaling_factor * (coeffs_f32 / coeffs_f32.sum(-1, keepdims=True))
  return coeffs_f32.astype(coeffs.dtype)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def expert_selection(
    x: jt.Num[jax.Array, "BT D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "E"],
    jt.Num[jax.Array, "BT E"],
]:
  """Selects experts for each token and calculates group sizes for each expert.

  Expert selection (projection accumulation, sigmoid, bias add, group
  selection, top-k and coefficient normalization) runs in `router_dtype`
  promoted with the dtypes of `x` and the router weights. With bf16 tokens and
  weights and `router_dtype=jnp.float32` (MaxText `float32_gate_logits=True`),
  the bf16 x bf16 projection is accumulated in fp32 and everything after it is
  fp32, while the router weights, and therefore their gradients and weight
  collectives, stay bf16. The coefficients and the returned `logits` are cast
  back to `x.dtype`, so the dispatch, expert and combine kernels and the
  auxiliary routing statistics are unaffected by the router dtype. The returned
  `logits` are the pre-bias scores `sigmoid(x @ kernel)` (MaxText's
  `pre_bias_logits`): the bias only steers the expert selection.

  Args:
    x: Flattened input tokens of shape (BT, D).
    w: MoE router weights.
    num_experts: Total number of routed experts.
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.
    router_dtype: Optional minimum dtype for expert selection. None computes in
      the promoted dtype of `x` and the router weights.

  Returns:
    A tuple of (selected_experts, coeffs, group_sizes, logits).
  """
  return _expert_selection_impl(
      x,
      w,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
      router_dtype=router_dtype,
  )[:4]


def _expert_selection_impl(
    x: jt.Num[jax.Array, "BT D"],
    w: dsv3_types.DSv3MoERouterWeightsPytree,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    router_dtype: jax.typing.DTypeLike | None = None,
) -> tuple[
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "E"],
    jt.Num[jax.Array, "BT E"],
    jt.Num[jax.Array, "BT E"],
]:
  """`expert_selection` that also returns the pre-bias scores.

  Returns:
    A tuple of (selected_experts, coeffs, group_sizes, logits, probs), where
    `probs = sigmoid(x @ kernel)` in the router compute dtype and `logits` is
    `probs` cast to `x.dtype`.
  """
  assert w.kernel is not None
  assert w.bias is not None
  input_dtype = jnp.result_type(x, w.kernel)
  compute_dtype = jnp.result_type(input_dtype, w.bias)
  if router_dtype is not None:
    compute_dtype = jnp.promote_types(compute_dtype, router_dtype)
  pre_bias_logits = jax.nn.sigmoid(
      jnp.tensordot(
          x.astype(input_dtype),
          w.kernel.astype(input_dtype),
          axes=1,
          preferred_element_type=compute_dtype,
      )
  )
  if jnp.finfo(compute_dtype).bits < 32:
    # XLA may keep the fused sigmoid in fp32 (excess precision). Round it so
    # the routing and coefficients use exactly the scores saved for the
    # backward replay.
    finfo = jnp.finfo(compute_dtype)
    pre_bias_logits = jax.lax.reduce_precision(pre_bias_logits, exponent_bits=finfo.nexp, mantissa_bits=finfo.nmant)
  logits = pre_bias_logits + jax.lax.stop_gradient(w.bias.astype(compute_dtype))

  selected_experts, coeffs = expert_indices_and_coeffs(
      logits,
      pre_bias_logits,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
  )
  coeffs = coeffs.astype(x.dtype)

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _bincount(x: jax.Array) -> jax.Array:
    return jnp.bincount(x, length=num_experts)

  group_sizes = _bincount(jnp.ravel(selected_experts))
  return (
      selected_experts,
      coeffs,
      group_sizes,
      pre_bias_logits.astype(x.dtype),
      pre_bias_logits,
  )
