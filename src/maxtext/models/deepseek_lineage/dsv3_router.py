# Copyright 2023–2026 Google LLC
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

try:
  from jax.experimental.scheduling_groups import xla_metadata_call
except ImportError:
  try:
    from jax._src.xla_metadata import xla_metadata_call
  except ImportError:

    def xla_metadata_call(*unused_args, **unused_kwargs):
      return lambda fn: fn


import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops

compute_on = jax.experimental.compute_on.compute_on
ArrayType = jax.Array
ShardingType = jax.sharding.PartitionSpec
T = TypeVar("T", ArrayType, ShardingType, Any)


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


def _ragged_all_to_all_sc0(
    operand: jax.Array,
    out: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    *,
    axis_name: str | tuple[str, ...],
) -> jax.Array:
  """Executes ragged_all_to_all on SparseCore 0."""

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _ra2a(
      operand: jax.Array,
      out: jax.Array,
      input_offsets: jax.Array,
      send_sizes: jax.Array,
      output_offsets: jax.Array,
      recv_sizes: jax.Array,
  ) -> jax.Array:
    return jax.lax.ragged_all_to_all(
        operand,
        out,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis_name,
    )

  return _ra2a(
      operand,
      out,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
  )


def _gather_tc(
    x: jax.Array,
    indices: jax.Array,
    integer_config: int = 1024,
) -> jax.Array:
  """Executes a gather on TensorCore.

  Args:
    x: Input array.
    indices: Indices to gather.
    integer_config: Configure number of concurrent DMAs for the gather.

  Returns:
    Gathered array.
  """

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  @xla_metadata_call(integer=str(integer_config))
  def _gather(
      x: jax.Array,
      indices: jax.Array,
  ) -> jax.Array:
    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=tuple(range(indices.ndim, indices.ndim + x.ndim - 1)),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    slice_sizes = (1,) + x.shape[1:]
    return jax.lax.gather(
        x,
        indices[..., None],
        dimension_numbers=dnums,
        slice_sizes=slice_sizes,
    )

  return _gather(x, indices)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RouterMetadata(Generic[T]):
  sort_indices: T
  inverse_sort_indices: T
  send_sizes: T


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ChunkMetadata(Generic[T]):
  chunk_send_sizes: T
  next_expert: T
  group_sizes: T


def _compute_all_offsets(
    send_sizes_3d: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes 3D input offsets, receive sizes, write offsets, and expert sizes."""
  num_devices, _, local_num_experts = send_sizes_3d.shape
  send_sizes_flat = jnp.reshape(send_sizes_3d, (num_devices, -1))
  all_input_offsets = jnp.concatenate(
      [
          jnp.zeros((num_devices, 1), dtype=jnp.int32),
          jnp.cumsum(send_sizes_flat, axis=-1)[:, :-1],
      ],
      axis=-1,
  )
  all_input_offsets_3d = jnp.reshape(all_input_offsets, (num_devices, num_devices, local_num_experts))

  recv_sizes_3d = jnp.swapaxes(send_sizes_3d, 0, 1)
  all_expert_sizes = jnp.sum(recv_sizes_3d, axis=1)

  all_expert_starts = jnp.concatenate(
      [
          jnp.zeros((num_devices, 1), dtype=jnp.int32),
          jnp.cumsum(all_expert_sizes, axis=-1)[:, :-1],
      ],
      axis=-1,
  )
  all_relative_offsets = jnp.concatenate(
      [
          jnp.zeros((num_devices, 1, local_num_experts), dtype=jnp.int32),
          jnp.cumsum(recv_sizes_3d, axis=1)[:, :-1, :],
      ],
      axis=1,
  )
  all_write_offsets_3d = all_expert_starts[:, None, :] + all_relative_offsets
  return (
      all_input_offsets_3d,
      recv_sizes_3d,
      all_write_offsets_3d,
      all_expert_sizes,
  )


def _compute_chunk_dispatch_offsets(
    chunk_send_sizes_3d: jax.Array,
    full_send_sizes_3d: jax.Array,
    device_index: int | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes 1D communication slices for forward dispatch of a chunk."""
  full_input_offsets_3d, _, _, _ = _compute_all_offsets(full_send_sizes_3d)
  _, chunk_recv_sizes_3d, chunk_write_offsets_3d, _ = _compute_all_offsets(chunk_send_sizes_3d)
  input_offsets = jnp.ravel(full_input_offsets_3d[device_index])
  send_sizes = jnp.ravel(chunk_send_sizes_3d[device_index])
  output_offsets = jnp.ravel(chunk_write_offsets_3d[:, device_index, :])
  recv_sizes = jnp.ravel(chunk_recv_sizes_3d[device_index])
  return input_offsets, send_sizes, output_offsets, recv_sizes


def _compute_chunk_combine_offsets(
    chunk_send_sizes_3d: jax.Array,
    full_send_sizes_3d: jax.Array,
    device_index: int | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Computes 1D communication slices for backward combine of a chunk."""
  full_input_offsets_3d, _, _, _ = _compute_all_offsets(full_send_sizes_3d)
  _, chunk_recv_sizes_3d, chunk_write_offsets_3d, _ = _compute_all_offsets(chunk_send_sizes_3d)
  combine_input_offsets = jnp.ravel(chunk_write_offsets_3d[device_index])
  send_sizes = jnp.ravel(chunk_recv_sizes_3d[device_index])
  combine_output_offsets = jnp.ravel(full_input_offsets_3d[:, device_index, :])
  recv_sizes = jnp.ravel(chunk_send_sizes_3d[device_index])
  return combine_input_offsets, send_sizes, combine_output_offsets, recv_sizes


def _routing_metadata_impl(
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
  ) = expert_selection(
      x_flat,
      w,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
  )

  # Metadata computation
  num_devices = _get_axis_size(physical_expert_axis_name)
  local_num_experts = num_experts // num_devices

  selected_experts_flat = jnp.ravel(selected_experts)

  # Sort indices
  sort_indices = jnp.argsort(selected_experts_flat)
  inverse_sort_indices = jnp.argsort(sort_indices)

  # Sizes and offsets
  all_group_sizes = jax.lax.all_gather(local_group_sizes, axis_name=physical_expert_axis_name, axis=0)
  all_send_sizes_3d = jnp.reshape(all_group_sizes, (num_devices, num_devices, local_num_experts))

  metadata = RouterMetadata(
      sort_indices=sort_indices,
      inverse_sort_indices=inverse_sort_indices,
      send_sizes=all_send_sizes_3d,
  )

  selected_experts = jnp.reshape(selected_experts, x.shape[:-1] + (num_experts_per_tok,))
  logits = jnp.reshape(logits, x.shape[:-1] + (num_experts,))
  aux = dsv3_types.DSv3RouterAux(
      group_sizes=local_group_sizes[None, :],
      selected_experts=selected_experts,
      logits=logits,
  )
  return metadata, coeffs, aux


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

  Returns:
    A tuple of (RouterMetadata, coeffs, aux).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec)

  coeffs_spec = jax.sharding.PartitionSpec(token_axis, None)
  token_spec_1d = jax.sharding.PartitionSpec(token_axis)
  token_pspec = x_pspec[:-1]

  metadata_out_spec = RouterMetadata[jax.sharding.PartitionSpec](
      sort_indices=token_spec_1d,
      inverse_sort_indices=token_spec_1d,
      send_sizes=jax.sharding.PartitionSpec(None, None, None),
  )
  aux_out_spec = dsv3_types.DSv3RouterAux[jax.sharding.PartitionSpec](
      group_sizes=jax.sharding.PartitionSpec(token_axis, None),
      selected_experts=jax.sharding.PartitionSpec(*token_pspec, None),
      logits=jax.sharding.PartitionSpec(*token_pspec, None),
  )
  out_specs = (metadata_out_spec, coeffs_spec, aux_out_spec)

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
      ),
      mesh=mesh,
      out_specs=out_specs,
      check_vma=False,
  )(x, w)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_chunk_metadata_impl(
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    current_expert: jt.Num[jax.Array, ""],
    max_capacity: int,
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> ChunkMetadata[jax.Array]:
  """Computes chunk send sizes, next expert index, and group sizes for current device."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  device_index = _get_axis_index(physical_expert_axis_name)

  local_num_experts = full_send_sizes_3d.shape[-1]
  recv_sizes_3d = jnp.swapaxes(full_send_sizes_3d, 0, 1)
  all_expert_sizes = jnp.sum(recv_sizes_3d, axis=1)

  expert_indices = jnp.arange(local_num_experts, dtype=jnp.int32)
  is_active = expert_indices >= current_expert
  active_expert_sizes = jnp.where(is_active[None, :], all_expert_sizes, 0)
  cum_tokens = jnp.cumsum(active_expert_sizes, axis=-1)

  valid_per_device = cum_tokens <= max_capacity
  valid_experts = jnp.all(valid_per_device, axis=0) & is_active
  num_valid = jnp.sum(valid_experts, dtype=jnp.int32)
  chunk_len = jnp.maximum(num_valid, 1)
  next_expert = jnp.minimum(current_expert + chunk_len, local_num_experts)

  chunk_mask = is_active & (expert_indices < next_expert)
  chunk_send_sizes_3d = jnp.where(chunk_mask[None, None, :], full_send_sizes_3d, 0)
  group_sizes = jnp.where(chunk_mask, all_expert_sizes[device_index], 0)

  return ChunkMetadata(
      chunk_send_sizes=chunk_send_sizes_3d,
      next_expert=next_expert,
      group_sizes=group_sizes,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_chunk_metadata(
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    current_expert: jt.Num[jax.Array, ""],
    max_capacity: int,
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> ChunkMetadata[jax.Array]:
  """Computes chunk metadata and group sizes for a chunk across mesh.

  Args:
    full_send_sizes_3d: Full 3D send sizes tensor of shape (N, N, E).
    current_expert: Starting expert index for the chunk.
    max_capacity: Maximum token capacity per device for the chunk.
    expert_axis_name: Logical expert axis name.
    axis_mapping: Logical to physical mesh axis mapping.
    mesh: JAX mesh.

  Returns:
    ChunkMetadata containing chunk_send_sizes, next_expert, and group_sizes.
  """
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  out_specs = ChunkMetadata[jax.sharding.PartitionSpec](
      chunk_send_sizes=jax.sharding.PartitionSpec(None, None, None),
      next_expert=jax.sharding.PartitionSpec(),
      group_sizes=jax.sharding.PartitionSpec(physical_expert_axis_name),
  )
  return jax.shard_map(
      functools.partial(
          dsv3_chunk_metadata_impl,
          max_capacity=max_capacity,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(full_send_sizes_3d, current_expert)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _permute_impl(
    x: jt.Num[jax.Array, "B T D"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> jt.Num[jax.Array, "BT_k D0 D1"]:
  return _permute_impl_fwd(x, metadata, num_experts_per_tok)[0]


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_fwd(
    x: jt.Num[jax.Array, "B T D"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
) -> tuple[
    jt.Num[jax.Array, "BT_k D0 D1"],
    tuple[
        RouterMetadata,
        tuple[int, ...],
    ],
]:
  """Flattens and sorts tokens based on router metadata sort indices."""
  x_flat = jnp.reshape(x, (-1, x.shape[-1] // 128, 128))
  out = _gather_tc(x_flat, metadata.sort_indices // num_experts_per_tok, integer_config=1024)
  return out, (metadata, x.shape)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def _permute_impl_bwd(
    num_experts_per_tok: int,
    res: tuple[
        RouterMetadata,
        tuple[int, ...],
    ],
    grads: jt.Num[jax.Array, "BT_k D0 D1"],
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    None,
]:
  """Backward pass for permute: explicitly gathers then reduces gradients."""
  metadata, x_shape = res
  unsorted_grads = _gather_tc(grads, metadata.inverse_sort_indices, integer_config=2048)
  unsorted_grads_3d = jnp.reshape(
      unsorted_grads,
      (-1, num_experts_per_tok, grads.shape[-2], grads.shape[-1]),
  )
  grad_x_flat = jnp.sum(unsorted_grads_3d, axis=1)
  grad_x = jnp.reshape(grad_x_flat, x_shape)
  return grad_x, None


_permute_impl.defvjp(_permute_impl_fwd, _permute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_permute(
    x: jt.Num[jax.Array, "B T D"],
    metadata: RouterMetadata,
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "BT_k D0 D1"]:
  """Permutes and flattens tokens based on router metadata.

  Args:
    x: Input tokens of shape (B, T, D).
    metadata: Router metadata.
    num_experts_per_tok: Number of experts selected per token.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Permuted tokens sorted by expert of shape (BT_k, D0, D1).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=1)

  return jax.shard_map(
      functools.partial(
          _permute_impl,
          num_experts_per_tok=num_experts_per_tok,
      ),
      mesh=mesh,
      out_specs=jax.sharding.PartitionSpec(token_axis, None, None),
  )(x, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch_chunk_impl(
    data_sorted: jt.Num[jax.Array, "BT_k ..."],
    chunk_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    *,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "kNBT ..."]:
  """Dispatches a chunk of tokens across devices."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  device_index = _get_axis_index(physical_expert_axis_name)

  out_shape = (max_capacity, *data_sorted.shape[1:])
  existing_out = jnp.empty_like(data_sorted, shape=out_shape)

  input_offsets, send_sizes, output_offsets, recv_sizes = _compute_chunk_dispatch_offsets(
      chunk_send_sizes_3d, full_send_sizes_3d, device_index
  )

  routed_data = _ragged_all_to_all_sc0(
      data_sorted,
      existing_out,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
      axis_name=physical_expert_axis_name,
  )
  return routed_data


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_dispatch_chunk(
    x: jt.Num[jax.Array, "BT_k D0 D1"],
    chunk_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    *,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "kNBT D0 D1"]:
  """Dispatches a chunk of tokens to their selected experts across mesh.

  Args:
    x: Permuted tokens of shape (BT_k, D0, D1).
    chunk_send_sizes_3d: Send sizes for the current chunk (N, N, E).
    full_send_sizes_3d: Full send sizes across all chunks (N, N, E).
    max_capacity: Maximum token capacity per device for the chunk.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Routed tokens of shape (kNBT, D0, D1).
  """
  x_pspec = jax.typeof(x).sharding.spec
  token_axis = dsv3_get_token_axis(x_pspec, num_model_dims=2)
  token_spec = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec = ops.physical_pspec(token_spec, axis_mapping)
  out_specs = token_spec
  return jax.shard_map(
      functools.partial(
          dsv3_dispatch_chunk_impl,
          max_capacity=max_capacity,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(x, chunk_send_sizes_3d, full_send_sizes_3d)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches_impl(
    data1: jt.Num[jax.Array, "BT_k ..."],
    data2: jt.Num[jax.Array, "BT_k ..."],
    chunk_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    *,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[jt.Num[jax.Array, "kNBT ..."], jt.Num[jax.Array, "kNBT ..."]]:
  """Dispatches two chunks of tokens across devices in a single SC0 compute_on block.

  Needed only for the backward pass when rematting dispatch and doing the
  transpose of combine.
  """
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  device_index = _get_axis_index(physical_expert_axis_name)

  out_shape1 = (max_capacity, *data1.shape[1:])
  out_shape2 = (max_capacity, *data2.shape[1:])
  existing_out1 = jnp.empty_like(data1, shape=out_shape1)
  existing_out2 = jnp.empty_like(data2, shape=out_shape2)

  input_offsets, send_sizes, output_offsets, recv_sizes = _compute_chunk_dispatch_offsets(
      chunk_send_sizes_3d, full_send_sizes_3d, device_index
  )

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _ra2a_two(
      operand1: jax.Array,
      out1: jax.Array,
      operand2: jax.Array,
      out2: jax.Array,
      input_offsets: jax.Array,
      send_sizes: jax.Array,
      output_offsets: jax.Array,
      recv_sizes: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    routed1 = jax.lax.ragged_all_to_all(
        operand1,
        out1,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=physical_expert_axis_name,
    )
    routed2 = jax.lax.ragged_all_to_all(
        operand2,
        out2,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=physical_expert_axis_name,
    )
    return routed1, routed2

  return _ra2a_two(
      data1,
      existing_out1,
      data2,
      existing_out2,
      input_offsets,
      send_sizes,
      output_offsets,
      recv_sizes,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_aggregated_dispatches(
    x1: jt.Num[jax.Array, "BT_k D0 D1"],
    x2: jt.Num[jax.Array, "BT_k D0 D1"],
    chunk_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    *,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[jt.Num[jax.Array, "kNBT D0 D1"], jt.Num[jax.Array, "kNBT D0 D1"]]:
  """Dispatches two chunks of tokens to their selected experts across mesh.

  Args:
    x1: First permuted tokens of shape (BT_k, D0, D1).
    x2: Second permuted tokens of shape (BT_k, D0, D1).
    chunk_send_sizes_3d: Send sizes for the current chunk (N, N, E).
    full_send_sizes_3d: Full send sizes across all chunks (N, N, E).
    max_capacity: Maximum token capacity per device for the chunk.
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Tuple of routed tokens, each of shape (kNBT, D0, D1).
  """
  x1_pspec = jax.typeof(x1).sharding.spec
  x2_pspec = jax.typeof(x2).sharding.spec
  token_axis = dsv3_get_token_axis(x1_pspec, num_model_dims=2) or dsv3_get_token_axis(x2_pspec, num_model_dims=2)
  token_spec = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec = ops.physical_pspec(token_spec, axis_mapping)
  out_specs = (token_spec, token_spec)
  return jax.shard_map(
      functools.partial(
          dsv3_aggregated_dispatches_impl,
          max_capacity=max_capacity,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(x1, x2, chunk_send_sizes_3d, full_send_sizes_3d)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_chunk_impl(
    routed_data: jt.Num[jax.Array, "kNBT ..."],
    existing_out: jt.Num[jax.Array, "BT_k ..."],
    chunk_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "BT_k ..."]:
  """Combines a chunk of routed expert output tokens across devices."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  device_index = _get_axis_index(physical_expert_axis_name)

  (
      combine_input_offsets,
      send_sizes,
      combine_output_offsets,
      recv_sizes,
  ) = _compute_chunk_combine_offsets(chunk_send_sizes_3d, full_send_sizes_3d, device_index)

  out_sorted = _ragged_all_to_all_sc0(
      routed_data,
      existing_out,
      input_offsets=combine_input_offsets,
      send_sizes=send_sizes,
      output_offsets=combine_output_offsets,
      recv_sizes=recv_sizes,
      axis_name=physical_expert_axis_name,
  )
  return out_sorted


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_combine_chunk(
    routed_data: jt.Num[jax.Array, "kNBT D0 D1"],
    existing_out: jt.Num[jax.Array, "BT_k D0 D1"],
    chunk_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    *,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "BT_k D0 D1"]:
  """Combines a chunk of routed expert output tokens across devices.

  Args:
    routed_data: Routed expert output tokens of shape (kNBT, D0, D1).
    existing_out: Output buffer to write into (BT_k, D0, D1).
    chunk_send_sizes_3d: Send sizes for the current chunk (N, N, E).
    full_send_sizes_3d: Full send sizes across all chunks (N, N, E).
    expert_axis_name: Name of the logical expert axis.
    axis_mapping: Mapping from logical to physical mesh axes.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    Combined expert output tokens of shape (BT_k, D0, D1).
  """
  routed_pspec = jax.typeof(routed_data).sharding.spec
  out_pspec = jax.typeof(existing_out).sharding.spec
  token_axis = dsv3_get_token_axis(routed_pspec, num_model_dims=2) or dsv3_get_token_axis(out_pspec, num_model_dims=2)
  token_spec = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec = ops.physical_pspec(token_spec, axis_mapping)
  return jax.shard_map(
      functools.partial(
          dsv3_combine_chunk_impl,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=token_spec,
  )(routed_data, existing_out, chunk_send_sizes_3d, full_send_sizes_3d)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _unpermute_impl(
    x: jt.Num[jax.Array, "BT_k D0 D1"],
    coeffs: jt.Num[jax.Array, "BT k"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
    seq_length: int,
) -> jt.Num[jax.Array, "B T D"]:
  """Unsorts, scales by router coeffs, and un-flattens combined tokens."""
  return _unpermute_impl_fwd(x, coeffs, metadata, num_experts_per_tok, seq_length)[0]


def _unpermute_impl_fwd(
    x: jt.Num[jax.Array, "BT_k D0 D1"],
    coeffs: jt.Num[jax.Array, "BT k"],
    metadata: RouterMetadata,
    num_experts_per_tok: int,
    seq_length: int,
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    tuple[
        jt.Num[jax.Array, "BT_k D0 D1"],
        jt.Num[jax.Array, "BT k"],
        RouterMetadata,
    ],
]:
  """Forward pass unsorting and scaling tokens by router coefficients."""
  unsorted_x = _gather_tc(x, metadata.inverse_sort_indices, integer_config=2048)
  unsorted_x_3d = jnp.reshape(
      unsorted_x,
      (-1, num_experts_per_tok, unsorted_x.shape[-2], unsorted_x.shape[-1]),
  )
  combined_x_flat = jnp.sum(coeffs[:, :, None, None] * unsorted_x_3d, axis=1)
  out = jnp.reshape(
      combined_x_flat,
      (-1, seq_length, combined_x_flat.shape[-2] * combined_x_flat.shape[-1]),
  )
  return out, (x, coeffs, metadata)


def _unpermute_impl_bwd(
    num_experts_per_tok: int,
    seq_length: int,
    res: tuple[
        jt.Num[jax.Array, "BT_k D0 D1"],
        jt.Num[jax.Array, "BT k"],
        RouterMetadata,
    ],
    cotangent: jt.Num[jax.Array, "B T D"],
) -> tuple[jt.Num[jax.Array, "BT_k D0 D1"], jt.Num[jax.Array, "BT k"], None]:
  """Backward pass computing gradients with respect to unpermuted tokens and coeffs."""
  del seq_length
  x, coeffs, metadata = res
  unsorted_x = _gather_tc(x, metadata.inverse_sort_indices, integer_config=2048)
  unsorted_x_3d = jnp.reshape(
      unsorted_x,
      (-1, num_experts_per_tok, unsorted_x.shape[-2], unsorted_x.shape[-1]),
  )
  grads_flat = jnp.reshape(cotangent, (-1, cotangent.shape[-1] // 128, 128))

  weighted_grads = jnp.reshape(
      coeffs[:, :, None, None] * grads_flat[:, None, :, :],
      (-1, grads_flat.shape[-2], grads_flat.shape[-1]),
  )
  grad_x = _gather_tc(weighted_grads, metadata.sort_indices, integer_config=2048)

  grad_coeffs = jnp.sum(grads_flat[:, None, :, :] * unsorted_x_3d, axis=(-2, -1))
  return grad_x, grad_coeffs, None


_unpermute_impl.defvjp(_unpermute_impl_fwd, _unpermute_impl_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute(
    x: jt.Num[jax.Array, "BT_k D0 D1"],
    coeffs: jt.Num[jax.Array, "BT k"],
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    local_seq_length: int,
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
) -> jt.Num[jax.Array, "B T D"]:
  """Unpermutes and reduces combined tokens back to original token layout.

  Args:
    x: Combined expert tokens of shape (BT_k, D0, D1).
    coeffs: Routing coefficients of shape (BT, k).
    metadata: Router metadata.
    num_experts_per_tok: Number of experts selected per token.
    local_seq_length: Sequence length per shard.
    mesh: JAX mesh over which the data and model are sharded.
    out_specs: Output PartitionSpec.

  Returns:
    Output tokens of shape (B, T, D).
  """
  return jax.shard_map(
      functools.partial(
          _unpermute_impl,
          num_experts_per_tok=num_experts_per_tok,
          seq_length=local_seq_length,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(x, coeffs, metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_unpermute_bwd(
    cotangent: jt.Num[jax.Array, "B T D"],
    x: jt.Num[jax.Array, "BT_k D0 D1"],
    coeffs: jt.Num[jax.Array, "BT k"],
    metadata: RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT_k D0 D1"],
    jt.Num[jax.Array, "BT k"],
]:
  """Backward pass for unpermute computing gradients wrt combined tokens and coeffs.

  Args:
    cotangent: Gradients with respect to unpermute output of shape (B, T, D).
    x: Combined expert tokens of shape (BT_k, D0, D1).
    coeffs: Routing coefficients of shape (BT, k).
    metadata: Router metadata.
    num_experts_per_tok: Number of experts selected per token.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    A tuple of (grad_x, grad_coeffs).
  """
  token_axis = dsv3_get_token_axis(jax.typeof(x).sharding.spec, num_model_dims=2)
  token_spec_3d = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec_2d = jax.sharding.PartitionSpec(token_axis, None)
  out_specs = (token_spec_3d, token_spec_2d)

  def _bwd_impl(cotangent, x, coeffs, metadata):
    grad_x, grad_coeffs, _ = _unpermute_impl_bwd(num_experts_per_tok, 0, (x, coeffs, metadata), cotangent)
    return grad_x, grad_coeffs

  return jax.shard_map(
      _bwd_impl,
      mesh=mesh,
      out_specs=out_specs,
  )(cotangent, x, coeffs, metadata)


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
  bt, num_e = pre_bias_logits.shape
  flat_logits = jnp.ravel(pre_bias_logits.astype(jnp.float32))
  flat_indices = jnp.ravel(indices + jnp.arange(bt, dtype=indices.dtype)[:, None] * num_e)

  # We don't pin to SC0 if padding is required because it results in
  # an XLA SC fusion error. This code path is only used for the tiny test.
  if flat_indices.shape[0] % 1024 != 0:
    coeffs = (
        flat_logits.at[flat_indices]
        .get(mode="promise_in_bounds", wrap_negative_indices=False)
        .reshape(bt, num_experts_per_tok)
        .astype(pre_bias_logits.dtype)
    )
  else:

    @compute_on(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )
    def _gather(flat_a: jax.Array, flat_idx: jax.Array) -> jax.Array:
      return flat_a.at[flat_idx].get(mode="promise_in_bounds", wrap_negative_indices=False)

    coeffs = _gather(flat_logits, flat_indices).reshape(bt, num_experts_per_tok).astype(pre_bias_logits.dtype)
  coeffs = routed_scaling_factor * (coeffs / coeffs.sum(-1, keepdims=True))
  return indices, coeffs


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
) -> tuple[
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "E"],
    jt.Num[jax.Array, "BT E"],
]:
  """Selects experts for each token and calculates group sizes for each expert.

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

  Returns:
    A tuple of (selected_experts, coeffs, group_sizes, logits).
  """
  assert w.kernel is not None
  assert w.bias is not None
  pre_bias_logits = jax.nn.sigmoid(jnp.tensordot(x, w.kernel, axes=1))
  logits = pre_bias_logits + w.bias

  selected_experts, coeffs = expert_indices_and_coeffs(
      logits,
      pre_bias_logits,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
  )

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [0]}},
  )
  def _bincount(x: jax.Array) -> jax.Array:
    return jnp.bincount(x, length=num_experts)

  group_sizes = _bincount(jnp.ravel(selected_experts))
  return selected_experts, coeffs, group_sizes, logits
