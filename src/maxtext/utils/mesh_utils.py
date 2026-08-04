# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mesh utilities for DiLoCo stack operations across submeshes."""

from typing import Any, Sequence
import jax
import jax.numpy as jnp
from jax import sharding
from jax.experimental.layout import Format, Layout
import numpy as np

from pathwaysutils.experimental.concatenate_by_mesh_axis import concatenate_by_mesh_axis


def partition_mesh_by_diloco_axis(
    global_mesh: jax.sharding.Mesh, num_replicas: int, diloco_axis_name: str = "diloco"
) -> list[jax.sharding.Mesh]:
  """Slices a global mesh along the diloco axis into multiple submeshes. Won't keep diloco dim."""
  if diloco_axis_name not in global_mesh.axis_names:
    raise ValueError(f"Axis {diloco_axis_name} not found in mesh axis names: {global_mesh.axis_names}")

  diloco_axis_index = global_mesh.axis_names.index(diloco_axis_name)
  diloco_axis_size = global_mesh.shape[diloco_axis_name]

  if diloco_axis_size != num_replicas:
    raise ValueError(f"Diloco axis size ({diloco_axis_size}) must match num_replicas ({num_replicas})")

  devices = global_mesh.devices
  submeshes = []
  axis_names = list(global_mesh.axis_names)
  axis_names.remove(diloco_axis_name)

  for i in range(num_replicas):
    sub_devices = np.take(devices, i, axis=diloco_axis_index)
    submesh = jax.sharding.Mesh(sub_devices, axis_names)
    submeshes.append(submesh)

  return submeshes


def _insert_axis_into_spec(
    spec: sharding.PartitionSpec,
    axis_index: int,
    axis_name: str | None,
) -> sharding.PartitionSpec:
  spec_list = list(spec)
  while len(spec_list) < axis_index:
    spec_list.append(None)
  spec_list.insert(axis_index, axis_name)
  return sharding.PartitionSpec(*spec_list)


def _replace_axis_in_spec(
    spec: sharding.PartitionSpec,
    axis_index: int,
    axis_name: str,
) -> sharding.PartitionSpec:
  spec_list = list(spec)
  spec_list[axis_index] = axis_name
  return sharding.PartitionSpec(*spec_list)


def _get_spec(leaf: jax.Array) -> sharding.PartitionSpec:
  if not isinstance(leaf.sharding, sharding.NamedSharding):
    raise ValueError(f"Expected NamedSharding, got {leaf.sharding=!r}")
  return leaf.sharding.spec


def _get_mesh_from_tree(tree: Any) -> sharding.Mesh:
  for leaf in jax.tree.leaves(tree):
    if isinstance(leaf, jax.Array):
      if isinstance(leaf.sharding, sharding.NamedSharding):
        if isinstance(leaf.sharding.mesh, sharding.Mesh):
          return leaf.sharding.mesh
        raise ValueError(f"Expected Mesh, got {leaf.sharding.mesh=!r}")
      raise ValueError(f"Expected NamedSharding, got {leaf.sharding=!r}")
  raise ValueError("PyTree has no jax.Array leaves.")


def _expand_mesh_by_axis(
    mesh: sharding.Mesh,
    axis_index: int,
    axis_name: str,
    axis_type: sharding.AxisType = sharding.AxisType.Auto,
) -> sharding.Mesh:
  axis_names = (
      *mesh.axis_names[:axis_index],
      axis_name,
      *mesh.axis_names[axis_index:],
  )
  axis_types = (
      *mesh.axis_types[:axis_index],
      axis_type,
      *mesh.axis_types[axis_index:],
  )
  devices = np.expand_dims(mesh.devices, axis=axis_index)
  return sharding.Mesh(devices, axis_names=axis_names, axis_types=axis_types)


def _expand_tree_on_mesh(
    tree: Any,
    mesh: sharding.Mesh,
    axis_index_to_expand: int,
    out_specs: Any,
    donate: bool = True,
) -> Any:
  """Lowers and compiles expand_dims on a physical submesh using pure NamedSharding."""
  def _expand_distributed_axis(t):
    return jax.tree.map(
        lambda x: jnp.expand_dims(x, axis=axis_index_to_expand)
        if isinstance(x, jax.Array)
        else x,
        t,
    )

  def _leaf_struct(leaf):
    if isinstance(leaf, jax.Array):
      return jax.ShapeDtypeStruct(
          leaf.shape, leaf.dtype
      )
    return leaf

  in_structs = jax.tree.map(_leaf_struct, tree)

  def _leaf_in_sharding(leaf):
    if isinstance(leaf, jax.Array):
      return sharding.NamedSharding(
          mesh=mesh,
          spec=_get_spec(leaf),
          memory_kind=leaf.sharding.memory_kind if hasattr(leaf.sharding, "memory_kind") else None,
      )
    return leaf

  in_shardings = jax.tree.map(_leaf_in_sharding, tree)

  def _leaf_out_sharding(spec, leaf):
    if isinstance(leaf, jax.Array):
      return sharding.NamedSharding(
          mesh=mesh,
          spec=spec,
          memory_kind=leaf.sharding.memory_kind if hasattr(leaf.sharding, "memory_kind") else None,
      )
    return leaf

  out_shardings = jax.tree.map(_leaf_out_sharding, out_specs, tree)

  lowered = (
      jax.jit(
          _expand_distributed_axis,
          in_shardings=(in_shardings,),
          out_shardings=out_shardings,
          donate_argnums=0 if donate else None,
      )
      .trace(in_structs)
      .lower(lowering_platforms=("cpu",))
  )
  compiled = lowered.compile(device_assignment=tuple(mesh.devices.flat))
  return compiled(tree)


def _put_tree_on_expanded_mesh(
    tree: Any,
    expanded_mesh: sharding.Mesh,
    axis_index: int,
    axis_name: str,
) -> Any:
  def _target_sharding(arr):
    if not isinstance(arr, jax.Array):
      return None
    if not isinstance(arr.sharding, sharding.NamedSharding):
      raise ValueError(f"Expected NamedSharding, got {arr.sharding=!r}")
    target_spec = _replace_axis_in_spec(
        arr.sharding.spec,
        axis_index=axis_index,
        axis_name=axis_name,
    )
    return sharding.NamedSharding(
        mesh=expanded_mesh,
        spec=target_spec,
        memory_kind=arr.sharding.memory_kind if hasattr(arr.sharding, "memory_kind") else None,
    )

  target_shardings = jax.tree.map(_target_sharding, tree)
  return jax.device_put(tree, target_shardings)


def stack_across_meshes_pytree(
    pytrees: Sequence[Any],
    global_mesh: jax.sharding.Mesh,
    axis_name: str,
) -> Any:
  """Stacks a list of PyTrees across submeshes into a single global PyTree."""
  if not pytrees:
    return pytrees

  meshes = [_get_mesh_from_tree(tree) for tree in pytrees]

  def _leaf_expanded_spec(leaf):
    if isinstance(leaf, jax.Array):
      return _insert_axis_into_spec(
          _get_spec(leaf),
          axis_index=0,
          axis_name=None,
      )
    return leaf

  specs_with_expanded_axis = jax.tree.map(
      _leaf_expanded_spec,
      pytrees[0],
  )

  expanded_trees_on_learner = [
      _expand_tree_on_mesh(
          tree,
          mesh,
          axis_index_to_expand=0,
          out_specs=specs_with_expanded_axis,
          donate=True,
      )
      for tree, mesh in zip(pytrees, meshes, strict=True)
  ]

  expanded_meshes = [
      _expand_mesh_by_axis(
          mesh,
          axis_index=0,
          axis_name=axis_name,
          axis_type=sharding.AxisType.Auto,
      )
      for mesh in meshes
  ]

  expanded_trees_on_expanded_mesh = [
      _put_tree_on_expanded_mesh(
          tree,
          expanded_mesh=mesh,
          axis_index=0,
          axis_name=axis_name,
      )
      for tree, mesh in zip(
          expanded_trees_on_learner, expanded_meshes, strict=True
      )
  ]

  return concatenate_by_mesh_axis(expanded_trees_on_expanded_mesh, axis_name)

