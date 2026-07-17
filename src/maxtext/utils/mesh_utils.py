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

from typing import Any
import jax
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


def _expand_array_dims_with_mesh(
    x: jax.Array,
    axis_name: str,
) -> jax.Array:
  """Expands array dimensions by introducing a new dim-1 at index 0 and expanding its mesh."""
  sharding = x.sharding
  assert isinstance(sharding, jax.sharding.NamedSharding)
  submesh = sharding.mesh

  expanded_devices = np.expand_dims(np.array(submesh.devices), axis=0)
  expanded_mesh = jax.sharding.Mesh(expanded_devices, axis_names=(axis_name,) + submesh.axis_names)
  expanded_sharding = jax.sharding.NamedSharding(
      expanded_mesh, jax.sharding.PartitionSpec(axis_name, *sharding.spec), memory_kind=sharding.memory_kind
  )

  # Pathways caches all jit-compiled ops (expand_dims, device_put_reshard) keyed by
  # shape/dtype/sharding WITHOUT layout. Different learner slices or jnp.take outputs
  # can produce arrays with different layouts (null vs tiled) for the same logical tensor,
  # causing the cached jit to reject the second layout variant.
  # Shard-level construction avoids every layout-sensitive jit entirely: np.expand_dims
  # is a pure-numpy op and make_array_from_single_device_arrays is a metadata operation.
  local_arrays = [
      jax.device_put(
          np.expand_dims(np.asarray(shard.data), axis=0),
          jax.sharding.SingleDeviceSharding(shard.device),
      )
      for shard in x.addressable_shards
  ]
  return jax.make_array_from_single_device_arrays(
      shape=(1,) + x.shape,
      sharding=expanded_sharding,
      arrays=local_arrays,
  )


def stack_across_meshes_pytree(trees: list[Any], global_mesh: jax.sharding.Mesh, axis_name: str) -> Any:
  """Stacks a list of PyTrees across submeshes into a single global PyTree."""
  # 1. Expand dimensions of all arrays in all PyTrees manually
  expanded_trees = []
  for tree in trees:
    exp_tree = jax.tree_util.tree_map(lambda x: _expand_array_dims_with_mesh(x, axis_name), tree)
    expanded_trees.append(exp_tree)

  # 2. Concatenate along the mesh axis using pathwaysutils
  return concatenate_by_mesh_axis(expanded_trees, mesh_axis=axis_name)
