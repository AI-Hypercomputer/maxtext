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

"""Mesh utilities for DiLoCo stack/unstack operations across submeshes."""

from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

from pathwaysutils.experimental.concatenate_by_mesh_axis import concatenate_by_mesh_axis
from pathwaysutils.experimental.split_by_mesh_axis import split_by_mesh_axis


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
  # TODO: verify the correctness of this function. compare to the g3 implementation.
  # 1. Expand the array shape natively to prepending a size-1 dimension
  expanded_x = jnp.expand_dims(x, axis=0)

  # 2. Get original sharding
  sharding = x.sharding
  assert isinstance(sharding, jax.sharding.NamedSharding)
  submesh = sharding.mesh

  # 3. Build expanded mesh (with axis_name of size 1 at the beginning)
  expanded_devices = np.expand_dims(np.array(submesh.devices), axis=0)
  expanded_mesh = jax.sharding.Mesh(expanded_devices, axis_names=(axis_name,) + submesh.axis_names)

  # 4. Build expanded NamedSharding and reshard (metadata-only operation in JAX)
  expanded_sharding = jax.sharding.NamedSharding(
      expanded_mesh, jax.sharding.PartitionSpec(axis_name, *sharding.spec), memory_kind=sharding.memory_kind
  )

  return jax.device_put(expanded_x, expanded_sharding)


def stack_across_meshes_pytree(trees: list[Any], global_mesh: jax.sharding.Mesh, axis_name: str) -> Any:
  """Stacks a list of PyTrees across submeshes into a single global PyTree."""
  # pathwaysutils' concatenate primitive always donates its inputs and its CPU
  # implementation is not safe when the source submeshes and destination mesh
  # overlap in one PJRT client. Use ordinary non-donating JAX on CPU; TPU keeps
  # the zero-copy Pathways mesh concatenation used in production.
  if all(device.platform == "cpu" for device in global_mesh.devices.flat):

    def stack_leaves(*leaves):
      source_sharding = leaves[0].sharding
      assert isinstance(source_sharding, jax.sharding.NamedSharding)
      output_sharding = jax.sharding.NamedSharding(
          global_mesh,
          jax.sharding.PartitionSpec(axis_name, *source_sharding.spec),
          memory_kind=source_sharding.memory_kind,
      )

      # A single jitted computation cannot accept arguments committed to
      # disjoint CPU meshes. Materialize through host memory in the CPU-only
      # fallback, then place the combined value on the global mesh.
      return jax.device_put(np.stack([np.asarray(value) for value in leaves], axis=0), output_sharding)

    return jax.tree_util.tree_map(stack_leaves, *trees)

  # 1. Expand dimensions of all arrays in all PyTrees manually
  expanded_trees = []
  for tree in trees:
    exp_tree = jax.tree_util.tree_map(lambda x: _expand_array_dims_with_mesh(x, axis_name), tree)
    expanded_trees.append(exp_tree)

  # 2. Concatenate along the mesh axis using pathwaysutils
  return concatenate_by_mesh_axis(expanded_trees, mesh_axis=axis_name)


def unstack_across_meshes_pytree(
    global_tree: Any,
    submeshes: list[jax.sharding.Mesh],
    axis_name: str,
) -> list[Any]:
  """Unstacks/splits a global PyTree into a list of submesh-local PyTrees."""
  num_replicas = len(submeshes)
  first_leaf = jax.tree_util.tree_leaves(global_tree)[0]
  if all(device.platform == "cpu" for device in first_leaf.sharding.device_set):

    def slice_tree(replica_idx, submesh):
      def slice_leaf(x):
        assert isinstance(x.sharding, jax.sharding.NamedSharding)
        target_sharding = jax.sharding.NamedSharding(
            submesh,
            jax.sharding.PartitionSpec(*x.sharding.spec[1:]),
            memory_kind=x.sharding.memory_kind,
        )

        return jax.device_put(np.asarray(x)[replica_idx], target_sharding)

      return jax.tree_util.tree_map(slice_leaf, global_tree)

    return [slice_tree(i, submeshes[i]) for i in range(num_replicas)]

  # 1. Split the global PyTree along the mesh axis using pathwaysutils
  split_trees = split_by_mesh_axis(global_tree, mesh_axis=axis_name)

  # 2. Squeeze out the leading size-1 axis and reshard each split tree to its submesh
  def squeeze_and_reshard_tree(tree, submesh):
    def squeeze_leaf(x):
      # Squeeze leading size-1 dimension
      squeezed = x[0]
      # Reshard to target submesh
      submesh_sharding = jax.sharding.NamedSharding(submesh, squeezed.sharding.spec)
      return jax.device_put(squeezed, submesh_sharding)

    return jax.tree_util.tree_map(squeeze_leaf, tree)

  return [squeeze_and_reshard_tree(split_trees[i], submeshes[i]) for i in range(num_replicas)]
