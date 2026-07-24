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

"""Mesh and layout utilities for non-SPMD DiLoCo."""

import functools
import threading
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

from pathwaysutils.experimental.concatenate_by_mesh_axis import concatenate_by_mesh_axis


def jit_with_layout_canonicalized_inputs(
    fun,
    *,
    in_shardings,
    out_shardings,
    donate_argnums=(),
):
  """JITs ``fun`` and adapts calls to the executable's physical formats.

  A logical ``NamedSharding`` does not describe a device-local physical layout.
  In particular, a TPU-compiled executable may require a tiled input while a
  transferred or restored value with the same logical sharding has a different
  layout.  The compiled executable is the authority: each call is placed into
  its concrete ``input_formats`` before execution.

  ``donate_argnums`` applies to both the format conversion and the executable.
  Callers must not retain or use donated arguments.
  """
  donate_argnums = tuple(donate_argnums)
  donated = frozenset(donate_argnums)
  jitted = jax.jit(
      fun,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      donate_argnums=donate_argnums,
  )
  executable = None
  input_formats = None
  compile_lock = threading.Lock()

  @functools.wraps(fun)
  def call(*args):
    nonlocal executable, input_formats
    if executable is None:
      with compile_lock:
        if executable is None:
          executable = jitted.lower(*args).compile()
          input_formats, keyword_formats = executable.input_formats
          if keyword_formats:
            raise ValueError("Layout-canonicalized JIT does not support keyword arguments")

    donate_mask = tuple(i in donated for i in range(len(args)))
    formatted_args = jax.device_put(args, input_formats, donate=donate_mask)
    return executable(*formatted_args)

  return call


@functools.cache
def _make_layout_safe_expand_dims(input_sharding, output_sharding, shape, dtype):
  """Caches one format-adapted leading-dimension expansion per signature."""
  del shape, dtype  # They intentionally specialize the cache key.
  return jit_with_layout_canonicalized_inputs(
      lambda value: jnp.expand_dims(value, axis=0),
      in_shardings=input_sharding,
      out_shardings=output_sharding,
  )


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
  retained_axes = [i for i, name in enumerate(global_mesh.axis_names) if name != diloco_axis_name]
  axis_names = tuple(global_mesh.axis_names[i] for i in retained_axes)
  axis_types = tuple(global_mesh.axis_types[i] for i in retained_axes)

  for i in range(num_replicas):
    sub_devices = np.take(devices, i, axis=diloco_axis_index)
    submesh = jax.sharding.Mesh(sub_devices, axis_names, axis_types=axis_types)
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
  expanded_mesh = jax.sharding.Mesh(
      expanded_devices,
      axis_names=(axis_name,) + submesh.axis_names,
      axis_types=(jax.sharding.AxisType.Auto,) + submesh.axis_types,
  )
  expanded_sharding = jax.sharding.NamedSharding(
      expanded_mesh,
      jax.sharding.PartitionSpec(axis_name, *sharding.spec),
      memory_kind=sharding.memory_kind,
  )

  # Never round-trip a remote CPU shard through controller-host NumPy. Besides
  # doubling the fragment, np.asarray(shard.data) centralizes all remote shards
  # in the single Pathways client. Adapt the small compiled operation to its
  # executable-selected layout instead.
  expand_dims = _make_layout_safe_expand_dims(sharding, expanded_sharding, x.shape, x.dtype)
  return expand_dims(x)


def stack_across_meshes_pytree(trees: list[Any], global_mesh: jax.sharding.Mesh, axis_name: str) -> Any:
  """Stacks a list of PyTrees across submeshes into a single global PyTree."""
  del global_mesh  # Retained for compatibility with the existing public API.
  # 1. Expand dimensions of all arrays in all PyTrees manually
  expanded_trees = []
  for tree in trees:
    exp_tree = jax.tree_util.tree_map(lambda x: _expand_array_dims_with_mesh(x, axis_name), tree)
    expanded_trees.append(exp_tree)

  # 2. Concatenate along the mesh axis using pathwaysutils
  return concatenate_by_mesh_axis(expanded_trees, mesh_axis=axis_name)
