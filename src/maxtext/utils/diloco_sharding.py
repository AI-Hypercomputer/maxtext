# Copyright 2023-2026 Google LLC
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

"""Mesh sharding, batch dimension reshaping, and metric extraction utilities for DiLoCo."""

from collections.abc import Sequence

import drjax
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import numpy as np


def add_diloco_to_sharding(pytree: PyTree) -> PyTree:
  """Recursively traverses a PyTree and prepends 'diloco' to the PartitionSpec
  of any NamedSharding object that contains 'diloco' in its mesh axis names.
  """

  def map_fn(leaf):
    if isinstance(leaf, jax.sharding.NamedSharding):
      if "diloco" not in leaf.mesh.axis_names:
        return leaf
      new_spec = jax.sharding.PartitionSpec("diloco", *leaf.spec)
      return jax.sharding.NamedSharding(mesh=leaf.mesh, spec=new_spec)
    return leaf

  return jax.tree_util.tree_map(map_fn, pytree)


def reshape_first_axis_with_diloco(num_diloco_replicas: int, pytree: PyTree) -> PyTree:
  """Reshapes the first dimension of each array in the PyTree to include a DiLoCo axis."""

  def extend_pspec(
      pspec: jax.sharding.PartitionSpec | Sequence[str | Sequence[str]] = (),
  ) -> jax.sharding.PartitionSpec:
    if pspec and isinstance(pspec[0], (tuple, list)) and len(pspec[0]) > 0 and pspec[0][0] == "diloco":
      remaining = tuple(pspec[0][1:])
      if len(remaining) == 1:
        return jax.sharding.PartitionSpec("diloco", remaining[0], *pspec[1:])
      elif len(remaining) > 1:
        return jax.sharding.PartitionSpec("diloco", remaining, *pspec[1:])
      else:
        return jax.sharding.PartitionSpec("diloco", *pspec[1:])
    return jax.sharding.PartitionSpec("diloco", *pspec)

  def reshape_for_diloco(arr):
    if not hasattr(arr, "shape"):
      return arr
    if (
        hasattr(arr, "ndim")
        and arr.ndim >= 3
        and arr.shape[0] == num_diloco_replicas
        and hasattr(arr, "sharding")
        and isinstance(arr.sharding, jax.sharding.NamedSharding)
        and arr.sharding.spec
        and arr.sharding.spec[0] == "diloco"
        and isinstance(arr.sharding.spec[0], str)
    ):
      return arr
    batch_dim, *example_shape = arr.shape
    if batch_dim % num_diloco_replicas != 0:
      raise ValueError(f"Batch dimension {batch_dim} is not divisible by num_diloco_replicas {num_diloco_replicas}.")
    diloco_shape = (num_diloco_replicas, batch_dim // num_diloco_replicas, *example_shape)
    if hasattr(arr, "sharding") and arr.sharding is not None:
      s = arr.sharding
      s = jax.sharding.NamedSharding(mesh=s.mesh, spec=extend_pspec(s.spec))
      return jax.lax.with_sharding_constraint(jnp.reshape(arr, shape=diloco_shape), s)
    return jnp.reshape(arr, shape=diloco_shape)

  return jax.tree.map(reshape_for_diloco, pytree)


def extract_replica_0(metrics: PyTree) -> PyTree:
  """Extracts metrics from replica 0 across DiLoCo islands."""

  def select_first_replica(x):
    if not hasattr(x, "shape") or len(x.shape) == 0:
      return x
    r = x.shape[0]
    mask = (jnp.arange(r) == 0).reshape((r,) + (1,) * (x.ndim - 1))
    return drjax.reduce_sum(x * mask)

  return jax.tree.map(select_first_replica, metrics)


def extract_per_island_metrics(metrics: PyTree, num_diloco_replicas: int) -> PyTree:
  """Extracts replica 0 metrics and appends per-island loss metrics."""
  default_metrics = extract_replica_0(metrics)
  if isinstance(metrics, dict) and "scalar" in metrics and "learning/loss" in metrics["scalar"]:
    for i in range(num_diloco_replicas):
      mask_i = jnp.arange(num_diloco_replicas) == i
      loss_i = drjax.reduce_sum(metrics["scalar"]["learning/loss"] * mask_i)
      default_metrics["scalar"][f"learning/loss_island_{i}"] = loss_i
  return default_metrics


def split_mesh_along_axis(mesh: jax.sharding.Mesh, axis_name: str = "diloco") -> list[jax.sharding.Mesh]:
  """Splits `mesh` into one submesh per index of `axis_name`, each without that axis.

  Submesh `i` holds the devices at position `i` along `axis_name`, with the remaining axes (and their axis types) in
  their original order, so any PartitionSpec that does not mention `axis_name` is valid on every submesh.
  """
  if axis_name not in mesh.axis_names:
    raise ValueError(f"Axis {axis_name!r} not found in mesh axis names {mesh.axis_names}.")
  axis_index = mesh.axis_names.index(axis_name)
  keep = [i for i in range(len(mesh.axis_names)) if i != axis_index]
  axis_names = tuple(mesh.axis_names[i] for i in keep)
  axis_types = tuple(mesh.axis_types[i] for i in keep)
  return [
      jax.sharding.Mesh(np.take(mesh.devices, i, axis=axis_index), axis_names, axis_types=axis_types)
      for i in range(mesh.shape[axis_name])
  ]


def remove_mesh_axis_from_rules(
    logical_axis_rules: Sequence[tuple[str, str | Sequence[str] | None]], axis_name: str = "diloco"
) -> list[tuple[str, str | tuple[str, ...] | None]]:
  """Drops `axis_name` from the physical axes of every logical axis rule.

  Rules that map only to `axis_name` are removed, so the logical axis falls through to the next matching rule, which
  is what a mesh built by `split_mesh_along_axis` needs.
  """
  new_rules = []
  for logical_axis, physical_axes in logical_axis_rules:
    if isinstance(physical_axes, str):
      if physical_axes == axis_name:
        continue
    elif physical_axes is not None:
      stripped = tuple(ax for ax in physical_axes if ax != axis_name)
      if physical_axes and not stripped:
        continue
      physical_axes = stripped
    new_rules.append((logical_axis, physical_axes))
  return new_rules
