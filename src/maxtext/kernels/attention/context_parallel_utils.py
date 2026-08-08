# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared helpers for attention context-parallel sharding metadata."""

from __future__ import annotations

from typing import Any

import jax


def mesh_axes_for_dim(axis_names: Any) -> tuple[Any, ...]:
  """Returns the mesh axes attached to one tensor dimension."""
  if axis_names is None:
    return ()
  if isinstance(axis_names, str):
    return (axis_names,)
  return tuple(axis for axis in axis_names if axis is not None)


def mesh_axes_size(mesh: Any, axes: tuple[Any, ...], *, label: str) -> int:
  """Returns the product of mesh sizes for a set of axes."""
  size = 1
  for axis in axes:
    if axis not in mesh.shape:
      raise ValueError(f"{label} requires mesh axis {axis!r} to exist.")
    size *= mesh.shape[axis]
  return size


def with_axis_on_dim(axis_names: Any, axis: Any, dim: int) -> Any:
  """Returns sharding axis names with one dimension replaced."""
  axes = list(axis_names)
  axes[dim] = axis
  if isinstance(axis_names, jax.sharding.PartitionSpec):
    return jax.sharding.PartitionSpec(*axes, unreduced=axis_names.unreduced, reduced=axis_names.reduced)
  if isinstance(axis_names, tuple):
    return tuple(axes)
  return axes
