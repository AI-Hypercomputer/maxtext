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

"""Utilities for reshaping and resharding tensors for Muon."""

import itertools
import math
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import jax
import jax.numpy as jnp

ReshapeFn = Callable[[jax.Array], jax.Array]
NamedSharding = jax.sharding.NamedSharding
TensorWithAxes = tuple[jax.Array, tuple[int, ...], tuple[int, ...], NamedSharding | None]


def get_dim_mesh_axes(
    sharding: jax.sharding.NamedSharding | None,
    dim: int,
) -> tuple[str, ...]:
  """Returns the mesh axes partitioned along a given array dimension.

  Args:
    sharding: The named sharding to extract partition spec from, or None.
    dim: The 0-based dimension index.

  Returns:
    A tuple of mesh axis names sharding dimension `dim`. Returns an empty tuple
    `()` if `sharding` is None, if the dimension is unpartitioned (`None`), if
    `dim < 0`, or if `dim >= len(sharding.spec)` (which occurs when trailing
    dimensions are omitted in the PartitionSpec and are replicated).
  """
  if sharding is None or dim < 0 or dim >= len(sharding.spec):
    return ()
  axis = sharding.spec[dim]
  if axis is None:
    return ()
  elif isinstance(axis, str):
    return (axis,)
  return tuple(axis)


def get_flat_mesh_axes(
    sharding: jax.sharding.NamedSharding | None,
    dims: Iterable[int],
) -> tuple[str, ...]:
  """Returns a concatenated tuple of mesh axes across specified dimensions.

  Args:
    sharding: The named sharding to extract partition specs from, or None.
    dims: Sequence of 0-based dimension indices to extract and concatenate.

  Returns:
    A single tuple containing all mesh axis names along the given dimensions in
    order.
  """
  if sharding is None:
    return ()
  axes = []
  for dim in dims:
    axes.extend(get_dim_mesh_axes(sharding, dim))
  return tuple(axes)


def _reshape(
    x: jax.Array,
    shape: tuple[int, ...],
    sharding: NamedSharding,
    *,
    explicit_axes: bool,
) -> jax.Array:
  """Reshapes `x`, passing `out_sharding` when `sharding` uses Explicit axes."""
  if explicit_axes:
    return jnp.reshape(x, shape, out_sharding=sharding)
  return jnp.reshape(x, shape)


def _get_3d_flatten_fns(
    shape: tuple[int, ...],
    reduction_axes: tuple[int, ...],
    output_axes: tuple[int, ...],
    sharding: NamedSharding | None,
) -> tuple[ReshapeFn, ReshapeFn, tuple[int, int, int], NamedSharding | None]:
  """Builds local transpose/reshape functions between `shape` and 3D `[batch, row, col]`."""
  ndim = len(shape)
  reduction_axes = tuple(ax % ndim for ax in reduction_axes)
  output_axes = tuple(ax % ndim for ax in output_axes)
  batch_axes = tuple(sorted(set(range(ndim)) - set(reduction_axes) - set(output_axes)))

  reduction_size = math.prod(shape[ax] for ax in reduction_axes)
  output_size = math.prod(shape[ax] for ax in output_axes)

  # Ensure row_size <= col_size so X @ X.T in Newton-Schulz has the smaller
  # [batch, row_size, row_size] shape, exploiting f(X^T)^T = f(X).
  # Note: `_pack_padded_group` emits canonical 3D tensors with output_axes=(1,)
  # and reduction_axes=(2,) so this tie-breaker keeps row_axes=(1,) even when
  # row_size == col_size, hitting `is_canonical_3d` below.
  if output_size > reduction_size:
    row_axes, col_axes = reduction_axes, output_axes
    row_size, col_size = reduction_size, output_size
  else:
    row_axes, col_axes = output_axes, reduction_axes
    row_size, col_size = output_size, reduction_size

  # Place sharded batch axes before unsharded batch axes so flattening batch
  # dims is a purely local reshape when at most one batch axis is sharded (or
  # leading sharded batch axes have local size 1).
  sharded_batch_axes = tuple(ax for ax in batch_axes if get_dim_mesh_axes(sharding, ax))
  unsharded_batch_axes = tuple(ax for ax in batch_axes if not get_dim_mesh_axes(sharding, ax))

  perm = sharded_batch_axes + unsharded_batch_axes + row_axes + col_axes
  permuted_shape = tuple(shape[ax] for ax in perm)
  inv_perm = [perm.index(i) for i in range(ndim)]
  is_canonical_3d = batch_axes == (0,) and row_axes == (1,) and col_axes == (2,)

  shape_3d = (math.prod(shape[ax] for ax in batch_axes), row_size, col_size)
  if sharding is None:
    if is_canonical_3d:
      return (lambda y: y), (lambda y_3d: y_3d), shape_3d, None
    return (
        lambda y: jnp.transpose(y, perm).reshape(shape_3d),
        lambda y_3d: jnp.reshape(y_3d, permuted_shape).transpose(inv_perm),
        shape_3d,
        None,
    )

  explicit_axes = is_explicit_axes(sharding)
  mesh = sharding.mesh
  permuted_sharding = NamedSharding(
      mesh,
      jax.P(*(get_dim_mesh_axes(sharding, ax) for ax in perm)),
  )
  sharding_3d = NamedSharding(
      mesh,
      jax.P(
          get_flat_mesh_axes(sharding, sharded_batch_axes),
          get_flat_mesh_axes(sharding, row_axes),
          get_flat_mesh_axes(sharding, col_axes),
      ),
  )

  if is_canonical_3d:
    if explicit_axes and sharding == sharding_3d:
      return (lambda y: y), (lambda y_3d: y_3d), shape_3d, sharding_3d
    return (
        lambda y: reshard_or_constrain(y, sharding_3d),
        lambda y_3d: reshard_or_constrain(y_3d, sharding),
        shape_3d,
        sharding_3d,
    )

  def to_3d(y: jax.Array) -> jax.Array:
    y_3d = _reshape(jnp.transpose(y, perm), shape_3d, sharding_3d, explicit_axes=explicit_axes)
    return y_3d if explicit_axes else jax.lax.with_sharding_constraint(y_3d, sharding_3d)

  def from_3d(y_3d: jax.Array) -> jax.Array:
    y_unreshaped = _reshape(y_3d, permuted_shape, permuted_sharding, explicit_axes=explicit_axes).transpose(inv_perm)
    return reshard_or_constrain(y_unreshaped, sharding)

  return to_3d, from_3d, shape_3d, sharding_3d


def _get_3d_unshard_fns(
    shape_3d: tuple[int, int, int],
    sharding_3d: NamedSharding,
) -> tuple[ReshapeFn, ReshapeFn, NamedSharding]:
  """Unshards matrix dims of a 3D `[batch, row, col]` tensor via replication or all-to-all."""
  mesh = sharding_3d.mesh
  batch_mesh = get_dim_mesh_axes(sharding_3d, 0)
  matrix_mesh = get_flat_mesh_axes(sharding_3d, (1, 2))
  matrix_shards = math.prod(mesh.shape[s] for s in matrix_mesh)
  if matrix_shards <= 1:
    return lambda y_3d: y_3d, lambda y_flat: y_flat, sharding_3d

  batch_size, row_size, col_size = shape_3d
  batch_shards = math.prod(mesh.shape[s] for s in batch_mesh)
  local_batch_size = batch_size // batch_shards

  # Keep batch_mesh before matrix_mesh so each batch shard stays on its existing
  # devices and all-to-all only spans matrix_mesh; when local_batch_size == 1,
  # drop matrix_mesh to replicate across matrix shards instead.
  flat_batch_axes = batch_mesh if local_batch_size == 1 else batch_mesh + matrix_mesh
  flat_sharding = NamedSharding(mesh, jax.P(flat_batch_axes, None, None))

  local_padding = -local_batch_size % matrix_shards if local_batch_size > 1 else 0
  if local_padding == 0:
    return (
        lambda y_3d: reshard_or_constrain(y_3d, flat_sharding),
        lambda y_flat: reshard_or_constrain(y_flat, sharding_3d),
        flat_sharding,
    )

  # Pad axis 1 of (batch_shards, local_batch_size, ...) so each batch shard is
  # padded locally rather than shifting data across batch_mesh.
  explicit_axes = is_explicit_axes(sharding_3d)
  padded_local_batch = local_batch_size + local_padding
  padded_shape_3d = (batch_shards * padded_local_batch, row_size, col_size)
  local_4d_sharding = NamedSharding(
      mesh,
      jax.P(sharding_3d.spec[0], None, sharding_3d.spec[1], sharding_3d.spec[2]),
  )

  def to_flat(y_3d: jax.Array) -> jax.Array:
    y_4d = _reshape(
        y_3d, (batch_shards, local_batch_size, row_size, col_size), local_4d_sharding, explicit_axes=explicit_axes
    )
    y_4d = jnp.pad(y_4d, ((0, 0), (0, local_padding), (0, 0), (0, 0)))
    y_padded = _reshape(y_4d, padded_shape_3d, sharding_3d, explicit_axes=explicit_axes)
    return reshard_or_constrain(y_padded, flat_sharding)

  def from_flat(y_flat: jax.Array) -> jax.Array:
    y_padded = reshard_or_constrain(y_flat, sharding_3d)
    y_4d = _reshape(
        y_padded, (batch_shards, padded_local_batch, row_size, col_size), local_4d_sharding, explicit_axes=explicit_axes
    )
    return _reshape(y_4d[:, :local_batch_size], shape_3d, sharding_3d, explicit_axes=explicit_axes)

  return to_flat, from_flat, flat_sharding


def _get_3d_group_key(
    dtype: jax.typing.DTypeLike,
    shape_3d: tuple[int, int, int],
    sharding_3d: NamedSharding,
) -> tuple[Any, ...] | None:
  """Returns a grouping key for a canonical 3D tensor if unsharding needs batch padding.

  Returns `(dtype, row_size, col_size, mesh, sharding_3d.spec)` when matrix axes
  are sharded (`matrix_shards > 1`) and `local_batch_size` is not divisible by
  `matrix_shards`; returns `None` for unsharded or already-divisible tensors so
  they skip concatenation/splitting overhead.
  """
  mesh = sharding_3d.mesh
  batch_shards = math.prod(mesh.shape[s] for s in get_dim_mesh_axes(sharding_3d, 0))
  matrix_shards = math.prod(mesh.shape[s] for s in get_flat_mesh_axes(sharding_3d, (1, 2)))
  local_batch_size = shape_3d[0] // batch_shards
  if matrix_shards <= 1 or local_batch_size % matrix_shards == 0:
    return None
  return (dtype, shape_3d[1], shape_3d[2], mesh, sharding_3d.spec)


def _concat_3d_batch(
    xs_3d: Sequence[jax.Array],
    sharding_3d: NamedSharding,
) -> tuple[jax.Array, Callable[[jax.Array], list[jax.Array]]]:
  """Concatenates compatible 3D `[batch, row, col]` tensors along their local batch dim."""
  if not xs_3d:
    raise ValueError("xs_3d must be non-empty.")
  if len(xs_3d) == 1:
    return xs_3d[0], lambda y: [y]

  shapes_3d = [x.shape for x in xs_3d]
  _, row_size, col_size = shapes_3d[0]
  batch_mesh = get_dim_mesh_axes(sharding_3d, 0)
  explicit_axes = is_explicit_axes(sharding_3d)
  if not batch_mesh:
    offsets = list(itertools.accumulate(s[0] for s in shapes_3d[:-1]))
    x_concat_3d = jnp.concatenate(xs_3d, axis=0)
    if not explicit_axes:
      x_concat_3d = jax.lax.with_sharding_constraint(x_concat_3d, sharding_3d)

    def split_unsharded(y: jax.Array) -> list[jax.Array]:
      parts = list(jnp.split(y, offsets, axis=0))
      if not explicit_axes:
        return [jax.lax.with_sharding_constraint(p, sharding_3d) for p in parts]
      return parts

    return x_concat_3d, split_unsharded

  batch_shards = math.prod(sharding_3d.mesh.shape[s] for s in batch_mesh)
  local_4d_sharding = NamedSharding(
      sharding_3d.mesh,
      jax.P(batch_mesh, None, get_dim_mesh_axes(sharding_3d, 1), get_dim_mesh_axes(sharding_3d, 2)),
  )
  local_batches = [s[0] // batch_shards for s in shapes_3d]
  total_local_batch = sum(local_batches)

  xs_4d = [
      _reshape(x, (batch_shards, local_b, row_size, col_size), local_4d_sharding, explicit_axes=explicit_axes)
      for x, local_b in zip(xs_3d, local_batches)
  ]
  x_concat_3d = _reshape(
      jnp.concatenate(xs_4d, axis=1),
      (batch_shards * total_local_batch, row_size, col_size),
      sharding_3d,
      explicit_axes=explicit_axes,
  )
  if not explicit_axes:
    x_concat_3d = jax.lax.with_sharding_constraint(x_concat_3d, sharding_3d)

  def split_3d(y_3d: jax.Array) -> list[jax.Array]:
    y_4d = _reshape(
        y_3d, (batch_shards, total_local_batch, row_size, col_size), local_4d_sharding, explicit_axes=explicit_axes
    )
    outputs = []
    offset = 0
    for shape_3d, local_b in zip(shapes_3d, local_batches):
      y_leaf_3d = _reshape(y_4d[:, offset : offset + local_b], shape_3d, sharding_3d, explicit_axes=explicit_axes)
      if not explicit_axes:
        y_leaf_3d = jax.lax.with_sharding_constraint(y_leaf_3d, sharding_3d)
      outputs.append(y_leaf_3d)
      offset += local_b
    return outputs

  return x_concat_3d, split_3d


class _PaddedLeaf(NamedTuple):
  """Internal record for a padded tensor eligible for post-3D batch grouping."""

  index: int
  item: TensorWithAxes
  to_3d: ReshapeFn
  from_3d: ReshapeFn
  sharding_3d: NamedSharding


def _pack_padded_group(
    group: Sequence[_PaddedLeaf],
) -> tuple[TensorWithAxes, tuple[list[int], Callable[[jax.Array], list[jax.Array]]]]:
  """Packs a multi-leaf `_PaddedLeaf` group into a single canonical 3D tensor."""
  sharding_3d = group[0].sharding_3d
  xs_3d = [entry.to_3d(entry.item[0]) for entry in group]
  x_concat_3d, split_3d = _concat_3d_batch(xs_3d, sharding_3d)
  idxs = [entry.index for entry in group]
  from_3d_fns = [entry.from_3d for entry in group]

  def unpack_group(y_concat: jax.Array) -> list[jax.Array]:
    return [from_3d(y_3d) for from_3d, y_3d in zip(from_3d_fns, split_3d(y_concat))]

  # Canonical 3D has row <= col; output_axes=(1,), reduction_axes=(2,) hits
  # `_get_3d_flatten_fns`'s `is_canonical_3d` fast path even when row == col.
  return (x_concat_3d, (2,), (1,), sharding_3d), (idxs, unpack_group)


def pack_compatible_tensors(
    items: Sequence[TensorWithAxes],
) -> tuple[list[TensorWithAxes], Callable[[Sequence[jax.Array]], list[jax.Array]]]:
  """Packs compatible padded tensors into batched canonical 3D tensors.

  Tensors whose canonical 3D `[batch, row, col]` form has sharded matrix axes
  and `local_batch_size % matrix_shards != 0` are grouped by
  `(dtype, row, col, mesh, sharding_3d.spec)` and concatenated locally along
  their batch axis. Unsharded, already-divisible, and singleton tensors are
  passed through unchanged.

  Args:
    items: Sequence of `(array, reduction_axes, output_axes, sharding)` tuples.

  Returns:
    `(packed_items, unpack_fn)` where `packed_items` uses the same tuple format
    (ready for `get_reshape_fns`) and `unpack_fn` maps outputs corresponding to
    `packed_items` back to a list aligned with `items`.
  """
  packed_items: list[TensorWithAxes] = []
  unpack_specs: list[tuple[list[int], Callable[[jax.Array], list[jax.Array]]]] = []
  padded_groups: dict[tuple[Any, ...], list[_PaddedLeaf]] = {}

  def identity_unpack(y: jax.Array) -> list[jax.Array]:
    return [y]

  for idx, item in enumerate(items):
    x, reduction_axes, output_axes, sharding = item
    if sharding is not None:
      to_3d, from_3d, shape_3d, sharding_3d = _get_3d_flatten_fns(x.shape, reduction_axes, output_axes, sharding)
      if sharding_3d is not None:
        group_key = _get_3d_group_key(x.dtype, shape_3d, sharding_3d)
        if group_key is not None:
          padded_groups.setdefault(group_key, []).append(
              _PaddedLeaf(index=idx, item=item, to_3d=to_3d, from_3d=from_3d, sharding_3d=sharding_3d)
          )
          continue

    packed_items.append(item)
    unpack_specs.append(([idx], identity_unpack))

  for group in padded_groups.values():
    if len(group) == 1:
      entry = group[0]
      packed_items.append(entry.item)
      unpack_specs.append(([entry.index], identity_unpack))
    else:
      packed_item, unpack_spec = _pack_padded_group(group)
      packed_items.append(packed_item)
      unpack_specs.append(unpack_spec)

  def unpack_fn(outputs: Sequence[jax.Array]) -> list[jax.Array]:
    if len(outputs) != len(unpack_specs):
      raise ValueError(f"Expected {len(unpack_specs)} outputs, got {len(outputs)}.")
    by_index: dict[int, jax.Array] = {}
    for y, (idxs, group_unpack) in zip(outputs, unpack_specs):
      for idx, leaf_out in zip(idxs, group_unpack(y)):
        by_index[idx] = leaf_out
    return [by_index[i] for i in range(len(items))]

  return packed_items, unpack_fn


def get_reshape_fns(
    x: jax.Array,
    reduction_axes: tuple[int, ...],
    output_axes: tuple[int, ...],
    sharding: NamedSharding | None = None,
    use_all_to_all: bool = True,
) -> tuple[ReshapeFn, ReshapeFn, NamedSharding | None]:
  """Computes reshape functions and flat 3D `[batch, row, col]` sharding."""
  to_3d, from_3d, shape_3d, sharding_3d = _get_3d_flatten_fns(x.shape, reduction_axes, output_axes, sharding)
  if sharding_3d is None or not use_all_to_all:
    return to_3d, from_3d, sharding_3d
  to_flat, from_flat, flat_sharding = _get_3d_unshard_fns(shape_3d, sharding_3d)
  return (
      lambda y: to_flat(to_3d(y)),
      lambda y_flat: from_3d(from_flat(y_flat)),
      flat_sharding,
  )


def is_explicit_axes(
    sharding: jax.sharding.NamedSharding | None,
) -> bool:
  """Returns True if the mesh uses Explicit axis types, False otherwise.

  Args:
    sharding: The sharding to check.

  Returns:
    True if the mesh uses Explicit axis types, False if Auto/Unconstrained or
    sharding is None.

  Raises:
    ValueError: If the mesh contains mixed axis types (some Explicit, some
      non-Explicit).
  """
  if sharding is None:
    return False

  axis_types = tuple(sharding.mesh.axis_types)
  if not axis_types:
    return False

  explicit_count = sum(1 for t in axis_types if t == jax.sharding.AxisType.Explicit)

  if 0 < explicit_count < len(axis_types):
    raise ValueError(
        "Mixed mesh axis types (both Explicit and Auto/Unconstrained) are not" f" supported. Found {axis_types=}."
    )

  return explicit_count == len(axis_types)


def reshard_or_constrain(
    x: jax.Array,
    target_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
  """Reshards x using jax.reshard for explicit mesh axes, or jax.lax.with_sharding_constraint for auto mesh axes."""
  if is_explicit_axes(target_sharding):
    return jax.reshard(x, target_sharding)
  return jax.lax.with_sharding_constraint(x, target_sharding)
