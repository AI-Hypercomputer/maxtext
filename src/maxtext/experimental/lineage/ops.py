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

"""Common operations shared across models."""

from collections.abc import Callable, Collection, Mapping
import dataclasses
import functools
import math
import string
from typing import Any, NamedTuple, Sequence, Union
import jax
import jax.experimental.compute_on
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

AnyJaxArray = jt.Num[jax.Array, "..."]
AnyJaxArrayOrPyTree = AnyJaxArray | jt.PyTree[AnyJaxArray]


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def rms_norm(
    x: jt.Num[jax.Array, "*input_dims"],
    scale: jt.Num[jax.Array, "..."] | None = None,
    *,
    epsilon: float,
    axis: Union[int, Sequence[int]] = -1,
    norm_dtype: jt.DTypeLike = jnp.float32,
) -> jt.Num[jax.Array, "*input_dims"]:
  """Performs RMS normalization.

  Args:
    x: Input array.
    scale: Scale array, which scales the last `scale.ndim` dimensions of x. The
      shape of scale must be a suffix of the shape of x.
    epsilon: Epsilon value for RMS normalization.
    axis: Optional, int or sequence of ints, default=-1. Axis or set of axes to
      use for normalization.
    norm_dtype: Dtype used for normalization. Defaults to float32.

  Returns:
    The normalized array.
  """
  assert scale is None or scale.shape == x.shape[-scale.ndim :], (
      f"scale.shape must be a suffix of x.shape, but got {scale.shape} and" f" {x.shape}"
  )

  x_dtype = x.dtype
  x = jnp.astype(x, norm_dtype)
  mean2 = jnp.mean(jnp.square(x), axis=axis, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), x_dtype)
  if scale is None:
    return y
  scale_dims = string.ascii_lowercase[: scale.ndim]
  return jnp.einsum(f"...{scale_dims},{scale_dims}->...{scale_dims}", y, scale)


def physical_pspec(
    logical_pspec: jax.sharding.PartitionSpec,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jax.sharding.PartitionSpec:
  """Translates logical axes in a PartitionSpec to physical mesh axes via axis_mapping."""

  def _map_axis(
      axis: str | tuple[str, ...] | None,
  ) -> str | tuple[str, ...] | None:
    if axis is None:
      return None
    if isinstance(axis, str):
      return axis_mapping.get(axis, axis)
    out = []
    for sub_axis in axis:
      mapped = axis_mapping.get(sub_axis, sub_axis)
      if isinstance(mapped, tuple):
        out.extend(mapped)
      else:
        out.append(mapped)
    return tuple(out)

  def _map_set(s: frozenset[str] | None) -> frozenset[str]:
    if not s:
      return frozenset()
    out = set()
    for p in s:
      mapped = _map_axis(p)
      if isinstance(mapped, tuple):
        out.update(mapped)
      elif mapped is not None:
        out.add(mapped)
    return frozenset(out)

  physical_partitions = tuple(_map_axis(p) for p in logical_pspec.partitions)
  reduced = _map_set(logical_pspec.reduced)
  unreduced = _map_set(logical_pspec.unreduced)
  return jax.sharding.PartitionSpec(*physical_partitions, reduced=reduced, unreduced=unreduced)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_along_axis(
    xs: jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]],
    axis_name: str | Sequence[str] | jt.PyTree[Any],
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    compute_type: str | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]]:
  """All-gathers each array along the given axis names if it is sharded along them.

  The dual of this function will perform a reduce-scatter along the given axes
  for each array that is sharded along them. When an array is not sharded
  along a given axis, the function is a no-op in the forward pass. However, the
  dual of this function will perform an all-reduce along that axis. This makes
  it useful for specifying when the all-reduces should occur for specific axes
  in the backward pass, such as for pipelining all-reduces across DCN.

  Args:
    xs: Input array or pytree of arrays, which may or may not be sharded along
      the given axis names.
    axis_name: Name of the axis, sequence of axis names, or pytree prefix of
      axis names to collect along. For weights, these should be the axes used
      for data parallelism for the activations that go through them.
    axis_mapping: Mapping from logical to physical mesh axes.
    compute_type: Optional `compute_on` target (e.g., `"tpu_sparsecore"`) to
      apply inside `shard_map`.
    compiler_options: Optional compiler options for `compute_on`.

  Returns:
    The array all-gathered along the given axis names if sharded along them.
  """

  def _to_tuple(m: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(m, str):
      return (m,)
    return tuple(m)

  def _is_axis_spec(x: Any) -> bool:
    return isinstance(x, str) or (
        isinstance(x, (tuple, list)) and not hasattr(x, "_fields") and all(isinstance(e, str) for e in x)
    )

  def _get_physical_axes(
      leaf_axis_name: str | Sequence[str],
  ) -> tuple[tuple[str, ...], frozenset[str]]:
    names: list[str] = []
    for name in _to_tuple(leaf_axis_name):
      for ax in _to_tuple(axis_mapping.get(name, name)):
        if ax not in names:
          names.append(ax)
    return tuple(names), frozenset(names)

  leaves = jax.tree.leaves(xs)
  if not leaves:
    return xs

  physical_axes_tree = jax.tree.map(
      lambda spec, sub_xs: jax.tree.map(lambda _: _get_physical_axes(spec), sub_xs),
      axis_name,
      xs,
      is_leaf=_is_axis_spec,
  )
  flat_physical_axes = jax.tree.leaves(
      physical_axes_tree,
      is_leaf=lambda x: (isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], frozenset)),
  )
  if not any(names for names, _ in flat_physical_axes):
    return xs

  mesh = jax.typeof(leaves[0]).sharding.mesh

  def _remove_axis_names(
      partitions: tuple[str | tuple[str, ...] | None, ...],
      physical_axis_set: frozenset[str],
  ) -> tuple[str | tuple[str, ...] | None, ...]:
    """Removes axis names from the partitions of a PartitionSpec to ensure replication."""
    out = []
    for a in partitions:
      if a is None:
        out.append(None)
      elif isinstance(a, str):
        out.append(a if a not in physical_axis_set else None)
      else:
        out.append(tuple(b for b in a if b not in physical_axis_set))
    return tuple(out)

  def _get_in_spec(arr):
    return jax.typeof(arr).sharding.spec

  def _get_target_spec(
      in_spec: jax.sharding.PartitionSpec,
      physical_axes: tuple[tuple[str, ...], frozenset[str]],
  ):
    _, physical_axis_set = physical_axes
    return jax.sharding.PartitionSpec(
        *_remove_axis_names(in_spec.partitions, physical_axis_set),
        # Concatenates the existing reduced axes with the new axes.
        reduced=in_spec.reduced | physical_axis_set,
        unreduced=in_spec.unreduced,
    )

  in_specs = jax.tree.map(_get_in_spec, xs)
  out_specs = jax.tree.map(_get_target_spec, in_specs, physical_axes_tree)

  def _collect_leaf(
      arr,
      in_spec: jax.sharding.PartitionSpec,
      physical_axes: tuple[tuple[str, ...], frozenset[str]],
  ):
    physical_axis_names, physical_axis_set = physical_axes
    sharded_axes = set()
    for dim_idx, part in enumerate(in_spec.partitions):
      if part is None:
        continue
      part_axes = (part,) if isinstance(part, str) else part
      sharded_axes.update(part_axes)
      for ax in reversed(part_axes):
        if ax in physical_axis_set:
          arr = jax.lax.all_gather(arr, ax, axis=dim_idx, tiled=True, to="reduced")
    for ax in physical_axis_names:
      if ax not in sharded_axes and ax not in in_spec.reduced and ax not in in_spec.unreduced:
        arr = jax.lax.pcast(arr, ax, to="reduced")
    return arr

  def _collect_pytree(xs_local):
    return jax.tree.map(_collect_leaf, xs_local, in_specs, physical_axes_tree)

  if compute_type is not None:
    _collect_pytree = jax.experimental.compute_on.compute_on(
        compute_type=compute_type,
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options=compiler_options,
    )(_collect_pytree)

  return jax.shard_map(
      _collect_pytree,
      mesh=mesh,
      in_specs=(in_specs,),
      out_specs=out_specs,
  )(xs)


@dataclasses.dataclass(frozen=True)
class DeferredReduce:
  """A reduction that the caller runs inside its own `shard_map`.

  `finish(jax.shard_map(local_fn, mesh=mesh, out_specs=out_specs)(operand))` is
  the output of the `reduce_along_axis` call that returned this. Running the
  reduction in the caller's `shard_map` lets it share, e.g., a SparseCore block
  with other collectives.

  Attributes:
    operand: The arrays to reduce. The arrays needing no reduction are left out,
      so they add no dependency to the caller's `shard_map`.
    local_fn: Per-shard reduction of `operand`.
    out_specs: PartitionSpecs of the outputs of `local_fn`.
    finish: Merges the outputs of the `shard_map` with the arrays left out of
      `operand` into the output of `reduce_along_axis`.
  """

  operand: list[jax.Array]
  local_fn: Callable[[list[jax.Array]], list[jax.Array]]
  out_specs: list[jax.sharding.PartitionSpec]
  finish: Callable[[list[jax.Array]], Any]


@jt.jaxtyped(typechecker=typeguard.typechecked)
def reduce_along_axis(
    xs: jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]],
    axis_name: str | Sequence[str] | jt.PyTree[Any],
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    like: jt.Shaped[jax.Array, "..."] | jt.PyTree[jt.Shaped[jax.Array, "..."]],
    *,
    axes: Collection[str] | None = None,
    defer: bool = False,
    compute_type: str | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]] | DeferredReduce:
  """Reduces cotangents of `collect_along_axis` back to its input shardings.

  `reduce_along_axis(ct, axis_name, axis_mapping, like=w)` is the VJP of
  `collect_along_axis(w, axis_name, axis_mapping)` applied to `ct`, but only
  reads w's shardings, so `like` may differ from w in dtype (e.g. the fp8 copy
  of bf16 weights). Each array is reduce-scattered along the given axes it is
  sharded along in `like` and all-reduced along the ones it is replicated
  along.

  The reduction can be split by mesh axis into calls issued separately, e.g.
  on different SparseCores or in different scan steps: `axes` restricts a call
  to the collected axes in it and leaves the arrays unreduced along the other
  collected axes for a later call. A call only reduces along the collected
  axes its input is still unreduced along, so the calls compose in any order.

  Args:
    xs: Cotangents of the `collect_along_axis` outputs, unreduced along the
      collected axes that no earlier call reduced.
    axis_name: Axis names passed to `collect_along_axis`.
    axis_mapping: Mapping from logical to physical mesh axes.
    like: Array or pytree of arrays with the shardings of the
      `collect_along_axis` inputs.
    axes: Optional physical mesh axes to reduce along in this call; `None` for
      all the collected axes. The collected axes sharding a dimension of `like`
      together must be all in or all out of `axes`, since reduce-scattering
      along only some of them would scramble the shards.
    defer: Whether to return the reduction as a `DeferredReduce` to run inside
      the caller's `shard_map`, instead of issuing it. Requires no
      `compute_type`.
    compute_type: Optional `compute_on` target (e.g., `"tpu_sparsecore"`) to
      apply inside `shard_map`.
    compiler_options: Optional compiler options for `compute_on`.

  Returns:
    The reduced cotangents, with the cotangent shardings of `like`, except
    unreduced along the collected axes left out of `axes`. A `DeferredReduce`
    of them if `defer`.
  """
  if defer and compute_type is not None:
    raise ValueError("defer runs the reduction in the caller's shard_map, which rules out" " compute_type.")
  if axes is not None:
    axes = frozenset(axes)

  def _to_tuple(m: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(m, str):
      return (m,)
    return tuple(m)

  def _is_axis_spec(x: Any) -> bool:
    return isinstance(x, str) or (
        isinstance(x, (tuple, list)) and not hasattr(x, "_fields") and all(isinstance(e, str) for e in x)
    )

  def _get_physical_axes(
      leaf_axis_name: str | Sequence[str],
  ) -> tuple[tuple[str, ...], frozenset[str]]:
    names: list[str] = []
    for name in _to_tuple(leaf_axis_name):
      for ax in _to_tuple(axis_mapping.get(name, name)):
        if ax not in names:
          names.append(ax)
    return tuple(names), frozenset(names)

  def _nothing_to_reduce() -> Any:
    if not defer:
      return xs
    return DeferredReduce(
        operand=[],
        local_fn=lambda operand: operand,
        out_specs=[],
        finish=lambda reduced: xs,
    )

  leaves = jax.tree.leaves(xs)
  if not leaves:
    return _nothing_to_reduce()

  physical_axes_tree = jax.tree.map(
      lambda spec, sub_xs: jax.tree.map(lambda _: _get_physical_axes(spec), sub_xs),
      axis_name,
      xs,
      is_leaf=_is_axis_spec,
  )
  flat_physical_axes = jax.tree.leaves(
      physical_axes_tree,
      is_leaf=lambda x: (isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], frozenset)),
  )
  if not any(names for names, _ in flat_physical_axes):
    return _nothing_to_reduce()

  mesh = jax.typeof(leaves[0]).sharding.mesh
  in_specs = jax.tree.map(lambda arr: jax.typeof(arr).sharding.spec, xs)
  like_specs = jax.tree.map(lambda arr: jax.typeof(arr).sharding.spec, like)

  def _leaf_collectives(
      in_spec: jax.sharding.PartitionSpec,
      like_spec: jax.sharding.PartitionSpec,
      physical_axes: tuple[tuple[str, ...], frozenset[str]],
  ) -> tuple[list[tuple[int, str]], list[str], frozenset[str]]:
    """Plans a leaf's collectives in this call.

    Args:
      in_spec: Sharding spec of the leaf of `xs`.
      like_spec: Sharding spec of the leaf of `like`.
      physical_axes: Physical axes the leaf is collected along, in order and as
        a set.

    Returns:
      The `(dimension, axis)` pairs to reduce-scatter along, the axes to
      all-reduce along, and the collected axes left unreduced for a later call.
      Only the collected axes the leaf is still unreduced along are pending: an
      earlier call may have reduced the others.
    """
    physical_axis_names, physical_axis_set = physical_axes
    pending = physical_axis_set & in_spec.unreduced
    selected = pending if axes is None else pending & axes
    sharded_axes = set()
    scatters = []
    # Transposes `collect_along_axis`'s `_collect_leaf` in reverse order.
    for dim_idx, part in reversed(list(enumerate(like_spec.partitions))):
      if part is None:
        continue
      part_axes = (part,) if isinstance(part, str) else part
      sharded_axes.update(part_axes)
      dim_pending = [ax for ax in part_axes if ax in pending]
      dim_selected = [ax for ax in dim_pending if ax in selected]
      if dim_selected and len(dim_selected) < len(dim_pending):
        raise ValueError(
            f"Reduce-scattering along only {dim_selected} of {dim_pending},"
            f" which shard dimension {dim_idx} of {like_spec} together, would"
            " scramble the shards."
        )
      scatters.extend((dim_idx, ax) for ax in dim_selected)
    psums = [
        ax
        for ax in reversed(physical_axis_names)
        if ax in selected and ax not in sharded_axes and ax not in like_spec.reduced and ax not in like_spec.unreduced
    ]
    return scatters, psums, pending - selected

  def _out_spec(
      in_spec: jax.sharding.PartitionSpec,
      like_spec: jax.sharding.PartitionSpec,
      physical_axes: tuple[tuple[str, ...], frozenset[str]],
  ) -> jax.sharding.PartitionSpec:
    spec = like_spec.to_ct_spec()
    _, _, skipped = _leaf_collectives(in_spec, like_spec, physical_axes)
    if not skipped:
      return spec

    def _unscattered(part):
      if part is None or isinstance(part, str):
        return None if part in skipped else part
      return tuple(ax for ax in part if ax not in skipped) or None

    # Stays unscattered along the skipped axes `like` is sharded along, and
    # unreduced along all of them.
    return jax.sharding.PartitionSpec(
        *map(_unscattered, spec.partitions),
        reduced=spec.reduced,
        unreduced=spec.unreduced | skipped,
    )

  out_specs = jax.tree.map(_out_spec, in_specs, like_specs, physical_axes_tree)

  def _reduce_leaf(
      arr,
      in_spec: jax.sharding.PartitionSpec,
      like_spec: jax.sharding.PartitionSpec,
      physical_axes: tuple[tuple[str, ...], frozenset[str]],
  ):
    scatters, psums, _ = _leaf_collectives(in_spec, like_spec, physical_axes)
    # The reduce-scatters first: they shrink the all-reduces, which commute
    # with them as they are along other axes.
    for dim_idx, ax in scatters:
      arr = jax.lax.psum_scatter(arr, ax, scatter_dimension=dim_idx, tiled=True)
    for ax in psums:
      arr = jax.lax.psum(arr, ax)
    return arr

  if defer:
    xs_leaves, xs_treedef = jax.tree.flatten(xs)
    leaf_specs = list(
        zip(
            xs_treedef.flatten_up_to(in_specs),
            xs_treedef.flatten_up_to(like_specs),
            xs_treedef.flatten_up_to(physical_axes_tree),
        )
    )
    leaf_out_specs = xs_treedef.flatten_up_to(out_specs)
    # The arrays this call reduces along no axis need no collective.
    reduced_idxs = []
    for i, specs in enumerate(leaf_specs):
      scatters, psums, _ = _leaf_collectives(*specs)
      if scatters or psums:
        reduced_idxs.append(i)

    def _local_fn(operand):
      return [_reduce_leaf(arr, *leaf_specs[i]) for arr, i in zip(operand, reduced_idxs)]

    def _finish(reduced):
      merged = list(xs_leaves)
      for i, arr in zip(reduced_idxs, reduced):
        merged[i] = arr
      return xs_treedef.unflatten(merged)

    return DeferredReduce(
        operand=[xs_leaves[i] for i in reduced_idxs],
        local_fn=_local_fn,
        out_specs=[leaf_out_specs[i] for i in reduced_idxs],
        finish=_finish,
    )

  def _reduce_pytree(xs_local):
    return jax.tree.map(_reduce_leaf, xs_local, in_specs, like_specs, physical_axes_tree)

  if compute_type is not None:
    _reduce_pytree = jax.experimental.compute_on.compute_on(
        compute_type=compute_type,
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options=compiler_options,
    )(_reduce_pytree)

  return jax.shard_map(
      _reduce_pytree,
      mesh=mesh,
      in_specs=(in_specs,),
      out_specs=out_specs,
  )(xs)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def w_collect_pipelined_scan(
    scan_fn: Callable[
        [AnyJaxArray, AnyJaxArrayOrPyTree],
        tuple[AnyJaxArray, AnyJaxArrayOrPyTree],
    ],
    x: AnyJaxArray,
    w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
) -> tuple[AnyJaxArray, AnyJaxArrayOrPyTree]:
  """Pipelines weight collection for scanning over scan_fn.

  Args:
    scan_fn: Function to execute at each scan iteration with signature `(x:
      AnyJaxArray, w: AnyJaxArrayOrPyTree) -> tuple[AnyJaxArray,
      AnyJaxArrayOrPyTree]` which takes current activations and collected
      weights and returns `(output, aux_step)`.
    x: Input array.
    w: Uncollected weights to scan over. Dimension 0 is the dimension to scan
      over and must be present.
    collect_fn_dcn: Function to collect weights over the DCN axis.
    collect_fn_ici: Function to collect weights over the ICI axes.

  Returns:
    A tuple of (output, aux) where aux is the stacked auxiliary outputs across
    all scan iterations.
  """
  w_first = collect_fn_ici(collect_fn_dcn(jax.tree.map(lambda _w: _w[0], w)))
  carry = (x, w_first)

  def body(carry, w_n_sharded):
    x, w = carry
    w_n = collect_fn_ici(collect_fn_dcn(w_n_sharded))
    x, aux_step = scan_fn(x, w)
    return (x, w_n), aux_step

  (x, w_last), aux_scanned = jax.lax.scan(body, carry, jax.tree.map(lambda _w: _w[1:], w))
  x, aux_last = scan_fn(x, w_last)
  aux = jax.tree.map(
      lambda s, l: jnp.concatenate([s, l[jnp.newaxis]], axis=0),
      aux_scanned,
      aux_last,
  )
  return x, aux


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_fwd(
    scan_fn: Callable[
        [AnyJaxArray, AnyJaxArrayOrPyTree],
        tuple[AnyJaxArray, AnyJaxArrayOrPyTree],
    ],
    x: AnyJaxArray,
    w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
) -> tuple[tuple[AnyJaxArray, AnyJaxArrayOrPyTree], Any]:
  """Forward pass of w_collect_pipelined_scan which does not save collected weights."""
  assert jax.tree.leaves(w)[0].shape[0] >= 2, (
      "there must be at least two scan iterations to perform a pipelined scan" " and its transpose"
  )
  # Derive the reduce functions (the duals of the collect functions)
  w_first_dcn, reduce_fn_dcn = jax.vjp(collect_fn_dcn, jax.tree.map(lambda _w: _w[0], w))
  getattr(reduce_fn_dcn, "args_res")[0] = None
  w_first, reduce_fn_ici = jax.vjp(collect_fn_ici, w_first_dcn)
  getattr(reduce_fn_ici, "args_res")[0] = None

  (x, aux_first), scan_fn_vjp_first = jax.vjp(scan_fn, x, w_first)
  getattr(scan_fn_vjp_first, "args_res")[1] = None

  w_second = collect_fn_ici(collect_fn_dcn(jax.tree.map(lambda _w: _w[1], w)))
  carry = (x, w_second)

  def body_fwd(carry, w_n_sharded):
    x, w = carry
    w_n = collect_fn_ici(collect_fn_dcn(w_n_sharded))
    (x, aux_step), scan_fn_vjp = jax.vjp(scan_fn, x, w)
    getattr(scan_fn_vjp, "args_res")[1] = None
    return (x, w_n), (scan_fn_vjp, aux_step)

  (x, w_last), (scan_fn_vjps, aux_scanned) = jax.lax.scan(body_fwd, carry, jax.tree.map(lambda _w: _w[2:], w))
  (x, aux_last), scan_fn_vjp_last = jax.vjp(scan_fn, x, w_last)
  getattr(scan_fn_vjp_last, "args_res")[1] = None
  aux = jax.tree.map(
      lambda first, scanned, last: jnp.concatenate([first[jnp.newaxis], scanned, last[jnp.newaxis]], axis=0),
      aux_first,
      aux_scanned,
      aux_last,
  )
  return (x, aux), (
      scan_fn_vjp_first,
      scan_fn_vjps,
      scan_fn_vjp_last,
      w,
      reduce_fn_dcn,
      reduce_fn_ici,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_bwd(
    _,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    res: Any,
    grad_outputs: tuple[AnyJaxArray, Any],
) -> tuple[AnyJaxArray, AnyJaxArrayOrPyTree]:
  """Backward pass of w_collect_pipelined_scan which re-collects weights."""
  (
      scan_fn_vjp_first,
      scan_fn_vjps,
      scan_fn_vjp_last,
      w,
      reduce_fn_dcn,
      reduce_fn_ici,
  ) = res

  x_grad, aux_grad = grad_outputs
  aux_step_last = jax.tree.map(lambda g: g[-1], aux_grad)
  step_grad_last = (x_grad, aux_step_last)

  w_last = collect_fn_ici(collect_fn_dcn(jax.tree.map(lambda _w: _w[-1], w)))
  getattr(scan_fn_vjp_last, "args_res")[1] = w_last
  x_grad, w_last_grad_unreduced = scan_fn_vjp_last(step_grad_last)

  w_second_last = collect_fn_ici(collect_fn_dcn(jax.tree.map(lambda _w: _w[-2], w)))
  carry = (x_grad, w_second_last, w_last_grad_unreduced)

  aux_scanned_grad = jax.tree.map(lambda g: g[1:-1], aux_grad)
  scan_xs = (
      scan_fn_vjps,
      jax.tree.map(lambda _w: _w[:-2], w),
      aux_scanned_grad,
  )

  def body_bwd(carry, xs):
    x_grad, w_n, w_prev_grad_unreduced = carry
    scan_fn_vjp, w_next_sharded, aux_step_grad = xs

    w_next = collect_fn_ici(collect_fn_dcn(w_next_sharded))
    w_prev_grad_reduced = reduce_fn_dcn(reduce_fn_ici(w_prev_grad_unreduced)[0])[0]

    getattr(scan_fn_vjp, "args_res")[1] = w_n
    x_grad, w_n_grad_unreduced = scan_fn_vjp((x_grad, aux_step_grad))
    return (x_grad, w_next, w_n_grad_unreduced), w_prev_grad_reduced

  (x_grad, w_first, w_second_grad_unreduced), w_grads_reduced = jax.lax.scan(
      body_bwd,
      carry,
      scan_xs,
      reverse=True,
  )

  w_second_grad_reduced = reduce_fn_dcn(reduce_fn_ici(w_second_grad_unreduced)[0])[0]
  getattr(scan_fn_vjp_first, "args_res")[1] = w_first
  aux_step_first = jax.tree.map(lambda g: g[0], aux_grad)
  step_grad_first = (x_grad, aux_step_first)
  x_grad, w_first_grad_unreduced = scan_fn_vjp_first(step_grad_first)
  w_first_grad_reduced = reduce_fn_dcn(reduce_fn_ici(w_first_grad_unreduced)[0])[0]
  # Likely better to initialize w_grad as a tree of zeros and insert into it
  # rather than concatenating at the end.
  w_grad = jax.tree.map(
      lambda x, y, z: jnp.concatenate([x[jnp.newaxis], y[jnp.newaxis], z], axis=0),
      w_first_grad_reduced,
      w_second_grad_reduced,
      w_grads_reduced,
  )
  return x_grad, w_grad


w_collect_pipelined_scan.defvjp(w_collect_pipelined_scan_fwd, w_collect_pipelined_scan_bwd)


class PipelinedScanLayer(NamedTuple):
  fwd: Callable[..., tuple[Any, Any]]
  bwd: Callable[..., Any]


def combine_w_grads(w_mla: AnyJaxArrayOrPyTree, w_expert: AnyJaxArrayOrPyTree) -> AnyJaxArrayOrPyTree:
  """Combines MLA/norm/router gradients and expert gradients for a layer."""

  def _add(a, b):
    if a is None:
      return b
    if b is None:
      return a
    return a + b

  return jax.tree.map(_add, w_mla, w_expert, is_leaf=lambda x: x is None)


def _interleave_arrays(a0: Any, a1: Any) -> Any:
  """Interleaves two arrays along axis 1 and merges leading dimensions."""
  if a0 is None or a1 is None:
    return None
  stacked = jnp.stack([a0, a1], axis=1)
  new_shape = (a0.shape[0] * 2,) + a0.shape[1:]
  s = getattr(jax.typeof(a0), "sharding", None)
  if s is not None:
    return jax.lax.reshape(stacked, new_shape, out_sharding=s)
  return stacked.reshape(new_shape)


def _layer_slice(tree: Any, idx: int | slice) -> Any:
  """Slices the layer dimension of every leaf; `None` trees stay `None`."""
  return jax.tree.map(lambda t: t[idx], tree)


def _layer_at(tree: Any, idx: int | jax.Array | None) -> Any:
  """Layer `idx` (static or traced) of every leaf; `None` for a `None` index."""
  if tree is None or idx is None:
    return None
  if isinstance(idx, int):
    return _layer_slice(tree, idx)
  return jax.tree.map(lambda t: jax.lax.dynamic_index_in_dim(t, idx, keepdims=False), tree)


def _w_aux_kwargs(w_aux_layer: Any) -> dict[str, Any]:
  """Keyword arguments forwarding a layer's `w_aux` slice, if any."""
  return {} if w_aux_layer is None else {"w_aux": w_aux_layer}


class _LayerCollect(NamedTuple):
  """Per-layer weight collection helpers of the pipelined scans.

  A layer's weights are collected as `collect_ici(pre(collect_dcn(w_layer)))`,
  where `pre` is the optional `collect_fn_pre` (identity if absent).

  Attributes:
    collect: `(w_layer, w_aux_layer) -> collected`, the full collection.
    pre: `(w_dcn, w_aux_layer) -> w_pre`, the pre-collect transform of
      DCN-collected weights.
    collect_ici: `(w_pre, w_aux_layer) -> collected`, the ICI collection of
      pre-collected weights.
    pre_carry: `idx -> (pre(collect_dcn(w[idx])), w_aux[idx])`, the
      pre-collection of layer `idx` (static or traced) to carry into the step
      that ICI-collects it. `None` without `collect_fn_pre` or for a `None`
      index.
  """

  collect: Callable[[Any, Any], Any]
  pre: Callable[[Any, Any], Any]
  collect_ici: Callable[[Any, Any], Any]
  pre_carry: Callable[[int | jax.Array | None], Any]


def _layer_collect(
    collect_fn_dcn: Callable[..., Any],
    collect_fn_ici: Callable[..., Any],
    collect_fn_pre: Callable[..., Any] | None,
    w: Any,
    w_aux: Any,
) -> _LayerCollect:
  """Builds the `_LayerCollect` helpers for a pipelined scan pass."""

  def pre(w_dcn, w_aux_layer):
    if collect_fn_pre is None:
      return w_dcn
    return collect_fn_pre(w_dcn, **_w_aux_kwargs(w_aux_layer))

  def collect_ici(w_pre, w_aux_layer):
    return collect_fn_ici(w_pre, **_w_aux_kwargs(w_aux_layer))

  def collect(w_layer, w_aux_layer):
    return collect_ici(pre(collect_fn_dcn(w_layer), w_aux_layer), w_aux_layer)

  def pre_carry(idx):
    if collect_fn_pre is None or idx is None:
      return None
    w_aux_layer = _layer_at(w_aux, idx)
    return pre(collect_fn_dcn(_layer_at(w, idx)), w_aux_layer), w_aux_layer

  return _LayerCollect(collect=collect, pre=pre, collect_ici=collect_ici, pre_carry=pre_carry)


def _shifted_indices(layers: slice | int, shift: int, num_layers: int) -> Any:
  """Layer indices `layers + shift`, clamped to `[0, num_layers)`.

  For a slice, returns a traced index array with out-of-range entries clamped
  (their pre-collection is computed but unused). For an int, returns `None`
  when out of range.

  Args:
    layers: Static layer slice or index.
    shift: Offset added to the layer indices.
    num_layers: Total number of layers.

  Returns:
    The shifted (clamped) indices, or `None`.
  """
  if isinstance(layers, slice):
    idx = jnp.arange(layers.start, layers.stop, layers.step) + shift
    return jnp.clip(idx, 0, num_layers - 1)
  idx = layers + shift
  return idx if 0 <= idx < num_layers else None


def _fwd_step_xs(w: Any, w_aux: Any, layers: slice | int, shifted: bool) -> tuple[Any, Any, Any]:
  """Inputs `(w_layer, w_aux_layer, pre_idx)` of forward steps for `layers`.

  Without `collect_fn_pre` (`shifted=False`) a step collects the layer it is
  given. With it, the layer arrives pre-collected in the carry and the step
  instead pre-collects the following layer, `pre_idx` (`None` past the last).

  Args:
    w: Stacked weights, layer dimension first.
    w_aux: Stacked non-differentiable per-layer inputs, or `None`.
    layers: Static layer slice or index of the steps.
    shifted: Whether `collect_fn_pre` is used.

  Returns:
    The tuple `(w_layer, w_aux_layer, pre_idx)`.
  """
  if shifted:
    num_layers = jax.tree.leaves(w)[0].shape[0]
    return None, None, _shifted_indices(layers, 1, num_layers)
  return _layer_slice(w, layers), _layer_slice(w_aux, layers), None


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16),
)
def w_collect_pipelined_scan_prologue_epilogue(
    prologue_fn: PipelinedScanLayer,
    scan_body_fn: PipelinedScanLayer,
    epilogue_fn: PipelinedScanLayer,
    x: AnyJaxArrayOrPyTree,
    w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici_fwd: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_dcn: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici_first: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_ici_second: Callable[..., AnyJaxArrayOrPyTree],
    collect_fn_ici_bwd: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    quantize_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None = None,
    offload_fn: Callable[[Any], Any] = lambda res: res,
    load_fn: Callable[[Any], Any] = lambda res: res,
    w_aux: Any = None,
    collect_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    reduce_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree]:
  """Pipelines weight collection for scanning with prologue and epilogue.

  Args:
    prologue_fn: PipelinedScanLayer to execute for the first layer with
      signatures `fwd(x, w) -> (carry, res)` and `bwd(res, carry_grad, w) ->
      (x_grad, w_mla_grad)`.
    scan_body_fn: PipelinedScanLayer to execute at each scan iteration with
      signatures `fwd(carry, w_curr, w_next) -> ((carry, aux_step), res)` and
      `bwd(res, (carry_grad, aux_step_grad), w_curr, w_next, w_next_expert_grad,
      reduce_ici_first, w_pending_grad_partial, reduce_ici_second) ->
      (carry_grad, w_curr_expert_grad, w_next_grad_partial, w_pending_grad)`.
      `bwd` completes the unreduced gradient of `w_next` by combining (see
      `combine_w_grads`) its own contribution with `w_next_expert_grad`, the
      `w_curr_expert_grad` returned by the previous backward step (that of
      `w_next`'s layer). It then reduces it over the ICI axes in two stages,
      each placed by the step that runs it: `reduce_ici_first(grad) ->
      w_next_grad_partial`, which the next backward step gets as
      `w_pending_grad_partial` (`None` for the first), then
      `reduce_ici_second(w_pending_grad_partial) -> w_pending_grad`.
      `reduce_ici_second(w_pending_grad_partial, defer=True)` instead defers the
      second stage (e.g. as `DeferredReduce`s, see `reduce_along_axis`) for
      `bwd` to run inside the `shard_map`s of other collectives and finish into
      `w_pending_grad`; the keyword is forwarded to `reduce_fn_ici_second`.
      `w_next_grad_partial` is only available once the weights that the step
      ICI-collects for the next one are too, which lets that collective be
      issued ahead of the first stage on a shared queue.
    epilogue_fn: PipelinedScanLayer to execute for the last layer with
      signatures `fwd(carry, w) -> ((out, aux_last), res)` and `bwd(res,
      (out_grad, aux_last_grad), w) -> (carry_grad, w_expert_grad)`.
    x: Input array.
    w: Uncollected weights to scan over. Dimension 0 is the dimension to scan
      over and must be present (num_layers >= 3).
    collect_fn_dcn: Function to collect weights over the DCN axis.
    collect_fn_ici_fwd: Function to collect weights over the ICI axes in forward
      pass.
    reduce_fn_dcn: Dual of `collect_fn_dcn` with signature `(grad, like) ->
      grad_reduced`, reducing the gradient of a layer's DCN-collected weights to
      the sharding of `like`, that layer's uncollected weights.
    reduce_fn_ici_first: First stage of the dual of `collect_fn_ici_bwd`, with
      the same signature as `reduce_fn_dcn` and `like` the layer's DCN-collected
      weights. It reduces the gradient over a part of the ICI axes and leaves it
      unreduced over the ones `reduce_fn_ici_second` reduces.
    reduce_fn_ici_second: Second stage of the dual of `collect_fn_ici_bwd`, with
      the same signature as `reduce_fn_ici_first`, taking its output as `grad`,
      so that `reduce_fn_ici_second(reduce_fn_ici_first(grad, like), like)` is
      the dual of `collect_fn_ici_bwd`. The scan body's `reduce_ici_second` also
      passes it keyword arguments, such as `defer`.
    collect_fn_ici_bwd: Function to collect weights over the ICI axes in
      backward pass.
    quantize_fn: Optional function applied once to all of `w` before any
      collection, e.g. to collect quantized weights. Only its output is
      collected and kept for the backward pass, which returns gradients of `w`
      as if it were the identity.
    offload_fn: Function to offload residuals to host memory.
    load_fn: Function to load residuals to device memory.
    w_aux: Optional pytree of non-differentiable per-layer inputs whose leaves
      have the layer dimension first and are sliced alongside `w`. When given,
      each layer's slice is passed as keyword argument `w_aux` to
      `collect_fn_pre`, `reduce_fn_pre`, `collect_fn_ici_fwd`,
      `collect_fn_ici_bwd`, `reduce_fn_ici_first` and `reduce_fn_ici_second`
      (the slice of the collected or reduced layer) and to the `fwd`/`bwd` of
      `prologue_fn`, `scan_body_fn` (the slice of `w_curr`) and `epilogue_fn`.
      Its cotangent is `None`.
    collect_fn_pre: Optional linear per-layer transform applied to the
      DCN-collected, still ICI-sharded weights before the ICI collect, so that a
      layer is collected as `collect_fn_ici(collect_fn_pre(collect_fn_dcn(w)))`
      (e.g. an expert shuffle). To keep it off the critical path of the ICI
      collect, its output for a layer is computed one scan step before the step
      that ICI-collects the layer and carried across, so both collectives
      overlap with a full step of compute.
    reduce_fn_pre: Dual of `collect_fn_pre` with signature `grad -> grad`,
      required with it. It is applied to the layer's gradient after the first
      stage of the ICI reduction, so it must commute with `reduce_fn_ici_second`
      (e.g. by only transforming leaves that `reduce_fn_ici_second` leaves
      unchanged).

  The last scan body step is peeled out of the scan so that the backward pass
  can pipeline the gradient reduction: a layer's gradient is reduced over a
  part of the ICI axes in the step that completes it, then over the rest of
  them and the DCN axis in the next step, overlapping the reduction with that
  step's backward compute.

  Returns:
    A tuple of (output, aux) where aux is the stacked auxiliary outputs across
    all scan iterations.
  """
  del offload_fn, load_fn, collect_fn_ici_bwd, reduce_fn_dcn
  del reduce_fn_ici_first, reduce_fn_ici_second, reduce_fn_pre
  num_layers = jax.tree.leaves(w)[0].shape[0]
  assert num_layers >= 3, (
      "there must be at least three layers to perform a pipelined scan with a" " peeled prologue and epilogue"
  )
  if quantize_fn is not None:
    w = quantize_fn(w)
  shifted = collect_fn_pre is not None
  fns = _layer_collect(collect_fn_dcn, collect_fn_ici_fwd, collect_fn_pre, w, w_aux)
  step_xs = functools.partial(_fwd_step_xs, w, w_aux, shifted=shifted)

  w_aux_0 = _layer_slice(w_aux, 0)
  w_first = fns.collect(_layer_slice(w, 0), w_aux_0)
  carry = prologue_fn.fwd(x, w_first, **_w_aux_kwargs(w_aux_0))[0]

  w_aux_1 = _layer_slice(w_aux, 1)
  w_second = fns.collect(_layer_slice(w, 1), w_aux_1)
  pre_carry = fns.pre_carry(2 if shifted else None)
  carry, aux_first = scan_body_fn.fwd(carry, w_first, w_second, **_w_aux_kwargs(w_aux_0))[0]

  def _scan_step(carry_tuple, xs):
    w_n_sharded, w_aux_n, pre_idx = xs
    activations, w_curr, w_aux_curr, pre_carry = carry_tuple
    if shifted:
      w_pre_n, w_aux_n = pre_carry
      w_next = fns.collect_ici(w_pre_n, w_aux_n)
    else:
      w_next = fns.collect(w_n_sharded, w_aux_n)
    pre_carry = fns.pre_carry(pre_idx)
    activations, aux_step = scan_body_fn.fwd(activations, w_curr, w_next, **_w_aux_kwargs(w_aux_curr))[0]
    return (activations, w_next, w_aux_n, pre_carry), aux_step

  def body(carry_tuple, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_tuple, aux_0 = _scan_step(carry_tuple, xs_0)
    (carry_tuple, aux_0), xs_1 = jax.lax.optimization_barrier(((carry_tuple, aux_0), xs_1))
    carry_tuple, aux_1 = _scan_step(carry_tuple, xs_1)
    return carry_tuple, (aux_0, aux_1)

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2
  even = slice(2, 2 + 2 * num_pairs, 2)
  odd = slice(3, 2 + 2 * num_pairs, 2)

  (carry, w_curr, w_aux_curr, pre_carry), (aux_0, aux_1) = jax.lax.scan(
      body,
      (carry, w_second, w_aux_1, pre_carry),
      (step_xs(even), step_xs(odd)),
  )
  aux_parts = [jax.tree.map(lambda x: x[jnp.newaxis], aux_first)]
  if num_pairs > 0:
    aux_scanned = jax.tree.map(_interleave_arrays, aux_0, aux_1)
    aux_parts.append(aux_scanned)
  if num_scanned % 2 == 1:
    rem = 2 + 2 * num_pairs
    (carry, w_curr, w_aux_curr, pre_carry), aux_rem = _scan_step((carry, w_curr, w_aux_curr, pre_carry), step_xs(rem))
    aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_rem))

  # Peeled final scan body step, whose backward pass primes the pipelined DCN
  # gradient reduction.
  (carry, w_last, w_aux_last, _), aux_last_sb = _scan_step(
      (carry, w_curr, w_aux_curr, pre_carry), step_xs(num_layers - 1)
  )
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last_sb))

  out, aux_last = epilogue_fn.fwd(carry, w_last, **_w_aux_kwargs(w_aux_last))[0]
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last))

  aux = jax.tree.map(
      lambda *parts: jnp.concatenate(parts, axis=0),
      *aux_parts,
  )
  return out, aux


def _is_pinned_host(x: Any) -> bool:
  sharding = getattr(jax.typeof(x), "sharding", None)
  return getattr(sharding, "memory_kind", None) == "pinned_host"


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_prologue_epilogue_fwd(
    prologue_fn: PipelinedScanLayer,
    scan_body_fn: PipelinedScanLayer,
    epilogue_fn: PipelinedScanLayer,
    x: AnyJaxArrayOrPyTree,
    w: AnyJaxArrayOrPyTree,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici_fwd: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_dcn: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici_first: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_ici_second: Callable[..., AnyJaxArrayOrPyTree],
    collect_fn_ici_bwd: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    quantize_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None = None,
    offload_fn: Callable[[Any], Any] = lambda res: res,
    load_fn: Callable[[Any], Any] = lambda res: res,
    w_aux: Any = None,
    collect_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
    reduce_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None = None,
) -> tuple[tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], Any]:
  """Forward pass of w_collect_pipelined_scan_prologue_epilogue."""
  del load_fn, collect_fn_ici_bwd, reduce_fn_dcn
  del reduce_fn_ici_first, reduce_fn_ici_second
  num_layers = jax.tree.leaves(w)[0].shape[0]
  assert num_layers >= 3, (
      "there must be at least three layers to perform a pipelined scan and its"
      " transpose, which peel a scan body step at each end of the scan"
  )
  assert (collect_fn_pre is None) == (reduce_fn_pre is None), "collect_fn_pre and reduce_fn_pre must be given together"
  # Only the quantized weights are collected, here and in the backward pass.
  if quantize_fn is not None:
    w = quantize_fn(w)
  shifted = collect_fn_pre is not None
  fns = _layer_collect(collect_fn_dcn, collect_fn_ici_fwd, collect_fn_pre, w, w_aux)
  step_xs = functools.partial(_fwd_step_xs, w, w_aux, shifted=shifted)

  w_aux_0 = _layer_slice(w_aux, 0)
  with jax.named_scope("prologue_fwd"):
    w_first = fns.collect(_layer_slice(w, 0), w_aux_0)
    carry, res_prologue = prologue_fn.fwd(x, w_first, **_w_aux_kwargs(w_aux_0))

  w_aux_1 = _layer_slice(w_aux, 1)
  with jax.named_scope("scan_step_fwd_0"):
    w_second = fns.collect(_layer_slice(w, 1), w_aux_1)
    pre_carry = fns.pre_carry(2 if shifted else None)
    res_prologue_host = offload_fn(res_prologue)
    (carry, aux_first), res_sb_0 = scan_body_fn.fwd(carry, w_first, w_second, **_w_aux_kwargs(w_aux_0))

  res_sb_0_shaped = jax.eval_shape(offload_fn, res_sb_0)
  has_host_offload = any(_is_pinned_host(s) for s in jax.tree.leaves(res_sb_0_shaped))

  def _split_for_offload(res):
    if not has_host_offload:
      return None, res
    res_to_offload = jax.tree.map(
        lambda x, s: x if _is_pinned_host(s) else None,
        res,
        res_sb_0_shaped,
        is_leaf=lambda x: x is None,
    )
    res_device = jax.tree.map(
        lambda x, s: None if _is_pinned_host(s) else x,
        res,
        res_sb_0_shaped,
        is_leaf=lambda x: x is None,
    )
    return res_to_offload, res_device

  def _offload(res_to_offload):
    return None if res_to_offload is None else offload_fn(res_to_offload)

  res_sb_0_offload, res_sb_0_device = _split_for_offload(res_sb_0)

  def _scan_step_fwd(carry_tuple, xs):
    w_n_sharded, w_aux_n, pre_idx = xs
    activations, w_curr, w_aux_curr, pre_carry, res_prev_offload = carry_tuple
    res_prev_host = _offload(res_prev_offload)
    if shifted:
      w_pre_n, w_aux_n = pre_carry
      w_next = fns.collect_ici(w_pre_n, w_aux_n)
    else:
      w_next = fns.collect(w_n_sharded, w_aux_n)
    pre_carry = fns.pre_carry(pre_idx)
    (activations, aux_step), res_curr = scan_body_fn.fwd(activations, w_curr, w_next, **_w_aux_kwargs(w_aux_curr))
    res_curr_offload, res_curr_device = _split_for_offload(res_curr)
    return (activations, w_next, w_aux_n, pre_carry, res_curr_offload), (
        res_prev_host,
        res_curr_device,
        aux_step,
    )

  @jax.named_call
  def body_fwd(carry_tuple, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_tuple, (res_prev_0_host, res_curr_0_device, aux_0) = jax.named_call(_scan_step_fwd, name="unrolled_0")(
        carry_tuple, xs_0
    )
    (carry_tuple, aux_0), xs_1 = jax.lax.optimization_barrier(((carry_tuple, aux_0), xs_1))
    carry_tuple, (res_prev_1_host, res_curr_1_device, aux_1) = jax.named_call(_scan_step_fwd, name="unrolled_1")(
        carry_tuple, xs_1
    )
    return carry_tuple, (
        (res_prev_0_host, res_prev_1_host),
        (res_curr_0_device, res_curr_1_device),
        (aux_0, aux_1),
    )

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2
  even = slice(2, 2 + 2 * num_pairs, 2)
  odd = slice(3, 2 + 2 * num_pairs, 2)

  (carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload), (
      (res_prev_0_scanned_host, res_prev_1_scanned_host),
      (res_curr_0_scanned_device, res_curr_1_scanned_device),
      (aux_0, aux_1),
  ) = jax.lax.scan(
      body_fwd,
      (carry, w_second, w_aux_1, pre_carry, res_sb_0_offload),
      (step_xs(even), step_xs(odd)),
  )

  aux_parts = [jax.tree.map(lambda x: x[jnp.newaxis], aux_first)]
  if num_pairs > 0:
    aux_scanned = jax.tree.map(_interleave_arrays, aux_0, aux_1)
    aux_parts.append(aux_scanned)
  res_rem_host = None
  res_rem_device = None
  if num_scanned % 2 == 1:
    rem = 2 + 2 * num_pairs
    (carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload), (
        res_rem_host,
        res_rem_device,
        aux_rem,
    ) = _scan_step_fwd(
        (carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload),
        step_xs(rem),
    )
    aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_rem))

  # Peeled final scan body step. Its transpose primes the pipelined DCN
  # gradient reduction, so its residual is kept separately from the scanned
  # residuals.
  with jax.named_scope("scan_step_fwd_last"):
    (carry, w_last, w_aux_last, _, res_last_sb_offload), (
        res_penultimate_sb_host,
        res_last_sb_device,
        aux_last_sb,
    ) = _scan_step_fwd(
        (carry, w_curr, w_aux_curr, pre_carry, res_curr_sb_offload),
        step_xs(num_layers - 1),
    )
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last_sb))

  with jax.named_scope("epilogue_fwd"):
    res_last_sb_host = _offload(res_last_sb_offload)
    (out, aux_last), res_epilogue = epilogue_fn.fwd(carry, w_last, **_w_aux_kwargs(w_aux_last))
  res_epilogue_host = offload_fn(res_epilogue)
  aux_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], aux_last))

  aux = jax.tree.map(
      lambda *parts: jnp.concatenate(parts, axis=0),
      *aux_parts,
  )
  res = (
      res_prologue_host,
      res_sb_0_device,
      (
          (res_prev_0_scanned_host, res_prev_1_scanned_host),
          (res_curr_0_scanned_device, res_curr_1_scanned_device),
          res_rem_host,
          res_rem_device,
      ),
      res_penultimate_sb_host,
      res_last_sb_host,
      res_last_sb_device,
      res_epilogue_host,
      w,
      w_aux,
  )
  return (out, aux), res


@jt.jaxtyped(typechecker=typeguard.typechecked)
def w_collect_pipelined_scan_prologue_epilogue_bwd(
    prologue_fn: PipelinedScanLayer,
    scan_body_fn: PipelinedScanLayer,
    epilogue_fn: PipelinedScanLayer,
    collect_fn_dcn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    collect_fn_ici_fwd: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_dcn: Callable[[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree],
    reduce_fn_ici_first: Callable[..., AnyJaxArrayOrPyTree],
    reduce_fn_ici_second: Callable[..., AnyJaxArrayOrPyTree],
    collect_fn_ici_bwd: Callable[..., AnyJaxArrayOrPyTree] | None,
    quantize_fn: Callable[[AnyJaxArrayOrPyTree], AnyJaxArrayOrPyTree] | None,
    offload_fn: Callable[[Any], Any],
    load_fn: Callable[[Any], Any],
    collect_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None,
    reduce_fn_pre: Callable[..., AnyJaxArrayOrPyTree] | None,
    res: Any,
    grad_outputs: tuple[AnyJaxArrayOrPyTree, Any],
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree, None]:
  """Backward pass of w_collect_pipelined_scan_prologue_epilogue."""
  del offload_fn, quantize_fn
  if collect_fn_ici_bwd is None:
    collect_fn_ici_bwd = collect_fn_ici_fwd
  del collect_fn_ici_fwd
  # w holds the quantized weights; the gradients are those of the unquantized
  # weights, as if quantization were the identity.
  (
      res_prologue_host,
      res_sb_0_device,
      (
          (res_prev_0_scanned_host, res_prev_1_scanned_host),
          (res_curr_0_scanned_device, res_curr_1_scanned_device),
          res_rem_host,
          res_rem_device,
      ),
      res_penultimate_sb_host,
      res_last_sb_host,
      res_last_sb_device,
      res_epilogue_host,
      w,
      w_aux,
  ) = res
  num_layers = jax.tree.leaves(w)[0].shape[0]
  shifted = collect_fn_pre is not None
  fns = _layer_collect(collect_fn_dcn, collect_fn_ici_bwd, collect_fn_pre, w, w_aux)

  def _load(res_host):
    return None if res_host is None else load_fn(res_host)

  def _merge_offloaded(res_loaded, res_device):
    if res_loaded is None:
      return res_device
    return jax.tree.map(
        lambda l, d: d if l is None else l,
        res_loaded,
        res_device,
        is_leaf=lambda x: x is None,
    )

  out_grad, aux_grad = grad_outputs
  aux_step_0 = jax.tree.map(lambda g: g[0], aux_grad)
  aux_scanned_grad = jax.tree.map(lambda g: g[1:-2], aux_grad)
  aux_last_sb_grad = jax.tree.map(lambda g: g[-2], aux_grad)
  aux_last_grad = jax.tree.map(lambda g: g[-1], aux_grad)

  # 1. Epilogue backward: load res_epilogue and concurrently load res_last_sb
  res_epilogue_device = load_fn(res_epilogue_host)
  w_last_sharded = _layer_slice(w, -1)
  w_last_dcn = collect_fn_dcn(w_last_sharded)
  w_aux_last = _layer_slice(w_aux, -1)
  w_last = fns.collect_ici(fns.pre(w_last_dcn, w_aux_last), w_aux_last)

  # Every layer's weights share the last layer's shardings, which are all the
  # reductions read from them (`collect_fn_pre` preserves shardings).
  def _reduce_ici_first(w_unreduced, w_aux_layer):
    """First stage of reducing a layer's gradient over the ICI axes."""
    return reduce_fn_ici_first(w_unreduced, w_last_dcn, **_w_aux_kwargs(w_aux_layer))

  def _reduce_ici_second(w_partial, w_aux_layer, **kwargs):
    """Second stage of reducing a layer's gradient over the ICI axes."""
    return reduce_fn_ici_second(w_partial, w_last_dcn, **_w_aux_kwargs(w_aux_layer), **kwargs)

  def _reduce_pre(w_partial, w_aux_layer):
    """Applies `reduce_fn_pre` (if any) to a partially reduced gradient."""
    if reduce_fn_pre is None:
      return w_partial
    return reduce_fn_pre(w_partial, **_w_aux_kwargs(w_aux_layer))

  def _reduce_dcn(w_ici):
    """Reduces an ICI-reduced gradient over the DCN axis."""
    return reduce_fn_dcn(w_ici, w_last_sharded)

  with jax.named_scope("epilogue_bwd"):
    res_last_sb_loaded = _load(res_last_sb_host)
    w_aux_penultimate = _layer_slice(w_aux, -2)
    w_curr = fns.collect(_layer_slice(w, -2), w_aux_penultimate)
    pre_carry = fns.pre_carry(num_layers - 3 if shifted else None)
    carry_grad, w_last_expert_grad_unreduced = epilogue_fn.bwd(
        res_epilogue_device,
        (out_grad, aux_last_grad),
        w_last,
        **_w_aux_kwargs(w_aux_last),
    )

  # 2. Reverse scan body backward.
  #
  # Each step completes one layer's gradient and runs the first stage of its
  # ICI reduction, but defers the rest of its reduction to the next step: that
  # step applies `reduce_fn_pre` to it before its own backward compute so that
  # the two overlap, runs the second stage of its ICI reduction (where its scan
  # body places that) and then reduces it over the DCN axis.
  # `w_partial_pending` is the partially reduced gradient of the layer
  # completed by the previous step (with the layer's `w_aux`), and is `None`
  # only for the peeled step that primes the pipeline.
  #
  # Each step also ICI-collects the weights of the layer whose backward pass
  # runs in the next step. Without `collect_fn_pre`, the step is given that
  # layer's sharded weights (`w_prev_sharded`); with it, they arrive
  # pre-collected in `pre_carry` and the step pre-collects the layer before
  # (`pre_idx`) for the step after it.
  def _scan_step_bwd(carry, xs):
    (
        carry_grad,
        w_prev_expert_grad_unreduced,
        w_curr,
        w_next,
        w_aux_curr,
        w_aux_next,
        res_curr_loaded,
        w_partial_pending,
        pre_carry,
    ) = carry
    (
        res_curr_device,
        res_prev_host,
        w_prev_sharded,
        w_aux_prev,
        aux_step_grad,
        pre_idx,
    ) = xs

    w_partial_pending_grad, w_aux_partial_pending = (None, None) if w_partial_pending is None else w_partial_pending
    if w_partial_pending is not None:
      w_partial_pending_grad = _reduce_pre(w_partial_pending_grad, w_aux_partial_pending)

    # Load previous iteration's offloaded residuals to device
    res_prev_loaded = _load(res_prev_host)
    res_curr_full = _merge_offloaded(res_curr_loaded, res_curr_device)

    if shifted:
      w_prev_pre, w_aux_prev = pre_carry
    else:
      w_prev_pre = collect_fn_dcn(w_prev_sharded)
    w_prev_collected = fns.collect_ici(w_prev_pre, w_aux_prev)
    pre_carry = fns.pre_carry(pre_idx)

    def _reduce_ici_first_with_collect(w_unreduced):
      # Collectives on a SparseCore queue complete in the order they were
      # issued. `scan_body_fn` needs the first reduction stage early in the
      # step, but nothing needs the all-gather above before the next step, so
      # the reduction would be due first and the all-gather could only start
      # behind it. Tying the all-gather's result to the reduction's gives both
      # the same deadline, which lets the all-gather, whose operand is
      # available first, be issued first.
      w_partial = _reduce_ici_first(w_unreduced, w_aux_next)
      w_partial, _ = jax.lax.optimization_barrier((w_partial, w_prev_collected))
      return w_partial

    (
        carry_grad,
        w_curr_expert_grad_unreduced,
        w_next_grad_partial,
        w_partial_pending_grad_ici,
    ) = scan_body_fn.bwd(
        res_curr_full,
        (carry_grad, aux_step_grad),
        w_curr,
        w_next,
        w_prev_expert_grad_unreduced,
        _reduce_ici_first_with_collect,
        w_partial_pending_grad,
        functools.partial(_reduce_ici_second, w_aux_layer=w_aux_partial_pending),
        **_w_aux_kwargs(w_aux_curr),
    )
    w_pending_reduced = None if w_partial_pending is None else _reduce_dcn(w_partial_pending_grad_ici)

    return (
        carry_grad,
        w_curr_expert_grad_unreduced,
        w_prev_collected,
        w_curr,
        w_aux_prev,
        w_aux_curr,
        res_prev_loaded,
        (w_next_grad_partial, w_aux_next),
        pre_carry,
    ), w_pending_reduced

  def _step_xs(res_curr_device, res_prev_host, layers, aux_step_grad):
    """Inputs of backward steps ICI-collecting `layers`."""
    if shifted:
      pre_idx = _shifted_indices(layers, -1, num_layers)
      return (
          res_curr_device,
          res_prev_host,
          None,
          None,
          aux_step_grad,
          pre_idx,
      )
    return (
        res_curr_device,
        res_prev_host,
        _layer_slice(w, layers),
        _layer_slice(w_aux, layers),
        aux_step_grad,
        None,
    )

  @jax.named_call
  def body_bwd(carry, xs_pair):
    xs_0, xs_1 = xs_pair
    carry_in = carry
    carry, w_1_reduced = jax.named_call(_scan_step_bwd, name="unrolled_1")(carry, xs_1)
    carry_for_barrier = jax.tree.map(
        lambda x_in, x_out: None if x_out is x_in else x_out,
        carry_in,
        carry,
    )
    res_curr_0_device, *xs_0_rest = xs_0
    (carry_for_barrier, w_1_reduced), xs_0_rest = jax.lax.optimization_barrier(
        ((carry_for_barrier, w_1_reduced), xs_0_rest)
    )
    xs_0 = (res_curr_0_device, *xs_0_rest)
    carry = jax.tree.map(
        lambda x_in, x_bar: x_in if x_bar is None else x_bar,
        carry_in,
        carry_for_barrier,
        is_leaf=lambda x: x is None,
    )
    carry, w_0_reduced = jax.named_call(_scan_step_bwd, name="unrolled_0")(carry, xs_0)
    return carry, (w_0_reduced, w_1_reduced)

  num_scanned = num_layers - 3
  num_pairs = num_scanned // 2

  carry_bwd = (
      carry_grad,
      w_last_expert_grad_unreduced,
      w_curr,
      w_last,
      w_aux_penultimate,
      w_aux_last,
      res_last_sb_loaded,
      None,
      pre_carry,
  )

  # Peeled scan body step. It completes the last layer and only runs the first
  # stage of its ICI reduction, priming the pipeline so that every step below
  # emits the fully reduced gradient of the layer completed by the step before
  # it.
  with jax.named_scope("scan_step_bwd_last"):
    carry_bwd, _ = _scan_step_bwd(
        carry_bwd,
        _step_xs(
            res_last_sb_device,
            res_penultimate_sb_host,
            num_layers - 3,
            aux_last_sb_grad,
        ),
    )

  w_rem_reduced = None
  if num_scanned % 2 == 1:
    xs_rem = _step_xs(
        res_rem_device,
        res_rem_host,
        num_scanned - 1,
        jax.tree.map(lambda g: g[-1], aux_scanned_grad),
    )
    carry_bwd, w_rem_reduced = _scan_step_bwd(carry_bwd, xs_rem)

  w_grads_reduced = None
  if num_pairs > 0:
    even = slice(0, 2 * num_pairs, 2)
    odd = slice(1, 1 + 2 * num_pairs, 2)
    aux_even_grad = jax.tree.map(lambda g: g[0 : 2 * num_pairs : 2], aux_scanned_grad)
    aux_odd_grad = jax.tree.map(lambda g: g[1 : 2 * num_pairs : 2], aux_scanned_grad)

    xs_0 = _step_xs(res_curr_0_scanned_device, res_prev_0_scanned_host, even, aux_even_grad)
    xs_1 = _step_xs(res_curr_1_scanned_device, res_prev_1_scanned_host, odd, aux_odd_grad)

    carry_bwd, (w_0_reduced_scanned, w_1_reduced_scanned) = jax.lax.scan(
        body_bwd,
        carry_bwd,
        (xs_0, xs_1),
        reverse=True,
    )

    w_grads_reduced = jax.tree.map(
        _interleave_arrays,
        w_0_reduced_scanned,
        w_1_reduced_scanned,
    )

  (
      carry_grad,
      w_second_expert_grad_unreduced,
      w_0,
      w_1,
      w_aux_0,
      w_aux_1,
      res_sb_0_loaded,
      (w_third_grad_partial, w_aux_2),
      _,
  ) = carry_bwd

  # 3. Scan body 0 backward: apply `reduce_fn_pre` to the third layer and load
  # the prologue residuals, both concurrently with this step's compute, then
  # finish reducing the third layer once its scan body ran the second stage of
  # its ICI reduction.
  with jax.named_scope("scan_step_bwd_0"):
    w_third_grad_partial = _reduce_pre(w_third_grad_partial, w_aux_2)
    res_prologue_device = load_fn(res_prologue_host)
    res_sb_0_full = _merge_offloaded(res_sb_0_loaded, res_sb_0_device)

    (
        carry_grad,
        w_0_expert_grad_unreduced,
        w_second_grad_partial,
        w_third_grad_ici,
    ) = scan_body_fn.bwd(
        res_sb_0_full,
        (carry_grad, aux_step_0),
        w_0,
        w_1,
        w_second_expert_grad_unreduced,
        functools.partial(_reduce_ici_first, w_aux_layer=w_aux_1),
        w_third_grad_partial,
        functools.partial(_reduce_ici_second, w_aux_layer=w_aux_2),
        **_w_aux_kwargs(w_aux_0),
    )
    w_third_grad_reduced = _reduce_dcn(w_third_grad_ici)

  # 4. Prologue backward: the second layer's reduction overlaps the prologue
  # compute, which leaves only the first layer's reduction to drain the
  # pipeline.
  with jax.named_scope("prologue_bwd"):
    w_second_grad_reduced = _reduce_dcn(_reduce_ici_second(_reduce_pre(w_second_grad_partial, w_aux_1), w_aux_1))

    x_grad, w_0_mla_grad_unreduced = prologue_fn.bwd(res_prologue_device, carry_grad, w_0, **_w_aux_kwargs(w_aux_0))

    w_0_unreduced = combine_w_grads(w_0_mla_grad_unreduced, w_0_expert_grad_unreduced)
    w_first_grad_reduced = _reduce_dcn(
        _reduce_ici_second(
            _reduce_pre(_reduce_ici_first(w_0_unreduced, w_aux_0), w_aux_0),
            w_aux_0,
        )
    )

  w_grad_parts = [
      jax.tree.map(lambda x: x[jnp.newaxis], g)
      for g in (
          w_first_grad_reduced,
          w_second_grad_reduced,
          w_third_grad_reduced,
      )
  ]
  if w_grads_reduced is not None:
    w_grad_parts.append(w_grads_reduced)
  if w_rem_reduced is not None:
    w_grad_parts.append(jax.tree.map(lambda x: x[jnp.newaxis], w_rem_reduced))
  w_grad = jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *w_grad_parts)
  return x_grad, w_grad, None


w_collect_pipelined_scan_prologue_epilogue.defvjp(
    w_collect_pipelined_scan_prologue_epilogue_fwd,
    w_collect_pipelined_scan_prologue_epilogue_bwd,
)


def silu_mul_tile(x: jax.Array, y: jax.Array, c: jax.Array, out_dtype: jt.DTypeLike) -> jax.Array:
  """Computes `(silu(x) * y * c).astype(out_dtype)` in-register."""
  return (jax.nn.silu(x) * y * c).astype(out_dtype)


def quantize_tile(x: jax.Array, inv_scale: jax.Array, qtype: jt.DTypeLike) -> jax.Array:
  """Quantizes tile `x` to `qtype` given `inv_scale = 1.0 / scale`."""
  q_max = float(jnp.finfo(qtype).max)
  return jnp.clip(x.astype(jnp.float32) * inv_scale, -q_max, q_max).astype(qtype)


def _ragged_silu_mul_kernel(
    x_ref: Any,
    y_ref: Any,
    c_ref: Any,
    out_ref: Any,
    *,
    out_scale_ref: Any = None,
) -> None:
  """Kernel body for ragged_silu_mul."""
  act = silu_mul_tile(x_ref[...], y_ref[...], c_ref[...], x_ref.dtype)
  if out_scale_ref is not None:
    inv_scale = 1.0 / out_scale_ref[0].astype(jnp.float32)
    out_ref[...] = quantize_tile(act, inv_scale, out_ref.dtype)
  else:
    out_ref[...] = act.astype(out_ref.dtype)


def _ragged_silu_mul_x_specs(
    x_offset: jax.Array,
    x_dtype: jnp.dtype,
    block_size_tokens: int,
    block_size_hidden: int,
    num_hidden_blocks: int,
) -> tuple[pl.BlockSpec, pl.BlockSpec]:
  """Block specs of g0 and g1 in rows [x_offset + i * block_size_tokens, ...).

  Rows are indexed by element so `x_offset` needs only be aligned to the
  sublane tiling, and `emit_pipeline` shrinks a block that runs past the end of
  x to the remaining rows.

  Args:
    x_offset: Row of x holding token 0.
    x_dtype: Dtype of x, which sets the sublane tiling `x_offset` is aligned to.
    block_size_tokens: Tile size along the token dim.
    block_size_hidden: Tile size along the hidden dim.
    num_hidden_blocks: Number of hidden blocks in each of g0 and g1.

  Returns:
    Block specs of (g0, g1).
  """
  sublanes = 8 * 4 // jnp.dtype(x_dtype).itemsize
  shape = (pl.Element(block_size_tokens), block_size_hidden)
  row = lambda i: pl.multiple_of(x_offset + i * block_size_tokens, sublanes)
  return (
      pl.BlockSpec(shape, lambda i, j: (row(i), j)),
      pl.BlockSpec(shape, lambda i, j: (row(i), j + num_hidden_blocks)),
  )


def _ragged_silu_mul_fwd_pipeline(
    x_offset_ref: Any,
    num_tokens_ref: Any,
    x_ref: Any,
    c_ref: Any,
    out_scale_ref: Any,
    out_ref: Any,
    *,
    block_size_tokens: int,
    block_size_hidden: int,
    num_hidden_blocks: int,
) -> None:
  """Runs `_ragged_silu_mul_kernel` over the valid token blocks."""
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    g0_spec, g1_spec = _ragged_silu_mul_x_specs(
        x_offset_ref[0],
        x_ref.dtype,
        block_size_tokens,
        block_size_hidden,
        num_hidden_blocks,
    )
    pltpu.emit_pipeline(
        functools.partial(_ragged_silu_mul_kernel, out_scale_ref=out_scale_ref),
        grid=(pl.cdiv(num_tokens, block_size_tokens), num_hidden_blocks),
        in_specs=[
            g0_spec,
            g1_spec,
            pl.BlockSpec((block_size_tokens, 1), lambda i, j: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_size_tokens, block_size_hidden), lambda i, j: (i, j)),
    )(x_ref, x_ref, c_ref, out_ref)


def _pallas_ragged_silu_mul(
    x: jt.Num[jax.Array, "x_rows double_hidden_dim"],
    x_offset: jt.Int[jax.Array, ""] | int,
    coeffs: jt.Num[jax.Array, "max_num_tokens 1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
    block_size_hidden: int,
    out_dtype: jt.DTypeLike | None = None,
    out_scale: jt.Float[jax.Array, "1 1"] | None = None,
) -> jt.Num[jax.Array, "max_num_tokens hidden_dim"]:
  """Pallas call wrapper for ragged_silu_mul forward pass."""
  max_num_tokens = coeffs.shape[0]
  hidden_dim = x.shape[1] // 2
  block_size_hidden = min(block_size_hidden, hidden_dim)
  num_hidden_blocks = (hidden_dim + block_size_hidden - 1) // block_size_hidden
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  target_dtype = jnp.dtype(out_dtype) if out_dtype is not None else x.dtype
  out_scale_smem = out_scale.astype(jnp.float32).reshape(1) if out_scale is not None else None
  out_scale_spec = smem if out_scale_smem is not None else None
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  call_kwargs = {}
  if out_dtype is not None or out_scale is not None:
    call_kwargs["name"] = "ragged_silu_mul_fwd-fp8"
  return pl.pallas_call(
      functools.partial(
          _ragged_silu_mul_fwd_pipeline,
          block_size_tokens=block_size_tokens,
          block_size_hidden=block_size_hidden,
          num_hidden_blocks=num_hidden_blocks,
      ),
      out_shape=jax.ShapeDtypeStruct(
          (max_num_tokens, hidden_dim),
          target_dtype,
          manual_axis_type=manual_axis_type,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm, hbm, out_scale_spec],
          out_specs=hbm,
      ),
      **call_kwargs,
  )(
      jnp.asarray(x_offset, jnp.int32)[None],
      jnp.asarray(num_tokens, jnp.int32)[None],
      x,
      coeffs,
      out_scale_smem,
  )


def _ragged_silu_mul_bwd_kernel(
    g_ref: Any,
    x_ref: Any,
    y_ref: Any,
    c_ref: Any,
    dx_ref: Any,
    dy_ref: Any,
    dc_ref: Any,
    *,
    g_mask_max: Any = None,
    num_tokens_ref: Any = None,
    amax_vmem_ref: Any = None,
) -> None:
  """Kernel body for ragged_silu_mul backward pass."""
  x_val = x_ref[...]
  y_val = y_ref[...]
  g_val = g_ref[...]
  c_val = c_ref[...]

  sig_x = jax.nn.sigmoid(x_val)
  silu_x = x_val * sig_x
  dsilu_x = sig_x * (1.0 + x_val * (1.0 - sig_x))

  if g_mask_max is not None:
    mask_bound = g_mask_max[0] if hasattr(g_mask_max, "ndim") and g_mask_max.ndim > 0 else g_mask_max
    act_bf16 = (silu_x * y_val * c_val).astype(x_val.dtype)
    g_val = jnp.where(jnp.abs(act_bf16) <= mask_bound, g_val, 0).astype(g_val.dtype)

  gc_val = g_val * c_val
  dx = (gc_val * y_val * dsilu_x).astype(dx_ref.dtype)
  dy = (gc_val * silu_x).astype(dy_ref.dtype)
  dx_ref[...] = dx
  dy_ref[...] = dy

  if amax_vmem_ref is not None and num_tokens_ref is not None:
    row_idx = pl.program_id(0) * dx.shape[0] + jax.lax.broadcasted_iota(jnp.int32, (dx.shape[0], 1), 0)
    valid = row_idx < num_tokens_ref[0]
    tile_abs = jnp.where(valid, jnp.maximum(jnp.abs(dx), jnp.abs(dy)), 0)
    tile_max = jnp.max(tile_abs, axis=0).astype(jnp.float32)
    amax_vmem_ref[0] = jnp.maximum(amax_vmem_ref[0], tile_max)

  # dc is reduced over the hidden dim, which is tiled along grid axis 1.
  @pl.when(pl.program_id(1) == 0)
  def _init():
    dc_ref[...] = jnp.zeros_like(dc_ref)

  f32 = jnp.float32
  dc_ref[...] += jnp.sum(
      g_val.astype(f32) * silu_x.astype(f32) * y_val.astype(f32),
      axis=-1,
      keepdims=True,
  ).astype(dc_ref.dtype)


def _ragged_silu_mul_bwd_pipeline(
    x_offset_ref: Any,
    num_tokens_ref: Any,
    g_ref: Any,
    x_ref: Any,
    c_ref: Any,
    g_mask_max_ref: Any,
    dxy_ref: Any,
    dc_ref: Any,
    amax_smem_ref: Any = None,
    amax_vmem_ref: Any = None,
    *,
    block_size_tokens: int,
    block_size_hidden: int,
    num_hidden_blocks: int,
    static_g_mask_max: float | None = None,
) -> None:
  """Runs `_ragged_silu_mul_bwd_kernel` over the valid token blocks."""
  num_tokens = num_tokens_ref[0]
  mask_max = g_mask_max_ref if g_mask_max_ref is not None else static_g_mask_max

  if amax_vmem_ref is not None:
    amax_vmem_ref[...] = jnp.zeros_like(amax_vmem_ref)

  @pl.when(num_tokens > 0)
  def _():
    g0_spec, g1_spec = _ragged_silu_mul_x_specs(
        x_offset_ref[0],
        x_ref.dtype,
        block_size_tokens,
        block_size_hidden,
        num_hidden_blocks,
    )
    hidden_shape = (block_size_tokens, block_size_hidden)
    hidden_spec = pl.BlockSpec(hidden_shape, lambda i, j: (i, j))
    dy_spec = pl.BlockSpec(hidden_shape, lambda i, j: (i, j + num_hidden_blocks))
    coeffs_spec = pl.BlockSpec((block_size_tokens, 1), lambda i, j: (i, 0))
    # dx and dy are written directly into the two halves of dxy_ref.
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_silu_mul_bwd_kernel,
            g_mask_max=mask_max,
            num_tokens_ref=num_tokens_ref,
            amax_vmem_ref=amax_vmem_ref,
        ),
        grid=(pl.cdiv(num_tokens, block_size_tokens), num_hidden_blocks),
        in_specs=[hidden_spec, g0_spec, g1_spec, coeffs_spec],
        out_specs=[hidden_spec, dy_spec, coeffs_spec],
    )(g_ref, x_ref, x_ref, c_ref, dxy_ref, dxy_ref, dc_ref)

  if amax_smem_ref is not None and amax_vmem_ref is not None:
    amax_smem_ref[0, 0] = jnp.max(amax_vmem_ref[0])


def _pallas_ragged_silu_mul_bwd(
    g: jt.Num[jax.Array, "max_num_tokens hidden_dim"],
    x: jt.Num[jax.Array, "x_rows double_hidden_dim"],
    x_offset: jt.Int[jax.Array, ""] | int,
    coeffs: jt.Num[jax.Array, "max_num_tokens 1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
    block_size_hidden: int,
    g_mask_max: jt.Float[jax.Array, ""] | float | None = None,
    with_absmax: bool = False,
) -> Any:
  """Computes backward pass gradients w.r.t. x and coeffs using Pallas."""
  max_num_tokens, hidden_dim = g.shape
  if (g_mask_max is not None or with_absmax) and block_size_tokens * block_size_hidden > 512 * 1024:
    block_size_tokens = min(block_size_tokens, 512)
  block_size_hidden = min(block_size_hidden, hidden_dim)
  num_hidden_blocks = (hidden_dim + block_size_hidden - 1) // block_size_hidden
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  static_g_mask_max = None
  g_mask_max_smem = None
  if isinstance(g_mask_max, (int, float)):
    static_g_mask_max = float(g_mask_max)
  elif g_mask_max is not None:
    g_mask_max_smem = jnp.asarray(g_mask_max, jnp.float32).reshape(1)
  g_mask_spec = smem if g_mask_max_smem is not None else None
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  dxy_struct = jax.ShapeDtypeStruct(
      (max_num_tokens, 2 * hidden_dim),
      x.dtype,
      manual_axis_type=manual_axis_type,
  )
  dc_struct = jax.ShapeDtypeStruct(
      coeffs.shape,
      jnp.float32,
      manual_axis_type=manual_axis_type,
  )
  if with_absmax:
    amax_struct = jax.ShapeDtypeStruct(
        (1, 1),
        jnp.float32,
        manual_axis_type=manual_axis_type,
    )
    out_shape = (dxy_struct, dc_struct, amax_struct)
    out_specs = [hbm, hbm, smem]
    scratch_shapes = [pltpu.VMEM((1, block_size_hidden), jnp.float32)]
  else:
    out_shape = (dxy_struct, dc_struct)
    out_specs = [hbm, hbm]
    scratch_shapes = []
  res = pl.pallas_call(
      functools.partial(
          _ragged_silu_mul_bwd_pipeline,
          block_size_tokens=block_size_tokens,
          block_size_hidden=block_size_hidden,
          num_hidden_blocks=num_hidden_blocks,
          static_g_mask_max=static_g_mask_max,
      ),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm, hbm, hbm, g_mask_spec],
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      name="ragged_silu_mul_bwd-absmax" if with_absmax else None,
  )(
      jnp.asarray(x_offset, jnp.int32)[None],
      jnp.asarray(num_tokens, jnp.int32)[None],
      g,
      x,
      coeffs,
      g_mask_max_smem,
  )
  if with_absmax:
    dxy, dc, amax = res
    return dxy, dc.astype(coeffs.dtype), amax
  dxy, dc = res
  return dxy, dc.astype(coeffs.dtype)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_silu_mul_fwd(
    x: jt.Num[jax.Array, "x_rows double_hidden_dim"],
    x_offset: jt.Int[jax.Array, ""] | int,
    coeffs: jt.Num[jax.Array, "max_num_tokens 1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens: int = 128,
    block_size_hidden: int = 128,
    out_dtype: jt.DTypeLike | None = None,
    out_scale: jt.Float[jax.Array, "1 1"] | None = None,
) -> jt.Num[jax.Array, "max_num_tokens hidden_dim"]:
  """Computes ragged silu(g0) * g1 * coeffs using Pallas, where x is [g0, g1].

  Token t is read from row `x_offset + t` of x, so x can be a larger buffer
  (e.g. the cross-layer activation bank). Not differentiable; use
  `ragged_silu_mul_bwd` for the backward pass.

  Args:
    x: 2D input array of shape (x_rows, 2 * hidden_dim).
    x_offset: Row of x holding token 0. Must be a multiple of the sublane tiling
      of x's dtype (8 for 32-bit, 16 for 16-bit dtypes).
    coeffs: Per-token scaling coefficients of shape (max_num_tokens, 1).
    num_tokens: Number of valid tokens to process, at most `max_num_tokens` and
      `x_rows - x_offset`. Leftover tokens in the output buffer are left
      uninitialized/unmasked for performance.
    block_size_tokens: Tile size along the token dim (default 128).
    block_size_hidden: Tile size along the hidden dim (default 128).
    out_dtype: Optional output dtype (e.g. fp8). Defaults to `x.dtype`.
    out_scale: Optional `[1, 1]` f32 static quantization scale when `out_dtype`
      is fp8.

  Returns:
    Result array of shape (max_num_tokens, hidden_dim).
  """
  return _pallas_ragged_silu_mul(
      x,
      x_offset,
      coeffs,
      num_tokens,
      block_size_tokens,
      block_size_hidden,
      out_dtype=out_dtype,
      out_scale=out_scale,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_silu_mul_bwd(
    g: jt.Num[jax.Array, "max_num_tokens hidden_dim"],
    x: jt.Num[jax.Array, "x_rows double_hidden_dim"],
    x_offset: jt.Int[jax.Array, ""] | int,
    coeffs: jt.Num[jax.Array, "max_num_tokens 1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens: int = 128,
    block_size_hidden: int = 128,
    g_mask_max: jt.Float[jax.Array, ""] | float | None = None,
    with_absmax: bool = False,
) -> Any:
  """Backward pass of `ragged_silu_mul_fwd` for the given output cotangent.

  Args:
    g: Cotangent of the `ragged_silu_mul_fwd` output.
    x: 2D input array of shape (x_rows, 2 * hidden_dim).
    x_offset: Row of x holding token 0 (see `ragged_silu_mul_fwd`).
    coeffs: Per-token scaling coefficients of shape (max_num_tokens, 1).
    num_tokens: Number of valid tokens (see `ragged_silu_mul_fwd`). Leftover
      rows of both gradients are left uninitialized.
    block_size_tokens: Tile size along the token dim (default 128).
    block_size_hidden: Tile size along the hidden dim (default 128).
    g_mask_max: Optional saturation threshold for the straight-through
      estimator. When provided, `g` is zeroed where `|silu(g0) * g1 * c| >
      g_mask_max`.
    with_absmax: If True, also returns the `[1, 1]` f32 absmax over the first
      `num_tokens` rows of `dx` (`[dg0, dg1]`).

  Returns:
    Tuple of `(dx, dcoeffs)` or `(dx, dcoeffs, dx_absmax)` when `with_absmax`
    is True.
  """
  return _pallas_ragged_silu_mul_bwd(
      g,
      x,
      x_offset,
      coeffs,
      num_tokens,
      block_size_tokens,
      block_size_hidden,
      g_mask_max=g_mask_max,
      with_absmax=with_absmax,
  )


# --- Pallas Kernels for Ragged FP8 Quantization ---


def _ragged_absmax_kernel(
    x_ref: Any,
    *,
    num_tokens_ref: Any,
    amax_ref: Any,
) -> None:
  """Accumulates the absmax of the valid rows in `x_ref` into `amax_ref`."""
  x_val = x_ref[...]
  abs_x = jnp.abs(x_val.astype(jnp.float32))
  block_tokens = x_val.shape[0]
  rem = num_tokens_ref[0] - pl.program_id(0) * block_tokens
  abs_x = jnp.where(
      (rem >= block_tokens) | (jax.lax.broadcasted_iota(jnp.int32, x_val.shape, 0) < rem),
      abs_x,
      0.0,
  )
  if x_val.ndim == 2:
    block_max = jnp.max(abs_x.reshape(-1, 8, x_val.shape[1]), axis=0)
  else:
    block_max = jnp.max(abs_x, axis=0)
  amax_ref[...] = jnp.maximum(amax_ref[...], block_max)


def _ragged_quantize_kernel(
    x_ref: Any,
    out_ref: Any,
    *,
    num_tokens_ref: Any,
    scale_ref: Any,
    fmax: float,
) -> None:
  """Quantizes valid rows of `x_ref` into `out_ref` using `scale_ref`."""
  x_val = x_ref[...]
  inv_scale = 1.0 / scale_ref[...].astype(jnp.float32)
  if x_val.ndim == 3:
    inv_scale = inv_scale[None]
  q = jnp.clip(x_val.astype(jnp.float32) * inv_scale, -fmax, fmax).astype(out_ref.dtype)
  block_tokens = x_val.shape[0]
  rem = num_tokens_ref[0] - pl.program_id(0) * block_tokens
  q = jnp.where(
      (rem >= block_tokens) | (jax.lax.broadcasted_iota(jnp.int32, x_val.shape, 0) < rem),
      q,
      0,
  )
  out_ref[...] = q


def _ragged_quantize_pipeline(
    num_tokens_ref: Any,
    x_ref: Any,
    scale_ref: Any,
    out_ref: Any,
    *,
    block_size_tokens: int,
    fmax: float,
) -> None:
  """Runs `_ragged_quantize_kernel` over `[0, num_tokens)`."""
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    x_spec = pl.BlockSpec(
        (block_size_tokens, *x_ref.shape[1:]),
        lambda i: (i,) + (0,) * (x_ref.ndim - 1),
    )
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_quantize_kernel,
            num_tokens_ref=num_tokens_ref,
            scale_ref=scale_ref,
            fmax=fmax,
        ),
        grid=(pl.cdiv(num_tokens, block_size_tokens),),
        in_specs=[x_spec],
        out_specs=x_spec,
    )(x_ref, out_ref)


def _ragged_absmax_quantize_pipeline(
    num_tokens_ref: Any,
    x_ref: Any,
    out_ref: Any,
    scale_out_ref: Any,
    amax_ref: Any,
    *,
    block_size_tokens: int,
    fmax: float,
) -> None:
  """Computes `scale` and quantizes `[0, num_tokens)` of `x_ref`."""
  scale_out_ref[...] = jnp.ones((1, 1), jnp.float32)
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    amax_ref[...] = jnp.zeros_like(amax_ref)
    num_blocks = pl.cdiv(num_tokens, block_size_tokens)
    x_spec = pl.BlockSpec(
        (block_size_tokens, *x_ref.shape[1:]),
        lambda i: (i,) + (0,) * (x_ref.ndim - 1),
    )
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_absmax_kernel,
            num_tokens_ref=num_tokens_ref,
            amax_ref=amax_ref,
        ),
        grid=(num_blocks,),
        in_specs=[x_spec],
    )(x_ref)
    amax = jnp.max(amax_ref[...], keepdims=True)
    scale_out_ref[...] = jnp.where(amax > 0, amax / fmax, 1.0)
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_quantize_kernel,
            num_tokens_ref=num_tokens_ref,
            scale_ref=scale_out_ref,
            fmax=fmax,
        ),
        grid=(num_blocks,),
        in_specs=[x_spec],
        out_specs=x_spec,
    )(x_ref, out_ref)


def ragged_quantize(
    x: jax.Array,
    scale: jax.Array,
    qtype: Any,
    num_tokens: jax.Array | int,
    *,
    block_size_tokens: int | None = None,
) -> jax.Array:
  """Quantizes the first `num_tokens` rows of `x` to `qtype` with `scale`."""
  qtype = jnp.dtype(qtype)
  fmax = float(jnp.finfo(qtype).max)
  if block_size_tokens is None:
    default_block = 512 if x.ndim == 2 else 256
    block_size_tokens = math.gcd(default_block, x.shape[0])
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(
          _ragged_quantize_pipeline,
          block_size_tokens=block_size_tokens,
          fmax=fmax,
      ),
      out_shape=jax.ShapeDtypeStruct(x.shape, qtype, manual_axis_type=manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[hbm, pl.BlockSpec((1, 1), lambda *_: (0, 0))],
          out_specs=hbm,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 64 * 1024 * 1024),
      ),
      name="ragged_quantize",
  )(
      jnp.asarray(num_tokens, jnp.int32)[None],
      x,
      scale.astype(jnp.float32).reshape(1, 1),
  )


def ragged_absmax_quantize(
    x: jax.Array,
    qtype: Any,
    num_tokens: jax.Array | int,
    *,
    block_size_tokens: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Dynamically quantizes the first `num_tokens` rows of `x` to `qtype`."""
  qtype = jnp.dtype(qtype)
  fmax = float(jnp.finfo(qtype).max)
  if block_size_tokens is None:
    default_block = 512 if x.ndim == 2 else 256
    block_size_tokens = math.gcd(default_block, x.shape[0])
  amax_shape = (8, x.shape[1]) if x.ndim == 2 else x.shape[1:]
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(
          _ragged_absmax_quantize_pipeline,
          block_size_tokens=block_size_tokens,
          fmax=fmax,
      ),
      out_shape=(
          jax.ShapeDtypeStruct(x.shape, qtype, manual_axis_type=manual_axis_type),
          jax.ShapeDtypeStruct((1, 1), jnp.float32, manual_axis_type=manual_axis_type),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[hbm],
          out_specs=[hbm, pl.BlockSpec((1, 1), lambda *_: (0, 0))],
          scratch_shapes=[pltpu.VMEM(amax_shape, jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * 64 * 1024 * 1024),
      ),
      name="ragged_absmax_quantize",
  )(
      jnp.asarray(num_tokens, jnp.int32)[None],
      x,
  )


# --- Pallas Kernels for Ragged Flatten / Unflatten ---


def _ragged_flatten_kernel(x_ref: Any, out_ref: Any) -> None:
  """Kernel body for flattening [X, D0, D1] -> [X, D]."""
  x_val = x_ref[...]
  out_ref[...] = jnp.reshape(x_val, (x_val.shape[0], -1))


def _pallas_ragged_flatten(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  """Pallas call wrapper for flattening [X, D0, D1] -> [X, D]."""
  max_num_tokens, d0, d1 = x.shape
  d = d0 * d1
  grid = ((num_tokens + block_size_tokens - 1) // block_size_tokens,)
  in_specs = [
      pl.BlockSpec(
          block_shape=(block_size_tokens, d0, d1),
          index_map=lambda i: (i, 0, 0),
      ),
  ]
  out_spec = pl.BlockSpec(
      block_shape=(block_size_tokens, d),
      index_map=lambda i: (i, 0),
  )
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_flatten_kernel,
      out_shape=jax.ShapeDtypeStruct(
          (max_num_tokens, d),
          x.dtype,
          manual_axis_type=manual_axis_type,
      ),
      in_specs=in_specs,
      out_specs=out_spec,
      grid=grid,
  )(x)


def _ragged_unflatten_kernel(x_ref: Any, out_ref: Any) -> None:
  """Kernel body for unflattening [X, D] -> [X, D0, D1]."""
  x_val = x_ref[...]
  out_ref[...] = jnp.reshape(x_val, out_ref.shape)


def _pallas_ragged_unflatten(
    x: jt.Num[jax.Array, "max_num_tokens d"],
    shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  """Pallas call wrapper for unflattening [X, D] -> [X, D0, D1]."""
  max_num_tokens, d = x.shape
  d0, d1 = shape
  assert d == d0 * d1, f"Shape mismatch: {d} != {d0} * {d1}"
  grid = ((num_tokens + block_size_tokens - 1) // block_size_tokens,)
  in_specs = [
      pl.BlockSpec(
          block_shape=(block_size_tokens, d),
          index_map=lambda i: (i, 0),
      ),
  ]
  out_spec = pl.BlockSpec(
      block_shape=(block_size_tokens, d0, d1),
      index_map=lambda i: (i, 0, 0),
  )
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_unflatten_kernel,
      out_shape=jax.ShapeDtypeStruct(
          (max_num_tokens, d0, d1),
          x.dtype,
          manual_axis_type=manual_axis_type,
      ),
      in_specs=in_specs,
      out_specs=out_spec,
      grid=grid,
  )(x)


# --- Custom VJPs (Dual of each other) ---


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _ragged_flatten_vjp(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  return _ragged_flatten_fwd(x, num_tokens, block_size_tokens)[0]


def _ragged_flatten_fwd(x, num_tokens, block_size_tokens):
  out = _pallas_ragged_flatten(x, num_tokens, block_size_tokens)
  return out, (num_tokens, x.shape[1:])


def _ragged_flatten_bwd(block_size_tokens, res, g):
  num_tokens, shape = res
  # Dual of Flatten backward pass is Unflatten forward pass!
  dx = _pallas_ragged_unflatten(g, shape, num_tokens, block_size_tokens)
  return dx, None


_ragged_flatten_vjp.defvjp(_ragged_flatten_fwd, _ragged_flatten_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 3))
def _ragged_unflatten_vjp(
    x: jt.Num[jax.Array, "max_num_tokens d"],
    shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  return _ragged_unflatten_fwd(x, shape, num_tokens, block_size_tokens)[0]


def _ragged_unflatten_fwd(x, shape, num_tokens, block_size_tokens):
  out = _pallas_ragged_unflatten(x, shape, num_tokens, block_size_tokens)
  return out, (num_tokens,)


def _ragged_unflatten_bwd(shape, block_size_tokens, res, g):
  del shape
  (num_tokens,) = res
  # Dual of Unflatten backward pass is Flatten forward pass!
  dx = _pallas_ragged_flatten(g, num_tokens, block_size_tokens)
  return dx, None


_ragged_unflatten_vjp.defvjp(_ragged_unflatten_fwd, _ragged_unflatten_bwd)


# --- Public APIs ---


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_flatten(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  """Flattens ragged tensor from [X, D0, D1] to [X, D] for the first num_tokens using Pallas.

  The dual backward pass (custom VJP) uses ragged_unflatten.

  Args:
    x: 3D input array of shape (max_num_tokens, d0, d1).
    num_tokens: Number of valid tokens to process. Leftover tokens in the output
      buffer are left uninitialized/unmasked for performance.
    block_size_tokens: Tile size along token dim (default 128).

  Returns:
    Reshaped 2D array of shape (max_num_tokens, d0 * d1).
  """
  return _ragged_flatten_vjp(x, num_tokens, block_size_tokens)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_unflatten(
    x: jt.Num[jax.Array, "max_num_tokens d"],
    shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  """Unflattens ragged tensor from [X, D] to [X, D0, D1] for the first num_tokens using Pallas.

  The dual backward pass (custom VJP) uses ragged_flatten.

  Args:
    x: 2D input array of shape (max_num_tokens, D).
    shape: Target (D0, D1) shape such that D0 * D1 == D.
    num_tokens: Number of valid tokens to process. Leftover tokens in the output
      buffer are left uninitialized/unmasked for performance.
    block_size_tokens: Tile size along token dim (default 128).

  Returns:
    Reshaped 3D array of shape (max_num_tokens, D0, D1).
  """
  return _ragged_unflatten_vjp(x, shape, num_tokens, block_size_tokens)


# --- Pallas Kernels for Ragged Write ---


def _ragged_write_kernel(
    dst_off_ref: Any,
    src_off_ref: Any,
    slice_size_ref: Any,
    dst_in_ref: Any,
    src0_ref: Any,
    src1_ref: Any,
    dst_out_ref: Any,
    *,
    block_size: int,
    max_src_blocks: int,
) -> None:
  """Kernel body for 2D ragged_write along axis 0."""
  b_size = jnp.int32(block_size)
  two_b_size = jnp.int32(2 * block_size)
  zero = jnp.int32(0)
  max_src_idx = jnp.int32(max_src_blocks - 1)

  block_idx = jnp.int32(pl.program_id(0))
  d_off = jnp.int32(dst_off_ref[()])
  s_off = jnp.int32(src_off_ref[()])
  s_size = jnp.int32(slice_size_ref[()])

  dst_block = (d_off // b_size) + block_idx
  dst_block_start = dst_block * b_size

  first_valid_dst_row = jnp.maximum(dst_block_start, d_off)
  first_needed_src_row = s_off + first_valid_dst_row - d_off
  src_block_0 = jnp.clip(first_needed_src_row // b_size, zero, max_src_idx).astype(jnp.int32)
  src_block_0_start = src_block_0 * b_size

  src_concat = jnp.concatenate([src0_ref[...], src1_ref[...]], axis=0)
  shift = s_off + dst_block_start - d_off - src_block_0_start
  roll_shift = ((two_b_size - shift) % two_b_size).astype(jnp.int32)
  rolled_src = pltpu.roll(src_concat, roll_shift, axis=0)[:block_size, :]

  global_dst_rows = dst_block_start + jax.lax.broadcasted_iota(jnp.int32, dst_in_ref.shape, dimension=0)
  write_mask = (global_dst_rows >= d_off) & (global_dst_rows < d_off + s_size)
  dst_out_ref[...] = jnp.where(write_mask, rolled_src, dst_in_ref[...])


def _pallas_ragged_write(
    dst: jt.Num[jax.Array, "dst_tokens embedding_dim"],
    src: jt.Num[jax.Array, "src_tokens embedding_dim"],
    dst_offset: jt.Num[jax.Array, ""] | int,
    src_offset: jt.Num[jax.Array, ""] | int,
    slice_size: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "dst_tokens embedding_dim"]:
  """Pallas call wrapper for 2D ragged_write along axis 0."""
  dst_tokens, embedding_dim = dst.shape
  src_tokens, _ = src.shape
  block_size = max(8, ((block_size_tokens + 7) // 8) * 8)

  max_dst_blocks = (dst_tokens + block_size - 1) // block_size
  max_src_blocks = (src_tokens + block_size - 1) // block_size

  d_off_i32 = jnp.int32(dst_offset)
  s_size_i32 = jnp.int32(slice_size)
  first_dst_block = d_off_i32 // jnp.int32(block_size)
  last_dst_block = (d_off_i32 + s_size_i32 - jnp.int32(1)) // jnp.int32(block_size)
  num_dst_blocks = jnp.where(
      s_size_i32 > jnp.int32(0),
      last_dst_block - first_dst_block + jnp.int32(1),
      jnp.int32(0),
  ).astype(jnp.int32)

  def dst_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref):
    del src_off_ref, slice_size_ref
    b_size = jnp.int32(block_size)
    dst_block = jnp.clip(
        (jnp.int32(dst_off_ref[()]) // b_size) + jnp.int32(block_idx),
        jnp.int32(0),
        jnp.int32(max_dst_blocks - 1),
    )
    return (dst_block, jnp.int32(0))

  def src0_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref):
    del slice_size_ref
    b_size = jnp.int32(block_size)
    d_off = jnp.int32(dst_off_ref[()])
    s_off = jnp.int32(src_off_ref[()])
    dst_block_start = ((d_off // b_size) + jnp.int32(block_idx)) * b_size
    first_valid_dst_row = jnp.maximum(dst_block_start, d_off)
    first_needed_src_row = s_off + first_valid_dst_row - d_off
    src_block_0 = jnp.clip(
        first_needed_src_row // b_size,
        jnp.int32(0),
        jnp.int32(max_src_blocks - 1),
    )
    return (src_block_0, jnp.int32(0))

  def src1_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref):
    src_block_0, _ = src0_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref)
    src_block_1 = jnp.clip(
        src_block_0 + jnp.int32(1),
        jnp.int32(0),
        jnp.int32(max_src_blocks - 1),
    )
    return (src_block_1, jnp.int32(0))

  block_shape = (block_size, embedding_dim)
  manual_axis_type = getattr(jax.typeof(dst), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(
          _ragged_write_kernel,
          block_size=block_size,
          max_src_blocks=max_src_blocks,
      ),
      out_shape=jax.ShapeDtypeStruct(
          dst.shape,
          dst.dtype,
          manual_axis_type=manual_axis_type,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              pl.BlockSpec(block_shape, dst_index_map),
              pl.BlockSpec(block_shape, src0_index_map),
              pl.BlockSpec(block_shape, src1_index_map),
          ],
          out_specs=pl.BlockSpec(block_shape, dst_index_map),
          grid=(num_dst_blocks,),
      ),
      input_output_aliases={3: 0},
  )(dst_offset, src_offset, slice_size, dst, src, src)


@functools.partial(
    jax.jit,
    donate_argnums=(0,),
    static_argnames=("block_size_tokens",),
)
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_write(
    dst: jt.Num[jax.Array, "dst_tokens embedding_dim"],
    src: jt.Num[jax.Array, "src_tokens embedding_dim"],
    dst_offset: jt.Num[jax.Array, ""] | int,
    src_offset: jt.Num[jax.Array, ""] | int,
    slice_size: jt.Num[jax.Array, ""] | int,
    *,
    block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "dst_tokens embedding_dim"]:
  """Updates a dynamic slice along axis 0 of a 2D array from a 2D source array.

  Conceptually, this operation is equivalent to:
      ```python
      dst_out = jnp.copy(dst)
      dst_out[dst_offset : dst_offset + slice_size, :] = src[
          src_offset : src_offset + slice_size, :
      ]
      return dst_out
      ```

  This function safely handles dynamic variables (JAX Tracers) for `dst_offset`,
  `src_offset`, and `slice_size` within JIT-compiled functions.
  It leverages dynamic grid scheduling on TPU and supports slice boundaries
  not aligned to `block_size_tokens`. If the requested write operation
  is out-of-bounds or `slice_size <= 0`, it gracefully acts as a no-op and
  returns `dst` unmodified.

  Args:
      dst: 2D destination array of shape (dst_tokens, embedding_dim).
      src: 2D source array of shape (src_tokens, embedding_dim).
      dst_offset: Integer scalar identifying the starting index along axis 0 in
        `dst`.
      src_offset: Integer scalar identifying the starting index along axis 0 in
        `src`.
      slice_size: Integer scalar indicating the number of rows to copy along
        axis 0.
      block_size_tokens: Tile size along token axis (axis 0). Defaults to 128.

  Returns:
      A 2D array with identical shape and dtype to `dst` containing the ragged
      write update.
  """
  if dst.ndim != 2 or src.ndim != 2:
    raise ValueError(f"dst and src must be 2D arrays, got ranks {dst.ndim} and {src.ndim}")
  if dst.dtype != src.dtype:
    raise ValueError(f"dst and src must have the same dtype, got {dst.dtype} and {src.dtype}")
  if dst.shape[1] != src.shape[1]:
    raise ValueError(f"dst and src embedding dimensions must match, got {dst.shape[1]} and" f" {src.shape[1]}")
  if block_size_tokens <= 0:
    raise ValueError(f"block_size_tokens must be positive, got {block_size_tokens}")
  if dst.shape[0] == 0 or src.shape[0] == 0 or dst.shape[1] == 0:
    return dst
  dst_limit = dst.shape[0]
  src_limit = src.shape[0]

  dst_offset = jnp.asarray(dst_offset)
  src_offset = jnp.asarray(src_offset)
  slice_size = jnp.asarray(slice_size)

  # 1. OOB Checking (No-op if any element is OOB or slice is invalid)
  is_oob = (
      (slice_size <= 0)
      | (slice_size > dst_limit)
      | (slice_size > src_limit)
      | (dst_offset < 0)
      | (dst_offset > dst_limit - jnp.clip(slice_size, 0, dst_limit))
      | (src_offset < 0)
      | (src_offset > src_limit - jnp.clip(slice_size, 0, src_limit))
  )

  zero = jnp.int32(0)
  safe_slice_size = jnp.where(is_oob, zero, jnp.asarray(slice_size, dtype=jnp.int32))
  safe_dst_offset = jnp.where(is_oob, zero, jnp.asarray(dst_offset, dtype=jnp.int32))
  safe_src_offset = jnp.where(is_oob, zero, jnp.asarray(src_offset, dtype=jnp.int32))

  return _pallas_ragged_write(
      dst,
      src,
      safe_dst_offset,
      safe_src_offset,
      safe_slice_size,
      block_size_tokens,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def split_microbatches(
    x: AnyJaxArrayOrPyTree,
    *,
    mesh: jax.sharding.Mesh,
) -> tuple[AnyJaxArrayOrPyTree, AnyJaxArrayOrPyTree]:
  """Splits the batch into two micro-batches locally in a shard map.

  Assumes the first dimension is the batch dimension and splits along it.
  Does not split any other dimensions, including the sequence dimension,
  which should already be split when pdbs < 1.

  Args:
    x: Input array or PyTree of arrays.
    mesh: JAX mesh over which the data and model are sharded.

  Returns:
    A tuple of two PyTrees with the same structure as the input, each containing
    half of the data along the first dimension.
  """
  if x is None:
    return None, None

  def _split_leaf(arr):
    s = jax.typeof(arr).sharding
    arr_pspec = getattr(s, "spec", jax.sharding.PartitionSpec())

    def _local_split(local_arr):
      return tuple(jnp.split(local_arr, 2, axis=0))

    return jax.shard_map(
        _local_split,
        mesh=mesh,
        in_specs=arr_pspec,
        out_specs=(arr_pspec, arr_pspec),
    )(arr)

  leaves, treedef = jax.tree.flatten(x)
  splits = [_split_leaf(leaf) for leaf in leaves]
  mb0 = jax.tree.unflatten(treedef, [s[0] for s in splits])
  mb1 = jax.tree.unflatten(treedef, [s[1] for s in splits])
  return mb0, mb1


@jt.jaxtyped(typechecker=typeguard.typechecked)
def merge_microbatches(
    mb0: AnyJaxArrayOrPyTree,
    mb1: AnyJaxArrayOrPyTree,
    *,
    mesh: jax.sharding.Mesh,
) -> AnyJaxArrayOrPyTree:
  """Merges two micro-batches back into a single batch in a shard map."""
  if mb0 is None:
    return None

  def _merge_leaf(arr0, arr1):
    s0 = jax.typeof(arr0).sharding
    s1 = jax.typeof(arr1).sharding
    spec0 = getattr(s0, "spec", jax.sharding.PartitionSpec())
    spec1 = getattr(s1, "spec", jax.sharding.PartitionSpec())
    arr_pspec = spec0 if spec0 is not None else spec1

    def _local_merge(local_arr0, local_arr1):
      return jnp.concatenate([local_arr0, local_arr1], axis=0)

    return jax.shard_map(
        _local_merge,
        mesh=mesh,
        in_specs=(arr_pspec, arr_pspec),
        out_specs=arr_pspec,
    )(arr0, arr1)

  return jax.tree.map(_merge_leaf, mb0, mb1)


# --- Pallas Kernels for Ragged Gather on TensorCore ---

_RAGGED_GATHER_TC_NUM_BUFFERS = 2


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_gather_tc(
    x: jt.Num[jax.Array, "num_input_tokens d0 d1"],
    indices: jt.Int[jax.Array, "num_indices"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""] | int,
    max_out_tokens: int,
    block_size: int = 2048,
    *,
    with_absmax: bool = False,
) -> Any:
  """Executes a ragged gather on TensorCore using a Pallas kernel.

  Gathers `out[i] = x[indices[start + i]]` for `i < num_tokens`. Output rows
  `[num_tokens, max_out_tokens)` are left untouched (garbage from the
  uninitialized output buffer).

  The caller must guarantee `0 <= num_tokens <= max_out_tokens` and
  `0 <= start` with `start + num_tokens <= num_indices`; these are not checked
  at runtime and violating them is undefined behavior.

  Args:
    x: Input array of shape (num_input_tokens, d0, d1).
    indices: Indices into `x` of shape (num_indices,).
    start: Offset into `indices` of the first index to gather. May be a traced
      scalar; the kernel does not recompile per value.
    num_tokens: Number of indices to gather. May be a traced scalar; the kernel
      does not recompile per value.
    max_out_tokens: Length of the output buffer along axis 0.
    block_size: Number of output rows gathered per grid step, i.e. the number of
      concurrent row DMAs in flight per buffer. Clamped to `max_out_tokens`.
    with_absmax: If True, also returns the `[1, 1]` f32 absmax over the first
      `num_tokens` gathered rows.

  Returns:
    Gathered array in an output buffer of shape (max_out_tokens, d0, d1), or
    `(out, absmax)` when `with_absmax` is True.
  """
  if max_out_tokens <= 0:
    raise ValueError(f"max_out_tokens must be positive, got {max_out_tokens}.")
  if isinstance(num_tokens, int) and not 0 <= num_tokens <= max_out_tokens:
    raise ValueError(f"num_tokens must be in [0, {max_out_tokens}], got {num_tokens}.")
  if block_size <= 0:
    raise ValueError(f"block_size must be positive, got {block_size}.")
  num_indices = indices.shape[0]
  _, dim0, dim1 = x.shape
  if num_indices == 0 or (isinstance(num_tokens, int) and num_tokens == 0):
    empty = jnp.empty((max_out_tokens, dim0, dim1), dtype=x.dtype)
    if with_absmax:
      return empty, jnp.zeros((1, 1), dtype=jnp.float32)
    return empty

  block_size = min(block_size, max_out_tokens)
  num_buffers = _RAGGED_GATHER_TC_NUM_BUFFERS
  num_full_blocks = max_out_tokens // block_size
  rem = max_out_tokens % block_size
  num_blocks = num_full_blocks + (1 if rem > 0 else 0)

  # `indices` may exceed SMEM capacity, so each block stages only its window of
  # indices in SMEM. The window starts at an offset aligned to the HBM tile
  # (128 x i32) so the dynamic DMA slice needs no retiling, and has room for the
  # misalignment. It is clamped to `max_win_start` to stay in bounds, which
  # requires the (padded) indices length to be aligned and at least one window
  # long.
  idx_align = 128
  idx_window = pl.cdiv(block_size, idx_align) * idx_align + idx_align
  idx_pad = max((-num_indices) % idx_align, idx_window - num_indices)
  if idx_pad > 0:
    indices = jnp.pad(indices, (0, idx_pad))
  max_win_start = indices.shape[0] - idx_window

  def _ragged_gather_kernel(*args):
    if with_absmax:
      (
          scalars_smem_ref,
          idx_hbm_ref,
          x_hbm_ref,
          o_hbm_ref,
          amax_smem_ref,
          vmem_ref,
          idx_smem_ref,
          idx_sem,
          data_recv_sem,
          data_send_sem,
          amax_vmem_ref,
      ) = args
    else:
      (
          scalars_smem_ref,
          idx_hbm_ref,
          x_hbm_ref,
          o_hbm_ref,
          vmem_ref,
          idx_smem_ref,
          idx_sem,
          data_recv_sem,
          data_send_sem,
      ) = args
      amax_smem_ref = amax_vmem_ref = None

    block_idx = pl.program_id(0)
    buf_idx = block_idx % num_buffers
    start_idx = scalars_smem_ref[0]
    num_valid = scalars_smem_ref[1]

    if with_absmax:
      assert amax_vmem_ref is not None

      @pl.when(block_idx == 0)
      def _init_amax():
        amax_vmem_ref[...] = jnp.zeros_like(amax_vmem_ref)

    def _run_step(step_idx, b_idx, action: str):
      base = step_idx * block_size
      row0 = start_idx + base
      win_start = jnp.minimum((row0 // idx_align) * idx_align, max_win_start)
      idx_off = row0 - win_start

      def _fetch_indices():
        # Synchronous; the row gathers read the indices as scalars when issued,
        # so a single SMEM buffer suffices.
        copy = pltpu.make_async_copy(
            idx_hbm_ref.at[pl.ds(pl.multiple_of(win_start, idx_align), idx_window)],
            idx_smem_ref,
            idx_sem,
        )
        copy.start()
        copy.wait()

      def _chunk(off, size: int, chunk_action: str):
        """Issues `chunk_action` for block rows [off, off + size)."""
        if chunk_action == "data_start":
          step_unroll = math.gcd(size, 128)

          @pl.loop(0, size // step_unroll)
          def _(chunk_i):
            base_i = off + chunk_i * step_unroll
            src_indices = [idx_smem_ref[idx_off + base_i + u] for u in range(step_unroll)]
            for u in range(step_unroll):
              pltpu.make_async_copy(
                  x_hbm_ref.at[pl.ds(src_indices[u], 1), :, :],
                  vmem_ref.at[b_idx, pl.ds(base_i + u, 1), :, :],
                  data_recv_sem.at[b_idx],
              ).start()

        elif chunk_action == "recv_wait":
          # Only the byte count matters for the wait; `o_hbm_ref` always has
          # at least `block_size` rows, whereas `x` may have fewer.
          pltpu.make_async_copy(
              o_hbm_ref.at[pl.ds(0, size), :, :],
              vmem_ref.at[b_idx, pl.ds(0, size), :, :],
              data_recv_sem.at[b_idx],
          ).wait()
        elif chunk_action == "absmax":
          assert amax_vmem_ref is not None
          step_unroll = math.gcd(size, 16)

          @pl.loop(0, size // step_unroll)
          def _(chunk_i):
            base_i = off + chunk_i * step_unroll
            tile = vmem_ref[b_idx, pl.ds(base_i, step_unroll), :, :]
            row_max = jnp.max(jnp.abs(tile).astype(jnp.float32), axis=(0, 1))
            amax_vmem_ref[0] = jnp.maximum(amax_vmem_ref[0], row_max)

        else:
          copy = pltpu.make_async_copy(
              vmem_ref.at[b_idx, pl.ds(off, size), :, :],
              o_hbm_ref.at[pl.ds(base + off, size), :, :],
              data_send_sem.at[b_idx],
          )
          if chunk_action == "send_start":
            copy.start()
          else:
            copy.wait()

      def _do_len(length: int):
        step_valid = jnp.minimum(jnp.maximum(num_valid - base, 0), length)

        def _over_valid_rows(chunk_action: str):
          # Fast path: the whole block is valid, so use one static-size chunk.
          @pl.when(step_valid == length)
          def _():
            _chunk(0, length, chunk_action)

          # Partially valid block: DMA sizes must be static, so split the
          # dynamic row count into power-of-two chunks, one per set bit.
          @pl.when(step_valid < length)
          def _():
            off = 0
            for k in reversed(range(length.bit_length())):
              size = 1 << k
              bit = step_valid & size

              @pl.when(bit != 0)
              def _(off=off, size=size):
                _chunk(off, size, chunk_action)

              off = off + bit

        if action == "data_start":

          @pl.when(step_valid > 0)
          def _():
            _fetch_indices()

          _over_valid_rows("data_start")
        elif action == "copy_out_start":
          # All recv waits must finish before any send starts, because the
          # per-row gather DMAs share one semaphore and complete out of order.
          _over_valid_rows("recv_wait")
          if with_absmax:
            _over_valid_rows("absmax")
          _over_valid_rows("send_start")
        elif action == "copy_out_wait":
          _over_valid_rows("send_wait")

      if rem == 0:
        _do_len(block_size)
      elif isinstance(step_idx, int):
        _do_len(rem if step_idx == num_blocks - 1 else block_size)
      else:
        jax.lax.cond(
            step_idx == num_blocks - 1,
            lambda: _do_len(rem),
            lambda: _do_len(block_size),
        )

    @pl.when(block_idx == 0)
    def _prologue():
      _run_step(0, 0, "data_start")

    _run_step(block_idx, buf_idx, "copy_out_start")

    # The grid covers only blocks overlapping `[0, num_tokens)`.
    num_steps = pl.num_programs(0)
    next_data_block = block_idx + 1

    @pl.when(next_data_block < num_steps)
    def _queue_next():
      next_data_buf = next_data_block % num_buffers

      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        _run_step(block_idx - 1, next_data_buf, "copy_out_wait")

      _run_step(next_data_block, next_data_buf, "data_start")

    @pl.when(block_idx == num_steps - 1)
    def _epilogue():
      @pl.when(block_idx >= 1)
      def _wait_prev_out():
        prev_block = block_idx - 1
        _run_step(prev_block, prev_block % num_buffers, "copy_out_wait")

      _run_step(block_idx, buf_idx, "copy_out_wait")
      if with_absmax:
        assert amax_smem_ref is not None and amax_vmem_ref is not None
        amax_smem_ref[0, 0] = jnp.max(amax_vmem_ref[0])

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_struct = jax.ShapeDtypeStruct(
      (max_out_tokens, dim0, dim1),
      x.dtype,
      manual_axis_type=manual_axis_type,
  )
  scalars = jnp.stack(
      [
          jnp.asarray(start, dtype=jnp.int32),
          jnp.asarray(num_tokens, dtype=jnp.int32),
      ]
  )
  # Only launch grid steps for blocks overlapping `[0, num_tokens)`. When
  # `num_tokens` is traced, this is a dynamic grid size.
  num_steps = pl.cdiv(num_tokens if isinstance(num_tokens, int) else scalars[1], block_size)
  scratch_shapes = [
      pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
      pltpu.SMEM((idx_window,), indices.dtype),
      pltpu.SemaphoreType.DMA(()),
      pltpu.SemaphoreType.DMA((num_buffers,)),
      pltpu.SemaphoreType.DMA((num_buffers,)),
  ]
  if with_absmax:
    num_steps = jnp.maximum(1, num_steps)
    amax_struct = jax.ShapeDtypeStruct(
        (1, 1),
        jnp.float32,
        manual_axis_type=manual_axis_type,
    )
    out_shape = (out_struct, amax_struct)
    out_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.SMEM),
    )
    scratch_shapes.append(pltpu.VMEM((1, dim1), jnp.float32))
    call_name = "ragged_gather_tc-absmax"
  else:
    out_shape = out_struct
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
    call_name = "ragged_gather_tc"

  return pl.pallas_call(
      _ragged_gather_kernel,
      out_shape=out_shape,
      grid=(num_steps,),
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      out_specs=out_specs,
      scratch_shapes=scratch_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(pltpu.ARBITRARY,),
      ),
      name=call_name,
  )(scalars, indices, x)


# --- Ragged Gather Reduce on TensorCore ---


# While block i is reduced, the row DMAs of block i + 2 are issued in the same
# loop and block i + 1's are in flight, and block i - 1 is being stored.
_RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS = 4
_RAGGED_GATHER_REDUCE_TC_LOOKAHEAD = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS - 2
# Static bodies per trip of the row DMA issue and reduce loops.
_RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL = 64
_RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL = 16
# Alignment of dynamic 1D int32 HBM slices.
_RAGGED_GATHER_REDUCE_TC_ALIGN = 128
# Slots per chunk of the two-level search for block boundaries.
_RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK = 256
# Block table entries per block: first slot, slot count, first token, token
# count.
_RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS = 4
# VMEM of the block buffers, under the kernel's VMEM limit.
_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT = 64 * 1024 * 1024
_RAGGED_GATHER_REDUCE_TC_VMEM_BUDGET = 56 * 1024 * 1024


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RaggedGatherReduceMetadata:
  """Routing of `ragged_gather_reduce_tc`.

  Built by `ragged_gather_reduce_tc_metadata`. The valid slots of `indices` are
  compacted so that each output token's valid rows are contiguous and in buffer
  order, and output tokens are split into blocks of consecutive tokens whose
  valid rows plus output rows fit in `block_size` VMEM rows.

  Attributes:
    src_rows: Buffer row of each compacted slot, padded to an aligned length of
      at least one SMEM slot window.
    tokens: Output token of each compacted slot, padded like `src_rows`.
    blocks: Number of blocks, followed by the first slot, slot count, first
      token and token count of each block in structure-of-arrays order.
    block_size: Maximum VMEM rows per block (valid input rows plus output
      tokens), which also bounds the row DMAs in flight per block.
  """

  src_rows: jax.Array
  tokens: jax.Array
  blocks: jax.Array
  block_size: int = jax.tree.static()


def _ragged_gather_reduce_tc_layout(num_indices: int, top_k: int, block_size: int) -> tuple[int, int, int]:
  """Returns (max_blocks, window, slots_len) of the metadata layout.

  A token costs its valid rows plus its output row. Block `b` takes the tokens
  whose exclusive cost prefix is in `[b * span, (b + 1) * span)`, where
  `span = block_size - top_k`, so it costs at most `block_size` VMEM rows. Each
  block stages a `window` of the compacted slots in SMEM, starting at an aligned
  offset, so the slot arrays are padded to `slots_len`.

  Args:
    num_indices: Length of the forward gather's indices.
    top_k: Number of occurrences of each token id in the indices.
    block_size: Maximum VMEM rows per block.
  """
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  max_blocks = pl.cdiv(num_indices + num_indices // top_k, block_size - top_k)
  window = pl.cdiv(block_size, align) * align + align
  slots_len = pl.cdiv(max(num_indices, window), align) * align
  return max_blocks, window, slots_len


def _check_ragged_gather_reduce_tc_args(top_k: int, block_size: int):
  if top_k <= 0:
    raise ValueError(f"top_k must be positive, got {top_k}.")
  if block_size <= top_k:
    raise ValueError(
        f"block_size must exceed top_k ({top_k}) so that a block fits a token" f" and its rows, got {block_size}."
    )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_gather_reduce_tc_metadata(
    indices: jt.Int[jax.Array, "num_indices"],
    start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""] | int,
    *,
    top_k: int,
    block_size: int = 896,
    max_num_tokens: int | None = None,
) -> RaggedGatherReduceMetadata:
  """Derives the routing consumed by `ragged_gather_reduce_tc`.

  The result depends only on the routing, not on the token data, so it can be
  computed once and shared by every `ragged_gather_reduce_tc` call with the
  same `indices`, `start` and `num_tokens`.

  The valid slots, those in the `[start, start + num_tokens)` window, are
  compacted by a stable sort on their token, so a token's valid rows are
  contiguous and in buffer order. Output tokens are then split into blocks of
  consecutive tokens that each fit `block_size` VMEM rows of input rows and
  output tokens, so sparse windows get blocks of many tokens and dense windows
  blocks of many rows.

  Each token id must appear exactly `top_k` times in `indices` (e.g.
  `indices = sort_indices // top_k`). The caller must guarantee `0 <= start`,
  `0 <= num_tokens` and `start + num_tokens <= num_indices`; these are not
  checked at runtime and violating them is undefined behavior.

  Args:
    indices: Indices used by the forward ragged gather, of shape (num_out_tokens
      * top_k,), with values in [0, num_out_tokens).
    start: Offset into `indices` of the index that formed the first buffer row.
      May be a traced scalar.
    num_tokens: Number of valid rows in the buffer. May be a traced scalar.
    top_k: Number of occurrences of each token id in `indices`.
    block_size: Maximum VMEM rows (valid input rows plus output tokens) per
      block of the kernel. Must exceed `top_k`, and `ragged_gather_reduce_tc`
      requires its block buffers to fit in VMEM for the token shape.
    max_num_tokens: Optional static bound on `num_tokens`. If set, only a window
      of `max_num_tokens` slots is sorted, which gives the same result faster.
      The caller must guarantee `num_tokens <= max_num_tokens`; this is not
      checked at runtime.

  Returns:
    The compacted slots and block table.
  """
  _check_ragged_gather_reduce_tc_args(top_k, block_size)
  num_indices = indices.shape[0]
  if num_indices % top_k:
    raise ValueError(f"indices length ({num_indices}) must be divisible by top_k ({top_k}).")
  if isinstance(num_tokens, int) and num_tokens < 0:
    raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
  if isinstance(start, int) and start < 0:
    raise ValueError(f"start must be non-negative, got {start}.")
  if isinstance(start, int) and isinstance(num_tokens, int) and start + num_tokens > num_indices:
    raise ValueError(f"start + num_tokens ({start + num_tokens}) must not exceed indices" f" length ({num_indices}).")
  num_out_tokens = num_indices // top_k
  max_blocks, _, slots_len = _ragged_gather_reduce_tc_layout(num_indices, top_k, block_size)
  if num_out_tokens == 0:
    empty_slots = jnp.zeros((slots_len,), jnp.int32)
    return RaggedGatherReduceMetadata(
        src_rows=empty_slots,
        tokens=empty_slots,
        blocks=jnp.zeros((1,), jnp.int32),
        block_size=block_size,
    )

  start = jnp.asarray(start, dtype=jnp.int32)
  num_tokens = jnp.asarray(num_tokens, dtype=jnp.int32)
  positions = jnp.arange(num_indices, dtype=jnp.int32)
  # Only the `window` slots from `window_start`, which hold the valid slots,
  # are sorted.
  window = num_indices
  if max_num_tokens is not None:
    window = min(max(max_num_tokens, 1), num_indices)
  window_start = jnp.minimum(start, num_indices - window)
  window_positions = window_start + jnp.arange(window, dtype=jnp.int32)
  rel = window_positions - start
  valid = (rel >= 0) & (rel < num_tokens)
  window_indices = jax.lax.dynamic_slice_in_dim(indices, window_start, window)
  # Valid slots first, by token and then buffer order; invalid slots last.
  key = jnp.where(valid, window_indices, num_out_tokens).astype(jnp.int32)
  sorted_key, sorted_positions = jax.lax.sort((key, window_positions), num_keys=1, is_stable=True)
  if window < num_indices:
    # The invalid slots follow the valid ones in buffer order: those before
    # `start`, then those after the valid slots.
    invalid_positions = jnp.where(positions < start + num_tokens, positions - num_tokens, positions)
    sorted_positions = jnp.where(
        positions < num_tokens,
        jnp.pad(sorted_positions, (0, num_indices - window)),
        invalid_positions,
    )
    sorted_key = jnp.pad(sorted_key, (0, num_indices - window), constant_values=num_out_tokens)
  pad = slots_len - num_indices
  src_rows = jnp.pad(sorted_positions - start, (0, pad))
  tokens = jnp.pad(sorted_key, (0, pad))

  # A token costs its valid rows plus its output row, so the tokens before
  # slot i's token cost run_start[i] + sorted_key[i], where run_start[i] is
  # the first slot of that token. Since each token appears at most `top_k`
  # times, `i - run_start[i]` is the number of preceding slots in `[i - top_k +
  # 1, i)` with the same token, avoiding a sequential cummax over all slots.
  # Packing `run_offset = i - run_start[i]` into the low bits of `cost`
  # preserves `< threshold << offset_bits` and lets `prev_key` and `next_key`
  # be recovered directly from the chunk reductions without 1D gathers.
  padded_key = jnp.pad(sorted_key, (top_k - 1, 0), constant_values=-1)
  run_offset = jnp.zeros_like(sorted_key)
  for k in range(1, top_k):
    run_offset = run_offset + (padded_key[top_k - 1 - k : top_k - 1 - k + num_indices] == sorted_key).astype(jnp.int32)
  offset_bits = (top_k - 1).bit_length()
  offset_mask = (1 << offset_bits) - 1
  int_max = jnp.iinfo(jnp.int32).max
  cost = (positions - run_offset) + sorted_key
  packed = jnp.where(
      sorted_key < num_out_tokens,
      (cost << offset_bits) + run_offset,
      int_max,
  )

  # Two-level search for the first slot whose token reaches each block's cost
  # budget: count full chunks first, then scan the single straddling chunk.
  chunk = _RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK
  num_chunks = pl.cdiv(num_indices, chunk)
  packed_chunks = jnp.pad(packed, (0, num_chunks * chunk - num_indices), constant_values=int_max).reshape(
      num_chunks, chunk
  )
  end_packed = packed_chunks[:, -1]
  threshold = jnp.arange(max_blocks + 1, dtype=jnp.int32) * (block_size - top_k)
  thr_packed = threshold[:, None] << offset_bits
  end_below = end_packed < thr_packed
  full_chunks = jnp.sum(end_below, axis=1, dtype=jnp.int32)
  row = jnp.minimum(full_chunks, num_chunks - 1)
  sel_packed = packed_chunks[row]
  sel_below = sel_packed < thr_packed
  in_chunk = jnp.sum(sel_below, axis=1, dtype=jnp.int32)
  slot_start = row * chunk + in_chunk

  # The tokens between slot_start - 1's token and slot_start's token have no
  # valid slots and cost 1 each, so block b starts at the first of them whose
  # tokens before it reach the threshold, or else at slot_start's token.
  prev_packed = jnp.maximum(
      jnp.max(jnp.where(end_below, end_packed, -1), axis=1),
      jnp.max(jnp.where(sel_below, sel_packed, -1), axis=1),
  )
  prev_key = jnp.where(
      slot_start > 0,
      (prev_packed >> offset_bits) + (prev_packed & offset_mask) - (slot_start - 1),
      -1,
  )
  next_packed = jnp.min(jnp.where(sel_below, int_max, sel_packed), axis=1)
  next_key = jnp.where(
      next_packed < int_max,
      (next_packed >> offset_bits) - slot_start,
      num_out_tokens,
  )
  token_start = jnp.clip(threshold - slot_start, prev_key + 1, next_key)
  num_blocks = jnp.sum(token_start[:-1] < num_out_tokens, dtype=jnp.int32)
  blocks = jnp.concatenate(
      [
          num_blocks[None],
          slot_start[:-1],
          jnp.diff(slot_start),
          token_start[:-1],
          jnp.diff(token_start),
      ]
  )
  return RaggedGatherReduceMetadata(
      src_rows=src_rows,
      tokens=tokens,
      blocks=blocks.astype(jnp.int32),
      block_size=block_size,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_gather_reduce_tc(
    x: jt.Num[jax.Array, "max_input_tokens d0 d1"],
    metadata: RaggedGatherReduceMetadata,
    *,
    num_out_tokens: int,
    top_k: int,
    out: jt.Num[jax.Array, "num_out_tokens d0 d1"] | None = None,
    zero_initialized: bool | None = None,
) -> jt.Num[jax.Array, "num_out_tokens d0 d1"]:
  """Scatter-reduce transpose of `ragged_gather_tc`, reducing duplicates by sum.

  `x` is a token buffer formed by `ragged_gather_tc(src, indices, start,
  num_tokens, ...)`, i.e. `x[i] = src[indices[start + i]]` for
  `i < num_tokens`. Each valid row is scattered to its destination token and
  rows that land on the same token are summed in float32 in buffer order:
  `out[t] = init[t] + sum_{i < num_tokens, indices[start + i] == t} x[i]`.
  Rows of `x` at or past `num_tokens` are never read.

  When `zero_initialized` is True (the default when `out` is None), `out` is
  assumed to be all zeros, so each block's output token tile is zeroed in VMEM,
  valid rows are summed in VMEM, and the result is written directly to `out`
  without reading existing values from HBM. When `zero_initialized` is False
  (the default when `out` is provided), existing values in `out` are read from
  HBM into VMEM, accumulated with valid rows in VMEM in float32, and written
  back to `out`.

  The routing is given by `metadata`, from
  `ragged_gather_reduce_tc_metadata(indices, start, num_tokens, top_k=top_k,
  ...)`. The caller must also guarantee `num_tokens <= max_input_tokens`; this
  is not checked at runtime and violating it is undefined behavior.

  Args:
    x: Ragged token buffer of shape (max_input_tokens, d0, d1).
    metadata: Routing from `ragged_gather_reduce_tc_metadata`.
    num_out_tokens: Number of output tokens, `len(indices) // top_k`.
    top_k: Number of occurrences of each token id in `indices`.
    out: Optional destination buffer of shape (num_out_tokens, d0, d1) to
      scatter-reduce into. If None, a zero-initialized buffer is allocated.
    zero_initialized: Whether `out` is guaranteed to be all zeros. Defaults to
      True when `out` is None and False when `out` is provided.

  Returns:
    Reduced tokens of shape (num_out_tokens, d0, d1).

  Raises:
    ValueError: If the metadata was not derived for `num_out_tokens * top_k`
      indices and `top_k`, `out` has an incompatible shape or dtype, or the
      block buffers do not fit in VMEM.
  """
  max_input_tokens, dim0, dim1 = x.shape
  block_size = metadata.block_size
  _check_ragged_gather_reduce_tc_args(top_k, block_size)
  if num_out_tokens < 0:
    raise ValueError(f"num_out_tokens must be non-negative, got {num_out_tokens}.")
  if out is not None:
    if out.shape != (num_out_tokens, dim0, dim1):
      raise ValueError(f"out must have shape {(num_out_tokens, dim0, dim1)}, got" f" {out.shape}.")
    if out.dtype != x.dtype:
      raise ValueError(f"out must have dtype {x.dtype}, got {out.dtype}.")
  if zero_initialized is None:
    zero_initialized = out is None

  num_buffers = _RAGGED_GATHER_REDUCE_TC_NUM_BUFFERS
  align = _RAGGED_GATHER_REDUCE_TC_ALIGN
  fields = _RAGGED_GATHER_REDUCE_TC_BLOCK_FIELDS
  max_blocks, window, slots_len = _ragged_gather_reduce_tc_layout(num_out_tokens * top_k, top_k, block_size)
  expected_shapes = ((slots_len,), (slots_len,), (1 + fields * max_blocks,))
  shapes = (
      metadata.src_rows.shape,
      metadata.tokens.shape,
      metadata.blocks.shape,
  )
  if shapes != expected_shapes:
    raise ValueError(
        "metadata (src_rows, tokens, blocks) must have shapes"
        f" {expected_shapes}, got {shapes}; was it derived from"
        f" {num_out_tokens * top_k} indices with top_k={top_k}?"
    )
  # A (d0, d1) token is padded to whole (sublanes, 128) tiles in VMEM.
  itemsize = jnp.dtype(x.dtype).itemsize
  sublanes = 8 * max(1, 4 // itemsize)
  token_vmem_bytes = pl.cdiv(dim0, sublanes) * sublanes * pl.cdiv(dim1, 128) * 128 * itemsize
  vmem_budget = _RAGGED_GATHER_REDUCE_TC_VMEM_BUDGET
  if num_buffers * block_size * token_vmem_bytes > vmem_budget:
    raise ValueError(
        f"block_size {block_size} does not fit {num_buffers} VMEM buffers of"
        f" ({dim0}, {dim1}) {x.dtype} tokens in {vmem_budget} bytes; use"
        f" block_size <= {vmem_budget // (num_buffers * token_vmem_bytes)}."
    )
  if num_out_tokens == 0 or max_input_tokens == 0:
    if out is not None and not zero_initialized:
      return out
    return jnp.zeros((num_out_tokens, dim0, dim1), x.dtype)
  if not zero_initialized and out is None:
    out = jnp.zeros((num_out_tokens, dim0, dim1), x.dtype)

  lookahead = _RAGGED_GATHER_REDUCE_TC_LOOKAHEAD
  # A block's slot window is loaded two steps before its rows are issued and
  # is live through its reduce, lookahead steps later, so windows cycle
  # through a ring of lookahead + 2 SMEM slots.
  ring = lookahead + 2
  has_out_in = out is not None

  def _ragged_gather_reduce_kernel(*args):
    if has_out_in:
      (
          blocks_smem_ref,
          src_hbm_ref,
          tok_hbm_ref,
          x_hbm_ref,
          o_in_hbm_ref,
          o_hbm_ref,
          vmem_ref,
          src_smem_ref,
          tok_smem_ref,
          recv_sem,
          send_sem,
          slots_sem,
      ) = args
    else:
      (
          blocks_smem_ref,
          src_hbm_ref,
          tok_hbm_ref,
          x_hbm_ref,
          o_hbm_ref,
          vmem_ref,
          src_smem_ref,
          tok_smem_ref,
          recv_sem,
          send_sem,
          slots_sem,
      ) = args
      o_in_hbm_ref = None

    blocks_smem_ref = pltpu.annotate(blocks_smem_ref, no_store=True, no_bank_conflict=True)
    src_smem_ref = pltpu.annotate(src_smem_ref, no_store=True, no_bank_conflict=True)
    tok_smem_ref = pltpu.annotate(tok_smem_ref, no_store=True, no_bank_conflict=True)
    vmem_ref = pltpu.annotate(vmem_ref, no_bank_conflict=True, no_hazard=True)

    num_blocks = blocks_smem_ref[0]

    # Block blk reduces compacted slots [slot_start, slot_start + num_slots)
    # into output tokens [token_start, token_start + num_tokens). In its VMEM
    # buffer, rows [0, num_slots) hold the gathered input rows and rows
    # [block_size - num_tokens, block_size) hold the output tokens.
    def _block(blk):
      return (
          blocks_smem_ref[1 + blk],
          blocks_smem_ref[1 + max_blocks + blk],
          blocks_smem_ref[1 + 2 * max_blocks + blk],
          blocks_smem_ref[1 + 3 * max_blocks + blk],
      )

    def _window_start(slot_start):
      return jnp.minimum((slot_start // align) * align, slots_len - window)

    def _slots_loads(blk):
      r = blk % ring
      window_start = pl.multiple_of(_window_start(_block(blk)[0]), align)
      return [
          pltpu.make_async_copy(
              hbm_ref.at[pl.ds(window_start, window)],
              smem_ref.at[pl.ds(r * window, window)],
              slots_sem.at[r],
          )
          for hbm_ref, smem_ref in (
              (src_hbm_ref, src_smem_ref),
              (tok_hbm_ref, tok_smem_ref),
          )
      ]

    def _slot_base(blk):
      """Returns the SMEM offset of block blk's first slot."""
      slot_start = _block(blk)[0]
      return (blk % ring) * window + slot_start - _window_start(slot_start)

    def _pow2_chunks(n, fn):
      """Calls fn(off, size) on static power-of-two sizes summing to n."""
      off = 0
      for k in reversed(range(block_size.bit_length())):
        size = 1 << k
        bit = n & size

        @pl.when(bit != 0)
        def _(off=off, size=size):
          fn(off, size)

        off = off + bit

    def _unrolled_chunks(n, max_unroll: int, fn):
      """Calls fn(base, unroll) on static-sized chunks covering [0, n)."""

      def _do_pow2(off, size):
        unroll = min(size, max_unroll)
        if size == unroll:
          fn(off, unroll)
        else:

          @pl.loop(0, size // unroll)
          def _(c):
            fn(off + c * unroll, unroll)

      _pow2_chunks(n, _do_pow2)

    def _wait_rows(b, n, sem):
      def _wait(off, size):
        del off
        rows = vmem_ref.at[b, pl.ds(0, size), :, :]
        pltpu.make_async_copy(rows, rows, sem.at[b]).wait()

      _pow2_chunks(n, _wait)

    def _in_out_start(blk):
      """Initializes block blk's output token region in VMEM."""
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)
      out_base = block_size - num_tokens
      if zero_initialized:

        def _zero_out(off, size):
          vmem_ref[b, pl.ds(out_base + off, size), :, :] = jnp.zeros((size, dim0, dim1), x.dtype)

        _unrolled_chunks(num_tokens, 256, _zero_out)
      else:
        assert o_in_hbm_ref is not None

        def _load_out(off, size):
          pltpu.make_async_copy(
              o_in_hbm_ref.at[pl.ds(token_start + off, size), :, :],
              vmem_ref.at[b, pl.ds(out_base + off, size), :, :],
              recv_sem.at[b],
          ).start()

        _pow2_chunks(num_tokens, _load_out)

    def _issue_in_chunk(b, slot_base, base_i, unroll):
      idx_base = slot_base + base_i
      srcs = [src_smem_ref[idx_base + u] for u in range(unroll)]
      for u in range(unroll):
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(srcs[u], 1), :, :],
            vmem_ref.at[b, pl.ds(base_i + u, 1), :, :],
            recv_sem.at[b],
        ).start()

    def _in_start(blk):
      b = blk % num_buffers
      slot_base = _slot_base(blk)
      _in_out_start(blk)
      _unrolled_chunks(
          _block(blk)[1],
          _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          lambda off, u: _issue_in_chunk(b, slot_base, off, u),
      )

    def _in_wait(blk):
      _, num_slots, _, num_tokens = _block(blk)
      wait_count = num_slots if zero_initialized else num_slots + num_tokens
      _wait_rows(blk % num_buffers, wait_count, recv_sem)

    def _out_start(blk):
      b = blk % num_buffers
      _, _, token_start, num_tokens = _block(blk)
      out_base = block_size - num_tokens

      def _store(off, size):
        pltpu.make_async_copy(
            vmem_ref.at[b, pl.ds(out_base + off, size), :, :],
            o_hbm_ref.at[pl.ds(token_start + off, size), :, :],
            send_sem.at[b],
        ).start()

      _pow2_chunks(num_tokens, _store)

    def _out_wait(blk):
      _wait_rows(blk % num_buffers, _block(blk)[3], send_sem)

    def _slots_wait(blk):
      for load in _slots_loads(blk):
        load.wait()

    for blk in range(ring):

      @pl.when(blk < num_blocks)
      def _load_first_slots(blk=blk):
        for load in _slots_loads(blk):
          load.start()

    for blk in range(lookahead):

      @pl.when(blk < num_blocks)
      def _start_first_rows(blk=blk):
        _slots_wait(blk)
        _in_start(blk)

    @pl.loop(0, num_blocks)
    def _step(i):
      _in_wait(i)
      nxt = i + lookahead
      has_nxt = nxt < num_blocks

      # Block nxt reuses the buffer of block nxt - num_buffers, whose store
      # must be done.
      @pl.when(nxt >= num_buffers)
      def _free_buffer():
        _out_wait(nxt - num_buffers)

      @pl.when(has_nxt)
      def _wait_next_slots():
        _slots_wait(nxt)

      nxt_clamped = jnp.minimum(nxt, num_blocks - 1)

      @pl.when(has_nxt)
      def _start_next_out():
        _in_out_start(nxt_clamped)

      b_i = i % num_buffers
      _, num_reduce, token_start_i, num_tokens_i = _block(i)
      reduce_base = _slot_base(i)
      token_bias_i = token_start_i - (block_size - num_tokens_i)

      # Block nxt's row DMAs are started in block i's reduce loop, so the
      # scalar DMA issue shares bundles with the vector reduce.
      b_nxt = nxt_clamped % num_buffers
      num_issue = jnp.where(has_nxt, _block(nxt_clamped)[1], 0)
      issue_base = _slot_base(nxt_clamped)

      unroll = _RAGGED_GATHER_REDUCE_TC_REDUCE_UNROLL
      num_both_chunks = jnp.minimum(num_reduce, num_issue) // unroll
      num_reduce_chunks = num_reduce // unroll
      base_issue_rem = num_both_chunks * unroll
      base_reduce_rem = num_reduce_chunks * unroll

      def _reduce_step(t, row, prev_t, total):
        if zero_initialized:
          total = jnp.where(t == prev_t, total, 0.0) + row
        else:
          init = vmem_ref[b_i, t, :, :].astype(jnp.float32)
          total = jnp.where(t == prev_t, total, init) + row
        vmem_ref[b_i, t, :, :] = total.astype(x.dtype)
        return t, total

      def _issue_and_reduce_chunk(c, carry):
        prev_t, total = carry
        base_j = c * unroll
        issue_idx = issue_base + base_j
        reduce_idx = reduce_base + base_j
        srcs = [src_smem_ref[issue_idx + u] for u in range(unroll)]
        toks = [tok_smem_ref[reduce_idx + u] - token_bias_i for u in range(unroll)]
        rows = vmem_ref[b_i, pl.ds(base_j, unroll), :, :].astype(jnp.float32)
        for u in range(unroll):
          pltpu.make_async_copy(
              x_hbm_ref.at[pl.ds(srcs[u], 1), :, :],
              vmem_ref.at[b_nxt, pl.ds(base_j + u, 1), :, :],
              recv_sem.at[b_nxt],
          ).start()
          prev_t, total = _reduce_step(toks[u], rows[u], prev_t, total)
        return prev_t, total

      def _reduce_chunk(c, carry):
        prev_t, total = carry
        base_j = c * unroll
        reduce_idx = reduce_base + base_j
        toks = [tok_smem_ref[reduce_idx + u] - token_bias_i for u in range(unroll)]
        rows = vmem_ref[b_i, pl.ds(base_j, unroll), :, :].astype(jnp.float32)
        for u in range(unroll):
          prev_t, total = _reduce_step(toks[u], rows[u], prev_t, total)
        return prev_t, total

      def _reduce_one(j, carry):
        prev_t, total = carry
        t = tok_smem_ref[reduce_base + j] - token_bias_i
        row = vmem_ref[b_i, j, :, :].astype(jnp.float32)
        return _reduce_step(t, row, prev_t, total)

      carry = (jnp.int32(-1), jnp.zeros((dim0, dim1), jnp.float32))
      carry = jax.lax.fori_loop(0, num_both_chunks, _issue_and_reduce_chunk, carry)
      _unrolled_chunks(
          num_issue - base_issue_rem,
          _RAGGED_GATHER_REDUCE_TC_ISSUE_UNROLL,
          lambda off, u: _issue_in_chunk(b_nxt, issue_base, base_issue_rem + off, u),
      )
      carry = jax.lax.fori_loop(num_both_chunks, num_reduce_chunks, _reduce_chunk, carry)
      jax.lax.fori_loop(base_reduce_rem, num_reduce, _reduce_one, carry)
      _out_start(i)

      # Block i's slot window is free now that its reduce is done.
      @pl.when(i + ring < num_blocks)
      def _load_later_slots():
        for load in _slots_loads(i + ring):
          load.start()

    # The last num_buffers - lookahead stores are still pending.
    for k in range(num_buffers - lookahead):

      @pl.when(num_blocks - 1 - k >= 0)
      def _wait_last_stores(k=k):
        _out_wait(num_blocks - 1 - k)

  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  out_shape = jax.ShapeDtypeStruct((num_out_tokens, dim0, dim1), x.dtype, manual_axis_type=manual_axis_type)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  dma_sems = pltpu.SemaphoreType.DMA((num_buffers,))
  in_specs = [
      pl.BlockSpec(memory_space=pltpu.SMEM),
      hbm_spec,
      hbm_spec,
      hbm_spec,
  ]
  inputs = [metadata.blocks, metadata.src_rows, metadata.tokens, x]
  input_output_aliases = {}
  if has_out_in:
    in_specs.append(hbm_spec)
    inputs.append(out)
    input_output_aliases[4] = 0
  scratch_shapes = [
      pltpu.VMEM((num_buffers, block_size, dim0, dim1), x.dtype),
      pltpu.SMEM((ring * window,), jnp.int32),
      pltpu.SMEM((ring * window,), jnp.int32),
      dma_sems,
      dma_sems,
      pltpu.SemaphoreType.DMA((ring,)),
  ]
  return pl.pallas_call(
      _ragged_gather_reduce_kernel,
      out_shape=out_shape,
      in_specs=in_specs,
      out_specs=hbm_spec,
      scratch_shapes=scratch_shapes,
      input_output_aliases=input_output_aliases,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=_RAGGED_GATHER_REDUCE_TC_VMEM_LIMIT,
          disable_bounds_checks=True,
      ),
      name="ragged_gather_reduce_tc",
  )(*inputs)
