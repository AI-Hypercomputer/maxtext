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

"""Fragment extraction, application and outer optimization for threaded streaming DiLoCo.

In threaded DiLoCo each learner extracts one parameter fragment per sync step, sends it to wherever the outer
optimizer runs, and applies the synced fragment a few steps later. This module holds the jitted pieces: fragment
extraction and application on a learner's mesh, and the outer optimizer step on transfer fragments.

A *transfer fragment* is a dict `key -> 1-D jax.Array`, keyed like `FragmentedTreeManipulator.get_flat_fragment`.
Each leaf is flattened shard-locally: every device flattens its own shard and the 1-D result is sharded over the
same mesh axes, so extraction and application never move data between chips and host transfers only see 1-D
arrays. The 1-D element order is therefore mesh-block-major (shard after shard), not the row-major order of the
fragment slice: a transfer fragment is a permutation of `get_flat_fragment(...)[key].reshape(-1)`, and only
`FragmentTransfer.apply` with the same layout restores it. Leaves whose fragment shape is not divisible by their
sharding fall back to a global (row-major) reshape.
"""

import dataclasses
import functools
import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from maxtext.trainers.diloco.utils.fragmenter import BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX, FragmentedTreeManipulator

TransferFragment = dict[str, jax.Array]


@dataclasses.dataclass(frozen=True)
class LeafLayout:
  """Static description of one entry of a transfer fragment."""

  key: str
  shape: tuple[int, ...]  # Shape of the fragment slice, not of the full parameter.
  spec: PartitionSpec | None  # Sharding used for the shard-local flatten; None means a global reshape.


def _spec_axes(spec: PartitionSpec) -> list[str]:
  axes = []
  for entry in spec:
    if entry is not None:
      axes.extend((entry,) if isinstance(entry, str) else entry)
  return axes


def _shard_local_spec(shape: tuple[int, ...], spec: PartitionSpec, mesh: Mesh) -> PartitionSpec | None:
  """Returns `spec` if `shape` splits evenly under it on `mesh` and it shards anything, else None."""
  if not _spec_axes(spec):
    return None
  for dim, entry in zip(shape, tuple(spec) + (None,) * (len(shape) - len(spec))):
    axes = () if entry is None else ((entry,) if isinstance(entry, str) else entry)
    if dim % math.prod(mesh.shape[a] for a in axes) != 0:
      return None
  return spec


def _flat_spec(spec: PartitionSpec | None) -> PartitionSpec:
  """PartitionSpec of the 1-D transfer array of a leaf flattened under `spec` (replicated for a global reshape)."""
  return PartitionSpec() if spec is None else PartitionSpec(tuple(_spec_axes(spec)))


def _flatten(x: jax.Array, mesh: Mesh, spec: PartitionSpec | None) -> jax.Array:
  if spec is None:
    return jnp.reshape(x, (-1,))
  flat_spec = _flat_spec(spec)
  return jax.shard_map(lambda a: jnp.reshape(a, (-1,)), mesh=mesh, in_specs=spec, out_specs=flat_spec, check_vma=False)(x)


def _unflatten(y: jax.Array, mesh: Mesh, leaf: LeafLayout) -> jax.Array:
  if leaf.spec is None:
    return jnp.reshape(y, leaf.shape)
  local_shape = list(leaf.shape)
  for i, entry in enumerate(leaf.spec):
    for axis in () if entry is None else ((entry,) if isinstance(entry, str) else entry):
      local_shape[i] //= mesh.shape[axis]
  flat_spec = _flat_spec(leaf.spec)
  return jax.shard_map(
      lambda a: jnp.reshape(a, local_shape), mesh=mesh, in_specs=flat_spec, out_specs=leaf.spec, check_vma=False
  )(y)


def _base_keystr(key: str) -> str:
  for suffix in (BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX):
    if key.endswith(suffix):
      return key[: -len(suffix)]
  return key


class FragmentTransfer:
  """Jitted fragment extraction and application for one learner's parameter tree.

  All layer fragments share one extraction and one application executable (the fragment index is a traced
  argument); fragment 0 has its own pair. Application pins its output shardings to the live parameters and donates
  them, so the train step sees unchanged shardings and the scanned layer stacks are updated in place.

  The parameter shardings are read once, here: the transfer layouts and the output shardings of `apply` are pinned
  to the shardings of `params` at construction. Build it from parameters that already have the shardings the train
  step uses (for example after a Zero-1 reshard); otherwise every `apply` returns parameters in the old layout.

  Args:
    manipulator: Fragmenter built from a tree with the same structure as `params`.
    params: The learner's parameter tree (concrete or abstract, with NamedShardings on one mesh).
    alpha: Weight of the learner's current value when applying a synced fragment
      (`communication_overlapping_alpha`); 0 replaces the fragment.

  Raises:
    ValueError: If a leaf of `params` does not have a NamedSharding, or the NamedShardings use different meshes.
  """

  def __init__(self, manipulator: FragmentedTreeManipulator, params: Any, alpha: float = 0.0):
    self.manipulator = manipulator
    keyed = jax.tree_util.tree_flatten_with_path(params)[0]
    shardings = {jax.tree_util.keystr(path): getattr(leaf, "sharding", None) for path, leaf in keyed}
    for key, sharding in shardings.items():
      if not isinstance(sharding, NamedSharding):
        raise ValueError(f"FragmentTransfer needs a NamedSharding on every parameter; {key} has {sharding!r}.")
    meshes = {sharding.mesh for sharding in shardings.values()}
    if len(meshes) != 1:
      raise ValueError(f"FragmentTransfer needs all parameters on one mesh; found {len(meshes)} meshes.")
    self.mesh = meshes.pop()
    self.num_fragments = manipulator.num_fragments

    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    # Every layer fragment has the same layout, so fragment 1 stands in for all of them.
    self.layouts = {f: self._build_layout(manipulator.get_flat_fragment(abstract, f), shardings) for f in (0, 1)}
    self._first_layer, self._layer_start_step, self._layer_stride, self._num_layers = self._layer_progression()
    # With interleaved layers (stride > 1), fragment `f`'s layers are one column of the scan axis viewed as
    # (num_layers, stride) whenever every fragment starts inside the first row.
    num_layer_fragments = len(manipulator.fragment_to_layer_indices)
    self._layer_grid = (
        self._layer_stride > 1
        and self._first_layer + (num_layer_fragments - 1) * self._layer_start_step < self._layer_stride
    )

    params_shardings = jax.tree.map(lambda x: x.sharding, params)
    apply_kwargs = {"out_shardings": params_shardings, "donate_argnums": 0}
    self._extract_non_scanned = self._jit_extract(layer_fragment=False)
    self._extract_layer = self._jit_extract(layer_fragment=True)
    self._apply_non_scanned = jax.jit(functools.partial(self._apply, layer_fragment=False, alpha=alpha), **apply_kwargs)
    self._apply_layer = jax.jit(functools.partial(self._apply, layer_fragment=True, alpha=alpha), **apply_kwargs)

  def _build_layout(self, abstract_fragment: dict[str, jax.ShapeDtypeStruct], shardings) -> tuple[LeafLayout, ...]:
    layout = []
    for key in sorted(abstract_fragment):
      shape = tuple(abstract_fragment[key].shape)
      spec = shardings[_base_keystr(key)].spec
      layout.append(LeafLayout(key, shape, _shard_local_spec(shape, spec, self.mesh)))
    return tuple(layout)

  def _jit_extract(self, *, layer_fragment: bool):
    """Jits `_extract`, pinning each 1-D output to its transfer sharding (replicated for a global reshape)."""
    out_shardings = {leaf.key: NamedSharding(self.mesh, _flat_spec(leaf.spec)) for leaf in self._layout(layer_fragment)}
    return jax.jit(functools.partial(self._extract, layer_fragment=layer_fragment), out_shardings=out_shardings)

  def _layer_progression(self) -> tuple[int, int, int, int]:
    """Describes fragment `f`'s layers as `first + (f - 1) * start_step + stride * arange(num_layers)`.

    Covers both sequential and interleaved layer assignment, which lets all layer fragments share one executable.
    """
    indices = self.manipulator.fragment_to_layer_indices
    if not indices:
      return 0, 0, 1, 0
    first = indices[1]
    num_layers = len(first)
    stride = first[1] - first[0] if num_layers > 1 else 1
    start_step = indices[2][0] - first[0] if len(indices) > 1 else 0
    for f, layers in indices.items():
      expected = tuple(first[0] + (f - 1) * start_step + stride * j for j in range(num_layers))
      if tuple(layers) != expected:
        raise ValueError(f"Layer fragment {f} indices {layers} are not an arithmetic progression {expected}.")
    return first[0], start_step, stride, num_layers

  def _as_layer_grid(self, leaf: jax.Array, axis: int) -> jax.Array | None:
    """Views `leaf`'s scan axis as (num_layers, stride), so an interleaved fragment is one slice; None if it can't."""
    if not self._layer_grid or leaf.shape[axis] != self._num_layers * self._layer_stride:
      return None
    return jnp.reshape(leaf, leaf.shape[:axis] + (self._num_layers, self._layer_stride) + leaf.shape[axis + 1 :])

  def _layer_slice(self, leaf: jax.Array, axis: int, fragment_idx: jax.Array) -> jax.Array:
    start = self._first_layer + (fragment_idx - 1) * self._layer_start_step
    if self._layer_stride == 1:
      return jax.lax.dynamic_slice_in_dim(leaf, start, self._num_layers, axis=axis)
    grid = self._as_layer_grid(leaf, axis)
    if grid is not None:
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(grid, start, 1, axis=axis + 1), axis + 1)
    return jnp.take(leaf, start + self._layer_stride * jnp.arange(self._num_layers), axis=axis)

  def _set_layer_slice(self, leaf: jax.Array, axis: int, fragment_idx: jax.Array, value: jax.Array) -> jax.Array:
    """Writes `value` into the layers of fragment `fragment_idx` along `axis` (inverse of `_layer_slice`)."""
    start = self._first_layer + (fragment_idx - 1) * self._layer_start_step
    if self._layer_stride == 1:
      return jax.lax.dynamic_update_slice_in_dim(leaf, value, start, axis=axis)
    grid = self._as_layer_grid(leaf, axis)
    if grid is not None:
      grid = jax.lax.dynamic_update_slice_in_dim(grid, jnp.expand_dims(value, axis + 1), start, axis=axis + 1)
      return jnp.reshape(grid, leaf.shape)
    index = [slice(None)] * leaf.ndim
    index[axis] = start + self._layer_stride * jnp.arange(self._num_layers)
    return leaf.at[tuple(index)].set(value)

  def _bucket_range(self, key: str, fragment_idx: jax.Array) -> tuple[Any, int, int]:
    """Returns `(start, size, axis)` of the slice of a bucketized leaf carried by `key`'s fragment."""
    spec = self.manipulator.bucketized_leaves[_base_keystr(key)]
    if key.endswith(BUCKET_KEY_SUFFIX):
      return (fragment_idx - 1) * spec.chunk_size, spec.chunk_size, spec.axis
    return self.manipulator.num_layer_fragments * spec.chunk_size, spec.remainder, spec.axis

  def _get(self, leaf: jax.Array, key: str, fragment_idx: jax.Array) -> jax.Array:
    if key.endswith((BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX)):
      start, size, axis = self._bucket_range(key, fragment_idx)
      return jax.lax.dynamic_slice_in_dim(leaf, start, size, axis=axis)
    if self.manipulator.keypath_to_is_scanned[key]:
      return self._layer_slice(leaf, self.manipulator.param_scan_axis, fragment_idx)
    return leaf

  def _set(self, leaf: jax.Array, key: str, fragment_idx: jax.Array, value: jax.Array) -> jax.Array:
    if key.endswith((BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX)):
      start, _, axis = self._bucket_range(key, fragment_idx)
      return jax.lax.dynamic_update_slice_in_dim(leaf, value, start, axis=axis)
    if self.manipulator.keypath_to_is_scanned[key]:
      return self._set_layer_slice(leaf, self.manipulator.param_scan_axis, fragment_idx, value)
    return value

  def _layout(self, layer_fragment: bool) -> tuple[LeafLayout, ...]:
    return self.layouts[1 if layer_fragment else 0]

  def _extract(self, params: Any, fragment_idx: jax.Array, *, layer_fragment: bool) -> TransferFragment:
    leaves = dict(zip(self.manipulator.leaf_keystrs, jax.tree.leaves(params)))
    return {
        leaf.key: _flatten(self._get(leaves[_base_keystr(leaf.key)], leaf.key, fragment_idx), self.mesh, leaf.spec)
        for leaf in self._layout(layer_fragment)
    }

  def _apply(
      self, params: Any, fragment_idx: jax.Array, fragment: TransferFragment, *, layer_fragment: bool, alpha: float
  ) -> Any:
    """Traced body of `apply`; `alpha` and `layer_fragment` are bound statically."""
    leaves, treedef = jax.tree.flatten(params)
    index = {k: i for i, k in enumerate(self.manipulator.leaf_keystrs)}
    for leaf in self._layout(layer_fragment):
      i = index[_base_keystr(leaf.key)]
      value = _unflatten(fragment[leaf.key], self.mesh, leaf).astype(leaves[i].dtype)
      if alpha:
        value = alpha * self._get(leaves[i], leaf.key, fragment_idx) + (1.0 - alpha) * value
      leaves[i] = self._set(leaves[i], leaf.key, fragment_idx, value)
    return jax.tree.unflatten(treedef, leaves)

  def _check_fragment_idx(self, fragment_idx: int) -> None:
    # Out-of-range indices would not fail inside the jitted functions: slices clamp, gathers fill and scatters drop.
    if not 0 <= fragment_idx < self.num_fragments:
      raise ValueError(f"fragment_idx ({fragment_idx}) must be in [0, {self.num_fragments}).")

  def extract(self, params: Any, fragment_idx: int) -> TransferFragment:
    """Returns fragment `fragment_idx` of `params` as a transfer fragment on the learner mesh."""
    self._check_fragment_idx(fragment_idx)
    fn = self._extract_layer if fragment_idx > 0 else self._extract_non_scanned
    # A host scalar: a jnp scalar would live on the default device and be copied to this learner's mesh every call.
    return fn(params, np.int32(fragment_idx))

  def apply(self, params: Any, fragment_idx: int, fragment: TransferFragment) -> Any:
    """Writes a synced transfer fragment into `params` (donated) and returns the updated tree."""
    self._check_fragment_idx(fragment_idx)
    fn = self._apply_layer if fragment_idx > 0 else self._apply_non_scanned
    return fn(params, np.int32(fragment_idx), fragment)


def move_fragment(fragment: TransferFragment, mesh: Mesh) -> TransferFragment:
  """Copies a transfer fragment onto `mesh`, keeping each array's PartitionSpec (one batched `device_put`)."""
  return jax.device_put(fragment, {k: NamedSharding(mesh, v.sharding.spec) for k, v in fragment.items()})


def nesterov_outer_step(
    outer: TransferFragment,
    trace: TransferFragment,
    learner_fragments: Sequence[TransferFragment],
    *,
    learning_rate: float,
    momentum: float,
) -> tuple[TransferFragment, TransferFragment]:
  """One `optax.sgd(learning_rate, momentum, nesterov=True)` step on the DiLoCo pseudo-gradient.

  The pseudo-gradient is `mean(outer - learner_fragments)`, the order SPMD DiLoCo uses, so an element that no learner
  changed gets an exactly zero pseudo-gradient. Arithmetic is done in float32; the new outer value and the momentum
  trace are kept in the dtypes of `outer` and `trace`.

  Args:
    outer: Outer parameters of the fragment.
    trace: Momentum trace of the fragment. It is donated: use the returned trace afterwards.
    learner_fragments: Each learner's current value of the fragment. All inputs must be on the same devices.
    learning_rate: Outer learning rate.
    momentum: Outer Nesterov momentum.

  Returns:
    `(new_outer, new_trace)`, with the shardings of `outer` and `trace`. `outer` is not donated, so callers may keep
    using the arrays they passed in (e.g. a synced fragment that was already handed on).
  """
  shardings = tuple((key, outer[key].sharding, trace[key].sharding) for key in sorted(outer))
  return _jit_outer_step(shardings, learning_rate, momentum)(outer, trace, learner_fragments)


@functools.lru_cache(maxsize=None)
def _jit_outer_step(shardings: tuple[tuple[str, Any, Any], ...], learning_rate: float, momentum: float):
  """Jits `_outer_step` with its outputs pinned to `shardings` (`(key, outer sharding, trace sharding)` triples)."""
  out_shardings = ({key: o for key, o, _ in shardings}, {key: t for key, _, t in shardings})
  return jax.jit(
      functools.partial(_outer_step, learning_rate=learning_rate, momentum=momentum),
      out_shardings=out_shardings,
      donate_argnums=1,
  )


def _outer_step(
    outer: TransferFragment,
    trace: TransferFragment,
    learner_fragments: Sequence[TransferFragment],
    *,
    learning_rate: float,
    momentum: float,
) -> tuple[TransferFragment, TransferFragment]:
  """Traced body of `nesterov_outer_step`."""
  new_outer, new_trace = {}, {}
  for key, outer_value in outer.items():
    outer_f32 = outer_value.astype(jnp.float32)
    pseudo_grad = sum(outer_f32 - f[key].astype(jnp.float32) for f in learner_fragments) / len(learner_fragments)
    trace_value = momentum * trace[key].astype(jnp.float32) + pseudo_grad
    update = learning_rate * (pseudo_grad + momentum * trace_value)
    new_outer[key] = (outer_f32 - update).astype(outer_value.dtype)
    new_trace[key] = trace_value.astype(trace[key].dtype)
  return new_outer, new_trace
