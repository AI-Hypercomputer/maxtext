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

"""Parameter fragment manipulation and scheduling utilities for Streaming DiLoCo (https://arxiv.org/abs/2501.18512).

A **fragment** is a disjoint subset of the model parameter PyTree (e.g., embeddings/head or a block of decoder
layers). While standard DiLoCo synchronizes the entire model at once, Streaming DiLoCo pipelines cross-island
communication by synchronizing one fragment per inner step to overlap inter-cluster communication with computation.

Fragment 0 holds the non-scanned parameters; fragments 1..N-1 each hold a subset of the scanned decoder layers.
With `diloco_bucketize_non_scanned`, non-scanned matrices (ndim >= 2, e.g. the embedding table and the output
projection) are instead split across the layer fragments along their longest unsharded axis that has at least N-1
indices, so that every fragment carries a similar number of bytes. There is no size threshold: any such matrix is
split, whatever its size. A matrix whose long axes are all sharded stays whole in fragment 0. Bucketization applies to
every streaming DiLoCo runner that builds its fragments with this class.
"""

import re
from typing import Any, Iterator, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.utils import max_logging

# Suffixes of flat-fragment keys that carry a slice of a bucketized non-scanned leaf.
BUCKET_KEY_SUFFIX = "#bucket"
REMAINDER_KEY_SUFFIX = "#remainder"


class BucketSpec(NamedTuple):
  """Split of a non-scanned leaf along one of its unsharded axes across the layer fragments.

  Layer fragment `f` (1-based) carries indices `[(f - 1) * chunk_size, f * chunk_size)` of `axis` (the leaf's longest
  unsharded axis, e.g. vocab for both the embedding table and the output head under FSDP); the trailing `remainder`
  indices stay in fragment 0.
  """

  chunk_size: int
  remainder: int
  axis: int = 0


def _slice_axis(leaf: Any, axis: int, start: int, size: int) -> Any:
  """Static slice `[start, start + size)` along `axis`; also accepts `jax.ShapeDtypeStruct`."""
  if isinstance(leaf, jax.ShapeDtypeStruct):
    shape = list(leaf.shape)
    shape[axis] = size
    return jax.ShapeDtypeStruct(tuple(shape), leaf.dtype, sharding=getattr(leaf, "sharding", None))
  return jax.lax.slice_in_dim(leaf, start, start + size, axis=axis)


def _is_sharding_leaf(x: Any) -> bool:
  return x is None or isinstance(x, jax.sharding.Sharding)


def _sharding_leaves(kvs: list[tuple[Any, Any]], shardings: Any) -> list[Any]:
  """Per-leaf shardings: from `shardings` if given, else from the leaves themselves (None for tracers).

  Args:
    kvs: `(keypath, leaf)` pairs of the parameter tree, as given by `jax.tree_util.tree_flatten_with_path`.
    shardings: Optional PyTree of shardings with the structure of the parameter tree.
  """
  if shardings is not None:
    sharding_kvs = jax.tree_util.tree_flatten_with_path(shardings, is_leaf=_is_sharding_leaf)[0]
    if len(sharding_kvs) != len(kvs):
      raise ValueError(f"shardings has {len(sharding_kvs)} leaves but the parameter tree has {len(kvs)}.")
    for (param_path, _), (sharding_path, _) in zip(kvs, sharding_kvs):
      param_key, sharding_key = jax.tree_util.keystr(param_path), jax.tree_util.keystr(sharding_path)
      if param_key != sharding_key:
        raise ValueError(
            f"shardings does not have the structure of the parameter tree: found {sharding_key} in shardings where"
            f" the parameter tree has {param_key}."
        )
    return [s for _, s in sharding_kvs]
  # Tracers carry no sharding (accessing it raises), so traced leaves count as unknown -> unsharded.
  return [None if isinstance(leaf, jax.core.Tracer) else getattr(leaf, "sharding", None) for _, leaf in kvs]


def _axis_shard_counts(shape: tuple[int, ...], sharding: Any) -> list[int]:
  """Number of shards along each axis of `shape` under `sharding`; 1 everywhere if it is not a `NamedSharding`.

  Entries of the PartitionSpec that are neither mesh axis names nor tuples of them (e.g. `UNCONSTRAINED`) give 0, so
  such an axis is never chosen for splitting.
  """
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return [1] * len(shape)
  spec = tuple(sharding.spec) + (None,) * (len(shape) - len(sharding.spec))
  counts = []
  for entry in spec[: len(shape)]:
    if entry is None:
      counts.append(1)
    elif isinstance(entry, str):
      counts.append(int(sharding.mesh.shape[entry]))
    elif isinstance(entry, tuple):
      counts.append(int(np.prod([sharding.mesh.shape[a] for a in entry])))
    else:
      counts.append(0)
  return counts


class FragmentedTreeManipulator:
  """For Streaming DiLoCo: Partitions and manipulates fragments of a JAX PyTree, supporting scanned layers."""

  def __init__(
      self,
      keypath_to_is_scanned: dict[str, bool],
      fragment_to_layer_indices: dict[int, tuple[int, ...]],
      num_fragments: int,
      param_scan_axis: int = 0,
      leaf_keystrs: list[str] | None = None,
      bucketized_leaves: dict[str, BucketSpec] | None = None,
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments
    self.param_scan_axis = param_scan_axis
    self.leaf_keystrs = leaf_keystrs or []
    self.bucketized_leaves = bucketized_leaves or {}

  @property
  def num_layer_fragments(self) -> int:
    return self.num_fragments - 1

  @classmethod
  def create(cls, params_tree: Any, config: Any, shardings: Any = None) -> "FragmentedTreeManipulator":
    """Creates a FragmentedTreeManipulator from the parameters PyTree and configuration.

    Args:
      params_tree: Parameter PyTree (concrete arrays, `jax.ShapeDtypeStruct`s or tracers).
      config: Config with the DiLoCo fragment settings.
      shardings: Optional PyTree of `NamedSharding`s with the structure of `params_tree`. Only used to pick the split
        axis of bucketized leaves. When omitted, the shardings of concrete (non-traced) leaves are used; leaves
        without a `NamedSharding` are treated as unsharded. Callers that build the manipulator inside a traced
        function (SPMD streaming DiLoCo) must pass it, since tracers carry no sharding.
    """
    kvs, _ = jax.tree_util.tree_flatten_with_path(params_tree)

    num_layers = config.num_decoder_layers
    num_fragments = config.num_diloco_fragments
    num_transformer_fragments = num_fragments - 1

    if num_transformer_fragments <= 0:
      raise ValueError(
          f"num_diloco_fragments ({num_fragments}) must be at least 2 (1 for non-scanned parameters, at least 1 for"
          " scanned layers)."
      )
    if num_layers % num_transformer_fragments != 0:
      raise ValueError(
          f"num_decoder_layers ({num_layers}) must be divisible by "
          f"num_diloco_fragments - 1 ({num_transformer_fragments}) for now."
      )

    num_synced = num_layers // num_transformer_fragments
    use_sequential = config.use_sequential_layers
    param_scan_axis = getattr(config, "param_scan_axis", 0)

    # Pre-compute layer indices for each fragment 1 ... num_transformer_fragments
    fragment_to_layer_indices = {}
    for i in range(1, num_fragments):
      sync_id = i - 1
      if use_sequential:
        indices = list(range(sync_id * num_synced, (sync_id + 1) * num_synced))
      else:
        indices = list(range(sync_id, num_layers, num_transformer_fragments))
      fragment_to_layer_indices[i] = tuple(indices)

    # Regex to identify scanned layer parameters
    scanned_regex = re.compile(r"/(?:layers|blocks|moe_layers|dense_layers|layers_outside_pipeline)(?:/|$)")
    keypath_to_is_scanned = {}
    leaf_keystrs = []

    for keypath, v in kvs:
      parts = []
      for k in keypath:
        parts.append(str(k.key) if hasattr(k, "key") else (str(k.idx) if hasattr(k, "idx") else str(k)))
      serialized_path = "/" + "/".join(parts)
      keystr = jax.tree_util.keystr(keypath)
      leaf_keystrs.append(keystr)

      is_scanned = (
          bool(scanned_regex.search(serialized_path))
          and hasattr(v, "shape")
          and len(v.shape) > 0
          and v.shape[param_scan_axis] == num_layers
      )
      keypath_to_is_scanned[keystr] = is_scanned

    # Matrices with an unsharded axis long enough to give every layer fragment at least one index are split along
    # the longest such axis (vocab for both the embedding table (vocab, embed) and the output head (embed, vocab)
    # when vocab is unsharded). Slicing a sharded axis would make XLA gather the whole leaf on every fragment, so a
    # leaf whose long axes are all sharded stays whole in fragment 0, as do vectors (e.g. norm scales).
    bucketized_leaves = {}
    if getattr(config, "diloco_bucketize_non_scanned", False):
      sharding_leaves = _sharding_leaves(kvs, shardings)
      if not any(isinstance(s, jax.sharding.NamedSharding) for s in sharding_leaves):
        max_logging.warning(
            "DiLoCo bucketization found no NamedSharding for any parameter (the given `shardings` are all None, or"
            " none were given and the leaves are tracers, host arrays or not placed with a NamedSharding); every axis"
            " is treated as unsharded, so a sharded axis may be split."
        )
      for (keypath, v), sharding in zip(kvs, sharding_leaves):
        keystr = jax.tree_util.keystr(keypath)
        shape = tuple(getattr(v, "shape", ()))
        if keypath_to_is_scanned[keystr] or len(shape) < 2:
          continue
        long_axes = [a for a in range(len(shape)) if shape[a] >= num_transformer_fragments]
        shard_counts = _axis_shard_counts(shape, sharding)
        candidates = [a for a in long_axes if shard_counts[a] == 1]
        if candidates:
          axis = max(candidates, key=shape.__getitem__)
          bucketized_leaves[keystr] = BucketSpec(*divmod(shape[axis], num_transformer_fragments), axis=axis)
          max_logging.log(f"DiLoCo: bucketizing {keystr} {shape} -> {bucketized_leaves[keystr]}")
        elif long_axes:
          max_logging.log(
              f"DiLoCo: not bucketizing {keystr} {shape}: every axis with >= {num_transformer_fragments} indices is"
              f" sharded (shards per axis {shard_counts}); it stays whole in fragment 0."
          )

    return cls(
        keypath_to_is_scanned=keypath_to_is_scanned,
        fragment_to_layer_indices=fragment_to_layer_indices,
        num_fragments=num_fragments,
        param_scan_axis=param_scan_axis,
        leaf_keystrs=leaf_keystrs,
        bucketized_leaves=bucketized_leaves,
    )

  def _iter_keyed_leaves(self, tree: Any) -> Iterator[tuple[str, Any]]:
    """Yields `(keystr, leaf)`, reusing the cached key strings when the tree matches the construction tree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == len(self.leaf_keystrs):
      yield from zip(self.leaf_keystrs, leaves)
    else:
      for keypath, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        yield jax.tree_util.keystr(keypath), leaf

  def _scan_axis(self, leaf: Any, has_replica_dim: bool) -> int:
    return self.param_scan_axis + 1 if has_replica_dim and leaf.ndim > self.param_scan_axis + 1 else self.param_scan_axis

  def _bucket_rows(self, keystr: str, fragment_idx: int) -> tuple[int, int] | None:
    """Returns `(start, size)` along the split axis of bucketized leaf `keystr` carried by `fragment_idx`, if any."""
    spec = self.bucketized_leaves[keystr]
    if fragment_idx > 0:
      return (fragment_idx - 1) * spec.chunk_size, spec.chunk_size
    if spec.remainder:
      return self.num_layer_fragments * spec.chunk_size, spec.remainder
    return None

  def _bucket_axis(self, keystr: str, has_replica_dim: bool) -> int:
    return self.bucketized_leaves[keystr].axis + (1 if has_replica_dim else 0)

  def _layer_indices(self, fragment_idx: int) -> tuple[tuple[int, ...], bool]:
    """Returns the layer indices of `fragment_idx` and whether they are contiguous."""
    layer_indices = tuple(int(x) for x in self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,)))
    is_contiguous = len(layer_indices) > 0 and (
        list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
    )
    return layer_indices, is_contiguous

  def _check_fragment_idx(self, fragment_idx: int) -> None:
    if not 0 <= fragment_idx < self.num_fragments:
      raise ValueError(f"fragment_idx ({fragment_idx}) must be in [0, {self.num_fragments}).")

  def get_flat_fragment(self, tree: Any, fragment_idx: int, has_replica_dim: bool = False) -> dict[str, Any]:
    """Extracts a flat dictionary containing parameters for the specified fragment index.

    Keys are leaf key strings; row slices of bucketized leaves use `BUCKET_KEY_SUFFIX` (layer fragments) or
    `REMAINDER_KEY_SUFFIX` (fragment 0).
    """
    self._check_fragment_idx(fragment_idx)
    flat_frag = {}
    layer_indices, is_contiguous = self._layer_indices(fragment_idx) if fragment_idx > 0 else ((), False)
    for keystr, v in self._iter_keyed_leaves(tree):
      if self.keypath_to_is_scanned.get(keystr, False):
        if fragment_idx == 0:
          continue
        axis = self._scan_axis(v, has_replica_dim)
        if isinstance(v, jax.ShapeDtypeStruct) or is_contiguous:
          flat_frag[keystr] = _slice_axis(v, axis, layer_indices[0], len(layer_indices))
        else:
          flat_frag[keystr] = jnp.take(v, np.array(layer_indices, dtype=np.int32), axis=axis)
      elif keystr in self.bucketized_leaves:
        rows = self._bucket_rows(keystr, fragment_idx)
        if rows is not None:
          suffix = BUCKET_KEY_SUFFIX if fragment_idx > 0 else REMAINDER_KEY_SUFFIX
          flat_frag[keystr + suffix] = _slice_axis(v, self._bucket_axis(keystr, has_replica_dim), *rows)
      elif fragment_idx == 0:
        flat_frag[keystr] = v
    return flat_frag

  def apply_flat_fragment(
      self,
      tree: Any,
      fragment_idx: int,
      flat_fragment: dict[str, Any],
      has_replica_dim: bool = False,
  ) -> Any:
    """Merges a flat fragment dictionary (as produced by `get_flat_fragment`) back into the full PyTree.

    Slices written into scanned or bucketized leaves are cast to the dtype of the leaf they are written into.
    """
    self._check_fragment_idx(fragment_idx)
    _, treedef = jax.tree_util.tree_flatten(tree)
    layer_indices, is_contiguous = self._layer_indices(fragment_idx) if fragment_idx > 0 else ((), False)
    new_leaves = []
    for keystr, v in self._iter_keyed_leaves(tree):
      if self.keypath_to_is_scanned.get(keystr, False):
        if fragment_idx > 0 and keystr in flat_fragment and not isinstance(v, jax.ShapeDtypeStruct):
          axis = self._scan_axis(v, has_replica_dim)
          update = jnp.asarray(flat_fragment[keystr], dtype=v.dtype)
          if is_contiguous:
            v = jax.lax.dynamic_update_slice_in_dim(v, update, layer_indices[0], axis=axis)
          else:
            index = tuple(slice(None) if i != axis else np.array(layer_indices, dtype=np.int32) for i in range(v.ndim))
            v = v.at[index].set(update)
      elif keystr in self.bucketized_leaves:
        suffix = BUCKET_KEY_SUFFIX if fragment_idx > 0 else REMAINDER_KEY_SUFFIX
        rows = self._bucket_rows(keystr, fragment_idx)
        if rows is not None and keystr + suffix in flat_fragment and not isinstance(v, jax.ShapeDtypeStruct):
          axis = self._bucket_axis(keystr, has_replica_dim)
          update = jnp.asarray(flat_fragment[keystr + suffix], dtype=v.dtype)
          v = jax.lax.dynamic_update_slice_in_dim(v, update, rows[0], axis=axis)
      elif fragment_idx == 0 and keystr in flat_fragment:
        v = flat_fragment[keystr]
      new_leaves.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def get_streaming_schedule(config: Any) -> tuple[int, int]:
  """Computes steps_between_syncs and synchronization period for streaming DiLoCo."""
  num_fragments = config.num_diloco_fragments
  steps_between_syncs = int(round(config.diloco_sync_period / num_fragments))
  steps_between_syncs = max(1, steps_between_syncs)
  period = num_fragments * steps_between_syncs
  return steps_between_syncs, period
