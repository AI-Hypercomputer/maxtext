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
"""

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class FragmentedTreeManipulator:
  """For Streaming DiLoCo: Partitions and manipulates fragments of a JAX PyTree, supporting scanned layers."""

  def __init__(
      self,
      keypath_to_is_scanned: dict[str, bool],
      fragment_to_layer_indices: dict[int, tuple[int, ...]],
      num_fragments: int,
      param_scan_axis: int = 0,
      leaf_keystrs: list[str] | None = None,
      fragment_to_leaf_indices: dict[int, list[int]] | None = None,
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments
    self.param_scan_axis = param_scan_axis
    self.leaf_keystrs = leaf_keystrs or []
    self.fragment_to_leaf_indices = fragment_to_leaf_indices or {}

  @classmethod
  def create(cls, params_tree: Any, config: Any) -> "FragmentedTreeManipulator":
    """Creates a FragmentedTreeManipulator from the parameters PyTree and configuration."""
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

    fragment_to_leaf_indices = {}
    for f in range(num_fragments):
      if f == 0:
        fragment_to_leaf_indices[0] = [i for i, k in enumerate(leaf_keystrs) if not keypath_to_is_scanned.get(k, False)]
      else:
        fragment_to_leaf_indices[f] = [i for i, k in enumerate(leaf_keystrs) if keypath_to_is_scanned.get(k, False)]

    return cls(
        keypath_to_is_scanned=keypath_to_is_scanned,
        fragment_to_layer_indices=fragment_to_layer_indices,
        num_fragments=num_fragments,
        param_scan_axis=param_scan_axis,
        leaf_keystrs=leaf_keystrs,
        fragment_to_leaf_indices=fragment_to_leaf_indices,
    )

  def get_flat_fragment(self, tree: Any, fragment_idx: int, has_replica_dim: bool = False) -> dict[str, Any]:
    """Extracts a flat dictionary containing parameters for the specified fragment index."""
    flat_frag = {}
    leaves = jax.tree_util.tree_leaves(tree)

    if len(leaves) == len(self.leaf_keystrs):
      leaf_indices = self.fragment_to_leaf_indices.get(fragment_idx, [])
      if fragment_idx == 0:
        for idx in leaf_indices:
          flat_frag[self.leaf_keystrs[idx]] = leaves[idx]
        return flat_frag

      raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
      layer_indices = tuple(int(x) for x in raw_indices)
      is_contiguous = len(layer_indices) > 0 and (
          list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
      )

      for idx in leaf_indices:
        keystr = self.leaf_keystrs[idx]
        v = leaves[idx]
        axis = self.param_scan_axis + 1 if has_replica_dim and v.ndim > self.param_scan_axis + 1 else self.param_scan_axis
        if isinstance(v, jax.ShapeDtypeStruct):
          new_shape = list(v.shape)
          new_shape[axis] = len(layer_indices)
          shd = getattr(v, "sharding", None)
          flat_frag[keystr] = jax.ShapeDtypeStruct(tuple(new_shape), v.dtype, sharding=shd)
        elif is_contiguous:
          slc = [slice(None)] * v.ndim
          slc[axis] = slice(layer_indices[0], layer_indices[-1] + 1)
          flat_frag[keystr] = v[tuple(slc)]
        else:
          flat_frag[keystr] = jnp.take(v, np.array(layer_indices, dtype=np.int32), axis=axis)
      return flat_frag

    # Fallback if tree structure does not match leaf count
    kvs, _ = jax.tree_util.tree_flatten_with_path(tree)
    for k, v in kvs:
      keystr = jax.tree_util.keystr(k)
      is_scanned = self.keypath_to_is_scanned.get(keystr, False)
      if fragment_idx == 0:
        if not is_scanned:
          flat_frag[keystr] = v
      else:
        if is_scanned:
          raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
          layer_indices = tuple(int(x) for x in raw_indices)
          is_contiguous = len(layer_indices) > 0 and (
              list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
          )
          axis = (
              self.param_scan_axis + 1 if has_replica_dim and v.ndim > self.param_scan_axis + 1 else self.param_scan_axis
          )
          if isinstance(v, jax.ShapeDtypeStruct):
            new_shape = list(v.shape)
            new_shape[axis] = len(layer_indices)
            shd = getattr(v, "sharding", None)
            flat_frag[keystr] = jax.ShapeDtypeStruct(tuple(new_shape), v.dtype, sharding=shd)
          elif is_contiguous:
            slc = [slice(None)] * v.ndim
            slc[axis] = slice(layer_indices[0], layer_indices[-1] + 1)
            flat_frag[keystr] = v[tuple(slc)]
          else:
            flat_frag[keystr] = jnp.take(v, np.array(layer_indices, dtype=np.int32), axis=axis)
    return flat_frag

  def apply_flat_fragment(
      self,
      tree: Any,
      fragment_idx: int,
      flat_fragment: dict[str, Any],
      has_replica_dim: bool = False,
  ) -> Any:
    """Merges a flat fragment dictionary back into the full parameters PyTree structure."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    new_leaves = list(leaves)

    if len(leaves) == len(self.leaf_keystrs):
      leaf_indices = self.fragment_to_leaf_indices.get(fragment_idx, [])
      if fragment_idx == 0:
        for idx in leaf_indices:
          keystr = self.leaf_keystrs[idx]
          if keystr in flat_fragment:
            new_leaves[idx] = flat_fragment[keystr]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

      raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
      layer_indices = tuple(int(x) for x in raw_indices)
      is_contiguous = len(layer_indices) > 0 and (
          list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
      )

      for idx in leaf_indices:
        keystr = self.leaf_keystrs[idx]
        if keystr not in flat_fragment:
          continue
        v = leaves[idx]
        axis = self.param_scan_axis + 1 if has_replica_dim and v.ndim > self.param_scan_axis + 1 else self.param_scan_axis
        if isinstance(v, jax.ShapeDtypeStruct):
          continue
        if is_contiguous:
          slc = [slice(None)] * v.ndim
          slc[axis] = slice(layer_indices[0], layer_indices[-1] + 1)
          new_leaves[idx] = v.at[tuple(slc)].set(flat_fragment[keystr])
        else:
          new_leaves[idx] = v.at[
              tuple(slice(None) if i != axis else np.array(layer_indices, dtype=np.int32) for i in range(v.ndim))
          ].set(flat_fragment[keystr])
      return jax.tree_util.tree_unflatten(treedef, new_leaves)

    # Fallback if tree structure does not match leaf count
    kvs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    new_kvs = []
    for k, v in kvs:
      keystr = jax.tree_util.keystr(k)
      is_scanned = self.keypath_to_is_scanned.get(keystr, False)
      if fragment_idx == 0:
        if not is_scanned and keystr in flat_fragment:
          new_kvs.append(flat_fragment[keystr])
        else:
          new_kvs.append(v)
      else:
        if is_scanned and keystr in flat_fragment:
          raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
          layer_indices = tuple(int(x) for x in raw_indices)
          is_contiguous = len(layer_indices) > 0 and (
              list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
          )
          axis = (
              self.param_scan_axis + 1 if has_replica_dim and v.ndim > self.param_scan_axis + 1 else self.param_scan_axis
          )
          if isinstance(v, jax.ShapeDtypeStruct):
            new_v = v
          elif is_contiguous:
            slc = [slice(None)] * v.ndim
            slc[axis] = slice(layer_indices[0], layer_indices[-1] + 1)
            new_v = v.at[tuple(slc)].set(flat_fragment[keystr])
          else:
            new_v = v.at[
                tuple(slice(None) if i != axis else np.array(layer_indices, dtype=np.int32) for i in range(v.ndim))
            ].set(flat_fragment[keystr])
          new_kvs.append(new_v)
        else:
          new_kvs.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_kvs)


def get_streaming_schedule(config: Any) -> tuple[int, int]:
  """Computes steps_between_syncs and synchronization period for streaming DiLoCo."""
  num_fragments = config.num_diloco_fragments
  steps_between_syncs = int(round(config.diloco_sync_period / num_fragments))
  steps_between_syncs = max(1, steps_between_syncs)
  period = num_fragments * steps_between_syncs
  return steps_between_syncs, period
