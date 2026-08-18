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

"""FragmentedTreeManipulator for parameter tree slicing."""

import functools
import re
from typing import Any
import jax
from jax.experimental.layout import Format, Layout
import jax.numpy as jnp
import numpy as np


def _get_tree_mesh(tree):
  """Finds the mesh corresponding to tree leaves, creating a single-device mesh if needed."""
  for leaf in jax.tree_util.tree_leaves(tree):
    if isinstance(leaf, jax.Array):
      sharding = getattr(leaf, "sharding", None)
      if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding.mesh
      elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
        dev = list(sharding.device_set)[0]
        return jax.sharding.Mesh(np.array([dev]).reshape(1, 1), ("diloco", "model"))
  return None




class FragmentedTreeManipulator:
  """Partitions and manipulates fragments of a JAX PyTree, supporting scanned layers."""

  def __init__(
      self,
      keypath_to_is_scanned: dict[str, bool],
      fragment_to_layer_indices: dict[int, jax.Array],
      num_fragments: int,
      param_scan_axis: int = 0,
      keypath_to_layout: dict[str, Any] = None,
      leaf_keystrs: list[str] = None,
      fragment_to_leaf_indices: dict[int, list[int]] = None,
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments
    self.param_scan_axis = param_scan_axis
    self.keypath_to_layout = keypath_to_layout or {}

    if leaf_keystrs is not None:
      self.leaf_keystrs = list(leaf_keystrs)
    else:
      self.leaf_keystrs = list(keypath_to_is_scanned.keys())

    self.keystr_to_leaf_index = {k: i for i, k in enumerate(self.leaf_keystrs)}

    if fragment_to_leaf_indices is not None:
      self.fragment_to_leaf_indices = fragment_to_leaf_indices
    else:
      self.fragment_to_leaf_indices = {}
      for f in range(num_fragments):
        if f == 0:
          self.fragment_to_leaf_indices[0] = [
              i for i, k in enumerate(self.leaf_keystrs) if not self.keypath_to_is_scanned.get(k, False)
          ]
        else:
          self.fragment_to_leaf_indices[f] = [
              i for i, k in enumerate(self.leaf_keystrs) if self.keypath_to_is_scanned.get(k, False)
          ]

  @classmethod
  def create(cls, params_tree, config):
    """Creates a FragmentedTreeManipulator from the parameters PyTree and configuration."""
    kvs, _ = jax.tree_util.tree_flatten_with_path(params_tree)

    num_layers = config.num_decoder_layers
    num_transformer_fragments = config.num_diloco_fragments

    # If user provided total fragments (e.g. 37 = 1 non-scanned + 36 layer fragments)
    if num_transformer_fragments == num_layers + 1:
      num_transformer_fragments = num_layers

    assert num_layers % num_transformer_fragments == 0, (
        f"num_decoder_layers ({num_layers}) must be divisible by "
        f"num_diloco_fragments ({num_transformer_fragments}) for now."
    )

    num_synced = num_layers // num_transformer_fragments
    use_sequential = config.use_sequential_layers
    num_fragments = 1 + num_transformer_fragments

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
    keypath_to_layout = {}
    leaf_keystrs = []

    for keypath, leaf in kvs:
      parts = []
      for k in keypath:
        parts.append(str(k.key) if hasattr(k, "key") else (str(k.idx) if hasattr(k, "idx") else str(k)))
      serialized_path = "/" + "/".join(parts)
      keystr = jax.tree_util.keystr(keypath)
      leaf_keystrs.append(keystr)
      keypath_to_is_scanned[keystr] = bool(scanned_regex.search(serialized_path))
      keypath_to_layout[keystr] = getattr(getattr(leaf, "format", None), "layout", None)

    fragment_to_leaf_indices = {}
    for f in range(num_fragments):
      if f == 0:
        fragment_to_leaf_indices[0] = [
            i for i, k in enumerate(leaf_keystrs) if not keypath_to_is_scanned.get(k, False)
        ]
      else:
        fragment_to_leaf_indices[f] = [
            i for i, k in enumerate(leaf_keystrs) if keypath_to_is_scanned.get(k, False)
        ]

    return cls(
        keypath_to_is_scanned,
        fragment_to_layer_indices,
        num_fragments,
        config.param_scan_axis,
        keypath_to_layout=keypath_to_layout,
        leaf_keystrs=leaf_keystrs,
        fragment_to_leaf_indices=fragment_to_leaf_indices,
    )

  def get_flat_fragment(
      self, tree, fragment_idx: int, has_replica_dim: bool = False, **kwargs
  ) -> dict[str, Any]:
    """Extracts a flat dictionary containing parameters for the specified fragment index.

    Args:
      tree: The full parameter PyTree to extract from.
      fragment_idx: Which fragment to extract (0 = non-scanned, >0 = scanned layer slice).
      has_replica_dim: Whether the tree has an extra leading replica dimension.
      **kwargs: Ignored legacy compatibility arguments (e.g. use_null_layout_jit).
    """
    flat_frag = {}
    leaves = jax.tree_util.tree_leaves(tree)

    if len(leaves) == len(self.leaf_keystrs):
      leaf_indices = sorted(self.fragment_to_leaf_indices.get(fragment_idx, []), key=lambda i: self.leaf_keystrs[i])
      if fragment_idx == 0:
        for idx in leaf_indices:
          flat_frag[self.leaf_keystrs[idx]] = leaves[idx]
        return flat_frag

      # Scanned fragment (fragment_idx > 0)
      raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
      if isinstance(raw_indices, (list, tuple)):
        layer_indices = tuple(int(x) for x in raw_indices)
      else:
        layer_indices = tuple(int(x) for x in np.asarray(raw_indices, dtype=np.int32).tolist())

      is_contiguous = len(layer_indices) > 0 and (
          list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
      )

      for idx in leaf_indices:
        keystr = self.leaf_keystrs[idx]
        v = leaves[idx]
        axis = (
            self.param_scan_axis + 1
            if has_replica_dim and v.ndim > self.param_scan_axis + 1
            else self.param_scan_axis
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

    # Defensive fallback if tree has unexpected structure
    kvs = jax.tree_util.tree_flatten_with_path(tree)[0]
    for keypath, v in sorted(kvs, key=lambda kv: jax.tree_util.keystr(kv[0])):
      keystr = jax.tree_util.keystr(keypath)
      is_scanned = self.keypath_to_is_scanned.get(keystr, False)
      if fragment_idx == 0:
        if not is_scanned:
          flat_frag[keystr] = v
      else:
        if is_scanned:
          raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
          if isinstance(raw_indices, (list, tuple)):
            layer_indices = tuple(int(x) for x in raw_indices)
          else:
            layer_indices = tuple(int(x) for x in np.asarray(raw_indices, dtype=np.int32).tolist())
          is_contiguous = len(layer_indices) > 0 and (
              list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
          )
          axis = (
              self.param_scan_axis + 1
              if has_replica_dim and v.ndim > self.param_scan_axis + 1
              else self.param_scan_axis
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

  def dynamic_extract_scanned_fragment(
      self, tree, layer_idx: jax.Array, has_replica_dim: bool = False
  ) -> dict[str, Any]:
    """Extracts a scanned layer fragment using a dynamic slice in XLA (single JIT for all layer fragments)."""
    flat_frag = {}
    leaves = jax.tree_util.tree_leaves(tree)
    leaf_indices = self.fragment_to_leaf_indices.get(1, [])
    raw_indices = self.fragment_to_layer_indices.get(1, (0,))
    if hasattr(raw_indices, "shape"):
      slice_len = int(raw_indices.shape[0]) if len(raw_indices.shape) > 0 else 1
    elif isinstance(raw_indices, (list, tuple)):
      slice_len = len(raw_indices)
    else:
      slice_len = 1
    start_idx = layer_idx * slice_len

    if len(leaves) == len(self.leaf_keystrs):
      for idx in sorted(leaf_indices, key=lambda i: self.leaf_keystrs[i]):
        keystr = self.leaf_keystrs[idx]
        v = leaves[idx]
        axis = (
            self.param_scan_axis + 1
            if has_replica_dim and v.ndim > self.param_scan_axis + 1
            else self.param_scan_axis
        )
        if isinstance(v, jax.ShapeDtypeStruct):
          new_shape = list(v.shape)
          new_shape[axis] = slice_len
          shd = getattr(v, "sharding", None)
          flat_frag[keystr] = jax.ShapeDtypeStruct(tuple(new_shape), v.dtype, sharding=shd)
        else:
          flat_frag[keystr] = jax.lax.dynamic_slice_in_dim(v, start_idx, slice_len, axis=axis)
      return flat_frag

    kvs = jax.tree_util.tree_flatten_with_path(tree)[0]
    for keypath, v in sorted(kvs, key=lambda kv: jax.tree_util.keystr(kv[0])):
      keystr = jax.tree_util.keystr(keypath)
      if self.keypath_to_is_scanned.get(keystr, False):
        axis = (
            self.param_scan_axis + 1
            if has_replica_dim and v.ndim > self.param_scan_axis + 1
            else self.param_scan_axis
        )
        if isinstance(v, jax.ShapeDtypeStruct):
          new_shape = list(v.shape)
          new_shape[axis] = slice_len
          shd = getattr(v, "sharding", None)
          flat_frag[keystr] = jax.ShapeDtypeStruct(tuple(new_shape), v.dtype, sharding=shd)
        else:
          flat_frag[keystr] = jax.lax.dynamic_slice_in_dim(v, start_idx, slice_len, axis=axis)
    return flat_frag

  def dynamic_apply_scanned_fragment(
      self,
      tree,
      layer_idx: jax.Array,
      flat_fragment: dict[str, Any],
      has_replica_dim: bool = False,
  ):
    """Merges a flat scanned fragment back into the parameter PyTree using dynamic slice updates (single JIT)."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    new_leaves = list(leaves)
    raw_indices = self.fragment_to_layer_indices.get(1, (0,))
    if hasattr(raw_indices, "shape"):
      slice_len = int(raw_indices.shape[0]) if len(raw_indices.shape) > 0 else 1
    elif isinstance(raw_indices, (list, tuple)):
      slice_len = len(raw_indices)
    else:
      slice_len = 1
    start_idx = layer_idx * slice_len

    if len(leaves) == len(self.leaf_keystrs):
      for keystr in sorted(flat_fragment.keys()):
        idx = self.keystr_to_leaf_index.get(keystr)
        if idx is None:
          continue
        v = leaves[idx]
        frag = flat_fragment[keystr]
        if isinstance(v, jax.ShapeDtypeStruct):
          new_leaves[idx] = v
          continue
        axis = (
            self.param_scan_axis + 1
            if has_replica_dim and v.ndim > self.param_scan_axis + 1
            else self.param_scan_axis
        )
        new_leaves[idx] = jax.lax.dynamic_update_slice_in_dim(v, frag, start_idx, axis=axis)
      return jax.tree_util.tree_unflatten(treedef, new_leaves)

    kvs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    new_kvs = []
    for k, v in kvs:
      keystr = jax.tree_util.keystr(k)
      if self.keypath_to_is_scanned.get(keystr, False) and keystr in flat_fragment:
        frag = flat_fragment[keystr]
        if isinstance(v, jax.ShapeDtypeStruct):
          new_kvs.append(v)
        else:
          axis = (
              self.param_scan_axis + 1
              if has_replica_dim and v.ndim > self.param_scan_axis + 1
              else self.param_scan_axis
          )
          new_kvs.append(jax.lax.dynamic_update_slice_in_dim(v, frag, start_idx, axis=axis))
      else:
        new_kvs.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_kvs)


  def apply_flat_fragment(
      self,
      tree,
      fragment_idx: int,
      flat_fragment: dict[str, Any],
      has_replica_dim: bool = False,
      **kwargs,
  ):
    """Merges a flat fragment dictionary back into the full parameters PyTree structure.

    Args:
      tree: The full parameter PyTree to update.
      fragment_idx: Which fragment to update (0 = non-scanned, >0 = scanned layer slice).
      flat_fragment: The fragment values to merge in.
      has_replica_dim: Whether the tree has an extra leading replica dimension.
      **kwargs: Ignored legacy compatibility arguments.
    """
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

      # fragment_idx > 0 (scanned parameters)
      raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
      if isinstance(raw_indices, (list, tuple)):
        layer_indices = tuple(int(x) for x in raw_indices)
      else:
        layer_indices = tuple(int(x) for x in np.asarray(raw_indices, dtype=np.int32).tolist())
      is_contiguous = len(layer_indices) > 0 and (
          list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
      )

      for idx in leaf_indices:
        keystr = self.leaf_keystrs[idx]
        if keystr not in flat_fragment:
          continue
        v = leaves[idx]
        frag = flat_fragment[keystr]

        if isinstance(v, jax.ShapeDtypeStruct):
          new_leaves[idx] = v
          continue

        axis = (
            self.param_scan_axis + 1
            if has_replica_dim and v.ndim > self.param_scan_axis + 1
            else self.param_scan_axis
        )
        if is_contiguous:
          start = layer_indices[0]
          end = layer_indices[-1] + 1
          if start == 0 and end == v.shape[axis]:
            new_leaves[idx] = frag
          else:
            parts = []
            if start > 0:
              slc_pre = [slice(None)] * v.ndim
              slc_pre[axis] = slice(0, start)
              parts.append(v[tuple(slc_pre)])
            parts.append(frag)
            if end < v.shape[axis]:
              slc_post = [slice(None)] * v.ndim
              slc_post[axis] = slice(end, v.shape[axis])
              parts.append(v[tuple(slc_post)])
            new_leaves[idx] = jnp.concatenate(parts, axis=axis)
        else:
          idx_tuple = tuple(
              slice(None) if i != axis else np.array(layer_indices, dtype=np.int32) for i in range(v.ndim)
          )
          new_leaves[idx] = v.at[idx_tuple].set(frag)

      return jax.tree_util.tree_unflatten(treedef, new_leaves)

    # Defensive fallback if tree has unexpected structure
    kvs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    new_kvs = []
    raw_indices = self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,))
    if isinstance(raw_indices, (list, tuple)):
      layer_indices = tuple(int(x) for x in raw_indices)
    else:
      layer_indices = tuple(int(x) for x in np.asarray(raw_indices, dtype=np.int32).tolist())
    is_contiguous = len(layer_indices) > 0 and (
        list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
    )

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
          frag = flat_fragment[keystr]
          if isinstance(v, jax.ShapeDtypeStruct):
            new_kvs.append(v)
          else:
            axis = (
                self.param_scan_axis + 1
                if has_replica_dim and v.ndim > self.param_scan_axis + 1
                else self.param_scan_axis
            )
            if is_contiguous:
              start = layer_indices[0]
              end = layer_indices[-1] + 1
              if start == 0 and end == v.shape[axis]:
                new_kvs.append(frag)
              else:
                parts = []
                if start > 0:
                  slc_pre = [slice(None)] * v.ndim
                  slc_pre[axis] = slice(0, start)
                  parts.append(v[tuple(slc_pre)])
                parts.append(frag)
                if end < v.shape[axis]:
                  slc_post = [slice(None)] * v.ndim
                  slc_post[axis] = slice(end, v.shape[axis])
                  parts.append(v[tuple(slc_post)])
                new_kvs.append(jnp.concatenate(parts, axis=axis))
            else:
              idx_tuple = tuple(
                  slice(None) if i != axis else np.array(layer_indices, dtype=np.int32) for i in range(v.ndim)
              )
              new_kvs.append(v.at[idx_tuple].set(frag))
        else:
          new_kvs.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_kvs)

  def get_leaves_for_fragment(self, tree, fragment_idx: int) -> dict[str, Any]:
    """Returns full leaf arrays for all parameters modified by fragment_idx."""
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == len(self.leaf_keystrs):
      leaf_indices = self.fragment_to_leaf_indices.get(fragment_idx, [])
      return {self.leaf_keystrs[idx]: leaves[idx] for idx in leaf_indices}

    leaves_dict = {}
    kvs = jax.tree_util.tree_leaves_with_path(tree)
    for keypath, val in kvs:
      keystr = jax.tree_util.keystr(keypath)
      is_scanned = self.keypath_to_is_scanned.get(keystr, False)
      if (fragment_idx == 0 and not is_scanned) or (fragment_idx > 0 and is_scanned):
        leaves_dict[keystr] = val
    return leaves_dict

  def replace_leaves_from_dict(self, tree, leaf_dict: dict[str, Any]):
    """Replaces leaves in tree matching keys in leaf_dict."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if len(leaves) == len(self.leaf_keystrs):
      new_leaves = list(leaves)
      for keystr, new_val in leaf_dict.items():
        idx = self.keystr_to_leaf_index.get(keystr)
        if idx is not None and idx < len(new_leaves):
          new_leaves[idx] = new_val
      return jax.tree_util.tree_unflatten(treedef, new_leaves)

    flat_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    new_leaves = []
    for keypath, val in flat_leaves:
      keystr = jax.tree_util.keystr(keypath)
      if keystr in leaf_dict:
        new_leaves.append(leaf_dict[keystr])
      else:
        new_leaves.append(val)
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
