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
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments
    self.param_scan_axis = param_scan_axis
    self.keypath_to_layout = keypath_to_layout or {}
    # Caches for full-fragment JIT extraction and insertion functions.
    self._extract_jit_cache: dict = {}
    self._apply_jit_cache: dict = {}
    self._layout_cast_cache: dict = {}

  def _get_extract_jit_fn(self, fragment_idx: int, has_replica_dim: bool):
    """Returns a cached JIT function that extracts a full flat fragment PyTree in a single graph."""
    cache_key = (fragment_idx, has_replica_dim)
    if cache_key not in self._extract_jit_cache:
      static_frag_idx = fragment_idx
      static_has_replica = has_replica_dim
      if static_frag_idx > 0:
        static_indices = tuple(int(x) for x in self.fragment_to_layer_indices[static_frag_idx])
        is_contiguous = (list(static_indices) == list(range(static_indices[0], static_indices[-1] + 1)))
      else:
        static_indices = ()
        is_contiguous = False

      def extract_fn(t):
        kvs, _ = jax.tree_util.tree_flatten_with_path(t)
        flat_frag = {}
        for k, v in kvs:
          keystr = jax.tree_util.keystr(k)
          is_scanned = self.keypath_to_is_scanned.get(keystr, False)
          if static_frag_idx == 0:
            if not is_scanned:
              flat_frag[keystr] = v
          else:
            if is_scanned:
              axis = self.param_scan_axis + 1 if static_has_replica else self.param_scan_axis
              if is_contiguous:
                slc = [slice(None)] * v.ndim
                slc[axis] = slice(static_indices[0], static_indices[-1] + 1)
                flat_frag[keystr] = v[tuple(slc)]
              else:
                flat_frag[keystr] = jnp.take(v, np.array(static_indices), axis=axis)
        return flat_frag

      self._extract_jit_cache[cache_key] = extract_fn
    return self._extract_jit_cache[cache_key]

  def _get_apply_jit_fn(self, fragment_idx: int, has_replica_dim: bool):
    """Returns a function that merges a flat fragment into a full parameter PyTree."""
    cache_key = (fragment_idx, has_replica_dim)
    if cache_key not in self._apply_jit_cache:
      static_frag_idx = fragment_idx
      static_has_replica = has_replica_dim
      if static_frag_idx > 0:
        static_indices = tuple(int(x) for x in self.fragment_to_layer_indices[static_frag_idx])
        is_contiguous = (list(static_indices) == list(range(static_indices[0], static_indices[-1] + 1)))
      else:
        static_indices = ()
        is_contiguous = False

      def apply_fn(t, flat_fragment):
        kvs, treedef = jax.tree_util.tree_flatten_with_path(t)
        new_kvs = []
        for k, v in kvs:
          keystr = jax.tree_util.keystr(k)
          is_scanned = self.keypath_to_is_scanned.get(keystr, False)
          if static_frag_idx == 0:
            if not is_scanned:
              new_kvs.append(flat_fragment[keystr])
            else:
              new_kvs.append(v)
          else:
            if is_scanned:
              axis = self.param_scan_axis + 1 if static_has_replica else self.param_scan_axis
              frag = flat_fragment[keystr]
              if is_contiguous:
                start = static_indices[0]
                end = static_indices[-1] + 1
                parts = []
                if start > 0:
                  slc_prefix = [slice(None)] * v.ndim
                  slc_prefix[axis] = slice(0, start)
                  parts.append(v[tuple(slc_prefix)])
                parts.append(frag)
                if end < v.shape[axis]:
                  slc_suffix = [slice(None)] * v.ndim
                  slc_suffix[axis] = slice(end, v.shape[axis])
                  parts.append(v[tuple(slc_suffix)])
                new_kvs.append(jnp.concatenate(parts, axis=axis))
              else:
                idx_tuple = tuple(slice(None) if i != axis else np.array(static_indices) for i in range(v.ndim))
                new_kvs.append(v.at[idx_tuple].set(frag))
            else:
              new_kvs.append(v)
        return jax.tree_util.tree_unflatten(treedef, new_kvs)

      self._apply_jit_cache[cache_key] = apply_fn
    return self._apply_jit_cache[cache_key]

  @classmethod
  def create(cls, params_tree, config):
    """Creates a FragmentedTreeManipulator from the parameters PyTree and configuration."""
    kvs, _ = jax.tree_util.tree_flatten_with_path(params_tree)

    num_layers = config.num_decoder_layers
    num_transformer_fragments = config.num_diloco_fragments

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

    for keypath, leaf in kvs:
      parts = []
      for k in keypath:
        parts.append(str(k.key) if hasattr(k, "key") else (str(k.idx) if hasattr(k, "idx") else str(k)))
      serialized_path = "/" + "/".join(parts)
      keystr = jax.tree_util.keystr(keypath)
      keypath_to_is_scanned[keystr] = bool(scanned_regex.search(serialized_path))
      keypath_to_layout[keystr] = getattr(getattr(leaf, "format", None), "layout", None)

    return cls(
        keypath_to_is_scanned,
        fragment_to_layer_indices,
        num_fragments,
        config.param_scan_axis,
        keypath_to_layout=keypath_to_layout,
    )

  def get_flat_fragment(
      self, tree, fragment_idx: int, has_replica_dim: bool = False
  ) -> dict[str, Any]:
    """Extracts a flat dictionary containing parameters for the specified fragment index.

    Args:
      tree: The full parameter PyTree to extract from.
      fragment_idx: Which fragment to extract (0 = non-scanned, >0 = scanned layer slice).
      has_replica_dim: Whether the tree has an extra leading replica dimension.
    """
    extract_fn = self._get_extract_jit_fn(fragment_idx, has_replica_dim)
    tree_mesh = _get_tree_mesh(tree)
    if tree_mesh is not None:
      with jax.set_mesh(tree_mesh):
        return extract_fn(tree)
    return extract_fn(tree)

  def apply_flat_fragment(
      self,
      tree,
      fragment_idx: int,
      flat_fragment: dict[str, Any],
      has_replica_dim: bool = False,
  ):
    """Merges a flat fragment dictionary back into the full parameters PyTree structure.

    Args:
      tree: The full parameter PyTree to update.
      fragment_idx: Which fragment to update (0 = non-scanned, >0 = scanned layer slice).
      flat_fragment: The fragment values to merge in.
      has_replica_dim: Whether the tree has an extra leading replica dimension.
    """
    apply_fn = self._get_apply_jit_fn(fragment_idx, has_replica_dim)
    tree_mesh = _get_tree_mesh(tree)
    if tree_mesh is not None:
      with jax.set_mesh(tree_mesh):
        return apply_fn(tree, flat_fragment)
    return apply_fn(tree, flat_fragment)

  def get_leaves_for_fragment(self, tree, fragment_idx: int) -> dict[str, Any]:
    """Returns full leaf arrays for all parameters modified by fragment_idx."""
    leaves = {}
    kvs = jax.tree_util.tree_leaves_with_path(tree)
    for keypath, val in kvs:
      keystr = jax.tree_util.keystr(keypath)
      is_scanned = self.keypath_to_is_scanned.get(keystr, False)
      if (fragment_idx == 0 and not is_scanned) or (fragment_idx > 0 and is_scanned):
        leaves[keystr] = val
    return leaves

  def replace_leaves_from_dict(self, tree, leaf_dict: dict[str, Any]):
    """Replaces leaves in tree matching keys in leaf_dict."""
    flat_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    new_leaves = []
    for keypath, val in flat_leaves:
      keystr = jax.tree_util.keystr(keypath)
      if keystr in leaf_dict:
        new_leaves.append(leaf_dict[keystr])
      else:
        new_leaves.append(val)
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
