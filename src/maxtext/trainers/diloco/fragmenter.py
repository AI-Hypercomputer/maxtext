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
import jax.numpy as jnp
from jax.experimental.layout import Format, Layout


class FragmentedTreeManipulator:
  """Partitions and manipulates fragments of a JAX PyTree, supporting scanned layers."""

  def __init__(
      self,
      keypath_to_is_scanned: dict[str, bool],
      fragment_to_layer_indices: dict[int, jax.Array],
      num_fragments: int,
      param_scan_axis: int = 0,
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments
    self.param_scan_axis = param_scan_axis
    # Caches for null-layout-aware jit functions used by the CPU syncer path.
    self._take_jit_cache: dict = {}
    self._scatter_jit_cache: dict = {}

  def _make_take_jit_null(self, v, indices, axis):
    """Returns a cached jit that extracts a fragment from a null-layout CPU array.

    Eager jnp.take on Pathways null-layout arrays compiles jit__take with tiled input
    expectations (XLA always prefers tiled for data-reading ops), which Pathways then
    rejects at the input boundary. Wrapping in a jit with explicit null-layout Format
    forces XLA to compile with null expectations, matching what device_put produces.
    """
    cache_key = (v.shape, v.dtype, str(v.sharding), tuple(int(i) for i in indices), axis)
    if cache_key not in self._take_jit_cache:
      null_layout = Layout(major_to_minor=tuple(range(len(v.shape))), tiling=None)
      in_format = Format(layout=null_layout, sharding=v.sharding)
      out_sharding = jax.sharding.NamedSharding(v.sharding.mesh, v.sharding.spec)
      static_indices = jnp.array([int(i) for i in indices])
      static_axis = axis

      @functools.partial(jax.jit, in_shardings=(in_format,), out_shardings=out_sharding)
      def take_fn(x):
        return jnp.take(x, static_indices, axis=static_axis)

      self._take_jit_cache[cache_key] = take_fn
    return self._take_jit_cache[cache_key]

  def _make_scatter_jit_null(self, v, frag_example, indices, axis):
    """Returns a cached jit that updates a null-layout CPU array with a null-layout fragment."""
    frag_indices = tuple(int(i) for i in indices)
    cache_key = (
        v.shape, v.dtype, str(v.sharding),
        frag_example.shape, frag_example.dtype, str(frag_example.sharding),
        frag_indices, axis,
    )
    if cache_key not in self._scatter_jit_cache:
      null_v_layout = Layout(major_to_minor=tuple(range(len(v.shape))), tiling=None)
      null_f_layout = Layout(major_to_minor=tuple(range(len(frag_example.shape))), tiling=None)
      in_format_v = Format(layout=null_v_layout, sharding=v.sharding)
      in_format_f = Format(layout=null_f_layout, sharding=frag_example.sharding)
      out_sharding = jax.sharding.NamedSharding(v.sharding.mesh, v.sharding.spec)
      static_indices = jnp.array(list(frag_indices))
      static_axis = axis

      @functools.partial(jax.jit, in_shardings=(in_format_v, in_format_f), out_shardings=out_sharding)
      def scatter_fn(full_x, frag_x):
        idx_tuple = tuple(slice(None) if i != static_axis else static_indices for i in range(full_x.ndim))
        return full_x.at[idx_tuple].set(frag_x)

      self._scatter_jit_cache[cache_key] = scatter_fn
    return self._scatter_jit_cache[cache_key]

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
      fragment_to_layer_indices[i] = jnp.array(indices)

    # Regex to identify scanned layer parameters
    scanned_regex = re.compile(r"/(?:layers|blocks|moe_layers|dense_layers|layers_outside_pipeline)(?:/|$)")
    keypath_to_is_scanned = {}

    for keypath, _ in kvs:
      parts = []
      for k in keypath:
        parts.append(str(k.key) if hasattr(k, "key") else (str(k.idx) if hasattr(k, "idx") else str(k)))
      serialized_path = "/" + "/".join(parts)
      keypath_to_is_scanned[jax.tree_util.keystr(keypath)] = bool(scanned_regex.search(serialized_path))

    return cls(keypath_to_is_scanned, fragment_to_layer_indices, num_fragments, config.param_scan_axis)

  def get_flat_fragment(
      self, tree, fragment_idx: int, has_replica_dim: bool = False, use_null_layout_jit: bool = False
  ) -> dict[str, Any]:
    """Extracts a flat dictionary containing parameters for the specified fragment index.

    Args:
      tree: The full parameter PyTree to extract from.
      fragment_idx: Which fragment to extract (0 = non-scanned, >0 = scanned layer slice).
      has_replica_dim: Whether the tree has an extra leading replica dimension.
      use_null_layout_jit: When True, wraps jnp.take in a null-layout-Format jit to avoid
        Pathways layout mismatches. Use for CPU syncer arrays; leave False for TPU learner arrays.
    """
    kvs, _ = jax.tree_util.tree_flatten_with_path(tree)
    flat_frag = {}
    for k, v in kvs:
      keystr = jax.tree_util.keystr(k)
      is_scanned = self.keypath_to_is_scanned.get(keystr, False)
      if fragment_idx == 0:
        if not is_scanned:
          flat_frag[keystr] = v
      else:
        if is_scanned:
          indices = self.fragment_to_layer_indices[fragment_idx]
          axis = self.param_scan_axis + 1 if has_replica_dim else self.param_scan_axis
          if use_null_layout_jit:
            take_fn = self._make_take_jit_null(v, indices, axis)
            flat_frag[keystr] = take_fn(v)
          else:
            flat_frag[keystr] = jnp.take(v, indices, axis=axis)
    return flat_frag

  def apply_flat_fragment(
      self,
      tree,
      fragment_idx: int,
      flat_fragment: dict[str, Any],
      has_replica_dim: bool = False,
      use_null_layout_jit: bool = False,
  ):
    """Merges a flat fragment dictionary back into the full parameters PyTree structure.

    Args:
      tree: The full parameter PyTree to update.
      fragment_idx: Which fragment to update (0 = non-scanned, >0 = scanned layer slice).
      flat_fragment: The fragment values to merge in.
      has_replica_dim: Whether the tree has an extra leading replica dimension.
      use_null_layout_jit: When True, wraps scatter in a null-layout-Format jit to avoid
        Pathways layout mismatches. Use for CPU syncer arrays; leave False for TPU learner arrays.
    """
    kvs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    new_kvs = []
    for k, v in kvs:
      keystr = jax.tree_util.keystr(k)
      is_scanned = self.keypath_to_is_scanned.get(keystr, False)
      if fragment_idx == 0:
        if not is_scanned:
          new_kvs.append(flat_fragment[keystr])
        else:
          new_kvs.append(v)
      else:
        if is_scanned:
          indices = self.fragment_to_layer_indices[fragment_idx]
          axis = self.param_scan_axis + 1 if has_replica_dim else self.param_scan_axis
          frag = flat_fragment[keystr]
          if use_null_layout_jit:
            scatter_fn = self._make_scatter_jit_null(v, frag, indices, axis)
            new_kvs.append(scatter_fn(v, frag))
          else:
            idx_tuple = tuple(slice(None) if i != axis else indices for i in range(v.ndim))
            new_kvs.append(v.at[idx_tuple].set(frag))
        else:
          new_kvs.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_kvs)
