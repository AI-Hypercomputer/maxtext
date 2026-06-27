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

"""Streaming DiLoCo fragment manipulation and synchronization utilities for SPMD."""

import re
from typing import Any

import drjax
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.train_state_nnx import TrainStateNNX
import optax


class FragmentedTreeManipulator:
  """For Streaming DiLoCo: Partitions and manipulates fragments of a JAX PyTree, supporting scanned layers."""

  def __init__(
      self,
      keypath_to_is_scanned: dict[str, bool],
      fragment_to_layer_indices: dict[int, jax.Array],
      num_fragments: int,
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments

  @classmethod
  def create(cls, params_tree, config):
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

    for keypath, v in kvs:
      parts = []
      for k in keypath:
        parts.append(str(k.key) if hasattr(k, "key") else (str(k.idx) if hasattr(k, "idx") else str(k)))
      serialized_path = "/" + "/".join(parts)
      is_scanned = (
          bool(scanned_regex.search(serialized_path))
          and hasattr(v, "shape")
          and len(v.shape) > 0
          and v.shape[0] == num_layers
      )
      keypath_to_is_scanned[jax.tree_util.keystr(keypath)] = is_scanned

    return cls(keypath_to_is_scanned, fragment_to_layer_indices, num_fragments)

  def get_flat_fragment(self, tree, fragment_idx: int, has_replica_dim: bool = False) -> dict[str, Any]:
    """Extracts a flat dictionary containing parameters for the specified fragment index."""
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
          if isinstance(v, jax.ShapeDtypeStruct):
            new_shape = (v.shape[0], len(indices), *v.shape[2:]) if has_replica_dim else (len(indices), *v.shape[1:])
            flat_frag[keystr] = jax.ShapeDtypeStruct(new_shape, v.dtype)
          elif has_replica_dim:
            flat_frag[keystr] = v[:, indices]  # Slice second dimension (layer axis)
          else:
            flat_frag[keystr] = v[indices]  # Slice first dimension (layer axis)
    return flat_frag

  def apply_flat_fragment(
      self,
      tree,
      fragment_idx: int,
      flat_fragment: dict[str, Any],
      has_replica_dim: bool = False,
  ):
    """Merges a flat fragment dictionary back into the full parameters PyTree structure."""
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
          if isinstance(v, jax.ShapeDtypeStruct):
            new_v = v
          elif has_replica_dim:
            new_v = v.at[:, indices].set(flat_fragment[keystr])
          else:
            new_v = v.at[indices].set(flat_fragment[keystr])
          new_kvs.append(new_v)
        else:
          new_kvs.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_kvs)


def get_streaming_schedule(config) -> tuple[int, int]:
  """Computes steps_between_syncs and synchronization period for streaming DiLoCo."""
  num_fragments = config.num_diloco_fragments
  steps_between_syncs = int(round(config.diloco_sync_period / num_fragments))
  steps_between_syncs = max(1, steps_between_syncs)
  period = num_fragments * steps_between_syncs
  return steps_between_syncs, period


def replace_nnx_model_params_frag(
    s,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    outer_frag_replica: dict[str, Any],
    alpha: float = 0.0,
):
  """Replaces a single parameter fragment in an NNX TrainState with optional alpha interpolation."""
  s_model = s["model"] if hasattr(s, "keys") else s.model
  graphdef, full_params, non_param_state = nnx.split(s_model, nnx.Param, ...)
  full_params_dict = full_params.to_pure_dict()
  if alpha > 0.0:
    inner_frag = manipulator.get_flat_fragment(full_params_dict, frag_idx, has_replica_dim=False)
    merged_frag = jax.tree.map(lambda i, o: alpha * i + (1 - alpha) * o, inner_frag, outer_frag_replica)
  else:
    merged_frag = outer_frag_replica

  new_full_params = manipulator.apply_flat_fragment(full_params_dict, frag_idx, merged_frag, has_replica_dim=False)
  new_model = nnx.merge(graphdef, new_full_params, non_param_state)

  if isinstance(s_model, nnx.State):
    new_model = nnx.state(new_model)
  elif isinstance(s_model, dict):
    new_model = nnx.to_pure_dict(new_model)

  if hasattr(s, "keys"):
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(s)
    new_model_iter = iter(jax.tree_util.tree_leaves(new_model))

    def _is_model_leaf(path):
      if not path:
        return False
      k = path[0]
      return getattr(k, "key", None) == "model" or getattr(k, "name", None) == "model"

    new_leaves = [next(new_model_iter) if _is_model_leaf(p) else leaf for p, leaf in leaves_with_paths]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
  else:
    s_opt = s["optimizer"] if hasattr(s, "keys") else s.optimizer
    return TrainStateNNX(new_model, s_opt)


def synchronize_fragment_state(
    state,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    outer_optimizer: optax.GradientTransformation,
    mesh: jax.sharding.Mesh | None = None,
):
  """Synchronizes a single parameter fragment across DiLoCo replicas in streaming DiLoCo."""
  # 1. Extract global and local parameters for the fragment
  outer_params_frag = manipulator.get_flat_fragment(state.params, frag_idx, has_replica_dim=False)
  inner_model_params = nnx.filter_state(state.inner_state.model, nnx.Param).to_pure_dict()
  inner_params_frag = manipulator.get_flat_fragment(inner_model_params, frag_idx, has_replica_dim=True)

  # 2. Compute the pseudo-gradient: outer - inner
  broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)
  unreduced_grads = jax.tree.map(lambda x, y: x - y, broadcast_outer_frag, inner_params_frag)

  # 3. Average gradients across replicas
  averaged_pseudo_grad = drjax.reduce_mean(unreduced_grads)

  # 4. Extract outer optimizer state for this fragment (TraceState is (trace, EmptyState))
  trace_frag = manipulator.get_flat_fragment(state.outer_opt_state[0].trace, frag_idx, has_replica_dim=False)
  opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

  # 5. Run outer optimizer on the fragment
  updates_frag, new_opt_state_frag = outer_optimizer.update(
      averaged_pseudo_grad, opt_state_frag, params=outer_params_frag
  )
  new_outer_params_frag = optax.apply_updates(outer_params_frag, updates_frag)

  # 6. Re-merge updated params and optimizer states back to full PyTree
  new_params = manipulator.apply_flat_fragment(state.params, frag_idx, new_outer_params_frag, has_replica_dim=False)
  new_trace = manipulator.apply_flat_fragment(
      state.outer_opt_state[0].trace, frag_idx, new_opt_state_frag[0].trace, has_replica_dim=False
  )
  new_outer_opt_state = (optax.TraceState(trace=new_trace), state.outer_opt_state[1])

  return state.replace(
      params=new_params,
      outer_opt_state=new_outer_opt_state,
  )


def apply_fragment_to_inner_state(
    state,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    alpha: float = 0.0,
    mesh: jax.sharding.Mesh | None = None,
):
  """Broadcasts synced outer parameter fragment and updates inner state across replicas."""
  outer_params_frag = manipulator.get_flat_fragment(state.params, frag_idx, has_replica_dim=False)
  broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)

  new_inner_state = drjax.map_fn(
      lambda s, frag: replace_nnx_model_params_frag(s, manipulator, frag_idx, frag, alpha=alpha),
      (state.inner_state, broadcast_outer_frag),
      mesh=mesh,
  )
  return state.replace(inner_state=new_inner_state)
