#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""An implementation of Distributed Low-Communication (DiLoCo) training.

This module contains implementations of:

-   DiLoCo: Distributed Low-Communication Training of Language Models
    https://arxiv.org/abs/2311.08105
-   Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch
    https://arxiv.org/abs/2501.18512
"""

from collections.abc import Sequence
import re
from typing import Any, Callable

import drjax
from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int32, Key, PyTree, UInt32
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.configs import pyconfig
import optax

Batch = Any
Params = PyTree
Metrics = PyTree
OptState = optax.OptState
InnerOptStates = optax.OptState
PRNGKey = Key[Array, ""] | UInt32[Array, "2"]
Step = Int32[Array, ""]


class DiLoCoTrainState(struct.PyTreeNode):
  """The state of the DiLoCo training process.

  Attributes:
    inner_state: A `flax.training.train_state.TrainState` of the state for each
      step of the inner optimization.  All arrays are expected to have a leading
      dimension with size of the number of diloco replicas so that training
      steps can be mapped over this dimension.
    params: A PyTree of the global model weights. These will mimic a
      sub-PyTree in `inner_state`, which rank-1 shape.
    outer_opt_state: The state for the outer Nesterov momentum optimizer.
    step: The step counter of the training process.
  """

  inner_state: Any
  params: Params
  outer_opt_state: OptState
  step: Step


def add_diloco_to_sharding(pytree):
  """
  Recursively traverses a PyTree and prepends 'diloco' to the PartitionSpec
  of any NamedSharding object that doesn't have an empty PartitionSpec.
  """

  def map_fn(leaf):
    if isinstance(leaf, jax.sharding.NamedSharding):
      new_spec = jax.sharding.PartitionSpec("diloco", *leaf.spec)
      return jax.sharding.NamedSharding(mesh=leaf.mesh, spec=new_spec)
    return leaf

  return jax.tree_util.tree_map(map_fn, pytree)


def reshape_first_axis_with_diloco(num_diloco_replicas: int, pytree: PyTree) -> PyTree:
  """Reshapes the first dimension of each array in the PyTree to include a DiLoCo axis.

  This function takes a a batch of data represented as a PyTree
  and reshapes the leading dimension of each array within it. The purpose is
  to introduce a new 'diloco' axis, which is used for distributing data
  across DiLoCo replicas.

  Args:
    num_diloco_replicas: The number of DiLoCo replicas. This determines the
      size of the new leading dimension.
    pytree: The input PyTree, where each array is expected to have a batch
      dimension as its first axis.

  Returns:
    A new PyTree with the same structure as the input, but with each array's
    first dimension reshaped to `(num_diloco_replicas, original_batch_dim // num_diloco_replicas, ...)`.
    The sharding specification is also updated to include the 'diloco' axis.
  """

  def extend_pspec(pspec: jax.sharding.PartitionSpec | Sequence[str | Sequence[str]] = ()) -> jax.sharding.PartitionSpec:
    if tuple(*pspec)[0] == "diloco":
      # pull out diloco axis if already present
      return jax.sharding.PartitionSpec("diloco", (*pspec[0][1:],), (*pspec[1:],))
    return jax.sharding.PartitionSpec("diloco", *pspec)

  def reshape_for_diloco(arr):
    batch_dim, *example_shape = arr.shape
    diloco_shape = (num_diloco_replicas, batch_dim // num_diloco_replicas, *example_shape)
    if hasattr(arr, "sharding"):
      s = arr.sharding
      s = jax.sharding.NamedSharding(mesh=s.mesh, spec=extend_pspec(s.spec))
      return jax.lax.with_sharding_constraint(jnp.reshape(arr, shape=diloco_shape), s)
    return jnp.reshape(arr, shape=diloco_shape)

  return jax.tree.map(reshape_for_diloco, pytree)


class FragmentedTreeManipulator:
  """Partitions and manipulates fragments of a JAX PyTree, supporting scanned layers."""

  def __init__(
      self, keypath_to_is_scanned: dict[str, bool], fragment_to_layer_indices: dict[int, jax.Array], num_fragments: int
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments

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
          if has_replica_dim:
            flat_frag[keystr] = v[:, indices]  # Slice second dimension (layer axis)
          else:
            flat_frag[keystr] = v[indices]  # Slice first dimension (layer axis)
    return flat_frag

  def apply_flat_fragment(self, tree, fragment_idx: int, flat_fragment: dict[str, Any], has_replica_dim: bool = False):
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
          if has_replica_dim:
            new_v = v.at[:, indices].set(flat_fragment[keystr])
          else:
            new_v = v.at[indices].set(flat_fragment[keystr])
          new_kvs.append(new_v)
        else:
          new_kvs.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_kvs)


def build_abstract_diloco_state(
    config: "pyconfig.HyperParameters",
    abstract_state: PyTree,
    state_mesh_shardings: PyTree,
    mesh: jax.sharding.Mesh,
) -> tuple[DiLoCoTrainState, DiLoCoTrainState, PyTree]:
  """Build abstract DiLoCo state and shardings for AOT compilation.

  This function creates abstract (shape-only) DiLoCo state suitable for
  ahead-of-time compilation, where we don't have actual arrays.

  Args:
    config: The config used to set up training.
    abstract_state: Abstract train state (ShapeDtypeStruct objects).
    state_mesh_shardings: Shardings for the regular train state.
    mesh: The mesh for sharding.

  Returns:
    A tuple of (abstract_diloco_state, diloco_state_shardings).
  """

  # Create inner state with diloco dimension prepended to all arrays
  def add_diloco_dim(x):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
      new_shape = (config.num_diloco_replicas,) + tuple(x.shape)
      return jax.ShapeDtypeStruct(new_shape, x.dtype)
    return x

  inner_state = jax.tree.map(add_diloco_dim, abstract_state)

  # Create outer optimizer state shape using eval_shape
  outer_optimizer = optax.sgd(
      config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )
  # For NNX, model params (Param variables only) live under abstract_state.model;
  # for Linen under abstract_state.params.
  if config.pure_nnx:
    _, model_params, _ = nnx.split(abstract_state.model, nnx.Param, ...)
    model_params = model_params.to_pure_dict()
    _, model_params_sharding, _ = nnx.split(
        state_mesh_shardings.model, nnx.Param, ...
    )
    model_params_sharding = model_params_sharding.to_pure_dict()
  else:
    model_params = abstract_state.params
    model_params_sharding = state_mesh_shardings.params
  outer_opt_state = jax.eval_shape(outer_optimizer.init, model_params)

  # Create abstract step
  abstract_step = jax.ShapeDtypeStruct((), jnp.int32)

  # Build abstract DiLoCo state
  diloco_state = DiLoCoTrainState(
      inner_state=inner_state,
      params=model_params,
      outer_opt_state=outer_opt_state,
      step=abstract_step,
  )

  # Build shardings
  inner_state_shardings = add_diloco_to_sharding(state_mesh_shardings)
  # Sharding for outer_opt_state. For SGD with momentum, it is (TraceState(trace=...), EmptyState())
  # We shard the momentum trace the same way as the parameters.
  outer_opt_state_sharding = (
      optax.TraceState(trace=model_params_sharding),
      optax.EmptyState(),
  )
  diloco_state_shardings = DiLoCoTrainState(
      inner_state=inner_state_shardings,
      params=model_params_sharding,
      outer_opt_state=outer_opt_state_sharding,
      step=None,
  )

  return diloco_state, diloco_state_shardings, inner_state_shardings


def build_diloco_state(
    config: "pyconfig.HyperParameters",
    initialize_state: Callable[[], Any],
    mesh: jax.sharding.Mesh | None = None,
) -> tuple[DiLoCoTrainState, PyTree]:
  """Given a non-DiLoCo train state, construct a DiLoCo training state."""
  outer_optimizer = optax.sgd(
      config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def init_diloco_state() -> tuple[DiLoCoTrainState, PyTree]:
    state = initialize_state()
    # Inner state must be broadcast across clients.
    # Pass mesh explicitly because jax.set_mesh() uses a different thread-local
    # than pxla.thread_resources (which drjax reads), so drjax cannot find the
    # mesh automatically when jax.set_mesh is used.
    inner_state = drjax.broadcast(state, mesh=mesh)
    # Outer state retains a single copy of the model parameters and optimizer state.
    # For NNX, model params (Param variables only) live under state.model;
    # for Linen under state.params.
    if config.pure_nnx:
      _, outer_params, _ = nnx.split(state.model, nnx.Param, ...)
      outer_params = outer_params.to_pure_dict()
    else:
      outer_params = state.params
    outer_opt_state = outer_optimizer.init(outer_params)
    outer_opt_state_sharding = jax.tree_util.tree_map(lambda x: x.sharding, outer_opt_state)
    # For NNX, the step counter lives at state.optimizer.step; for Linen at state.step.
    step = state.optimizer.step if config.pure_nnx else state.step
    return (
        DiLoCoTrainState(inner_state=inner_state, params=outer_params, outer_opt_state=outer_opt_state, step=step),
        outer_opt_state_sharding,
    )

  return init_diloco_state()


def build_diloco_train_step(
    config: pyconfig.HyperParameters,
    train_step: Callable[[Any, Batch, PRNGKey], tuple[Any, Metrics]],
    mesh: jax.sharding.Mesh | None = None,
) -> Callable[[DiLoCoTrainState, Batch, PRNGKey], tuple[DiLoCoTrainState, Metrics]]:
  """Convert a local state and train step into DiLoCo-compatible versions.

  This is an implementation of the original (non-streaming) DiLoCo algorithm
  which syncs all model parameters across  the replicas every
  `config.diloco_sync_period` steps, treating the difference accumulated over
  non-sync steps as a pseudo gradient and applying SGD with Nesterov momentum on
  the "global" model.

  Args:
    config: The config used to set up training.
    train_step: A local train step. This will be executed independently within
      each replica.
  """
  outer_optimizer = optax.sgd(
      config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )

  def synchronize(state):
    # Calculate the delta between the current replica's state and the global
    # state (since last synchronization).
    broadcast_outer_params = drjax.broadcast(state.params, mesh=mesh)
    # For NNX, model Param vars live under inner_state.model; for Linen under inner_state.params.
    if config.pure_nnx:
      _, inner_model_params, _ = nnx.split(
          state.inner_state.model, nnx.Param, ...
      )
      inner_model_params = inner_model_params.to_pure_dict()
    else:
      inner_model_params = state.inner_state.params
    model_delta = jax.tree.map(lambda x, y: y - x, inner_model_params, broadcast_outer_params)
    # Treat the average delta as the outer optimizer's gradient and apply to
    # the global (outer) model params.
    averaged_pseudo_grad = drjax.reduce_mean(model_delta)
    updates, new_opt_state = outer_optimizer.update(averaged_pseudo_grad, state.outer_opt_state, state.params)
    new_outer_params = optax.apply_updates(state.params, updates)
    # Replace inner model params with the new global model params.
    # NOTE: inner optimizer state is retained despite the change in parameters,
    # see section 6.1 in https://arxiv.org/pdf/2311.08105.
    if config.pure_nnx:
      # For NNX: merge new Param vars back with the non-Param model vars (e.g. RNG state).
      def replace_nnx_model_params(s, new_params):
        s_model = s["model"] if hasattr(s, "keys") else s.model
        s_opt = s["optimizer"] if hasattr(s, "keys") else s.optimizer

        graphdef, _, non_param_state = nnx.split(s_model, nnx.Param, ...)
        new_model = nnx.merge(graphdef, new_params, non_param_state)

        if type(s_model).__name__ == "State":
          new_model = nnx.state(new_model)
        elif isinstance(s_model, dict):
          new_model = nnx.to_pure_dict(new_model)

        if hasattr(s, "keys"):
          # Replace "model" leaves by path, keeping s's treedef. Picking by position
          # (leaves[N:]) breaks if a key sorts before "model"; reconstructing via
          # type(s)({...}) breaks the lax.cond match — nnx.State recursive-wraps.
          leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(s)
          new_model_iter = iter(jax.tree_util.tree_leaves(new_model))

          def _is_model_leaf(path):
            if not path:
              return False
            k = path[0]
            return (
                getattr(k, "key", None) == "model"
                or getattr(k, "name", None) == "model"
            )

          new_leaves = [
              next(new_model_iter) if _is_model_leaf(p) else leaf
              for p, leaf in leaves_with_paths
          ]
          return jax.tree_util.tree_unflatten(treedef, new_leaves)
        else:
          return TrainStateNNX(new_model, s_opt)

      new_inner_state = drjax.map_fn(
          lambda s: replace_nnx_model_params(s, new_outer_params),
          state.inner_state,
          mesh=mesh,
      )
    else:
      new_inner_state = drjax.map_fn(lambda s: s.replace(params=new_outer_params), state.inner_state, mesh=mesh)
    return state.replace(
        params=new_outer_params,
        outer_opt_state=new_opt_state,
        inner_state=new_inner_state,
    )

  def typed_reduce_mean(in_tree):
    total = drjax.reduce_sum(in_tree)
    avg = jax.tree.map(lambda x: (x / config.num_diloco_replicas).astype(x.dtype), total)
    return avg

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def diloco_train_step(state, batch, prng):
    # Broadcast the RNG across replicas.
    broadcast_rng = drjax.broadcast(prng, mesh=mesh)
    inner_state, metrics = drjax.map_fn(train_step, (state.inner_state, batch, broadcast_rng), mesh=mesh)
    avg_metrics = typed_reduce_mean(metrics)
    # For NNX, the step counter lives at inner_state.optimizer.step; for Linen at inner_state.step.
    new_step = inner_state.optimizer.step[0] if config.pure_nnx else inner_state.step[0]
    state = state.replace(
        inner_state=inner_state,
        step=new_step,
    )

    if config.enable_streaming_diloco:
      manipulator = FragmentedTreeManipulator.create(state.params, config)

      num_transformer_fragments = config.num_diloco_fragments
      num_fragments = 1 + num_transformer_fragments

      steps_between_syncs_plus_1 = int(round(config.diloco_sync_period / num_fragments))
      steps_between_syncs_plus_1 = max(1, steps_between_syncs_plus_1)
      period = num_fragments * steps_between_syncs_plus_1

      def synchronize_fragment(state, idx):
        # 1. Extract global and local parameters for the fragment
        outer_params_frag = manipulator.get_flat_fragment(state.params, idx, has_replica_dim=False)
        inner_model_params = (
            nnx.filter_state(state.inner_state.model, nnx.Param) if config.pure_nnx else state.inner_state.params
        )
        inner_params_frag = manipulator.get_flat_fragment(inner_model_params, idx, has_replica_dim=True)

        # 2. Compute the pseudo-gradient: outer - inner
        broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)
        unreduced_grads = jax.tree.map(lambda x, y: x - y, broadcast_outer_frag, inner_params_frag)

        # 3. Average gradients across replicas
        averaged_pseudo_grad = drjax.reduce_mean(unreduced_grads)

        # 4. Extract outer optimizer state for this fragment (TraceState is (trace, EmptyState))
        trace_frag = manipulator.get_flat_fragment(state.outer_opt_state[0].trace, idx, has_replica_dim=False)
        opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

        # 5. Run outer optimizer on the fragment
        updates_frag, new_opt_state_frag = outer_optimizer.update(
            averaged_pseudo_grad, opt_state_frag, params=outer_params_frag
        )
        new_outer_params_frag = optax.apply_updates(outer_params_frag, updates_frag)

        # 6. Re-merge updated params and optimizer states back to full PyTree
        new_params = manipulator.apply_flat_fragment(state.params, idx, new_outer_params_frag, has_replica_dim=False)
        new_trace = manipulator.apply_flat_fragment(
            state.outer_opt_state[0].trace, idx, new_opt_state_frag[0].trace, has_replica_dim=False
        )
        new_outer_opt_state = (optax.TraceState(trace=new_trace), state.outer_opt_state[1])

        return state.replace(
            params=new_params,
            outer_opt_state=new_outer_opt_state,
        )

      def apply_fragment(state, idx):
        # Get synced global params fragment
        outer_params_frag = manipulator.get_flat_fragment(state.params, idx, has_replica_dim=False)

        # Broadcast to replicas
        broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)

        # Interpolation functions per replica
        def replace_nnx_model_params_frag(s, outer_frag_replica):
          full_params = nnx.filter_state(s.model, nnx.Param)
          if config.communication_overlapping_alpha > 0.0:
            inner_frag = manipulator.get_flat_fragment(full_params, idx, has_replica_dim=False)
            alpha = config.communication_overlapping_alpha
            merged_frag = jax.tree.map(lambda i, o: alpha * i + (1 - alpha) * o, inner_frag, outer_frag_replica)
          else:
            merged_frag = outer_frag_replica

          new_full_params = manipulator.apply_flat_fragment(full_params, idx, merged_frag, has_replica_dim=False)
          non_param_model = nnx.filter_state(s.model, nnx.Not(nnx.Param))
          new_model = nnx.merge_state(non_param_model, new_full_params)

          result = type(s)({})
          result["model"] = new_model
          result["optimizer"] = s["optimizer"]
          return result

        def replace_linen_model_params_frag(s, outer_frag_replica):
          if config.communication_overlapping_alpha > 0.0:
            inner_frag = manipulator.get_flat_fragment(s.params, idx, has_replica_dim=False)
            alpha = config.communication_overlapping_alpha
            merged_frag = jax.tree.map(lambda i, o: alpha * i + (1 - alpha) * o, inner_frag, outer_frag_replica)
          else:
            merged_frag = outer_frag_replica
          new_params = manipulator.apply_flat_fragment(s.params, idx, merged_frag, has_replica_dim=False)
          return s.replace(params=new_params)

        # Apply to replica inner states
        replace_fn = replace_nnx_model_params_frag if config.pure_nnx else replace_linen_model_params_frag
        new_inner_state = drjax.map_fn(replace_fn, (state.inner_state, broadcast_outer_frag), mesh=mesh)
        return state.replace(inner_state=new_inner_state)

      # Step 1: Run the synchronization logic if we hit a sync step
      is_sync_step = jax.lax.bitwise_and(new_step > 0, new_step % steps_between_syncs_plus_1 == 0)

      def do_sync(s):
        frag_idx = (new_step % period) // steps_between_syncs_plus_1
        return jax.lax.switch(
            frag_idx, [lambda state_arg, idx=i: synchronize_fragment(state_arg, idx) for i in range(num_fragments)], s
        )

      state = jax.lax.cond(is_sync_step, do_sync, lambda s: s, state)

      # Step 2: Apply the synced parameters (with delay V)
      V = config.num_communication_overlapping_steps
      is_apply_step = jax.lax.bitwise_and(new_step - V > 0, (new_step - V) % steps_between_syncs_plus_1 == 0)

      def do_apply(s):
        frag_idx = ((new_step - V) % period) // steps_between_syncs_plus_1
        return jax.lax.switch(
            frag_idx, [lambda state_arg, idx=i: apply_fragment(state_arg, idx) for i in range(num_fragments)], s
        )

      state = jax.lax.cond(is_apply_step, do_apply, lambda s: s, state)

    else:
      # Either synchronize the model, or no-op, depending on whether the current
      # step falls on the synchronization period.
      state = jax.lax.cond(
          new_step % config.diloco_sync_period == 0,
          synchronize,
          lambda x: x,  # no-op
          state,
      )
    return state, avg_metrics

  return diloco_train_step
