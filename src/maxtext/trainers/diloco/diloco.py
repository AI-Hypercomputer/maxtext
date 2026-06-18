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
from typing import Any, Callable

import drjax
from flax import nnx
from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int32, Key, PyTree, UInt32
import optax

from maxtext.configs import pyconfig

Batch = Any
Params = PyTree
Metrics = PyTree
OptState = optax.OptState
InnerOptStates = optax.OptState
PRNGKey = Key[Array, ""] | UInt32[Array, "2"]
Step = Int32[Array, ""]


import re


def _pure_nnx(config):
  return getattr(config, "pure_nnx", False)


def _enable_streaming_diloco(config):
  return getattr(config, "enable_streaming_diloco", False)


def _num_synced_layers_per_step(config):
  return getattr(config, "num_synced_layers_per_step", config.num_decoder_layers)


def _use_sequential_layers(config):
  return getattr(config, "use_sequential_layers", False)


class FragmentedTreeManipulator:
  """Partitions and manipulates fragments of a JAX PyTree."""

  def __init__(
      self, keypath_to_fragment_idx: dict[tuple, int], num_fragments: int
  ):
    self.keypath_to_fragment_idx = keypath_to_fragment_idx
    self.num_fragments = num_fragments

  @classmethod
  def create(cls, params_tree, config):
    kvs, _ = jax.tree_util.tree_flatten_with_path(params_tree)

    num_layers = config.num_decoder_layers
    num_synced = _num_synced_layers_per_step(config)
    use_sequential = _use_sequential_layers(config)

    num_transformer_fragments = num_layers // num_synced
    num_fragments = 1 + num_transformer_fragments

    keypath_to_fragment_idx = {}
    layer_regex = re.compile(r"(?:layers_|blocks_)(\d+)")

    for keypath, _ in kvs:
      # Serialize keypath to a string representation
      parts = []
      for k in keypath:
        parts.append(
            str(k.key)
            if hasattr(k, "key")
            else (str(k.idx) if hasattr(k, "idx") else str(k))
        )
      serialized_path = "/".join(parts)

      # Extract layer index
      match = layer_regex.search(serialized_path)
      if match:
        lid = int(match.group(1))
        if use_sequential:
          transformer_frag_idx = lid // num_synced
        else:
          transformer_frag_idx = lid % num_transformer_fragments
        frag_idx = 1 + transformer_frag_idx
      else:
        frag_idx = 0  # Base fragment

      keypath_to_fragment_idx[keypath] = frag_idx

    return cls(keypath_to_fragment_idx, num_fragments)

  def get_flat_fragment(self, tree, fragment_idx: int) -> dict[tuple, Any]:
    kvs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        k: v
        for k, v in kvs
        if self.keypath_to_fragment_idx.get(k) == fragment_idx
    }

  def apply_flat_fragment(
      self, tree, fragment_idx: int, flat_fragment: dict[tuple, Any]
  ):
    kvs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    new_kvs = []
    for k, v in kvs:
      if self.keypath_to_fragment_idx.get(k) == fragment_idx:
        new_kvs.append(flat_fragment[k])
      else:
        new_kvs.append(v)
    return jax.tree_util.tree_unflatten(treedef, new_kvs)


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
  """Reshapes the first dimension of each array in the PyTree to include a DiLoCo axis."""

  def extend_pspec(pspec: jax.sharding.PartitionSpec | Sequence[str | Sequence[str]] = ()) -> jax.sharding.PartitionSpec:
    pspec_tuple = tuple(pspec)
    if not pspec_tuple:
      return jax.sharding.PartitionSpec("diloco")
    first = pspec_tuple[0]
    if isinstance(first, (list, tuple)):
      if len(first) > 0 and first[0] == "diloco":
        remaining = tuple(first[1:])
        if len(remaining) == 1:
          remaining = remaining[0]
        elif len(remaining) == 0:
          remaining = None
        if remaining is None:
          return jax.sharding.PartitionSpec("diloco", *pspec_tuple[1:])
        else:
          return jax.sharding.PartitionSpec("diloco", remaining, *pspec_tuple[1:])
    elif first == "diloco":
      return jax.sharding.PartitionSpec("diloco", *pspec_tuple[1:])
    return jax.sharding.PartitionSpec("diloco", *pspec_tuple)

  def reshape_for_diloco(path, arr):
    print(f"DEBUG DILOCO DATA BATCH leaf path: {path}, shape: {arr.shape if hasattr(arr, 'shape') else 'None'}, type: {type(arr)}", flush=True)
    if not hasattr(arr, "shape") or len(arr.shape) == 0:
      print(f"DEBUG DILOCO DATA BATCH skipping reshape for scalar/empty leaf: {path}", flush=True)
      return arr
    batch_dim, *example_shape = arr.shape
    diloco_shape = (num_diloco_replicas, batch_dim // num_diloco_replicas, *example_shape)
    if hasattr(arr, "sharding"):
      s = arr.sharding
      print(f"DEBUG DILOCO DATA BATCH leaf has sharding: {s}", flush=True)
      s = jax.sharding.NamedSharding(mesh=s.mesh, spec=extend_pspec(s.spec))
      res = jax.lax.with_sharding_constraint(jnp.reshape(arr, shape=diloco_shape), s)
      print(f"DEBUG DILOCO DATA BATCH leaf reshaped shape: {res.shape}, sharding: {res.sharding}", flush=True)
      return res
    res = jnp.reshape(arr, shape=diloco_shape)
    print(f"DEBUG DILOCO DATA BATCH leaf reshaped shape: {res.shape}", flush=True)
    return res

  return jax.tree_util.tree_map_with_path(reshape_for_diloco, pytree)


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
  if _pure_nnx(config):
    model_params = abstract_state.model.filter(nnx.Param)
    model_params_sharding = state_mesh_shardings.model.filter(nnx.Param)
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
    outer_params = state.model.filter(nnx.Param) if _pure_nnx(config) else state.params
    outer_opt_state = outer_optimizer.init(outer_params)
    outer_opt_state_sharding = jax.tree_util.tree_map(lambda x: x.sharding, outer_opt_state)
    # For NNX, the step counter lives at state.optimizer.step; for Linen at state.step.
    step = state.optimizer.step if _pure_nnx(config) else state.step
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
    inner_model_params = (
        nnx.filter_state(state.inner_state.model, nnx.Param) if _pure_nnx(config) else state.inner_state.params
    )
    model_delta = jax.tree.map(lambda x, y: y - x, inner_model_params, broadcast_outer_params)
    # Treat the average delta as the outer optimizer's gradient and apply to
    # the global (outer) model params.
    averaged_pseudo_grad = drjax.reduce_mean(model_delta)
    updates, new_opt_state = outer_optimizer.update(averaged_pseudo_grad, state.outer_opt_state, state.params)
    new_outer_params = optax.apply_updates(state.params, updates)
    # Replace inner model params with the new global model params.
    # NOTE: inner optimizer state is retained despite the change in parameters,
    # see section 6.1 in https://arxiv.org/pdf/2311.08105.
    if _pure_nnx(config):
      # For NNX: merge new Param vars back with the non-Param model vars (e.g. RNG state).
      def replace_nnx_model_params(s, new_params):
        non_param_model = nnx.filter_state(s.model, nnx.Not(nnx.Param))
        new_model = nnx.merge_state(non_param_model, new_params)
        # Assign via __setitem__ so nested States are stored as plain dicts (matching
        # nnx.state()'s pytree structure). The dict-literal constructor keeps them as
        # State objects, which makes jax.lax.cond see mismatched pytree structures.
        result = type(s)({})
        result["model"] = new_model
        result["optimizer"] = s["optimizer"]
        return result

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

  # Streaming scheduling parameters
  num_layers = config.num_decoder_layers
  num_synced = _num_synced_layers_per_step(config)
  num_transformer_fragments = num_layers // num_synced
  num_fragments = 1 + num_transformer_fragments

  desired_sync_period = config.diloco_sync_period
  steps_between_syncs_plus_1 = max(
      1, round(desired_sync_period / num_fragments)
  )
  period = num_fragments * steps_between_syncs_plus_1

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
    new_step = inner_state.optimizer.step[0] if _pure_nnx(config) else inner_state.step[0]
    state = state.replace(
        inner_state=inner_state,
        step=new_step,
    )

    if _enable_streaming_diloco(config):
      manipulator = FragmentedTreeManipulator.create(state.params, config)

      def synchronize_fragment(s, idx):
        outer_params_frag = manipulator.get_flat_fragment(s.params, idx)
        inner_model_params = (
            nnx.filter_state(s.inner_state.model, nnx.Param)
            if _pure_nnx(config)
            else s.inner_state.params
        )
        inner_params_frag = manipulator.get_flat_fragment(
            inner_model_params, idx
        )

        unreduced_grads = jax.tree.map(
            lambda x, y: x[None] - y, outer_params_frag, inner_params_frag
        )
        averaged_pseudo_grad = drjax.reduce_mean(unreduced_grads)

        trace_frag = manipulator.get_flat_fragment(
            s.outer_opt_state[0].trace, idx
        )
        opt_state_frag = (
            optax.TraceState(trace=trace_frag),
            optax.EmptyState(),
        )

        updates_frag, new_opt_state_frag = outer_optimizer.update(
            averaged_pseudo_grad, opt_state_frag, params=outer_params_frag
        )
        new_outer_params_frag = optax.apply_updates(
            outer_params_frag, updates_frag
        )

        new_params = manipulator.apply_flat_fragment(
            s.params, idx, new_outer_params_frag
        )
        new_trace = manipulator.apply_flat_fragment(
            s.outer_opt_state[0].trace, idx, new_opt_state_frag[0].trace
        )
        new_outer_opt_state = (
            optax.TraceState(trace=new_trace),
            s.outer_opt_state[1],
        )

        return s.replace(
            params=new_params,
            outer_opt_state=new_outer_opt_state,
        )

      def apply_fragment(s, idx):
        outer_params_frag = manipulator.get_flat_fragment(s.params, idx)
        broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)

        def replace_nnx_model_params_frag(inner_s, outer_frag_replica):
          full_params = nnx.filter_state(inner_s.model, nnx.Param)
          if config.communication_overlapping_alpha > 0.0:
            inner_frag = manipulator.get_flat_fragment(full_params, idx)
            alpha = config.communication_overlapping_alpha
            merged_frag = jax.tree.map(
                lambda i, o: alpha * i + (1 - alpha) * o,
                inner_frag,
                outer_frag_replica,
            )
          else:
            merged_frag = outer_frag_replica

          new_full_params = manipulator.apply_flat_fragment(
              full_params, idx, merged_frag
          )
          non_param_model = nnx.filter_state(inner_s.model, nnx.Not(nnx.Param))
          new_model = nnx.merge_state(non_param_model, new_full_params)

          result = type(inner_s)({})
          result["model"] = new_model
          result["optimizer"] = inner_s["optimizer"]
          return result

        def replace_linen_model_params_frag(inner_s, outer_frag_replica):
          if config.communication_overlapping_alpha > 0.0:
            inner_frag = manipulator.get_flat_fragment(inner_s.params, idx)
            alpha = config.communication_overlapping_alpha
            merged_frag = jax.tree.map(
                lambda i, o: alpha * i + (1 - alpha) * o,
                inner_frag,
                outer_frag_replica,
            )
          else:
            merged_frag = outer_frag_replica
          new_params = manipulator.apply_flat_fragment(
              inner_s.params, idx, merged_frag
          )
          return inner_s.replace(params=new_params)

        replace_fn = (
            replace_nnx_model_params_frag
            if _pure_nnx(config)
            else replace_linen_model_params_frag
        )
        new_inner_state = drjax.map_fn(
            replace_fn, (s.inner_state, broadcast_outer_frag), mesh=mesh
        )
        return s.replace(inner_state=new_inner_state)

      # 1. Sync step logic
      is_sync_step = jax.lax.bitwise_and(
          new_step > 0, new_step % steps_between_syncs_plus_1 == 0
      )

      def do_sync(s):
        frag_idx = (new_step % period) // steps_between_syncs_plus_1
        return jax.lax.switch(
            frag_idx,
            [
                lambda state_arg, idx=i: synchronize_fragment(state_arg, idx)
                for i in range(num_fragments)
            ],
            s,
        )

      state = jax.lax.cond(is_sync_step, do_sync, lambda s: s, state)

      # 2. Apply step logic (with delay V)
      V = config.num_communication_overlapping_steps
      is_apply_step = jax.lax.bitwise_and(
          new_step - V > 0, (new_step - V) % steps_between_syncs_plus_1 == 0
      )

      def do_apply(s):
        frag_idx = ((new_step - V) % period) // steps_between_syncs_plus_1
        return jax.lax.switch(
            frag_idx,
            [
                lambda state_arg, idx=i: apply_fragment(state_arg, idx)
                for i in range(num_fragments)
            ],
            s,
        )

      state = jax.lax.cond(is_apply_step, do_apply, lambda s: s, state)

    else:
      # Fallback to vanilla DiLoCo
      state = jax.lax.cond(
          new_step % config.diloco_sync_period == 0,
          synchronize,
          lambda x: x,  # no-op
          state,
      )

    return state, avg_metrics

  return diloco_train_step
