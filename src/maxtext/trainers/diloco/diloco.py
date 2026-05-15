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

  inner_state: train_state.TrainState
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
    initialize_state: Callable[[], train_state.TrainState],
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
    outer_params = state.model.filter(nnx.Param) if config.pure_nnx else state.params
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
    train_step: Callable[[train_state.TrainState, Batch, PRNGKey], tuple[train_state.TrainState, Metrics]],
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
        nnx.filter_state(state.inner_state.model, nnx.Param) if config.pure_nnx else state.inner_state.params
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
    if config.pure_nnx:
      # For NNX: merge new Param vars back with the non-Param model vars (e.g. RNG state).
      def replace_nnx_model_params(s, new_params):
        non_param_model = nnx.filter_state(s.model, nnx.Not(nnx.Param))
        new_model = nnx.merge_state(non_param_model, new_params)
        # Build result via __setitem__ so nested States are stored as plain dicts
        # internally, matching the pytree structure produced by nnx.state().
        # (Passing State objects via the constructor dict literal stores them
        # as-is, causing jax.lax.cond to see mismatched pytree structures.)
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
