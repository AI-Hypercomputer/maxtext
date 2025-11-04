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
from flax import struct
from flax.training import train_state
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int32, Key, PyTree, UInt32
import optax

from MaxText import pyconfig

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
    outer_params: A PyTree of the global model weights. These will mimic a
      sub-PyTree in `inner_state`, which rank-1 shape.
    outer_opt_state: The state for the outer Nesterov momentum optimizer.
    step: The step counter of the training process.
  """

  inner_state: train_state.TrainState
  outer_params: Params
  outer_opt_state: OptState
  step: Step


def reshape_first_axis_with_diloco(num_diloco_replicas: int, pytree: PyTree) -> PyTree:
  """Reshapses the first dimension of each array int he PyTree to include a DiLoCo axis."""

  def extend_pspec(pspec: jax.sharding.PartitionSpec | Sequence[str | Sequence[str]] = ()) -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec("diloco", pspec)

  def reshape_for_diloco(arr):
    batch_dim, *example_shape = arr.shape
    diloco_shape = (num_diloco_replicas, batch_dim // num_diloco_replicas, *example_shape)
    s = arr.sharding
    s = jax.sharding.NamedSharding(mesh=s.mesh, spec=extend_pspec(s.spec))
    return jax.lax.with_sharding_constraint(jnp.reshape(arr, shape=diloco_shape), s)

  return jax.tree.map(reshape_for_diloco, pytree)


def build_diloco_state(
    config: "pyconfig.HyperParameters",
    initialize_state: Callable[[], train_state.TrainState],
) -> DiLoCoTrainState:
  """Given a non-DiLoCo train state, construct a DiLoCo training state."""
  outer_optimizer = optax.sgd(
      config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def init_diloco_state() -> DiLoCoTrainState:
    state = initialize_state()
    # Inner state must be broadcast across clients.
    inner_state = drjax.broadcast(state)
    # Outer state retains a single copy of the model parameters and optimizer state.
    outer_params = state.params
    outer_opt_state = outer_optimizer.init(outer_params)
    return DiLoCoTrainState(
        inner_state=inner_state, outer_params=outer_params, outer_opt_state=outer_opt_state, step=state.step
    )

  return init_diloco_state()


def build_diloco_train_step(
    config: "pyconfig.HyperParameters",
    train_step: Callable[[train_state.TrainState, Batch, PRNGKey], tuple[train_state.TrainState, Metrics]],
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
    broadcast_outer_params = drjax.broadcast(state.outer_params)
    model_delta = jax.tree.map(lambda x, y: y - x, state.inner_state.params, broadcast_outer_params)
    # Treat the average delta as the outer optimizer's gradient and apply to
    # the global (outer) model params.
    averaged_pseudo_grad = drjax.reduce_mean(model_delta)
    updates, new_opt_state = outer_optimizer.update(averaged_pseudo_grad, state.outer_opt_state, state.outer_params)
    new_outer_params = optax.apply_updates(state.outer_params, updates)
    # Replace inner model params with the new global model params.
    # NOTE: inner optimizer state is retained despite the change in parameters,
    # see section 6.1 in https://arxiv.org/pdf/2311.08105.
    new_inner_state = drjax.map_fn(lambda state: state.replace(params=new_outer_params), state.inner_state)
    return state.replace(
        outer_params=new_outer_params,
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
    broadcast_rng = drjax.broadcast(prng)
    inner_state, metrics = drjax.map_fn(train_step, (state.inner_state, batch, broadcast_rng))
    avg_metrics = typed_reduce_mean(metrics)
    state = state.replace(
        inner_state=inner_state,
        step=inner_state.step[0],
    )
    # Either synchronize the model, or no-op, depending on whether the current
    # step falls on the synchronization period.
    state = jax.lax.cond(
        inner_state.step[0] % config.diloco_sync_period == 0,
        synchronize,
        lambda x: x,  # no-op
        state,
    )
    return state, avg_metrics

  return diloco_train_step
