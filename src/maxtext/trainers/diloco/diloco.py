# Copyright 2025 Google LLC
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

"""An implementation of Distributed Low-Communication (DiLoCo) training.

This module contains implementations of:

-   DiLoCo: Distributed Low-Communication Training of Language Models
    https://arxiv.org/abs/2311.08105
-   Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch
    https://arxiv.org/abs/2501.18512
"""

from typing import Any, Callable

import drjax
from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int32, Key, PyTree, UInt32
from maxtext.configs import pyconfig
from maxtext.trainers.diloco import utils as diloco_utils
from maxtext.utils import diloco_sharding
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
      step of the inner optimization. All arrays are expected to have a leading
      dimension with size of the number of diloco replicas so that training
      steps can be mapped over this dimension.
    params: A PyTree of the global model weights. These will mimic a sub-PyTree
      in `inner_state`, which rank-1 shape.
    outer_opt_state: The state for the outer Nesterov momentum optimizer.
    step: The step counter of the training process.
  """

  inner_state: Any
  params: Params
  outer_opt_state: OptState
  step: Step


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
  # Model params (Param variables only) live under abstract_state.model.
  _, model_params, _ = nnx.split(abstract_state.model, nnx.Param, ...)
  model_params = model_params.to_pure_dict()  # pyrefly: ignore[missing-attribute]
  _, model_params_sharding, _ = nnx.split(state_mesh_shardings.model, nnx.Param, ...)
  model_params_sharding = model_params_sharding.to_pure_dict()  # pyrefly: ignore[missing-attribute]
  outer_opt_state = jax.eval_shape(outer_optimizer.init, model_params)

  # Create abstract step
  abstract_step = jax.ShapeDtypeStruct((), jnp.int32)

  # Build abstract DiLoCo state
  diloco_state = DiLoCoTrainState(
      inner_state=inner_state,
      params=model_params,
      outer_opt_state=outer_opt_state,
      step=abstract_step,  # pyrefly: ignore[bad-argument-type]
  )

  # Build shardings
  inner_state_shardings = diloco_utils.add_diloco_to_sharding(state_mesh_shardings)
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
      step=None,  # pyrefly: ignore[bad-argument-type]
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
    # Model params (Param variables only) live under state.model.
    _, outer_params, _ = nnx.split(state.model, nnx.Param, ...)
    outer_params = outer_params.to_pure_dict()  # pyrefly: ignore[missing-attribute]
    outer_opt_state = outer_optimizer.init(outer_params)
    outer_opt_state_sharding = jax.tree_util.tree_map(lambda x: x.sharding, outer_opt_state)
    step = state.optimizer.step
    return (
        DiLoCoTrainState(inner_state=inner_state, params=outer_params, outer_opt_state=outer_opt_state, step=step),
        outer_opt_state_sharding,
    )

  return init_diloco_state()


def build_vanilla_diloco_train_step(
    config: pyconfig.HyperParameters,
    train_step: Callable[[Any, Batch, PRNGKey], tuple[Any, Metrics]],
    mesh: jax.sharding.Mesh | None = None,
) -> Callable[[DiLoCoTrainState, Batch, PRNGKey], tuple[DiLoCoTrainState, Metrics]]:
  """Convert a local state and train step into vanilla DiLoCo train step."""
  outer_optimizer = optax.sgd(
      config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def vanilla_diloco_train_step(state: DiLoCoTrainState, batch: Batch, prng: PRNGKey):
    keys = jax.random.split(prng, config.num_diloco_replicas) if prng is not None else None
    inner_state, metrics = drjax.map_fn(train_step, (state.inner_state, batch, keys), mesh=mesh)
    default_metrics = diloco_utils.extract_per_island_metrics(metrics, config.num_diloco_replicas)
    new_step = inner_state.optimizer.step[0]
    state = state.replace(
        inner_state=inner_state,
        step=new_step,
    )

    state = jax.lax.cond(
        new_step % config.diloco_sync_period == 0,
        lambda s: diloco_utils.synchronize_full_state(s, outer_optimizer, mesh=mesh),
        lambda x: x,
        state,
    )
    return state, default_metrics

  return vanilla_diloco_train_step


def build_streaming_diloco_train_step(
    config: pyconfig.HyperParameters,
    train_step: Callable[[Any, Batch, PRNGKey], tuple[Any, Metrics]],
    mesh: jax.sharding.Mesh | None = None,
) -> Callable[[DiLoCoTrainState, Batch, PRNGKey], tuple[DiLoCoTrainState, Metrics]]:
  """Convert a local state and train step into streaming DiLoCo train step."""
  outer_optimizer = optax.sgd(
      config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )
  num_fragments = config.num_diloco_fragments
  steps_between_syncs, period = diloco_utils.get_streaming_schedule(config)
  delay_v = config.num_communication_overlapping_steps
  alpha = config.communication_overlapping_alpha

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def streaming_diloco_train_step(state: DiLoCoTrainState, batch: Batch, prng: PRNGKey):
    keys = jax.random.split(prng, config.num_diloco_replicas) if prng is not None else None
    inner_state, metrics = drjax.map_fn(train_step, (state.inner_state, batch, keys), mesh=mesh)
    default_metrics = diloco_utils.extract_per_island_metrics(metrics, config.num_diloco_replicas)
    new_step = inner_state.optimizer.step[0]
    state = state.replace(
        inner_state=inner_state,
        step=new_step,
    )

    manipulator = diloco_utils.FragmentedTreeManipulator.create(state.params, config)

    # Step 1: Run the synchronization logic if we hit a sync step
    is_sync_step = (new_step > 0) & (new_step % steps_between_syncs == 0)

    def do_sync(s):
      frag_idx = (new_step % period) // steps_between_syncs
      return jax.lax.switch(
          frag_idx,
          [
              lambda s_arg, idx=i: diloco_utils.synchronize_fragment_state(
                  s_arg, manipulator, idx, outer_optimizer, mesh=mesh
              )
              for i in range(num_fragments)
          ],
          s,
      )

    state = jax.lax.cond(is_sync_step, do_sync, lambda s: s, state)

    # Step 2: Apply the synced parameters (with delay V)
    is_apply_step = (new_step - delay_v > 0) & ((new_step - delay_v) % steps_between_syncs == 0)

    def do_apply(s):
      frag_idx = ((new_step - delay_v) % period) // steps_between_syncs
      return jax.lax.switch(
          frag_idx,
          [
              lambda s_arg, idx=i: diloco_utils.apply_fragment_to_inner_state(
                  s_arg, manipulator, idx, alpha=alpha, mesh=mesh
              )
              for i in range(num_fragments)
          ],
          s,
      )

    state = jax.lax.cond(is_apply_step, do_apply, lambda s: s, state)

    return state, default_metrics

  return streaming_diloco_train_step


def build_diloco_train_step(
    config: pyconfig.HyperParameters,
    train_step: Callable[[Any, Batch, PRNGKey], tuple[Any, Metrics]],
    mesh: jax.sharding.Mesh | None = None,
) -> Callable[[DiLoCoTrainState, Batch, PRNGKey], tuple[DiLoCoTrainState, Metrics]]:
  """Convert a local state and train step into DiLoCo-compatible versions.

  Args:
    config: The config used to set up training.
    train_step: A local train step. This will be executed independently within
      each replica.
    mesh: The mesh for sharding.
  """
  if config.enable_streaming_diloco:
    return build_streaming_diloco_train_step(config, train_step, mesh=mesh)
  return build_vanilla_diloco_train_step(config, train_step, mesh=mesh)


# Re-exports for backward compatibility
reshape_first_axis_with_diloco = diloco_sharding.reshape_first_axis_with_diloco
add_diloco_to_sharding = diloco_sharding.add_diloco_to_sharding
