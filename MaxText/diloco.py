import jax
import jax.numpy as jnp
import optax
import pyconfig

from flax import struct
from typing import Any, Callable, Tuple


Batch = Any
State = Any
Params = Any
Loss = jnp.float32
OptState = optax.OptState
InnerOptStates = optax.OptState
PRNGKey = jax.Array
Step = jnp.int32


class DiLoCoTrainState(struct.PyTreeNode):
  inner_state: State
  outer_params: Params
  outer_opt_state: OptState
  step: Step


def build_diloco_train_step(
    config: pyconfig.HyperParameters,
    train_step: Callable[[State, Batch, PRNGKey], Loss],
    state: State,
) -> Tuple[State, Callable[[State, Batch, PRNGKey], Loss]]:
  """Convert a local state and train step into DiLoCo-compatible versions.
  
  DiLoCo runs multiple replicas of training independently and periodically
  synchronizes between them using SGD with Nesterov momentum.

  Args:
    config: The config used to set up training.
    train_step: A local train step. This will be executed independently within
      each replica.
    state: The flax train state of the standard train loop. This will be
      modified to create DiLoCo state.
  """
  import drjax

  # TODO(jonbolin): Keep this as part of DiLoCoTrainState?
  outer_optimizer = optax.sgd(
    config.diloco_outer_lr,
    momentum=config.diloco_outer_momentum,
    nesterov=True,
  )

  @drjax.program(placements={'diloco': config.num_diloco_replicas})
  def init_train_state(state: State) -> DiLoCoTrainState:
    # Inner state must be broadcast across clients
    inner_state = drjax.broadcast(state)

    # Outer state retains a single copy
    outer_params = state.params
    outer_opt_state = outer_optimizer.init(outer_params)

    return DiLoCoTrainState(
      inner_state=inner_state,
      outer_params=outer_params,
      outer_opt_state=outer_opt_state,
      step=state.step,
    )

  @drjax.program(placements={'diloco': config.num_diloco_replicas})
  def synchronize(state):
    jax.debug.print('synchronizing')
    # Calculate the delta between the current replica's state and the global state
    broadcast_outer_params = drjax.broadcast(state.outer_params)
    model_delta = jax.tree.map(lambda x, y: y - x, state.inner_state.params, broadcast_outer_params)

    # Treat the average delta as the outer optimizer's gradient
    average_grad = drjax.reduce_mean(model_delta)
    updates, new_opt_state = outer_optimizer.update(average_grad, state.outer_opt_state, state.outer_params)
    new_outer_params = optax.apply_updates(state.outer_params, updates)

    # Replace inner parameters with the new global parameters.
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

  @drjax.program(placements={'diloco': config.num_diloco_replicas})
  def diloco_train_step(state, batch, prng):
    # Batch shape is [pdbs, S]. To use in DiLoCo, we need to add a leading axis
    # to vmap over.
    per_replica_batch = config.global_batch_size_to_train_on // config.num_diloco_replicas
    batch_shape = (config.num_diloco_replicas, per_replica_batch, -1)
    batch = jax.tree.map(lambda x: x.reshape(batch_shape), batch)

    # Broadcast the RNG across replicas
    broadcast_rng = drjax.broadcast(prng)

    inner_state, metrics = drjax.map_fn(
      train_step,
      (state.inner_state, batch, broadcast_rng)
    )
    avg_metrics = typed_reduce_mean(metrics)

    state = state.replace(
      inner_state=inner_state,
      step=inner_state.step[0],
    )

    state = jax.lax.cond(
      (inner_state.step[0] + 1) % config.diloco_sync_period == 0,
      synchronize,
      lambda x: x,  # no-op
      state
    )

    return state, avg_metrics


  diloco_state = init_train_state(state)

  return diloco_state, diloco_train_step