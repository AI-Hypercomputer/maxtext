import jax
import jax.numpy as jnp
import max_logging
import optax
from flax.training import train_state
from flax import struct

from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


class DiLoCoTrainState(train_state.TrainState):
  adam_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  adam_state: optax.OptState = struct.field(pytree_node=True)

  @classmethod
  def create(cls, config, *, apply_fn, params, tx, **kwargs):
    state = super().create(
      apply_fn=apply_fn,
      params=params,
      tx=tx.sgd,
      adam_tx=tx.adam,
      adam_state=None,
    )

    # Defer import until needed
    import drjax

    @drjax.program(placements={'clients': config.diloco_num_workers})
    def _broadcast(adam_state):
      return drjax.broadcast(adam_state)

    adam_state = _broadcast(tx.adam.init(state.params))
    return state.replace(adam_state=adam_state)


def get_diloco_train_step(config, mesh, train_step):
  # Defer import until needed
  import drjax

  @drjax.program(placements={'clients': config.diloco_num_workers})
  def diloco_train_step(model, config, state, data, dropout_rng):
    """
    Run a DiLoCo round. DiLoCo executes multiple optimization steps within
    each worker, then synchronizes the net change in the model to perform a
    global state update.
    """

    def scan_fn(carry, data):
      """ Executes a single inner optimization step. """
      state, step = carry
      nextrng = jax.jit(jax.random.fold_in)(dropout_rng, step)
      state, metrics = train_step(model, config, state, data, nextrng)
      return (state, step + 1), metrics

    def worker_round(start_step, params, opt_state, worker_inputs):
      """
      Execute one local round of optimization. This executes
      `config.diloco_sync_period` steps locally without any cross-client
      communication.
      """
      # Initialize an AdamW optimizer for the local worker.
      initial_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=state.adam_tx,
      ).replace(
        # Load the client's opt_state into the optimizer.
        opt_state=opt_state,
        step=start_step,
      )

      # Scan over local steps, carrying the current step number and updated state.
      (final_state, _), metrics = jax.lax.scan(scan_fn, (initial_state, start_step), worker_inputs)
      metrics = jax.tree.map(jnp.average, metrics)

      # Calculate the net change in model state.
      model_delta = jax.tree.map(lambda x, y: x - y, params, final_state.params)
      return model_delta, final_state.opt_state, metrics

    max_logging.log('Running training with DiLoCo')

    # Reshard the inputs to the appropriate shape. We reshape each device's data
    # locally using shard_map, since reshaping the global tensor will incur
    # cross-worker collectives due to device placement of the shards.
    # TODO(jonbolin): Can this be avoided?
    @partial(shard_map, mesh=mesh, in_specs=P(*config.data_sharding),
             out_specs=P('clients', None, config.data_sharding[0][1:]))
    def reshape_inputs(inputs):
      # Within shard_map, we are reshaping a single device's input data.
      # Global shape will be NumClients x StepsBetweenSync x ClientBatch x Seq
      return inputs.reshape((1, config.diloco_sync_period, int(config.per_device_batch_size), -1))

    # Broadcast model parameters
    params_in_clients = drjax.broadcast(state.params)
    opt_state_in_clients = state.adam_state
    # The inner step is the global step times diloco_sync_period
    start_step_in_clients = drjax.broadcast(state.step * config.diloco_sync_period)
    reshaped_data = jax.tree_map(reshape_inputs, data)

    # Run optimization locally on each worker. The final state within each worker
    # is discarded, only the aggregate change from each worker is reported.
    local_grads, local_opt_state, local_metrics = \
      drjax.map_fn(worker_round, (start_step_in_clients, params_in_clients, opt_state_in_clients, reshaped_data))

    # DiLoCo Algorithm
    # Average the outer gradients across workers
    average_grad = drjax.reduce_mean(local_grads)
    total_metrics = drjax.reduce_mean(local_metrics)
    # Update global state.
    state = state.apply_gradients(grads=average_grad, adam_state=local_opt_state)

    return state, total_metrics

  return diloco_train_step
