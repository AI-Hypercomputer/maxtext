import jax
import jax.numpy as jnp
import max_logging
import optimizers
from flax.training import train_state


def get_diloco_train_step(config, train_step):
  # Defer import until needed
  import drjax

  @drjax.program(placements={'clients': config.diloco_num_workers})
  def diloco_train_step(model, config, state, data, dropout_rng):
    """
    Run a DiLoCo round. DiLoCo executes multiple optimization steps within
    each worker, then synchronizes the net change in the model to perform a
    global state update.

    In this implementation, each worker initializes its own AdamW optimizer
    during each round of optimization.
    """

    def scan_fn(carry, data):
      """ Executes a single inner optimization step.  """
      state, step = carry
      nextrng = jax.jit(jax.random.fold_in)(dropout_rng, step)
      state, metrics = train_step(model, config, state, data, nextrng)
      return (state, step + 1), metrics

    def worker_round(start_step, params, worker_inputs):
      """
      Execute one local round of optimization. This executes
      `config.diloco_sync_period` steps locally without any cross-client
      communication.
      """
      # Initialize an AdamW optimizer for the local worker.
      # TODO(jonbolin): Need to preserve optimizer state to use an LR schedule
      adamw_tx = optimizers.get_optimizer(config, config.learning_rate, inner_diloco=True)
      state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=adamw_tx)

      # Scan over local steps, carrying the current step number and updated state.
      (final_state, _), metrics = jax.lax.scan(scan_fn, (state, start_step), worker_inputs)
      metrics = jax.tree.map(lambda x: jnp.average(x), metrics)

      # Calculate the net change in model state.
      model_delta = jax.tree.map(lambda x, y: x - y, params, final_state.params)
      return model_delta, metrics

    max_logging.log('Running training with DiLoCo')

    # Broadcast model parameters
    params_in_clients = drjax.broadcast(state.params)
    start_step_in_clients = drjax.broadcast(state.step)
    #init_rng_in_clients = drjax.broadcast(init_rng)

    # Shape must be NumClients x StepsBetweenSync x ClientBatch x Seq.
    # DrJax will map over the NumClients axis, and DiLoCo will scan over the
    # StepsBetweenSync axis.
    data_shape = (config.diloco_num_workers, config.diloco_sync_period, config.global_batch_size_to_load // config.diloco_num_workers // config.diloco_sync_period, -1)
    data_in_clients = jax.tree_map(lambda x: x.reshape(data_shape), data)

    # Run optimization locally on each worker. The final state within each worker
    # is discarded, only the aggregate change from each worker is reported.
    local_grads, local_metrics = drjax.map_fn(worker_round, (start_step_in_clients, params_in_clients, data_in_clients))

    # DiLoCo Algorithm
    # Average the outer gradients across workers
    average_grad = drjax.reduce_mean(local_grads)
    total_metrics = drjax.reduce_mean(local_metrics)
    # Update global state.
    state = state.apply_gradients(grads=average_grad)

    return state, total_metrics

  return diloco_train_step
