# pylint: disable=g-bad-todo, abstract-method
"""Training loop and Decoding of the model."""
import functools
from typing import Callable, Iterable, Sequence, Tuple, Union

import os
import datetime
from absl import app
from etils import epath
import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
import numpy as np
import optax
from orbax.checkpoint.checkpoint_manager import CheckpointManager
from orbax.checkpoint.checkpointer import Checkpointer
from orbax import checkpoint
from orbax.checkpoint import type_handlers
from tensorboardX import SummaryWriter

from layers import Transformer
import pyconfig
from input_pipeline import get_datasets
import temperature_sampler



import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.maps import Mesh



os.environ["TFDS_DATA_DIR"] = "gs://tensorflow-datasets/datasets"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = "1"



# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[
    [PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array
]

# Learning Rate Schedule
# -----------------------------------------------------------------------------
# learning rate scheduling
def rsqrt_schedule(init_value: float, shift: int = 0):
  def schedule(count):
    return init_value * (count + shift)**-.5 * shift**.5
  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0,
          end_value=learning_rate,
          transition_steps=warmup_steps
          ),
      rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
      ], boundaries=[warmup_steps],
      )


def write_metrics(writer, metrics, step):
  if jax.process_index() == 0:
    for metric_name in metrics.get("scalar",[]):
      writer.add_scalar(metric_name, metrics["scalar"][metric_name], step)
    for metric_name in metrics.get("scalars",[]):
      writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)
    for metric_name in metrics.get("histogram",[]):
      writer.add_histogram(metric_name, metrics["histogram"][metric_name], step)


def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  return total_parameters


# Tokenization and De-tokenization helpers.
# ---------------------------------------------------------------------------


def decode_tokens(toks, tokenizer, eos_id):
  valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
  return tokenizer.detokenize(valid_toks).numpy().decode("utf-8")


def encode_strings(strs, max_len, tokenizer):
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  for i, s in enumerate(strs):
    toks = tokenizer.tokenize(s).numpy()
    # Remove EOS token in prompt.
    tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
  return tokenized_batch


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def init_train_state(model, tx, config, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, key
  """
  input_shape = (
      len(jax.devices()) * config.per_device_batch_size,
      config.max_target_length
  )
  model_vars = model.init({'params': key, 'dropout': key},
                          jnp.ones(input_shape),
                          jnp.ones(input_shape))
  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=model_vars['params'],
      tx=tx)
  return state


def count_dead_neurons(activations):
  # activation_mat is batch x target_length x mlp (hidden layer size.)
  # A neuron is counted as dead if it has
  # an activation of zero among the entire batch.
  batch_sum_activations = jnp.ravel(jnp.sum(activations, axis=0))
  return jnp.sum(batch_sum_activations == 0)


def train_step(model, state, data, dropout_rng):
  # inputs, targets, segments, positions = apply_args
  rng1, rng2 = jax.random.split(dropout_rng)

  def loss_fn(params):
    logits, mod_vars = model.apply({'params': params},
                         data['inputs'],
                         data['targets'],
                         data['inputs_segmentation'],
                         data['inputs_position'],
                         rngs={'dropout': rng1}, mutable='intermediates')
    # TODO: is optax xent as good as custom T5X one?
    xent = optax.softmax_cross_entropy_with_integer_labels(logits, data['targets'])
    # Mask out paddings at the end of each example.
    xent = xent * (data['inputs_segmentation'] != 0)
    # TODO: mask out the prompt if training prefix-LM
    return jnp.sum(xent)/jnp.size(xent), mod_vars

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, mod_vars), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)

  metrics = {'scalar':dict(), 'scalars': dict()}
  metrics['scalar'].update({'loss': loss})

  # Record dead neuron count in each decoder layer in the MLPBlock.
  dead_neuron_metric_dict = {
      f'dead_neurons_layers_{layer_num}':
      count_dead_neurons(mod_vars['intermediates']['decoder']
                         [f'layers_{layer_num}']['mlp']['activations'][0])
      for layer_num in range(model.config.num_decoder_layers)
  }
  metrics['scalars'].update({'dead_neurons': dead_neuron_metric_dict})

  return new_state, metrics, rng2


def predict_step(inputs,
                 state,
                 rngkey,
                 model,
                 config):
  """Predict language model on a batch."""
  # NOTE: wtf are we adding inputs.shape[2:] here?  it's almost always empty??
  target_shape = (inputs.shape[0], config.max_predict_length) + inputs.shape[2:]

  initial_variables = model.init(
      jax.random.PRNGKey(0),
      jnp.ones(target_shape, config.dtype),
      None,
      enable_dropout=False,
      decode=True,
      max_decode_length=config.max_predict_length
  )
  cache = initial_variables["cache"]

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = model.apply(
        {
            "params": state.params,
            "cache": flat_cache
        },
        flat_ids,
        None,
        enable_dropout=False,
        decode=True,
        max_decode_length=config.max_predict_length,
        mutable=["cache"])
    new_flat_cache = new_vars["cache"]
    # Remove singleton sequence-length dimension:
    # [batch, 1, vocab] --> [batch, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # search over possible sequences given input encoding.
  seqs = temperature_sampler.temperature_sample(
      inputs,
      cache,
      tokens_ids_to_logits,
      rngkey,
      temperature=config.sampling_temperature,
      topk=config.sampling_top_k,
      eos_token=config.eos_id)

  return seqs

def create_orbax_checkpoint_manager(checkpoint_dir: str):
  #TODO(this logic is in Orbax. Once that version is in Pip, delete)
  # https://github.com/google/orbax/commit/e9113ce7e52916fba96a1983089a1788d55a0735 #pylint: disable=line-too-long
  p = epath.Path(checkpoint_dir)
  if jax.process_index() == 0:
    if not p.is_dir():
      p.mkdir()
  #END(TODO)
  return CheckpointManager(p,
                           Checkpointer(checkpoint.PyTreeCheckpointHandler()))

def unbox_logicallypartioned_trainstate(
    boxed_train_state: train_state.TrainState):
  """ Unboxes the flax.LogicallyPartitioned pieces in a train state.

    Args:
      boxed_train_state: a train state that includes LogicallyPartitioned
        leaves.
    Returns:
      a TrainState where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(lambda x: x.unbox() if \
        isinstance(x, flax.linen.spmd.LogicallyPartitioned) \
        else x, boxed_train_state, \
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned))

def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           first_checkpoint_path: str,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           mesh,
                           state_mesh_annotations):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    first_checkpoint_path: if there is no checkpoint in the checkpoint manager,
      return the Params from the first_checkpoint_path if they exist. This
      enables loading just the parameters and is intended for finetuning.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    mesh: a physical TPU mesh
    state_mesh_annotation: a PyTree of sharding rules, matching
      abstract_unboxed_pre_state.

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """
  def map_to_pspec(data, pspec):
    if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)) \
          and pspec is not None:
      return type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)
    else:
      return type_handlers.RestoreArgs()

  restore_args = jax.tree_util.tree_map(map_to_pspec,
                                        abstract_unboxed_pre_state,
                                        state_mesh_annotations)
  latest_step = checkpoint_manager.latest_step()
  if latest_step is not None:
    print(f"restoring state from this run's directory latest step \
        {latest_step}")
    return checkpoint_manager.restore(latest_step, abstract_unboxed_pre_state,
                                      {"restore_args" : restore_args}), None
  elif first_checkpoint_path != "":
    print(f"restoring state from first_checkpoint_path {first_checkpoint_path}")
    p = epath.Path(first_checkpoint_path)
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
    return None, checkpointer.restore(p,
                                      item=abstract_unboxed_pre_state,
                                      restore_args=restore_args).params
  else:
    print("not restoring checkpoint")
    return None, None

def train_loop(config, state=None):
  """Main Training loop.

  Args:
    config:
    state:
    ckpt_path:

  Returns:

  """
  writer = SummaryWriter(config.tensorboard_dir)
  checkpoint_manager = create_orbax_checkpoint_manager(config.checkpoint_dir)
  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(0), 2)

  # Model and Optimizer definition
  model = Transformer(config)
  learning_rate_schedule = create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  tx = optax.adam(
      create_learning_rate_schedule(
          learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
      )
  )

  # Mesh definition
  mesh_shape_1d = (len(jax.devices()),)
  print(
      f"number jax devices {len(jax.devices())}, exact devices: {jax.devices()}"
  )
  mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), config.mesh_axes)

  # Set up datasets.
  train_iter, _, _, sp_tokenizer = get_datasets(
      config=config,
      global_mesh=mesh,
      vocab_path=config.vocab_path)

  # Abstract initialization
  init_train_state_partial = functools.partial(init_train_state, model, tx,
                                               config)
  abstract_state = jax.eval_shape(init_train_state_partial, init_rng)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  num_model_parameters = calculate_num_params_from_pytree(abstract_state.params)
  unboxed_abstract_state = unbox_logicallypartioned_trainstate(abstract_state)
  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
    state, raw_params = load_state_if_possible(checkpoint_manager,
                                               config.load_parameters_path,
                                               unboxed_abstract_state,
                                               mesh,
                                               state_mesh_annotations)
    data_pspec = P('data', None) # Dataset Partitioning is batch-parallel.

    if not state:
      state = pjit(
          init_train_state_partial,
          in_axis_resources=None,
          out_axis_resources=state_mesh_annotations
      )(init_rng)
      if raw_params:
        state = state.replace(params = raw_params)
    raw_params = None

  state = unbox_logicallypartioned_trainstate(state)




  # Define compiled top-level functions.
  p_train_step = pjit(
    train_step,
    in_axis_resources=(state_mesh_annotations,
                       data_pspec,
                       None),
    out_axis_resources=(state_mesh_annotations, None, None),
    static_argnums=(0,))

  # TODO: add held-out p_eval_step.

  p_predict_step = pjit(
      functools.partial(predict_step, model=model, config=config),
      in_axis_resources=(P(None, None),
                        state_mesh_annotations,
                        None),
      out_axis_resources=None
  )

  # Encode the demo prompt.
  tokenized_prompts = encode_strings(
      [config.prompt], config.max_predict_length, sp_tokenizer)

  last_step_completion = datetime.datetime.now()
  # Main Loop
  for step in np.arange(state.step, config.steps):
    print('step: ', step)
    example_batch = next(train_iter)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics, nextrng = p_train_step(
          model, state, example_batch, nextrng
      )

    new_time = datetime.datetime.now()
    metrics['scalar'].update({
        'step_time_seconds': (new_time - last_step_completion).total_seconds()
    })
    metrics['scalar'].update({
        'per_device_tflops':
            6 * num_model_parameters * config.max_target_length *
            config.per_device_batch_size / 10**12
    })
    metrics['scalar'].update({
        'per_device_tflops/sec':
            metrics['scalar']['per_device_tflops'] /
            metrics['scalar']['step_time_seconds']
    })
    metrics['scalar'].update({
        'current_learning_rate':
            learning_rate_schedule(state.opt_state[1].count)
    })
    last_step_completion = new_time
    if step % config.log_weight_histogram_period == 0:
      histogram_dict = dict()
      for layer_num in range(model.config.num_decoder_layers):
        histogram_dict[f'wo/layers_{layer_num}'] = state.params['decoder'][
            f'layers_{layer_num}']['mlp']['wo']['kernel']
        histogram_dict[f'wi/layers_{layer_num}'] = state.params['decoder'][
            f'layers_{layer_num}']['mlp']['wi']['kernel']
      metrics['histogram'] = histogram_dict
    write_metrics(writer, metrics, step)

    # Log some stuff.
    if step % config.log_period == 0:
      print(f"completed {step}, per-step scalar metrics {metrics['scalar']}")
      print(
          f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'"
      )
      writer.flush()
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        seqs = p_predict_step(tokenized_prompts, state, nextrng)
        print(decode_tokens(np.array(seqs)[0], sp_tokenizer, config.eos_id))

    if step > 0 and step % config.save_period == 0:
      checkpoint_manager.save(step, state)
      print("saved a checkpoint")

  return state


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  train_loop(pyconfig.config)


if __name__ == "__main__":
  app.run(main)

