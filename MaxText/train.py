"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method
"""Training loop and Decoding of the model."""
from typing import Sequence

import os
import datetime
from absl import app
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
import numpy as np
import optax
from tensorboardX import SummaryWriter

from layers import Transformer
import pyconfig
from input_pipeline import get_datasets
from input_pipeline import preprocess_dataset
import max_utils
import temperature_sampler
import checkpointing



import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental.maps import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

import max_logging

cc.initialize_cache(os.path.expanduser("~/jax_cache"))


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def get_first_step(state):
  with jax.spmd_mode('allow_all'):
    return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons """

  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return train_iter()

def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr):
  """Records scalar metrics to be written to tensorboard"""
  metrics['scalar'].update({
      'perf/step_time_seconds': step_time_delta.total_seconds()
  })
  metrics['scalar'].update({
      'perf/per_device_tflops' : per_device_tflops
  })
  metrics['scalar'].update({
      'perf/per_device_tflops_per_sec':
          per_device_tflops /
          step_time_delta.total_seconds()
  })
  metrics['scalar'].update({'learning/current_learning_rate': lr })


def write_metrics(writer, metrics, step, config):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode('allow_all'):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar",[]):
        writer.add_scalar(metric_name, metrics["scalar"][metric_name], step)
      for metric_name in metrics.get("scalars",[]):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    full_log = step % config.log_period == 0

    max_logging.log(f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
          f"TFLOP/s: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
          f"loss: {metrics['scalar']['learning/loss']:.3f}")

    if full_log:
      max_logging.log(
          f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'"
      )
      writer.flush()



def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  return total_parameters

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


def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """ Adds the activation metrics to the metrics dict"""

  if config.scan_layers:
    metrics_dict = intermediate_outputs['intermediates']['decoder']['decoder']

    for layer_num in range(config.num_decoder_layers):
      output_metrics['scalar'][f'activ_fraction_zero/layer_{layer_num:03d}'] = \
        metrics_dict["activation_fraction_zero"][0][layer_num]
      output_metrics['scalar'][f'activ_mean/layer_{layer_num:03d}'] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics['scalar'][f'activ_stdev/layer_{layer_num:03d}'] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs['intermediates']['decoder'][f'layers_{layer_num}']
      output_metrics['scalar'][f'activ_fraction_zero/layer_{layer_num:03d}'] = layer["activation_fraction_zero"][0]
      output_metrics['scalar'][f'activ_mean/layer_{layer_num:03d}'] = layer["activation_mean"][0]
      output_metrics['scalar'][f'activ_stdev/layer_{layer_num:03d}'] = layer["activation_stdev"][0]

def train_step(model, config, state, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  # inputs, targets, segments, positions = apply_args
  rng1, rng2 = jax.random.split(dropout_rng)

  def loss_fn(params):
    logits, intermediate_outputs = model.apply({'params': params},
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
    return jnp.sum(xent)/jnp.size(xent), intermediate_outputs

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, intermediate_outputs), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)
  metrics = {'scalar': {'learning/loss': loss, 'learning/grad_norm' : max_utils.l2norm_pytree(grads),
             'learning/param_norm' : max_utils.l2norm_pytree(new_state.params)}, 'scalars': {}}
  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

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

def train_loop(config, state=None):
  """Main Training loop.

  Args:
    config:
    state:
    ckpt_path:

  Returns:

  """
  writer = SummaryWriter(config.tensorboard_dir)
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config.checkpoint_dir, config.enable_checkpointing)
  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(0), 2)

  # Model and Optimizer definition
  model = Transformer(config)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  tx = optax.adam(
      max_utils.create_learning_rate_schedule(
          learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
      )
  )

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Set up datasets.
  train_ds, eval_ds = get_datasets(
      config=config,
  )
  train_iter, _, _, _ = preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds,
    vocab_path=os.path.join(config.base_output_directory, config.vocab_relative_path),
  )

  state, state_mesh_annotations = max_utils.setup_initial_state(model, tx, config, init_rng, mesh, checkpoint_manager)
  data_pspec = P(*config.data_sharding)

  num_model_parameters = calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")
  per_device_tflops =  6 * num_model_parameters * config.max_target_length * config.per_device_batch_size / 10**12

  # Define compiled top-level functions.
  p_train_step = pjit(
    train_step,
    in_axis_resources=(state_mesh_annotations,
                       data_pspec,
                       None),
    out_axis_resources=(state_mesh_annotations, None, None),
    static_argnums=(0,1,),
    donate_argnums=2)

  example_batch = None
  last_step_completion = datetime.datetime.now()

  for step in np.arange(get_first_step(state), config.steps):
    example_batch = load_next_batch(train_iter, example_batch, config)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics, nextrng = p_train_step(
          model, config, state, example_batch, nextrng
      )

    new_time = datetime.datetime.now()
    record_scalar_metrics(metrics, new_time - last_step_completion,  per_device_tflops, learning_rate_schedule(step))
    write_metrics(writer, metrics, step, config)
    last_step_completion = new_time

    if step > 0 and step % config.save_period == 0 and checkpoint_manager is not None:
      checkpoint_manager.save(step, state)
      max_logging.log("saved a checkpoint")

    # Start profiling at end of first step to avoid compilation.
    # Move before for loop to include.
    if step == 0:
      max_utils.activate_profiler(config)

  max_utils.deactivate_profiler(config)
  writer.close()
  return state

def update_libtpu_init_args(flag,value):
  cur_args = [arg for arg in os.environ.get('LIBTPU_INIT_ARGS', "").split(" ") if f"--{flag}=" not in arg]
  cur_args.append(f"--{flag}={value}")
  os.environ["LIBTPU_INIT_ARGS"] = " ".join(cur_args)

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = pyconfig.config.use_pjrt
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  update_libtpu_init_args("xla_tpu_enable_megascale_barrier", pyconfig.config.mxla_barrier)
  train_loop(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
