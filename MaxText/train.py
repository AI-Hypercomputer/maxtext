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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
import sys

from typing import Sequence
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
import numpy as np
import optax
from tensorboardX import SummaryWriter

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import pyconfig

from input_pipeline import create_data_iterator_with_tokenizer
from layers import models

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import llama2

models = llama2
Transformer = models.Transformer


def validate_train_config(config):
  """ Validates the configuration is set correctly for train.py"""

  assert config.run_name, "Erroring out, need a real run_name"
  if not config.dataset_path.startswith('gs://'):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith('gs://'):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")

  assert ((config.load_parameters_path=="" and config.load_from_other_directory=="") or
    config.enable_checkpointing), "You must set enable_checkpointing to load a checkpoint"
  assert config.load_parameters_path=="" or config.load_from_other_directory=="",\
  "At most one of load_parameters_path or load_from_other_directory should be set"
  assert config.load_from_other_directory_step==-1 or config.load_from_other_directory!="",\
  "You must specify the loading directory if you specify the loading step"
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive interger."

# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_training_tflops(num_model_parameters, config):
  """ Calculate training TFLOP"""
  learnable_weight_tflops = 6 * num_model_parameters * config.max_target_length * config.per_device_batch_size \
                                   / 10**12
  noncasual_attention_flops = 12 * config.num_heads * config.num_decoder_layers * config.head_dim \
                      * config.max_target_length**2 * config.per_device_batch_size / 10**12
  causal_attention_tflops = noncasual_attention_flops / 2 # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops
  print(f'Per train step, total TFLOPs will be {total_tflops:.2f},',
        f'split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops',
        f'and {100 * causal_attention_tflops/total_tflops:.2f}% attention flops')
  return total_tflops

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
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars",[]):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    full_log = step % config.log_period == 0

    max_logging.log(f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
          f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
          f"loss: {metrics['scalar']['learning/loss']:.3f}")

    if full_log and jax.process_index() == 0:
      max_logging.log(
          f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'"
      )
      writer.flush()

# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------

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
  rng1, gen_aqt_rng = jax.random.split(dropout_rng)
  aqt_rng, rng2 = jax.random.split(gen_aqt_rng)

  # decimate proportion of data when per_device_batch_size<1
  for k, v in data.items():
    data[k] = v[:config.global_batch_size_to_train_on,:]

  def loss_fn(params):
    logits, intermediate_outputs = model.apply({'params': params},
                         data['inputs'],
                         data['targets'],
                         data['inputs_segmentation'],
                         data['inputs_position'],
                         enable_dropout=config.enable_dropout,
                         rngs={'dropout': rng1, 'aqt': aqt_rng}, mutable='intermediates')
    one_hot_targets = jax.nn.one_hot(data['targets'], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    xent = nn.with_logical_constraint(xent, ('activation_batch', 'activation_length'))
    # Mask out paddings at the end of each example.
    xent = xent * (data['inputs_segmentation'] != 0)
    return jnp.sum(xent)/jnp.size(xent), intermediate_outputs

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, intermediate_outputs), raw_grads = grad_fn(state.params)
  if config.gradient_clipping_threshold > 0:
    grads, _ = optax.clip_by_global_norm(config.gradient_clipping_threshold).update(raw_grads, state, None)
  else:
    grads = raw_grads
  new_state = state.apply_gradients(grads=grads)
  metrics = {'scalar': {'learning/loss': loss, 'learning/grad_norm': max_utils.l2norm_pytree(grads),
             'learning/raw_grad_norm': max_utils.l2norm_pytree(raw_grads),
             'learning/param_norm': max_utils.l2norm_pytree(new_state.params)}, 'scalars': {}}
  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  return new_state, metrics, rng2

def train_loop(config, state=None):
  """Main Training loop.

  Args:
    config:
    state:
    ckpt_path:

  Returns:

  """
  writer = SummaryWriter(config.tensorboard_dir)
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.save_period,
  )
  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(config.init_weights_seed), 2)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  model = Transformer(config, mesh)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = maxtext_utils.get_optimizer(config, learning_rate_schedule)

  data_iterator, _ = create_data_iterator_with_tokenizer(config, mesh)

  state, state_mesh_annotations = max_utils.setup_training_state(model, tx, config, init_rng, mesh, checkpoint_manager)
  functional_train, in_shard, out_shard, static_argnums, donate_argnums = maxtext_utils.get_functional_train_with_signature(
    train_step,
    mesh,
    state_mesh_annotations,
    model,
    config
  )

  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")
  per_device_tflops = calculate_training_tflops(num_model_parameters, config)

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != '':
    print("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    print("Loaded compiled function!", flush=True)
  else:
    p_train_step = jax.jit(
      functional_train,
      in_shardings=in_shard,
      out_shardings=out_shard,
      static_argnums=static_argnums,
      donate_argnums=donate_argnums)

  example_batch = None
  last_step_completion = datetime.datetime.now()

  local_metrics_file = open(config.metrics_file, 'a', encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  start_step = get_first_step(state) # this is the start_step for training
  first_profiling_step = start_step + config.skip_first_n_steps_for_profiler
  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step:
      max_utils.activate_profiler(config)

    example_batch = load_next_batch(data_iterator, example_batch, config)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics, nextrng = p_train_step(
          state, example_batch, nextrng
      )

    new_time = datetime.datetime.now()
    record_scalar_metrics(metrics, new_time - last_step_completion,  per_device_tflops, learning_rate_schedule(step))
    write_metrics(writer, metrics, step, config)
    last_step_completion = new_time

    if checkpoint_manager is not None:
      if checkpoint_manager.save(step, state):
        max_logging.log(f"saved a checkpoint at step {step}")
      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()

    if config.metrics_file:
      max_utils.write_metrics_locally(metrics, step, config, local_metrics_file)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(metrics, step, config, running_gcs_metrics)

  max_utils.deactivate_profiler(config)
  writer.close()
  return state

def main(argv: Sequence[str]) -> None:
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  print(f"Found {jax.device_count()} devices.")
  config = pyconfig.config
  validate_train_config(config)
  cc.initialize_cache(os.path.expanduser(config.jax_cache_dir))
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  debug_config = debug_configuration.DebugConfig(
    stack_trace_config = stack_trace_configuration.StackTraceConfig(
      collect_stack_trace = config.collect_stack_trace,
      stack_trace_to_cloud = config.stack_trace_to_cloud,
      stack_trace_interval_seconds = config.stack_trace_interval_seconds))
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config)


if __name__ == "__main__":
  app.run(main)
