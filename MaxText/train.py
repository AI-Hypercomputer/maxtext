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

from input_pipeline import create_data_iterator_with_tokenizer, SyntheticDataIterator
from layers import models

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration
import functools
import multihost_dataloading
from jax.experimental.multihost_utils import process_allgather
import math
from mlperf_logging import mllog


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
    return next(train_iter)

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

def train_step(model, config, state, data, dropout_rng, is_train: bool = True):
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

  if is_train:
    # decimate proportion of data when per_device_batch_size<1
    for k, v in data.items():
      data[k] = v[:config.global_batch_size_to_train_on,:]

  def loss_fn(params, is_train=True):
    logits, intermediate_outputs = model.apply({'params': params},
                         data['inputs'],
                         data['targets'],
                         data['inputs_segmentation'],
                         data['inputs_position'],
                         padding_mask=data['targets_segmentation'] != 0,
                         enable_dropout=config.enable_dropout if is_train else False,
                         rngs={'dropout': rng1, 'aqt': aqt_rng}, mutable='intermediates')
    
    one_hot_targets = jax.nn.one_hot(data['targets'], config.vocab_size, dtype=jnp.float32)
    # add
    logits = logits.astype(jnp.float32)
    if config.stable_cross_entropy_loss:
      xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
    else:
      xent = -jnp.sum(jax.nn.log_softmax(logits) * one_hot_targets, axis=-1, dtype=jnp.float32)
    xent = nn.with_logical_constraint(xent, ('activation_batch', 'activation_length'))
    # Mask out paddings at the end of each example.
    padding_mask = data['targets_segmentation'] != 0
    xent = xent * padding_mask
    cum_loss = jnp.sum(xent).astype(jnp.float32)
    cum_weights = jnp.sum(padding_mask).astype(jnp.float32)
    loss = (cum_loss / (cum_weights + 1e-6)).astype(jnp.float32)
    aux = {
      'intermediate_outputs': intermediate_outputs,
      'cum_loss': cum_loss,
      'cum_weights': cum_weights,
    }
    return loss, aux

  if is_train:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), raw_grads = grad_fn(state.params)
    intermediate_outputs = aux['intermediate_outputs']

    if config.gradient_clipping_threshold > 0:
      grads, _ = optax.clip_by_global_norm(config.gradient_clipping_threshold).update(raw_grads, state, None)
    else:
      grads = raw_grads
    new_state = state.apply_gradients(grads=grads)
    metrics = {'scalar': {'learning/loss': loss, 'learning/grad_norm': max_utils.l2norm_pytree(grads),
            'learning/raw_grad_norm': max_utils.l2norm_pytree(raw_grads),
            'learning/param_norm': max_utils.l2norm_pytree(new_state.params)}, 'scalars': {}}
  else:
    loss, aux = loss_fn(state.params, is_train=False)
    cum_loss = aux['cum_loss']
    cum_weights = aux['cum_weights']
    metrics = {'scalar': {'evaluation/loss': loss, 'evaluation/cum_loss': cum_loss, 'evaluation/cum_weights': cum_weights}}
    new_state = state

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

  train_data_iterator, eval_data_iterator, _ = create_data_iterator_with_tokenizer(config, mesh)

  state, state_mesh_annotations = max_utils.setup_training_state(model, tx, config, init_rng, mesh, checkpoint_manager)

  # hack overwrite state
  def map_fn(key_path, value):
    key_path_str = jax.tree_util.keystr(key_path)
    if key_path_str in  (".step", ".opt_state[0].count", ".opt_state[1].count", "opt_state.count", ".opt_state[<flat index 0>]"):
      max_logging.log(f"overwrite step: {key_path_str}")
      return jnp.array(config.overwrite_ckpt_step, dtype=value.dtype)
    elif key_path_str in (".params['decoder']['decoder']['pre_self_attention_norm']['scale']",  ".params['decoder']['decoder']['mlp']['mlp_layer_norm']['scale']", ".params['decoder']['decoder_norm']['scale']"):
      max_logging.log(f"replaced {key_path_str}")
      with jax.spmd_mode('allow_all'):
        return value - 1.
    else:
      return value

  state = jax.tree_util.tree_map_with_path(map_fn, state)
  start_step = get_first_step(state) # this is the start_step for training

  eval_interval = math.ceil(24567 / config.global_batch_size_to_train_on)
  if jax.process_index() == 0:
    mllogger = mllog.get_mllogger()
    mllogger.event(mllog.constants.CACHE_CLEAR)
    mllogger.start(mllog.constants.INIT_START)
    mllogger.event(mllog.constants.SUBMISSION_ORG, 'Google')
    mllogger.event(mllog.constants.SUBMISSION_PLATFORM, 'tpu-v5p')
    mllogger.event(mllog.constants.SUBMISSION_STATUS, mllog.constants.CLOUD)
    mllogger.event(mllog.constants.SUBMISSION_DIVISION, mllog.constants.CLOSED)
    mllogger.event(mllog.constants.SUBMISSION_BENCHMARK, mllog.constants.GPT3)
    mllogger.event(mllog.constants.OPT_NAME, mllog.constants.ADAM)
    mllogger.event(mllog.constants.OPT_BASE_LR, config.learning_rate)
    mllogger.event(mllog.constants.OPT_END_LR, config.cosine_learning_rate_final_fraction)
    mllogger.event(mllog.constants.OPT_WEIGHT_DECAY, config.adam_weight_decay)
    mllogger.event(mllog.constants.OPT_LR_DECAY_STEPS, int(config.learning_rate_schedule_steps * (1 - config.warmup_steps_fraction)))
    mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS, int(config.learning_rate_schedule_steps * config.warmup_steps_fraction + 1))
    mllogger.event(mllog.constants.OPT_LR_DECAY_SCHEDULE, 'cosine with linear warmup')
    mllogger.event(mllog.constants.INIT_CHECKPOINT_STEP, start_step)
    mllogger.event(mllog.constants.OPT_ADAM_BETA_1, config.adam_b1)
    mllogger.event(mllog.constants.OPT_ADAM_BETA_2, config.adam_b2)
    mllogger.event(mllog.constants.OPT_ADAM_EPSILON, config.adam_eps)
    mllogger.event(mllog.constants.OPT_GRADIENT_CLIP_NORM, config.gradient_clipping_threshold)
    mllogger.event(mllog.constants.GLOBAL_BATCH_SIZE, config.global_batch_size_to_train_on)
    mllogger.event(mllog.constants.MAX_SEQUENCE_LENGTH, config.max_target_length)
    mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS, 1)
    mllogger.event(mllog.constants.EVAL_SAMPLES, 24567)

  functional_train, in_shard_train, out_shard_train, static_argnums_train, donate_argnums_train = maxtext_utils.get_functional_train_with_signature(
    train_step,
    mesh,
    state_mesh_annotations,
    model,
    config,
    is_train=True,
  )

  if eval_data_iterator:
    functional_eval, in_shard_eval, out_shard_eval, static_argnums_eval, donate_argnums_eval = maxtext_utils.get_functional_train_with_signature(
      train_step,
      mesh,
      state_mesh_annotations,
      model,
      config,
      is_train=False,
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
      in_shardings=in_shard_train,
      out_shardings=out_shard_train,
      static_argnums=static_argnums_train,
      donate_argnums=donate_argnums_train)
  if eval_data_iterator:
    p_eval_step = jax.jit(
      functional_eval,
      in_shardings=in_shard_eval,
      out_shardings=out_shard_eval,
      static_argnums=static_argnums_eval,
      donate_argnums=donate_argnums_eval,
    )

  # pre compile graph
  synthetic_data_batch = SyntheticDataIterator(config, mesh)()
  # eval first since state_copy will be donated in train_step
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    # have to use copy and put p_eval_step first since p_train_step will consume the tensor buffer with donate_argnums feature
    state_copy = jax.tree_map(lambda x: x.copy(), state)
    p_eval_step(state_copy, synthetic_data_batch, nextrng)
    p_train_step(state_copy, synthetic_data_batch, nextrng)

  example_batch = None
  last_step_completion = datetime.datetime.now()

  local_metrics_file = open(config.metrics_file, 'a', encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  first_profiling_step = start_step + config.skip_first_n_steps_for_profiler
 
  if jax.process_index() == 0:
    mllogger.end(mllog.constants.INIT_STOP)
    mllogger.start(mllog.constants.RUN_START)
    eval_frequency_tokens = eval_interval * config.global_batch_size_to_train_on * config.max_target_length

  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step:
      max_utils.activate_profiler(config)

    example_batch = load_next_batch(train_data_iterator, example_batch, config)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics, nextrng = p_train_step(
          state, example_batch, nextrng
      )

    new_time = datetime.datetime.now()
    record_scalar_metrics(metrics, new_time - last_step_completion,  per_device_tflops, learning_rate_schedule(step))
    write_metrics(writer, metrics, step, config)
    last_step_completion = new_time

    if checkpoint_manager is not None:
      if step > 0 and checkpoint_manager.save(step, state):
        max_logging.log(f"saved a checkpoint at step {step}")
      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()

    if config.metrics_file:
      max_utils.write_metrics_locally(metrics, step, config, local_metrics_file)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(metrics, step, config, running_gcs_metrics)

    if step > start_step and step % eval_interval == 0:
      valid_cum_loss = 0
      valid_cum_weights = 0
      for batch in eval_data_iterator:
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          _, metrics, _ = p_eval_step(
            state, batch, nextrng
          )
        batch_valid_cum_loss = float(metrics['scalar']['evaluation/cum_loss'])
        batch_valid_cum_weights = float(metrics['scalar']['evaluation/cum_weights'])
        valid_cum_loss += batch_valid_cum_loss
        valid_cum_weights += batch_valid_cum_weights

      weighted_mean_valid_loss = valid_cum_loss / (valid_cum_weights + 1e-8)
      max_logging.log(f"average loss after {start_step}: weighted_mean={weighted_mean_valid_loss}, total_weights={valid_cum_weights}")
      if jax.process_index() == 0:
        current_epoch_num = step * config.global_batch_size_to_train_on * config.max_target_length
        first_epoch_num = current_epoch_num - eval_frequency_tokens
        mllogger.end(
          mllog.constants.BLOCK_STOP,
          metadata={'first_epoch_num': first_epoch_num},
        )
        mllogger.event(
          mllog.constants.EVAL_ACCURACY,
          weighted_mean_valid_loss,
          metadata={'epoch_num': current_epoch_num},
          )

      if config.target_valid_loss and weighted_mean_valid_loss <= config.target_valid_loss:
        if jax.process_index() == 0:
          mllogger.end(mllog.constants.RUN_STOP, metadata={'status': 'success'})
          mllogger.event(mllog.constants.TRAIN_SAMPLES, current_epoch_num)
        break

      if jax.process_index() == 0:
        mllogger.start(
          mllog.constants.BLOCK_START,
          metadata={
              'epoch_count': eval_frequency_tokens,
              'first_epoch_num': current_epoch_num,
              },
          )

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
