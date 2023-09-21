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
import jax
import os

jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
print(f"Found {jax.device_count()} devices.")
from jax.experimental.topologies import get_topology_desc
from typing import Sequence
import datetime
from input_pipeline import create_data_iterator_with_tokenizer
from absl import app
from flax.linen import partitioning as nn_partitioning
import numpy as np
import optax
import functools
import pickle
from jax.experimental.serialize_executable import serialize, deserialize_and_load
from tensorboardX import SummaryWriter

from layers import Transformer
import pyconfig
import tensorflow_datasets as tfds
from input_pipeline import get_datasets
from input_pipeline import preprocess_dataset
import max_utils
import checkpointing

import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

import max_logging
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_training_tflops(num_model_parameters, config):
  learnable_weight_tflops = 6 * num_model_parameters * config.max_target_length * config.per_device_batch_size \
                                   / 10**12
  attention_tflops = 12 * config.num_heads * config.num_decoder_layers * config.head_dim * config.max_target_length**2 \
                     * config.per_device_batch_size / 10**12
  total_tflops = learnable_weight_tflops + attention_tflops
  print(f'Per train step, total TFLOPs will be {total_tflops:.2f},',
        f'split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops',
        f'and {100 * attention_tflops/total_tflops:.2f}% attention flops')
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

def train_loop(config, state=None):
  """Main Training loop.

  Args:
    config:
    state:
    ckpt_path:

  Returns:

  """
  writer = SummaryWriter(config.tensorboard_dir)
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config.checkpoint_dir,
                                                                     config.enable_checkpointing,
                                                                     config.async_checkpointing)
  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(config.init_weights_seed), 2)

  # Model and Optimizer definition
  model = Transformer(config)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  tx = optax.adam(
      max_utils.create_learning_rate_schedule(
          learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
      ),
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      eps_root=config.adam_eps_root
  )

  # Mesh definition
  # Mattdavidow: xaot
  topology_devices = get_topology_desc(
      platform='tpu',
      topology_name=f'v4:2x2x1',
      chip_config_name='megacore',
      chips_per_host_bounds=(2, 2, 1),
      num_slices=1,
  ).devices
  use_devices = topology_devices # either topology_devices or jax.devices()
  devices_array = max_utils.create_device_mesh(config, use_devices) # alter
  mesh = Mesh(devices_array, config.mesh_axes)

  # Set up datasets.
  # read_config = tfds.ReadConfig(
  #   shuffle_seed = config.data_shuffle_seed,
  # )
  # train_ds, eval_ds = get_datasets(
  #     config=config,
  #     read_config = read_config,
  # )
  # train_iter, _, _, _ = preprocess_dataset(
  #   config,
  #   mesh,
  #   train_ds, eval_ds,
  #   vocab_path=os.path.join(config.base_output_directory, config.vocab_relative_path),
  #   data_shuffle_seed = config.data_shuffle_seed,
  # )

  state, state_mesh_annotations = max_utils.setup_initial_state(model, tx, config, init_rng, mesh, checkpoint_manager)
  data_pspec = P(*config.data_sharding)

  data_iterator, _ = create_data_iterator_with_tokenizer(config, mesh)
  # num_model_parameters = calculate_num_params_from_pytree(state.params)
  # max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")
  # per_device_tflops = calculate_training_tflops(num_model_parameters, config)


  with mesh,nn_partitioning.axis_rules(config.logical_axis_rules):
    print("Jitting train step...")
    jitted = pjit(
        train_step,
        in_shardings=(state_mesh_annotations, data_pspec, None),
        out_shardings=(state_mesh_annotations, None, None),
        static_argnums=(0,1,),
        donate_argnums=2
    )
    print("Jitted train step!!!")

    example_batch = None
    example_rng = jax.random.PRNGKey(0)
    example_batch = load_next_batch(data_iterator, example_batch, config)
    print("Lowering jitted train step...")
    lowered = jitted.lower(model, config, state, example_batch, example_rng)
    print("Lowered jitted train step!!!")

  orig_compiled = lowered.compile()

  serialized, in_tree, out_tree = serialize(orig_compiled)
  print(f"{in_tree=}")
  print(f"{out_tree=}")

  #out_shape = jax.eval_shape(orig_compiled, state, example_batch, example_rng)
  train_pytree = functools.partial(train_step, model, config)
  out_shaped = jax.eval_shape(train_pytree, state, example_batch, example_rng)
  # print(f"{out_shape=}")
  flat_out_shaped, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
  print(f"{out_tree_recreated=}")
  
  def my_silly_func(*args):
    flat_in_shaped, in_tree_recreated = jax.tree_util.tree_flatten(args)
    print(f"{in_tree_recreated=}")

  #my_silly_func(state, example_batch, example_rng)

  #top_input_tree = {'state':state, 'batch':example_batch, 'rng':example_rng}
  #flat_in_shaped, in_tree_recreated_almost = jax.tree_util.tree_flatten(top_input_tree)
  #print(f"{in_tree_recreated_almost=}")
  input_args = (state, example_batch, example_rng)
  flat_in_shaped, in_tree_recreated = jax.tree_util.tree_flatten((input_args,{}))
  print(f"{in_tree_recreated=}")
  # in_tree_recreated = jax.tree_util.tree_flatten((state, example_batch, example_rng))
  # top_input_tree = {'state':state, 'batch':example_batch, 'rng':example_rng}
  # flat_in_shaped, in_tree_recreated_almost = jax.tree_util.tree_flatten(top_input_tree)
  # in_tree_recreated = [in_tree_recreated_almost['state'], in_tree_recreated_almost['batch'], in_tree_recreated_almost['rng']]
  # print(f"{in_tree_recreated=}")

  # save the serialized via pickle
  if 0:
    print("Saving the serialized compiled train step...")
    with open("x_aot_train.pickle", "wb") as f:
        pickle.dump(serialized, f)
    with open("x_in_tree_train.pickle", "wb") as f:
        pickle.dump(in_tree, f)
    with open("x_out_tree_train.pickle", "wb") as f:
        pickle.dump(out_tree, f)
    print("Saved the serialized compiled train step!!!")
  else:
    print("Not saving the compiled train step because cannot pickle PyTreeDef")

  ## Run locally instead of loading the pickle

  compiled = deserialize_and_load(serialized, in_tree, out_tree)

  # simpler than getting the inputs to train_step correct, just call cost_analysis
  cost = compiled.cost_analysis()[0]['flops']
  print(f"{cost=}")

  # one step using the locally compiled object
  print("Running one step using the compiled object...")
  # one_step_output = compiled(model, config, state, example_batch, example_rng)
  one_step_output = compiled(state, example_batch, example_rng)
  print("One step of compiled successfully ran!")
  






  # Define compiled top-level functions.
#   p_train_step = pjit(
#     train_step,
#     in_shardings=(state_mesh_annotations,
#                        data_pspec,
#                        None),
#     out_shardings=(state_mesh_annotations, None, None),
#     static_argnums=(0,1,),
#     donate_argnums=2)

  return None

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  debug_config = debug_configuration.DebugConfig(
    stack_trace_config = stack_trace_configuration.StackTraceConfig(
      collect_stack_trace = pyconfig.config.collect_stack_trace,
      stack_trace_to_cloud = pyconfig.config.stack_trace_to_cloud,
      stack_trace_interval_seconds = pyconfig.config.stack_trace_interval_seconds))
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
