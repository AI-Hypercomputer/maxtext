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

# pylint: disable=bare-except, consider-using-generator
"""Utils that are only interesting to MaxText. """

import jax
import max_utils
from jax.sharding import PartitionSpec as P
from jax.experimental.serialize_executable import deserialize_and_load


import pickle
import functools
from input_pipeline import input_pipeline_interface



def get_functional_train_with_signature(train_step, mesh, state_mesh_annotations, model, config):
  """ Get the shardings (both state and data) for train_step """
  functional_train = get_functional_train_step(train_step, model, config)
  functional_train.__name__ = "train_step"
  data_pspec = P(*config.data_sharding)
  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  data_sharding = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None) # State, batch, rng
  out_shardings = (state_mesh_shardings, None) # State, metrics
  static_argnums = () # We partial out the static argnums of model and config
  donate_argnums = 0 # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums

def get_functional_train_step(train_step, model, config):
  return functools.partial(train_step, model, config)

def get_functional_eval_with_signature(eval_step, mesh, state_mesh_annotations, model, config):
  """ Get the shardings (both state and data) for eval_step """
  functional_eval = get_functional_eval_step(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  data_pspec = P(*config.data_sharding)
  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  data_sharding = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None) # State, batch, rng
  out_shardings = None # metrics
  static_argnums = () # We partial out the static argnums of model, config
  donate_argnums = () # state will be kept instead of being donated in eval_step
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums

def get_functional_eval_step(eval_step, model, config):
  return functools.partial(eval_step, model, config)

def load_compiled(config, partial_train, state):
  """ # Loading a serialized compiled train step function."""
  # Currently partial_train and state  are needed to reconstruct
  # input/output shapes to construct the in_trees and out_trees for load API
  # Parker is working on a serializing these
  def load_serialized_compiled(save_name):
    with open(save_name, "rb") as f:
      serialized_compiled = pickle.load(f)
    return serialized_compiled

  def get_train_input_output_trees(func, input_args, input_kwargs):
    _, in_tree_recreated = jax.tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(func, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

  serialized_compiled = load_serialized_compiled(config.compiled_trainstep_file)
  shaped_batch = input_pipeline_interface.get_shaped_batch(config)
  example_rng = jax.random.PRNGKey(0)
  shaped_input_args = (state, shaped_batch, example_rng)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree)
  return p_train_step

# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_tflops_training_per_device(num_model_parameters, config, log=True):
  """ Calculate training TFLOP"""
  learnable_weight_tflops = 6 * num_model_parameters * config.max_target_length * config.per_device_batch_size \
                                   / 10**12
  noncasual_attention_flops = 12 * config.num_query_heads * config.num_decoder_layers * config.head_dim \
                      * config.max_target_length**2 * config.per_device_batch_size / 10**12
  causal_attention_tflops = noncasual_attention_flops / 2 # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops

  if log:
    print('Per train step:\n',
          f'Total TFLOPs: {total_tflops:.2f} \n',
          f'split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops',
          f'and {100 * causal_attention_tflops/total_tflops:.2f}% attention flops')
  return total_tflops, learnable_weight_tflops, causal_attention_tflops

# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_tflops_prefill(num_model_parameters, prefill_length, config, log=True):
  """ Calculate training TFLOP"""
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length \
                                   / 10**12
  noncasual_attention_flops = 4 * config.num_query_heads * config.num_decoder_layers * config.head_dim \
                      * prefill_length**2 * config.per_device_batch_size / 10**12
  causal_attention_tflops = noncasual_attention_flops / 2 # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops

  if log:
    print('Per prefill step: \n',
          f'\tTotal TFLOPs: {total_tflops:.2f} \n',
          f'\t\tLearnable weight TFLOPs: {learnable_weight_tflops} ',
          f'({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n',
          f'\t\tCausal attention TFLOPs: {causal_attention_tflops} ',
          f'({100 * causal_attention_tflops/total_tflops:.2f})% of Total')
  return total_tflops, learnable_weight_tflops, causal_attention_tflops


def assert_params_sufficiently_sharded(params, mesh, tolerance=0.01):
  """Checks whether most params are sharded across sharding axis.

  This function determines whether the majority of parameters  are distributed
  across a specified sharding axes with an acceptable tolerance. It compares the
  current distribution to a scenario where all parameters are fully sharded
  across the 'fsdp', 'fsdp_transpose', 'sequence', and 'tensor' axes.

  Args:
    params: params of the model state
    mesh: mesh constructed from config
    tolerance: float between 0.0 and 1.0 representing the allowed percentage of
    non-sharded parameters.
  Returns:
    bool: True if the majority of parameters are sufficiently sharded
  """
  total_num_params = max_utils.calculate_num_params_from_pytree(params)
  product_num_devices_for_weight_sharding  = 1
  for axis in ['fsdp', 'fsdp_transpose', 'sequence', 'tensor']:
    product_num_devices_for_weight_sharding *= mesh.shape[axis]
  total_num_params_per_chip = (
    max_utils.calculate_total_params_per_chip(
      params)
  )
  perfectly_sharded_params_per_chip = (
    total_num_params / product_num_devices_for_weight_sharding
  )
  assert total_num_params_per_chip >= perfectly_sharded_params_per_chip, (
    'Number of parameters per chip must not be less than in the ideal sharded '
    'scenario accross `fsdp`, `fsdp_transpose`,`sequence`, `tensor` axes.'
    )
  assert (
    total_num_params_per_chip/perfectly_sharded_params_per_chip - 1 < tolerance
    ), (f'Number of unsharded parameters exceeds tolerance {tolerance * 100}% '
      'of total parameters.')

