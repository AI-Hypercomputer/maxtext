import jax
import numpy as np
from jax import numpy as jnp
from jax import tree_map
from flax import linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax.experimental import mesh_utils
from typing import Sequence
from absl import app
import os
import argparse
from typing import Optional
from layers import quantizations
from layers import simple_decoder_layer
import common_types
import pyconfig
import functools
import max_utils
from layers import pipeline
from layers import llama2

import jax.numpy as jnp
import timing_util

def pretty_print_pytree(pytree, indent_level=0):
  """Pretty-prints a JAX PyTree, showing the shapes of each leaf.

  Args:
    pytree: The JAX PyTree to print.
    indent_level: The initial indentation level (default: 0).
  """

  for key, value in pytree.items():
    indent = "  " * indent_level  # Calculate indentation
    if isinstance(value, jnp.ndarray):
      print(f"{indent}{key} {value.shape}")  # Print arrays with shape
    else:
      print(f"{indent}{key}")  # Print other leaves as they are
      pretty_print_pytree(value, indent_level + 1)  # Recurse for nested structures

def get_weights_and_inputs(batch_size, sequence, features, n_layers):
    '''Get random weights, random inputs, and random targets
        Returns
            weights: [n_layers, features, features]
            inputs: [global_batch, sequence, features]
            targets: [global_batch, sequence, features]
    '''
    weights_shape = jnp.array([n_layers, features, features]) # pytree in real cases instead of single array
    k = jax.random.PRNGKey(1)
    weights = jax.random.normal(k,weights_shape, dtype=jnp.float32)

    # we pass in input with global batch, its up to the pipeline function to reshape to microbatches
    input_shape = [batch_size, sequence, features]
    k = jax.random.PRNGKey(2)
    inputs = jax.random.normal(k,input_shape, dtype=jnp.float32)

    # dummy targets same shape as inputs to use for a dummy loss funciton to check gradient correctness
    k = jax.random.PRNGKey(3)
    dummy_targets = jax.random.normal(k,input_shape, dtype=jnp.float32)

    inputs_position = jnp.array([jnp.arange(sequence, dtype=jnp.int32) for _ in range(batch_size)], dtype=jnp.int32)
    inputs_segmentation = jnp.ones((batch_size, sequence), dtype=jnp.int32)

    return weights, inputs, dummy_targets, inputs_position, inputs_segmentation

def assert_same_output_and_grad(f1,f2, targets, *inputs, f1_extra_inputs=[], f2_extra_inputs=[],f1_name="regular",f2_name="pipeline"):
  f1_inputs = (*inputs, *f1_extra_inputs)
  f2_inputs = (*inputs, *f2_extra_inputs)
  def f1_loss(*f1_inputs):
    return jnp.linalg.norm(f1(*f1_inputs) - targets)

  def f2_loss(*f2_inputs):
    return jnp.linalg.norm(f2(*f2_inputs) - targets)

  def print_norms(a,b,f1_name="regular",f2_name="pipeline",diff_name="diff"):
    a_norm = jnp.linalg.norm(a)
    b_norm = jnp.linalg.norm(b)
    diff_norm = jnp.linalg.norm(a-b)

    print(f"{diff_name} norm of {diff_norm}")
    print(f"{f1_name} norm of {a_norm}")
    print(f"{f2_name} norm of {b_norm}")

  def my_ravel(pytree):
    ravelled_tree = jax.tree_map(jnp.ravel, pytree)
    ravelled_leaves, _ = jax.tree_util.tree_flatten(ravelled_tree)
    return jnp.concatenate(ravelled_leaves)

  f1_value = f1(*f1_inputs)
  f2_value = f2(*f2_inputs)
  _, f1_grad = jax.value_and_grad(f1_loss)(*f1_inputs)
  _, f2_grad = jax.value_and_grad(f2_loss)(*f2_inputs)

  f1_grad = my_ravel(f1_grad)
  f2_grad = my_ravel(f2_grad)
  print(f"{f1_grad.shape=}")

  print_norms(f1_value, f2_value, f1_name=f"{f1_name} output", f2_name=f"{f2_name} output", diff_name="Output difference")
  print_norms(f1_grad, f2_grad, f1_name=f"{f1_name} grad", f2_name=f"{f2_name} grad", diff_name="Gradient difference")

def main(argv: Sequence[str]) -> None:
  # TODO: Reformat this test into the same format as other MaxText tests

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config

  # TODO: determine if num_stages should be added to pyconfig or elsewhere
  num_stages = config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism
  layers_per_stage = config.num_decoder_layers / (num_stages * config.num_pipeline_repeats)
  #assert layers_per_stage==1,"Currently only supporting 1 layer per pipeline stage"

  _, inputs, targets, inputs_position, inputs_segmentation = get_weights_and_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim, config.num_decoder_layers)
  deterministic = False
  model_mode = common_types.MODEL_MODE_TRAIN

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  #decoder_layer = simple_decoder_layer.SimpleDecoderLayer
  #from layers import nested_simple_decoder_layer
  #decoder_layer = nested_simple_decoder_layer.SimpleDecoderLayer
  #decoder_layer = simple_decoder_layer.SimpleDecoderLayer(config=config,mesh=mesh).apply
  #decoder_layer = llama2.LlamaDecoderLayer

  # my_pipeline = pipeline.Pipeline(
  #   config=config,
  #   decoder_layer_class=decoder_layer,
  #   mesh=mesh
  # )
  # from layers import pipeline_shard
  # my_pipeline = pipeline_shard.Pipeline(
  #   config=config,
  #   decoder_layer_class=decoder_layer,
  #   mesh=mesh
  # )

  # decoder_layer_instance = simple_decoder_layer.SimpleDecoderLayer(config=config, mesh=mesh)
  # from layers import pipeline_flax_vmap
  # my_pipeline = pipeline_flax_vmap.Pipeline(
  #   config=config,
  #   decoder_layer_instance=decoder_layer_instance,
  #   mesh=mesh
  # )

  from layers import simple_dg
  #decoder_layer_class = simple_decoder_layer.SimpleDecoderLayer
  decoder_layer_class = llama2.LlamaDecoderLayer
  from layers import pipeline_shard_init
  from layers import pipeline
  from layers import pipeline_circular_shard_init
  my_pipeline = pipeline_circular_shard_init.Pipeline(
    config=config,
    decoder_layer_class=decoder_layer_class,
    mesh=mesh
  )


  init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
  #pipeline_out = my_pipeline.apply(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)



  def run_regular_pipeline(params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode):
    reg_layer_activations = inputs

    def get_cur_layer_params(params, layer_idx):
      def get_cur_layer_params_arr(leaf):
        if config.num_pipeline_repeats > 1 and True:
          new_shape = (leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:]
          leaf = jnp.reshape(leaf, new_shape)
        return leaf[layer_idx]
      return jax.tree.map(get_cur_layer_params_arr, params)

    old=False
    for layer in range(config.num_decoder_layers):
      if old:
        cur_layer_params = params['params'][f'layers_{layer}']
        cur_layer_params = {'params':cur_layer_params}
      else:
        cur_layer_params = get_cur_layer_params(params, layer)
        cur_layer_params['params'] = cur_layer_params['params']['layers']
      reg_layer_activations, _ = decoder_layer_class(config=config,mesh=mesh).apply(cur_layer_params, reg_layer_activations, inputs_position, inputs_segmentation, deterministic, model_mode)
    return reg_layer_activations


  
  reg_layers = run_regular_pipeline
  pipeline_func = my_pipeline.apply
  pipeline_func(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
  #assert_same_output_and_grad(reg_layers,pipeline_func, targets, init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

  partial_pipeline_func = functools.partial(pipeline_func, deterministic=deterministic, model_mode=model_mode)
  jit_pipeline_func = jax.jit(partial_pipeline_func)
  timing_util.simple_timeit(jit_pipeline_func, init_pipeline_params, inputs, inputs_position, inputs_segmentation, tries = 3, task = 'basic_pp')


if __name__ == "__main__":
  app.run(main)
  # Circular
  # python3 MaxText/pipeline_parallelism_test.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False base_emb_dim=28 ici_pipeline_parallelism=4 base_num_decoder_layers=8 scan_layers=True num_pipeline_microbatches=12 num_pipeline_repeats=2
  # Non-circular
  # python3 MaxText/pipeline_parallelism_test.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False base_emb_dim=28 ici_pipeline_parallelism=4 base_num_decoder_layers=4 scan_layers=True num_pipeline_microbatches=12 num_pipeline_repeats=1
  # For timing:
  # python3 MaxText/pipeline_parallelism_test.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False ici_pipeline_parallelism=4 base_num_decoder_layers=4 scan_layers=True num_pipeline_microbatches=4 num_pipeline_repeats=1 base_emb_dim=2560