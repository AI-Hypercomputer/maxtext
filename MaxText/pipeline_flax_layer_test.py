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
import pipeline_flax_layer

def stack_pytrees(*pytrees):
  """Stacks pytrees with identical structure along a new leading dimension."""
  def stacking_fn(*leaves):
    return jnp.stack(leaves)
  return tree_map(stacking_fn, *pytrees)

def create_mesh(n_stages, tp_axis, dp_axis):
  devices = mesh_utils.create_device_mesh((n_stages, tp_axis, dp_axis))
  

  mesh = Mesh(devices, axis_names=('stage', 'tensor', 'data'))
  return mesh

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
    print(f"{inputs_position.shape}") 
    #inputs_position = jnp.arange((batch_size, sequence), dtype = jnp.int32)
    inputs_segmentation = jnp.ones((batch_size, sequence), dtype=jnp.int32)

    return weights, inputs, dummy_targets, inputs_position, inputs_segmentation

def main(argv: Sequence[str]) -> None:
  # This only exists for convenient testing

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
  #mesh = create_mesh(num_stages, config.ici_tensor_parallelism, config.ici_data_parallelism)

  decoder_layer = simple_decoder_layer.SimpleDecoderLayer
  my_pipeline = pipeline_flax_layer.Pipeline(
    config=config,
    decoder_layer_class=decoder_layer,
    mesh=mesh
  )
  init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

  pipeline_out = my_pipeline.apply(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

  # def get_layer_params(params,layer_idx):
  #   def get_layer(leaf,layer_idx):
  #     return leaf[layer_idx]
  #   my_get_layer = functools.partial(get_layer,layer_idx=layer_idx)
  #   return jax.tree_map(my_get_layer, params)
  
  reg_layer_activations = inputs
  for layer in range(config.num_decoder_layers):
    cur_layer_params = init_pipeline_params['params'][f'layers_{layer}']
    cur_layer_params = {'params':cur_layer_params}
    reg_layer_activations=decoder_layer(config=config,mesh=mesh).apply(cur_layer_params, reg_layer_activations, inputs_position, inputs_segmentation, deterministic, model_mode)
  
  diff = pipeline_out - reg_layer_activations
  print(f"diff norm of {jnp.linalg.norm(diff)}")

  print(f"pipeline output norm of {jnp.linalg.norm(pipeline_out)}")

  print(f"reg output norm of s{jnp.linalg.norm(reg_layer_activations)}")





if __name__ == "__main__":
  app.run(main)