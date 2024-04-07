import jax
from jax import numpy as jnp
from jax import tree_map
from flax import linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import os
import argparse

import common_types
import pyconfig

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

    return weights, inputs, dummy_targets

class SimpleDecoderLayer(nn.Module):
  embed_size: int

  def setup(self):
    self.weight_mat = self.param('weights', nn.initializers.ones, (self.embed_size, self.embed_size))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x @ self.weight_mat

# Pipeline is made up of several SimpleDecoderLayers 
class Pipeline(nn.Module):
  
  config: common_types.Config
  decoder_layer_class: nn.Module
  mesh: common_types.Mesh

  def setup(self):
    # TODO: See what Inputs are needed to initialize DecoderLayers e.g. LlamaDecoderLayer
    decoder_layers = [self.decoder_layer_class(self.config.emb_dim) for _ in range(self.config.n_layers)]
    self.decoder_layers = decoder_layers

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # We want to access the variables of the decoder_layer, the below loop fills in the variables dictionary (previously empty dict)
    for decoder in self.decoder_layers:
      #print(dir(decoder))
      _ = decoder(x)

    # decoder.variables is an empty dictionary until the loop above is executed
    decoder_params = [decoder.variables for decoder in self.decoder_layers] 
    stacked_params = stack_pytrees(*decoder_params)
    stacked_inputs = stack_pytrees(*([x] * self.n_layers))
    pipeline_output = jax.vmap(self.decoder_layers[0].apply)(stacked_params, stacked_inputs)
    return pipeline_output

def main() -> None:
  # This only exists for convenient testing

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config

  # TODO: determine if num_stages should be added to pyconfig or elsewhere
  # TODO place this logic in lass
  num_stages = config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism
  layers_per_stage = config.num_decoder_layers / (num_stages * config.num_pipeline_repeats)
  assert layers_per_stage==1,"Currently only supporting 1 layer per pipeline stage"

  # TODO: place this in class
  use_circ_storage = config.num_pipeline_repeats > 1 and config.num_pipeline_microbatches > num_stages

  
  _, inputs, targets = get_weights_and_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim, config.num_decoder_layers)

  mesh = create_mesh(num_stages, config['ici_tensor_parallelism'], config['ici_data_parallelism'])

  my_pipeline = Pipeline(
    config=config,
    decoder_layer_class=SimpleDecoderLayer,
    mesh=mesh
  )
  init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), inputs)
  my_pipeline.apply(init_pipeline_params, inputs)

if __name__ == "__main__":
  main()