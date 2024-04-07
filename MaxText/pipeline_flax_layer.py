import jax
from jax import numpy as jnp
from jax import tree_map
from flax import linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
from typing import Sequence
from absl import app
import os
import argparse
from typing import Optional
from layers import quantizations
import common_types
import pyconfig
import functools

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

class SimpleDecoderLayer(nn.Module):
  config: common_types.Config
  mesh: Mesh
  quant: Optional[quantizations.AqtQuantization] = None

  def setup(self):
    self.weight_mat = self.param('weights', nn.initializers.ones, (self.config.emb_dim, self.config.emb_dim))

  def __call__(self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode) -> jnp.ndarray:
    return inputs @ self.weight_mat

# Pipeline is made up of several SimpleDecoderLayers 
class Pipeline(nn.Module):
  
  config: common_types.Config
  decoder_layer_class: nn.Module
  mesh: common_types.Mesh
  quant: Optional[quantizations.AqtQuantization] = None

  def setup(self):
    # TODO: See what Inputs are needed to initialize DecoderLayers e.g. LlamaDecoderLayer
    decoder_layers = [self.decoder_layer_class(config=self.config, mesh=self.mesh, name=f'layers_{lyr}', quant=self.quant) for lyr in range(self.config.num_decoder_layers)]
    self.decoder_layers = decoder_layers
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.layers_per_stage = self.config.num_decoder_layers / (self.num_stages * self.config.num_pipeline_repeats)
    assert self.layers_per_stage==1,"Currently only supporting 1 layer per pipeline stage"
    self.use_circ_storage = self.config.num_pipeline_repeats > 1 and self.config.num_pipeline_microbatches > self.num_stages
    

  def __call__(self, inputs: jnp.ndarray, positions: jnp.ndarray, segment_ids:jnp.ndarray, deterministic: bool, model_mode=common_types.MODEL_MODE_TRAIN) -> jnp.ndarray:
    # We want to access the variables of the decoder_layer, the below loop fills in the variables dictionary (previously empty dict)
    for decoder in self.decoder_layers:
      #print(dir(decoder))
      _ = decoder(inputs, positions, segment_ids, deterministic, model_mode)

    # TODO: This stacking is silly - its passing entire inputs to both instead of microbatching first
    # decoder.variables is an empty dictionary until the loop above is executed
    decoder_params = [decoder.variables for decoder in self.decoder_layers] 
    stacked_params = stack_pytrees(*decoder_params)
    stacked_inputs = stack_pytrees(*([inputs] * self.num_stages))
    stacked_positions = stack_pytrees(*([positions] * self.num_stages))
    stacked_segment_ids = stack_pytrees(*([segment_ids] * self.num_stages))

    decoder_apply=functools.partial(self.decoder_layers[0].apply, deterministic=deterministic, model_mode=model_mode)
    pipeline_output = jax.vmap(decoder_apply)(stacked_params, stacked_inputs, stacked_positions, stacked_segment_ids)
    #pipeline_output = jax.vmap(self.decoder_layers[0].apply, in_axes=[0,0,0,0,None,None])(stacked_params, inputs, positions, segment_ids, deterministic, model_mode)
    return pipeline_output

def main(argv: Sequence[str]) -> None:
  # This only exists for convenient testing

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config

  # TODO: determine if num_stages should be added to pyconfig or elsewhere
  num_stages = config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism
  layers_per_stage = config.num_decoder_layers / (num_stages * config.num_pipeline_repeats)
  assert layers_per_stage==1,"Currently only supporting 1 layer per pipeline stage"

  _, inputs, targets, inputs_position, inputs_segmentation = get_weights_and_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim, config.num_decoder_layers)
  deterministic = False
  model_mode = common_types.MODEL_MODE_TRAIN

  mesh = create_mesh(num_stages, config.ici_tensor_parallelism, config.ici_data_parallelism)

  my_pipeline = Pipeline(
    config=config,
    decoder_layer_class=SimpleDecoderLayer,
    mesh=mesh
  )
  init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

  my_pipeline.apply(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

if __name__ == "__main__":
  app.run(main)