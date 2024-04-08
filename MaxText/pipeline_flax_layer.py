import jax
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
    self.microbatch_size = self.config.global_batch_size_to_train_on // self.config.num_pipeline_microbatches
    
  def S(self, *specs):
    return NamedSharding(self.mesh, PartitionSpec(*specs))

  def shard_dim_by_stages(self, x):
   '''Assumes the stages dimension is leading and the mesh has name stages.'''
   specs = ['stage'] + [None] * (x.ndim - 1)
   stage_sharding = S(self.mesh, *specs)
   return jax.lax.with_sharding_constraint(x, stage_sharding)
  
  def init_states(self, inputs):
    '''Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
        Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

        Returns
          shift: zeros shape [n_stages, micro_size, sequence, embed]
          state_io: reshaped inputs [n_stages, microbatches/stages, micro_size, sequence, embed]
          circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed]
          circ_storage_mover: zeros[n_stages, micro_size, sequence, embed]
    
    '''

    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [n_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:])
    shift = self.shard_dim_by_stages(shift, self.mesh)
    # TODO: Use logical names e.g. "activation_embed" for mixed sharding strategies below?
    #shift = jax.lax.with_sharding_constraint(shift, S(self.mesh, 'stage', 'data', None, 'tensor'))

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs as the pipeline runs/finishes
    # state_io has shape [n_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (self.num_stages, self.num_pipeline_microbatches // self.num_stages) + inputs.shape[1:])
    state_io = self.shard_dim_by_stages(state_io)
    # TODO: Use logical names e.g. "activation_embed" for mixed sharding strategies below?
    #state_io = jax.lax.with_sharding_constraint(state_io, S(self.mesh, 'stage', None, 'data', None, 'tensor'))

    # TODO: verify comment below
    # The data/fsdp can shard over microbatch_size, not number of microbatches. The num_microbatches is looped over so should not be sharded.

    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only needed
    # when num_microbatches > num_stages, else instead the final stage can immediately pass to the first without additional storage.
    # Alternative name is "between_repeats_storage"
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed] -- this is huge btw, it should be reducible by a factor of num_stages
    if self.use_circ_storage:
        circ_storage = jnp.zeros((self.num_stages,) + inputs.shape )
    else:
       circ_storage = None

    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage
    # circ_storage_mover shape is same as shift: [n_stages, micro_size, sequence, embed]
    # This mover is one iteration behind before being pushed into storage - which is why we can't just re-use output
    # However shouldn't we be able to keep only the last stage's output instead of all stages?
    if self.use_circ_storage:
        circ_storage_mover = shift
    else:
       circ_storage_mover = None

    return state_io, shift, circ_storage, circ_storage_mover

  def __call__(self, inputs: jnp.ndarray, positions: jnp.ndarray, segment_ids:jnp.ndarray, deterministic: bool, model_mode=common_types.MODEL_MODE_TRAIN) -> jnp.ndarray:
    # We want to access the variables of the decoder_layer, the below loop fills in the variables dictionary (previously empty dict)
    for decoder in self.decoder_layers:
      #print(dir(decoder))
      _ = decoder(inputs, positions, segment_ids, deterministic, model_mode)


    ##### Begin real implementation ####
    #Reshape from [global_batch, ...] to [num_micro_batches, micro_batch_size, ...]
    inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length, self.config.emb_dim))
    positions = positions.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
    segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))

    state_io, shift, circ_storage, circ_storage_mover = self.init_states(inputs)








    # Fake implementation
    # TODO: This stacking is silly - its passing entire inputs to both instead of microbatching first
    # decoder.variables is an empty dictionary until the loop above is executed
    decoder_params = [decoder.variables for decoder in self.decoder_layers] 
    stacked_params = stack_pytrees(*decoder_params)
    stacked_inputs = stack_pytrees(*([inputs] * self.num_stages))
    stacked_positions = stack_pytrees(*([positions] * self.num_stages))
    stacked_segment_ids = stack_pytrees(*([segment_ids] * self.num_stages))

    decoder_apply=functools.partial(self.decoder_layers[0].apply, deterministic=deterministic, model_mode=model_mode)
    #pipeline_output = jax.vmap(decoder_apply)(stacked_params, stacked_inputs, stacked_positions, stacked_segment_ids)
    # Alternatively use in_axis instead of partial:
    pipeline_output = jax.vmap(self.decoder_layers[0].apply, in_axes=[0,0,0,0,None,None])(stacked_params, stacked_inputs, stacked_positions, stacked_segment_ids, deterministic, model_mode)

    pipeline_output = pipeline_output[0] # correct for shape of above hack
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
    decoder_layer_class=simple_decoder_layer.SimpleDecoderLayer,
    mesh=mesh
  )
  init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

  my_pipeline.apply(init_pipeline_params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

if __name__ == "__main__":
  app.run(main)