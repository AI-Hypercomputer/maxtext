from jax import numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh
from typing import Optional
from layers import quantizations
from layers import simple_decoder_layer
import common_types
import jax
import jax
import numpy as np
from jax import numpy as jnp
from jax import tree_map
from flax import linen as nn
from jax.sharding import Mesh
from typing import Optional
from layers import quantizations
from layers import simple_decoder_layer
import common_types
import max_utils
import os
import pyconfig
from absl import app
import optax
from flax.training import train_state
import flax
from flax.linen import partitioning as nn_partitioning

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

class MultipleSimpleDecoderLayer(nn.Module):
  config: common_types.Config
  mesh: Mesh
  num_layers: int
  quant: Optional[quantizations.AqtQuantization] = None
  decoder_layer_class: nn.Module
  

  def setup(self):
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.layers_per_stage = self.config.num_decoder_layers / (self.num_stages * self.config.num_pipeline_repeats)
    # TODO: should this assert be in this class or in the initial pyconfig check?
    assert self.layers_per_stage==1,f"Currently only supporting 1 layer per pipeline stage, but {self.config.num_decoder_layers} layers were requested with {self.num_stages} stages"
    self.use_circ_storage = self.config.num_pipeline_repeats > 1 and self.config.num_pipeline_microbatches > self.num_stages
    self.microbatch_size = self.config.global_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    # TODO: should this assert be in this class or pyconfig check?
    assert microbatches_per_stage * self.num_stages == self.config.num_pipeline_microbatches, f"Currently the number of microbatches ({self.config.num_pipeline_microbatches}) must be divisible by the number of stages ({self.num_stages})"
    self.microbatches_per_stage = microbatches_per_stage


  @nn.compact
  def __call__(self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode) -> jnp.ndarray:

    inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length, self.config.emb_dim))
    if positions is not None:
      positions = positions.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      positions_0 = positions[0]
    else:
      positions_0 = None
    if segment_ids is not None:
      segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      segment_ids_0 = segment_ids[0]
    else:
      segment_ids_0 = None

    if positions is not None:
    stages_positions = self.get_microbatches_for_stages(positions, loop_iteration)
    positions_stage_idx = 0
   else:
     stages_positions = None
     positions_stage_idx = 0
   if segment_ids is not None:
    stages_segment_ids = self.get_microbatches_for_stages(segment_ids, loop_iteration)
    segment_stage_idx = 0
   else:
    stages_segment_ids
    segment_stage_idx = None

    initializing = self.is_mutable_collection('params')
    vmapped_fn = nn.vmap(
       simple_decoder_layer.SimpleDecoderLayer,
       in_axes=[0,0,positions_stage_idx, segment_stage_idx, None, None],
       out_axes = 0,
       spmd_axis_name="stage",
       variable_axes={'params': 0},
       split_rngs={'params': True}
    )

    vmapped_fn()
       return inputs
def unbox_logicallypartioned(
    boxed_pytree):
  """ Unboxes the flax.LogicallyPartitioned pieces

    Args:
      boxed_pytree: a pytree that includes LogicallyPartitioned
        leaves.
    Returns:
      a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(lambda x: x.unbox() if \
        isinstance(x, flax.linen.spmd.LogicallyPartitioned) \
        else x, boxed_pytree, \
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned))

def main(argv) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    pyconfig.initialize(argv)
    config = pyconfig.config

    _, inputs, targets, inputs_position, inputs_segmentation = get_weights_and_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim, config.num_decoder_layers)
    deterministic = False
    model_mode = common_types.MODEL_MODE_TRAIN

    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    msdl = MultipleSimpleDecoderLayer(
        config=config,
        mesh=mesh,
        num_layers=4
    )

    print("success")

    # def init_initial_state(model, tx, config, is_training, key):
    #     input_shape = (
    #         config.global_batch_size_to_load,
    #         config.max_target_length
    #     )
    #     model_vars = model.init({'params': key}, inputs,
    #       segmentation,
    #       positions,
    #       deterministic,
    #       model_mode,
    # return init_training_state(model.apply, model_vars, tx)
    variables = msdl.init(jax.random.PRNGKey(0), inputs, inputs_segmentation,inputs_position,deterministic,model_mode)
    tx = optax.adam(learning_rate=0.01)


    def create_state():
        return train_state.TrainState.create(apply_fn=msdl.apply, params=variables,tx=tx )
    abstract_state = jax.eval_shape(create_state)

    state_logical_annotations = nn.get_partition_spec(abstract_state)

    state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh,
                                                     config.logical_axis_rules)

    abstract_sharded_state = jax.jit(
      create_state,
      in_shardings=None,
      out_shardings=state_mesh_shardings
    ).eval_shape()

    unboxed_abstract_sharded_state = unbox_logicallypartioned(abstract_sharded_state)

    # Initialization
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
    breakpoint()

if __name__ == "__main__":
  app.run(main)

#python3 MaxText/state_annotations_sandbox.py MaxText/configs/base.yml run_name=mattdavidow-train-base base_output_directory=gs://maxtext-experiments-multipod dataset_path=gs://max-datasets-rogue steps=5 enable_checkpointing=False base_emb_dim=28 ici_pipeline_parallelism=4 base_num_decoder_layers=4 scan_layers=False num_pipeline_microbatches=12