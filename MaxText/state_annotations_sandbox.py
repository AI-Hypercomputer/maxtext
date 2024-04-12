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
  


  @nn.compact
  def __call__(self, inputs: jnp.ndarray, positions, segmentation, deterministic, model_mode) -> jnp.ndarray:
    initializing = self.is_mutable_collection('params')
    params_spec = (1 if initializing else common_types.ScanIn(1))
    if self.config.scan_layers:
        scan_fn= nn.scan(
        simple_decoder_layer.SimpleDecoderLayer,
        length = self.num_layers,
        variable_axes={
              'params': params_spec,
              'cache': 0,
              'intermediates': 0,
              'aqt':0,
              '_overwrite_with_gradient': 0,
          },
        split_rngs={
            'params': True,
            'dropout': False
        },
        #variable_broadcast=False,
        in_axes=(
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
        ),
        metadata_params={nn.PARTITION_NAME: 'layers'},
        )
        
        print("Finishing scanning!",flush=True)
        ouputs, _ = scan_fn(config=self.config, mesh=self.mesh, name='layers', quant=self.quant)(inputs,
            segmentation,
            positions,
            deterministic,
            model_mode,)
        return ouputs
    else:
       for layer_idx in range(self.num_layers):
            inputs = simple_decoder_layer.SimpleDecoderLayer(config=self.config, mesh=self.mesh, name=f'layers_{layer_idx}',quant=self.quant)(
            inputs,
            segmentation,
            positions,
            deterministic,
            model_mode,
        )
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