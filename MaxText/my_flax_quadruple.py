import functools
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax.experimental import shard_map
from flax import linen as nn

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

devices = jax.devices()


BATCH=32768
EMBED=4096
REPEATS=3
stage = 2
GLOBAL_BATCH = stage * BATCH # assumes 1 microbatch per stage
fsdp = 2
tensor = 1


mesh = Mesh(mesh_utils.create_device_mesh((stage, fsdp, tensor), jax.devices()), ["stage", "fsdp", "tensor"])
logical_axis_rules =(
   ("batch", "fsdp"),
   ("embed", "tensor"),
   ("stage", "stage")
)


def S(*specs):
  return NamedSharding(mesh, PartitionSpec(*specs))

class SimpleDecoderLayer(nn.Module):
  def setup(self):
    self.weight_mat = self.param(
      'weights',
      nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
      (EMBED, EMBED)
    )

  def __call__(self, inputs: jnp.ndarray):
    # Assuming inputs are [batch, embed]
    return inputs @ self.weight_mat.astype(inputs.dtype)

class MyMultipleBeast(nn.Module):
    layers: nn.Module # The name of this property (layers) is reflected in the state pytree and thus also checkpoints.

    def get_main_vmap_func(self):
      def func_to_vmap(body_instance, inputs): #(body_instance, params, inputs)
        # nn.vmap requires either a nn.module class or a function whose first argument is a nn.module instance.
        return body_instance(inputs)
        #return body_instance.apply(params, inputs)
      return func_to_vmap

    def shard_dim_by_stages(self, x, dim: int):
      # Shards a dimension by stages. Currently the sharding of other dimensions are left up the compiler, alternatively
      # we may want to copy over the sharding from the other input axes
      dims_mapping = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
      dims_mapping[dim] = "stage"
      dims_mapping = tuple(dims_mapping)
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*dims_mapping))
      return jax.lax.with_sharding_constraint(x, sharding)

    def vmap_parallel_gather(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
      """Use vmap to implement a sharded parallel gather.
      Parallel gather means each stage has its own weights, and gets one slice from it.
      Args:
        weights: Per-stage data to be gathered from.
        repeat_ids: Integer tensor of shape [num_stages], the repeats of the stages.
        repeat_dim_in_weights: The dimension in weights where repeat_ids are applied. The output will not
          have this dimension.
        stages_dim_in_weights: The dimension in weights that represents parallel stages.
      Returns:
        The per-stage gathered values. The shape is weights.shape but with repeat_dim_in_weights
          removed.
      """
      def _gather_one(x, repeat_id):
        return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

      gathered_weights_stage_dim = 0
      repeat_ids = self.shard_dim_by_stages(repeat_ids, 0)
      weights = self.shard_dim_by_stages(weights, stages_dim_in_weights)
      stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(weights, repeat_ids)

      stage_weights = self.shard_dim_by_stages(stage_weights, gathered_weights_stage_dim)

      
      # may need to remove axis here = meta.remove_axis(weights, 0, circular_metadata_params)
      return stage_weights

    def gather_weights(self, repeat_ids):
       def gather_leaf(weights):
        return self.vmap_parallel_gather(weights, repeat_ids, 0, 1)
       return jax.tree.map(gather_leaf, self.variables)
       
    @nn.compact
    def __call__(self, stage_inputs: jnp.ndarray):
       # Inputs should be [stage, batch, embed]
       #return self.layers(stage_inputs)
       if 1:
        func_to_vmap = self.get_main_vmap_func() # sdl

        if self.is_initializing():
          vmap_sdl = nn.vmap(
              func_to_vmap,
              in_axes=(0),
              spmd_axis_name='stage',
              variable_axes={'params': 0},
              split_rngs={'params':  self.is_initializing()},
              metadata_params={
                nn.PARTITION_NAME: "layers",
                'sub_weight_split_dims_mapping': (None),
                "is_initializing": self.is_initializing(),
                "x_times": stage}
          )



          circ_vmap_sdl = nn.vmap(
          vmap_sdl,
          in_axes=(0),
            variable_axes={
              'params': 0,
              "non_trainable": 0,
              "hyper_params": 0,
            },
            split_rngs={'params': True},
            metadata_params={
              nn.PARTITION_NAME: "circular_repeats",
              'sub_weight_split_dims_mapping': (None,), 
              "is_initializing": True,
              "x_times": REPEATS,
              'optimizer_dims_mapping': None,
            }
          ) 

          example_inputs = jax.lax.broadcast(stage_inputs, [REPEATS])
          # We only need to run one set of stages to initialize the variables, instead of looping over all microbatches for the full total_iterations.
          stage_outputs = circ_vmap_sdl(self.layers, example_inputs)
          stage_outputs = stage_outputs[0] # Remove extra dimension created for the circular vmap
          broadcasted_stage_outpus = jax.lax.broadcast(stage_outputs[0], [GLOBAL_BATCH // BATCH])
          return jnp.reshape(broadcasted_stage_outpus, [GLOBAL_BATCH, EMBED])
       
       if not self.is_initializing():
          breakpoint()
          repeat_ids = jnp.array([2,0])
          stage_weights = self.gather_weights(repeat_ids)
          circular_metadata_params={
            nn.PARTITION_NAME: "circular_repeats",
            'sub_weight_split_dims_mapping': (None,),
            "is_initializing": self.is_initializing(),
            "x_times": REPEATS,
            'optimizer_dims_mapping': None,
          }
          from flax.core import meta
          stage_weights = meta.remove_axis(stage_weights, 0, circular_metadata_params)
          # remove layers key from pytree, e.g. instead of params -> layers -> weights, just params -> weights
          stage_weights['params'] = stage_weights['params']['layers']

          def get_stage_partition_spec(weight_partition_spec):
             def get_stage_partition_spec_leaf(leaf_partition_spec):
                new_partition_spec = [None] * len(leaf_partition_spec)
                new_partition_spec[0] = "stage"
                return PartitionSpec(*new_partition_spec)
             partition_spec_tree = jax.tree.map(get_stage_partition_spec_leaf, weight_partition_spec)

             return partition_spec_tree

          stage_weight_partition_spec = get_stage_partition_spec(nn.get_partition_spec(stage_weights))
          #breakpoint()

          def remove_leading_singleton_stage_dim(arr_pytree):
             def remove_leading_singleton_stage_dim_leaf(arr):
                assert arr.shape[0] == 1
                # may need some flax remove metadata
                return arr[0]
             return jax.tree.map(remove_leading_singleton_stage_dim_leaf, arr_pytree)

          @functools.partial(
            shard_map.shard_map,
            mesh=mesh,
            in_specs=(
                PartitionSpec("stage", None, None), # [stage, batch ,embed]
                stage_weight_partition_spec, # weights partition spec
            ),
            out_specs=PartitionSpec("stage", None, None),
            check_rep=False,
            auto = {"fsdp", "tensor"} # everything except stage
          )
          def run_microbatch_iteration(individual_activations, individual_weights):
              # Remove leading singleton stage dimension
              individual_activations = remove_leading_singleton_stage_dim(individual_activations)
              individual_weights = remove_leading_singleton_stage_dim(individual_weights)
              
              # run computation
              #breakpoint()
              output_activation = self.layers.apply(individual_weights, individual_activations)

              # permute
              jax.lax.ppermute(output_activation, 'stage', [(i, (i+1) % stage) for i in range(stage)])

              # add back a stage dimension
              output_activation=jnp.expand_dims(output_activation, axis=0)


              return output_activation
          
          return run_microbatch_iteration(stage_inputs, stage_weights)

def create_inputs():
    def create_activations():
        activations = jnp.zeros((BATCH, EMBED), dtype=jnp.bfloat16)
        activations = lax.with_sharding_constraint(activations, S("fsdp", "tensor"))
        return activations

    def create_weights():
        weights = jnp.zeros((EMBED, EMBED), dtype=jnp.bfloat16)
        weights = lax.with_sharding_constraint(weights, S("fsdp" , "tensor"))
        return weights
    return create_activations(), create_weights()


class myDecoder(nn.Module):
   @nn.compact
   def __call__(self, inputs):
      sdl = SimpleDecoderLayer()
      return MyMultipleBeast(layers=sdl)(inputs)

class myTransformer(nn.Module):

   def setup(self):
      self.decoder = myDecoder()
   def __call__(self, inputs):
    return self.decoder(inputs)










with mesh:
    activations, weights = jax.jit(create_inputs)()
    stage_inputs = jax.lax.broadcast(activations, [stage])
    sdl = SimpleDecoderLayer()
    my_pipeline = MyMultipleBeast(layers=sdl)

    run_transformer = True
    run_pipeline = False
    if run_transformer:
        # transformer
        my_transformer = myTransformer()
        init_transformer_params= my_transformer.init(jax.random.PRNGKey(0), stage_inputs)
        jit_transformer = jax.jit(my_transformer.apply)
        #timing_util.simple_timeit(jit_transformer, init_transformer_params, stage_inputs, task = 'transformer')
        outputs = jit_transformer(init_transformer_params, stage_inputs)
        sum_outputs = jnp.sum(outputs)
        print(f"{sum_outputs=}", flush=True)


    if run_pipeline:
        # pipeline 
        import timing_util
        init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), stage_inputs)
        jit_pipeline = jax.jit(my_pipeline.apply)
        #timing_util.simple_timeit(jit_pipeline, init_pipeline_params, stage_inputs, task = 'partial_shard_pp')
        outputs = jit_pipeline(init_pipeline_params, stage_inputs)
        sum_outputs = jnp.sum(outputs)
        print(f"{sum_outputs=}", flush=True)


