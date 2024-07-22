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


devices = jax.devices()


BATCH=512
EMBED=2048
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
          repeat_ids = jnp.array([2,0])
          #breakpoint()
          weights = self.gather_weights(repeat_ids)
          breakpoint()

       
       

       

@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(
        PartitionSpec("stage", None, None), #S("stage", "fsdp", "tensor"),
        PartitionSpec("stage", None, None)
    ),
    out_specs=PartitionSpec("stage", None, None),
    check_rep=False,
    auto = {"fsdp", "tensor"} # everything except stage
)
def my_partial_shard_map(activations, weights):
    # activations are shape [stage, batch, embed], the blocked version in this partial shard map
    # will be blocked along the stage axis stage times, so each block is only [1, batch, embed]
    # we want to remove this leading 1 dimension so it is the original shape w/o pipeling of [batch, embed]
    return my_regular_activation_times_weight(activations[0], weights[0])

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

with mesh:
    # regular w/o stages and w/o shard_map over stages
    if 1:
        activations, weights = jax.jit(create_inputs)()
        stage_inputs = jax.lax.broadcast(activations, [stage])
        sdl = SimpleDecoderLayer()
        my_pipeline = MyMultipleBeast(layers=sdl)
        init_pipeline_params = my_pipeline.init(jax.random.PRNGKey(0), stage_inputs)
        jit_pipeline = jax.jit(my_pipeline.apply)
        outputs = jit_pipeline(init_pipeline_params, stage_inputs)
        sum_outputs = jnp.sum(outputs)
        print(f"{sum_outputs=}", flush=True)

    # with stages and with shard map
    # create_input_vmap_func = jax.vmap(create_inputs, axis_size=stage, axis_name="stage")
    # activations, weights = jax.jit(create_input_vmap_func)()
    # print(f"{activations.shape}")
    # jit_my_shard_map = jax.jit(my_partial_shard_map)
    # ret = jit_my_shard_map(activations, weights)
    # sum_outputs = jnp.sum(ret)
    # print(f"{sum_outputs=}", flush=True)





