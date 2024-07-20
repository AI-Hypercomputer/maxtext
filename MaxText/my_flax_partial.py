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

batch=512
embed=2048
stage = 2
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
      (embed, embed)
    )

  def __call__(self, inputs: jnp.ndarray):
    # Assuming inputs are [batch, embed]
    return inputs @ self.weight_mat.astype(inputs.dtype)

sdl = SimpleDecoderLayer()
def func_to_pipeline(body_instance, inputs):
   return body_instance(inputs)

class MyMultipleBeast(nn.Module):
    @nn.compact
    def __call__(self, stage_inputs: jnp.ndarray):
       # Inputs should be [stage, batch, embed]
       vmap_sdl = nn.vmap(
          func_to_pipeline,
          in_axes=(0),
          spmd_axis_name='stage',
          variable_axes={'params': 0},
          split_rngs={'params':  self.is_initializing()},
          metadata_params={
            nn.PARTITION_NAME: "layers",
            'sub_weight_split_dims_mapping': (None),
            "is_initializing": self.is_initializing(),
            "x_times": self.num_stages}
       )
       return vmap_sdl(stage_inputs)

       

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
        activations = jnp.zeros((batch, embed), dtype=jnp.bfloat16)
        activations = lax.with_sharding_constraint(activations, S("fsdp", "tensor"))
        return activations

    def create_weights():
        weights = jnp.zeros((embed, embed), dtype=jnp.bfloat16)
        weights = lax.with_sharding_constraint(weights, S("fsdp" , "tensor"))
        return weights
    return create_activations(), create_weights()

with mesh:
    # regular w/o stages and w/o shard_map over stages
    if 0:
        activations, weights = jax.jit(create_inputs)()
        jit_my_func = jax.jit(my_regular_activation_times_weight)
        output = jit_my_func(activations, weights)
        sum_outputs = jnp.sum(output)
        print(f"{sum_outputs=}", flush=True)

    # with stages and with shard map
    create_input_vmap_func = jax.vmap(create_inputs, axis_size=stage, axis_name="stage")
    activations, weights = jax.jit(create_input_vmap_func)()
    print(f"{activations.shape}")
    jit_my_shard_map = jax.jit(my_partial_shard_map)
    ret = jit_my_shard_map(activations, weights)
    sum_outputs = jnp.sum(ret)
    print(f"{sum_outputs=}", flush=True)





