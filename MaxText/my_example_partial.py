import functools
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax.experimental import shard_map


devices = jax.devices()

batch=512
embed=2048
stage = 2
fsdp = 2
tensor = 1

mesh = Mesh(mesh_utils.create_device_mesh((stage, fsdp, tensor), jax.devices()), ["stage", "fsdp", "tensor"])

def S(*specs):
  return NamedSharding(mesh, PartitionSpec(*specs))

def my_regular_activation_times_weight(activations, weights):
    assert activations.ndim == 2
    activations = lax.with_sharding_constraint(activations, S("fsdp", "tensor"))
    weights = lax.with_sharding_constraint(weights, S("fsdp", "tensor"))
    return activations @ weights

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
    # regular
    if 0:
        activations, weights = jax.jit(create_inputs)()
        jit_my_func = jax.jit(my_regular_activation_times_weight)
        output = jit_my_func(activations, weights)
        sum_outputs = jnp.sum(output)
        print(f"{sum_outputs=}", flush=True)

    # with stages
    create_input_vmap_func = jax.vmap(create_inputs, axis_size=stage, axis_name="stage")
    activations, weights = jax.jit(create_input_vmap_func)()
    print(f"{activations.shape}")
    jit_my_shard_map = jax.jit(my_partial_shard_map)
    ret = jit_my_shard_map(activations, weights)
    sum_outputs = jnp.sum(ret)
    print(f"{sum_outputs=}", flush=True)





