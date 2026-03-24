import jax
import jax.numpy as jnp
from jax import lax, jit
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto
from jax.sharding import PartitionSpec as P
import tokamax

GLOBAL_BATCH=131072
MODEL=2048
FF=8192
NUM_EXP=8
EP=8
EXP_PER_SHARD = NUM_EXP // EP
assert EP * EXP_PER_SHARD == NUM_EXP, "Experts must be divisible by EP"
BATCH_PER_EP_SHARD = GLOBAL_BATCH // EP
assert EP * BATCH_PER_EP_SHARD == GLOBAL_BATCH, "Global Batch must be a multiple of EP"
BATCH_PER_EP_SHARD_PER_EXP = BATCH_PER_EP_SHARD // NUM_EXP # aka block assignment per shard or something....
assert NUM_EXP * BATCH_PER_EP_SHARD_PER_EXP == BATCH_PER_EP_SHARD, "Global Batch must be a multiple of (EP * EXP)"

 # 1D EP mesh
device_mesh_array = mesh_utils.create_device_mesh((EP,))
mesh = Mesh(device_mesh_array, ("expert")) 

activation_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
weight_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
groups_partition_spec = jax.sharding.PartitionSpec("expert", None)



@functools.partial(
    shard_map,
    mesh=mesh,
    in_specs=(
        activation_partition_spec,
        weight_partition_spec,
        groups_partition_spec,
        ),
    out_specs=(activation_partition_spec),
    check_vma=False,
)
def minimal_tokamax(activations, weights, group_sizes):
    # remove leading singleton dimension of activations
    activations = activations.reshape(activations.shape[1:])
    group_sizes = group_sizes.reshape(group_sizes.shape[1:])

    output = tokamax.ragged_dot(activations, weights, group_sizes, implementation="mosaic")
    output = jnp.expand_dims(output, axis=0)
    return output



activations = jnp.zeros([EP, BATCH_PER_EP_SHARD, MODEL])
weights = jnp.zeros([NUM_EXP, MODEL, FF])

tokens_per_local_expert = BATCH_PER_EP_SHARD / EXP_PER_SHARD
group_sizes = jnp.array([tokens_per_local_expert for _ in range(EXP_PER_SHARD)], dtype=jnp.int32)
group_sizes = jnp.tile(jnp.expand_dims(group_sizes, axis=0), (EP, 1))

jit_wrapper = jax.jit(minimal_tokamax)
tokamax_output = jit_wrapper(activations, weights, group_sizes)
breakpoint()