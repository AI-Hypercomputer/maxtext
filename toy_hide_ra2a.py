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

GLOBAL_BATCH=1024
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

print_a2a_input_vars=True



 # 1D EP mesh
device_mesh_array = mesh_utils.create_device_mesh((EP,))
mesh = Mesh(device_mesh_array, ("expert"))
x_partition_spec = jax.sharding.PartitionSpec("expert", None)
explicit_ep_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
x_sharding = NamedSharding(mesh, x_partition_spec)

@functools.partial(
    shard_map,
    mesh=mesh,
    in_specs=(
        explicit_ep_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        explicit_ep_partition_spec

        ),
    out_specs=(explicit_ep_partition_spec),
    check_vma=False,
)
def ra2a_gmm(x_input, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes, group_sizes, weights):
    # remove singleton leading axis of x
    x = x_input.reshape(x_input.shape[1:])
    output_shape = output_shape.reshape(output_shape.shape[1:])
    group_sizes = group_sizes.reshape(group_sizes.shape[1:])

    input_offsets = input_offsets.reshape(input_offsets.shape[1:])
    send_sizes = send_sizes.reshape(send_sizes.shape[1:])
    output_offsets = output_offsets.reshape(output_offsets.shape[1:])
    recv_sizes = recv_sizes.reshape(recv_sizes.shape[1:])

    ra2a_output = jax.lax.ragged_all_to_all(
        x,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )

    output = tokamax.ragged_dot(ra2a_output, weights, group_sizes, implementation="mosaic")
    output = jnp.expand_dims(output, axis=0)

    return output


# create an array x which is [batch, model] and has elements like
# [[0,0,0],
#  [0,0,0],
#  [1,1,1],
#  [1,1,1],
#  ...
# where each block of equal values is of size BATCH_PER_EP_SHARD_PER_EXP
x = [[exp_idx for _ in range(BATCH_PER_EP_SHARD_PER_EXP)] for exp_idx in range(NUM_EXP)]
x = jnp.array(x).flatten()
x = jnp.expand_dims(x, axis=1)
x = jnp.tile(x, (1, MODEL))
x = jnp.expand_dims(x, axis=0)
x = jnp.tile(x, (EP, 1, 1))

output_shape = x.copy()

input_offsets = [[BATCH_PER_EP_SHARD_PER_EXP * exp_idx for exp_idx in range(NUM_EXP)] for _ in range(EP)]
input_offsets = jnp.array(input_offsets, dtype=jnp.int32)

output_offsets = jnp.transpose(input_offsets)

send_sizes = jnp.array([[BATCH_PER_EP_SHARD_PER_EXP for _ in range(NUM_EXP)] for _ in range(EP)], dtype=jnp.int32)
recv_sizes = jnp.array([[BATCH_PER_EP_SHARD_PER_EXP for _ in range(NUM_EXP)] for _ in range(EP)], dtype=jnp.int32)

tokens_per_local_expert = BATCH_PER_EP_SHARD / EXP_PER_SHARD
group_sizes = jnp.array([tokens_per_local_expert for _ in range(EXP_PER_SHARD)], dtype=jnp.int32)
group_sizes = jnp.tile(jnp.expand_dims(group_sizes, axis=0), (EP, 1))

weights = jnp.zeros([NUM_EXP, MODEL, FF])

jit_wrapper = jax.jit(ra2a_gmm)
output = jit_wrapper(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes, group_sizes, weights)

breakpoint()
