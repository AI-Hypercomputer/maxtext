import jax
import jax.numpy as jnp
from jax import lax, jit
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto
from jax.sharding import PartitionSpec as P

GLOBAL_BATCH=128
MODEL=2048
FF=8192
NUM_EXP=8
EP=8
BATCH_PER_EP_SHARD = GLOBAL_BATCH // EP
assert EP * BATCH_PER_EP_SHARD == GLOBAL_BATCH, "Global Batch must be a multiple of EP"
BATCH_PER_EP_SHARD_PER_EXP = BATCH_PER_EP_SHARD // NUM_EXP # aka block assignment per shard or something....
assert NUM_EXP * BATCH_PER_EP_SHARD_PER_EXP == BATCH_PER_EP_SHARD, "Global Batch must be a multiple of (EP * EXP)"

print_a2a_input_vars=True



 # 1D EP mesh
device_mesh_array = mesh_utils.create_device_mesh((EP,))
mesh = Mesh(device_mesh_array, ("expert"))
x_partition_spec = jax.sharding.PartitionSpec("expert", None)
weight_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
explicit_ep_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
x_sharding = NamedSharding(mesh, x_partition_spec)

@functools.partial(
    shard_map,
    mesh=mesh,
    in_specs=(
        explicit_ep_partition_spec,
        weight_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        ),
    out_specs=(explicit_ep_partition_spec),
)
def ra2a_and_compute(x_input, weights, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes):
    # remove singleton leading axis of x
    x = x_input.reshape(x_input.shape[1:])
    output_shape = output_shape.reshape(output_shape.shape[1:])

    input_offsets = input_offsets.reshape(input_offsets.shape[1:])
    send_sizes = send_sizes.reshape(send_sizes.shape[1:])
    output_offsets = output_offsets.reshape(output_offsets.shape[1:])
    recv_sizes = recv_sizes.reshape(recv_sizes.shape[1:])

    if print_a2a_input_vars:
        print("Printing shapes right before ra2a call:", flush=True)
        print(f"{x=}\n")
        print(f"{output_shape=}\n")
        print(f"{input_offsets=}\n")
        print(f"{send_sizes=}\n")
        print(f"{output_offsets=}\n")
        print(f"{recv_sizes=}\n")
    output = jax.lax.ragged_all_to_all(
        x,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name="expert",
    )
    # add a leading singleton dimension to output
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

jit_wrapper = jax.jit(ra2a_wrapper)
x_a2a = jit_wrapper(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes)

