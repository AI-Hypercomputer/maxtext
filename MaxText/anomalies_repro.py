import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto
from jax.sharding import PartitionSpec as P
import tempfile
import os
import datetime




num_devices = len(jax.devices()) # we expect either 4 or 8 total devices
expert_parallelism = 4
pipeline_parallelism = num_devices // expert_parallelism 


 # Define a mesh with PP + EP
device_mesh_array = mesh_utils.create_device_mesh((expert_parallelism, pipeline_parallelism))
mesh = Mesh(device_mesh_array, ("expert", "dummy"))
x_partition_spec = jax.sharding.PartitionSpec("expert", None)
x_sharding = NamedSharding(mesh, x_partition_spec)
axis_name = "expert"

@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        x_partition_spec,
        ),
    out_specs=(x_partition_spec),
    check_rep=False,
)
def ra2a_wrapper(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes):
    input_offsets = input_offsets.reshape(input_offsets.shape[1:])
    send_sizes = send_sizes.reshape(send_sizes.shape[1:])
    output_offsets = output_offsets.reshape(output_offsets.shape[1:])
    recv_sizes = recv_sizes.reshape(recv_sizes.shape[1:])

    output = jax.lax.ragged_all_to_all(
        x,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis_name,
    )
    return output



def run_ragged_a2a(batch_per_ep, model):
    batch = batch_per_ep * expert_parallelism

    def create_x():
      x = jnp.arange(0.0, batch)
      x = jnp.expand_dims(x, axis=1)
      x = jnp.tile(x, (1, model))
      x = jnp.astype(x, jnp.float32)
      x = jax.device_put(x, x_sharding)
      return x
    x = jax.jit(create_x)()
    x.block_until_ready()

    output_shape = x.copy()

    
    #input_offsets_i_j is where in EP_i input we start grabbing index to send to EP_j
    ep_0 = [batch_per_ep * n for n in range(expert_parallelism)]
    input_offsets = [ep_0.copy() for _ in range(expert_parallelism)]
    input_offsets = jnp.array(input_offsets, dtype=jnp.int32)
    input_offsets = jax.device_put(input_offsets, x_sharding)

    # send_sizes_i_j is the ith EP send size to EP j
    ep_0 = [batch_per_ep for _ in range(expert_parallelism)]
    send_sizes = [ep_0.copy() for _ in range(expert_parallelism)]
    send_sizes = jnp.array(send_sizes, dtype=jnp.int32)
    send_sizes = jax.device_put(send_sizes, x_sharding)

    #output_offsets_i_j is where in EP_i needs to write the outputs in EP_j
    output_offsets = [[batch_per_ep * n for _ in range(expert_parallelism)] for n in range(expert_parallelism)]
    output_offsets = jnp.array(output_offsets, dtype=jnp.int32)
    output_offsets = jax.device_put(output_offsets, x_sharding)

    # recv_sizes_i_j is the ith EP rec size from EP j
    recv_sizes = send_sizes.copy()
    recv_sizes = jax.device_put(recv_sizes, x_sharding)
    jit_a2a_func = jax.jit(ra2a_wrapper)


    compiled = jit_a2a_func.lower(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes).compile()
    compiled_stats = compiled.memory_analysis()
    if compiled_stats is not None:
        print(
            f"Output size: {compiled_stats.output_size_in_bytes/10**9}, "
            f"temp size: {compiled_stats.temp_size_in_bytes/10**9}, "
            f"argument size: {compiled_stats.argument_size_in_bytes/10**9}, "
            f"host temp size: {compiled_stats.host_temp_size_in_bytes/10**9}, in GB."
        )

    output_a2a = jit_a2a_func(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes)
    return output_a2a


batch_per_ep = 1024

model = 4096 # Works with both bf16 and float32
model = 8192 # Works with bf16, crashes with float32
model = 16384 # Crashes with both bf16 and float32


output_a2a = run_ragged_a2a(batch_per_ep, model)
jax.block_until_ready(output_a2a)
print("successfully ran ragged a2a!")
