import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto
from jax.sharding import PartitionSpec as P

num_devices = len(jax.devices()) # we expect either 4 or 8 total devices
expert_parallelism = 2
assert expert_parallelism==2, "This script only supports EP=2"
pipeline_parallelism = num_devices // expert_parallelism # We expect this is either 2 or 4
batch = 2 * expert_parallelism**2
model = 3
axis_name = "expert"
print_a2a_input_vars=True

 # Define a mesh with PP + EP
device_mesh_array = mesh_utils.create_device_mesh((expert_parallelism, pipeline_parallelism))
mesh = Mesh(device_mesh_array, ("expert", "pipeline"))
x_partition_spec = jax.sharding.PartitionSpec("expert", None)
x_sharding = NamedSharding(mesh, x_partition_spec)

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
        axis_name=axis_name,
    )
    return output


# create an array x which is [batch, model] and has elements like
# [[0,0,0],
#  [1,1,1],
#  ...
x = jnp.arange(0.0, batch)
x = jnp.expand_dims(x, axis=1)
x = jnp.tile(x, (1, model))
x = jax.device_put(x, x_sharding)

output_shape = x.copy()

input_offsets = jnp.array([[0, 2],[0,2]], dtype=jnp.int32)
input_offsets = jax.device_put(input_offsets, x_sharding)

send_sizes = jnp.array([[2, 2],[2,2]], dtype=jnp.int32)
send_sizes = jax.device_put(send_sizes, x_sharding)

output_offsets = jnp.array([[0, 0],[2,2]], dtype=jnp.int32)
output_offsets = jax.device_put(output_offsets, x_sharding)

recv_sizes = jnp.array([[2, 2],[2,2]], dtype=jnp.int32)
recv_sizes = jax.device_put(recv_sizes, x_sharding)

expected_array = jnp.array([[0,0,0],[1,1,1],[4,4,4],[5,5,5],[2,2,2],[3,3,3],[6,6,6],[7,7,7]], dtype=jnp.int32)

##### Non-vmap #####
jit_wrapper = jax.jit(ra2a_wrapper)
print(f"{x.shape=}", flush=True)
x_a2a = jit_wrapper(x, output_shape, input_offsets, send_sizes, output_offsets, recv_sizes)
print(f"output of a2a without vmap:\n {x_a2a}")
assert x_a2a.shape == (batch, model)
print("Successfully ran wrapper (non - vmap)")
print("Now running expected assert...")
assert  jnp.array_equal(x_a2a, expected_array)
print("Regular (non-vmap) output has expected values!")


#### Vmap #####
vmap_func = jax.vmap(
    ra2a_wrapper,
    #spmd_axis_name="pipeline",
)
jit_vmap_func = jax.jit(vmap_func)

vmap_sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", "expert", None))

def expand_array_for_vmap(arr):
    arr = jnp.expand_dims(arr, axis=0)
    arr = jnp.tile(arr, (pipeline_parallelism, 1, 1))
    arr = jax.device_put(arr, vmap_sharding)
    return arr


x_vmap = expand_array_for_vmap(x)
output_shape_vmap = expand_array_for_vmap(output_shape)
input_offsets_vmap = expand_array_for_vmap(input_offsets)
send_sizes_vmap = expand_array_for_vmap(send_sizes)
output_offsets_vmap = expand_array_for_vmap(output_offsets)
recv_sizes_vmap = expand_array_for_vmap(recv_sizes)

vmap_output = jit_vmap_func(x_vmap, output_shape_vmap, input_offsets_vmap, send_sizes_vmap, output_offsets_vmap, recv_sizes_vmap)
print(f"output of a2a WITH vmap:\n {vmap_output}")
print(f"vmap_output.shape = {vmap_output.shape}")
print("Successfully ran vmap!!")
# This will fail! The output shape is [PP, 2, 4, 3] but we expect [PP, 8, 3] - the same shape as both x_vmap and output_shape_vmap
# vmap_output = vmap_output.reshape([4,8,3])
# print(f"output of a2a WITH vmap:\n {vmap_output}")
assert vmap_output.shape == (pipeline_parallelism, batch, model)
print("Now running expected assert...")
for i in range(pipeline_parallelism):
    assert  jnp.array_equal(vmap_output[i,:,:], expected_array)
print("Vmapped output has expected values!")