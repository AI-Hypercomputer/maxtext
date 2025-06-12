import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto


# Define inputs
num_devices = len(jax.devices())
batch = 8
model = 5
expert_parallelism = 2
pipeline_parallelism = num_devices // expert_parallelism
axis_name = "expert"

 # Define a 1D mesh only EP
device_mesh_array = mesh_utils.create_device_mesh((expert_parallelism, pipeline_parallelism))
mesh = Mesh(device_mesh_array, ("expert", "pipeline"))

# create an array x which is [batch, model] and has elements like
# [[0,0,0],
#  [1,1,1],
#  ...
x = jnp.arange(0.0, batch)
x = jnp.expand_dims(x, axis=1)
x = jnp.tile(x, (1, model))

x_partition_spec = jax.sharding.PartitionSpec("expert", None)
x_sharding = NamedSharding(mesh, x_partition_spec)
x = jax.device_put(x, x_sharding)

out_partition_spec = x_partition_spec
output_sharding = x_sharding

@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(x_partition_spec),
    out_specs=(out_partition_spec),
    check_rep=False,
)
def wrapper(x):
    # x is [batch, embed] (we can imagine batch=pdb * seq * exp_per_tok)
    batch_shard, _ = x.shape
    batch_per_ep_shard = batch_shard / expert_parallelism

    # output_shape is worst case [batch * EP, embed]
    output_shape = jnp.tile(x, (expert_parallelism, 1))
    output_shape = x # This is best case, which our test case achieves

    # input_offsets,
    input_offsets = jnp.array([i * batch_per_ep_shard for i in range(expert_parallelism)], dtype=jnp.int32)

    # send_sizes is [EP]
    send_sizes =jnp.array([batch_per_ep_shard for _ in range(expert_parallelism)], dtype=jnp.int32)

    # output_offsets is [EP]
    #output_offsets = jnp.array([i * batch_per_ep_shard for i in range(expert_parallelism)], dtype=jnp.int32)
    local_id = jax.lax.axis_index("expert")
    output_offsets = local_id * batch_per_ep_shard * jnp.ones((expert_parallelism,), dtype=jnp.int32)
    output_offsets = output_offsets.astype(jnp.int32)

    # recv_sizes is [EP]
    recv_sizes=jnp.array([batch_per_ep_shard for _ in range(expert_parallelism)], dtype=jnp.int32)
    

    print(f"{x=}\n")
    print(f"{output_shape=}\n")
    print(f"{input_offsets=}\n")
    print(f"{send_sizes=}\n")
    print(f"{output_offsets=}\n")
    print(f"{recv_sizes=}\n")
    # Strangely jax.debug.print doesn't print anything...?
    # jax.debug.print("{}\n", x)
    # jax.debug.print("{}\n", output_shape)
    # jax.debug.print("{}\n", input_offsets)
    # jax.debug.print("{}\n", send_sizes)
    # jax.debug.print("{}\n", output_offsets)
    # jax.debug.print("{}\n", recv_sizes)
    output = jax.lax.ragged_all_to_all(
        x,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis_name,
    )
    print(f"{output.shape=}\n")
    return output


# Non vmap version - this should work
# jit_wrapper = jax.jit(wrapper)
# print(f"{x.shape=}", flush=True)
# x_a2a = jit_wrapper(x)
# print("Successfully ran wrapper (non - vmap)")
# print(x_a2a)


# Vmapped version - this will fail
# 
# Traceback (most recent call last):
#   File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
#     return _run_code(code, main_globals, None,
#   File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
#     exec(code, run_globals)
#   File "/home/mattdavidow/maxtext/MaxText/rv_minimal.py", line 99, in <module>
#     x_vmap_a2a = jit_vmap_func(x_vmap)
#   File "/home/mattdavidow/maxtext/MaxText/rv_minimal.py", line 69, in wrapper
#     output = jax.lax.ragged_all_to_all(
# TypeError: tuple indices must be integers or slices, not NoneType
# 
# Likely due to shapes that look like below
# x=Traced<float32[4,5]>with<BatchTrace> with
#   val = Traced<float32[1,4,5]>with<DynamicJaxprTrace>
#   batch_dim = 0

# output_shape=Traced<float32[4,5]>with<BatchTrace> with
#   val = Traced<float32[1,4,5]>with<DynamicJaxprTrace>
#   batch_dim = 0

# input_offsets=Traced<int32[2]>with<DynamicJaxprTrace>

# send_sizes=Traced<int32[2]>with<DynamicJaxprTrace>

vmap_func = jax.vmap(
    wrapper,
    spmd_axis_name="pipeline",
)
jit_vmap_func = jax.jit(vmap_func)
x_vmap = jnp.expand_dims(x, axis=0)
x_vmap = jnp.tile(x_vmap, (pipeline_parallelism, 1, 1))
x_vmap = jax.device_put(x_vmap, NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", "expert", None)))
x_vmap_a2a = jit_vmap_func(x_vmap)
print(x_vmap_a2a.shape)
print("Successfully ran vmapped wrapper!", flush=True)


