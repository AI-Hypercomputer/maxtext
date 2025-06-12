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
expert_parallelism = 2
pipeline_parallelism = num_devices // expert_parallelism
#batch = 2 * expert_parallelism**2
batch = 32
model = 5
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


tokens_routed_per_ep_group = batch / expert_parallelism**2
# Create a routing table representing an equal routing
all_shard_group_sizes = tokens_routed_per_ep_group * jnp.ones((expert_parallelism,expert_parallelism), dtype=jnp.int32)
all_shard_group_sizes = all_shard_group_sizes.astype(jnp.int32)

@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(x_partition_spec, None),
    out_specs=(out_partition_spec),
    check_rep=False,
)
def wrapper(x, all_shards_group_sizes):
    # x is [batch, embed] (we can imagine batch=pdb * seq * exp_per_tok)
    batch_shard, _ = x.shape
    batch_per_ep_shard = batch_shard / expert_parallelism

    # output_shape is worst case [batch * EP, embed]
    output_shape = jnp.tile(x, (expert_parallelism, 1))
    output_shape = x # This is best case, which our test case achieves

    input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params(all_shards_group_sizes)

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

def get_all_to_all_params(all_shards_group_sizes):
    class TransformStrategy(Enum):
        INPUT_OFFSET = auto()
        SEND_SIZE = auto()
        OUTPUT_OFFSET = auto()
        RECV_SIZE = auto()

    def transform_array(input_array, shard_id, strategy):
        """This function transforms the input array based on the specified strategy,
        preparing it for the usage with `ragged_all_to_all` API. The transformation
        determines how data is sent and received between shards.
        """
        if strategy == TransformStrategy.INPUT_OFFSET:
            # Index of input array for the send
            local_array = input_array[shard_id]
            return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
        elif strategy == TransformStrategy.SEND_SIZE:
            # Size of input array for the send
            return input_array[shard_id]
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
            # Received index in the target output
            zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
            array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
            cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
            return cumulated_array[shard_id]
        elif strategy == TransformStrategy.RECV_SIZE:
            # Received size in the target output
            return input_array[:, shard_id]
        else:
            raise ValueError(f"Unknown transform array strategy: {strategy}")

    local_id = jax.lax.axis_index("expert")
    input_offsets = transform_array(all_shards_group_sizes, local_id, TransformStrategy.INPUT_OFFSET)
    send_sizes = transform_array(all_shards_group_sizes, local_id, TransformStrategy.SEND_SIZE)
    output_offsets = transform_array(all_shards_group_sizes, local_id, TransformStrategy.OUTPUT_OFFSET)
    recv_sizes = transform_array(all_shards_group_sizes, local_id, TransformStrategy.RECV_SIZE)
    return input_offsets, send_sizes, output_offsets, recv_sizes




# Run without vmap: This should work
jit_wrapper = jax.jit(wrapper)
print(f"{x.shape=}", flush=True)
x_a2a = jit_wrapper(x, all_shard_group_sizes)
print("Successfully ran wrapper (non - vmap)")
print(x_a2a)



# Run with vmap: This doesn't work
# Errors with
# ValueError: all operands must have the same batch sizes
# I assume this is related to the shapes that are printed, such as
#
# input_offsets=Traced<int32[2]>with<BatchTrace> with
#   val = Traced<int32[1,2]>with<DynamicJaxprTrace>
#   batch_dim = 0

# send_sizes=Traced<int32[2]>with<BatchTrace> with
#   val = Traced<int32[4,2]>with<DynamicJaxprTrace>
#   batch_dim = 0

vmap_func = jax.vmap(
    wrapper,
    spmd_axis_name="pipeline",
)
jit_vmap_func = jax.jit(vmap_func)

x_vmap = jnp.expand_dims(x, axis=0)
x_vmap = jnp.tile(x_vmap, (pipeline_parallelism, 1, 1))
x_vmap = jax.device_put(x_vmap, NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", "expert", None)))

all_shard_group_sizes_vmap = jnp.expand_dims(all_shard_group_sizes, axis=0)
all_shard_group_sizes_vmap = jnp.tile(all_shard_group_sizes_vmap, (pipeline_parallelism, 1, 1))
all_shard_group_sizes_vmap = jax.device_put(all_shard_group_sizes_vmap, NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", None, None)))

x_vmap_a2a = jit_vmap_func(x_vmap, all_shard_group_sizes_vmap)
print(x_vmap_a2a.shape)
print("Successfully ran vmapped wrapper!", flush=True)


