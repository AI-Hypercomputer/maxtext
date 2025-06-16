import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto
from jax.sharding import PartitionSpec as P


# Define inputs
num_devices = len(jax.devices())
expert_parallelism = 2
pipeline_parallelism = num_devices // expert_parallelism
batch = 2 * expert_parallelism**2
model = 3
axis_name = "expert"

 # Define a mesh with PP + EP
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

# Assign rows to expert groups in a symmetric balanced way
routing_table = jnp.zeros((2 * expert_parallelism, expert_parallelism), dtype=jnp.int32)
for ep_shard in range(expert_parallelism):
    routing_table = routing_table.at[2 * ep_shard, ep_shard].set(1)
    routing_table = routing_table.at[2 * ep_shard + 1, ep_shard].set(1)
routing_table = jnp.tile(routing_table, (expert_parallelism, 1))

routing_table_partition_spec = jax.sharding.PartitionSpec("expert", None)
routing_table_sharding = NamedSharding(mesh, routing_table_partition_spec)
routing_table = jax.device_put(routing_table, routing_table_sharding)


out_partition_spec = x_partition_spec
output_sharding = x_sharding


def main_wrapper(x, routing_table):
    # x is [batch, embed] (we can imagine batch=pdb * seq * exp_per_tok)
    # routing table is [batch, ep_groups] one hot
    input_offsets, send_sizes, output_offsets, recv_sizes = get_ra2a_inputs(routing_table)
    output_shape = x
    # output_shape = jnp.tile(x, (expert_parallelism, 1)) # necessary for worst case in real models
    output = ra2a_wrapper(
        x,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
    )
    return output

@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(
        x_partition_spec,
    ),
    out_specs=(
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
    ),
    check_rep=False,
)
def get_ra2a_inputs(routing_table):
    # routing table is [batch, ep_groups] one hot
    local_expert_size, num_ep_groups = routing_table.shape
    reshaped_group_sizes = jnp.sum(routing_table, axis=0)
    all_shards_group_sizes = lax.all_gather(reshaped_group_sizes, axis_name="expert")
    
    input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params(all_shards_group_sizes)
    input_offsets = jnp.expand_dims(input_offsets, axis=0)
    send_sizes = jnp.expand_dims(send_sizes, axis=0)
    output_offsets = jnp.expand_dims(output_offsets, axis=0)
    recv_sizes = jnp.expand_dims(recv_sizes, axis=0)
    
    return input_offsets, send_sizes, output_offsets, recv_sizes

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
    out_specs=(out_partition_spec),
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

    all_shards_group_sizes = 2 * jnp.ones((expert_parallelism,expert_parallelism), dtype=jnp.int32)
    input_offsets_2, send_sizes_2, output_offsets_2, recv_sizes_2 = get_all_to_all_params(all_shards_group_sizes)
    output_2 = jax.lax.ragged_all_to_all(
        x,
        output_shape,
        input_offsets_2,
        send_sizes_2,
        output_offsets+2,
        recv_sizes_2,
        axis_name=axis_name,
    )
    print(f"{output_2.shape=}\n")
    return output_2


##### Non-vmap #####
jit_wrapper = jax.jit(main_wrapper)
print(f"{x.shape=}", flush=True)
x_a2a = jit_wrapper(x, routing_table)
print("Successfully ran wrapper (non - vmap)")
print(x_a2a)
breakpoint()


##### Vmap #####
vmap_func = jax.vmap(
    main_wrapper,
    spmd_axis_name="pipeline",
)
jit_vmap_func = jax.jit(vmap_func)

x_vmap = jnp.expand_dims(x, axis=0)
x_vmap = jnp.tile(x_vmap, (pipeline_parallelism, 1, 1))
x_vmap = jax.device_put(x_vmap, NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", "expert", None)))

routing_table = jnp.expand_dims(routing_table, axis=0)
routing_table = jnp.tile(routing_table, (pipeline_parallelism, 1, 1))
routing_table = jax.device_put(routing_table, NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", "expert", None)))
x_a2a = jit_vmap_func(x_vmap, routing_table)
print("Successfully ran vmap!!")
print(x_a2a)
breakpoint()