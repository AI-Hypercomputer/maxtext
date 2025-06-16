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
batch = 128
model = 5
expert_parallelism = 8
axis_name = "expert"

 # Define a mesh with PP + EP
device_mesh_array = mesh_utils.create_device_mesh((expert_parallelism,))
mesh = Mesh(device_mesh_array, ("expert",))

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


def vmap_wrapper(x, routing_table):
    # x is [batch, embed] (we can imagine batch=pdb * seq * exp_per_tok)
    # routing table is [batch, ep_groups] one hot
    output = xx

    input_offsets = yy


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
def wrapper(x, output, input_offsets, send_sizes, output_offsets, recv_sizes):
    # x is [batch, embed] (we can imagine batch=pdb * seq * exp_per_tok)
    batch_shard, _ = x.shape
    batch_per_ep_shard = batch_shard / expert_parallelism

    # output_shape is [batch * EP, embed]
    output_shape = jnp.tile(x, (expert_parallelism, 1))
    output_shape = x

    # input_offsets is [EP]
    input_offsets = jnp.array([i * batch_per_ep_shard for i in range(expert_parallelism)], dtype=jnp.int32)

    # send_sizes is [EP]
    send_sizes =jnp.array([batch_per_ep_shard for _ in range(expert_parallelism)], dtype=jnp.int32)

    # output_offsets is [EP]?
    output_offsets = jnp.array([i * batch_per_ep_shard for i in range(expert_parallelism)], dtype=jnp.int32)

    # recv_sizes is [EP]
    recv_sizes=jnp.array([batch_per_ep_shard for _ in range(expert_parallelism)], dtype=jnp.int32)
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


jit_wrapper = jax.jit(wrapper)
print(f"{x.shape=}", flush=True)
x_a2a = jit_wrapper(x)
print("Successfully ran wrapper (non - vmap)")
breakpoint()