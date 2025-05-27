import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.experimental import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto


def get_all_to_all_params(all_shards_group_sizes, local_expert_size, num_expert_shards):
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

def permute(inputs, gate_logits):
    """Permute tokens to group by expert to fit gmm call."""
    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
    weights, selected_experts = get_topk(gate_logits)

    flatten_selected_experts = jnp.ravel(selected_experts)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    sorted_indices = sorted_selected_experts // num_experts_per_tok
    # sort inputs for number of selected experts
    sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(jnp.bfloat16)
    group_size = jnp.bincount(flatten_selected_experts, length=num_experts)
    return sorted_inputs, sorted_selected_experts, weights, group_size

def get_topk(gate_logits):
    top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, num_experts_per_tok)
    return top_k_weights, top_k_indices

# Define inputs
per_device_batch = 2
num_devices = len(jax.devices())
batch_size = per_device_batch * num_devices
sequence_length = 3
model_dim = 5
num_experts = 8
num_experts_per_tok = 2
expert_parallelism = 4
pipeline_parallelism = num_devices // expert_parallelism
axis_name = "expert"
random_routing = False
print_a2a_input_vars = False
hack_output_for_vmap = True
run_vmap=True

 # Define a mesh with one axis named "expert"
device_mesh_array = mesh_utils.create_device_mesh((pipeline_parallelism, expert_parallelism))
mesh = Mesh(device_mesh_array, ("pipeline", "expert"))

# Create inputs x and logits which are sharded only by expert
total_elements = batch_size * sequence_length * model_dim
x = jnp.arange(1.0, total_elements + 1.0).reshape(batch_size, sequence_length, model_dim)
x_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
x_sharding = NamedSharding(mesh, x_partition_spec)
x = jax.device_put(x, x_sharding)

logits = jnp.zeros((batch_size, sequence_length, model_dim))
logits_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
logits_sharding = NamedSharding(mesh, logits_partition_spec)
logits = jax.device_put(logits, logits_sharding)

output_sharding = jax.sharding.PartitionSpec("expert", None)

@functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(x_partition_spec, logits_partition_spec),
    out_specs=(x_partition_spec), #output_sharding
    check_rep=False,
)
def wrapper(x, logits):
    # get group sizes for all shards
    batch_size, sequence_length, _ = x.shape
    x, sorted_selected_experts, weights, group_sizes = permute(x, logits)
    local_expert_size = num_experts // expert_parallelism
    reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
    all_shards_group_sizes = lax.all_gather(reshaped_group_sizes, axis_name=axis_name)
    # calculate offsets and sizes for ragged_all_to_all operation
    input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params(
        all_shards_group_sizes, local_expert_size, expert_parallelism
    )

    buffer_size = int(expert_parallelism * per_device_batch * sequence_length * num_experts_per_tok)
    output_shape = jnp.zeros((buffer_size, model_dim), dtype=x.dtype)

    if hack_output_for_vmap:
        # returns the same shape, but this will get vmap transformed since it builds from x??
        output_shape = jnp.tile(x, (expert_parallelism, 1))

    if print_a2a_input_vars:
        jax.debug.print(f"{x=}\n")
        jax.debug.print(f"{output_shape=}\n")
        jax.debug.print(f"{input_offsets=}\n")
        jax.debug.print(f"{send_sizes=}\n")
        jax.debug.print(f"{output_offsets=}\n")
        jax.debug.print(f"{recv_sizes=}\n")

    # The main event: A2A - this is where things crash in the vmap case unless we set hack_output_for_vmap=True
    x = jax.lax.ragged_all_to_all(
        x,
        output_shape,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis_name,
    )
    jax.debug.print(f"{x.shape=}\n")
    return x



# This works
if not run_vmap:
    jit_wrapper = jax.jit(wrapper)
    x_a2a = jit_wrapper(x, logits)
    print("Successfully ran wrapper (non - vmap)")

else:
    # We now want to try to emulate how maxtext (and pax) SPMD PP works - which is by vmapping
    # over the PP axis. All inputs will get a new leading dimension of length PP that is sharded by
    # PP and will be vmapped over
    vmap_func = jax.vmap(
        wrapper,
        spmd_axis_name="pipeline",
    )
    jit_vmap_func = jax.jit(vmap_func)

    # Add a leading dimension to x and logits of length pipeline_parallelism, and shard it by pipeline
    x_vmap = jnp.expand_dims(x, axis=0)
    x_vmap = jnp.tile(x_vmap, (pipeline_parallelism, 1, 1, 1))
    x_vmap = jax.device_put(x_vmap, NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", "expert", None, None)))

    logits_vmap = jnp.expand_dims(logits, axis=0)
    logits_vmap = jnp.tile(logits_vmap, (pipeline_parallelism, 1, 1, 1))
    logits_vmap = jax.device_put(logits_vmap, NamedSharding(mesh, jax.sharding.PartitionSpec("pipeline", "expert", None, None)))

    # Run the wrapper with vmapped it should really just run #pipeline number of times on #pipeline parallelism groups
    # This fails with "TypeError: tuple indices must be integers or slices, not NoneType" if hack_output_for_vmap=False
    # with hack_output_for_vmap=True this returns a 1D a2a instead of 2D
    x_vmap_a2a = jit_vmap_func(x_vmap, logits_vmap)
    print(x_vmap_a2a.shape)