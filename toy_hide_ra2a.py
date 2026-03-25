import jax
import jax.numpy as jnp
from jax import lax, jit
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import functools
from enum import Enum, auto
from jax.sharding import PartitionSpec as P
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu
#import tokamax
import datetime

def simple_timeit(f, *args, tries=3, task=None, enable_profile=True):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"{task}"  # + '_' ]+ ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"gs://mattdavidow-maxtext-br/mintext-mpmd/a8/{trace_name}"

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  import time
  time.sleep(5) #profile is messed up?
  if enable_profile:
    jax.profiler.start_trace(trace_dir)
    print(f"{trace_dir=}")
  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  if enable_profile:
    jax.profiler.stop_trace()
  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"Average time ms for mm for {task} is {round(average_time_ms, 3)}")
  return average_time_ms / 1000





GLOBAL_BATCH=1048576
MODEL=2048
FF=16384
NUM_EXP=8
EP=8
EXP_PER_SHARD = NUM_EXP // EP
assert EP * EXP_PER_SHARD == NUM_EXP, "Experts must be divisible by EP"
BATCH_PER_EP_SHARD = GLOBAL_BATCH // EP
assert EP * BATCH_PER_EP_SHARD == GLOBAL_BATCH, "Global Batch must be a multiple of EP"
BATCH_PER_EP_SHARD_PER_EXP = BATCH_PER_EP_SHARD // NUM_EXP # aka block assignment per shard or something....
assert NUM_EXP * BATCH_PER_EP_SHARD_PER_EXP == BATCH_PER_EP_SHARD, "Global Batch must be a multiple of (EP * EXP)"

NUM_A2A_CHUNKS=4
MODEL_CHUNK_SIZE = MODEL // NUM_A2A_CHUNKS


 # 1D EP mesh
device_mesh_array = mesh_utils.create_device_mesh((EP,))
mesh = Mesh(device_mesh_array, ("expert"))
x_partition_spec = jax.sharding.PartitionSpec("expert", None)
explicit_ep_partition_spec = jax.sharding.PartitionSpec("expert", None, None)
x_sharding = NamedSharding(mesh, x_partition_spec)

# This usage (API) is surprisingly hard to discover
config = pallas_mosaic_tpu.Config(tile_m=1024, tile_k=512, tile_n=2048)
tpu_ragged_dot = pallas_mosaic_tpu.PallasMosaicTpuRaggedDot(config=config)

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
        explicit_ep_partition_spec

        ),
    out_specs=(explicit_ep_partition_spec),
    check_vma=False,
)
def ra2a_gmm(x_input, input_offsets, send_sizes, output_offsets, recv_sizes, group_sizes, weights):
    # remove singleton leading axis of x
    x_input = x_input.reshape(x_input.shape[1:])
    output_ra2a_chunk_shape = jnp.empty((x_input.shape[0], MODEL_CHUNK_SIZE), dtype=jnp.bfloat16)
                                                 
    group_sizes = group_sizes.reshape(group_sizes.shape[1:])

    input_offsets = input_offsets.reshape(input_offsets.shape[1:])
    send_sizes = send_sizes.reshape(send_sizes.shape[1:])
    output_offsets = output_offsets.reshape(output_offsets.shape[1:])
    recv_sizes = recv_sizes.reshape(recv_sizes.shape[1:])

    def ra2a_chunk(ra2a_input):
        return jax.lax.ragged_all_to_all(
            ra2a_input,
            output_ra2a_chunk_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )

    def compute_and_ra2a(activation_chunk, weight_chunk, ra2a_input, gmm_accum):
        #gmm_output_partial = tokamax.ragged_dot(activation_chunk, weight_chunk, group_sizes, implementation="mosaic", tile_sizes=TILE_SIZE)
        gmm_output_partial = tpu_ragged_dot(activation_chunk, weight_chunk, group_sizes=group_sizes)
        gmm_accum = gmm_accum + gmm_output_partial
        ra2a_output = ra2a_chunk(ra2a_input)
        return gmm_accum, ra2a_output


    # first ra2a is exposed
    first_chunk = jax.lax.dynamic_slice_in_dim(x_input, 0, MODEL_CHUNK_SIZE, 1)
    ra2a_output = ra2a_chunk(first_chunk)
    gmm_output = jnp.zeros((x_input.shape[0], FF), dtype=jnp.bfloat16) # initialize accumulation of chunks to zeros
    next_ra2a_input = jax.lax.dynamic_slice_in_dim(x_input, MODEL_CHUNK_SIZE, MODEL_CHUNK_SIZE, 1)

    for i in range(NUM_A2A_CHUNKS - 1):
        weight_chunk = jax.lax.dynamic_slice_in_dim(weights, i * MODEL_CHUNK_SIZE, MODEL_CHUNK_SIZE, 1)
        gmm_output, ra2a_output = compute_and_ra2a(ra2a_output, weight_chunk, next_ra2a_input, gmm_output)
        next_ra2a_input = jax.lax.dynamic_slice_in_dim(x_input, MODEL_CHUNK_SIZE * (i + 2), MODEL_CHUNK_SIZE, 1)
    
    # last compute has no ra2a
    last_weight_chunk = jax.lax.dynamic_slice_in_dim(weights, (NUM_A2A_CHUNKS - 1) * MODEL_CHUNK_SIZE, MODEL_CHUNK_SIZE, 1)
    #last_gmm_partial = tokamax.ragged_dot(ra2a_output, last_weight_chunk, group_sizes, implementation="mosaic", tile_sizes=TILE_SIZE)
    last_gmm_partial = tpu_ragged_dot(ra2a_output, last_weight_chunk, group_sizes=group_sizes)
    gmm_output = gmm_output + last_gmm_partial
    
    output = jnp.expand_dims(gmm_output, axis=0)

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
x = jnp.array(x, dtype=jnp.bfloat16)

input_offsets = [[BATCH_PER_EP_SHARD_PER_EXP * exp_idx for exp_idx in range(NUM_EXP)] for _ in range(EP)]
input_offsets = jnp.array(input_offsets, dtype=jnp.int32)

output_offsets = jnp.transpose(input_offsets)

send_sizes = jnp.array([[BATCH_PER_EP_SHARD_PER_EXP for _ in range(NUM_EXP)] for _ in range(EP)], dtype=jnp.int32)
recv_sizes = jnp.array([[BATCH_PER_EP_SHARD_PER_EXP for _ in range(NUM_EXP)] for _ in range(EP)], dtype=jnp.int32)

tokens_per_local_expert = BATCH_PER_EP_SHARD / EXP_PER_SHARD
group_sizes = jnp.array([tokens_per_local_expert for _ in range(EXP_PER_SHARD)], dtype=jnp.int32)
group_sizes = jnp.tile(jnp.expand_dims(group_sizes, axis=0), (EP, 1))

weights = jnp.zeros([NUM_EXP, MODEL, FF], dtype=jnp.bfloat16)

jit_wrapper = jax.jit(ra2a_gmm)
output = jit_wrapper(x, input_offsets, send_sizes, output_offsets, recv_sizes, group_sizes, weights)
simple_timeit(jit_wrapper, x, input_offsets, send_sizes, output_offsets, recv_sizes, group_sizes, weights, task="overlap_ra2a")



