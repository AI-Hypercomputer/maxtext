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


def simple_timeit(f, *args, tries=10, task=None, enable_profile=False):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"{task}"  # + '_' ]+ ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  temp_dir = tempfile.gettempdir()
  trace_dir = os.path.join(temp_dir, trace_name)
  #print(trace_dir)

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  if enable_profile:
    jax.profiler.start_trace(trace_dir)
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

num_devices = len(jax.devices()) # we expect either 4 or 8 total devices
expert_parallelism = 2
assert expert_parallelism==2, "This script only supports EP=2"
pipeline_parallelism = num_devices // expert_parallelism # We expect this is either 2 or 4


 # Define a mesh with PP + EP
device_mesh_array = mesh_utils.create_device_mesh((expert_parallelism, pipeline_parallelism))
mesh = Mesh(device_mesh_array, ("expert", "pipeline"))
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


# create an array x which is [batch, model] and has elements like
# [[0,0,0],
#  [1,1,1],
#  ...


def run_and_time_vmap_ep(n_batch, model):
    #n_batch = 512
    #model = 2048
    batch = n_batch * expert_parallelism**2
    

    x = jnp.arange(0.0, batch)
    x = jnp.expand_dims(x, axis=1)
    x = jnp.tile(x, (1, model))
    x = jax.device_put(x, x_sharding) 

    output_shape = x.copy()

    input_offsets = jnp.array([[0, n_batch],[0,n_batch]], dtype=jnp.int32)
    input_offsets = jax.device_put(input_offsets, x_sharding)

    send_sizes = jnp.array([[n_batch, n_batch],[n_batch,n_batch]], dtype=jnp.int32)
    send_sizes = jax.device_put(send_sizes, x_sharding)

    output_offsets = jnp.array([[0, 0],[n_batch,n_batch]], dtype=jnp.int32)
    output_offsets = jax.device_put(output_offsets, x_sharding)

    recv_sizes = jnp.array([[n_batch, n_batch],[n_batch, n_batch]], dtype=jnp.int32)
    recv_sizes = jax.device_put(recv_sizes, x_sharding)

    vmap_func = jax.vmap(
        ra2a_wrapper,
        spmd_axis_name="pipeline",
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


    simple_timeit(jit_vmap_func, x_vmap, output_shape_vmap, input_offsets_vmap, send_sizes_vmap, output_offsets_vmap, recv_sizes_vmap, tries=10, task="vmap_ra2a", enable_profile=False)

    vmap_output = jit_vmap_func(x_vmap, output_shape_vmap, input_offsets_vmap, send_sizes_vmap, output_offsets_vmap, recv_sizes_vmap)
    #print(f"output of a2a WITH vmap:\n {vmap_output}")
    # print(f"vmap_output.shape = {vmap_output.shape}")
    # print("Successfully ran vmap!!")
    # This will fail! The output shape is [PP, 2, 4, 3] but we expect [PP, 8, 3] - the same shape as both x_vmap and output_shape_vmap
    #assert vmap_output.shape == (pipeline_parallelism, batch, model)
    # print("Now running expected assert...")
    # for i in range(pipeline_parallelism):
    #     assert  jnp.array_equal(vmap_output[i,:,:], expected_array)
    # print("Vmapped output has expected values!")


run_and_time_vmap_ep(512, 2048)

run_and_time_vmap_ep(512, 4096)

run_and_time_vmap_ep(1024, 2048)