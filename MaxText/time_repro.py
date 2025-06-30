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
  print(f"Average time ms for {task} is {round(average_time_ms, 3)}")
  return average_time_ms / 1000

num_devices = len(jax.devices()) # we expect either 4 or 8 total devices
expert_parallelism = 4
#assert expert_parallelism==2, "This script only supports EP=2"
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


def run_and_time_vmap_ep(batch_per_ep, model):
    #model = 2048
    #batch = batch_per_ep * expert_parallelism**2
    batch = batch_per_ep * expert_parallelism
    

    def create_x():
      x = jnp.arange(0.0, batch)
      x = jnp.expand_dims(x, axis=1)
      x = jnp.tile(x, (1, model))
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
    #output_offsets = jnp.array([[0, 0],[n_batch,n_batch]], dtype=jnp.int32)
    output_offsets = jax.device_put(output_offsets, x_sharding)

    # recv_sizes_i_j is the ith EP rec size from EP j
    recv_sizes = send_sizes.copy()
    #recv_sizes = jnp.array([[n_batch, n_batch],[n_batch, n_batch]], dtype=jnp.int32)
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
    x_vmap = jax.jit(expand_array_for_vmap)(x)
    x_vmap.block_until_ready()

    output_shape_vmap = expand_array_for_vmap(output_shape)
    input_offsets_vmap = expand_array_for_vmap(input_offsets)
    send_sizes_vmap = expand_array_for_vmap(send_sizes)
    output_offsets_vmap = expand_array_for_vmap(output_offsets)
    recv_sizes_vmap = expand_array_for_vmap(recv_sizes)


    output_time = simple_timeit(jit_vmap_func, x_vmap, output_shape_vmap, input_offsets_vmap, send_sizes_vmap, output_offsets_vmap, recv_sizes_vmap, tries=10, task="vmap_ra2a", enable_profile=False)
    return output_time

def get_roofline_time(batch_per_ep, model, ici_speed_bytes):
  # Assume perfectly balanced a2a, cost is 1/4 that of an all-gather
  all_gather_size_bytes = 2 * batch_per_ep * model * expert_parallelism
  all_gather_time = all_gather_size_bytes / ici_speed_bytes
  ep_roofline_time = all_gather_time / 4
  return ep_roofline_time



batch_per_ep = 1024

model_vec = [512 * 2**n for n in range(1, 5)]

output_time = [run_and_time_vmap_ep(batch_per_ep, model) for model in model_vec]

ici_speed_bytes = 180e9
roofline_time = [get_roofline_time(batch_per_ep, model, ici_speed_bytes) for model in model_vec]
frac = [roof/output for output, roof in zip(output_time, roofline_time)]

print(output_time)
print(roofline_time)
print(frac)

# run_and_time_vmap_ep(512, 2048)

# run_and_time_vmap_ep(512, 4096)

# run_and_time_vmap_ep(1024, 2048)