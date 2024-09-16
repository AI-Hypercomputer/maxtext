import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import datetime
import jax
import random
import string
import os
from jax.experimental import shard_map
from jax.experimental.compilation_cache import compilation_cache
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


#!!!! Internally in google3 set trace_dir to CNS path or other profiling solution
def simple_timeit(f, *args, tries=10, task=None):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"t_{task}_" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"gs://mattdavidow-br/{trace_name}" 

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  jax.profiler.start_trace(trace_dir)

  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  jax.profiler.stop_trace()

  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"{task}: average time milliseconds: {average_time_ms:.2f}, trace {trace_dir}")
  return average_time_ms


# Baseline non-overlapped implementation to compare against
# In some ideal world compiler comes up with an overlapped solution even with naive code
def blocking_a2a(input_activations, weights):
    
    outputs = jnp.einsum("BXM,XEM -> BXE", input_activations, weights)
    outputs = jax.lax.with_sharding_constraint(outputs, NamedSharding(mesh, P('expert', None, 'model'))) #A2A B,EXP/X -> B/X,EXP
    return outputs

# Necessary explicit communication (use shard map)
def a2a(input_chunk):
  return jax.lax.all_to_all(input_chunk, 'expert', 0, 1, tiled=True)

# Desired overlapped implementaion
def overlap_a2a(input_activations, weights):
    num_chunks = 4
    chunk_size = EMBED // num_chunks


    ff_output_post_a2a = jnp.zeros((BATCH_PER_EXP, EXP, EMBED), dtype=input_activations.dtype)
    ff_output_post_a2a = jax.lax.with_sharding_constraint(ff_output_post_a2a, NamedSharding(mesh, P('expert', None, 'model')))

    output_list=[None for _ in range(num_chunks)]
    for i in range(num_chunks):
        chunk_start = chunk_size * i

        #input_chunk = jax.lax.dynamic_slice_in_dim(input_activations, chunk_start, chunk_size, 2)s
        weight_chunk = jax.lax.dynamic_slice_in_dim(weights, chunk_start, chunk_size, 1)
        result_chunk_before_a2a = jnp.einsum("BXM,XEM -> BXE", input_activations, weight_chunk)

        # a2a result from B/X,EXP -> B, EXP/X
        result_chunk = shard_map.shard_map(a2a, mesh, in_specs=P(None, 'expert', 'model'), out_specs=P('expert', None, 'model'))(result_chunk_before_a2a)
        #result_chunk = jax.lax.with_sharding_constraint(result_chunk, NamedSharding(mesh, P('expert', None, 'model'))) 
        #print(f"{result_chunk.shape=}", flush=True)
        #output_list[i] = result_chunk_before_a2a
        ff_output_post_a2a = jax.lax.dynamic_update_slice(ff_output_post_a2a, result_chunk, (0,0,chunk_start))

        # Alterantive at API
        #ff_output_post_a2a = ff_output_post_a2a.at[:,:,chunk_start:chunk_start+chunk_size].set(result_chunk)

    # to_ret = jnp.concatenate(output_list, axis=-1)
    # print(f"{to_ret.shape=}", flush=True)
    # to_ret = jax.lax.with_sharding_constraint(to_ret, NamedSharding(mesh, P('expert', None, 'model')))

    ff_output_post_a2a = jax.lax.with_sharding_constraint(ff_output_post_a2a, NamedSharding(mesh, P('expert', None, 'model'))) 
    return ff_output_post_a2a 
    # outputs = jnp.concatenate        
    # return ff_output_post_a2a

def create_inputs():
    input_activations = jax.random.normal(jax.random.PRNGKey(0), (BATCH_PER_EXP, EXP, MLP), dtype=jnp.bfloat16)
    # Inputs start out expert sharded
    input_activations = jax.lax.with_sharding_constraint(input_activations, NamedSharding(mesh, P(None, 'expert','model')))
    
    weights = jax.random.normal(jax.random.PRNGKey(1), (EXP, EMBED, MLP), dtype=jnp.bfloat16)
    weights = jax.lax.with_sharding_constraint(weights, NamedSharding(mesh, P('expert', None, 'model')))
    return input_activations, weights

BATCH_PER_EXP = 16384
EMBED = 4096
MLP = 8192
EXP = 4

global mesh
data_parallelism, model_parallelism, expert_parallelism = 1, 1, 4
ici_parallelism = [data_parallelism, model_parallelism, expert_parallelism]
devices_array = mesh_utils.create_device_mesh(ici_parallelism)
mesh = Mesh(devices_array, ["data", "model", "expert"])

input_activations, weights = jax.jit(create_inputs)()

# correctness test
# overlapped_results = jax.jit(overlap_a2a)(input_activations, weights)
# blocking_results = jax.jit(blocking_a2a)(input_activations, weights)
# # assert overlapped_results and blocking_results are close
# assert jnp.allclose(overlapped_results, blocking_results, rtol=1e-3, atol=1e-2)

# Profile overlap solution
jit_overlap_a2a = jax.jit(overlap_a2a)
simple_timeit(jit_overlap_a2a, input_activations, weights, task="hide_a2a")

# Profile blocking solution
# jit_blocking_a2a = jax.jit(blocking_a2a)
# simple_timeit(jit_blocking_a2a, input_activations, weights, task="blocking_a2a")
