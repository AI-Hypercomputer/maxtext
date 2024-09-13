import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import datetime
import jax
import random
import string
import os
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
def blocking_a2a(input_activations, weights):
    input_activations = jax.lax.with_sharding_constraint(input_activations, NamedSharding(mesh, P(None, 'expert', None)))
    return jnp.einsum("BXE,XEM -> BXM", input_activations, weights)

# Desired overlapped implementaion
def overlap_a2a(input_activations, weights):
    num_chunks = 4
    chunk_size = EMBED // num_chunks

    partial_sum = jnp.zeros((BATCH_PER_EXP, EXP, MLP))
    partial_sum = jax.lax.with_sharding_constraint(partial_sum, NamedSharding(mesh, P(None, 'expert', None)))
    for i in range(num_chunks):
        chunk_start = chunk_size * i

        input_chunk = jax.lax.dynamic_slice_in_dim(input_activations, chunk_start, chunk_size, 2)
        input_chunk = jax.lax.with_sharding_constraint(input_chunk, NamedSharding(mesh, P(None, 'expert', None)))
        weight_chunk = jax.lax.dynamic_slice_in_dim(weights, chunk_start, chunk_size, 1)

        partial_sum = partial_sum + jnp.einsum("BXE,XEM -> BXM", input_chunk, weight_chunk)
    return partial_sum

def create_inputs():
    input_activations = jnp.ones((BATCH_PER_EXP, EXP, EMBED),dtype=jnp.bfloat16)
    input_activations = jax.lax.with_sharding_constraint(input_activations, NamedSharding(mesh, P('expert', None,None)))
    
    weights = jnp.ones((EXP, EMBED, MLP),dtype=jnp.bfloat16)
    weights = jax.lax.with_sharding_constraint(weights, NamedSharding(mesh, P(None, 'expert', None)))
    return input_activations, weights

BATCH_PER_EXP = 2048
EMBED = 4096
MLP = 32768
EXP = 4

global mesh
mesh = Mesh(jax.devices(), ('expert',))

input_activations, weights = jax.jit(create_inputs)()

jit_overlap_a2a = jax.jit(overlap_a2a)
simple_timeit(jit_overlap_a2a, input_activations, weights, task="hide_a2a")
