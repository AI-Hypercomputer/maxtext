from functools import partial

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import jax.numpy as jnp
import numpy as np
import datetime
import jax
import random
import string

def simple_timeit(f, *args, tries = 2, task = None):
    '''Simple utility to time a function for multiple runs'''
    assert task is not None

    trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    trace_dir = f"gs://runner-maxtext-logs/rwitten/{trace_name}"

    outcomes_ms = []
    jax.block_until_ready(f(*args)) #warm it up!
    jax.profiler.start_trace(trace_dir)

    for _ in range(tries):
        s = datetime.datetime.now()
        jax.block_until_ready(f(*args))
        e = datetime.datetime.now()
        outcomes_ms.append(1000*(e-s).total_seconds())
    jax.profiler.stop_trace()

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    print(f"{task}: average time milliseconds: {average_time_ms:.2f}, trace {trace_dir}")
    return average_time_ms

raw_len = 32768
chunk_size = 1024
grid_pts = raw_len//chunk_size

v1 = jax.random.normal(jax.random.key(0), (raw_len, raw_len), dtype=jnp.bfloat16)
v2 = jax.random.normal(jax.random.key(1), (raw_len, raw_len), dtype=jnp.bfloat16)

@jax.jit
def multiply_tensors_normal(x: jax.Array, y: jax.Array):
  return x @ y

def multiply_tensors_kernel_original(x_ref, y_ref, o_ref):
  with pltpu.trace("body"):
    with pltpu.trace("clear"):

      @pl.when(pl.program_id(axis=2) == 0)
      def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    with pltpu.trace("dot"):
      dot = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

    with pltpu.trace("accum"):
      o_ref[...] += dot

def multiply_tensors_kernel_original_fast(x_ref, y_ref, o_ref):
  @pl.when(pl.program_id(axis=2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)

  dot = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

  o_ref[...] += dot

def multiply_tensors_kernel(x_tile_ref, y_tile_ref, o_tile_ref, acc_ref):
  @pl.when(pl.program_id(2) == 0)
  def init():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] = acc_ref[...] + jnp.dot(
      x_tile_ref[...],
      y_tile_ref[...],
      preferred_element_type=acc_ref.dtype,
  )
  # It is possible to make this conditional but in general this bundle packs
  # quite well for a simple matmul kernel
  o_tile_ref[...] = acc_ref[...].astype(o_tile_ref.dtype)



@jax.jit
def multiply_tensors_pallas(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(multiply_tensors_kernel,
                        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
                        grid_spec=pltpu.PrefetchScalarGridSpec(
                          num_scalar_prefetch=0,
                          in_specs=[
                            pl.BlockSpec(lambda i,j,k: (i,k), (chunk_size, chunk_size,)),
                            pl.BlockSpec(lambda i,j,k: (k,j), (chunk_size, chunk_size,))
                          ],
                          out_specs=pl.BlockSpec(lambda i,j,k: (i,j), (chunk_size, chunk_size,)),
                          scratch_shapes=[pltpu.VMEM((chunk_size, chunk_size), jnp.float32)],
                          grid = (grid_pts, grid_pts, grid_pts),
                        ),
                        
                        compiler_params=dict(mosaic=dict(dimension_semantics=("parallel","parallel","arbitrary",))),
                        )(x, y)

@jax.jit
def multiply_tensors_pallas_original(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(multiply_tensors_kernel_original,
                        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
                        in_specs=[
                          pl.BlockSpec(lambda i,j,k: (i,k), (chunk_size, chunk_size,)),
                          pl.BlockSpec(lambda i,j,k: (k,j), (chunk_size, chunk_size,))
                        ],
                        out_specs=pl.BlockSpec(lambda i,j,k: (i,j), (chunk_size, chunk_size,)),
                        grid = (grid_pts, grid_pts, grid_pts),
                        compiler_params=dict(mosaic=dict(dimension_semantics=("parallel","parallel","arbitrary",))),
                        )(x, y)

pallas = multiply_tensors_pallas_original(v1, v2)
normal = multiply_tensors_normal(v1, v2)

assert jax.numpy.allclose(pallas, normal, atol=1e-2, rtol=1e-2)

simple_timeit(multiply_tensors_pallas_original, v1, v2, task="pallas")
simple_timeit(multiply_tensors_normal, v1, v2, task="normal")


