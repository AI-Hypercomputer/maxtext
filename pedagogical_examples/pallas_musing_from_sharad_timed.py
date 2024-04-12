import math
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

import random
import string
import datetime


def simple_timeit(f, *args, tries = 10, task = None):
    '''Simple utility to time a function for multiple runs'''
    assert task is not None

    trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    trace_dir = f"gs://runner-maxtext-logs/rwittensuper/{trace_name}"

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


def matmul_kernel(x_ref, y_ref, o_ref, o_scratch_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

  x = x_ref[...]
  y = y_ref[...]
  o_scratch_ref[...] += jnp.matmul(
      x, y, preferred_element_type=o_scratch_ref.dtype
  )

  @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
  def _():
    o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)


@jax.jit
def matmul(x: jax.Array, y: jax.Array) -> jax.Array:
  m, _ = x.shape
  k, n = y.shape
  tile_m = 768
  tile_n = 768
  tile_k = 768
  grid = (math.ceil(m / tile_m), math.ceil(n / tile_n), math.ceil(k / tile_k))

  block_spec_a = pl.BlockSpec(lambda i, j, h: (i, h), (tile_m, tile_k))
  block_spec_b = pl.BlockSpec(lambda i, j, h: (h, j), (tile_k, tile_n))
  block_spec_out = pl.BlockSpec(lambda i, j, h: (i, j), (tile_m, tile_n))
  out_shape = (m, n)
  dimension_semantics = ("parallel", "parallel", "arbitrary")
  if jnp.issubdtype(x.dtype, jnp.integer):
    accum_dtype = jnp.int32
  else:
    accum_dtype = jnp.float32
  out_dtype = x.dtype
  return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct(out_shape, out_dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          scratch_shapes=[pltpu.VMEM((tile_m, tile_n), accum_dtype)],
          in_specs=[block_spec_a, block_spec_b],
          out_specs=block_spec_out,
      ),
      compiler_params=dict(
          mosaic=dict(dimension_semantics=dimension_semantics)
      ),
  )(x, y)


if __name__ == "__main__":
  x = jnp.ones((7680, 7680), dtype=jnp.bfloat16)
  y = jnp.ones((7680, 7680), dtype=jnp.bfloat16)

  assert jax.numpy.allclose(matmul(x,y), jnp.matmul(x,y), atol=1e-2, rtol=1e-2)

  simple_timeit(matmul, x, y, task = "pallas")
  simple_timeit(jnp.matmul, x, y, task = "xla")