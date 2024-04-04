import math
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp



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
  tile_m = 512
  tile_n = 1024
  tile_k = 1024
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
  x = jnp.ones((4096, 4096), dtype=jnp.int8)
  y = jnp.ones((4096, 4096), dtype=jnp.int8)
  jax.block_until_ready(matmul(x, y))
  jax.block_until_ready(jnp.matmul(x, y))
