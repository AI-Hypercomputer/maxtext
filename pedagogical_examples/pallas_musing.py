from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

raw_len = 512
chunk_size = 128

v1 = jax.random.normal(jax.random.key(0), (raw_len, raw_len))
v2 = jax.random.normal(jax.random.key(1), (raw_len, raw_len))

def add_vectors_kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]

def add_vector_normal(x: jax.Array, y: jax.Array):
  return x + y

@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(add_vectors_kernel,
                        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                        in_specs=[
                          pl.BlockSpec(lambda i,j: (i,j), (chunk_size, chunk_size,)),
                          pl.BlockSpec(lambda i,j: (i,j), (chunk_size, chunk_size,))
                        ],
                        out_specs=pl.BlockSpec(lambda i,j: (i,j), (chunk_size, chunk_size,)),
                        grid = (raw_len//chunk_size, raw_len//chunk_size),
                        )(x, y)


pallas = add_vectors(v1, v2)
normal = add_vector_normal(v1, v2)

assert jax.numpy.allclose(pallas, normal)
