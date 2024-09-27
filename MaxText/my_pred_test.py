import jax
from jax import numpy as jnp
from functools import partial
from jax import lax

from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

global mesh
mesh = Mesh(jax.devices(), ('x',))


a_arr = jax.device_put(
    jnp.arange(4 * 4).reshape((4, 4)),
    jax.sharding.NamedSharding(mesh, P('x', None)))

@jax.jit
@partial(
    shard_map, mesh=mesh, in_specs=(P('x', None),), out_specs=P('x', None)
)
def fwd(a_arr):
    axis_size = lax.psum(1, 'x')
    perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
    return lax.ppermute(a_arr, 'x', perm=perm)

c_arr = fwd(a_arr)

# assert two jax arrays are close
are_close = jnp.isclose(c_arr[1, :], a_arr[0, :], atol=1e-05, rtol=1e-05)
all_close = jnp.all(are_close)

breakpoint()
