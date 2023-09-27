import jax
from jax import numpy as jnp
import numpy as np
from functools import partial

from jax.sharding import Mesh
from jax.sharding import PartitionSpec



def _sum(x):
  return jax.tree_map(partial(jnp.sum, axis=0), x)

def broadcast_one_slice_to_all(in_tree, is_source, mesh, per_slice_sharding):

  @partial(jax.jit, static_argnums=0)
  def fake_zero_data(sharding):
    x = jnp.zeros_like(x)
    return jax.lax.with_sharding_constraint(x, sharding)

  def pre_jit(x):
    if is_source:
      inp = x
    else:
      inp = fake_zero_data(per_slice_sharding)
    inp = np.expand_dims(inp, axis=0)
    in_spec = P('data', *x.sharding.spec)
    return jax.experimental.host_local_array_to_global_array(inp, full_mesh, in_spec)

  out_sharding = jax.tree_map(lambda x: x.sharding, in_tree)

  in_tree = jax.tree_map(pre_jit, in_tree)
  out_tree = jax.jit(_sum, out_shardings=out_sharding)(in_tree)
  return out_tree

m = jax.devices()

m = np.reshape(m, (2, 2))



mesh = Mesh(m, ["axis1", "axis2"])

print(mesh)

def make_data():
  return jax.numpy.ones(100)

from jax.experimental.pjit import pjit
pjit_make_data = pjit(make_data, in_shardings=None, out_shardings=PartitionSpec(("axis1", "axis2")))

with mesh:
  data = pjit_make_data()
  jax.debug.visualize_array_sharding(data)

