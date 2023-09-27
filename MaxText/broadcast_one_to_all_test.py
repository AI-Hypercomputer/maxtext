import jax
from jax import numpy as jnp
import numpy as np
from functools import partial

from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.experimental.pjit import pjit

import sys

def make_data():
  return jax.numpy.ones(100)


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
    in_spec = PartitionSpec('data', *x.sharding.spec)
    return jax.experimental.host_local_array_to_global_array(inp, full_mesh, in_spec)

  out_sharding = jax.tree_map(lambda x: x.sharding, in_tree)

  in_tree = jax.tree_map(pre_jit, in_tree)
  out_tree = jax.jit(_sum, out_shardings=out_sharding)(in_tree)
  return out_tree





##Running normally on single host
sys.exit(0)

m = jax.devices()

m = np.reshape(m, (4, 1))

submeshes = []
isolated_data = []

for i in range(4):
  sub_devices = m[i:i+1, :]
  submeshes.append(Mesh(sub_devices, ["axis1", "axis2"]))
  pjit_make_data = pjit(make_data, in_shardings=None, out_shardings=PartitionSpec(("axis1", "axis2")))
  with submeshes[-1]:
    isolated_data.append(pjit_make_data())
    jax.debug.visualize_array_sharding(isolated_data[-1])



##Running normally on one slice
sys.exit(0)

full_mesh = Mesh(m, ["axis1", "axis2"])


from jax.experimental.pjit import pjit
pjit_make_data = pjit(make_data, in_shardings=None, out_shardings=PartitionSpec(("axis1", "axis2")))

with mesh:
  data = pjit_make_data()
  jax.debug.visualize_array_sharding(data)

