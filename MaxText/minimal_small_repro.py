import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

mesh_axes = ["axis_1", "axis_2"]
ici_parallelism = [1,4]
dcn_parallelism = [2,1]
device_mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
mesh = Mesh(device_mesh, mesh_axes)
data_sharding = (("axis_1", "axis_2"),)
data_pspec = P(*data_sharding)
data_pspec_shardings = jax.sharding.NamedSharding(mesh, data_pspec)
data = jax.numpy.ones(1024)
data = jax.lax.with_sharding_constraint(data, data_pspec_shardings)
jit_sum = jax.jit(jnp.sum)
jit_sum(data) # runtime crash