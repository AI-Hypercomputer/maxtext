import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils

def my_concatenate(x):
    first_row = jax.lax.slice_in_dim(x, 0, 1, axis=0)
    other_rows = jax.lax.slice_in_dim(x, 1, 4, axis=0)
    return jnp.concatenate([first_row, other_rows], axis=0)

my_input = jnp.arange(4)
my_concatenate(my_input) # This works fine

# Create a mesh with 1D axis length 4 (v4-8) Define a sharding over the axis
devices = mesh_utils.create_device_mesh((4,))
mesh = jax.sharding.Mesh(devices, axis_names=('data'))
data_sharding =  jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data'))
my_input_sharded = jax.lax.with_sharding_constraint(my_input, data_sharding)

# If we jit my_concatenate, things work
jit_my_concatenate = jax.jit(my_concatenate, in_shardings=data_sharding, out_shardings=data_sharding)
jit_my_concatenate(my_input_sharded) # works

# If we don't jit, we get a sharding error
my_concatenate(my_input_sharded) # sharding error


