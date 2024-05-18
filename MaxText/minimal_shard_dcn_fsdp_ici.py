import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
import timing_util

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

mesh_axes = ["stage", "fsdp"]
ici_parallelism = [1,4]
dcn_parallelism = [2,1]
device_mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
mesh = Mesh(device_mesh, mesh_axes)

num_stages = 2
batch_size = 16
embed_size = 3072
embed_out_size = 5120

# Create data
data_shape = (num_stages, batch_size, embed_size)
data_sharding = ("stage", "fsdp", None)
data_pspec = P(*data_sharding)
data_pspec_shardings = jax.sharding.NamedSharding(mesh, data_pspec)
data = jax.numpy.ones(data_shape)
data = jax.lax.with_sharding_constraint(data, data_pspec_shardings)

# Create weight matrix
weight_sharding = ("stage", "fsdp", None)
weight_pspec = jax.sharding.NamedSharding(mesh, P(*weight_sharding))
weights = jax.numpy.ones((num_stages, embed_size, embed_out_size))
weights = jax.lax.with_sharding_constraint(weights, weight_pspec)

def layer_matmul(acitivations, weights):
    return acitivations @ weights

vmapped_layer_matmul = jax.vmap(layer_matmul)

jitted_vmapped_layer_matmul = jax.jit(vmapped_layer_matmul)

print("About to perform the matmul...",flush=True)
outputs = jitted_vmapped_layer_matmul(data, weights)
outputs.block_until_ready()
print(f"Outputs done!", flush=True)

timing_util.simple_timeit(jitted_vmapped_layer_matmul, data, weights, tries = 3, task = 'vmap_dcn_fsdp')


print(f"Printing to guarantee execution: {jnp.sum(outputs)}", flush=True)
