import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
import timing_util

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

num_stages = 3
num_ici = 64
per_device_batch_size = 2048 * 5
batch_size = num_ici * per_device_batch_size # really microbatch_size, num_micro = num_stages
embed_size = 3072
embed_out_size = embed_size
num_iter= 4

mesh_axes = ["stage", "fsdp"]
ici_parallelism = [1,64]
dcn_parallelism = [num_stages,1]
device_mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
mesh = Mesh(device_mesh, mesh_axes)



# Create data
def create_data():
    data_shape = (num_stages, batch_size, embed_size)
    data_sharding = ("stage", "fsdp", None)
    data_pspec = P(*data_sharding)
    data_pspec_shardings = jax.sharding.NamedSharding(mesh, data_pspec)
    data = jax.numpy.ones(data_shape)
    data = jax.lax.with_sharding_constraint(data, data_pspec_shardings)
    return data

# Create weight matrix
def create_weights():
    weight_sharding = ("stage", "fsdp", None)
    weight_pspec = jax.sharding.NamedSharding(mesh, P(*weight_sharding))
    weights = jax.numpy.ones((num_stages, embed_size, embed_out_size))
    weights = jax.lax.with_sharding_constraint(weights, weight_pspec)
    return weights

def layer_matmul(acitivations, weights):
    return acitivations @ weights

vmapped_layer_matmul = jax.vmap(layer_matmul)

# Shift becomes a rotated-right version of the previous output
def _rotate_right(output_in):
    # Use lax.slice to avoid generating a gather.
    last = jax.lax.slice_in_dim(output_in, num_stages - 1, num_stages, axis=0)
    except_last = jax.lax.slice_in_dim(output_in, 0, num_stages - 1, axis=0)
    return jnp.concatenate([last, except_last], axis=0)

def pipeline_imitator(data, weights):
    for _ in range(num_iter):
        data = vmapped_layer_matmul(data, weights)
        data = _rotate_right(data)
    return data

jit_pipeline_imitator = jax.jit(pipeline_imitator)

jit_create_data = jax.jit(create_data)
data = jit_create_data()
jit_create_weights = jax.jit(create_weights)
weights = jit_create_weights()

print("About to perform the imitator...",flush=True)
outputs = jit_pipeline_imitator(data, weights)
outputs.block_until_ready()
print(f"Outputs done!", flush=True)

timing_util.simple_timeit(jit_pipeline_imitator, data, weights, tries = 3, task = 'vmap_dcn_fsdp')

