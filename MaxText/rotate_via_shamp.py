import jax
import jax.ad_checkpoint
import numpy as np
from jax import numpy as jnp
from flax.core import meta
from flax import linen as nn
import common_types
import functools
from typing import Any
from jax.experimental import shard_map
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

NUM_STAGES=2
BATCH_SHARD=2
SEQUENCE_SHARD=1
EMBED_SHARD=1


BATCH=2
SEQUENCE=6
EMBED=3

AXIS_NAMES = ('stage','batch', 'sequence', 'embed')
ici_parallelism = [NUM_STAGES, BATCH_SHARD, SEQUENCE_SHARD ,EMBED_SHARD]
devices_array = mesh_utils.create_device_mesh(ici_parallelism)
global mesh
mesh = Mesh(devices_array, AXIS_NAMES)

def rotate_right_shmap(arr):
    partition_spec = P(*AXIS_NAMES)
    print(f"{partition_spec=}", flush=True)
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=partition_spec,
        out_specs=partition_spec,
        check_rep=False,
    )
    def rotate_shmap(arr):
        arr = jax.lax.ppermute(arr, 'stage', [(i, (i+1) % NUM_STAGES) for i in range(NUM_STAGES)])
        return arr
    return rotate_shmap(arr)

def rotate_right(arr):
    # Use lax.slice to avoid generating a gather.
    last = jax.lax.slice_in_dim(arr, NUM_STAGES - 1, NUM_STAGES, axis=0)
    except_last = jax.lax.slice_in_dim(arr, 0, NUM_STAGES - 1, axis=0)
    return jnp.concatenate([last, except_last], axis=0)

def create_random_arr():
    shape = (NUM_STAGES, BATCH, SEQUENCE, EMBED)
    total_elements = np.prod(shape)  # Calculate the total number of elements
    sequential_values = jnp.arange(1, total_elements + 1)  # Create a 1D array with sequential values
    return jnp.reshape(sequential_values, shape)

arr1 = create_random_arr()
arr2 = create_random_arr()

print(f"{jnp.linalg.norm(arr1)=}",flush=True)

rot_shmap = rotate_right_shmap(arr1)
rot_regular = rotate_right(arr2)
diff = rot_shmap - rot_regular

print(f"{jnp.linalg.norm(rot_shmap)=}",flush=True)
print(f"{jnp.linalg.norm(rot_regular)=}",flush=True)
print(f"{jnp.linalg.norm(diff)=}",flush=True)