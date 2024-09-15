import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import jax
import os
from jax.debug import visualize_array_sharding
from jax.experimental import shard_map
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

def input_a2a(input_chunk):
    return jax.lax.all_to_all(input_chunk, 'expert', 1, 0, tiled=True)

BATCH_PER_EXP = 2048
EXP = 4

global mesh
mesh = Mesh(jax.devices(), ('expert',))

input_activations = jnp.ones((BATCH_PER_EXP, EXP),dtype=jnp.bfloat16)
input_activations = jax.lax.with_sharding_constraint(input_activations, NamedSharding(mesh, P('expert', None)))

print(f"{input_activations.shape=}")
visualize_array_sharding(input_activations)

inputs_before_a2a_spec = P("expert", None) 
inputs_after_a2a_spec = P(None, "expert")
# Perform a2a on input_chunk B/X, Exp -> B, Exp/X
input_after_a2a = shard_map.shard_map(input_a2a, mesh, in_specs=inputs_before_a2a_spec, out_specs=inputs_after_a2a_spec)(input_activations)

print(f"{input_after_a2a.shape=}")
visualize_array_sharding(input_after_a2a)