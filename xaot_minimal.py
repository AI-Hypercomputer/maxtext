from jax.experimental.topologies import get_topology_desc
import jax
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import numpy as np
from jax.experimental.serialize_executable import serialize, deserialize_and_load
import pickle
import jax.numpy as jnp
from flax.serialization import to_bytes
import msgpack
from jax import tree_util



topo='v4-16'
if topo=='v4-8':
    topology_devices = get_topology_desc(
        platform='tpu',
        topology_name=f'v4:2x2x1',
        chip_config_name='megacore',
        chips_per_host_bounds=(2, 2, 1),
        num_slices=1,
    ).devices
elif topo=='v4-16':
    topology_devices = get_topology_desc(
    platform='tpu',
    topology_name=f'v4:2x2x2',
    chip_config_name='megacore',
    chips_per_host_bounds=(2, 2, 2),
    num_slices=1,
).devices


print(f"{topology_devices=}")
jax_devices = jax.devices()
print(f"{jax_devices=}")

use_devices = topology_devices

def fun(x):
    return x * x

with jax.sharding.Mesh(np.array(use_devices), ('data',)):
    jitted = pjit.pjit(
        fun, in_shardings=P('data'), out_shardings=P(None, 'data')
    )
    lowered = jitted.lower(
        jax.core.ShapedArray(shape=(128, 128), dtype=np.float32)
    )
orig_compiled = lowered.compile()


serialized, in_tree, out_tree = serialize(orig_compiled)
print(f"{in_tree=}")
print(f"{out_tree=}")

with open(f"x_aot_{topo}.pickle", "wb") as f:
    pickle.dump(serialized, f)
with open("x_in_tree_{topo}.pickle", "wb") as f:
    pickle.dump(in_tree, f)
with open("x_out_tree_{topo}.pickle", "wb") as f:
    pickle.dump(out_tree, f)

ex_input = 2.0 * jnp.ones((128, 128), dtype=jnp.float32)

if topo=='v4-8':
    ## Run locally instead of loading the pickle
    compiled = deserialize_and_load(serialized, in_tree, out_tree)

    cost = compiled.cost_analysis()[0]['flops']
    print(f"{cost=}")

    
    print(f"{compiled(ex_input)}")

out_shaped = jax.eval_shape(fun, ex_input)
flat_out_shaped, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
print(f"{out_tree_recreated=}")

ex_input = jax.core.ShapedArray(shape=(128, 128), dtype=np.float32)
flat_in_shaped, in_tree_recreated = tree_util.tree_flatten(ex_input)