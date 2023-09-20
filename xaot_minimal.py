from jax.experimental.topologies import get_topology_desc
import jax
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import numpy as np
from jax.experimental.serialize_executable import serialize, deserialize_and_load
import pickle
import jax.numpy as jnp



topology_devices = get_topology_desc(
    platform='tpu',
    topology_name=f'v4:2x2x1',
    chip_config_name='megacore',
    chips_per_host_bounds=(2, 2, 1),
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

with open("x_aot.pickle", "wb") as f:
    pickle.dump(serialized, f)
with open("x_in_tree.pickle", "wb") as f:
    pickle.dump(in_tree, f)
with open("x_out_tree.pickle", "wb") as f:
    pickle.dump(out_tree, f)


## Run locally instead of loading the pickle
compiled = deserialize_and_load(serialized, in_tree, out_tree)

cost = compiled.cost_analysis()[0]['flops']
print(f"{cost=}")

ex_input = 2.0 * jnp.ones((128, 128), dtype=jnp.float32)
print(f"{compiled(ex_input)}")