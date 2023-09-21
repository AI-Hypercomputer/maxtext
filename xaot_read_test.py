import pickle
from jax.experimental.serialize_executable import deserialize_and_load
import jax.numpy as jnp
topo='v4-16'
print("Start pickle reading")

with open(f"x_aot_{topo}.pickle", "rb") as f:
    serialized_compiled = pickle.load(f)
with open(f"x_in_tree_{topo}.pickle", "rb") as f:
    in_tree = pickle.load(f)
with open(f"x_out_tree_{topo}.pickle", "rb") as f:
    out_tree = pickle.load(f)

print("Finish pickle reading")

compiled = deserialize_and_load(serialized_compiled, in_tree, out_tree)

print("finished deserializing")

cost = compiled.cost_analysis()[0]['flops']
print(f"{cost=}")

ex_input = 2.0 * jnp.ones((128, 128), dtype=jnp.float32)
print(f"{compiled(ex_input)}")