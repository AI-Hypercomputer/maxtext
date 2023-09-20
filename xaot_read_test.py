import pickle
from jax.experimental.serialize_executable import deserialize_and_load
import jax.numpy as jnp

print("Start pickle reading")

with open("x_aot.pickle", "rb") as f:
    serialized_compiled = pickle.load(f)
with open("x_in_tree.pickle", "rb") as f:
    in_tree = pickle.load(f)
with open("x_out_tree.pickle", "rb") as f:
    out_tree = pickle.load(f)

print("Finish pickle reading")

compiled = deserialize_and_load(serialized_compiled, in_tree, out_tree)

print("finished deserializing")

cost = compiled.cost_analysis()[0]['flops']
print(f"{cost=}")

ex_input = 2.0 * jnp.ones((128, 128), dtype=jnp.float32)
print(f"{compiled(ex_input)}")