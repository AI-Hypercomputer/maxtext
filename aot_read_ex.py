import pickle
from jax.experimental.serialize_executable import deserialize_and_load

print("Start pickle reading")

with open("aot.pickle", "rb") as f:
    serialized_compiled = pickle.load(f)
with open("in_tree.pickle", "rb") as f:
    in_tree = pickle.load(f)
with open("out_tree.pickle", "rb") as f:
    out_tree = pickle.load(f)

print("Finish pickle reading")

compiled = deserialize_and_load(serialized_compiled, in_tree, out_tree)

print("finished deserializing")

cost = compiled.cost_analysis()[0]['flops']
print(f"{cost=}")



#f(x, y):=    2 * x + y

x, y = 8, 4
res = compiled(x, y)
print(f"{res=}")
# DeviceArray(10, dtype=int32)