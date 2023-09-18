import jax
import jax.numpy as jnp
import numpy as np
import pickle
from jax.experimental.serialize_executable import serialize

def f(x, y):
    return 2 * x + y
x, y = 3, 4

lowered = jax.jit(f).lower(x, y)

# Print lowered HLO
print("Printing Lowered HLO:")
print(lowered.as_text())
# module @jit_f.0 {
#   func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
#     %0 = stablehlo.constant dense<2> : tensor<i32>
#     %1 = stablehlo.multiply %0, %arg0 : tensor<i32>
#     %2 = stablehlo.add %1, %arg1 : tensor<i32>
#     return %2 : tensor<i32>
#   }
# }

compiled = lowered.compile()

# blob = compiled.serialize()
serialized, in_tree, out_tree = serialize(compiled)
# with open('my_blob_aot', 'w') as f:
#   f.write(serialized)

with open("aot.pickle", "wb") as f:
    pickle.dump(serialized, f)
with open("in_tree.pickle", "wb") as f:
    pickle.dump(in_tree, f)
with open("out_tree.pickle", "wb") as f:
    pickle.dump(out_tree, f)

# Query for cost analysis, print FLOP estimate
cost = compiled.cost_analysis()[0]['flops']
print(f"{cost=}")
#2.0

# Execute the compiled function!
res = compiled(x, y)
print(f"{res=}")
# DeviceArray(10, dtype=int32)