import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
print("importing jax", flush=True)


import jax
print("jax imported", flush=True)


print(jax.devices())