print("Importing torch...")
import torch
print("Importing jax...")
import jax
import jax.numpy as jnp
print("Running a simple JAX calculation...")
x = jnp.arange(10)
print(f"JAX works: {x.sum().item()}")

print("Importing tensorflow...")
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

print("Importing tensorstore...")
import tensorstore as ts
print("TensorStore imported successfully!")

print("Importing orbax...")
import orbax.checkpoint
print("Orbax imported successfully!")

print("All imports and basic operations completed successfully!")
