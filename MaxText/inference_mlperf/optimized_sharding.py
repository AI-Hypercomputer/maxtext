import jax
from jax.experimental import mesh_utils
import jaxtyping
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from typing import Sequence, Mapping


GLOBAL_BATCH = 8*1024
EMBED = 8192
MLP = 28672
LAYERS = 80

devices = jax.devices()
print()
print(devices)
print()

mesh_axis_names=("model", "seq")
mesh_dim = (8, 1)
mesh = Mesh(devices=mesh_utils.create_device_mesh(mesh_dim), axis_names=mesh_axis_names)
print("Mesh")
print(mesh)
print()

A = jax.numpy.ones((GLOBAL_BATCH, EMBED), dtype=jax.numpy.bfloat16)
activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model"))
A_ = jax.device_put(A, activation_sharding)
print("Sharding A")
jax.debug.visualize_array_sharding(A_)

print()
print()
W1 = jax.numpy.ones((EMBED, MLP), dtype=jax.numpy.bfloat16)
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("model",))
W1_ = jax.device_put(W1, weight_sharding)
print("Sharding W1")
jax.debug.visualize_array_sharding(W1_)



# ------------------------------------------------------------------------------------

mesh_axis_names=("model", "seq")
mesh_dim = (2, 4)
mesh = Mesh(devices=mesh_utils.create_device_mesh(mesh_dim), axis_names=mesh_axis_names)
print("Mesh")
print(mesh)
print()

A = jax.numpy.ones((GLOBAL_BATCH, EMBED), dtype=jax.numpy.bfloat16)
activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model"))
A_ = jax.device_put(A, activation_sharding)
print(f"Sharding A: {A.shape=}, {mesh_dim=}")
jax.debug.visualize_array_sharding(A_)

print()
print()
W1 = jax.numpy.ones((EMBED, MLP), dtype=jax.numpy.bfloat16)
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("model",))
W1_ = jax.device_put(W1, weight_sharding)
print(f"Sharding W1: {W1.shape=}, {mesh_dim=}")
jax.debug.visualize_array_sharding(W1_)


# -----------------------------------------------------------------------------------


mesh_axis_names=("model", "seq")
mesh_dim = (4, 2)
mesh = Mesh(devices=mesh_utils.create_device_mesh(mesh_dim), axis_names=mesh_axis_names)
print("Mesh")
print(mesh)
print()

A = jax.numpy.ones((GLOBAL_BATCH, EMBED), dtype=jax.numpy.bfloat16)
activation_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "model"))
A_ = jax.device_put(A, activation_sharding)
print(f"Sharding A: {A.shape=}, {mesh_dim=}")
jax.debug.visualize_array_sharding(A_)

print()
print()
W1 = jax.numpy.ones((EMBED, MLP), dtype=jax.numpy.bfloat16)
weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("model",))
W1_ = jax.device_put(W1, weight_sharding)
print(f"Sharding W1: {W1.shape=}, {mesh_dim=}")
jax.debug.visualize_array_sharding(W1_)