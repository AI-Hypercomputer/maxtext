#!/usr/bin/python3

# SPDX-License-Identifier: Apache-2.0

"""
This programs demonstrates embarrassingly parallelizable non-SPMD computations in Jax, in this case by having each
process_index run its own computation.
The same approach can be extended for non-embarrassingly parallelizable computations.
The simplest way to do that would be by running embarrassingly parallelizable computations on arbitrary submeshes,
then using a `host_local_array_to_global_array` to reshard into a new global array.
An important limitation of this approach is that we cannot overlap communication and computation between the different
kernel calls.
"""

import numpy as np

import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh


# Notice this is jax.local_devices(), not jax.devices(). Hence each process (on TPUVMs, each VM) will run separate programs
# on its mesh.
mesh = Mesh(np.array(jax.local_devices()), ["data"])
sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(None))
idx = jax.process_index()


# Example step depends on idx which is different on each program
def example_step():
  return idx * jax.numpy.ones((idx + 1))


jit_func = jax.jit(
    example_step,
    out_shardings=sharding,
)

# pylint: disable=not-callable
print(f"{idx=} -> {jit_func()=}")
