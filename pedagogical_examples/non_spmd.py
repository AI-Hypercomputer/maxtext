#!/usr/bin/python3

"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

'''
This programs demonstrates embarrassingly parallelizable non-SPMD computations in Jax, in this case by having each
process_index run its own computation.
The same approach can be extended for non-embarrassingly parallelizable computations.
The simplest way to do that would be by running embarrassingly parallelizable computations on arbitrary submeshes,
then using a `host_local_array_to_global_array` to reshard into a new global array.
An important limitation of this approach is that we cannot overlap communication and computation between the different
kernel calls.
'''


import jax
from jax.sharding import PartitionSpec
from jax.sharding import Mesh

import numpy as np




# Notice this is jax.local_devices(), not jax.devices(). Hence each process (on TPUVMs, each VM) will run separate programs
# on its mesh.
mesh = Mesh(np.array(jax.local_devices()), ["data"])
sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(None))
idx = jax.process_index()

# Example step depends on idx which is different on each program
def example_step():
  return idx * jax.numpy.ones((idx+1))

jit_func = jax.jit(
        example_step,
        out_shardings=sharding,
      )

print(f"{idx=} -> {jit_func()=}")


