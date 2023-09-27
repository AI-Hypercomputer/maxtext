import jax
from jax import numpy as jnp
import numpy as np
from functools import partial

from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.experimental.pjit import pjit

from jax.experimental.multihost_utils import host_local_array_to_global_array, process_allgather, _local_to_global_aval

from jax._src.interpreters import pxla

import sys

import max_utils
jax.config.update("jax_spmd_mode", "allow_all")

def find_np_idx(array, filter):
  for idx, val in np.ndenumerate(array):
    if filter(val):
      return idx

def my_slice_devices(device_array):
  ### slices are assumed to be restricted to the first axis
  idx = find_np_idx(device_array, lambda x : x.process_index == jax.process_index())
  zeroth_idx = idx[0]
  sliced_result =  device_array[zeroth_idx:zeroth_idx+1, :,:]
  return zeroth_idx, sliced_result

class DummyConfig(object):
  def __init__(self):
    self.dcn_data_parallelism = -1
    self.dcn_fsdp_parallelism = 1
    self.dcn_tensor_parallelism = 1
    self.ici_data_parallelism = 1
    self.ici_fsdp_parallelism = -1
    self.ici_tensor_parallelism = 1
    self.mesh_axes = ['data', 'fsdp', 'tensor']

config = DummyConfig()

def make_data():
  return jax.numpy.ones(1024)

def _sum(x):
  return jax.tree_map(partial(jnp.sum, axis=0), x)

def broadcast_one_slice_to_all(in_tree, is_source, full_mesh, per_slice_sharding, num_slices):
  @partial(jax.jit, static_argnums=0)
  def fake_zero_data(sharding, x):
    x = jnp.zeros_like(x)
    return jax.lax.with_sharding_constraint(x, sharding)

  def pre_jit(x):
    if is_source:
      inp = x
    else:
      inp = fake_zero_data(per_slice_sharding, x)
    inp = jnp.expand_dims(inp, axis=0)
    in_spec = PartitionSpec('data', *x.sharding.spec)
    global_shape = (num_slices, *x.shape)
    global_sharding = jax.sharding.NamedSharding(full_mesh, in_spec)
    print(global_shape , global_sharding, x.shape)
    return jax.make_array_from_single_device_arrays(global_shape, global_sharding, [s.data for s in inp.addressable_shards])

  out_sharding = jax.tree_map(lambda x: jax.sharding.NamedSharding(full_mesh, PartitionSpec(*x.sharding.spec)), in_tree)

  in_tree = jax.tree_map(pre_jit, in_tree)
  print(f"{in_tree.shape=}")
  out_tree = jax.jit(_sum, out_shardings=out_sharding)(in_tree)
  return out_tree

# steps:
### create global mesh
### create per slice mesh
### run the lift.

def main():

  global_devices = max_utils.create_device_mesh(config, logging=False)
  num_slices = global_devices.shape[0]
  global_mesh = Mesh(global_devices, config.mesh_axes)
  print(f"Global device shape {global_devices.shape}")
  this_worker_slice_idx, slice_devices  = my_slice_devices(global_devices)
  slice_mesh = Mesh(slice_devices, config.mesh_axes)
  print(f"{slice_devices.shape=} and {this_worker_slice_idx=}")

  pjit_make_data = pjit(make_data, in_shardings=None, out_shardings=PartitionSpec(("fsdp", "tensor")))
  with slice_mesh:
    data = 1 + this_worker_slice_idx * pjit_make_data()
    jax.debug.visualize_array_sharding(data)

  per_slice_sharding = jax.sharding.NamedSharding(slice_mesh, PartitionSpec(("fsdp", "tensor")))
  output = broadcast_one_slice_to_all(data, this_worker_slice_idx==0, global_mesh, per_slice_sharding, num_slices)

  for s in output.addressable_shards:
    print(f"{s.data.shape=}, {s.index=}, {s.replica_id=}, {output.sharding=}")

  print(f"{process_allgather(output)=}, {output.shape=}")

  sys.exit(0)

if __name__ == "__main__":
  main()

### generating data per chip
start_mesh = max_utils.create_device_mesh(config)

submeshes = []
isolated_data = []

for i in range(4):
  sub_devices = m[i:i+1, :]
  submeshes.append(Mesh(sub_devices, ["axis1", "axis2"]))
  pjit_make_data = pjit(make_data, in_shardings=None, out_shardings=PartitionSpec(("axis1", "axis2")))
  with submeshes[-1]:
    isolated_data.append(pjit_make_data())
    jax.debug.visualize_array_sharding(isolated_data[-1])



##Running normally on one slice
sys.exit(0)

full_mesh = Mesh(m, ["axis1", "axis2"])


from jax.experimental.pjit import pjit
pjit_make_data = pjit(make_data, in_shardings=None, out_shardings=PartitionSpec(("axis1", "axis2")))

with mesh:
  data = pjit_make_data()
  jax.debug.visualize_array_sharding(data)

