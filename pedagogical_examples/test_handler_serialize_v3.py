"""" Test saving and restoring a simple pytree.

You can specify different ways for restoring and different meshes

Example run

python3 pedagogical_examples/test_single_replica_mesh2.py \
  --path gs://your_path  \
  --restore-method singlereplica \
  --mesh-shape 1 4 \
  --tree-size 16
"""
import argparse
import asyncio
import jax
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import utils
import string
import sys

from etils import epath
from etils import epath
from jax import sharding
from orbax.checkpoint import type_handlers

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


LETTERS = list(string.ascii_lowercase)

PyTree = Any
LegacyTransformFn = Callable[[PyTree, PyTree, PyTree], Tuple[PyTree, PyTree]]
SingleReplicaArrayHandler = type_handlers.SingleReplicaArrayHandler


def is_leaf(x):
  return (
      isinstance(x, np.ndarray)
      or isinstance(x, jax.sharding.Mesh)
      or isinstance(x, jax.sharding.PartitionSpec)
      or isinstance(x, type_handlers.ParamInfo)
  )


def create_sharded_array(arr, mesh, mesh_axes):
  """Create sharded jax.Array."""
  if isinstance(arr, (int, float)):
    arr = np.asarray(arr)
  return jax.make_array_from_callback(
      arr.shape, sharding.NamedSharding(mesh, mesh_axes), lambda idx: arr[idx]
  )


def make_pytree(N):
  pytree = {
        'a': np.arange(N*8).reshape((8, N)) * 1,
        'b': np.arange(N*4).reshape((4, N)) * 2,
        'c': {
            'a': np.arange(N*8).reshape((8, N)) * 3,
            'e': np.arange(N*4).reshape((4, N)) * 4,
        },
    }
  return pytree


def make_array(N):
  arr = (np.arange(N*8).reshape((8, N)) * 1,
         np.arange(N*4).reshape((4, N)) * 2,
         np.arange(N*8).reshape((8, N)) * 3,
         np.arange(N*4).reshape((4, N)) * 4,)

  return arr


def setup_sharded_pytree(
    pytree_N,
    shape: List,
):
  """Creates a PyTree of sharded arrays for testing."""

  pytree = make_pytree(pytree_N)
  devices = jax.devices()
  devices = np.asarray(devices)

  dim = len(shape)
  mesh = jax.sharding.Mesh(
      devices.reshape(shape), LETTERS[-dim:]
  )

  # mesh_axes = jax.sharding.PartitionSpec(LETTERS[-dim+1:])
  mesh_axes = jax.sharding.PartitionSpec(None, LETTERS[-dim+1:])
  # print('-------pspec', mesh_axes)
  mesh_tree = {
      'a': mesh,
      'b': mesh,
      'c': {
          'a': mesh,
          'e': mesh,
      },
  }
  axes_tree = {
      'a': mesh_axes,
      'b': mesh_axes,
      'c': {
          'a': mesh_axes,
          'e': mesh_axes,
      },
  }

  pytree = jax.tree_util.tree_map(
      create_sharded_array, pytree, mesh_tree, axes_tree, is_leaf=is_leaf
  )
  flat_pytree, _ = jax.tree_util.tree_flatten(pytree)
  return pytree, flat_pytree, mesh, mesh_axes
  # return pytree, mesh_tree, axes_tree, data_axis_name


def setup_sharded_array(
    pytree_N,
    shape: List,
):
  """Creates a PyTree of sharded arrays for testing."""

  arr = make_array(pytree_N)
  devices = jax.devices()
  devices = np.asarray(devices)

  dim = len(shape)
  mesh = jax.sharding.Mesh(
      devices.reshape(shape), LETTERS[-dim:]
  )

  # mesh_axes = jax.sharding.PartitionSpec(LETTERS[-dim+1:])
  mesh_axes = jax.sharding.PartitionSpec(None, LETTERS[-dim+1:])
  # print('-------pspec', mesh_axes)

  sharded_arr = [create_sharded_array(a, mesh, mesh_axes) for a in arr]
  # flat_pytree, _ = jax.tree_util.tree_flatten(pytree)
  return sharded_arr, mesh, mesh_axes


def check_trees_equal(tree1, tree2):
  def check_same(key, v1, v2):
     assert jax.numpy.allclose(
            v1, v2, rtol=1e-06, atol=1e-06
        ), 'Error!!! Restored values are not close enough'
  jax.tree_util.tree_map_with_path(check_same, tree1, tree2)
  print('Hooray, values are close enough!')


def _find_zeroth_idx(array):
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[0]


def _replica_devices(device_array):
  zeroth_idx =  _find_zeroth_idx(device_array)
  replica_result = device_array[zeroth_idx : zeroth_idx + 1]
  return replica_result


def _create_restore_args(data, mesh, pspec):
  replica_devices = _replica_devices(mesh.devices)
  replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
  ss_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

  return type_handlers.SingleReplicaArrayRestoreArgs(
      sharding=jax.sharding.NamedSharding(mesh, pspec),
      single_replica_sharding=ss_sharding,
      replica_axis_index=0,
      global_shape=data.shape,
      dtype=data.dtype,
  )


def main(args):

  path = epath.Path(args.path)
  flat_state, mesh, mesh_axes = setup_sharded_array(
    args.tree_size,
    args.mesh_shape
    )

  handler = SingleReplicaArrayHandler()
  type_handlers.register_type_handler(jax.Array,
                                      SingleReplicaArrayHandler(),
                                      override=True)

  flat_infos = [type_handlers.ParamInfo(name=str(i), path=(path / str(i)), parent_dir=path) for i in range(len(flat_state))]
  asyncio.run(handler.serialize(flat_state, flat_infos))
  utils.sync_global_devices('Serialization complite')

  restore_args_new = [_create_restore_args(data, mesh, mesh_axes) for data in flat_state]

  restored = asyncio.run(handler.deserialize(flat_infos, restore_args_new))

  print(len(restored), type(restored), restored)

  return


def parser(args):
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--tree-size',
    type=int,
    default=16,
    help='length of the array to construc the pytree from'
    )

  parser.add_argument(
    '--path',
    type=str,
    default='/tmp/checkpoint_manager/',
    help='whether save ckpt in new folder or reuse the path'
  )

  parser.add_argument(
    '--mesh-shape',
    type=int,
    nargs="+",
    default=None,
    help='dimension of data mesh'
  )

  return parser.parse_args(args)

if __name__ == '__main__':
  args = parser(sys.argv[1:])
  main(args)
