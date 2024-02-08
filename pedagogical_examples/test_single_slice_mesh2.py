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
import jax
import numpy as np
import os
import orbax.checkpoint as ocp
import string
import sys

from etils import epath
from etils import epath
from jax import sharding
from jax import numpy as jnp
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
from typing import cast, List


LETTERS = list(string.ascii_lowercase)


def is_leaf(x):
  return (
      isinstance(x, np.ndarray)
      or isinstance(x, jax.sharding.Mesh)
      or isinstance(x, jax.sharding.PartitionSpec)
      or isinstance(x, pytree_checkpoint_handler.ParamInfo)
  )


def create_sharded_array(arr, mesh, mesh_axes):
  """Create sharded jax.Array."""
  if isinstance(arr, (int, float)):
    arr = np.asarray(arr)
  return jax.make_array_from_callback(
      arr.shape, sharding.NamedSharding(mesh, mesh_axes), lambda idx: arr[idx]
  )


def make_pytree(N):
  # pytree = {
  #       'a': np.arange(N*8).reshape((8, N)) * 1,
  #       'b': np.arange(N*4).reshape((4, N)) * 2,
  #       'c': {
  #           'a': np.arange(N*8).reshape((8, N)) * 3,
  #           'e': np.arange(N*4).reshape((4, N)) * 4,
  #       },
  #   }
  pytree = {
        'a': np.arange(N*N).reshape((N, N)) * 1,
        'b': np.arange(N*4*N).reshape((4*N, N)) * 2,
        'c': {
            'a': np.arange(N*N).reshape((N, N)) * 3,
            'e': np.arange(N*4*N).reshape((4*N, N)) * 4,
        },
    }
  return pytree


def setup_sharded_pytree(
    pytree_N,
    shape: List,
    replica_idx: int = 0,
):
  """Creates a PyTree of sharded arrays for testing."""

  pytree = make_pytree(pytree_N)
  devices = jax.devices()
  num_devices = len(devices)
  devices = np.asarray(devices)

  dim = len(shape)
  data_axis_name = LETTERS[-dim + replica_idx]
  mesh = jax.sharding.Mesh(
      devices.reshape(shape), LETTERS[-dim:]
  )

  mesh_axes = jax.sharding.PartitionSpec(None, LETTERS[-dim+1:])

  print('-------pspec', mesh_axes)
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
  return pytree, mesh_tree, axes_tree, data_axis_name


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
  ### replicas are assumed to be restricted to the first axis
  zeroth_idx =  _find_zeroth_idx(device_array)
  replica_result = device_array[zeroth_idx : zeroth_idx + 1]
  return replica_result


def main(args):

  train_state, mesh_tree, axes_tree, data_axis_name = setup_sharded_pytree(
    args.tree_size,
    args.mesh_shape
    )

  path = epath.Path(args.path)

  options = ocp.CheckpointManagerOptions(
      max_to_keep=3,
      save_interval_steps=2
  )
  mngr = ocp.CheckpointManager(
      path,
      ocp.PyTreeCheckpointer(),
      options=options
  )

  @jax.jit
  def train_fn(state):
    return jax.tree_util.tree_map(lambda x: x + 1, state)

  # num_steps = 2
  # for step in range(num_steps):
  #   train_state = train_fn(train_state)
  #   mngr.save(step, train_state)
  mngr.save(0, train_state)
  # print('state after training', train_state)

  empty_state = jax.tree_util.tree_map(
          lambda x: x, train_state, is_leaf=is_leaf
      )

  if args.restore_method == 'singlereplica':
    type_handlers.register_type_handler(jax.Array,
                                      type_handlers.SingleReplicaArrayHandler(),
                                      override=True)
    # print('Restoring with single replica', jax.process_index())

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

    restore_args = jax.tree_util.tree_map(
      _create_restore_args,
      empty_state,
      mesh_tree,
      axes_tree
      )

    restored = mngr.restore(
      mngr.latest_step(),
      items=empty_state,
      restore_kwargs={'restore_args':
        cast(type_handlers.SingleReplicaArrayRestoreArgs, restore_args)}
      )
  elif args.restore_method == 'arrayrestore':
    print('Restoring with ArrayRestoreArgs')
    def map_to_pspec(data, mesh, pspec):
      return type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)

    restore_args = jax.tree_util.tree_map(
      map_to_pspec,
      empty_state,
      mesh_tree,
      axes_tree
      )

    restored = mngr.restore(
      mngr.latest_step(),
      empty_state,
      restore_kwargs={'restore_args': restore_args}
      )
  elif args.restore_method == 'orig':
    print('Restoring in default way')
    shardings = jax.tree_map(lambda x: x.sharding, empty_state)
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        empty_state, shardings)
    print('restore args', restore_args)

    restored = mngr.restore(
      mngr.latest_step(),
      items=empty_state,
      restore_kwargs={'restore_args': restore_args},
    )
  else:
    raise ValueError(f'Wrong value for restore-method is passed. Must be one '
                     'of ["singlereplica", "arrayrestore", "orig"]')

  check_trees_equal(train_state, restored)

  return


def parser(args):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--restore-method',
    type=str,
    default='restoreargs',
    help='specifies how to restore the ckpt'
    )

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
