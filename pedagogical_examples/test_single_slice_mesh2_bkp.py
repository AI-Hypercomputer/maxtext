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
  pytree = {
        'a': np.arange(N*8).reshape((8, N)) * 1,
        'b': np.arange(N*4).reshape((4, N)) * 2,
        'c': {
            'a': np.arange(N*8).reshape((8, N)) * 3,
            'e': np.arange(N*4).reshape((4, N)) * 4,
        },
    }
  return pytree


def setup_sharded_pytree(
    pytree_N,
    shape: List,
):
  """Creates a PyTree of sharded arrays for testing."""

  pytree = make_pytree(pytree_N)
  devices = jax.devices()
  num_devices = len(devices)
  devices = np.asarray(devices)

  dim = len(shape)
  data_axis_name = LETTERS[-dim]
  mesh = jax.sharding.Mesh(
      devices.reshape(shape), LETTERS[-dim:]
  )
  # mesh_axes = jax.sharding.PartitionSpec(LETTERS[-dim+1:])
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


def get_replica_pids(rep_id, mesh):
  replica_devices = np.take(mesh.devices, rep_id, axis=0).flatten()
  pids = set([d.process_index for d in replica_devices])
  ids = set([d.id for d in replica_devices])
  return pids, ids


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

  if args.restore_method == 'singleslice':
    type_handlers.register_type_handler(jax.Array,
                                      type_handlers.SingleSliceArrayHandler(),
                                      override=True)
    print('Restoring with single slice')
    def _create_restore_args(data, mesh, pspec):
      rep0_pids, rep0_ids = get_replica_pids(0, mesh)
      return type_handlers.SingleSliceArrayRestoreArgs(
          sharding=jax.sharding.NamedSharding(mesh, pspec),
          single_slice_sharding=jax.sharding.NamedSharding(mesh, pspec),
          single_replica_ids = rep0_ids,
          single_replica_pids = rep0_pids,
          replica_axis_name=data_axis_name,
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
        cast(type_handlers.SingleSliceArrayRestoreArgs, restore_args)}
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
                     'of ["singleslice", "arrayrestore", "orig"]')

  # print('restored', restored)
  print('--------------')
  check_trees_equal(train_state, restored)
  # print(jax.tree_util.tree_map(lambda x: x.sharding, restored))

def check_trees_equal(tree1, tree2):
  def check_same(key, v1, v2):
     assert jax.numpy.allclose(
            v1, v2, rtol=1e-06, atol=1e-06
        )
  jax.tree_util.tree_map_with_path(check_same, tree1, tree2)
  print('Hooray, values are close enough!')

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
