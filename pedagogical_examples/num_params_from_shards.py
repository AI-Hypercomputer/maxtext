import argparse
from collections import defaultdict
import jax
import numpy as np
import sys


def is_leaf(x):
  return (
      isinstance(x, np.ndarray)
      or isinstance(x, jax.sharding.Mesh)
      or isinstance(x, jax.sharding.PartitionSpec)
      # or isinstance(x, pytree_checkpoint_handler.ParamInfo)
  )


def create_sharded_array(arr, mesh, mesh_axes):
  """Create sharded jax.Array."""
  if isinstance(arr, (int, float)):
    arr = np.asarray(arr)
  return jax.make_array_from_callback(
      arr.shape,
      jax.sharding.NamedSharding(mesh, mesh_axes),
      lambda idx: arr[idx]
  )


def make_pytree(N):
  pytree = {
        'a': np.arange(N).reshape((2, N//2)) * 1,
        'b': {
            'a': np.arange(N).reshape(1, N),
            'e': np.arange(N//2).reshape((2, N//4)),
        },
    }
  return pytree


def setup_sharded_pytree(
):
  """Creates a PyTree of sharded arrays for testing."""

  pytree = make_pytree(16)
  devices = np.asarray(jax.devices())

  mesh = jax.sharding.Mesh(devices.reshape((2, 2)), ['x', 'y'])
  print('mesh', mesh)
  print(mesh.shape)

  mesh_axes = jax.sharding.PartitionSpec(None, 'y')

  mesh_tree = {
      'a': mesh,
      'b': {
          'a': mesh,
          'e': mesh,
      },
  }
  axes_tree = {
      'a': mesh_axes,
      'b': {
          'a': mesh_axes,
          'e': mesh_axes,
      },
  }

  pytree = jax.tree_util.tree_map(
      create_sharded_array, pytree, mesh_tree, axes_tree, is_leaf=is_leaf
  )
  return pytree, mesh_tree, axes_tree


def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  print('params sizes', params_sizes)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  assert total_parameters >= 0
  return total_parameters


def main():
  devices = np.asarray(jax.devices())
  mesh = jax.sharding.Mesh(devices.reshape((2,2)), ['x','y'])
  mesh_axis = jax.sharding.PartitionSpec(None,'y')

  a = np.arange(2*8).reshape((2, 8))
  sharded_a = jax.make_array_from_callback(
    a.shape,
    jax.sharding.NamedSharding(mesh, mesh_axis),
    lambda idx: a[idx])
  params_per_shard = defaultdict(int)
  print('addressable shards')
  print(sharded_a.addressable_shards)
  for shard in sharded_a.addressable_shards:
    dev_id = shard.device.id
    params_per_shard[dev_id] += np.prod(shard.data.shape)
  print('Done')
  print(params_per_shard)

def main_pytree():
  pytree, mesh_tree, axes_tree = setup_sharded_pytree()
  sharded = jax.tree_util.tree_map(lambda x: x.addressable_shards, pytree, is_leaf=is_leaf)
  print(sharded)

  print(calculate_num_params_from_pytree(pytree))
  # print(jax.tree_util.tree_map(lambda x: np.prod(x.data.shape), pytree))
  print('==========')
  params_per_device = defaultdict(int)
  def calc_shard_params(shard):
    dev_id = shard.device.id
    params_per_device[dev_id] += np.prod(shard.data.shape)
  jax.tree_util.tree_map(calc_shard_params, sharded)
  print(params_per_device)
  # print(jax.tree_util.tree_map(lambda x: x.addressable_shards, pytree, is_leaf=is_leaf))
  # print(pytree.addressable_shards)
  # def params_per_device():
  # print(calculate_num_params_from_pytree(pytree))

def parser(args):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--is-pytree',
    action='store_true',
    help='whether test pytree or simply jax array'
    )
  return parser.parse_args(args)

if __name__ == '__main__':
  args = parser(sys.argv[1:])
  if args.is_pytree:
    main_pytree()
  else:
    main()

