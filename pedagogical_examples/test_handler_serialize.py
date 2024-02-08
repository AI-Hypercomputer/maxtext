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
import dataclasses
import json
import jax
import numpy as np
import os
import orbax.checkpoint as ocp
from orbax.checkpoint import future
from orbax.checkpoint import utils
import string
import sys

from etils import epath
from etils import epath
from jax import sharding
from jax import numpy as jnp
# from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import checkpoint_args
# from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
# from orbax.checkpoint import type_handlers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast


LETTERS = list(string.ascii_lowercase)

PyTree = Any
LegacyTransformFn = Callable[[PyTree, PyTree, PyTree], Tuple[PyTree, PyTree]]


# _VALUE_TYPE = 'value_type'
# _SKIP_DESERIALIZE = 'skip_deserialize'
# _METADATA_FILE = '_METADATA'
# _TREE_METADATA_KEY = 'tree_metadata'
# _KEY_METADATA_KEY = 'key_metadata'
# _VALUE_METADATA_KEY = 'value_metadata'
# _USE_ZARR3 = 'use_zarr3'


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
  return pytree, mesh_tree, axes_tree, data_axis_name


def check_trees_equal(tree1, tree2):
  def check_same(key, v1, v2):
     assert jax.numpy.allclose(
            v1, v2, rtol=1e-06, atol=1e-06
        ), 'Error!!! Restored values are not close enough'
  jax.tree_util.tree_map_with_path(check_same, tree1, tree2)
  print('Hooray, values are close enough!')


def is_sharding_valid(single_replica_ids, single_replica_pids):
  if jax.process_index() in single_replica_pids:
    loc_devieces_in_replica = single_replica_ids.intersection(set([d.id for d in jax.local_devices()]))
    # print(f' host ID is {jax.process_index()}, '
          # f'num of devices in replica 0 is {len(loc_devieces_in_replica)}')
    assert len(loc_devieces_in_replica) == 4, (
      ' !!! Provided sharding is not valid. There is a host with part'
      ' of devices outside of replica 0')


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


class SingleReplicaArrayHandler(
    type_handlers.SingleReplicaArrayHandler
):

  def _get_param_names(self, item: PyTree) -> PyTree:
    """Gets parameter names for PyTree elements."""

    def _param_name_from_keypath(keypath: Tuple[Any, ...]) -> str:
      return '.'.join([str(utils.get_key_name(k)) for k in keypath])

    return jax.tree_util.tree_map_with_path(
        lambda kp, _: _param_name_from_keypath(kp),
        item,
        is_leaf=utils.is_empty_or_leaf,
  )

  def _get_param_infos(
      self,
      item: PyTree,
      directory: epath.Path,
  ) -> Tuple[PyTree, bool]:
    """Returns parameter information for elements in `item`.

    Args:
      item: a PyTree to extract information from.
      directory: a directory where checkpoint files are located.

    Returns:
      A PyTree matching `item` of ParamInfo
    """
    if not item:
      raise ValueError('Found empty item')
    names = self._get_param_names(item)

    def _param_info(name):
      return type_handlers.ParamInfo(
          name=name,
          path=(directory / name),
          parent_dir=directory,
      )

    return jax.tree_util.tree_map(_param_info, names)


  def get_serialization_requests(
    self,
    tree,
    param_infos: type_handlers.ParamInfo,
    request_args
    ):
    values = []
    infos = []
    args = []
    keys = []

    def _group_value(
        keypath: Tuple[Any, ...],
        info: type_handlers.ParamInfo,
        value: Any,
        arg: Union[type_handlers.SaveArgs, type_handlers.SingleReplicaArrayRestoreArgs],
    ):
      nonlocal values
      nonlocal infos
      nonlocal args
      nonlocal keys
      tuple_key = utils.tuple_path_from_keypath(keypath)
      keys.append(tuple_key)
      values.append(value)
      infos.append(info)
      args.append(arg)

    jax.tree_util.tree_map_with_path(
        _group_value,
        param_infos,
        tree,
        request_args
    )
    return keys, values, infos, args


  async def async_save(self,
        directory: epath.Path,
        item: Optional[PyTree] = None,
        save_args = None,
    ) -> Optional[List[future.Future]]:

      save_args = jax.tree_util.tree_map(
          type_handlers._maybe_set_default_save_args,
          item,
          save_args,
          is_leaf=utils.is_empty_or_leaf,
      )

      param_infos = self._get_param_infos(
          item, directory
      )

      _, serial_values, serial_infos, serial_args = self.get_serialization_requests(
            item, param_infos, save_args
        )

      serialize_ops = [self.serialize(serial_values, serial_infos, serial_args)]
      commit_futures = await asyncio.gather(*serialize_ops)
      commit_futures, _ = jax.tree_util.tree_flatten(commit_futures)

      return commit_futures

  async def _maybe_deserialize(
      self,
      structure: PyTree,
      param_infos: PyTree,
      restore_args: PyTree
  ) -> PyTree:
    """Deserializes values or gets them from the aggregate file."""

    serial_keys, _, serial_infos, serial_args = (
      self.get_serialization_requests(
            structure, param_infos, restore_args
        )
     )

    deserialized_batches_ops = [self.deserialize(serial_infos, serial_args)]
    deserialized_batches = await asyncio.gather(*deserialized_batches_ops)

    flat_restored = {}
    for key, value in zip(serial_keys, deserialized_batches[0]):
      flat_restored[key] = value

    return utils.from_flat_dict(flat_restored, target=structure)

  def async_restore(
      self,
      structure,
      directory: epath.Path,
      restore_args: Optional[PyTree] = None,
  ) -> PyTree:

    param_infos = self._get_param_infos(
          structure, directory
      )
    restored_item = asyncio.run(
        self._maybe_deserialize(structure, param_infos, restore_args)
    )

    utils.sync_global_devices('SingleReplicaArrayHandler:restore')
    return restored_item


def main(args):

  train_state, mesh_tree, axes_tree, data_axis_name = setup_sharded_pytree(
    args.tree_size,
    args.mesh_shape
    )

  path = epath.Path(args.path)

  handler = SingleReplicaArrayHandler()
  type_handlers.register_type_handler(jax.Array,
                                      SingleReplicaArrayHandler(),
                                      override=True)
  handler = SingleReplicaArrayHandler()
  save_args = jax.tree_util.tree_map(lambda x: type_handlers.SaveArgs(), train_state)

  asyncio.run(handler.async_save(path,
                    item=train_state,
                    save_args=save_args,)
              )

  # # print('Data is saved, now restoring!')
  empty_state = jax.tree_util.tree_map(
          # lambda x: x+1,
          lambda x: jax.numpy.zeros_like(x),
          train_state,
          is_leaf=is_leaf
      )

  # empty_state = empty_tree = jax.numpy.zeros_like(train_state)

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
  restored = handler.async_restore(
        empty_state,
        path,
        restore_args=restore_args,
    )

  # print('============================================================')
  # print(restored)

  check_trees_equal(train_state, restored)

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
