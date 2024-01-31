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
from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
# from orbax.checkpoint import type_handlers
from typing import Any, Callable, Dict, List, Optional, Tuple, cast


LETTERS = list(string.ascii_lowercase)

PyTree = Any
LegacyTransformFn = Callable[[PyTree, PyTree, PyTree], Tuple[PyTree, PyTree]]


_VALUE_TYPE = 'value_type'
_SKIP_DESERIALIZE = 'skip_deserialize'
_METADATA_FILE = '_METADATA'
_TREE_METADATA_KEY = 'tree_metadata'
_KEY_METADATA_KEY = 'key_metadata'
_VALUE_METADATA_KEY = 'value_metadata'
_USE_ZARR3 = 'use_zarr3'


class SingleReplicaArrayHandler(
    type_handlers.SingleReplicaArrayHandler
):
  def __init__(self,):
    super().__init__()
    self._aggregate_handler = aggregate_handlers.MsgpackHandler()
    self._aggregate_filename = 'checkpoint'
    self._concurrent_gb = 96

  async def _write_aggregate_file(
      self,
      directory: epath.Path,
      item: PyTree,
      param_infos: PyTree,
      save_args: PyTree,
  ) -> future.Future:
    ser_item = pytree_checkpoint_handler._get_tree_for_aggregation(
      param_infos,
      save_args, item
    )
    return await self._aggregate_handler.serialize(
        directory / self._aggregate_filename, ser_item
    )

  def _get_param_names(self, item: PyTree) -> PyTree:
    """Gets parameter names for PyTree elements."""
    return pytree_checkpoint_handler._get_param_names(item)

  def _get_param_infos(
      self, item: PyTree, directory: epath.Path, save_args: PyTree
  ) -> Tuple[PyTree, bool]:
    """Returns parameter information for elements in `item`.

    At minimum, this method should extract the names of each parameter for
    saving/restoring.

    Args:
      item: a PyTree to extract information from.
      directory: a directory where checkpoint files are located.
      save_args: PyTree matching item containing SaveArgs.

    Returns:
      A PyTree matching `item` of ParamInfo, and a bool indicating whether all
      parameters were aggregated. The bool can enable us to skip some steps
      later, potentially saving time.
    """
    if not item:
      raise ValueError('Found empty item')
    names = self._get_param_names(item)
    all_params_aggregated = True

    def _param_info(name, args):
      nonlocal all_params_aggregated
      all_params_aggregated &= args.aggregate
      return type_handlers.ParamInfo(
          name=name,
          path=(directory / name),
          parent_dir=directory,
          skip_deserialize=args.aggregate,
      )

    return (
        jax.tree_util.tree_map(_param_info, names, save_args),
        all_params_aggregated,
    )

  async def async_save(self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      save_args = None,
      args = None,
  ) -> Optional[List[future.Future]]:
    if args is None:
      args = SingleReplicaSaveArgs(
          item=item,
          save_args=save_args,
      )

    item = args.item
    save_args = args.save_args

    save_args = jax.tree_util.tree_map(
        pytree_checkpoint_handler._maybe_set_default_save_args,
        item,
        item if save_args is None else save_args,
        is_leaf=utils.is_empty_or_leaf,
    )
    import pdb; pdb.set_trace
    param_infos, all_params_aggregated = self._get_param_infos(
        item, directory, save_args
    )

    request = pytree_checkpoint_handler._batched_serialization_requests(
          item, param_infos, save_args
      )
    request = request[0]
    serialize_ops = [request.handler.serialize(
                request.values, request.infos, request.args
            )
        ]
    commit_futures = await asyncio.gather(*serialize_ops)
    commit_futures, _ = jax.tree_util.tree_flatten(commit_futures)

    aggregate_commit_future = await self._write_aggregate_file(
        directory, item, param_infos, save_args
    )
    return [aggregate_commit_future]

  def save(self, directory, *args, **kwargs):

    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)

    asyncio.run(async_save(directory, *args, **kwargs))
    utils.sync_global_devices('Handler:save')

  def _read_metadata_file(
      self, directory: epath.Path, keep_empty_nodes: bool = False
  ) -> Tuple[PyTree, bool]:
    """Reads metadata file and returns a tree of restore types.

    Args:
      directory: directory
      keep_empty_nodes: If True, does not discard empty nodes in the tree.

    Returns:
      Tree with _InternalValueMetadata as values.

    Raises:
      FileNotFoundError: if the metadata file is not found.
    """
    path = directory / _METADATA_FILE
    if not path.exists():
      raise FileNotFoundError(
          f'Metadata file (named {_METADATA_FILE}) does not exist at'
          f' {directory}.'
      )

    metadata_dict = json.loads(path.read_text())

    if _USE_ZARR3 in metadata_dict:
      use_zarr3_metadata = metadata_dict[_USE_ZARR3]
    else:
      use_zarr3_metadata = False

    tree_metadata = cast(
        Dict[Any, Any],
        metadata_dict,
    )[_TREE_METADATA_KEY]
    flat_tree_metadata = []
    for metadata in tree_metadata.values():
      keypath = pytree_checkpoint_handler._keypath_from_metadata(metadata[_KEY_METADATA_KEY])
      value_meta = metadata[_VALUE_METADATA_KEY]
      restore_type, skip_deserialize = (
          value_meta[_VALUE_TYPE],
          value_meta[_SKIP_DESERIALIZE],
      )
      if type_handlers.is_empty_typestr(restore_type) and not keep_empty_nodes:
        # Return node as the empty value itself rather than as
        # _InternalValueMetadata.
        value_meta = type_handlers.get_empty_value_from_typestr(restore_type)
      else:
        value_meta = pytree_checkpoint_handler._InternalValueMetadata(
            restore_type=restore_type,
            skip_deserialize=skip_deserialize,
        )
      flat_tree_metadata.append((keypath, value_meta))

    return (
        utils.from_flattened_with_keypath(flat_tree_metadata),
        use_zarr3_metadata,
    )

  def _read_aggregate_file(self, directory: epath.Path) -> PyTree:
    """Restores the aggregate file representing PyTree structure."""
    checkpoint_path = directory / self._aggregate_filename
    if checkpoint_path.exists():
      return self._aggregate_handler.deserialize(checkpoint_path)
    else:
      return utils.pytree_structure(directory)

  def _get_internal_metadata(
      self, directory: epath.Path
  ) -> Tuple[PyTree, Optional[bool]]:

    aggregate_tree = self._read_aggregate_file(directory)
    flat_aggregate = utils.to_flat_dict(aggregate_tree, keep_empty_nodes=True)
    try:
      metadata_tree, use_zarr3 = self._read_metadata_file(
          directory, keep_empty_nodes=True
      )
      flat_metadata = utils.to_flat_dict(metadata_tree, keep_empty_nodes=True)
    except FileNotFoundError:
      metadata_tree = None
      flat_metadata = None
      use_zarr3 = None
    if flat_metadata is None:
      flat_metadata = jax.tree_util.tree_map(
          lambda _: None, flat_aggregate, is_leaf=utils.is_empty_or_leaf
      )

    def _get_internal_value_metadata(value_meta, value):
      if value_meta is None:
        if utils.is_supported_empty_aggregation_type(value):
          return value
        restore_type = None
        skip_deserialize = not utils.leaf_is_placeholder(value)
      else:
        if type_handlers.is_empty_typestr(value_meta.restore_type):
          return type_handlers.get_empty_value_from_typestr(
              value_meta.restore_type
          )
        restore_type, skip_deserialize = (
            value_meta.restore_type,
            value_meta.skip_deserialize,
        )
      return pytree_checkpoint_handler._InternalValueMetadata(
          restore_type=restore_type,
          skip_deserialize=skip_deserialize,
          aggregate_value=value,
      )

    result = {}
    for tuple_key in flat_metadata.keys():
      result[tuple_key] = _get_internal_value_metadata(
          flat_metadata[tuple_key], flat_aggregate[tuple_key]
      )
    target = metadata_tree if metadata_tree is not None else aggregate_tree
    return utils.from_flat_dict(result, target=target), use_zarr3

  async def _maybe_deserialize(
      self, structure: PyTree, param_infos: PyTree, restore_args: PyTree
  ) -> PyTree:
    """Deserializes values or gets them from the aggregate file."""

    # Handle parameters from aggregate file.
    def _process_aggregated_value(info, meta, args):
      value = meta.aggregate_value
      if info.skip_deserialize:
        value = pytree_checkpoint_handler._try_array_cast(value, args.dtype)
        value = pytree_checkpoint_handler._maybe_shard_array(value, args)
      return value

    flat_aggregate = utils.to_flat_dict(
        jax.tree_util.tree_map(
            _process_aggregated_value, param_infos, structure, restore_args
        ),
    )

    batch_requests = pytree_checkpoint_handler._batched_serialization_requests(
        structure, param_infos, restore_args
    )
    deserialized_batches = []
    deserialized_batches_ops = []
    for request in batch_requests:
      deserialized_batches_ops.append(
          request.handler.deserialize(request.infos, request.args)
      )
    deserialized_batches += await asyncio.gather(*deserialized_batches_ops)

    flat_restored = {}
    for request, deserialized in zip(batch_requests, deserialized_batches):
      for key, value in zip(request.keys, deserialized):
        flat_restored[key] = value
    # Add in any values which were not deserialized, coming from aggregate file.
    for key in flat_aggregate.keys():
      if key not in flat_restored:
        flat_restored[key] = flat_aggregate[key]
    return utils.from_flat_dict(flat_restored, target=structure)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      restore_args: Optional[PyTree] = None,
      transforms: Optional[PyTree] = None,
      transforms_default_to_original: bool = True,
      legacy_transform_fn: Optional[LegacyTransformFn] = None,
      args: Optional['SingleReplicaRestoreArgs'] = None,
  ) -> PyTree:
    if args is None:
      args = SingleReplicaRestoreArgs(
          item,
          restore_args,
          transforms,
          transforms_default_to_original,
          legacy_transform_fn,
      )
    item = args.item
    restore_args = args.restore_args
    transforms = args.transforms
    transforms_default_to_original = args.transforms_default_to_original
    legacy_transform_fn = args.legacy_transform_fn

    byte_limiter = pytree_checkpoint_handler.get_byte_limiter(self._concurrent_gb)
    structure, use_zarr3_metadata = self._get_internal_metadata(directory)
    # `checkpoint_restore_args` has a structure relative to the checkpoint,
    # while `restore_args` remains structured relative to the output.
    param_infos, checkpoint_restore_args = pytree_checkpoint_handler._get_restore_parameters(
        directory,
        item,
        structure,
        transforms,
        restore_args,
        byte_limiter=byte_limiter,
        transforms_default_to_original=transforms_default_to_original,
        use_zarr3=use_zarr3_metadata
        if use_zarr3_metadata is not None
        else False,
    )

    def _maybe_set_default_restore_types(
        meta: pytree_checkpoint_handler._InternalValueMetadata,
        arg: type_handlers.RestoreArgs
    ):
      if not meta.skip_deserialize and meta.restore_type is None:
        return dataclasses.replace(
            meta, restore_type=type_handlers.default_restore_type(arg)
        )
      return meta

    structure = jax.tree_util.tree_map(
        _maybe_set_default_restore_types, structure, checkpoint_restore_args
    )

    restored_item = asyncio.run(
        self._maybe_deserialize(structure, param_infos, checkpoint_restore_args)
    )

    if not legacy_transform_fn:
      restored_item = pytree_checkpoint_handler._transform_checkpoint(
          item,
          restored_item,
          restore_args,
          transforms,
          transforms_default_to_original,
      )
    utils.sync_global_devices('SingleReplicaArrayHandler:restore')
    return restored_item


@checkpoint_args.register_with_handler(SingleReplicaArrayHandler, for_save=True)
@dataclasses.dataclass
class SingleReplicaSaveArgs(checkpoint_args.CheckpointArgs):
  """Parameters for saving a PyTree.

  Attributes:
    item (required): a PyTree to be saved.
    save_args: a PyTree with the same structure of `item`, which consists of
      `ocp.SaveArgs` objects as values. `None` can be used for values where no
      `SaveArgs` are specified.
  """

  item: PyTree
  save_args: Optional[PyTree] = None


@checkpoint_args.register_with_handler(SingleReplicaArrayHandler, for_restore=True)
@dataclasses.dataclass
class SingleReplicaRestoreArgs(checkpoint_args.CheckpointArgs):
  item: Optional[PyTree] = None
  restore_args: Optional[PyTree] = None
  transforms: Optional[PyTree] = None
  transforms_default_to_original: bool = True
  legacy_transform_fn: Optional[LegacyTransformFn] = None



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


def check_trees_equal(tree1, tree2):
  def check_same(key, v1, v2):
     assert jax.numpy.allclose(
            v1, v2, rtol=1e-06, atol=1e-06
        ), 'Error!!! Restored values are not close enough'
  jax.tree_util.tree_map_with_path(check_same, tree1, tree2)
  print('Hooray, values are close enough!')


def get_replica_pids(rep_id, mesh):
  print(f' I am host {jax.process_index()} w/ local devices ',
        [d.id for d in jax.local_devices()])
  replica_devices = np.take(mesh.devices, rep_id, axis=0).flatten()
  pids = set([d.process_index for d in replica_devices])
  ids = set([d.id for d in replica_devices])
  return ids, pids


def is_sharding_valid(single_replica_ids, single_replica_pids):
  if jax.process_index() in single_replica_pids:
    loc_devieces_in_replica = single_replica_ids.intersection(set([d.id for d in jax.local_devices()]))
    print(f' host ID is {jax.process_index()}, '
          f'num of devices in replica 0 is {len(loc_devieces_in_replica)}')
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
  handler.save(path, args=SingleReplicaSaveArgs(train_state, save_args))

  print('Checkpointing done, now restoring!')
  empty_state = jax.tree_util.tree_map(
          lambda x: x, train_state, is_leaf=is_leaf
      )


  print('Restoring with single replica', jax.process_index())

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
  restored = handler.restore(
        path,
        args=SingleReplicaRestoreArgs(restore_args=restore_args),
    )
  
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
