"""PyTreeCheckpointHandler class.

Implementation of `CheckpointHandler` interface dealing with JAX PyTrees. Much
of the underlying reading/writing logic for individual leaf types can be
customized, and is delegated to the `TypeHandler` class.
"""

import asyncio
import collections
import dataclasses
import enum
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from absl import logging
from etils import epath
import jax
from jax.experimental.array_serialization import serialization
import numpy as np
from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import async_checkpoint_handler
from orbax.checkpoint import future
from orbax.checkpoint import json_checkpoint_handler
from orbax.checkpoint import transform_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
#from orbax.checkpoint import value_metadata


PyTree = Any
TupleKey = Tuple[str, ...]
RestoreArgs = type_handlers.RestoreArgs
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs
SaveArgs = type_handlers.SaveArgs
ParamInfo = type_handlers.ParamInfo
TypeHandler = type_handlers.TypeHandler
AggregateHandler = aggregate_handlers.AggregateHandler
MsgpackHandler = aggregate_handlers.MsgpackHandler
LegacyTransformFn = Callable[[PyTree, PyTree, PyTree], Tuple[PyTree, PyTree]]
Transform = transform_utils.Transform
RestoreTransform = transform_utils.RestoreTransform
JsonCheckpointHandler = json_checkpoint_handler.JsonCheckpointHandler
# TODO(b/298487158) Clean up protected access.
LimitInFlightBytes = serialization._LimitInFlightBytes  # pylint: disable=protected-access


_METADATA_FILE = '_METADATA'
_CHECKPOINT_FILE = 'checkpoint'


### Metadata utils
_KEY_NAME = 'key'
_KEY_TYPE = 'key_type'
_VALUE_TYPE = 'value_type'
_SKIP_DESERIALIZE = 'skip_deserialize'

_TREE_METADATA_KEY = 'tree_metadata'
_KEY_METADATA_KEY = 'key_metadata'
_VALUE_METADATA_KEY = 'value_metadata'


class _KeyType(enum.Enum):
  """Enum representing PyTree key type."""

  SEQUENCE = 1
  DICT = 2

  def to_json(self) -> int:
    return self.value

  @classmethod
  def from_json(cls, value: int) -> '_KeyType':
    return cls(value)


def _get_key_metadata_type(key: Any) -> _KeyType:
  """Translates the JAX key class into a proto enum."""
  if utils.is_sequence_key(key):
    return _KeyType.SEQUENCE
  elif utils.is_dict_key(key):
    return _KeyType.DICT
  else:
    raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')


def _keypath_from_key_type(key_name: str, key_type: _KeyType) -> Any:
  """Converts from Key in TreeMetadata to JAX keypath class."""
  if key_type == _KeyType.SEQUENCE:
    return jax.tree_util.SequenceKey(int(key_name))
  elif key_type == _KeyType.DICT:
    return jax.tree_util.DictKey(key_name)
  else:
    raise ValueError(f'Unsupported KeyEntry: {key_type}')


def _get_keypath_metadata(keypath: Any) -> Tuple[Dict[str, Any]]:
  """Gets JSON metadata for a JAX keypath."""
  keypath_serialized = []
  for k in keypath:
    keypath_serialized.append({
        _KEY_NAME: str(utils.get_key_name(k)),
        _KEY_TYPE: _get_key_metadata_type(k).to_json(),
    })
  return tuple(keypath_serialized)


def _keypath_from_metadata(keypath_serialized: Tuple[Dict[str, Any]]) -> Any:
  """Creates a JAX keypath from JSON metadata."""
  keypath = []
  for k in keypath_serialized:
    keypath.append(
        _keypath_from_key_type(k[_KEY_NAME], _KeyType.from_json(k[_KEY_TYPE]))
    )
  return tuple(keypath)


def _get_value_metadata(value: Any, save_arg: SaveArgs) -> Dict[str, Any]:
  """Gets JSON metadata for a given value."""
  if utils.is_supported_empty_aggregation_type(value):
    typestr = type_handlers.get_empty_value_typestr(value)
    skip_deserialize = True
  else:
    try:
      handler = type_handlers.get_type_handler(type(value))
      typestr = handler.typestr()
      skip_deserialize = save_arg.aggregate
    except ValueError:
      # Not an error because users' training states often have a bunch of
      # random unserializable objects in them (empty states, optimizer
      # objects, etc.). An error occurring due to a missing TypeHandler
      # will be surfaced elsewhere.
      typestr = type_handlers.RESTORE_TYPE_NONE
      skip_deserialize = True
  return {
      _VALUE_TYPE: typestr,
      _SKIP_DESERIALIZE: skip_deserialize,
  }


### End metadata utils


def get_byte_limiter(concurrent_gb: int):
  async def _create_byte_limiter():
    # Wrap creation in async function to avoid issues on python<=3.9.
    concurrent_bytes = concurrent_gb * 10**9
    # Construction must take place here so that it is within the same async
    # method, to prevent errors resulting from different event loops, and
    # cannot be created below this level because there must be a single object
    # for the entire restore call.
    return LimitInFlightBytes(concurrent_bytes)  # pylint: disable=protected-access

  return asyncio.run(_create_byte_limiter())


async def _create_param_save_dir(param_info: ParamInfo, args: SaveArgs):
  # Directory will be unused.
  path = param_info.path
  if path is None or args.aggregate:
    return
  if jax.process_index() == 0:
    # TODO(b/273803615): Note that keys with slashes ('/', generated by Haiku,
    # for example) will result in the creation of nested sub-directories, rather
    # than flat parameter directories like for a standard neste PyTree. This
    # discrepancy, while potentially problematic, will not be addressed since we
    # anticipate moving fully to OCDBT within a quarter or two.
    await utils.async_makedirs(path, parents=True)


def _maybe_set_default_save_args(value, args):
  # If already set, return.
  if isinstance(args, SaveArgs):
    return args
  aggregate = not type_handlers.has_type_handler(type(value))
  return SaveArgs(aggregate=aggregate)


def _maybe_set_default_restore_args(args):
  if isinstance(args, RestoreArgs):
    return args
  return RestoreArgs(restore_type=None)


def _try_array_cast(arr, dtype):
  if dtype is not None:
    if utils.is_scalar(arr):
      arr = np.asarray(arr).astype(dtype).item()
    else:
      if hasattr(arr, 'astype'):
        arr = arr.astype(dtype)
  return arr


def _maybe_shard_array(value, args):
  if hasattr(value, 'reshape') and isinstance(args, ArrayRestoreArgs):
    value = value.reshape(args.global_shape)
    sharding = args.sharding or jax.sharding.NamedSharding(
        args.mesh, args.mesh_axes
    )
    value = jax.make_array_from_callback(
        value.shape, sharding, lambda idx: value[idx]
    )
  return value


def _get_param_names(item: PyTree) -> PyTree:
  """Gets parameter names for PyTree elements."""

  def _param_name_from_keypath(keypath: Tuple[Any, ...]) -> str:
    return '.'.join([str(utils.get_key_name(k)) for k in keypath])

  return jax.tree_util.tree_map_with_path(
      lambda kp, _: _param_name_from_keypath(kp),
      item,
      is_leaf=utils.is_empty_or_leaf,
  )


def _keystr(key: Tuple[Any, ...]) -> str:
  return '/'.join(key)


def _find_matching_input_args(
    input_key: TupleKey,
    flat_item: Dict[TupleKey, Any],
    flat_transforms: Dict[TupleKey, Transform],
    flat_restore_args: Dict[TupleKey, RestoreArgs],
) -> Optional[RestoreArgs]:
  """Given an input_key, tries to find matching RestoreArgs for the input.

  Args:
    input_key: A key in the input tree.
    flat_item: The flattened, user-provided item.
    flat_transforms: Flattened transformations dict.
    flat_restore_args: Flattened tree of RestoreArgs, relative to item.

  Returns:
    RestoreArgs that match the given input_key, according to the
    transformations, or None if no match is found.
  """
  for transform_key, transform in flat_transforms.items():
    if transform.multi_value_fn is not None:
      if not isinstance(transform, RestoreTransform):
        raise ValueError(
            'Must use RestoreTransform in order to use multi_value_fn'
            ' during restore.'
        )
      if transform.multi_value_fn_input_args is None:
        raise ValueError(
            '`multi_value_fn` was specified, but'
            ' `multi_value_fn_input_args` were not. The latter must be'
            ' specified to identify inputs for the function.'
        )
      for (
          input_key_regex,
          input_args,
      ) in transform.multi_value_fn_input_args.items():
        if re.fullmatch(input_key_regex, _keystr(input_key)):
          return input_args
    elif not transform.use_fallback:
      # The following is done to reverse-engineer the regex for the key in
      # the original tree.
      for output_key in flat_item:
        match = re.fullmatch(_keystr(transform_key), _keystr(output_key))
        if match:
          if transform.original_key is None:
            # If transform.original_key is not specified, this transform
            # does not rename the original key. We can reuse the key from
            # the item.
            input_key_pattern = _keystr(output_key)
          else:
            input_key_pattern = match.expand(transform.original_key)
          if input_key_pattern == _keystr(input_key):
            return flat_restore_args[output_key]
  return None


def _has_use_fallback_transform(
    input_key: TupleKey, flat_transforms: Dict[TupleKey, Transform]
) -> bool:
  result = False
  for transform_key, transform in flat_transforms.items():
    match = re.fullmatch(_keystr(transform_key), _keystr(input_key))
    if match and transform.use_fallback:
      result = True
  return result


@dataclasses.dataclass
class _InternalValueMetadata:
  restore_type: Optional[str]
  skip_deserialize: bool = False
  aggregate_value: Optional[Any] = None


def _get_restore_parameters(
    directory: epath.Path,
    item: Optional[PyTree],
    structure: PyTree,
    transforms: Optional[PyTree],
    restore_args: Optional[PyTree],
    byte_limiter: Optional[LimitInFlightBytes] = None,
    transforms_default_to_original: bool = True,
) -> Tuple[PyTree, PyTree]:
  """Construct parameters needed for restoration.

  If transforms are not provided, the method is pretty simple: param_infos are
  constructed from the structure of the original checkpoint, and restore_args
  are serialized to a tree structure compatible with param_infos and structure.

  If transforms are provided, things become more complicated because we must
  determine exactly which parameters the user desires to restore, and construct
  param_infos and restore_args for these, while discarding unneeded parameters.
  In essence, the process can be thought of as reversing the transformations.
  This happens differently for different types of transforms.
  1. Renamed key: Identify the original key name (in the checkpoint) and carry
    over the provided restore args for the parameter.
  2. multi_value_fn: Users are required to specify multi_value_fn_input_args.
    Any keys named here must be loaded, and their restore args are also given
    here.
  3. Unspecified key: A key which is unspecified in the transforms but present
    in the `item` is a key that is carried over from the checkpoint unchanged.
  4. Fallback key: This is a key that is present in the `item` but not in the
    original checkpoint. It does not need to be restored.
  5. Keys present in the original checkpoint but not in the `item`/`transforms`
    are implicitly ignored, and not restored.

  Args:
    directory: Checkpoint directory.
    item: Optional reference item.
    structure: The structure of the original checkpoint.
    transforms: User-provided transformations. If None, they were not provided.
      Has the structure of the desired output tree.
    restore_args: User-provided restoration arguments. If None, they were not
      provided. Otherwise, the tree has the same structure as the desired output
      tree.
    byte_limiter: A _LimitInFlightBytes object.
    transforms_default_to_original: See transform_utils.apply_transformations.

  Returns:
    Tuple of param_infos, and restore_args.
  """
  flat_structure = utils.to_flat_dict(structure, keep_empty_nodes=True)
  if restore_args is None:
    restore_args = jax.tree_util.tree_map(lambda x: RestoreArgs(), structure)
  flat_restore_args = utils.to_flat_dict(restore_args, keep_empty_nodes=True)
  flat_param_infos = {}
  flat_input_restore_args = {}
  is_ocdbt_checkpoint = type_handlers.is_ocdbt_checkpoint(directory)

  def _get_param_info(
      nested_name: Tuple[str, ...],
      meta: _InternalValueMetadata,
  ) -> Union[ParamInfo, Any]:
    if utils.is_supported_empty_aggregation_type(meta):
      # Empty node, ParamInfo should not be returned.
      return meta
    name = '.'.join(nested_name)
    return ParamInfo(
        name=name,
        path=directory / name,
        skip_deserialize=meta.skip_deserialize,
        is_ocdbt_checkpoint=is_ocdbt_checkpoint,
        byte_limiter=byte_limiter,
    )

  if transforms is None:
    for key, meta in flat_structure.items():
      flat_param_infos[key] = _get_param_info(key, meta)
    restore_args = utils.serialize_tree(restore_args, keep_empty_nodes=True)
  else:
    if item is None:
      raise ValueError(
          'If providing `transforms`, must provide `item` matching structure'
          ' of expected result.'
      )
    flat_item = utils.to_flat_dict(item, keep_empty_nodes=True)
    flat_transforms = utils.to_flat_dict(transforms)

    for input_key, meta in flat_structure.items():
      maybe_input_args = _find_matching_input_args(
          input_key, flat_item, flat_transforms, flat_restore_args
      )
      if maybe_input_args:
        flat_param_infos[input_key] = _get_param_info(input_key, meta)
        flat_input_restore_args[input_key] = maybe_input_args
      elif input_key in flat_item and input_key in flat_structure:
        # Key is present in both input and output.
        if _has_use_fallback_transform(input_key, flat_transforms):
          # Indicates that a `use_fallback` transformation was specified.
          if transforms_default_to_original:
            # Specified `use_fallback`, but key was also present in the
            # checkpoint. This means we should skip loading, since it will be
            # overridden with a new value.
            flat_param_infos[input_key] = ParamInfo(skip_deserialize=True)
            flat_input_restore_args[input_key] = RestoreArgs()
          else:
            # Specified `use_fallback`, but `transforms_default_to_original`
            # is False. This means we draw the value from the user-provided
            # `item`.
            flat_param_infos[input_key] = _get_param_info(input_key, meta)
            flat_input_restore_args[input_key] = flat_restore_args[input_key]
        else:
          # Transform not specified.
          if transforms_default_to_original:
            # Key/value is carried over from the original unchanged.
            flat_param_infos[input_key] = _get_param_info(input_key, meta)
            flat_input_restore_args[input_key] = flat_restore_args[input_key]
          else:
            # Take the value from the user-provided `item`, ignoring any value
            # in the checkpoint.
            flat_param_infos[input_key] = ParamInfo(skip_deserialize=True)
            flat_input_restore_args[input_key] = RestoreArgs()
      else:
        # No match, restoration not required since it will be dropped from the
        # output.
        flat_param_infos[input_key] = ParamInfo(skip_deserialize=True)
        flat_input_restore_args[input_key] = RestoreArgs()

    restore_args = utils.from_flat_dict(
        flat_input_restore_args, target=structure
    )

  return (
      utils.from_flat_dict(flat_param_infos, target=structure),
      restore_args,
  )


def _get_tree_for_aggregation(param_infos, save_args, item):
  """Get tree for aggregated checkpoint."""

  # TODO(b/283164080): These type checks result in logic from the lower layer
  # (TypeHandler/AggregateHandler) leaking into the upper layer
  # (CheckpointHandler). Ideally, AggregateHandler could define its own
  # supported values and error conditions.
  def _get_leaf_for_aggregation(param_info, arg, value):
    if arg.aggregate:  # Param was aggregated, return value after cast.
      if isinstance(value, jax.Array) and not value.is_fully_replicated:
        raise ValueError(
            'jax.Array must be fully replicated to be saved in aggregate file.'
        )
      if not utils.is_supported_aggregation_type(value):
        # Not an error because users' training states often have a bunch of
        # random unserializable objects in them (empty states, optimizer
        # objects, etc.).
        value = None
      return _try_array_cast(value, arg.dtype)
    else:  # Placeholder string for non-aggregated value.
      return utils.leaf_placeholder(param_info.name)

  return jax.tree_util.tree_map(
      _get_leaf_for_aggregation, param_infos, save_args, item
  )


@dataclasses.dataclass
class _BatchRequest:
  """Represents a a request for batched serialization or deserialization."""

  handler: TypeHandler
  values: List[Any]
  infos: List[ParamInfo]
  args: List[Union[SaveArgs, RestoreArgs]]


def _batched_serialization_requests(
    tree: PyTree, param_infos: PyTree, args: PyTree
) -> List[_BatchRequest]:
  """Gets a list of batched serialization or deserialization requests."""
  grouped = {}

  def _group_value(
      info: ParamInfo,
      value: Union[Any, _InternalValueMetadata],
      arg: RestoreArgs,
  ):
    nonlocal grouped
    # Exclude from serialize/deserialize with TypeHandler if aggregated.
    if info.skip_deserialize:
      return
    if isinstance(arg, RestoreArgs):
      assert isinstance(value, _InternalValueMetadata)
      restore_type = value.restore_type
      if arg.restore_type is not None:
        # Give user the chance to override restore_type if they want.
        restore_type = arg.restore_type
      handler = type_handlers.get_type_handler(restore_type)
    else:
      handler = type_handlers.get_type_handler(type(value))
    if handler not in grouped:
      grouped[handler] = _BatchRequest(handler, [], [], [])
    request = grouped[handler]
    grouped[handler] = dataclasses.replace(
        request,
        values=request.values + [value],
        infos=request.infos + [info],
        args=request.args + [arg],
    )

  jax.tree_util.tree_map(
      _group_value,
      param_infos,
      tree,
      args,
  )
  return list(grouped.values())


def _multi_value_fns_with_args(
    transforms: PyTree, restore_args: PyTree
) -> PyTree:
  """Constructs a wrapper for multi_value_fn including RestoreArgs."""
  flat_restore_args = utils.to_flat_dict(restore_args, sep='/')

  def _maybe_wrap_transform(transform: Transform):
    def _multi_value_fn_with_args(transform_key: str, tree: PyTree) -> Any:
      nonlocal transform
      transform = typing.cast(RestoreTransform, transform)
      return transform.multi_value_fn(
          transform_key, tree, flat_restore_args[transform_key]
      )

    if transform.multi_value_fn is not None:
      return Transform(multi_value_fn=_multi_value_fn_with_args)
    else:
      return transform

  return jax.tree_util.tree_map(_maybe_wrap_transform, transforms)


def _transform_checkpoint(
    item: PyTree,
    restored: PyTree,
    restore_args: Optional[PyTree],
    transforms: Optional[PyTree],
    transforms_default_to_original: bool,
) -> PyTree:
  """Optionally transforms the restored PyTree to the structure of `item`.

  Args:
    item: a PyTree representing the result structure ("new tree structure").
    restored: a PyTree representing the original tree structure.
    restore_args: tree of RestoreArgs, with the same structure as `item`.
    transforms: provides instructions on how to transform the input trees. See
      transform_utils.
    transforms_default_to_original: See transform_utils.

  Returns:
    A transformed PyTree.
  """
  if item is None:
    if transforms is not None:
      msg = (
          'If providing `transforms`, must provide `item` matching structure'
          ' of expected result.'
      )
      raise ValueError(msg)
    item = restored
  else:
    if transforms is None:
      item = utils.deserialize_tree(restored, item)
    else:
      if restore_args is None:
        raise ValueError(
            'If providing `transforms`, must provide `restore_args` matching'
            ' structure of expected result.'
        )
      transforms = _multi_value_fns_with_args(transforms, restore_args)
      item = transform_utils.apply_transformations(
          restored, transforms, item, transforms_default_to_original
      )
  return item


class MyTypeHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """A CheckpointHandler implementation for any PyTree structure.

  See JAX documentation for more information on what consistutes a "PyTree".
  This handler is capable of saving and restoring any leaf object for which a
  `TypeHandler` (see documentation) is registered. By default, `TypeHandler`s
  for standard types like `np.ndarray`, `jax.Array`, Python scalars, and others
  are registered.

  As with all `CheckpointHandler` subclasses, `PyTreeCheckpointHandler` should
  only be used in conjunction with a `Checkpointer` (or subclass). By itself,
  the `CheckpointHandler` is non-atomic.

  Example::

    ckptr = Checkpointer(PyTreeCheckpointHandler())
  """

  def __init__(
      self,
      aggregate_filename: Optional[str] = None,
      concurrent_gb: int = 96,
      use_ocdbt: bool = True,
      restore_with_serialized_types: bool = True,
      write_tree_metadata: bool = False,
  ):
    """Creates PyTreeCheckpointHandler.

    Args:
      aggregate_filename: name that the aggregated checkpoint should be saved
        as.
      concurrent_gb: max concurrent GB that are allowed to be read. Can help to
        reduce the possibility of OOM's when large checkpoints are restored.
      use_ocdbt: enables Tensorstore OCDBT driver. This option allows using a
        different checkpoint format which is faster to read and write, as well
        as more space efficient. Currently, it is not yet enabled as the default
        option.
      restore_with_serialized_types: If True, the values with unspecified
        restore types will be restored using the typing information in the
        checkpoint. Otherwise, arrays will be restored as either np.ndarray or
        jax.Array, and will ignore any typing information present in the
        checkpoint. Note: this option is not applicable to most users.
      write_tree_metadata: Writes tree metadata in JSON format. The tree
        metadata is used to enable a checkpoint which is fully self-describing.
    """
    self._aggregate_handler = MsgpackHandler()
    if aggregate_filename is None:
      aggregate_filename = _CHECKPOINT_FILE
    self._aggregate_filename = aggregate_filename
    self._concurrent_gb = concurrent_gb
    self._use_ocdbt = use_ocdbt
    self._restore_with_serialized_types = restore_with_serialized_types
    self._write_tree_metadata = write_tree_metadata
    self._metadata_handler = JsonCheckpointHandler(_METADATA_FILE)

    # BEGIN GOOGLE-INTERNAL
    if utils.is_pathways_backend():
      if self._use_ocdbt:
        raise ValueError('OCDBT not currently supported for Pathways backend.')
      if (
          type_handlers.get_type_handler(jax.Array).__class__.__name__
          != 'PathwaysArrayHandler'
      ):
        logging.warning(
            'Pathways backend detected, but `PathwaysArrayHandler` was not'
            ' registered. This will likely result in significantly slower save'
            ' and restore performance. Please first call'
            ' `pathways_type_handler.smaybe_register_pathways_handlers()`.'
        )
    # END GOOGLE-INTERNAL

    if self._use_ocdbt:
      jax.monitoring.record_event(
          '/jax/orbax/pytree_checkpoint_handler/init/ocdbt'
      )
      type_handlers.start_coordinator_server_and_create_context()

  def _get_param_names(self, item: PyTree) -> PyTree:
    """Gets parameter names for PyTree elements."""
    return _get_param_names(item)

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
      return ParamInfo(
          name=name, path=(directory / name), skip_deserialize=args.aggregate
      )

    return (
        jax.tree_util.tree_map(_param_info, names, save_args),
        all_params_aggregated,
    )

  async def async_save(
      self,
      directory: epath.Path,
      item: PyTree,
      save_args: Optional[PyTree] = None,
  ) -> Optional[List[future.Future]]:
    """Saves a PyTree to a given directory.

    This operation is compatible with a multi-host, multi-device setting. Tree
    leaf values must be supported by type_handlers. Standard supported types
    include Python scalars, `np.ndarray`, `jax.Array`, and strings.

    After saving, all files will be located in "directory/". The exact files
    that are saved depend on the specific combination of options, including
    `use_ocdbt` and `write_tree_metadata`. If `write_tree_metadata` is
    enabled, a JSON metadata file will be present to store the tree structure.
    In addition, a msgpack file may be present, allowing users to store
    aggregated values (see below).

    Example usage::

      ckptr = Checkpointer(PyTreeCheckpointHandler())
      item = {
          'layer0': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
          'layer1': {
              'w': np.ndarray(...),
              'b': np.ndarray(...),
          },
      }
      # Note: save_args may be None if no customization is desired for saved
      # parameters.
      # In this case, we "aggregate" small parameters into a single file to
      # allow for greater file read/write efficiency (and potentially less)
      # wasted space). With OCDBT format active, this parameter is obsolete.
      save_args =
        jax.tree_util.tree_map(
            lambda x: SaveArgs(aggregate=x.size < some_size), item)
      # Eventually calls through to `async_save`.
      ckptr.save(path, item, save_args)

    Args:
      directory: save location directory.
      item: a PyTree to be saved.
      save_args: a PyTree matching `item` which consists of SaveArgs objects as
        values.

    Returns:
      A Future that will commit the data to `directory` when awaited. Copying
      the data from its source will be awaited in this function.
    """
    # Because of empty states, the user-provided args may not contain
    # all necessary arguments. These should be filled in with default args.
    save_args = jax.tree_util.tree_map(
        _maybe_set_default_save_args,
        item,
        item if save_args is None else save_args,
        is_leaf=utils.is_empty_or_leaf,
    )
    param_infos, all_params_aggregated = self._get_param_infos(
        item, directory, save_args
    )
    if not self._use_ocdbt and not all_params_aggregated:
      # Create directories in parallel.
      await asyncio.gather(
          *jax.tree_util.tree_flatten(
              jax.tree_util.tree_map(
                  _create_param_save_dir, param_infos, save_args
              )
          )[0]
      )
      utils.sync_global_devices(
          'PyTreeCheckpointHandler:create_param_save_dirs'
      )

    if all_params_aggregated:
      commit_futures = []
    else:
      serialize_ops = []
      batch_requests = _batched_serialization_requests(
          item, param_infos, save_args
      )
      for request in batch_requests:
        serialize_ops += [
            request.handler.serialize(
                request.values, request.infos, request.args
            )
        ]
      # Await copy futures. Returns list of lists.
      commit_futures = await asyncio.gather(*serialize_ops)
      commit_futures, _ = jax.tree_util.tree_flatten(commit_futures)

    # TODO(b/285888834): Allow this to be asynchronous.
    self._write_metadata_file(directory, item, save_args)
    aggregate_commit_future = await self._write_aggregate_file(
        directory, item, param_infos, save_args
    )
    return commit_futures + [aggregate_commit_future]

  def save(self, directory: epath.Path, item: Any, *args, **kwargs):
    """Saves the provided item.

    Blocks until both copy and commit complete.

    See async_save.

    Args:
      directory: the directory to save to.
      item: the item to be saved.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """

    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for f in commit_futures:
          f.result()  # Block on result.

    asyncio.run(async_save(directory, item, *args, **kwargs))
    utils.sync_global_devices('PyTreeCheckpointHandler:save')

  async def _maybe_deserialize(
      self, structure: PyTree, param_infos: PyTree, restore_args: PyTree
  ) -> PyTree:
    """Deserializes values or gets them from the aggregate file."""

    # Handle parameters from aggregate file.
    def _process_aggregated_value(info, meta, args):
      value = meta.aggregate_value
      if info.skip_deserialize:
        value = _try_array_cast(value, args.dtype)
        value = _maybe_shard_array(value, args)
      return value

    flat_aggregate = utils.to_flat_dict(
        jax.tree_util.tree_map(
            _process_aggregated_value, param_infos, structure, restore_args
        ),
        sep='.',
    )

    batch_requests = _batched_serialization_requests(
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
      for info, value in zip(request.infos, deserialized):
        assert not info.skip_deserialize
        flat_restored[info.name] = value
    # Add in any values which were not deserialized, coming from aggregate file.
    for key in flat_aggregate.keys():
      if key not in flat_restored:
        flat_restored[key] = flat_aggregate[key]
    return utils.from_flat_dict(flat_restored, target=structure, sep='.')

  def restore(
      self,
      directory: epath.Path,
      item: Optional[PyTree] = None,
      restore_args: Optional[PyTree] = None,
      transforms: Optional[PyTree] = None,
      transforms_default_to_original: bool = True,
      legacy_transform_fn: Optional[LegacyTransformFn] = None,
  ) -> PyTree:
    """Restores a PyTree from the checkpoint directory at the given path.

    In the most basic case, only `directory` is required. The tree will be
    restored exactly as saved, and all leaves will be restored as the correct
    types (assuming the tree metadata is present).

    However, `restore_args` is often required as well. This PyTree gives a
    `RestoreArgs` object (or subclass) for every leaf in the tree. Many types,
    such as string or `np.ndarray` do not require any special options for
    restoration. When restoring an individual leaf as `jax.Array`, however,
    some properties may be required.

    One example is `sharding`, which defines how a `jax.Array` in the restored
    tree should be partitioned. `mesh` and `mesh_axes` can also be used to
    specify `sharding`, but `sharding` is the preferred way of specifying this
    partition since `mesh` and `mesh_axes` only constructs
    `jax.sharding.NamedSharding`. For more information, see `ArrayTypeHandler`
    documentation and JAX sharding documentation.

    Example::

      ckptr = Checkpointer(PyTreeCheckpointHandler)
      restore_args = {
          'layer0': {
              'w': RestoreArgs(),
              'b': RestoreArgs(),
          },
          'layer1': {
              'w': ArrayRestoreArgs(
                  # Restores as jax.Array, regardless of how it was saved.
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  # Warning: may truncate or pad!
                  global_shape=(x, y),
                ),
              'b': ArrayRestoreArgs(
                  restore_type=jax.Array,
                  sharding=jax.sharding.Sharding(...),
                  global_shape=(x, y),
                ),
          },
      }
      ckptr.restore(path, restore_args=restore_args)

    Providing `item` is typically only necessary when restoring a custom PyTree
    class (or when using transformations). In this case, the restored object
    will take on the same structure as `item`.

    Example::

      @flax.struct.dataclass
      class TrainState:
        layer0: dict[str, jax.Array]
        layer1: dict[str, jax.Array]

      ckptr = Checkpointer(PyTreeCheckpointHandler)
      train_state = TrainState(
          layer0={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
          layer1={
              'w': jax.Array(...),  # zeros
              'b': jax.Array(...),  # zeros
          },
      )
      restore_args = jax.tree_util.tree_map(_make_restore_args, train_state)
      ckptr.restore(path, item=train_state, restore_args=restore_args)
      # restored tree is of type `TrainState`.

    Args:
      directory: saved checkpoint location directory.
      item: provides the tree structure for the restored item. If not provided,
        will infer the structure from the saved checkpoint. Transformations will
        not be run in this case. Necessary particularly in the case where the
        caller needs to restore the tree as a custom object.
      restore_args: optional object containing additional arguments for
        restoration. It should be a PyTree matching the structure of `item`, or
        if `item` is not provided, then it should match the structure of the
        checkpoint. Each value in the tree should be a `RestoreArgs` object (OR
        a subclass of `RestoreArgs`). Importantly, note that when restoring a
        leaf as a certain type, a specific subclass of `RestoreArgs` may be
        required. `RestoreArgs` also provides the option to customize the
        restore type of an individual leaf.
      transforms: a PyTree of transformations that should be applied to the
        saved tree in order to obtain a final structure. The `transforms` tree
        structure should conceptually match that of `item`, but the use of
        regexes and implicit keys means that it does not need to match
        completely. See `transform_utils` for further information.
      transforms_default_to_original: See transform_utils.apply_transformations.
      legacy_transform_fn: WARNING: NOT GENERALLY SUPPORTED. A function which
        accepts the `item` argument, a PyTree checkpoint structure and a PyTree
        of ParamInfos based on the checkpoint. Returns a transformed PyTree
        matching the desired return tree structure, and a matching ParamInfo
        tree.

    Returns:
      A PyTree matching the structure of `item`.

    Raises:
      FileNotFoundError: `directory` does not exist or is missing required files
      ValueError: `transforms` is provided without `item`.
      ValueError: `transforms` contains elements with `multi_value_fn`.
    """
    if not directory.exists():
      raise FileNotFoundError(
          f'Requested directory for restore does not exist at {directory}'
      )
    byte_limiter = get_byte_limiter(self._concurrent_gb)
    structure = self._get_internal_metadata(directory)
    # `checkpoint_restore_args` has a structure relative to the checkpoint,
    # while `restore_args` remains structured relative to the output.
    param_infos, checkpoint_restore_args = _get_restore_parameters(
        directory,
        item,
        structure,
        transforms,
        restore_args,
        byte_limiter=byte_limiter,
        transforms_default_to_original=transforms_default_to_original,
    )

    if legacy_transform_fn is not None and transforms is not None:
      raise ValueError(
          'Cannot provide both `transforms` and `legacy_transform_fn`.'
      )
    if legacy_transform_fn is not None:
      structure, param_infos = legacy_transform_fn(item, structure, param_infos)
      if restore_args is None:
        restore_args = jax.tree_util.tree_map(lambda x: RestoreArgs(), item)
      checkpoint_restore_args = restore_args

    def _maybe_set_default_restore_types(
        meta: _InternalValueMetadata, arg: RestoreArgs
    ):
      if not meta.skip_deserialize and meta.restore_type is None:
        return dataclasses.replace(
            meta, restore_type=type_handlers.default_restore_type(arg)
        )
      return meta

    # If metadata file was missing in the checkpoint, we need to decide
    # restore_type based on RestoreArgs.
    structure = jax.tree_util.tree_map(
        _maybe_set_default_restore_types, structure, checkpoint_restore_args
    )

    restored_item = asyncio.run(
        self._maybe_deserialize(structure, param_infos, checkpoint_restore_args)
    )

    if not legacy_transform_fn:
      restored_item = _transform_checkpoint(
          item,
          restored_item,
          restore_args,
          transforms,
          transforms_default_to_original,
      )
    utils.sync_global_devices('PyTreeCheckpointHandler:restore')
    return restored_item

  async def _write_aggregate_file(
      self,
      directory: epath.Path,
      item: PyTree,
      param_infos: PyTree,
      save_args: PyTree,
  ) -> future.Future:
    ser_item = _get_tree_for_aggregation(param_infos, save_args, item)
    return await self._aggregate_handler.serialize(
        directory / self._aggregate_filename, ser_item
    )

  def _read_aggregate_file(self, directory: epath.Path) -> PyTree:
    """Restores the aggregate file representing PyTree structure."""
    checkpoint_path = directory / self._aggregate_filename
    if checkpoint_path.exists():
      return self._aggregate_handler.deserialize(checkpoint_path)
    elif self._use_ocdbt:
      raise ValueError(
          f'Checkpoint structure file does not exist at {directory}.'
      )
    else:
      return utils.pytree_structure(directory)

  def _write_metadata_file(
      self, directory: epath.Path, item: PyTree, save_args: PyTree
  ):
    """Write PyTree metadata.

    Uses JSON format::

    {
        _TREE_METADATA_KEY: {
          "(top_level_key, lower_level_key)": {
              _KEY_METADATA_KEY: (
                  {_KEY_NAME: "top_level_key", _KEY_TYPE: <_KeyType (int)>},
                  {_KEY_NAME: "lower_level_key", _KEY_TYPE: <_KeyType (int)>},
              )
              _VALUE_METADATA_KEY: {
                  _VALUE_TYPE: "jax.Array",
                  _SKIP_DESERIALIZE: True/False,
              }
          }
          ...
      }
    }

    Args:
      directory: directory
      item: item to save
      save_args: save_args

    Returns:
      None
    """
    if not self._write_tree_metadata:
      return []

    flat_with_keys, _ = jax.tree_util.tree_flatten_with_path(
        item, is_leaf=utils.is_empty_or_leaf
    )
    flat_save_args_with_keys, _ = jax.tree_util.tree_flatten_with_path(
        save_args, is_leaf=utils.is_empty_or_leaf
    )

    flat_metadata_with_keys = {}
    for (keypath, value), (_, save_arg) in zip(
        flat_with_keys, flat_save_args_with_keys
    ):
      tuple_keypath = str(tuple([str(utils.get_key_name(k)) for k in keypath]))
      flat_metadata_with_keys[tuple_keypath] = {
          _KEY_METADATA_KEY: _get_keypath_metadata(keypath),
          _VALUE_METADATA_KEY: _get_value_metadata(value, save_arg),
      }

    metadata = {_TREE_METADATA_KEY: flat_metadata_with_keys}
    self._metadata_handler.save(directory, metadata)

  def _read_metadata_file(
      self, directory: epath.Path, keep_empty_nodes: bool = False
  ) -> PyTree:
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

    tree_metadata = typing.cast(
        Dict[Any, Any], self._metadata_handler.restore(directory)
    )[_TREE_METADATA_KEY]
    flat_tree_metadata = []
    for metadata in tree_metadata.values():
      keypath = _keypath_from_metadata(metadata[_KEY_METADATA_KEY])
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
        value_meta = _InternalValueMetadata(
            restore_type=restore_type,
            skip_deserialize=skip_deserialize,
        )
      flat_tree_metadata.append((keypath, value_meta))

    return utils.from_flattened_with_keypath(flat_tree_metadata)

  def _get_internal_metadata(self, directory: epath.Path) -> PyTree:
    """Gets limited information needed to fully restore the checkpoint.

    This information just consists of the restore type for each leaf, as well
    as the aggregated value (from the msgpack file) if present, and determines
    whether we need to deserialize the parameter using TypeHandler later.

    Args:
      directory: directory

    Returns:
      A PyTree of _InternalValueMetadata with the tree structure of the
      checkpoint.
    """
    aggregate_tree = self._read_aggregate_file(directory)
    flat_aggregate = utils.to_flat_dict(aggregate_tree, keep_empty_nodes=True)
    try:
      metadata_tree = self._read_metadata_file(directory, keep_empty_nodes=True)
      flat_metadata = utils.to_flat_dict(metadata_tree, keep_empty_nodes=True)
    except FileNotFoundError:
      metadata_tree = None
      flat_metadata = None
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
      return _InternalValueMetadata(
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
    return utils.from_flat_dict(result, target=target)

  def _get_user_metadata(self, directory: epath.Path) -> PyTree:
    """Reads metadata file and constructs user-friendly metadata.

    This will involve more file reads than are necessary for internal metadata.
    Typically, we will need to perform extra reads in order to get metadata
    about individual arrays.

    Args:
      directory: directory

    Returns:
      A PyTree of value_metadata.Metadata matching the checkpoint tree
      structure.
    """
    is_ocdbt_checkpoint = type_handlers.is_ocdbt_checkpoint(directory)

    flat_param_infos = {}
    flat_restore_types = {}
    metadata = self._read_metadata_file(directory, keep_empty_nodes=False)
    for keypath, value_meta in utils.to_flat_dict(metadata).items():
      param_name = '.'.join(keypath)
      restore_type, skip_deserialize = (
          value_meta.restore_type,
          value_meta.skip_deserialize,
      )
      flat_param_infos[keypath] = ParamInfo(
          name=param_name,
          path=directory / param_name,
          skip_deserialize=skip_deserialize,
          is_ocdbt_checkpoint=is_ocdbt_checkpoint,
      )
      flat_restore_types[keypath] = restore_type

    flat_metadatas = {}
    batched_param_infos = collections.defaultdict(list)
    batched_keypaths = collections.defaultdict(list)
    for keypath in flat_param_infos:
      param_info = flat_param_infos[keypath]
      restore_type = flat_restore_types[keypath]
      if param_info.skip_deserialize:
        flat_metadatas[keypath] = value_metadata.Metadata()
      else:
        batched_keypaths[restore_type].append(keypath)
        batched_param_infos[restore_type].append(param_info)

    metadata_ops = []
    for restore_type, param_infos in batched_param_infos.items():
      handler = type_handlers.get_type_handler(restore_type)
      metadata_ops.append(handler.metadata(param_infos))

    async def _get_metadata():
      return await asyncio.gather(*metadata_ops)

    batched_metadatas = asyncio.run(_get_metadata())
    for keypath_batch, metadata_batch in zip(
        batched_keypaths.values(), batched_metadatas
    ):
      for keypath, value in zip(keypath_batch, metadata_batch):
        flat_metadatas[keypath] = value
    return utils.from_flat_dict(flat_metadatas, target=metadata)

  def metadata(self, directory: epath.Path) -> Optional[PyTree]:
    """Returns tree metadata.

    The result will be a PyTree matching the structure of the saved checkpoint.
    Note that if the item saved was a custom class, the restored metadata will
    be returned as a nested dictionary representation.

    Example::

      {
        'layer0': {
            'w': ArrayMetadata(dtype=jnp.float32, shape=(8, 8), shards=(1, 2)),
            'b': ArrayMetadata(dtype=jnp.float32, shape=(8,), shards=(1,)),
        },
        'step': ScalarMetadata(dtype=jnp.int64),
      }

    If the required metadata file is not present, this method will raise an
    error.

    Args:
      directory: checkpoint location.

    Returns:
      tree containing metadata.
    """
    try:
      return self._get_user_metadata(directory)
    except FileNotFoundError as e:
      raise FileNotFoundError('Could not locate metadata file.') from e

  def close(self):
    """Closes the handler. Called automatically by Checkpointer."""
    self._aggregate_handler.close()