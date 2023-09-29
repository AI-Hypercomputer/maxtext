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

"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

from etils import epath
import jax
import portpicker
from jax.experimental import multihost_utils
from orbax import checkpoint
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions, Checkpointer, AsyncCheckpointer
from orbax.checkpoint import type_handlers
import orbax.checkpoint as ocp
import socket
import time

import max_logging

from flax.training import train_state


import asyncio
import dataclasses
import functools
import socket
from typing import Any, Optional, Sequence
from typing import cast
from etils import epath
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization
import max_logging
import numpy as np
import orbax.checkpoint as ocp
import portpicker
from local_checkpoint_manager import MyCheckpointManager

jax.config.update('jax_spmd_mode', 'allow_all')

def _multislice_distribute_initialize():
  """Calls jax.distribute.initialize() with appropriate multislice arguments."""

  def gen_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

  def gen_local_ip_nums():
    return [int(num) for num in gen_local_ip().split(':')[-1].split('.')]

  def get_coordinator_ip():
    local_ip_nums = jax.numpy.array(gen_local_ip_nums())
    coordinator_ip_nums = multihost_utils.broadcast_one_to_all(local_ip_nums)
    coordinator_ip_strings = [str(num) for num in list(coordinator_ip_nums)]
    return '.'.join(coordinator_ip_strings)

  port = multihost_utils.broadcast_one_to_all(jax.numpy.array(portpicker.pick_unused_port()))
  coordinator_address = get_coordinator_ip() + ':' + str(port)
  jax.distributed.initialize(coordinator_address=coordinator_address,
                             num_processes=jax.process_count(),
                             process_id=jax.process_index())

def _sum(x):
  return jax.tree_map(functools.partial(jnp.sum, axis=0), x)


def broadcast_one_slice_to_all(
    in_tree, global_mesh, per_slice_shardings, is_source
):
  num_slices = global_mesh.devices.shape[0]

  @functools.partial(jax.jit, static_argnums=0)
  def fake_zero_data(sharding, x):
    x = jnp.zeros_like(x)
    return jax.lax.with_sharding_constraint(x, sharding)

  def pre_jit(x, per_slice_sharding):
    if is_source:
      inp = x
    else:
      inp = fake_zero_data(per_slice_sharding, x)
    inp = jnp.expand_dims(inp, axis=0)
    in_spec = jax.sharding.PartitionSpec("data", *x.sharding.spec)
    global_shape = (num_slices, *x.shape)
    global_sharding = jax.sharding.NamedSharding(global_mesh, in_spec)
    arr = jax.make_array_from_single_device_arrays(
        global_shape, global_sharding, [s.data for s in inp.addressable_shards]
    )
    # if is_source:
    #   x.delete()
    # else:
    #   x.delete()
    #   inp.delete()
    return arr

  out_sharding = jax.tree_map(
      lambda x: jax.sharding.NamedSharding(
          global_mesh, jax.sharding.PartitionSpec(*x.sharding.spec)
      ),
      in_tree,
  )

  # in_tree = jax.tree_map(pre_jit, in_tree, per_slice_shardings)
  # out_tree = jax.jit(_sum, out_shardings=out_sharding)(in_tree)
  in_tree_sharded = jax.tree_map(pre_jit, in_tree, per_slice_shardings)
  jax.tree_map(lambda x: x.delete(), in_tree)
  out_tree = jax.jit(_sum, out_shardings=out_sharding)(in_tree_sharded)
  return out_tree


def _is_host_for_slice(idx: int) -> bool:
  try:
    return jax.local_devices()[0].slice_index == idx
  except:
    return True


@dataclasses.dataclass
class SingleSliceArrayRestoreArgs(ocp.ArrayRestoreArgs):
  single_slice_sharding: Optional[jax.sharding.NamedSharding] = None


class SingleSliceArrayHandler(ocp.type_handlers.ArrayHandler):

  async def deserialize(
      self,
      infos: Sequence[ocp.type_handlers.ParamInfo],
      args: Optional[Sequence[ocp.RestoreArgs]] = None,
  ) -> Sequence[jax.Array]:
    """See superclass documentation.

    Args:
      infos: ParamInfo.
      args: must be of type `ArrayRestoreArgs`.

    Returns:
      The deserialized parameter.

    Raises:
      ValueError if `args` is not provided.
      ValueError if `args.sharding` is not provided or `args.mesh` and
      `args.mesh_axes` are not provided.
    """
    if args is None:
      raise ValueError('Must provide ArrayRestoreArgs to restore as jax.Array.')
    ocp.type_handlers.check_input_arguments(infos, args)
    deserialize_ops = []
    shardings = []
    single_slice_shardings = []
    start_time = time.time()
    for info, arg in zip(infos, args):
      arg = cast(SingleSliceArrayRestoreArgs, arg)
      if isinstance(arg, SingleSliceArrayRestoreArgs):
        if arg.sharding is not None:
          sharding = arg.sharding
          shardings.append(sharding)
        else:
          raise ValueError('Must provide `sharding`.')
        if arg.single_slice_sharding is not None:
          single_slice_sharding = arg.single_slice_sharding
          single_slice_shardings.append(single_slice_sharding)
        else:
          raise ValueError('Must provide `sharding`.')
      else:
        raise ValueError('Must provide `ArrayRestoreArgs`.')
      if not info.is_ocdbt_checkpoint:
        await ocp.type_handlers._assert_parameter_files_exist(  # pylint: disable=protected-access
            info.path, self._metadata_key
        )
      # Using OCDBT, but existing checkpoint may be stored in old format.
      use_ocdbt = ocp.type_handlers._use_ocdbt_for_restore(  # pylint: disable=protected-access
          self._use_ocdbt, info.is_ocdbt_checkpoint
      )
      tspec = self._get_json_tspec_read(info, use_ocdbt=use_ocdbt)
      tspec = ocp.type_handlers._get_cast_tspec_deserialize(tspec, arg)  # pylint: disable=protected-access

      if _is_host_for_slice(0):
        deserialize_ops += [
            serialization.async_deserialize(
                single_slice_sharding,
                tspec,
                global_shape=arg.global_shape
                if hasattr(arg, "global_shape")
                else None,
                byte_limiter=info.byte_limiter,
                context=self._ts_context,
            )
        ]

    print("Finished for loop!!", flush=True)
    if _is_host_for_slice(0):
      print("Before gather!!")
      deserialized = await asyncio.gather(*deserialize_ops)
    else:
      # def make_data():
      #   return jax.numpy.ones(1024)

      # pjit_make_data = jax.jit(make_data, out_shardings=single_slice_shardings)
      # deserialized = [pjit_make_data() for _ in deserialize_ops]

      print("Creating dummy arrays.", flush=True)

      single_slice_shardings = [arg.single_slice_sharding for arg in args]
      shape_dtype = [
          jax.ShapeDtypeStruct(arg.global_shape, arg.dtype) for arg in args
      ]

      @functools.partial(
          jax.jit, static_argnums=0, out_shardings=tuple(single_slice_shardings)
      )
      def create_zeros(shape_dtype_tup):
        return jax.tree_util.tree_map(lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup)

      deserialized = create_zeros(tuple(shape_dtype))

      # dummy_arrs = [np.zeros(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)]
      # deserialized = jax.jit(
      #     lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype),
      #     out_shardings=single_slice_shardings,
      #     static_argnums=0,
      # )(shape_dtype)

    # deserialized = [deserialized[0]]
    # single_slice_shardings = [single_slice_shardings[0]]
    deserialized = tuple(deserialized)
    single_slice_shardings = tuple(single_slice_shardings)

    # for d in deserialized:
    #   print((d.shape, d.dtype, d.sharding), flush=True)

    print("Finished loading on slice 0!!", flush=True)
    end_loading = time.time()
    print(f"Deserializing in time {end_loading - start_time}", flush=True)
    start_broadcast = time.time()
    shared_state = []
    for i in range(len(deserialized)):
      r = broadcast_one_slice_to_all(
          [deserialized[i]],
          shardings[0].mesh,
          [single_slice_shardings[i]],
          is_source=_is_host_for_slice(0),
      )         
      shared_state += r

    print("Finished broadcasting shared state!", flush=True)
    #print(shared_state, flush=True)
    print(shared_state[0], flush=True)
    print(shared_state[0].addressable_shards, flush=True)

    jax.block_until_ready(shared_state)
    end_broadcast = time.time()
    print("Blocked Finished for shared state!", flush=True)
    print(f"Finished broadcasting in {end_broadcast - start_broadcast}", flush=True)
    # print(shared_state[0])
    # print(shared_state.sharding)
    # for s in shared_state.addressable_shards:
    #   print(f"\n{s}\n", flush=True)
    # assert 1 > 2
    print("Returning", flush=True)
    return shared_state


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    enable_single_slice_checkpointing: bool = False,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(checkpoint_dir)
  if use_async:
    _multislice_distribute_initialize()
    checkpointer = AsyncCheckpointer(PyTreeCheckpointHandler())
  else:
    checkpointer = Checkpointer(PyTreeCheckpointHandler())

  if enable_single_slice_checkpointing:
    # ocp.type_handlers.register_type_handler(
    #     jax.Array, SingleSliceArrayHandler(), override=True
    # )
    ocp.type_handlers.register_type_handler(jax.Array, SingleSliceArrayHandler(), override=True)
    ocp.type_handlers._enable_ocdbt_for_handlers()
    #ocp.type_handlers.register_type_handler(jax.Array, SingleSliceArrayHandler(use_ocdbt=True, ts_context=ocp.type_handlers._OCDBT_TS_CONTEXT), override=True)

  mngr = MyCheckpointManager(
      p,
      checkpointer,
      options=CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps
      )
  )
  max_logging.log("Checkpoint manager created!")
  return mngr


def _find_np_idx(array, filter_fn):
  for idx, val in np.ndenumerate(array):
    if filter_fn(val):
      return idx


def _slice_devices(device_array):
  ### slices are assumed to be restricted to the first axis
  idx = _find_np_idx(
      device_array, lambda x: x.process_index == jax.process_index()
  )
  zeroth_idx = idx[0]
  sliced_result = device_array[zeroth_idx : zeroth_idx + 1, :, :]
  return sliced_result

def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           first_checkpoint_path: str,
                           load_from_other_directory: str,
                           load_from_other_directory_step: int,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           mesh,
                           state_mesh_annotations):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    first_checkpoint_path: if there is no checkpoint in the checkpoint manager,
      return the Params from the first_checkpoint_path if they exist. This
      enables loading just the parameters and is intended for finetuning.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    mesh: a physical TPU mesh
    state_mesh_annotation: a PyTree of sharding rules, matching
      abstract_unboxed_pre_state.

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """
  if checkpoint_manager is None:
    max_logging.log("no checkpoint manager, not restoring checkpoint")
    return None, None
  def map_to_pspec(data, pspec):
    if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)) \
          and pspec is not None:
      slice_devices = _slice_devices(mesh.devices)
      slice_mesh = jax.sharding.Mesh(slice_devices, mesh.axis_names)
      return SingleSliceArrayRestoreArgs(
          sharding=jax.sharding.NamedSharding(mesh, pspec),
          single_slice_sharding=jax.sharding.NamedSharding(slice_mesh, pspec),
          global_shape=data.shape,
          dtype=data.dtype,
      )
    else:
      return type_handlers.RestoreArgs()

  restore_args = jax.tree_util.tree_map(map_to_pspec,
                                        abstract_unboxed_pre_state,
                                        state_mesh_annotations)
  latest_step = checkpoint_manager.latest_step()

  if latest_step is not None:
    max_logging.log(f"restoring state from this run's directory latest step \
        {latest_step}")
    return checkpoint_manager.restore(latest_step, abstract_unboxed_pre_state,
                                      {"restore_args" : restore_args}), None
  elif first_checkpoint_path != "":
    max_logging.log(f"restoring state from first_checkpoint_path {first_checkpoint_path}")
    p = epath.Path(first_checkpoint_path)
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    return None, checkpointer.restore(p,
                                      item=abstract_unboxed_pre_state,
                                      restore_args=restore_args).params
  elif load_from_other_directory != "":
    p = epath.Path(load_from_other_directory)
    checkpointer_loader = Checkpointer(PyTreeCheckpointHandler())
    mngr_loader = CheckpointManager(p, checkpointer_loader, options=CheckpointManagerOptions(create=True))
    if load_from_other_directory_step == -1:
      step = mngr_loader.latest_step()
      max_logging.log(f"restoring state from {load_from_other_directory} latest step {step}")
    else:
      step = load_from_other_directory_step
      max_logging.log(f"restoring state from {load_from_other_directory} step {step}")
    return mngr_loader.restore(step, abstract_unboxed_pre_state,
                                      {"restore_args" : restore_args}), None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


class PyTreeCheckpointHandler(ocp.PyTreeCheckpointHandler):
  
  async def _maybe_deserialize(
      self, structure, param_infos, restore_args
  ):
    """Deserializes values or gets them from the aggregate file."""

    # Handle parameters from aggregate file.
    def _process_aggregated_value(info, meta, args):
      value = meta.aggregate_value
      if info.skip_deserialize:
        value = ocp.pytree_checkpoint_handler._try_array_cast(value, args.dtype)
        value = ocp.pytree_checkpoint_handler._maybe_shard_array(value, args)
      return value

    flat_aggregate = ocp.utils.to_flat_dict(
        jax.tree_util.tree_map(
            _process_aggregated_value, param_infos, structure, restore_args
        ),
        sep='.',
    )

    batch_requests = ocp.pytree_checkpoint_handler._batched_serialization_requests(
        structure, param_infos, restore_args
    )
    print('before deserialize', flush=True)
    deserialized_batches = []
    deserialized_batches_ops = []
    for request in batch_requests:
      deserialized_batches_ops.append(
          request.handler.deserialize(request.infos, request.args)
      )
    deserialized_batches += await asyncio.gather(*deserialized_batches_ops)
    print('after deserialize', flush=True)

    flat_restored = {}
    for request, deserialized in zip(batch_requests, deserialized_batches):
      for info, value in zip(request.infos, deserialized):
        assert not info.skip_deserialize
        flat_restored[info.name] = value
    # Add in any values which were not deserialized, coming from aggregate file.
    for key in flat_aggregate.keys():
      if key not in flat_restored:
        flat_restored[key] = flat_aggregate[key]
    return ocp.utils.from_flat_dict(flat_restored, target=structure, sep='.')

  def restore(
      self,
      directory,
      item=None,
      restore_args=None,
      transforms=None,
      transforms_default_to_original=True,
      legacy_transform_fn=None,
  ):
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
    byte_limiter = ocp.pytree_checkpoint_handler.get_byte_limiter(
        self._concurrent_gb
    )
    structure = self._get_internal_metadata(directory)
    # `checkpoint_restore_args` has a structure relative to the checkpoint,
    # while `restore_args` remains structured relative to the output.
    param_infos, checkpoint_restore_args = (
        ocp.pytree_checkpoint_handler._get_restore_parameters(
            directory,
            item,
            structure,
            transforms,
            restore_args,
            byte_limiter=byte_limiter,
            transforms_default_to_original=transforms_default_to_original,
        )
    )

    if legacy_transform_fn is not None and transforms is not None:
      raise ValueError(
          'Cannot provide both `transforms` and `legacy_transform_fn`.'
      )
    if legacy_transform_fn is not None:
      structure, param_infos = legacy_transform_fn(item, structure, param_infos)
      if restore_args is None:
        restore_args = jax.tree_util.tree_map(lambda x: ocp.RestoreArgs(), item)
      checkpoint_restore_args = restore_args

    def _maybe_set_default_restore_types(meta, arg):
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
    print('here1', flush=True)

    if not legacy_transform_fn:
      restored_item = ocp.pytree_checkpoint_handler._transform_checkpoint(
          item,
          restored_item,
          restore_args,
          transforms,
          transforms_default_to_original,
      )
    print('here2', flush=True)
    ocp.utils.sync_global_devices("PyTreeCheckpointHandler:restore")
    print('here3', flush=True)
    return restored_item
