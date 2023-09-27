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

import max_logging

from flax.training import train_state

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
    in_tree, full_mesh, per_slice_sharding, is_source
):
  num_slices = full_mesh.devices.shape[0]

  @functools.partial(jax.jit, static_argnums=0)
  def fake_zero_data(sharding, x):
    x = jnp.zeros_like(x)
    return jax.lax.with_sharding_constraint(x, sharding)

  def pre_jit(x):
    if is_source:
      inp = x
    else:
      inp = fake_zero_data(per_slice_sharding, x)
    inp = jnp.expand_dims(inp, axis=0)
    in_spec = jax.sharding.PartitionSpec("data", *x.sharding.spec)
    global_shape = (num_slices, *x.shape)
    global_sharding = jax.sharding.NamedSharding(full_mesh, in_spec)
    print(global_shape, global_sharding, x.shape)
    return jax.make_array_from_single_device_arrays(
        global_shape, global_sharding, [s.data for s in inp.addressable_shards]
    )

  out_sharding = jax.tree_map(
      lambda x: jax.sharding.NamedSharding(
          full_mesh, jax.sharding.PartitionSpec(*x.sharding.spec)
      ),
      in_tree,
  )

  in_tree = jax.tree_map(pre_jit, in_tree)
  print(f"{in_tree.shape=}")
  out_tree = jax.jit(_sum, out_shardings=out_sharding)(in_tree)
  return out_tree


def _is_host_for_slice(idx: int) -> bool:
  return jax.local_devices()[0].slice_index == idx


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
    deserialized = await asyncio.gather(*deserialize_ops)
    return broadcast_one_slice_to_all(
        deserialized,
        shardings[0].mesh,
        single_slice_shardings[0],
        is_source=_is_host_for_slice(0),
    )












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
    checkpointer = AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())
  else:
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())

  if enable_single_slice_checkpointing:
    ocp.type_handlers.register_type_handler(
        jax.Array, SingleSliceArrayHandler(), override=True
    )

  mngr = CheckpointManager(
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
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
    return None, checkpointer.restore(p,
                                      item=abstract_unboxed_pre_state,
                                      restore_args=restore_args).params
  elif load_from_other_directory != "":
    p = epath.Path(load_from_other_directory)
    checkpointer_loader = Checkpointer(checkpoint.PyTreeCheckpointHandler())
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
