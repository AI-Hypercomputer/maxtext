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

import jax
import numpy as np
import max_logging
from etils import epath
from flax.training import train_state
from orbax import checkpoint
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions, Checkpointer, AsyncCheckpointer
from orbax.checkpoint import type_handlers


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(checkpoint_dir)
  if use_async:
    checkpointer = AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler(),
                                     timeout_secs=900)
  else:
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())

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


def _find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = _find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array,
                           idx,
                           axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           load_parameters_from_path: str,
                           load_full_state_from_path: str,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           mesh,
                           state_mesh_annotations,
                           enable_single_replica_ckpt_restoring):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint manager,
      load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint manager,
      load full state from a full state checkpoint at this path.
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
  def map_to_pspec(data, pspec):
    if isinstance(data, (jax.Array, jax.ShapeDtypeStruct)) \
          and pspec is not None:
      if enable_single_replica_ckpt_restoring:
        type_handlers.register_type_handler(jax.Array,
                                            type_handlers.SingleReplicaArrayHandler(),
                                            override=True)
        replica_axis_index = 0  # for maxtext data is the first dimension
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)
        # print('restore_args: mesh and pspec', mesh.shape, pspec)
        # print('restore_args: single_replica_sharding', single_replica_sharding)
        return type_handlers.SingleReplicaArrayRestoreArgs(
          sharding=jax.sharding.NamedSharding(mesh, pspec),
          single_replica_sharding=single_replica_sharding,
          replica_axis_index=replica_axis_index,
          global_shape=data.shape,
          dtype=data.dtype,
          )
      else:
        return type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)

    return type_handlers.RestoreArgs()
  restore_args = jax.tree_util.tree_map(map_to_pspec,
                                        abstract_unboxed_pre_state,
                                        state_mesh_annotations)

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
      max_logging.log(f"restoring state from this run's directory latest step \
          {latest_step}")
      return checkpoint_manager.restore(latest_step, abstract_unboxed_pre_state,
                                        {"restore_args" : restore_args}), None
    else:
      max_logging.log("failed to find preexisting checkpoint for this run")

  if load_parameters_from_path != "":
    max_logging.log(f"restoring params from {load_parameters_from_path=}")
    p = epath.Path(load_parameters_from_path)
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
    restore_args_param_train_state = train_state.TrainState(step = restore_args.step, params = restore_args.params,\
                                                            tx=None,  opt_state = {}, apply_fn=None) # type: ignore
    abstract_param_train_state = train_state.TrainState(step = abstract_unboxed_pre_state.step,\
                                                        params = abstract_unboxed_pre_state.params,\
                                                        tx=None,opt_state = {}, apply_fn=None) # type: ignore
    full_restored_state = checkpointer.restore(p, item = abstract_param_train_state,\
                                               restore_args = restore_args_param_train_state)
    return None, full_restored_state.params
  elif load_full_state_from_path != "":
    max_logging.log(f"restoring full state from {load_full_state_from_path=}")
    p = epath.Path(load_full_state_from_path)
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
    return checkpointer.restore(p,
                                item=abstract_unboxed_pre_state,
                                restore_args=restore_args), None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None
