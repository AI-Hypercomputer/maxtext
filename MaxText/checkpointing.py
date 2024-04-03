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

from typing import Optional, Union
from etils import epath
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions
import jax
import numpy as np
import orbax.checkpoint
import grain.python as grain

import max_logging
from multihost_dataloading import MultiHostDataLoadIterator
from flax.training import train_state

def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: Optional[str] = 'c4'
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(checkpoint_dir)

  if dataset_type=='c4-array_record':
    item_names = ('items', 'iter')
  else:
    item_names = ('items',)

  mngr = CheckpointManager(
      p,
      item_names = item_names,
      options = CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
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
                           data_iterator: Union[MultiHostDataLoadIterator, None],
                           load_parameters_from_path: str,
                           load_full_state_from_path: str,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           enable_single_replica_ckpt_restoring: Optional[bool] = False,
                           dataset_type: Optional[str] = 'c4'):
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
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
      with SingleReplicaArrayHandler

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
      max_logging.log(f"restoring from this run's directory latest step \
          {latest_step}")

      def  map_to_pspec(data):
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        if not enable_single_replica_ckpt_restoring:
          return orbax.checkpoint.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)
        orbax.checkpoint.type_handlers.register_type_handler(
          jax.Array,
          orbax.checkpoint.type_handlers.SingleReplicaArrayHandler(),
          override=True)
        orbax.checkpoint.type_handlers.register_type_handler(
          jax.Array,
          orbax.checkpoint.type_handlers.SingleReplicaArrayHandler(),
          override=True)
        replica_axis_index = 0  # for maxtext data is the first dimension
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)
        return orbax.checkpoint.type_handlers.SingleReplicaArrayRestoreArgs(
          sharding=jax.sharding.NamedSharding(mesh, pspec),
          single_replica_sharding=single_replica_sharding,
          replica_axis_index=replica_axis_index,
          global_shape=data.shape,
          dtype=data.dtype,
          )

      restore_args = jax.tree_util.tree_map(map_to_pspec,
                                            abstract_unboxed_pre_state,
                                            )
      if dataset_type == 'c4-array_record' and data_iterator is not None:
        return checkpoint_manager.restore(
          latest_step,
          args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeRestore(
            item=abstract_unboxed_pre_state,
            restore_args=restore_args),
            iter=grain.PyGrainCheckpointRestore(data_iterator.local_iterator))
          ), None
      else:
        return (
          checkpoint_manager.restore(
            latest_step,
            args=orbax.checkpoint.args.Composite(
              items=orbax.checkpoint.args.PyTreeRestore(
              item=abstract_unboxed_pre_state,
              restore_args=restore_args)
            ),
          ),
          None)

  if load_parameters_from_path != "":
    max_logging.log(f"restoring params from {load_parameters_from_path=}")
    p = epath.Path(load_parameters_from_path)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
    # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
    # memory, we instead specify here that we are just restoring the params field of the checkpoint
    # (which itself may be a dictionary containing a key named 'params').
    restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(abstract_unboxed_pre_state.params)
    restored = ckptr.restore(p, item = {'params': abstract_unboxed_pre_state.params}, transforms={},
                                restore_args = {'params': restore_args})
    return None, restored['params']

  elif load_full_state_from_path != "":
    max_logging.log(f"restoring full state from {load_full_state_from_path=}")
    p = epath.Path(load_full_state_from_path)
    ckptr = orbax.checkpoint.StandardCheckpointer()
    restored = ckptr.restore(p, args=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state))
    return  {'items': restored}, None

  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None
