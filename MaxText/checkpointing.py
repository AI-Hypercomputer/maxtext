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
    item_names = ('default', 'iter')
  elif dataset_type=='c4':
    item_names = ('default',)
  else:
    raise ValueError(f"Unknown dataset_type {dataset_type}. dataset_type must be c4, c4-array_record or synthetic")

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


def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           data_iterator: Union[MultiHostDataLoadIterator, None],
                           load_parameters_from_path: str,
                           load_full_state_from_path: str,
                           abstract_unboxed_pre_state: train_state.TrainState,
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
    mesh: a physical TPU mesh
    state_mesh_annotation: a PyTree of sharding rules, matching
      abstract_unboxed_pre_state.

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
      if dataset_type == 'c4-array_record' and data_iterator is not None:
        return checkpoint_manager.restore(latest_step,
                                      args=orbax.checkpoint.args.Composite(
                                      default=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state),
                                      iter=grain.PyGrainCheckpointRestore(data_iterator.local_iterator)
                                    )), None
      elif dataset_type == 'c4':
        return checkpoint_manager.restore(latest_step,
                                      args=orbax.checkpoint.args.Composite(
                                      default=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state),
                                    )), None
      else:
        raise ValueError(f"Unknown dataset_type {dataset_type}. dataset_type must be c4, c4-array_record or synthetic")

  if load_parameters_from_path != "":
    max_logging.log(f"restoring params from {load_parameters_from_path=}")
    p = epath.Path(load_parameters_from_path)
    ckptr = orbax.checkpoint.PyTreeCheckpointer()
    metadata = ckptr.metadata(p)
    restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(metadata)
    restored = ckptr.restore(p, item = {'params': metadata['params']}, transforms={},
                             restore_args = {'params': restore_args['params']})

    return None, restored['params']

  elif load_full_state_from_path != "":
    max_logging.log(f"restoring full state from {load_full_state_from_path=}")
    p = epath.Path(load_full_state_from_path)
    ckptr = orbax.checkpoint.StandardCheckpointer()
    restored = ckptr.restore(p, args=orbax.checkpoint.args.StandardRestore(abstract_unboxed_pre_state))
    return  {'default': restored, 'iter': None}, None

  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None
