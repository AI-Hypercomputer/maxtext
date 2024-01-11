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
# pylint: disable=line-too-long

from etils import epath
import jax
from orbax import checkpoint
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions, Checkpointer, AsyncCheckpointer
from orbax.checkpoint import type_handlers
import grain.python as pygrain

import max_logging

from flax.training import train_state

def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(checkpoint_dir)
  if use_async:
    checkpointer = AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())
  else:
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())

  mngr = CheckpointManager(
      p,
      checkpointer,
      options=CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps
      ),
      metadata={'iter': False}
  )
  max_logging.log("Checkpoint manager created!")
  return mngr

def create_orbax_checkpoint_manager_pygrain(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager (including data iterator)...")
  p = epath.Path(checkpoint_dir)
  if use_async:
    checkpointer = {'default':AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler()),
                       'iter':Checkpointer(pygrain.PyGrainCheckpointHandler())}
  else:
    checkpointer = {'default':Checkpointer(checkpoint.PyTreeCheckpointHandler()),
                       'iter':Checkpointer(pygrain.PyGrainCheckpointHandler())}

  mngr = CheckpointManager(
      p,
      checkpointer,
      options=CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps
      ),
      metadata={'iter': True}
  )
  max_logging.log("Checkpoint manager created!")
  return mngr

def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           load_parameters_path: str,
                           load_from_other_directory: str,
                           load_from_other_directory_step: int,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           load_data_iterator_from_checkpoint,
                           iterator,
                           mesh,
                           state_mesh_annotations):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_path: This enables loading just the parameters and is intended 
      for finetuning.
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
      return type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)
    else:
      return type_handlers.RestoreArgs()

  # Set restore_args based whether to load data iterator
  if load_parameters_path != "" or not load_data_iterator_from_checkpoint:
    restore_args = jax.tree_util.tree_map(map_to_pspec,
                                          abstract_unboxed_pre_state,
                                          state_mesh_annotations)
  else:
    restore_state = jax.tree_util.tree_map(map_to_pspec,
                                          abstract_unboxed_pre_state,
                                          state_mesh_annotations)
    restore_args = {'default':restore_state, 'iter':iterator}

  # Get step to load
  if load_from_other_directory != "" and load_from_other_directory_step != -1:
    step = load_from_other_directory_step
  else:
    step = checkpoint_manager.latest_step()

  # if load_from_other_directory is set, load from it.
  # Otherwise, load from checkpoint_dir (base_output_directory/checkpoints)
  if step is not None:
    if load_data_iterator_from_checkpoint:
      if load_from_other_directory != "":
        p = epath.Path(load_from_other_directory)
        iter_path = epath.Path(p / str(step) / 'iter')
      else:
        iter_path = epath.Path(checkpoint_manager.directory / str(step) / 'iter')

      if not iter_path.exists():
        raise FileNotFoundError(f"Data iterator path not found: {iter_path}. Please set load_data_iterator_from_checkpoint to False if loading is not intended.")

      max_logging.log(f"restoring step {step} state from {checkpoint_manager.directory} and data iterator from {iter_path}")
      return checkpoint_manager.restore(step, {'default':abstract_unboxed_pre_state,'iter':iterator},
                                      {"restore_args" : restore_args}), None
    else:
      max_logging.log(f"restoring step {step} state from {checkpoint_manager.directory}")
      return checkpoint_manager.restore(step, abstract_unboxed_pre_state,
                              {"restore_args" : restore_args}), None

  # load params only from load_parameters_path
  elif load_parameters_path != "":
    max_logging.log(f"restoring params from load_parameters_path {load_parameters_path}")
    p = epath.Path(load_parameters_path)
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
    return None, checkpointer.restore(p,
                                      item=abstract_unboxed_pre_state,
                                      restore_args=restore_args).params

  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None
