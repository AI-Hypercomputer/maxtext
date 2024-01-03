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
    dataset_type: str,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None
  max_logging.log("Creating checkpoint manager...")
  p = epath.Path(checkpoint_dir)
  if use_async:
    if dataset_type == "array_record":
      checkpointer = {'state':AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler()),
                      'iter':Checkpointer(pygrain.PyGrainCheckpointHandler())}
    else:
      checkpointer = AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler()) 
  else:
    if dataset_type == "array_record":
      checkpointer = {'state':Checkpointer(checkpoint.PyTreeCheckpointHandler()),
                      'iter':Checkpointer(pygrain.PyGrainCheckpointHandler())}
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


def load_state_if_possible(checkpoint_manager: CheckpointManager,
                           first_checkpoint_path: str,
                           load_from_other_directory: str,
                           load_from_other_directory_step: int,
                           abstract_unboxed_pre_state: train_state.TrainState,
                           dataset_type,
                           iterator,
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
      return type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)
    else:
      return type_handlers.RestoreArgs()

  if dataset_type=="array_record":
    restore_state = jax.tree_util.tree_map(map_to_pspec,
                                          abstract_unboxed_pre_state,
                                          state_mesh_annotations)
    restore_args = {'state':restore_state, 'iter':iterator}
  else:
    restore_args = jax.tree_util.tree_map(map_to_pspec,
                                          abstract_unboxed_pre_state,
                                          state_mesh_annotations) 

  latest_step = checkpoint_manager.latest_step()
  if latest_step is not None:
    max_logging.log(f"restoring state from this run's directory latest step \
        {latest_step}")
    if dataset_type=="array_record":
      return checkpoint_manager.restore(latest_step, {'state':abstract_unboxed_pre_state,'iter':iterator},
                                      {"restore_args" : restore_args}), None
    else:
      return checkpoint_manager.restore(latest_step, abstract_unboxed_pre_state,
                                      {"restore_args" : restore_args}), None
  elif first_checkpoint_path != "":
    max_logging.log(f"restoring state from first_checkpoint_path {first_checkpoint_path}")
    p = epath.Path(first_checkpoint_path)
    checkpointer = Checkpointer(checkpoint.PyTreeCheckpointHandler())
    if dataset_type=="array_record":
      return None, checkpointer.restore(p,
                                      item={'state':abstract_unboxed_pre_state,'iter':iterator},
                                      restore_args=restore_args).params
    else:
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
    if dataset_type=="array_record":
      return mngr_loader.restore(step, {'state':abstract_unboxed_pre_state,'iter':iterator},
                                      {"restore_args" : restore_args}), None
    else:
      return mngr_loader.restore(step, abstract_unboxed_pre_state,
                                      {"restore_args" : restore_args}), None                                        
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None
