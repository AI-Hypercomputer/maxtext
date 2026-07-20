# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orbax v0 emergency / multi-tier checkpoint managers (no v1 equivalent).

MaxText's standard checkpointing runs on Orbax v1 (see ``common/checkpointing.py``).
The emergency and multi-tier replicator managers have no v1 counterpart, so they
stay on Orbax v0 — isolated here so the standard path stays v0-free.
"""

from typing import Any

from absl import flags
from etils import epath
from flax import nnx
import jax
from maxtext.common import train_state_nnx
from maxtext.utils import gcs_utils
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import max_logging
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager


DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = maxtext_globals.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
CheckpointManager = emergency_checkpoint_manager.CheckpointManager
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = emergency_checkpoint_manager.PersistentCheckpointOptions
ReplicatorCheckpointManager = emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager

_MANAGERS = (CheckpointManager, ReplicatorCheckpointManager)


def create_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: Any,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
    orbax_logger: Any = None,  # pytype: disable=attribute-error
):
  """Returns an emergency checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency checkpoint manager...")

  # Only create local directories if running on GPUs as the previous directory structure might be assumed by TPUs.
  if global_mesh.devices.flatten()[0].platform == "gpu":
    # pylint: disable=protected-access
    local_checkpoint_dir = f"{local_checkpoint_dir}/{jax._src.distributed.global_state.process_id}"
    local_p = epath.Path(local_checkpoint_dir)
    local_p.mkdir(exist_ok=True, parents=True)

  persistent_p = gcs_utils.mkdir_and_check_permissions(persistent_checkpoint_dir)

  # pure_nnx saves via to_checkpoint_dict (Linen params/opt_state/step plus an nnx_aux
  # subtree), but the emergency manager restores against the abstract it is built with.
  # Convert it the same way so it matches what is on disk; restore reshapes back to NNX.
  if isinstance(abstract_state, nnx.State):
    abstract_state = train_state_nnx.to_checkpoint_dict(abstract_state)

  manager = CheckpointManager(
      local_checkpoint_dir,
      persistent_p,
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=emergency_checkpoint_manager.CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=local_save_interval_steps),
          persistent=PersistentCheckpointOptions(save_interval_steps=persistent_save_interval_steps),
      ),
      logger=orbax_logger,
  )

  max_logging.log("Emergency checkpoint manager created!")
  return manager


def create_replicator_checkpoint_manager(
    local_checkpoint_dir: str,
    save_interval_steps: int,
    global_mesh: jax.sharding.Mesh,
    colocated_python_checkpointing: bool = False,
):
  """Returns an emergency replicator checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency replicator checkpoint manager...")

  manager = ReplicatorCheckpointManager(
      epath.Path(local_checkpoint_dir),
      options=emergency_replicator_checkpoint_manager.ReplicatorCheckpointManagerOptions(
          save_interval_steps=save_interval_steps,
          use_colocated_python=colocated_python_checkpointing,
      ),
      global_mesh=global_mesh,
  )

  max_logging.log("Emergency replicator checkpoint manager created!")
  return manager


def save(checkpoint_manager, step, state, config, force):
  """v0 emergency save: writes the state under the ``state`` checkpointable."""
  chunk_byte_size = (
      config.checkpoint_storage_target_data_file_size_bytes if config else DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  )
  checkpoint_args = ocp.args.PyTreeSave(
      item=state,
      save_args=jax.tree.map(lambda _: ocp.SaveArgs(chunk_byte_size=chunk_byte_size), state),
      ocdbt_target_data_file_size=chunk_byte_size,
  )
  replicator_error_handler(config)
  return checkpoint_manager.save(step, args=ocp.args.Composite(state=checkpoint_args), force=force)


def restore(checkpoint_manager, step, abstract_unboxed_pre_state):
  """v0 emergency restore: returns the restored state pytree."""
  restore_target = abstract_unboxed_pre_state
  if isinstance(abstract_unboxed_pre_state, nnx.State):
    restore_target = abstract_unboxed_pre_state.to_pure_dict()
  restore_args = jax.tree_util.tree_map(
      lambda data: ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding), restore_target
  )
  checkpoint_args = ocp.args.PyTreeRestore(item=restore_target, restore_args=restore_args, partial_restore=True)
  return checkpoint_manager.restore(step, args=ocp.args.Composite(state=checkpoint_args)).state


def replicator_error_handler(config: Any):
  """Replicator error handler to handle errors in replicator service."""
  if config.enable_multi_tier_checkpointing:
    local_dir = config.local_checkpoint_directory
    replicator_errors_file = f"{local_dir}/replicator.errors"
    replicator_failed_file = f"{local_dir}/replicator.failed"
    process_replicator_error_file(replicator_errors_file)

    # if the replicator.failed file exists, then we have a fatal error
    is_fatal = process_replicator_error_file(replicator_failed_file)
    if is_fatal:
      raise ValueError("Replicator fatal error found in replicator.failed file.")


def process_replicator_error_file(error_file: str) -> bool:
  """Handles replicator errors by reading, logging, cleaning the error file."""
  error_file_path_exists = epath.Path(error_file).exists()
  if error_file_path_exists:
    max_logging.log(f"replicator_error_handler: file found: {error_file}.")
    read_replicator_error_file(error_file)
    cleanup_replicator_error_file(error_file)

  return error_file_path_exists


def read_replicator_error_file(error_file: str):
  """Read replicator errors file."""
  try:
    error_data = epath.Path(error_file).read_text()
    max_logging.log(f"Contents of replicator error file:\n{error_data}")
  except (OSError, ValueError) as e:
    max_logging.log("replicator_error_handler: Failed to read contents of failed" f" file: {e}")


def cleanup_replicator_error_file(error_file: str):
  """Clean up replicator errors file."""
  try:
    epath.Path(error_file).unlink()
  except (OSError, ValueError) as e:
    max_logging.log("replicator_error_handler: Failed to remove replicator errors file:" f" {e}")
