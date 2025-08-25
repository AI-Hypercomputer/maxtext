# Copyright 2023–2025 Google LLC
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

"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

import time
from typing import Any

from absl import flags
from etils import epath
from flax.training import train_state
import grain.python as grain
import jax
from MaxText import exceptions
from MaxText import max_logging
from MaxText.globals import DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
from MaxText.multihost_dataloading import MultiHostDataLoadIterator
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager
# pylint: disable=too-many-positional-arguments

CheckpointManager = ocp.CheckpointManager
CheckpointManagerOptions = ocp.CheckpointManagerOptions
Composite = ocp.args.Composite
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
EmergencyCheckpointManager = emergency_checkpoint_manager.CheckpointManager
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = emergency_checkpoint_manager.PersistentCheckpointOptions
EmergencyReplicatorCheckpointManager = emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager


def _load_full_state_from_path(
    path,
    abstract_unboxed_pre_state,
    enable_orbax_v1,
    checkpoint_conversion_fn,
    source_checkpoint_layout,
):
  """Load full state from checkpoint at specified path.

  Args:
    path: path to checkpoint
    abstract_unboxed_pre_state: an abstract state that Orbax matches type
      against.
    enable_orbax_v1: whether to use orbax v1 or the previously supported v0.
    checkpoint_conversion_fn: user-provided function to convert checkpoint to
      maxtext-supported state.
    source_checkpoint_layout: String representation of the checkpoint layout of
      the source checkpoint.

  Returns:
    The loaded state.
  """

  if enable_orbax_v1:
    if source_checkpoint_layout == "orbax":
      context = ocp_v1.Context(
          checkpoint_layout=ocp_v1.options.CheckpointLayout.ORBAX
      )
      with context:
        return ocp_v1.load_pytree(path, abstract_unboxed_pre_state)
    elif source_checkpoint_layout == "safetensors":
      context = ocp_v1.Context(
          checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS
      )
      with context:
        metadata = ocp_v1.pytree_metadata(path)
        simple_abstract_state = metadata.metadata
        shardings = sharding_utils.construct_maximal_shardings(
            simple_abstract_state
        )

        def combine_sharding(sds, shardings):
          return jax.ShapeDtypeStruct(
              shape=sds.shape, dtype=sds.dtype, sharding=shardings
          )

        sharded_abstract_state = jax.tree.map(
            combine_sharding, simple_abstract_state, shardings
        )
        pre_transformed_state = ocp_v1.load_pytree(
            path, sharded_abstract_state
        )
      state = checkpoint_conversion_fn(pre_transformed_state)
      return state
    else:
      raise ocp_v1.errors.InvalidLayoutError(
          f"Unknown checkpoint layout: {source_checkpoint_layout}"
      )
  else:
    # Original v0 logic.
    p = epath.Path(path)
    return ocp.StandardCheckpointer().restore(p, abstract_unboxed_pre_state)


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: None | str = "tfds",
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    max_to_keep: int = 5,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None

  max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

  if dataset_type == "grain":
    item_names = ("items", "iter")
  else:
    item_names = ("items",)

  # local storage checkpoint needs parent directory created
  p = epath.Path(checkpoint_dir)
  p.mkdir(exist_ok=True, parents=True)
  # we need to use ocdbt and zarr3 to control max file size in the checkpoint
  # omitting `iter` uses default handler for `iter`
  item_handlers = {"items": PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)}
  manager = CheckpointManager(
      p,
      item_names=item_names,
      item_handlers=item_handlers,
      options=CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
          max_to_keep = max_to_keep,
      ),
      logger=orbax_logger,
  )

  max_logging.log("Checkpoint manager created!")
  return manager


def create_orbax_emergency_checkpoint_manager(
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

  # Only create directories if running on GPUs as the previous
  # directory structure might be assumed by TPUs
  if global_mesh.devices.flatten()[0].platform == "gpu":
    # pylint: disable=protected-access
    local_checkpoint_dir = f"{local_checkpoint_dir}/{jax._src.distributed.global_state.process_id}"
    local_p = epath.Path(local_checkpoint_dir)
    persistent_p = epath.Path(persistent_checkpoint_dir)
    local_p.mkdir(exist_ok=True, parents=True)
    persistent_p.mkdir(exist_ok=True, parents=True)

  manager = EmergencyCheckpointManager(
      local_checkpoint_dir,
      epath.Path(persistent_checkpoint_dir),
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


def create_orbax_emergency_replicator_checkpoint_manager(
    local_checkpoint_dir: str,
    save_interval_steps: int,
    global_mesh: jax.sharding.Mesh,
):
  """Returns an emergency replicator checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency replicator checkpoint manager...")

  manager = EmergencyReplicatorCheckpointManager(
      epath.Path(local_checkpoint_dir),
      options=emergency_replicator_checkpoint_manager.ReplicatorCheckpointManagerOptions(
          save_interval_steps=save_interval_steps,
      ),
      global_mesh=global_mesh,
  )

  max_logging.log("Emergency replicator checkpoint manager created!")
  return manager


def replicator_error_handler(config: Any):
  """Replicator error handler to handle errors in replicator service."""
  if config.enable_emergency_checkpoint and config.use_replicator_service and config.local_checkpoint_directory:
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


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


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
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


def load_state_if_possible(
    checkpoint_manager: CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState,
    enable_single_replica_ckpt_restoring: bool | None = False,
    dataset_type: str | None = "tfds",
    step: int = -1,  # -1 means latest
    use_ocdbt=True,
    use_zarr3=True,
    enable_orbax_v1=False,
    checkpoint_conversion_fn=None,
    source_checkpoint_layout="orbax",
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint
      manager, load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint
      manager, load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
      with SingleReplicaArrayHandler
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    enable_orbax_v1: bool flag for enabling Orbax v1.
    checkpoint_conversion_fn: function for converting checkpoint to Orbax v1.
    source_checkpoint_layout: Optional checkpoint context to use for loading, 
    provided in string format with the default being "orbax".

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    step = checkpoint_manager.latest_step() if step < 0 else step
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      def map_to_pspec(data):
        if not enable_single_replica_ckpt_restoring:
          return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        replica_axis_index = 0
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

        return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(mesh, pspec),
            single_replica_sharding=single_replica_sharding,
            global_shape=data.shape,
            dtype=data.dtype,
        )

      if enable_single_replica_ckpt_restoring:
        array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
            replica_axis_index=0,
            broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
        )
        ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

      restore_args = jax.tree_util.tree_map(map_to_pspec, abstract_unboxed_pre_state)
      checkpoint_args = ocp.args.PyTreeRestore(item=abstract_unboxed_pre_state, restore_args=restore_args)

      match (checkpoint_manager, dataset_type, data_iterator):
        # Case 1: Matches if 'checkpoint_manager' is an instance of either EmergencyCheckpointManager
        # or EmergencyReplicatorCheckpointManager. The '_' indicates that 'dataset_type' and
        # 'data_iterator' can be any value and aren't used in this pattern.
        case (checkpoint_manager, _, _) if isinstance(
            checkpoint_manager,
            (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager),
        ):
          return (
              checkpoint_manager.restore(
                  step, args=Composite(state=checkpoint_args)
              ).state,
              None,
          )
        # Case 2: Matches if dataset type is "grain" and a specific checkpoint file exits for the iterator
        # exists within the checkpoint manager's directory for the given step.
        case (checkpoint_manager, dataset_type, data_iterator) if dataset_type == "grain" and data_iterator and (
            checkpoint_manager.directory / str(step) / "iter"
        ).exists():
          grain_iter = grain.PyGrainCheckpointRestore(data_iterator.local_iterator)
          return (checkpoint_manager.restore(step, args=Composite(items=checkpoint_args, iter=grain_iter)), None)
        # Case 3: Default/Fallback case.
        # This case acts as a wildcard ('_') and matches if none of the preceding cases were met.
        case _:
          return (checkpoint_manager.restore(step, args=Composite(items=checkpoint_args)), None)

  if load_parameters_from_path != "":
    restored_params = load_params_from_path(
        load_parameters_from_path,
        abstract_unboxed_pre_state.params,
        checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
    restored_state = _load_full_state_from_path(
        path=load_full_state_from_path,
        abstract_unboxed_pre_state=abstract_unboxed_pre_state,
        enable_orbax_v1=enable_orbax_v1,
        checkpoint_conversion_fn=checkpoint_conversion_fn,
        source_checkpoint_layout=source_checkpoint_layout,
    )
    return {"items": restored_state}, None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:  # pytype: disable=attribute-error
  """Setup checkpoint logger.
  Args:
    config
  Returns:
    CloudLogger
  """
  orbax_cloud_logger = None
  max_logging.log("Setting up checkpoint logger...")
  if config.enable_checkpoint_cloud_logger:
    logger_name = f"goodput_{config.run_name}"
    orbax_cloud_logger = ocp.logging.CloudLogger(
        options=ocp.logging.CloudLoggerOptions(job_name=config.run_name, logger_name=logger_name)
    )
    max_logging.log("Successfully set up checkpoint cloud logger.")

  return orbax_cloud_logger


def load_params_from_path(
    load_parameters_from_path, abstract_unboxed_params, checkpoint_storage_concurrent_gb, use_ocdbt=True, use_zarr3=True
):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")

  # *_concurrent_gb should be set for large models, the default is 96.
  max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=checkpoint_storage_concurrent_gb,
          save_concurrent_gb=checkpoint_storage_concurrent_gb,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
      )
  )

  # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
  # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
  # memory, we instead specify here that we are just restoring the params field of the checkpoint
  # (which itself may be a dictionary containing a key named 'params').
  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_unboxed_params)
  restored = ckptr.restore(
      epath.Path(load_parameters_from_path),
      item={"params": abstract_unboxed_params},
      transforms={},
      restore_args={"params": restore_args},
  )
  return restored["params"]


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  print(f"Saving quantized params checkpoint with use_ocdbt = {use_ocdbt} and use_zarr3 = {use_zarr3}")
  orbax_checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
  orbax_checkpointer.save(checkpoint_dir, {"params": params}, force=True)
  print(f"Quantized params checkpoint saved at: {checkpoint_dir}")


def maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None):
  """Save checkpoint if checkpointing is enabled."""
  if checkpoint_manager is None:
    return

  # Determine the effective step for saving a checkpoint.
  # If 'step' is not provided, this call is for a potential final checkpoint
  # and use the last completed step from the state.
  actual_step = (int(state.step) - 1) if step is None else int(step)

  # Determine if a checkpoint save should be forced, overriding the usual `config.checkpoint_period` logic.
  # This occurs if this function was called:
  # without an explicit 'step' (implying it's a checkpoint save for final step),
  # AND the 'actual_step' is a valid step,
  # AND it's not a step that would normally trigger a checkpoint save.
  force_ckpt_save = step is None and actual_step != -1 and (actual_step % config.checkpoint_period != 0)

  try:
    checkpoint_saved = save_checkpoint(checkpoint_manager, actual_step, state, config, data_iterator, force_ckpt_save)
    if checkpoint_saved:
      print_save_message(actual_step, config.async_checkpointing)
  except Exception as e:
    raise exceptions.StopTraining(f"Checkpointing failed. {str(e)}") from e

  # Wait for any pending checkpoint save to finish during preemption or final step save
  if force_ckpt_save or checkpoint_manager.reached_preemption(actual_step):
    checkpoint_manager.wait_until_finished()

  # Raise exception upon preemption
  if checkpoint_manager.reached_preemption(actual_step):
    raise exceptions.StopTraining("Job is preempted.")


def save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False):
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (
        force
        or (step % config.checkpoint_period == 0)
        or (config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0)
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = (
      config.checkpoint_storage_target_data_file_size_bytes if config else DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  )

  checkpoint_args = ocp.args.PyTreeSave(
      item=state,
      save_args=jax.tree.map(lambda _: ocp.SaveArgs(chunk_byte_size=chunk_byte_size), state),
      ocdbt_target_data_file_size=chunk_byte_size,
  )

  match (checkpoint_manager, config):
    case (checkpoint_manager, _) if isinstance(
        checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)
    ):
      replicator_error_handler(config)
      return checkpoint_manager.save(step, args=Composite(state=checkpoint_args), force=force)
    case (_, config) if config and config.dataset_type == "grain":
      grain_iter = grain.PyGrainCheckpointSave(data_iterator.local_iterator)
      return checkpoint_manager.save(step, args=Composite(items=checkpoint_args, iter=grain_iter), force=force)
    case _:
      return checkpoint_manager.save(step, args=Composite(items=checkpoint_args), force=force)
