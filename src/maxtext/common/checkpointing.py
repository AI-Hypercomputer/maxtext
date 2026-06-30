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

import contextlib
import datetime
import time
from typing import Any

from etils import epath
from flax import nnx
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
from maxtext.common import checkpoint_context
from maxtext.common import emergency_checkpointing
from maxtext.common import grain_utility
from maxtext.common import train_state_nnx
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline import synthetic_data_processing
from maxtext.utils import elastic_utils
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.checkpoint_conversion.utils.load_dynamic import load_safetensors_dynamic_state


from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.arrays import sharding as sharding_utils

PlaceHolderDataIterator = synthetic_data_processing.PlaceHolderDataIterator
MultiHostDataLoadIterator = multihost_dataloading.MultiHostDataLoadIterator


# Backward compatibility aliases for v0 emergency managers.
EmergencyCheckpointManager = emergency_checkpointing.CheckpointManager
EmergencyReplicatorCheckpointManager = emergency_checkpointing.ReplicatorCheckpointManager
create_orbax_emergency_checkpoint_manager = emergency_checkpointing.create_emergency_checkpoint_manager
create_orbax_emergency_replicator_checkpoint_manager = emergency_checkpointing.create_replicator_checkpoint_manager

# Union of v1 Checkpointer / the emergency factories return; used in type hints.
CheckpointManager = (
    ocp.training.Checkpointer
    | EmergencyCheckpointManager
    | EmergencyReplicatorCheckpointManager
)


def _load_linen_checkpoint_into_nnx(
    path,
    abstract_nnx_state,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
):
  """Restores a Linen-layout checkpoint into an NNX state (pure_nnx resume).

  Restores against a Linen-shape abstract, reshapes back via
  `from_linen_checkpoint_dict`, then fills NNX-only rngs/dropout with defaults.
  """
  max_logging.log(f"Restoring Linen-layout checkpoint into NNX state at {path}")
  nnx_abstract_pure = abstract_nnx_state.to_pure_dict()
  linen_abstract = train_state_nnx.to_linen_checkpoint_dict(nnx_abstract_pure)
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
  )
  with context:
    restored = ocp.load(epath.Path(path), linen_abstract)
  partial_nnx = train_state_nnx.from_linen_checkpoint_dict(restored)
  return train_state_nnx.populate_pure_dict_from_partial(nnx_abstract_pure, partial_nnx)


def _load_linen_params_into_nnx(
    path,
    nnx_params_abstract,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
):
  """Weight-only load of a Linen-layout checkpoint into an NNX params state.

  Reuses `to_linen_checkpoint_dict` (wrapping the params under `model`) to build
  the
  `params/params/...` restore target, then rebinds the restored weights into the
  NNX params Variables.
  """
  max_logging.log(f"Restoring Linen-layout params into NNX state at {path}")
  linen_abstract = train_state_nnx.to_linen_checkpoint_dict({"model": nnx_params_abstract.to_pure_dict()})
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
  )
  with context:
    restored = ocp.load(epath.Path(path), linen_abstract)
  return train_state_nnx.rebuild_nnx_with_values(nnx_params_abstract, restored["params"]["params"])


def _load_full_state_from_path(
    path,
    abstract_unboxed_pre_state,
    checkpoint_conversion_fn,
    source_checkpoint_layout,
    checkpoint_storage_concurrent_gb,
    use_ocdbt,
    use_zarr3,
    enable_single_replica_ckpt_restoring: bool = False,
):
  """Load full state from a checkpoint at the specified path.

  Args:
    path: path to checkpoint
    abstract_unboxed_pre_state: an abstract state that Orbax matches type
      against.
    checkpoint_conversion_fn: user-provided function to convert a checkpoint to
      a maxtext-supported state.
    source_checkpoint_layout: String representation of the checkpoint's on-disk
      layout, "orbax" or "safetensors".
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    use_ocdbt: Whether to use OCDBT format.
    use_zarr3: Whether to use Zarr3 format.

  Returns:
    The loaded state.
  """
  # pure_nnx checkpoints are stored in the Linen on-disk layout; reshape to NNX.
  if isinstance(abstract_unboxed_pre_state, nnx.State):
    return _load_linen_checkpoint_into_nnx(
        path,
        abstract_unboxed_pre_state,
        checkpoint_storage_concurrent_gb,
        use_ocdbt,
        use_zarr3,
    )

  if source_checkpoint_layout == "orbax":
    context = checkpoint_context.build_context(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.ORBAX,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )
    with context:
      return ocp.load(path, abstract_unboxed_pre_state)

  if source_checkpoint_layout == "safetensors":
    context = checkpoint_context.build_context(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
        checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS,
        enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
    )
    with context:
      metadata = ocp.metadata(path)
      simple_abstract_state = metadata.metadata
      shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

      def combine_sharding(sds, shardings):
        return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

      sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
      pre_transformed_state = ocp.load(path, sharded_abstract_state)
    return checkpoint_conversion_fn(pre_transformed_state)

  raise ocp.errors.InvalidLayoutError(f"Unknown checkpoint layout: {source_checkpoint_layout}")


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: None | str = None,
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    enable_continuous_checkpointing: bool = False,
    max_num_checkpoints_to_keep: int = 10,
    checkpoint_storage_concurrent_gb: int = 96,
    enable_single_controller: bool = False,
    colocated_python_checkpointing: bool = False,
    enable_single_replica_ckpt_restoring: bool = False,
    enable_autocheckpoint: bool = False,
    todelete_subdir: str | None = None,
    todelete_full_path: str | None = None,
    ocdbt_target_data_file_size_bytes: int | None = None,
):
  """Returns an Orbax v1 training ``Checkpointer``, or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None

  # TODO: b/529622681 - Remove deprecated settings.
  if orbax_logger is not None:
    max_logging.warning(
        "Cloud logging (enable_checkpoint_cloud_logger) is disabled because"
        " Orbax v1 now configures its own logger internally. This config"
        " setting is ignored and will be removed."
    )
  if dataset_type is not None:
    max_logging.warning(
        "Specifying dataset_type upon checkpointer creation is deprecated and"
        " will be removed soon, this is now handled dynamically by Orbax"
        " Checkpointer."
    )

  max_logging.log(f"Creating checkpointer with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

  validated_path = gcs_utils.mkdir_and_check_permissions(checkpoint_dir)

  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      ocdbt_target_data_file_size_bytes=ocdbt_target_data_file_size_bytes,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      async_timeout_secs=(
          int(datetime.timedelta(minutes=60).total_seconds()) if enable_continuous_checkpointing else None
      ),
      todelete_full_path=todelete_full_path,
      todelete_subdir=todelete_subdir,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
      colocated_python_checkpointing=(enable_single_controller and colocated_python_checkpointing),
      partial_load=True,
  )

  manager = ocp.training.Checkpointer(
      validated_path,
      context=context,
      save_decision_policy=checkpoint_context.build_save_decision_policy(
          save_interval_steps=save_interval_steps,
          enable_continuous_checkpointing=enable_continuous_checkpointing,
          enable_autocheckpoint=enable_autocheckpoint,
      ),
      preservation_policy=checkpoint_context.build_preservation_policy(
          max_num_checkpoints_to_keep=max_num_checkpoints_to_keep,
      ),
  )
  manager.use_async = use_async

  max_logging.log("Checkpoint manager created!")
  return manager


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


def latest_step(checkpoint_manager):
  """Latest saved step or None, across the v0 emergency manager and the v1 Checkpointer."""
  if isinstance(checkpoint_manager, (emergency_checkpointing.CheckpointManager, emergency_checkpointing.ReplicatorCheckpointManager)):
    return checkpoint_manager.latest_step()
  latest = checkpoint_manager.latest
  return latest.step if latest is not None else None


def wait_until_finished(checkpoint_manager):
  """Blocks until pending saves finish, across the v0 emergency manager and the v1 Checkpointer."""
  if emergency_checkpointing.is_emergency_manager(checkpoint_manager):
    checkpoint_manager.wait_until_finished()
  else:
    checkpoint_manager.wait()


def reached_preemption(step: int) -> bool:
  """Whether a preemption sync point has been reached at ``step`` (manager-agnostic)."""
  return multihost_utils.reached_preemption_sync_point(step)


def is_structural_or_shape_mismatch(e: Exception) -> bool:
  """Helper to check if an exception is likely a PyTree structure or shape mismatch."""
  if not isinstance(e, (ValueError, TypeError)):
    return False
  msg = str(e).lower()
  mismatch_keywords = [
      "mismatch",
      "structure",
      "shape",
      "tree",
      "leaf",
      "leaves",
      "paths matched",
      "shapedtypestruct",
      "invalid type",
  ]
  return any(kw in msg for kw in mismatch_keywords)


def _assert_no_shaped_dtype_struct(pytree):
  """Asserts that there are no jax.ShapeDtypeStruct leaves in the restored pytree."""
  if isinstance(pytree, jax.ShapeDtypeStruct):
    raise ValueError(
        "Some parameters in the restored state remained as ShapeDtypeStruct"
        f" (indicating structure mismatch): {pytree}."
    )

  if hasattr(pytree, "keys") and hasattr(pytree, "__getitem__"):
    for k in pytree.keys():
      _assert_no_shaped_dtype_struct(pytree[k])
  elif isinstance(pytree, (list, tuple)):
    for v in pytree:
      _assert_no_shaped_dtype_struct(v)
  else:
    leaves = jax.tree_util.tree_leaves(pytree)
    if len(leaves) == 1 and leaves[0] is pytree:
      return
    for leaf in leaves:
      _assert_no_shaped_dtype_struct(leaf)


@contextlib.contextmanager
def handle_checkpoint_mismatch(context_name: str, path: str):
  """Context manager to intercept PyTree/shape mismatches and raise descriptive errors."""
  try:
    yield
  except Exception as e:
    if is_structural_or_shape_mismatch(e):
      raise ValueError(
          f"Failed to {context_name} from {path}. This is often caused by a"
          " mismatch in the 'scan_layers' configuration (stacked vs unstacked)"
          " between your current execution command and the saved checkpoint."
          f" Original error: {e}"
      ) from e
    raise


def load_state_if_possible(
    checkpoint_manager: CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState | nnx.State,
    enable_single_replica_ckpt_restoring: bool | None = False,
    dataset_type: str | None = "tfds",
    step: int = -1,  # -1 means latest
    use_ocdbt=True,
    use_zarr3=True,
    enable_orbax_v1=False,
    checkpoint_conversion_fn=None,
    source_checkpoint_layout="orbax",
    expansion_factor_real_data: int = -1,
    maxtext_config: Any | None = None,
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
    checkpoint_conversion_fn: function for converting a safetensors checkpoint
      to a maxtext-supported state.
    source_checkpoint_layout: Optional checkpoint context to use for loading,
      provided in string format with the default being "orbax".

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  # Orbax Checkpointer expects a root dir (e.g. checkpoints/) or a step dir (e.g. checkpoints/0/).
  # V0 consumers often passed paths ending in `/items` or `/params` directly referencing the Pytree.
  # V1 auto-discovers these buffers via checkpointable_name. Strip them from the paths.
  if load_parameters_from_path:
    load_parameters_from_path = load_parameters_from_path.removesuffix("/items").removesuffix("/params")
  if load_full_state_from_path:
    load_full_state_from_path = load_full_state_from_path.removesuffix("/items").removesuffix("/params")

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    step = latest_step(checkpoint_manager) if step < 0 else step
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      # pure_nnx saves in the Linen on-disk layout; reshape it back into the NNX state.
      # (Emergency managers use their own restore path below.)
      if isinstance(abstract_unboxed_pre_state, nnx.State) and not emergency_checkpointing.is_emergency_manager(checkpoint_manager):
        assert isinstance(checkpoint_manager, ocp.training.Checkpointer)
        nnx_abstract_pure = abstract_unboxed_pre_state.to_pure_dict()
        linen_abstract = train_state_nnx.to_linen_checkpoint_dict(
            nnx_abstract_pure
        )
        restored_linen = checkpoint_manager.load(
            step, linen_abstract, checkpointable_name="items"
        )
        partial_nnx = train_state_nnx.from_linen_checkpoint_dict(restored_linen)
        restored_nnx = train_state_nnx.populate_pure_dict_from_partial(
            nnx_abstract_pure, partial_nnx
        )
        return ({"items": restored_nnx}, None)

      checkpoint_path = str(checkpoint_manager.directory / str(step))
      with handle_checkpoint_mismatch("restore checkpoint", checkpoint_path):
        # Case 1: Matches if 'checkpoint_manager' is an instance of either EmergencyCheckpointManager
        # or EmergencyReplicatorCheckpointManager.
        if emergency_checkpointing.is_emergency_manager(checkpoint_manager):
          return (
              emergency_checkpointing.restore(
                  checkpoint_manager, step, abstract_unboxed_pre_state
              ),
              None,
          )
        # Case 2: Matches if dataset type is "grain" and the data iterator is not a
        # PlaceHolderDataIterator and a specific checkpoint file exists for the iterator
        assert isinstance(checkpoint_manager, ocp.training.Checkpointer)
        abstract_checkpointables = {"items": abstract_unboxed_pre_state}
        if (
            dataset_type == "grain"
            and data_iterator
            and not isinstance(data_iterator, PlaceHolderDataIterator)
            and (checkpoint_manager.directory / str(step) / "iter").exists()
        ):
          assert isinstance(checkpoint_manager, ocp.training.Checkpointer)
          abstract_checkpointables["iter"] = grain_utility.for_restore(
              checkpoint_manager, step, data_iterator, expansion_factor_real_data
          )
        # Load the checkpointables from the checkpoint.
        return (
            checkpoint_manager.load_checkpointables(
                step, abstract_checkpointables
            ),
            None,
        )

  if source_checkpoint_layout == "safetensors_dynamic":
    path = load_parameters_from_path or load_full_state_from_path
    max_logging.log(
        f"Dynamic On-the-Fly Formatting: Loading SafeTensors from {path}"
    )
    # Delegate to custom MaxText dynamic conversion module
    return load_safetensors_dynamic_state(
        path, abstract_unboxed_pre_state, maxtext_config
    )
  elif load_parameters_from_path != "":
    if isinstance(abstract_unboxed_pre_state, nnx.State):
      _, params, _ = nnx.split(abstract_unboxed_pre_state.model, nnx.Param, ...)
    else:
      params = abstract_unboxed_pre_state.params

    with handle_checkpoint_mismatch("load parameters", load_parameters_from_path):
      restored_params = load_params_from_path(
          load_parameters_from_path,
          params,
          checkpoint_storage_concurrent_gb,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
          enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
      )
      _assert_no_shaped_dtype_struct(restored_params)
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
    with handle_checkpoint_mismatch("load full state", load_full_state_from_path):
      restored_state = _load_full_state_from_path(
          path=load_full_state_from_path,
          abstract_unboxed_pre_state=abstract_unboxed_pre_state,
          checkpoint_conversion_fn=checkpoint_conversion_fn,
          source_checkpoint_layout=source_checkpoint_layout,
          checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
          enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
      )
      _assert_no_shaped_dtype_struct(restored_state)
    return {"items": restored_state}, None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:  # pytype: disable=attribute-error
  """DEPRECATED: Setup checkpoint logger."""
  # TODO: b/529622681 - Remove this config option entirely.
  if config.enable_checkpoint_cloud_logger:
    max_logging.warning(
        "Cloud logging (enable_checkpoint_cloud_logger) is disabled because"
        " Orbax v1 now configures its own logger internally. This config"
        " setting is ignored and will be removed."
    )
  return None


def load_params_from_path(
    load_parameters_from_path,
    abstract_unboxed_params,
    checkpoint_storage_concurrent_gb,
    use_ocdbt=True,
    use_zarr3=True,
    enable_single_replica_ckpt_restoring: bool = False,
):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")

  # NNX target: the on-disk checkpoint is in Linen layout; reshape it into the
  # NNX params state.
  if isinstance(abstract_unboxed_params, nnx.State):
    return _load_linen_params_into_nnx(
        load_parameters_from_path,
        abstract_unboxed_params,
        checkpoint_storage_concurrent_gb,
        use_ocdbt,
        use_zarr3,
    )

  # Memory optimization: restore only the "params" key (the checkpoint may also hold opt_state/step);
  # partial_load drops the rest. The abstract carries shape/dtype/sharding directly.
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt,
      use_zarr3=use_zarr3,
      checkpoint_storage_concurrent_gb=checkpoint_storage_concurrent_gb,
      partial_load=True,
      enable_single_replica_ckpt_restoring=enable_single_replica_ckpt_restoring,
  )
  with context:
    restored = ocp.load(
        epath.Path(load_parameters_from_path),
        {"params": abstract_unboxed_params},
        checkpointable_name="items",
    )
  return restored["params"]


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  max_logging.log(
      f"Saving params checkpoint with use_ocdbt={use_ocdbt} and"
      f" use_zarr3={use_zarr3}"
  )
  context = checkpoint_context.build_context(
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3
  )
  with context:
    ocp.save(
        checkpoint_dir, {"params": params}, checkpointable_name="items", overwrite=True
    )
  max_logging.log(f"Params checkpoint saved at: {checkpoint_dir}")


def maybe_save_checkpoint(
    checkpoint_manager, state, config, data_iterator, step=None
):
  """Save checkpoint if checkpointing is enabled."""
  if checkpoint_manager is None:
    return

  # Determine the effective step for saving a checkpoint.
  # If 'step' is not provided, this call is for a potential final checkpoint
  # and use the last completed step from the state.
  if step is not None:
    actual_step = int(step)
  else:
    if config.pure_nnx:
      # Under DiLoCo the step lives on the DiLoCoTrainState; otherwise on the optimizer.
      actual_step = int(state.step if config.enable_diloco else state.optimizer.step) - 1
    else:
      # Linen TrainState has .step attribute
      actual_step = int(state.step) - 1

  if latest_step(checkpoint_manager) == actual_step:
    max_logging.log(f"Checkpoint for step {actual_step} already exists, skipping save.")
    return

  if config.pure_nnx:
    # Save in the Linen on-disk layout so pure_nnx and Linen checkpoints are interchangeable.
    if config.enable_diloco:
      # DiLoCoTrainState: persist the synchronized global model (outer params).
      # The per-replica inner optimizer / outer-momentum state is not checkpointed.
      step_value = state.step.get_value() if hasattr(state.step, "get_value") else state.step
      state = train_state_nnx.to_linen_checkpoint_dict({"model": state.params, "optimizer": {"step": step_value}})
    else:
      state = train_state_nnx.to_linen_checkpoint_dict(state.to_pure_dict())

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
      if config.elastic_enabled:
        elastic_utils.maybe_elastic_scale_up(config, checkpoint_manager)
  except elastic_utils.manager.ScaleUpSignalError as e:
    if config.elastic_enabled:
      max_logging.log(f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except jax.errors.JaxRuntimeError as e:
    if config.elastic_enabled:
      max_logging.log(f"Elastic event detected, letting exception bubble up: {e}")
      raise
    else:
      raise exceptions.StopTraining("Job is preempted.") from e
  except Exception as e:
    raise exceptions.StopTraining(f"Checkpointing failed. {str(e)}") from e

  # Wait for any pending checkpoint save to finish during preemption or final step save
  is_emergency = isinstance(checkpoint_manager, (emergency_checkpointing.CheckpointManager, emergency_checkpointing.ReplicatorCheckpointManager))
  manager_preempted = checkpoint_manager.reached_preemption(actual_step) if is_emergency else reached_preemption(actual_step)

  if force_ckpt_save or manager_preempted:
    wait_until_finished(checkpoint_manager)

  # Raise exception upon preemption
  if manager_preempted:
    raise exceptions.StopTraining("Job is preempted.")


def save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False):
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    is_emergency = isinstance(checkpoint_manager, (emergency_checkpointing.CheckpointManager, emergency_checkpointing.ReplicatorCheckpointManager))
    manager_preempted = checkpoint_manager.reached_preemption(step) if is_emergency else reached_preemption(step)
    if (
        force
        or (step % config.checkpoint_period == 0 and not config.enable_continuous_checkpointing)
        or (config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0)
        or (config.enable_autocheckpoint and manager_preempted)
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

  # Emergency / replicator managers keep the v0 save path.
  if emergency_checkpointing.is_emergency_manager(checkpoint_manager):
    return emergency_checkpointing.save(checkpoint_manager, step, state, config, force)

  # Standard path: Orbax v1 Checkpointer. Storage/chunk options live on the manager's Context.
  checkpointables = {"items": state}
  if config and config.dataset_type == "grain" and not isinstance(data_iterator, PlaceHolderDataIterator):
    checkpointables["iter"] =  grain_utility.for_save(checkpoint_manager, step, data_iterator, config.expansion_factor_real_data)
  return checkpoint_manager.save_checkpointables(step, checkpointables, force=force)
