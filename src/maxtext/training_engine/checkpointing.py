# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpointing utilities for MaxText training engine."""

from collections.abc import Mapping
import dataclasses
import os
from typing import Any

from absl import logging
from flax import nnx
import jax
from maxtext.configs import pyconfig
import orbax.checkpoint as ocp


@dataclasses.dataclass
class CheckpointState:
  """Container for the model and optimizer to checkpoint.

  Checkpoints are taken only between optimizer steps, so there is no partially accumulated
  gradient or metric state to carry.
  """

  model: nnx.Module
  optimizer: nnx.optimizer.Optimizer | None = None


class CheckpointRestoreError(RuntimeError):
  """A checkpoint exists but cannot be restored by this engine."""


_PATHWAYS_PERSISTENCE_REGISTERED = False


def _maybe_register_pathways_persistence() -> None:
  """Registers Orbax Pathways persistence array handler, if applicable.

  When ENABLE_PATHWAYS_PERSISTENCE=1, delegates checkpoint saving directly
  from TPU workers to storage (GCS), bypassing host RAM staging.
  """
  global _PATHWAYS_PERSISTENCE_REGISTERED
  if _PATHWAYS_PERSISTENCE_REGISTERED:
    return
  if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") != "1":
    return

  _PATHWAYS_PERSISTENCE_REGISTERED = True
  try:
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import orbax.checkpoint.pathways as ocp_pathways
    from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
    from orbax.checkpoint._src.serialization import type_handler_registry

    # pylint: enable=g-import-not-at-top,import-outside-toplevel

    ocp_pathways.register_type_handlers(
        use_single_replica_array_handler=False,
        checkpointing_impl=ocp_pathways.CheckpointingImpl.PERSISTENCE,
        # Preserve array metadata store required for pytrees with typed PRNG keys.
        array_metadata_store=array_metadata_store_lib.Store(),
    )

    handler = type_handler_registry.get_type_handler(jax.Array)
    handler_name = type(handler).__name__
    store = getattr(handler, "_array_metadata_store", None)
    if handler_name in ("CloudPathwaysArrayHandler", "PathwaysPersistenceArrayHandler"):
      if store is None:
        logging.error(
            "Registered %s but array metadata store is None; saving may fail on typed PRNG keys.",
            handler_name,
        )
      else:
        logging.info(
            "Registered Pathways persistence array handler (%s, store=%s); TPUs will write directly to storage.",
            handler_name,
            type(store).__name__,
        )
    else:
      logging.warning(
          "Pathways persistence registration fell back: jax.Array is handled by %s, not a persistence handler.",
          handler_name,
      )
  except (ImportError, AttributeError, ModuleNotFoundError, NotImplementedError) as e:
    logging.warning(
        "Pathways persistence requested but unavailable on this backend (%s). Falling back to host-staged checkpointing.",
        e,
    )


def _check_step_boundary_checkpoint(step: int, custom_metadata: Any, item_metadata: Any) -> None:
  """Raises if the checkpoint at `step` was taken mid-step, before anything is restored from it.

  Checkpoints written by earlier versions of this engine could hold a partially accumulated
  step: `micro_step_count > 0` in their metadata and an `accumulated_grads` item. This engine
  resumes only from step boundaries, so such a checkpoint is rejected rather than restored
  into the live model and optimizer.

  Args:
    step: The checkpoint's step, for the error message.
    custom_metadata: The checkpoint's custom metadata.
    item_metadata: The checkpoint's per-item metadata.

  Raises:
    CheckpointRestoreError: If the checkpoint was taken mid-step.
  """
  micro_step_count = custom_metadata.get("micro_step_count", 0) if isinstance(custom_metadata, Mapping) else 0
  if not isinstance(micro_step_count, int) or isinstance(micro_step_count, bool):
    micro_step_count = 0
  if micro_step_count > 0 or "accumulated_grads" in item_metadata:
    raise CheckpointRestoreError(
        f"Mid-step checkpoints are not supported; found micro_step_count={micro_step_count} > 0 at step {step}."
        if micro_step_count > 0
        else f"Mid-step checkpoints are not supported; the checkpoint at step {step} holds accumulated gradients."
    )


class CheckpointManager:
  """CheckpointManager wrapper for MaxText training engine."""

  def __init__(
      self,
      checkpoint_dir: str,
      config: pyconfig.HyperParameters,
  ) -> None:
    """Initializes the CheckpointManager.

    Args:
      checkpoint_dir: The root directory for saving checkpoints.
      config: The training configuration.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if checkpoint_dir:
      _maybe_register_pathways_persistence()

      # Use configured array format (e.g. use_ocdbt=False for Pathways).
      # Build a fresh handler per item as Orbax handlers carry per-item state.
      def _pytree_handler() -> ocp.PyTreeCheckpointHandler:
        return ocp.PyTreeCheckpointHandler(
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
            save_device_host_concurrent_gb=config.checkpoint_storage_device_host_concurrent_gb,
        )

      self._checkpoint_manager = ocp.CheckpointManager(
          directory=checkpoint_dir,
          options=ocp.CheckpointManagerOptions(
              save_interval_steps=config.checkpoint_period,
              max_to_keep=config.max_num_checkpoints_to_keep,
              enable_async_checkpointing=config.async_checkpointing,
          ),
          item_handlers={
              "model_params": _pytree_handler(),
              "optimizer_state": _pytree_handler(),
          },
      )

  def get_latest_step(self) -> int | None:
    """Returns the latest checkpoint step."""
    if self._checkpoint_manager:
      return self._checkpoint_manager.latest_step()
    return None

  def wait_until_finished(self) -> None:
    """Waits for any ongoing async checkpoint saves to finish."""
    if self._checkpoint_manager:
      self._checkpoint_manager.wait_until_finished()

  def save_checkpoint(
      self,
      step: int,
      checkpoint_state: CheckpointState,
      custom_metadata: Any = None,
      force: bool = False,
  ) -> bool:
    """Saves the model and optimizer state for the given step.

    Args:
      step: The step to save the params for.
      checkpoint_state: CheckpointState object containing model and optimizer.
      custom_metadata: Custom metadata to save with the checkpoint.
      force: Save even if Orbax's save-interval policy would skip this step.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping save.")
      return False

    custom_metadata = dict(custom_metadata) if custom_metadata else {}

    # Checkpoints are taken between steps only, so one already saved at this step holds the
    # same state.
    if self.get_latest_step() == step:
      logging.info("Checkpoint already saved at step %d, skipping save.", step)
      return False

    params = nnx.state(checkpoint_state.model)
    jax.block_until_ready(params)
    model_cp_args = ocp.args.PyTreeSave(
        item=params,
        save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params),
    )
    save_args = {"model_params": model_cp_args}

    if checkpoint_state.optimizer:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      jax.block_until_ready(optimizer_state)
      optimizer_cp_args = ocp.args.PyTreeSave(
          item=optimizer_state,
          save_args=jax.tree.map(lambda _: ocp.SaveArgs(), optimizer_state),
      )
      save_args["optimizer_state"] = optimizer_cp_args

    return self._checkpoint_manager.save(
        step=step,
        args=ocp.args.Composite(**save_args),
        custom_metadata=custom_metadata,
        force=force,
    )

  def restore_checkpoint(
      self,
      checkpoint_state: CheckpointState,
      step: int | None = None,
  ) -> tuple[int | None, CheckpointState, Any]:
    """Restores items from the checkpoint at the given step.

    Args:
      checkpoint_state: CheckpointState object containing model and optimizer.
      step: Optional step index to restore from.

    Returns:
      A tuple of (step, checkpoint_state, custom metadata).
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, checkpoint_state, None

    if step is None:
      step = self.get_latest_step()
      if step is None:
        logging.info("No checkpoint found, skipping restore.")
        return None, checkpoint_state, None

    metadata = self._checkpoint_manager.metadata(step)
    restore_args: dict[str, Any] = {}

    abstract_params = nnx.state(checkpoint_state.model)
    restore_args["model_params"] = ocp.args.PyTreeRestore(
        item=abstract_params,
        restore_args=ocp.checkpoint_utils.construct_restore_args(target=abstract_params),
    )

    if checkpoint_state.optimizer is not None and "optimizer_state" in metadata.item_metadata:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      restore_args["optimizer_state"] = ocp.args.PyTreeRestore(
          item=optimizer_state,
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              target=nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
          ),
      )

    custom_metadata = None
    if metadata and hasattr(metadata, "custom_metadata"):
      custom_metadata = metadata.custom_metadata
    _check_step_boundary_checkpoint(step, custom_metadata, getattr(metadata, "item_metadata", None) or {})

    try:
      restored_items = self._checkpoint_manager.restore(
          step=step,
          args=ocp.args.Composite(**restore_args),
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Failed to restore checkpoint: %s", e)
      return None, None, None

    if "model_params" in restored_items:
      nnx.update(checkpoint_state.model, restored_items["model_params"])
    if checkpoint_state.optimizer is not None and "optimizer_state" in restored_items:
      nnx.update(checkpoint_state.optimizer, restored_items["optimizer_state"])

    return step, checkpoint_state, custom_metadata

  def close(self) -> None:
    """Closes the checkpoint manager."""
    if self._checkpoint_manager:
      self._checkpoint_manager.close()
