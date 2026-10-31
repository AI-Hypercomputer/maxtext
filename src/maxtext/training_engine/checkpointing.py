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

import copy
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


# Handlers that dispatch writes from the TPU workers directly to storage. Anything else means
# Orbax fell back to the controller-side handler, which stages every shard through host RAM.
_PATHWAYS_PERSISTENCE_HANDLERS = ("CloudPathwaysArrayHandler", "PathwaysPersistenceArrayHandler")

_PATHWAYS_PERSISTENCE_REGISTERED = False


class PathwaysCheckpointingUnavailableError(RuntimeError):
  """Pathways persistence checkpointing was explicitly requested but could not be registered."""


def _maybe_register_pathways_persistence() -> None:
  """Registers Orbax Pathways persistence array handler, if applicable.

  When ENABLE_PATHWAYS_PERSISTENCE=1, delegates checkpoint saving directly
  from TPU workers to storage (GCS), bypassing host RAM staging.

  Setting the environment variable is an explicit request, so a failure to install the
  handler raises rather than degrading. The silent fallback is controller-side host
  staging, which can exhaust the controller's memory for large models, and it surfaces
  only once the first save is attempted -- long after the run looks healthy.

  Raises:
    PathwaysCheckpointingUnavailableError: If ENABLE_PATHWAYS_PERSISTENCE=1 but a
      persistence handler could not be registered for jax.Array.
  """
  global _PATHWAYS_PERSISTENCE_REGISTERED
  if _PATHWAYS_PERSISTENCE_REGISTERED:
    return
  if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") != "1":
    return

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
  except (ImportError, AttributeError, NotImplementedError) as e:
    raise PathwaysCheckpointingUnavailableError(
        "ENABLE_PATHWAYS_PERSISTENCE=1 explicitly requested Pathways persistence checkpointing, but "
        f"registering the Orbax Pathways array handler failed ({type(e).__name__}: {e}). Refusing to "
        "fall back to controller-side host-staged checkpointing. Unset ENABLE_PATHWAYS_PERSISTENCE to "
        "run without Pathways persistence."
    ) from e

  handler_name = type(handler).__name__
  if handler_name not in _PATHWAYS_PERSISTENCE_HANDLERS:
    raise PathwaysCheckpointingUnavailableError(
        "ENABLE_PATHWAYS_PERSISTENCE=1 explicitly requested Pathways persistence checkpointing, but "
        f"after registration jax.Array is handled by {handler_name}, not one of "
        f"{_PATHWAYS_PERSISTENCE_HANDLERS}. Aborting before the first save rather than staging every "
        "shard through controller host memory."
    )

  store = getattr(handler, "_array_metadata_store", None)
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

  _PATHWAYS_PERSISTENCE_REGISTERED = True


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

    Raises:
      ValueError: If ENABLE_PATHWAYS_PERSISTENCE=1, persistence is selected, and `checkpoint_dir`
        is not a gs:// URI.
      PathwaysCheckpointingUnavailableError: If ENABLE_PATHWAYS_PERSISTENCE=1 but the requested
        Pathways checkpointing handler could not be registered.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    # The step of the save most recently handed to Orbax, so a failure that surfaces later can
    # be attributed to it.
    self._in_flight_step: int | None = None
    if checkpoint_dir:
      if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") == "1" and not str(checkpoint_dir).startswith("gs://"):
        raise ValueError(
            "ENABLE_PATHWAYS_PERSISTENCE=1 dispatches persistence writes to every "
            f"pathways-worker; checkpoint_dir must be a gs:// URI, got {checkpoint_dir!r}."
        )
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
              # Deadline for the background half of a save (storage writes and finalization).
              async_options=ocp.AsyncOptions(timeout_secs=config.async_checkpointing_timeout_secs),
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
    """Waits for any ongoing async checkpoint save, attributing any failure to its step."""
    if self._checkpoint_manager is None:
      return
    step = self._in_flight_step
    try:
      self._checkpoint_manager.wait_until_finished()
    except Exception as e:
      self._in_flight_step = None
      if step is not None:
        logging.error(
            "The background half of the checkpoint save at step %s failed with %s: %s; re-raising it.",
            step,
            type(e).__name__,
            e,
        )
      raise
    self._in_flight_step = None

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

    # Orbax waits for the previous save only for a step it is going to save; mirror that, so a
    # step the interval policy declines does not block on the in-flight save. Waiting here also
    # surfaces a failed previous save before this one starts, and keeps `get_latest_step()` below
    # from reporting a step whose save has not been finalized.
    if force or self._checkpoint_manager.should_save(step):
      self.wait_until_finished()
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

    custom_metadata_payload = copy.deepcopy(custom_metadata) if custom_metadata is not None else None
    saved = self._checkpoint_manager.save(
        step=step,
        args=ocp.args.Composite(**save_args),
        custom_metadata=custom_metadata_payload,
        force=force,
    )
    if saved:
      self._in_flight_step = step
    return saved

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

    Raises:
      CheckpointRestoreError: If a checkpoint exists at `step` but cannot be restored, including
        when it was taken mid-step, or when an optimizer is supplied but the checkpoint holds no
        optimizer state.
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

    if checkpoint_state.optimizer is not None and "optimizer_state" not in metadata.item_metadata:
      # Resuming training with a freshly initialized optimizer would silently change the run.
      raise CheckpointRestoreError(
          f"Checkpoint at step {step} has no optimizer state; restore it with optimizer=None, or "
          "resume from a step that has the optimizer state."
      )
    if checkpoint_state.optimizer is not None:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)

      # The Pathways array handler ignores `memory_kind="pinned_host"` for physical placement
      # (allocating restored buffers in device HBM) while keeping `pinned_host` on the returned
      # array's sharding metadata. Request `device` placement explicitly so the restored sharding
      # matches physical placement; the engine moves host-offloaded state back after restoring.
      def _device_restore_target(leaf: Any) -> Any:
        sharding = getattr(leaf, "sharding", None)
        if getattr(sharding, "memory_kind", None) == "pinned_host":
          return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding.with_memory_kind("device"))
        return leaf

      optimizer_target = jax.tree.map(_device_restore_target, optimizer_state)
      restore_args["optimizer_state"] = ocp.args.PyTreeRestore(
          item=optimizer_target,
          restore_args=ocp.checkpoint_utils.construct_restore_args(target=optimizer_target),
      )

    custom_metadata = None
    if metadata and hasattr(metadata, "custom_metadata"):
      custom_metadata = metadata.custom_metadata

    try:
      restored_items = self._checkpoint_manager.restore(
          step=step,
          args=ocp.args.Composite(**restore_args),
      )
    except Exception as e:
      # Returning "no checkpoint" here would make the caller start a fresh run from the base
      # weights while the checkpoint it failed to read is still on disk.
      raise CheckpointRestoreError(
          f"Checkpoint at step {step} exists but could not be restored; refusing to silently start a "
          f"fresh run over it. {type(e).__name__}: {e}"
      ) from e

    if "model_params" in restored_items:
      nnx.update(checkpoint_state.model, restored_items["model_params"])
    if checkpoint_state.optimizer is not None and "optimizer_state" in restored_items:
      nnx.update(checkpoint_state.optimizer, restored_items["optimizer_state"])

    return step, checkpoint_state, custom_metadata

  def close(self) -> None:
    """Closes the checkpoint manager once the in-flight save, if any, has finished."""
    if self._checkpoint_manager:
      self.wait_until_finished()
      self._checkpoint_manager.close()
