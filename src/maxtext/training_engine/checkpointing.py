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
from maxtext.common import checkpoint_context
from maxtext.configs import pyconfig
from orbax.checkpoint import v1 as ocp


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


# Handler that dispatches writes from the TPU workers directly to storage. Anything else means
# Orbax would use the controller-side handler, which stages every shard through host RAM.
_PATHWAYS_PERSISTENCE_HANDLERS = ("CloudPathwaysArrayHandler",)


class PathwaysCheckpointingUnavailableError(RuntimeError):
  """Pathways persistence checkpointing was explicitly requested but is not available."""


def _pathways_persistence_requested() -> bool:
  """Returns whether ENABLE_PATHWAYS_PERSISTENCE=1 asks for Pathways persistence checkpointing."""
  # Upstream Orbax reads the same environment variable in
  # `orbax.checkpoint._src.multihost.multihost.is_cloud_pathways_persistence_enabled()`,
  # called from `orbax.checkpoint._src.serialization.pathways_handler_registry`.
  return os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") == "1"


def _resolve_array_handler(context: ocp.Context) -> Any:
  """Returns the array handler Orbax will save and restore `jax.Array` leaves with under `context`."""
  # pylint: disable-next=g-import-not-at-top,import-outside-toplevel
  from orbax.checkpoint.experimental.v1._src.serialization import registration

  return registration.get_array_handler(context)


def _verify_pathways_persistence(context: ocp.Context) -> None:
  """Checks that `context` resolves to a Pathways persistence array handler.

  With ENABLE_PATHWAYS_PERSISTENCE=1, checkpoint writes go from the TPU workers directly to
  storage (GCS), bypassing host RAM staging. Orbax picks the array handler from the context when
  it saves; off a Pathways backend it silently uses the controller-side handler instead.

  Setting the environment variable is an explicit request, so a handler that cannot be resolved,
  or is not a persistence handler, raises here, at construction, rather than degrading. The
  silent fallback is controller-side host staging, which can exhaust the controller's memory for
  large models, and it would surface only once the first save is attempted -- long after the run
  looks healthy.

  Args:
    context: The Orbax context the checkpointer saves and restores with.

  Raises:
    PathwaysCheckpointingUnavailableError: If `context` does not resolve to a Pathways
      persistence handler.
  """
  try:
    handler = _resolve_array_handler(context)
  except (ImportError, AttributeError, NotImplementedError) as e:
    raise PathwaysCheckpointingUnavailableError(
        "ENABLE_PATHWAYS_PERSISTENCE=1 requires a Pathways persistence array handler, but "
        f"resolving it failed ({type(e).__name__}: {e})."
    ) from e

  handler_name = type(handler).__name__
  if handler_name not in _PATHWAYS_PERSISTENCE_HANDLERS:
    raise PathwaysCheckpointingUnavailableError(
        "ENABLE_PATHWAYS_PERSISTENCE=1 requires a Pathways persistence array handler "
        f"({_PATHWAYS_PERSISTENCE_HANDLERS}), got {handler_name}."
    )

  store = getattr(handler, "_array_metadata_store", None)
  logging.info(
      "Using Pathways persistence array handler (%s, store=%s); TPUs will write directly to storage.",
      handler_name,
      type(store).__name__,
  )


class CheckpointManager:
  """Orbax `training.Checkpointer` wrapper for MaxText training engine."""

  def __init__(
      self,
      checkpoint_dir: str,
      config: pyconfig.HyperParameters,
  ) -> None:
    """Initializes the CheckpointManager.

    Args:
      checkpoint_dir: The root directory for saving checkpoints. Empty disables checkpointing.
      config: The training configuration.

    Raises:
      ValueError: If ENABLE_PATHWAYS_PERSISTENCE=1 and `checkpoint_dir` is not a gs:// URI, or if
        `checkpoint_storage_use_ocdbt` or `checkpoint_storage_use_zarr3` is enabled.
      PathwaysCheckpointingUnavailableError: If ENABLE_PATHWAYS_PERSISTENCE=1 but no Pathways
        persistence handler is available.
    """
    self._checkpoint_manager: ocp.training.Checkpointer | None = None
    self._use_async = bool(config.async_checkpointing)
    # The step of the save most recently handed to Orbax, so a failure that surfaces later can
    # be attributed to it.
    self._in_flight_step: int | None = None
    if not checkpoint_dir:
      return

    enable_pathways_persistence = _pathways_persistence_requested()
    if enable_pathways_persistence:
      if config.checkpoint_storage_use_ocdbt or config.checkpoint_storage_use_zarr3:
        raise ValueError(
            "ENABLE_PATHWAYS_PERSISTENCE=1 is incompatible with OCDBT and Zarr3; set "
            "checkpoint_storage_use_ocdbt=False and checkpoint_storage_use_zarr3=False, got "
            f"use_ocdbt={config.checkpoint_storage_use_ocdbt}, "
            f"use_zarr3={config.checkpoint_storage_use_zarr3}."
        )
      if not str(checkpoint_dir).startswith("gs://"):
        raise ValueError(
            "ENABLE_PATHWAYS_PERSISTENCE=1 dispatches persistence writes to every "
            f"pathways-worker; checkpoint_dir must be a gs:// URI, got {checkpoint_dir!r}."
        )
    # TODO(igorts): Support config.colocated_python_checkpointing when single-controller Pathways is used.
    context = checkpoint_context.build_context(
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
        checkpoint_storage_device_host_concurrent_gb=config.checkpoint_storage_device_host_concurrent_gb,
        # Deadline for the background half of a save (storage writes and finalization).
        async_checkpointing_timeout_secs=config.async_checkpointing_timeout_secs,
        enable_pathways_persistence=enable_pathways_persistence,
    )
    if enable_pathways_persistence:
      _verify_pathways_persistence(context)
    policies = ocp.training.save_decision_policies
    save_policies = [
        policies.FixedIntervalPolicy(config.checkpoint_period),
        policies.PreemptionCheckpointingPolicy(),
    ]
    if config.save_checkpoint_on_start:
      save_policies.append(policies.InitialSavePolicy())
    self._checkpoint_manager = ocp.training.Checkpointer(
        checkpoint_dir,
        context=context,
        # Saves every `checkpoint_period` steps, on a step where Orbax has detected a preemption,
        # and (when `save_checkpoint_on_start` is True) on the first step of a run with no checkpoint.
        save_decision_policy=policies.AnySavePolicy(save_policies),
        preservation_policy=checkpoint_context.build_preservation_policy(
            max_to_keep=config.max_num_checkpoints_to_keep,
        ),
    )

  def get_latest_step(self) -> int | None:
    """Returns the latest checkpoint step."""
    if self._checkpoint_manager is None:
      return None
    latest = self._checkpoint_manager.latest
    return latest.step if latest is not None else None

  def wait_until_finished(self) -> None:
    """Waits for any ongoing async checkpoint save, attributing any failure to its step."""
    if self._checkpoint_manager is None:
      return
    step = self._in_flight_step
    try:
      self._checkpoint_manager.wait()
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

  def should_save(self, step: int, force: bool = False) -> bool:
    """Returns whether `step` should be checkpointed under the current save policy."""
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping save.")
      return False
    return force or self._checkpoint_manager.should_save(step)

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
      force: Save even if the save-interval policy would skip this step.

    Returns:
      Whether the checkpoint was saved (or, for an async save, started).
    """
    # Decide before anything else whether this step is saved at all, so a step the interval
    # policy declines neither blocks on the in-flight save nor waits for the state to be ready.
    if not self.should_save(step, force=force):
      return False
    # Surfaces a failed previous save before this one starts, and finalizes it so the step
    # listing below includes it.
    self.wait_until_finished()
    # Checkpoints are taken between steps only, so one already saved at this step holds the
    # same state. This includes a step before the latest, e.g. after resuming from it.
    if any(checkpoint.step == step for checkpoint in self._checkpoint_manager.checkpoints):
      logging.info("Checkpoint already saved at step %d, skipping save.", step)
      return False

    params = nnx.state(checkpoint_state.model)
    jax.block_until_ready(params)
    checkpointables: dict[str, Any] = {"model_params": params}
    if checkpoint_state.optimizer is not None:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      jax.block_until_ready(optimizer_state)
      checkpointables["optimizer_state"] = optimizer_state

    custom_metadata_payload = copy.deepcopy(custom_metadata)
    if self._use_async:
      saved = (
          self._checkpoint_manager.save_checkpointables_async(
              step, checkpointables, custom_metadata=custom_metadata_payload, force=True
          )
          is not None
      )
      if saved:
        self._in_flight_step = step
    else:
      saved = self._checkpoint_manager.save_checkpointables(
          step, checkpointables, custom_metadata=custom_metadata_payload, force=True
      )
    return saved

  def restore_checkpoint(
      self,
      checkpoint_state: CheckpointState,
      step: int | None = None,
  ) -> tuple[int | None, Any]:
    """Restores items from the checkpoint at the given step into `checkpoint_state`.

    Args:
      checkpoint_state: CheckpointState object containing model and optimizer, mutated in place.
      step: Optional step index to restore from.

    Returns:
      A tuple of (step, custom metadata), or (None, None) when checkpointing is disabled or no
      checkpoint exists.

    Raises:
      CheckpointRestoreError: If a checkpoint exists at `step` but cannot be restored, or when
        an optimizer is supplied but the checkpoint holds no optimizer state.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, None

    if step is None:
      step = self.get_latest_step()
      if step is None:
        logging.info("No checkpoint found, skipping restore.")
        return None, None

    try:
      metadata = self._checkpoint_manager.checkpointables_metadata(step)
    except Exception as e:
      raise CheckpointRestoreError(
          f"Checkpoint at step {step} exists but its metadata could not be read; refusing to silently "
          f"start a fresh run over it. {type(e).__name__}: {e}"
      ) from e
    # The names of the checkpointables saved at `step`, e.g. "model_params" and "optimizer_state".
    saved_checkpointables = metadata.metadata or {}

    if checkpoint_state.optimizer is not None and "optimizer_state" not in saved_checkpointables:
      # Resuming training with a freshly initialized optimizer would silently change the run.
      raise CheckpointRestoreError(
          f"Checkpoint at step {step} has no optimizer state; restore it with optimizer=None, or "
          "resume from a step that has the optimizer state."
      )

    # Each target leaf carries the shape, dtype and sharding the restored array is placed with.
    abstract_checkpointables: dict[str, Any] = {"model_params": nnx.state(checkpoint_state.model)}
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

      abstract_checkpointables["optimizer_state"] = jax.tree.map(_device_restore_target, optimizer_state)

    try:
      restored_items = self._checkpoint_manager.load_checkpointables(step, abstract_checkpointables)
    except Exception as e:
      # Returning "no checkpoint" here would make the caller start a fresh run from the base
      # weights while the checkpoint it failed to read is still on disk.
      raise CheckpointRestoreError(
          f"Checkpoint at step {step} exists but could not be restored; refusing to silently start a "
          f"fresh run over it. {type(e).__name__}: {e}"
      ) from e

    nnx.update(checkpoint_state.model, restored_items["model_params"])
    if checkpoint_state.optimizer is not None:
      nnx.update(checkpoint_state.optimizer, restored_items["optimizer_state"])

    return step, metadata.custom_metadata

  def close(self) -> None:
    """Closes the checkpoint manager once the in-flight save, if any, has finished."""
    if self._checkpoint_manager is None:
      return
    try:
      self.wait_until_finished()
    finally:
      self._checkpoint_manager.close()
