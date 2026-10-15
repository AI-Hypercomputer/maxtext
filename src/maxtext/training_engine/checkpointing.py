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
from typing import Any, List

from absl import logging
from etils import epath
from flax import nnx
import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
import orbax.checkpoint as ocp


@dataclasses.dataclass
class CheckpointState:
  """Container for model, optimizer, accumulated metrics and intra-step states to checkpoint."""

  model: nnx.Module
  optimizer: nnx.optimizer.Optimizer | None = None
  accumulated_metrics: List[abstract_engine.MetricsBuffer] | None = None
  accumulated_grads: Any = None
  # How many micro-batches of `step` are folded into `accumulated_grads`. 0 for a complete
  # step, whose gradients have already been applied and discarded.
  micro_step_count: int = 0


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
    self._config = config
    self._checkpoint_dir = checkpoint_dir
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if checkpoint_dir:
      self._checkpoint_manager = self._create_orbax_checkpoint_manager(checkpoint_dir)

  def _create_orbax_checkpoint_manager(
      self,
      checkpoint_dir: str,
  ) -> ocp.CheckpointManager:
    """Creates an Orbax CheckpointManager for the given directory."""
    _maybe_register_pathways_persistence()

    # Use configured array format (e.g. use_ocdbt=False for Pathways).
    # Build a fresh handler per item as Orbax handlers carry per-item state.
    def _pytree_handler() -> ocp.PyTreeCheckpointHandler:
      return ocp.PyTreeCheckpointHandler(
          use_ocdbt=self._config.checkpoint_storage_use_ocdbt,
          use_zarr3=self._config.checkpoint_storage_use_zarr3,
          save_device_host_concurrent_gb=self._config.checkpoint_storage_device_host_concurrent_gb,
      )

    return ocp.CheckpointManager(
        directory=checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=self._config.checkpoint_period,
            max_to_keep=self._config.max_num_checkpoints_to_keep,
            enable_async_checkpointing=self._config.async_checkpointing,
        ),
        item_handlers={
            "model_params": _pytree_handler(),
            "optimizer_state": _pytree_handler(),
            "accumulated_metrics": _pytree_handler(),
            "accumulated_grads": _pytree_handler(),
        },
    )

  def _resolve_restore_directory(self, directory: str) -> str:
    """Resolves a checkpoint restore directory, handling nested MaxText paths."""
    run_name = getattr(self._config, "run_name", None)
    candidates = []
    if run_name:
      candidates.append(os.path.join(directory, str(run_name), "checkpoints"))
    candidates.append(os.path.join(directory, "checkpoints"))
    for candidate in candidates:
      try:
        if epath.Path(candidate).is_dir():
          return candidate
      except OSError:
        if os.path.isdir(candidate):
          return candidate
    return directory

  def _resolve_restore_source(self, directory: str, step: int | None) -> tuple[str | None, int | None]:
    """Resolves the directory and step that `restore_checkpoint` reads from.

    Args:
      directory: The requested checkpoint restore directory.
      step: The requested step, or None for the latest step.

    Returns:
      A tuple (directory, step). `directory` is None when the restore should read from the configured
      checkpoint directory.

    Raises:
      ValueError: If `directory` resolves to the configured checkpoint directory and `step` is not its
        latest step.
    """
    resolved_dir = self._resolve_restore_directory(directory)
    root_latest_step = self.get_latest_step()
    if self._checkpoint_dir and epath.Path(resolved_dir) == epath.Path(self._checkpoint_dir):
      if step is not None and step != root_latest_step:
        raise ValueError(
            f"Cannot restore step {step} from {directory}: it is also the checkpoint root directory, whose"
            f" latest step is {root_latest_step}. Checkpoints saved after restoring an older step would be"
            " mixed with the newer ones already there. Use a different checkpoint directory, or leave the"
            " step unset to resume from the latest step."
        )
      logging.info(
          "Checkpoint restore directory %s is the same as the root checkpoint directory; resuming from its"
          " latest step (%s).",
          directory,
          root_latest_step,
      )
      return None, root_latest_step
    if root_latest_step is not None:
      logging.info(
          "Root checkpoint directory already has checkpoint at step %d; resuming from root directory instead"
          " of restore directory %s.",
          root_latest_step,
          directory,
      )
      return None, root_latest_step
    source = "the latest checkpoint" if step is None else f"step {step}"
    if self._checkpoint_dir:
      logging.info(
          "Restoring %s from restore directory %s; future checkpoints will be written to root directory %s.",
          source,
          resolved_dir,
          self._checkpoint_dir,
      )
    else:
      logging.info(
          "Restoring %s from restore directory %s; no checkpoint directory is configured, so checkpoint saving"
          " is disabled.",
          source,
          resolved_dir,
      )
    return resolved_dir, step

  def get_latest_step(self) -> int | None:
    """Returns the latest checkpoint step."""
    if self._checkpoint_manager:
      return self._checkpoint_manager.latest_step()
    return None

  def wait_until_finished(self) -> None:
    """Waits for any ongoing async checkpoint saves to finish."""
    if self._checkpoint_manager:
      self._checkpoint_manager.wait_until_finished()

  def get_saved_micro_step_count(self, step: int) -> int:
    """Returns how far into `step` the checkpoint already on disk got.

    Args:
      step: The step whose saved checkpoint should be inspected.

    Returns:
      0 if that checkpoint covers a complete step, otherwise the number of micro-batches
      accumulated into it.
    """
    if self._checkpoint_manager is None:
      return 0
    try:
      metadata = self._checkpoint_manager.metadata(step)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Could not read metadata for step %d, treating it as complete: %s", step, e)
      return 0
    custom_metadata = getattr(metadata, "custom_metadata", None)
    if not isinstance(custom_metadata, Mapping):
      return 0
    saved = custom_metadata.get("micro_step_count", 0)
    return saved if isinstance(saved, int) else 0

  def _supersedes_saved_checkpoint(self, step: int, micro_step_count: int) -> bool:
    """Returns whether a new checkpoint is more complete than the one saved at `step`.

    A complete step is never superseded: once the optimizer update for `step` has been
    checkpointed there is nothing more to record for it. A partial one is superseded by a
    complete step, and by a partial one that got further through the same step.

    Args:
      step: The step both checkpoints belong to.
      micro_step_count: The new checkpoint's progress through `step`.

    Returns:
      Whether the new checkpoint should replace the saved one.
    """
    saved_micro_step_count = self.get_saved_micro_step_count(step)
    if saved_micro_step_count == 0:
      return False
    return micro_step_count == 0 or micro_step_count > saved_micro_step_count

  def _delete_saved_step(self, step: int) -> None:
    """Deletes the checkpoint at `step` to make room for a more complete one.

    Orbax refuses to write a step that already exists, so superseding one means removing it
    first. An async save for this step may still be in flight, so drain before deleting.

    Args:
      step: The step to delete.
    """
    logging.info("Deleting intra-step checkpoint at step %d so a more complete one can replace it.", step)
    self._checkpoint_manager.wait_until_finished()
    self._checkpoint_manager.delete(step)

  def save_checkpoint(
      self,
      step: int,
      checkpoint_state: CheckpointState,
      custom_metadata: Any = None,
      **kwargs,
  ) -> bool:
    """Saves the params for the given step along with optional intra-step state.

    Args:
      step: The step to save the params for.
      checkpoint_state: CheckpointState object containing model, optimizer, and
        optional intra_step_state.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping save.")
      return False

    # Record micro_step_count on every checkpoint, complete or not, so that a later save at the same step
    # can tell whether it supersedes what is already on disk.
    custom_metadata = dict(custom_metadata) if custom_metadata else {}
    custom_metadata["micro_step_count"] = checkpoint_state.micro_step_count

    # A checkpoint already exists at this step. Skip, unless this one is more complete --
    # the case that matters is a step resumed from an intra-step checkpoint and then run to
    # completion, whose finished state would otherwise never reach disk.
    if self.get_latest_step() == step:
      if not self._supersedes_saved_checkpoint(step, checkpoint_state.micro_step_count):
        logging.info(
            "Checkpoint already saved at step %d, skipping save.",
            step,
        )
        return False
      self._delete_saved_step(step)
      # Orbax's save-interval policy declines a step it has already saved, so the
      # replacement has to be forced through.
      kwargs["force"] = True

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

    if checkpoint_state.accumulated_metrics:
      jax.block_until_ready(checkpoint_state.accumulated_metrics)
      metrics_cp_args = ocp.args.PyTreeSave(
          item=checkpoint_state.accumulated_metrics,
          save_args=jax.tree.map(
              lambda _: ocp.SaveArgs(),
              checkpoint_state.accumulated_metrics,
          ),
      )
      save_args["accumulated_metrics"] = metrics_cp_args

    if checkpoint_state.accumulated_grads:
      jax.block_until_ready(checkpoint_state.accumulated_grads)
      grads_cp_args = ocp.args.PyTreeSave(
          item=checkpoint_state.accumulated_grads,
          save_args=jax.tree.map(
              lambda _: ocp.SaveArgs(),
              checkpoint_state.accumulated_grads,
          ),
      )
      save_args["accumulated_grads"] = grads_cp_args

    return self._checkpoint_manager.save(
        step=step,
        args=ocp.args.Composite(**save_args),
        custom_metadata=custom_metadata,
        **kwargs,
    )

  def restore_checkpoint(
      self,
      checkpoint_state: CheckpointState,
      step: int | None = None,
      directory: str | None = None,
  ) -> tuple[int | None, CheckpointState, Any]:
    """Restores items from the checkpoint at the given step.

    Args:
      checkpoint_state: CheckpointState object containing model and optimizer.
      step: Optional step index to restore from.
      directory: Optional directory to restore checkpoints from. Defaults to the
        manager's configured checkpoint directory. If it resolves to the
        configured checkpoint directory, `step` must be None or the latest step,
        and the latest checkpoint is restored. Otherwise, if the configured
        checkpoint directory already contains checkpoints (e.g., after a
        preemption restart), its latest checkpoint takes precedence over
        `directory` and `step`.

    Returns:
      A tuple of (step, checkpoint_state, custom metadata).

    Raises:
      ValueError: If `directory` resolves to the configured checkpoint directory
        and `step` is not its latest step.
    """
    temp_manager: ocp.CheckpointManager | None = None
    if directory:
      directory, step = self._resolve_restore_source(directory, step)
    if directory:
      temp_manager = self._create_orbax_checkpoint_manager(directory)
      checkpoint_manager = temp_manager
    else:
      checkpoint_manager = self._checkpoint_manager

    if checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, checkpoint_state, None

    try:
      if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
          logging.info("No checkpoint found, skipping restore.")
          return None, checkpoint_state, None

      metadata = checkpoint_manager.metadata(step)
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

      if "accumulated_metrics" in metadata.item_metadata:
        restore_args["accumulated_metrics"] = ocp.args.PyTreeRestore()

      if "accumulated_grads" in metadata.item_metadata:
        accumulated_grads_target = nnx.state(checkpoint_state.model, nnx.Param)
        restore_args["accumulated_grads"] = ocp.args.PyTreeRestore(
            item=accumulated_grads_target,
            restore_args=ocp.checkpoint_utils.construct_restore_args(target=accumulated_grads_target),
        )

      custom_metadata = None
      if metadata and hasattr(metadata, "custom_metadata"):
        custom_metadata = metadata.custom_metadata

      try:
        restored_items = checkpoint_manager.restore(
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
      if "accumulated_metrics" in restored_items:
        checkpoint_state.accumulated_metrics = restored_items["accumulated_metrics"]
      if "accumulated_grads" in restored_items:
        checkpoint_state.accumulated_grads = restored_items["accumulated_grads"]

      return step, checkpoint_state, custom_metadata
    finally:
      if temp_manager is not None:
        temp_manager.close()

  def close(self) -> None:
    """Closes the checkpoint manager."""
    if self._checkpoint_manager:
      self._checkpoint_manager.close()
