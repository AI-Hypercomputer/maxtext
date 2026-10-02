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


_PERSISTENCE = "persistence"
_COLOCATED_PYTHON = "colocated_python"
_PATHWAYS_CHECKPOINTING_IMPLS = (_PERSISTENCE, _COLOCATED_PYTHON)

# Which impl this process registered, or None if registration has not succeeded yet. Orbax
# registers type handlers process-globally, so this doubles as the conflict detector.
_REGISTERED_IMPL: str | None = None

# Handlers that actually dispatch writes off the controller under `persistence`. Anything else
# means Orbax fell back to the controller-side handler, which stages every shard through
# proxy-pod host RAM and OOMs at 397B scale.
_PATHWAYS_PERSISTENCE_HANDLERS = ("CloudPathwaysArrayHandler", "PathwaysPersistenceArrayHandler")


class PathwaysCheckpointingUnavailableError(RuntimeError):
  """Pathways checkpointing was explicitly requested but could not be registered."""


def _maybe_register_pathways_persistence(impl_name: str = _PERSISTENCE) -> None:
  """Registers the Orbax Pathways array handler for `impl_name`, if applicable.

  When ENABLE_PATHWAYS_PERSISTENCE=1, delegates checkpoint saving directly
  from TPU workers to storage (GCS), bypassing host RAM staging.

  Setting the environment variable is an explicit request, so a failure to install
  the handler raises rather than degrading. The silent fallback is controller-side
  host staging, which exhausts proxy-pod memory at 397B scale, and it surfaces only
  once the first save is attempted -- long after the run looks healthy.

  Args:
    impl_name: Which Pathways implementation to register; one of
      `_PATHWAYS_CHECKPOINTING_IMPLS`. `persistence` is the shipped default;
      `colocated_python` additionally requires a version-matched sidecar container.

  Raises:
    ValueError: If `impl_name` is not a known implementation.
    PathwaysCheckpointingUnavailableError: If ENABLE_PATHWAYS_PERSISTENCE=1 but the
      requested handler could not be registered for jax.Array, or if a different
      implementation is already registered in this process.
  """
  global _REGISTERED_IMPL
  if impl_name not in _PATHWAYS_CHECKPOINTING_IMPLS:
    raise ValueError(
        f"unknown pathways_checkpointing_impl {impl_name!r}; expected one of {_PATHWAYS_CHECKPOINTING_IMPLS}."
    )
  if _REGISTERED_IMPL is not None:
    if _REGISTERED_IMPL != impl_name:
      raise PathwaysCheckpointingUnavailableError(
          f"Orbax type handlers are registered process-globally and this process already "
          f"registered {_REGISTERED_IMPL!r}; cannot also serve {impl_name!r}. The impl affects "
          "every save and restore, so mixing them would silently apply one mode to the other's "
          "checkpoints."
      )
    return
  if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") != "1":
    return

  try:
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import orbax.checkpoint.pathways as ocp_pathways
    from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
    from orbax.checkpoint._src.serialization import type_handler_registry

    # pylint: enable=g-import-not-at-top,import-outside-toplevel

    if impl_name == _COLOCATED_PYTHON:
      impl = ocp_pathways.CheckpointingImpl.COLOCATED_PYTHON
    else:
      impl = ocp_pathways.CheckpointingImpl.PERSISTENCE

    ocp_pathways.register_type_handlers(
        use_single_replica_array_handler=False,
        checkpointing_impl=impl,
        # Preserve array metadata store required for pytrees with typed PRNG keys.
        array_metadata_store=array_metadata_store_lib.Store(),
    )

    handler = type_handler_registry.get_type_handler(jax.Array)
    handler_name = type(handler).__name__
  except (ImportError, AttributeError, ModuleNotFoundError, NotImplementedError) as e:
    raise PathwaysCheckpointingUnavailableError(
        f"ENABLE_PATHWAYS_PERSISTENCE=1 explicitly requested Pathways checkpointing "
        f"(impl={impl_name!r}), but registering the Orbax Pathways array handler failed "
        f"({type(e).__name__}: {e}). Refusing to fall back to controller-side host-staged "
        "checkpointing, which exhausts proxy-pod memory at 397B scale. Unset "
        "ENABLE_PATHWAYS_PERSISTENCE to run without Pathways checkpointing."
    ) from e

  if impl_name == _COLOCATED_PYTHON:
    # COLOCATED_PYTHON is the only impl in this Orbax build that attaches a dispatcher, so
    # has_dispatcher() distinguishes it from NO_DISPATCHER -- the controller-side fallback --
    # without reaching into Orbax internals. Checking the class name alone would not: the
    # NO_DISPATCHER fallback yields an ArrayHandler too, just without a dispatcher.
    registered_as_requested = handler_name == "ArrayHandler" and handler.has_dispatcher()
    expected = "an ArrayHandler with a dispatcher attached"
  else:
    registered_as_requested = handler_name in _PATHWAYS_PERSISTENCE_HANDLERS
    expected = f"one of {_PATHWAYS_PERSISTENCE_HANDLERS}"

  if not registered_as_requested:
    raise PathwaysCheckpointingUnavailableError(
        f"ENABLE_PATHWAYS_PERSISTENCE=1 explicitly requested Pathways checkpointing "
        f"(impl={impl_name!r}), but after registration jax.Array is handled by {handler_name}, "
        f"not {expected}. This is the controller-side staging path that exhausts proxy-pod "
        "memory at 397B scale; aborting before the first save rather than OOM-ing mid-run."
    )

  store = getattr(handler, "_array_metadata_store", None)
  if store is None:
    logging.error(
        "Registered %s but array metadata store is None; saving may fail on typed PRNG keys.",
        handler_name,
    )
  else:
    logging.info(
        "Registered Pathways array handler (impl=%s, %s, store=%s); "
        "TPUs will write directly to storage.",
        impl.name,
        handler_name,
        type(store).__name__,
    )

  # Latch only on success, so a failed attempt re-raises on the next construction
  # instead of being swallowed by the early return above.
  _REGISTERED_IMPL = impl_name


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
      if (
          os.environ.get("ENABLE_PATHWAYS_PERSISTENCE") == "1"
          and not str(checkpoint_dir).startswith("gs://")
      ):
        raise ValueError(
            "ENABLE_PATHWAYS_PERSISTENCE=1 dispatches persistence writes to every "
            f"pathways-worker; checkpoint_dir must be a gs:// URI, got {checkpoint_dir!r}."
        )
      _maybe_register_pathways_persistence(getattr(config, "pathways_checkpointing_impl", _PERSISTENCE))

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
              "accumulated_metrics": _pytree_handler(),
              "accumulated_grads": _pytree_handler(),
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
    if "accumulated_metrics" in restored_items:
      checkpoint_state.accumulated_metrics = restored_items["accumulated_metrics"]
    if "accumulated_grads" in restored_items:
      checkpoint_state.accumulated_grads = restored_items["accumulated_grads"]

    return step, checkpoint_state, custom_metadata

  def close(self) -> None:
    """Closes the checkpoint manager."""
    if self._checkpoint_manager:
      self._checkpoint_manager.close()
