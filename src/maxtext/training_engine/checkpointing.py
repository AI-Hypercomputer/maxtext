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
import concurrent.futures
import dataclasses
import os
from typing import Any, List

from absl import logging
from flax import nnx
import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
import orbax.checkpoint as ocp

# How many times `CheckpointManager._delete_saved_step` removes a step directory that is still on
# disk after Orbax deleted the step.
_MAX_STEP_DIR_REMOVALS = 3


@dataclasses.dataclass
class CheckpointState:
  """Container for model, optimizer, accumulated metrics and intra-step states to checkpoint."""

  model: nnx.Module
  optimizer: nnx.optimizer.Optimizer | None = None
  accumulated_metrics: List[abstract_engine.MetricsBuffer] | None = None
  accumulated_grads: Any = None
  accumulated_denominator: Any = None
  # How many micro-batches of `step` are folded into `accumulated_grads`. 0 for a complete
  # step, whose gradients have already been applied and discarded.
  micro_step_count: int = 0


@jax.jit
def _jit_copy_tree(tree: Any) -> Any:
  """Clones an entire pytree of JAX arrays in a single fused JIT dispatch to break aliasing."""
  return jax.tree.map(jax.numpy.copy, tree)


def _tree_nbytes(tree: Any) -> int:
  """Returns the total byte size of all jax.Array leaves in `tree`."""
  return sum(
      int(leaf.nbytes) for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array) and hasattr(leaf, "nbytes")
  )


def _has_pinned_host_memory(shd: Any) -> bool:
  """Returns whether `shd` is on a device memory kind that supports `pinned_host` staging."""
  if shd is None or getattr(shd, "memory_kind", None) == "pinned_host":
    return False
  for dev in getattr(shd, "addressable_devices", ()):
    for mem in getattr(dev, "addressable_memories", lambda: ())():
      if getattr(mem, "kind", None) == "pinned_host":
        return True
  return False


def _isolate_for_async_save(
    tree: Any,
    *,
    already_isolated: bool = False,
    max_pinned_host_bytes: int | None = None,
) -> Any:
  """Isolates a pytree of device arrays from subsequent `donate_argnums` reuse without host-stalling."""
  if already_isolated:
    return tree
  leaves = jax.tree.leaves(tree)
  first_shd = next(
      (
          getattr(leaf, "sharding", None)
          for leaf in leaves
          if isinstance(leaf, jax.Array) and getattr(leaf, "sharding", None) is not None
      ),
      None,
  )
  within_pinned_budget = (
      max_pinned_host_bytes is None or max_pinned_host_bytes <= 0 or _tree_nbytes(tree) <= max_pinned_host_bytes
  )
  has_pinned = not _PATHWAYS_PERSISTENCE_REGISTERED and within_pinned_budget and _has_pinned_host_memory(first_shd)
  if has_pinned:
    with jax.transfer_guard("allow"):
      pinned_targets = jax.tree.map(
          lambda leaf: leaf.sharding.with_memory_kind("pinned_host")
          if isinstance(leaf, jax.Array) and getattr(leaf, "sharding", None) is not None
          else None,
          tree,
      )
      return jax.device_put(tree, pinned_targets, may_alias=False)
  return _jit_copy_tree(tree)


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
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    self._saved_step_micro_counts: dict[int, int] = {}
    self._async_checkpointing: bool = bool(getattr(config, "async_checkpointing", False))
    self._async_save_executor: concurrent.futures.ThreadPoolExecutor | None = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) if checkpoint_dir and self._async_checkpointing else None
    )
    self._pending_async_futures: list[concurrent.futures.Future[Any]] = []
    concurrent_gb = getattr(config, "checkpoint_storage_device_host_concurrent_gb", None)
    self._max_pinned_host_bytes: int | None = (
        int(concurrent_gb * (1024**3)) if concurrent_gb and concurrent_gb > 0 else None
    )
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
              "accumulated_metrics": _pytree_handler(),
              "accumulated_grads": _pytree_handler(),
              "accumulated_denominator": _pytree_handler(),
          },
      )

  def wait_for_inflight_save_staging(self) -> None:
    """Waits for any background intra-step `.save()` staging call to hand off to Orbax."""
    while self._pending_async_futures:
      fut = self._pending_async_futures.pop(0)
      fut.result()

  def get_latest_step(self) -> int | None:
    """Returns the latest checkpoint step."""
    if self._checkpoint_manager:
      latest = self._checkpoint_manager.latest_step()
      if self._saved_step_micro_counts:
        max_tracked = max(self._saved_step_micro_counts)
        return max_tracked if latest is None else max(latest, max_tracked)
      return latest
    return None

  def wait_until_finished(self) -> None:
    """Waits for any ongoing async checkpoint saves to finish."""
    self.wait_for_inflight_save_staging()
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
    if step in self._saved_step_micro_counts:
      return self._saved_step_micro_counts[step]
    if self._checkpoint_manager is None:
      return 0
    # Orbax counts an async save as the latest step from the moment it starts, but cannot read its
    # metadata until it finishes. The fallback below would then take a partial checkpoint for a
    # complete one, and a step-boundary save would decline to replace it.
    self._checkpoint_manager.wait_until_finished()
    try:
      metadata = self._checkpoint_manager.metadata(step)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Could not read metadata for step %d, treating it as complete: %s", step, e)
      return 0
    custom_metadata = getattr(metadata, "custom_metadata", None)
    if not isinstance(custom_metadata, Mapping):
      return 0
    saved = custom_metadata.get("micro_step_count", 0)
    count = saved if isinstance(saved, int) else 0
    self._saved_step_micro_counts[step] = count
    return count

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
    self.wait_until_finished()
    try:
      self._checkpoint_manager.delete(step)
    except FileNotFoundError:
      # Recorded, but never written: Orbax declined the save for `step` once it ran.
      logging.info("No checkpoint on disk at step %d; nothing to delete.", step)
    if isinstance(self._checkpoint_manager, ocp.CheckpointManager) and jax.process_index() == 0:
      # On GCS, Orbax delete() can leave step files behind; remove any leftover directory while
      # tolerating FileNotFoundError.
      step_dir = ocp.step.build_step_path(self._checkpoint_manager.directory, ocp.step.standard_name_format(), step)
      for _ in range(_MAX_STEP_DIR_REMOVALS):
        if not step_dir.exists():
          break
        logging.warning("%s is still on disk after deleting step %d; removing it again.", step_dir, step)
        try:
          step_dir.rmtree()
        except FileNotFoundError:
          pass  # An object an earlier attempt already removed; check the directory again.
    self._saved_step_micro_counts.pop(step, None)

  def should_save_checkpoint(self, step: int, micro_step_count: int, force: bool = False) -> bool:
    """Returns whether `save_checkpoint` would proceed at `step` without modifying disk state.

    Advisory for step-boundary saves: Orbax re-evaluates its save policy when `save()` runs.

    Args:
      step: The step the checkpoint would be saved under.
      micro_step_count: How far into `step` the checkpoint would be; 0 for a complete step.
      force: Whether the save bypasses Orbax's save-interval policy.

    Returns:
      Whether a save at `step` would proceed.
    """
    if self._checkpoint_manager is None:
      return False
    latest = self.get_latest_step()
    # Reject steps behind the latest checkpoint even when forced or intra-step.
    if latest is not None and step < latest:
      logging.warning("Not saving a checkpoint at step %d: it is behind the latest checkpoint, step %d.", step, latest)
      return False
    # If a checkpoint already exists at `step`, save only if the new one is more complete.
    if latest == step or step in self._saved_step_micro_counts:
      return self._supersedes_saved_checkpoint(step, micro_step_count)
    # `checkpoint_period` applies only to unforced step-boundary saves.
    if force or micro_step_count > 0 or not isinstance(self._checkpoint_manager, ocp.CheckpointManager):
      return True
    return self._checkpoint_manager.should_save(step)

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

    grads_already_isolated = bool(kwargs.pop("grads_already_isolated", False))

    # Record micro_step_count on every checkpoint, complete or not, so that a later save at the same step
    # can tell whether it supersedes what is already on disk.
    custom_metadata = dict(custom_metadata) if custom_metadata else {}
    custom_metadata["micro_step_count"] = checkpoint_state.micro_step_count

    if not self.should_save_checkpoint(step, checkpoint_state.micro_step_count, force=kwargs.get("force", False)):
      logging.info(
          "Skipping checkpoint at step %d: already saved, behind the latest one, or not due under the save interval.",
          step,
      )
      return False
    if self.get_latest_step() == step or step in self._saved_step_micro_counts:
      # This one supersedes the checkpoint already at `step`, and Orbax refuses to write a step
      # that exists.
      self._delete_saved_step(step)
      # Orbax's save-interval policy declines a step it has already saved, so the
      # replacement has to be forced through.
      kwargs["force"] = True

    is_intra_step = checkpoint_state.micro_step_count > 0
    if is_intra_step:
      # Not subject to the save interval (see `should_save_checkpoint`), so Orbax must not apply it.
      kwargs["force"] = True

    params = nnx.state(checkpoint_state.model)
    if not is_intra_step:
      jax.block_until_ready(params)
    model_cp_args = ocp.args.PyTreeSave(
        item=params,
        save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params),
    )
    save_args = {"model_params": model_cp_args}

    if checkpoint_state.optimizer:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      if not is_intra_step:
        jax.block_until_ready(optimizer_state)
      optimizer_cp_args = ocp.args.PyTreeSave(
          item=optimizer_state,
          save_args=jax.tree.map(lambda _: ocp.SaveArgs(), optimizer_state),
      )
      save_args["optimizer_state"] = optimizer_cp_args

    if checkpoint_state.accumulated_metrics:
      if not is_intra_step:
        jax.block_until_ready(checkpoint_state.accumulated_metrics)
      metrics_cp_args = ocp.args.PyTreeSave(
          item=checkpoint_state.accumulated_metrics,
          save_args=jax.tree.map(
              lambda _: ocp.SaveArgs(),
              checkpoint_state.accumulated_metrics,
          ),
      )
      save_args["accumulated_metrics"] = metrics_cp_args

    grads_to_save = checkpoint_state.accumulated_grads
    denom_to_save = checkpoint_state.accumulated_denominator

    if is_intra_step:
      # Isolate `accumulated_grads` and `accumulated_denominator` from subsequent
      # `fwd_bwd_accum` buffer donation (`donate_argnums=(3, 4)`) asynchronously on the
      # device stream without blocking the Python host thread.
      if grads_to_save is not None and denom_to_save is not None and not grads_already_isolated:
        grads_to_save, denom_to_save = _isolate_for_async_save(
            (grads_to_save, denom_to_save),
            already_isolated=False,
            max_pinned_host_bytes=self._max_pinned_host_bytes,
        )
      else:
        if grads_to_save is not None:
          grads_to_save = _isolate_for_async_save(
              grads_to_save,
              already_isolated=grads_already_isolated,
              max_pinned_host_bytes=self._max_pinned_host_bytes,
          )
        if denom_to_save is not None:
          denom_to_save = _isolate_for_async_save(
              denom_to_save,
              already_isolated=False,
              max_pinned_host_bytes=self._max_pinned_host_bytes,
          )
    else:
      if grads_to_save is not None:
        jax.block_until_ready(grads_to_save)

    if grads_to_save:
      save_args["accumulated_grads"] = ocp.args.PyTreeSave(
          item=grads_to_save,
          save_args=jax.tree.map(
              lambda _: ocp.SaveArgs(),
              grads_to_save,
          ),
      )

    if denom_to_save is not None:
      # Wrap in a dict so the leaf has a non-empty Orbax array name for Pathways persistence.
      denom_item = {"value": denom_to_save}
      save_args["accumulated_denominator"] = ocp.args.PyTreeSave(
          item=denom_item,
          save_args=jax.tree.map(
              lambda _: ocp.SaveArgs(),
              denom_item,
          ),
      )

    composite_args = ocp.args.Composite(**save_args)
    if (
        is_intra_step
        and self._async_save_executor is not None
        and isinstance(self._checkpoint_manager, ocp.CheckpointManager)
    ):
      # Offload Orbax's synchronous D2H staging and metadata creation to the background thread.
      micro_step_count = checkpoint_state.micro_step_count
      self._saved_step_micro_counts[step] = micro_step_count

      def _bg_save() -> bool:
        saved = False
        try:
          saved = self._checkpoint_manager.save(
              step=step,
              args=composite_args,
              custom_metadata=custom_metadata,
              **kwargs,
          )
          return saved
        except Exception:
          logging.exception("Background checkpoint save failed at step %d.", step)
          raise
        finally:
          # Forget a save that never reached disk, so `get_latest_step` does not report it. Only
          # this save's own entry is removed, in case a newer save at `step` has recorded its own.
          if not saved and self._saved_step_micro_counts.get(step) == micro_step_count:
            self._saved_step_micro_counts.pop(step, None)

      fut = self._async_save_executor.submit(_bg_save)
      self._pending_async_futures.append(fut)
      return True

    self.wait_for_inflight_save_staging()
    saved = self._checkpoint_manager.save(
        step=step,
        args=composite_args,
        custom_metadata=custom_metadata,
        **kwargs,
    )
    if saved:
      self._saved_step_micro_counts[step] = checkpoint_state.micro_step_count
    return saved

  def restore_checkpoint(
      self,
      checkpoint_state: CheckpointState,
      step: int | None = None,
      mesh: jax.sharding.Mesh | None = None,
  ) -> tuple[int | None, CheckpointState, Any]:
    """Restores items from the checkpoint at the given step.

    Args:
      checkpoint_state: CheckpointState object containing model and optimizer.
      step: Optional step index to restore from.
      mesh: Optional mesh to restore the metrics history and accumulated denominator onto,
        replicated. Defaults to the mesh the model's parameters are on.

    Returns:
      A tuple of (step, checkpoint_state, custom metadata).
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, checkpoint_state, None

    # A checkpoint that an async save is still writing cannot be read yet.
    self.wait_until_finished()
    if step is None:
      step = self.get_latest_step()
      if step is None:
        logging.info("No checkpoint found, skipping restore.")
        return None, checkpoint_state, None

    metadata = self._checkpoint_manager.metadata(step)
    restore_args: dict[str, Any] = {}

    abstract_params = nnx.state(checkpoint_state.model)
    param_mesh = mesh or next(
        (
            getattr(getattr(leaf, "sharding", None), "mesh", None)
            for leaf in jax.tree.leaves(abstract_params)
            if getattr(getattr(leaf, "sharding", None), "mesh", None) is not None
        ),
        None,
    )
    replicated_sharding = (
        jax.sharding.NamedSharding(param_mesh, jax.sharding.PartitionSpec()) if param_mesh is not None else None
    )
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
      metrics_meta = metadata.item_metadata["accumulated_metrics"]
      # Unlike the other items, the metrics history has no restore target to take shardings from,
      # and Pathways persistence cannot restore an array without one. Its arrays come back replicated.
      if replicated_sharding is not None and metrics_meta and jax.tree.leaves(metrics_meta):
        restore_args["accumulated_metrics"] = ocp.args.PyTreeRestore(
            restore_args=ocp.checkpoint_utils.construct_restore_args(
                metrics_meta,
                sharding_tree=jax.tree.map(lambda _: replicated_sharding, metrics_meta),
            ),
        )
      else:
        restore_args["accumulated_metrics"] = ocp.args.PyTreeRestore()

    if "accumulated_grads" in metadata.item_metadata:
      accumulated_grads_target = nnx.state(checkpoint_state.model, nnx.Param)
      restore_args["accumulated_grads"] = ocp.args.PyTreeRestore(
          item=accumulated_grads_target,
          restore_args=ocp.checkpoint_utils.construct_restore_args(target=accumulated_grads_target),
      )

    if "accumulated_denominator" in metadata.item_metadata:
      denom_leaf_target = (
          jax.ShapeDtypeStruct((), jax.numpy.float32, sharding=replicated_sharding)
          if replicated_sharding is not None
          else jax.ShapeDtypeStruct((), jax.numpy.float32)
      )
      denom_target = {"value": denom_leaf_target}
      restore_args["accumulated_denominator"] = ocp.args.PyTreeRestore(
          item=denom_target,
          restore_args=ocp.checkpoint_utils.construct_restore_args(target=denom_target),
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

    restored_micro = custom_metadata.get("micro_step_count", 0) if isinstance(custom_metadata, Mapping) else 0
    if isinstance(restored_micro, int):
      self._saved_step_micro_counts[step] = restored_micro

    if "model_params" in restored_items:
      nnx.update(checkpoint_state.model, restored_items["model_params"])
    if checkpoint_state.optimizer is not None and "optimizer_state" in restored_items:
      nnx.update(checkpoint_state.optimizer, restored_items["optimizer_state"])
    if "accumulated_metrics" in restored_items:
      checkpoint_state.accumulated_metrics = restored_items["accumulated_metrics"]
    if "accumulated_grads" in restored_items:
      checkpoint_state.accumulated_grads = restored_items["accumulated_grads"]
    if "accumulated_denominator" in restored_items:
      checkpoint_state.accumulated_denominator = restored_items["accumulated_denominator"]["value"]

    return step, checkpoint_state, custom_metadata

  def close(self) -> None:
    """Closes the checkpoint manager."""
    self.wait_for_inflight_save_staging()
    if self._async_save_executor is not None:
      self._async_save_executor.shutdown(wait=True)
      self._async_save_executor = None
    if self._checkpoint_manager:
      self._checkpoint_manager.close()
