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
from typing import Any, List

from absl import logging
from flax import nnx
import jax
from maxtext.common import checkpoint_context
from maxtext.configs import pyconfig
from maxtext.training_engine import abstract_engine
from orbax.checkpoint import v1 as ocp


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
    self._checkpointer: ocp.training.Checkpointer | None = None
    # The v0 manager took async-ness as an option; the v1 Checkpointer chooses per call,
    # so remember it and dispatch in save_checkpoint.
    self._use_async = bool(config.async_checkpointing)
    if checkpoint_dir:
      preservation_policy = None
      if config.max_num_checkpoints_to_keep is not None:
        preservation_policy = checkpoint_context.build_preservation_policy(max_to_keep=config.max_num_checkpoints_to_keep)
      self._checkpointer = ocp.training.Checkpointer(
          checkpoint_dir,
          context=checkpoint_context.build_context(),
          save_decision_policy=checkpoint_context.build_save_decision_policy(
              save_interval_steps=config.checkpoint_period
          ),
          preservation_policy=preservation_policy,
      )

  def get_latest_step(self) -> int | None:
    """Returns the latest checkpoint step."""
    if self._checkpointer:
      latest = self._checkpointer.latest
      return latest.step if latest is not None else None
    return None

  def wait_until_finished(self) -> None:
    """Waits for any ongoing async checkpoint saves to finish."""
    if self._checkpointer:
      self._checkpointer.wait()

  def get_saved_micro_step_count(self, step: int) -> int:
    """Returns how far into `step` the checkpoint already on disk got.

    Args:
      step: The step whose saved checkpoint should be inspected.

    Returns:
      0 if that checkpoint covers a complete step, otherwise the number of micro-batches
      accumulated into it.
    """
    if self._checkpointer is None:
      return 0
    try:
      metadata = self._checkpointer.checkpointables_metadata(step)
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
    if self._checkpointer is None:
      return
    logging.info("Deleting intra-step checkpoint at step %d so a more complete one can replace it.", step)
    self._checkpointer.wait()
    # The v1 Checkpointer exposes no public delete yet; use the
    # underlying v0 manager's delete (the same call Orbax makes internally).
    self._checkpointer._manager.delete(step)  # pylint: disable=protected-access
    # The v1 `checkpoints`/`latest` properties cache their step listing and are
    # unaware of deletes made below them; refresh so the replacement save isn't
    # misjudged.
    self._checkpointer.reload()

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
    if self._checkpointer is None:
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
    checkpointables = {"model_params": params}

    if checkpoint_state.optimizer:
      optimizer_state = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)
      jax.block_until_ready(optimizer_state)
      checkpointables["optimizer_state"] = optimizer_state

    if checkpoint_state.accumulated_metrics is not None:
      jax.block_until_ready(checkpoint_state.accumulated_metrics)
      checkpointables["accumulated_metrics"] = checkpoint_state.accumulated_metrics

    if checkpoint_state.accumulated_grads:
      jax.block_until_ready(checkpoint_state.accumulated_grads)
      checkpointables["accumulated_grads"] = checkpoint_state.accumulated_grads

    try:
      if self._use_async:
        response = self._checkpointer.save_checkpointables_async(
            step, checkpointables, custom_metadata=custom_metadata, **kwargs
        )
        return response is not None
      return self._checkpointer.save_checkpointables(step, checkpointables, custom_metadata=custom_metadata, **kwargs)
    except FileExistsError as e:  # ocp.training StepAlreadyExistsError subclasses FileExistsError
      logging.info("Checkpoint for step %d already exists, skipping save. (%s)", step, e)
      return False

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
    if self._checkpointer is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, checkpoint_state, None

    if step is None:
      step = self.get_latest_step()
      if step is None:
        logging.info("No checkpoint found, skipping restore.")
        return None, checkpoint_state, None

    metadata = self._checkpointer.checkpointables_metadata(step)
    saved_checkpointables = metadata.metadata if isinstance(metadata.metadata, Mapping) else {}

    abstract_checkpointables: dict[str, Any] = {"model_params": nnx.state(checkpoint_state.model)}

    if checkpoint_state.optimizer is not None and "optimizer_state" in saved_checkpointables:
      abstract_checkpointables["optimizer_state"] = nnx.state(checkpoint_state.optimizer, nnx.optimizer.OptState)

    if "accumulated_metrics" in saved_checkpointables:
      abstract_checkpointables["accumulated_metrics"] = saved_checkpointables["accumulated_metrics"]

    if "accumulated_grads" in saved_checkpointables:
      abstract_checkpointables["accumulated_grads"] = nnx.state(checkpoint_state.model, nnx.Param)

    custom_metadata = None
    if metadata and hasattr(metadata, "custom_metadata"):
      custom_metadata = metadata.custom_metadata

    try:
      restored_items = self._checkpointer.load_checkpointables(step, abstract_checkpointables)
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
    if self._checkpointer:
      self._checkpointer.close()
