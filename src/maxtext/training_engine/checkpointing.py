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

import dataclasses
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
      self._checkpoint_manager = ocp.CheckpointManager(
          directory=checkpoint_dir,
          options=ocp.CheckpointManagerOptions(
              save_interval_steps=config.checkpoint_period,
              max_to_keep=config.max_num_checkpoints_to_keep,
              enable_async_checkpointing=config.async_checkpointing,
          ),
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
      force_ckpt_save: bool = False,
  ) -> bool:
    """Saves the params for the given step along with optional intra-step state.

    Args:
      step: The step to save the params for.
      checkpoint_state: CheckpointState object containing model, optimizer, and
        optional intra_step_state.
      custom_metadata: Custom metadata to save with the checkpoint.
      force_ckpt_save: Whether to force save the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping save.")
      return False

    # Check if the checkpoint already exists at the current step.
    if not force_ckpt_save and self.get_latest_step() == step:
      logging.info(
          "Checkpoint already saved at step %d, skipping save.",
          step,
      )
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
        force=force_ckpt_save,
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
