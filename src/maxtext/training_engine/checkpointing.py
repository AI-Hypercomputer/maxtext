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

from typing import Any
from absl import logging
from flax import nnx
import jax
from maxtext.configs import pyconfig
import orbax.checkpoint as ocp


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
              save_interval_steps=getattr(config, "checkpoint_period", 1),
              max_to_keep=getattr(config, "max_num_checkpoints_to_keep", None),
              enable_async_checkpointing=getattr(
                  config, "async_checkpointing", True
              ),
          ),
      )

  def get_latest_step(self) -> int | None:
    """Returns the latest checkpoint step."""
    if self._checkpoint_manager:
      return self._checkpoint_manager.latest_step()
    return None

  def save_checkpoint(
      self,
      step: int,
      model: nnx.Module,
      optimizer: nnx.optimizer.Optimizer | None,
      custom_metadata: Any,
  ) -> bool:
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save.
      optimizer: The optimizer to save.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping save.")
      return False

    # Check if the checkpoint already exists at the current step.
    if self.get_latest_step() == step:
      logging.info(
          "Checkpoint already saved at step %d, skipping save.",
          step,
      )
      return False

    params = nnx.state(model)
    jax.block_until_ready(params)
    model_cp_args = ocp.args.PyTreeSave(
        item=params,
        save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params),
    )
    save_args = {"model_params": model_cp_args}
    if optimizer is not None:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
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
    )

  def restore_checkpoint(
      self,
      model: nnx.Module,
      optimizer: nnx.optimizer.Optimizer | None,
      step: int | None = None,
  ) -> Any:
    """Restores items from the checkpoint at the given step.

    Args:
      model: The model to restore the params to.
      optimizer: The optimizer to restore the state to.
      step: Optional step index to restore from.

    Returns:
      The metadata of the restored checkpoint, or None if no checkpoint was
      restored.
    """
    if self._checkpoint_manager is None:
      logging.info("Checkpointing is disabled, skipping restore.")
      return None, {}

    if step is None:
      step = self.get_latest_step()
      if step is None:
        logging.info("No checkpoint found, skipping restore.")
        return None, {}

    metadata = self._checkpoint_manager.metadata(step)

    abstract_params = nnx.state(model)
    model_args = ocp.args.PyTreeRestore(
        item=abstract_params,
        restore_args=ocp.checkpoint_utils.construct_restore_args(
            target=abstract_params
        ),
    )
    if optimizer is not None and "optimizer_state" in metadata.item_metadata:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      optimizer_args = ocp.args.PyTreeRestore(
          item=optimizer_state,
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              target=optimizer_state
          ),
      )
      restore_args = {
          "model_params": model_args,
          "optimizer_state": optimizer_args,
      }
    else:
      restore_args = {"model_params": model_args}

    restored_items = self._checkpoint_manager.restore(
        step=step,
        args=ocp.args.Composite(**restore_args),
    )
    if "model_params" in restored_items:
      nnx.update(model, restored_items["model_params"])
    if optimizer is not None and "optimizer_state" in restored_items:
      nnx.update(optimizer, restored_items["optimizer_state"])

    return step, metadata

  def close(self) -> None:
    """Closes the checkpoint manager."""
    if self._checkpoint_manager:
      self._checkpoint_manager.close()
