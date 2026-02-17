# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility classes for MaxText Distillation with Tunix.

This module contains adapter classes that bridge MaxText's data loading and
model structures with Tunix's training interfaces.
"""

from typing import Any, Iterator

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint

from maxtext.utils import max_logging
# Reuse MaxText's native checkpointing logic
from maxtext.common.checkpointing import GrainCheckpointHandler, GrainCheckpointSave, GrainCheckpointRestore
from tunix.distillation import distillation_trainer
from tunix.distillation.strategies import logit
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager


# -----------------------------------------------------------------------------
# Custom Data Structures
# -----------------------------------------------------------------------------


@flax.struct.dataclass(frozen=True)
class MaxTextTrainingInput(distillation_trainer.TrainingInput):
  """Extended TrainingInput dataclass to carry MaxText-specific fields."""

  #: Position indices for the tokens (for RoPE).
  positions: jax.Array = None
  #: Segment IDs for packed sequences (0=padding, 1+=examples).
  decoder_segment_ids: jax.Array = None
  #: Ground truth target tokens (used for loss calculation and logging).
  targets: jax.Array = None


# -----------------------------------------------------------------------------
# Data Loading Adapter
# -----------------------------------------------------------------------------


class MaxTextToTunixIterator:
  """Adapts the raw dictionary output of MaxText's data loader to Tunix objects.

  MaxText's `input_pipeline_interface.create_data_iterator` yields a dictionary.
  Tunix expects an object with specific attributes (input_tokens, etc.).
  """

  def __init__(self, maxtext_iterator: Iterator):
    """Initializes the adapter.

    Args:
      maxtext_iterator: The upstream iterator created by MaxText's input pipeline.
    """
    self._iterator = maxtext_iterator

  def __iter__(self):
    """Returns self as the iterator."""
    return self

  def __next__(self) -> MaxTextTrainingInput:
    """Fetches the next batch and converts it to the Tunix data class.

    Returns:
      A MaxTextTrainingInput object containing the batch data.

    Raises:
      StopIteration: If the upstream iterator is exhausted.
    """
    batch = next(self._iterator)

    # Ensure segmentation exists, default to ones if missing (standard non-packed)
    if "inputs_segmentation" in batch:
      input_mask = batch["inputs_segmentation"] != 0
      seg_ids = batch["inputs_segmentation"]
    else:
      # Fallback for non-packed datasets
      input_mask = jnp.ones_like(batch["inputs"], dtype=bool)
      seg_ids = None

    # pylint: disable=unexpected-keyword-arg
    return MaxTextTrainingInput(
        input_tokens=batch["inputs"],
        input_mask=input_mask,
        teacher_output=None,
        positions=batch["inputs_position"],
        decoder_segment_ids=seg_ids,
        targets=batch["targets"],
    )


# -----------------------------------------------------------------------------
# Distillation Strategy
# -----------------------------------------------------------------------------
class MonitoredLogitStrategy(logit.LogitStrategy):
  """Logit Strategy that returns detailed metrics for TensorBoard."""

  def compute_loss(
      self,
      student_output: jax.Array,
      teacher_output: jax.Array,
      labels: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes Loss and Auxiliary Metrics."""
    # Calculate Distillation Loss (KL Divergence)
    # Scale logits by temperature T for soft targets
    # We use explicit float32 casting for stability in loss calculation
    s_logits = student_output.astype(jnp.float32)
    t_logits = teacher_output.astype(jnp.float32)

    log_student_probs_temp = jax.nn.log_softmax(s_logits / self.temperature, axis=-1)
    teacher_probs_temp = jax.nn.softmax(t_logits / self.temperature, axis=-1)

    # KL(Teacher || Student)
    kl_div = optax.kl_divergence(log_student_probs_temp, teacher_probs_temp)

    # Scale gradients by T^2 (Hinton et al.)
    soft_loss = jnp.mean(kl_div) * (self.temperature**2)

    # 1. Student Hard Loss (Existing)
    ce_loss_student = optax.softmax_cross_entropy(logits=s_logits, labels=labels)
    hard_loss = jnp.mean(ce_loss_student)

    # 2. Teacher Hard Loss (For Verification)
    ce_loss_teacher = optax.softmax_cross_entropy(logits=t_logits, labels=labels)
    teacher_hard_loss = jnp.mean(ce_loss_teacher)

    # 3. Combine losses
    total_loss = (self.alpha * soft_loss) + ((1.0 - self.alpha) * hard_loss)

    # 4. Return Loss AND Metrics
    metrics = {
        "distill/soft_loss": soft_loss,
        "distill/hard_loss": hard_loss,
        "distill/kl_div": jnp.mean(kl_div),
        "distill/teacher_loss": teacher_hard_loss,
    }
    return total_loss, metrics

  def compute_eval_loss(
      self,
      student_output: jax.Array,
      labels: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes Eval Loss and returns empty aux dict (required for consistency)."""
    # Parent logic for task loss
    # We re-implement simple CE here to ensure float32 casting
    s_logits = student_output.astype(jnp.float32)
    ce_loss = optax.softmax_cross_entropy(logits=s_logits, labels=labels)
    task_loss = jnp.mean(ce_loss)

    # Must return a tuple because _has_aux=True expects it
    return task_loss, {}


# -----------------------------------------------------------------------------
# Checkpoint Manager
# -----------------------------------------------------------------------------


class MaxTextCheckpointManager(tunix_checkpoint_manager.CheckpointManager):
  """Custom CheckpointManager that uses MaxText's native handlers.

  This manager extends Tunix to support saving/restoring the MaxText input pipeline
  (Grain) alongside the model and optimizer.
  """

  def __init__(
      self,
      raw_iterator: Any | None,
      root_directory: str | None = None,
      options: checkpoint.CheckpointManagerOptions | None = None,
  ):
    super().__init__(root_directory=root_directory, options=options)
    self._iterator = raw_iterator

    # Re-initialize internal Orbax manager with MaxText's Grain handler
    # pylint: disable=access-member-before-definition
    if self._checkpoint_manager is not None:
      root_directory = self._checkpoint_manager.directory

      if options is None:
        options = getattr(self._checkpoint_manager, "options", None)

      item_handlers = {
          "model_params": checkpoint.PyTreeCheckpointHandler(),
          "optimizer_state": checkpoint.PyTreeCheckpointHandler(),
          "custom_metadata": checkpoint.JsonCheckpointHandler(),
          # Use MaxText's handler for the iterator
          "iter": GrainCheckpointHandler(),
      }

      self._checkpoint_manager.close()
      self._checkpoint_manager = checkpoint.CheckpointManager(
          root_directory,
          item_handlers=item_handlers,
          options=options,
      )
    # pylint: enable=access-member-before-definition

  def save(self, step, model, optimizer=None, save_only_lora_params=False, force=False, custom_metadata=None):
    """Saves the checkpoint including the input pipeline state (if available)."""
    if self._checkpoint_manager is None:
      return False
    if not force and not self._checkpoint_manager.should_save(step):
      return False

    # Standard Tunix Logic for Model/Optimizer
    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)

    # Define standard SaveArgs once to reuse
    default_save_args = checkpoint.SaveArgs()
    cp_save_args = {
        "model_params": checkpoint.args.PyTreeSave(
            item=params, save_args=jax.tree.map(lambda _: default_save_args, params)
        ),
    }
    if optimizer is not None:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      cp_save_args["optimizer_state"] = checkpoint.args.PyTreeSave(
          item=optimizer_state, save_args=jax.tree.map(lambda _: default_save_args, optimizer_state)
      )

    if self._iterator is not None:
      # Follow MaxText's logic to handle multi-process saving
      # Logic extracted from src/maxtext/common/checkpointing.py:save_checkpoint
      data_iterator = self._iterator
      if not isinstance(data_iterator, list):
        data_iterator = [data_iterator]

      grain_iters_to_save = []
      process_count_total = jax.process_count() * len(data_iterator)

      for i, data_iter in enumerate(data_iterator):
        process_index = jax.process_index() + i * jax.process_count()
        # MaxText iterators (MultiHostDataLoadIterator) wrap the actual Grain iterator in .local_iterator
        local_iter = data_iter.local_iterator if hasattr(data_iter, "local_iterator") else data_iter
        grain_iters_to_save.append((local_iter, process_index, process_count_total))

      # Use GrainCheckpointSave wrapper
      cp_save_args["iter"] = GrainCheckpointSave(item=grain_iters_to_save)

    return self._checkpoint_manager.save(
        step,
        args=checkpoint.args.Composite(**cp_save_args),
        custom_metadata=custom_metadata or {},
        force=force,
    )

  def restore_iterator(self):
    """Restores the iterator using MaxText's logic."""
    if self._checkpoint_manager is None or self._iterator is None:
      return None

    step = self._checkpoint_manager.latest_step()
    if step is None:
      return None

    try:
      # MaxText logic for restoration (simplified for standard case)
      # We assume 1-to-1 process mapping for now (no elasticity logic here yet)
      data_iter = self._iterator
      local_iter = data_iter.local_iterator if hasattr(data_iter, "local_iterator") else data_iter

      restore_args = GrainCheckpointRestore(item=local_iter)

      self._checkpoint_manager.restore(step, args=checkpoint.args.Composite(iter=restore_args))
      # Since Grain restores in-place via set_state(), we return the original object
      return self._iterator

    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Warning: Could not restore input pipeline: {e}")
      return None

  def wait_until_finished(self):
    """Blocks until all outstanding checkpoint operations are complete."""
    if self._checkpoint_manager is not None:
      self._checkpoint_manager.wait_until_finished()
