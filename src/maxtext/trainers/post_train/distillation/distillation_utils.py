# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
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

import pickle
import tensorflow as tf
from array_record.python import array_record_module

import abc
from typing import Any, Iterator, Optional, List, Callable, Literal

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint

from maxtext.utils import max_logging
# Reuse MaxText's native checkpointing logic
from maxtext.common.checkpointing import GrainCheckpointHandler, GrainCheckpointSave, GrainCheckpointRestore
from tunix.sft import peft_trainer
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager


# -----------------------------------------------------------------------------
# Custom Data Structures
# -----------------------------------------------------------------------------


@flax.struct.dataclass(frozen=True)
class DistillationForwardOutput:
  """Dataclass to carry MaxText-specific output fields."""

  #: logits
  logits: jax.Array
  #: out_projection_activations
  out_projection_activations: jax.Array | None = None


@flax.struct.dataclass(frozen=True)
class MaxTextTrainingInput(peft_trainer.TrainingInput):
  """Extended TrainingInput dataclass to carry MaxText-specific fields."""

  #: Position indices for the tokens (for RoPE).
  positions: jax.Array | None = None
  #: Segment IDs for packed sequences (0=padding, 1+=examples).
  decoder_segment_ids: jax.Array | None = None
  #: Ground truth target tokens (used for loss calculation and logging).
  targets: jax.Array | None = None
  #: Position indices for the target tokens.
  targets_position: jax.Array | None = None
  #: Segment IDs for packed target tokens.
  targets_segmentation: jax.Array | None = None
  #: Top-K logits from the teacher model.
  top_k_logits: jax.Array | None = None
  top_k_indices: jax.Array | None = None


# -----------------------------------------------------------------------------
# Data Loading Adapter
# -----------------------------------------------------------------------------


class OfflineArrayRecordIterator:
  """Reads the pre-generated global top-k logits file."""

  def __init__(self, data_dir: str, epochs: int = 100):
    self.filepath = data_dir

    if not tf.io.gfile.exists(self.filepath):
      raise FileNotFoundError(f"Offline distillation file not found: {self.filepath}")

    self.reader = array_record_module.ArrayRecordReader(self.filepath)
    self.num_records = self.reader.num_records()
    self.epochs = epochs
    self.current_epoch = 0
    self.record_index = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self.record_index >= self.num_records:
      self.current_epoch += 1
      if self.current_epoch >= self.epochs:
        raise StopIteration

      self.record_index = 0
      self.reader = array_record_module.ArrayRecordReader(self.filepath)

    record = self.reader.read()
    self.record_index += 1
    data = pickle.loads(record)

    # Map the arrays to match MaxText's expected dictionary
    batch = {
        "inputs": data["tokens"],
        "top_k_logits": data["top_k_logits"],
        "top_k_indices": data["top_k_indices"],
    }
    for key in ["inputs_position", "inputs_segmentation", "targets_segmentation", "targets"]:
      if key in data:
        batch[key] = data[key]

    return batch


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

    # If in SFT-mode, 'targets' contains prompts which should be masked out when computing the loss.
    # If using with packing the targets_segmentation mask is supposed to be a combined target+packing mask
    targets_segmentation = batch.get("targets_segmentation", jnp.ones_like(batch["targets"]))
    targets_position = batch.get("targets_position", batch.get("inputs_position"))

    # pylint: disable=unexpected-keyword-arg
    return MaxTextTrainingInput(
        input_tokens=batch["inputs"],
        input_mask=input_mask,
        positions=batch["inputs_position"],
        decoder_segment_ids=seg_ids,
        targets=batch["targets"],
        targets_position=targets_position,
        targets_segmentation=targets_segmentation,
        top_k_logits=batch.get("top_k_logits"),
        top_k_indices=batch.get("top_k_indices"),
    )


# -----------------------------------------------------------------------------
# Distillation Strategy
# -----------------------------------------------------------------------------


class DistillationStrategy(abc.ABC):
  """Abstract base class for MaxText Distillation Strategies."""

  def __init__(
      self, student_forward_fn: Callable, teacher_forward_fn: Callable, vocab_size: int, pad_id: int = 0, **kwargs
  ):
    """Initializes the generic distillation strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        vocab_size: The size of the model's vocabulary.
        pad_id: The ID used for padding tokens.
    """
    self.student_forward_fn = student_forward_fn
    self.teacher_forward_fn = teacher_forward_fn
    self.vocab_size = vocab_size
    self.pad_id = pad_id

  @abc.abstractmethod
  def compute_loss(
      self,
      student_output: "DistillationForwardOutput",
      teacher_output: "DistillationForwardOutput",
      labels: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes the distillation loss.

    Args:
        student_output: The forward pass output of the student model.
        teacher_output: The forward pass output of the frozen teacher model.
        labels: The masked one-hot encoded ground truth labels.

    Returns:
        A tuple containing the scalar loss and a dictionary of auxiliary metrics
        (e.g., {"distill/soft_loss": ..., "distill/total_loss": ...})
    """
    raise NotImplementedError

  @abc.abstractmethod
  def compute_eval_loss(
      self,
      student_output: "DistillationForwardOutput",
      labels: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes the evaluation loss (typically just the task loss).

    Args:
        student_output: The forward pass output of the student model.
        labels: The masked one-hot encoded ground truth labels.

    Returns:
        A tuple containing the scalar loss and an empty (or auxiliary) dict.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def create_labels(self, targets: jax.Array, targets_segmentation: Optional[jax.Array] = None, **kwargs) -> jax.Array:
    """
    Creates labels tensor to compute the loss
    """
    raise NotImplementedError


class CombinedDistillationStrategy(DistillationStrategy):
  """Strategy that returns detailed metrics for TensorBoard."""

  def __init__(
      self,
      student_forward_fn: Callable[..., DistillationForwardOutput],
      teacher_forward_fn: Callable[..., DistillationForwardOutput],
      pad_id: int = 0,
      temperature: float = 2.0,
      alpha: float = 0.5,
      beta_feature: float = 0.0,
      layer_indices: Optional[List[int]] = None,
      feature_loss_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
      feature_loss_type: Literal["cosine", "l2"] = "cosine",
      cosine_distance_axis: int | tuple[int, ...] = -1,
      vocab_size: int = 0,
  ):
    """Initializes the Combined strategy using tunix logit.LogitStrategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        labels_fn: Function to compute labels from model inputs.
        temperature: Temperature for softening probabilities (> 0).
        alpha: Weight to balance distillation loss and task loss (0.0 to 1.0).
        beta_feature: Weight to balance feature loss (0.0 to 1.0). 0.0 disables feature loss.
        layer_indices: Layer indices to apply feature loss.
        feature_loss_type: The type of feature loss to use if `feature_loss_fn` is None.
          Can be "cosine" (default) or "l2".
        feature_loss_fn: A function that takes two jax. Arrays (student_map,
          teacher_map) and returns a scalar loss. Defaults to Cosine Distance.
        cosine_distance_axis: The axis to use for cosine distance computation if
          feature_loss_fn is not provided. Defaults to -1.
    """

    super().__init__(
        student_forward_fn=student_forward_fn,
        teacher_forward_fn=teacher_forward_fn,
        vocab_size=vocab_size,
        pad_id=pad_id,
    )

    self.temperature = temperature
    self.alpha = alpha
    self.beta_feature = beta_feature
    self.layer_indices = jnp.array(layer_indices) if layer_indices is not None else None

    self.feature_loss_fn = feature_loss_fn
    if feature_loss_fn is None:
      if feature_loss_type == "cosine":
        self.feature_loss_fn = lambda student_features, teacher_features: jnp.mean(
            optax.cosine_distance(student_features, teacher_features, axis=cosine_distance_axis)
        )
      elif feature_loss_type == "l2":
        self.feature_loss_fn = lambda student_features, teacher_features: jnp.mean(
            optax.l2_loss(student_features, teacher_features)
        )
      else:
        raise ValueError(f"Unsupported feature_loss_type: {feature_loss_type!r}")

  def compute_loss(
      self,
      student_output: DistillationForwardOutput,
      teacher_output: DistillationForwardOutput,
      labels: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes Loss and Auxiliary Metrics."""
    # Calculate Distillation Loss (KL Divergence)
    # Scale logits by temperature T for soft targets
    # We use explicit float32 casting for stability in loss calculation
    s_logits = student_output.logits.astype(jnp.float32)
    t_logits = teacher_output.logits.astype(jnp.float32)

    # Shape: [num_layers, batch, seq, hidden_dim]
    s_features = student_output.out_projection_activations
    t_features = teacher_output.out_projection_activations

    if (s_features is None or t_features is None) and self.beta_feature > 0.0:
      raise ValueError(
          "Features extracted from student or teacher model are None, but distill_beta > 0.0. "
          "Ensure the model architecture supports feature extraction (e.g., 'out_projection_activations' is sowed)."
      )

    log_student_probs_temp = jax.nn.log_softmax(s_logits / self.temperature, axis=-1)
    teacher_probs_temp = jax.nn.softmax(t_logits / self.temperature, axis=-1)
    # labels are supposed to have all sft masks applied by this moment
    labels_mask = jnp.any(labels != 0, axis=-1, keepdims=True)
    mean_mask = jnp.squeeze(labels_mask, axis=-1)

    # KL(Teacher || Student)
    kl_div = optax.kl_divergence(log_student_probs_temp, teacher_probs_temp, where=labels_mask)

    # Scale gradients by T^2 (Hinton et al.)
    soft_loss = jnp.mean(kl_div, where=mean_mask) * (self.temperature**2)

    # 1. Student Hard Loss (Existing)
    ce_loss_student = optax.softmax_cross_entropy(logits=s_logits, labels=labels, where=labels_mask)
    hard_loss = jnp.mean(ce_loss_student, where=mean_mask)

    # 2. Teacher Hard Loss (For Verification)
    ce_loss_teacher = optax.softmax_cross_entropy(logits=t_logits, labels=labels, where=labels_mask)
    teacher_hard_loss = jnp.mean(ce_loss_teacher, where=mean_mask)

    # 3. Combine losses
    base_logit_loss = (self.alpha * soft_loss) + ((1.0 - self.alpha) * hard_loss)

    feature_loss = jnp.array(0.0)
    if self.beta_feature > 0.0:

      if self.layer_indices is not None:
        # jnp.take slices along axis=0 (the layer dimension)
        s_features_sliced = jnp.take(s_features, self.layer_indices, axis=0)
        t_features_sliced = jnp.take(t_features, self.layer_indices, axis=0)
      else:
        s_features_sliced = s_features
        t_features_sliced = t_features

      s_features_sliced = s_features_sliced.astype(jnp.float32)
      t_features_sliced = t_features_sliced.astype(jnp.float32)

      feature_loss = self.beta_feature * self.feature_loss_fn(s_features_sliced, t_features_sliced)

    total_loss = base_logit_loss + feature_loss

    # 4. Return Loss AND Metrics
    metrics = {
        "distill/soft_loss": soft_loss,
        "distill/hard_loss": hard_loss,
        "distill/kl_div": jnp.mean(kl_div, where=mean_mask),
        "distill/teacher_loss": teacher_hard_loss,
        "distill/out_proj_feature_loss": feature_loss,
        "distill/total_loss": total_loss,
        "distill/temperature": self.temperature,
        "distill/alpha": self.alpha,
        "distill/beta_feature": self.beta_feature,
    }
    return total_loss, metrics

  def compute_eval_loss(
      self,
      student_output: DistillationForwardOutput,
      labels: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes Eval Loss and returns empty aux dict (required for consistency)."""
    # Parent logic for task loss
    # We re-implement simple CE here to ensure float32 casting
    s_logits = student_output.logits.astype(jnp.float32)

    labels_mask = jnp.any(labels != 0, axis=-1, keepdims=True)
    mean_mask = jnp.squeeze(labels_mask, axis=-1)
    ce_loss = optax.softmax_cross_entropy(logits=s_logits, labels=labels, where=labels_mask)
    task_loss = jnp.mean(ce_loss, where=mean_mask)

    # Must return a tuple because _has_aux=True expects it
    return task_loss, {}

  def create_labels(self, targets, targets_segmentation=None, **kwargs):
    """Converts integer targets to masked one-hot vectors for hard label loss."""
    del kwargs  # Unused
    one_hot = jax.nn.one_hot(targets, self.vocab_size)
    mask = jnp.not_equal(targets, self.pad_id).astype(one_hot.dtype)[..., None]
    if targets_segmentation is not None:
      mask = mask * (targets_segmentation != 0)[..., None]
    return one_hot * mask


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
      params = nnx.state(model.student_model, nnx.LoRAParam)
    else:
      params = nnx.state(model.student_model)

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

  def maybe_restore(
      self,
      model: Any,
      optimizer: Any = None,
      restore_only_lora_params: bool = False,
  ) -> tuple[int, dict[str, Any]]:
    """Restores model and optimizer state if a checkpoint exists, using correct sharding specs.

    This method checks for the latest available checkpoint. If found, it restores the
    model parameters and optionally the optimizer state in-place. It automatically
    maps the parameter's `sharding` attributes to Orbax restore arguments to ensure
    the tensors are placed on the correct device meshes.

    Args:
      model: The model to restore. If a `ModelBundle` is provided, it automatically
        extracts and restores only the `student_model`.
      optimizer: The optimizer state to restore. If None, optimizer restoration is skipped.
      restore_only_lora_params: If True, restricts restoration to parameters marked
        as `nnx.LoRAParam`.

    Returns:
      A tuple containing the restored step number (0 if no checkpoint was found)
      and a dictionary of custom metadata.
    """
    if self._checkpoint_manager is None:
      return 0, {}

    step = self._checkpoint_manager.latest_step()
    if step is None:
      return 0, {}

    max_logging.log(f"Restoring from checkpoint step {step}...")

    # Extract student model safely
    target_model = getattr(model, "student_model", model)

    if restore_only_lora_params:
      params = nnx.state(target_model, nnx.LoRAParam)
    else:
      params = nnx.state(target_model)

    def map_to_pspec(data):
      if hasattr(data, "sharding"):
        return checkpoint.type_handlers.ArrayRestoreArgs(sharding=data.sharding)
      return None

    restore_args = jax.tree.map(map_to_pspec, params)

    cp_restore_args = {
        "model_params": checkpoint.args.PyTreeRestore(
            item=params,
            restore_args=restore_args,
        )
    }

    if optimizer is not None:
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      opt_restore_args = jax.tree.map(map_to_pspec, optimizer_state)
      cp_restore_args["optimizer_state"] = checkpoint.args.PyTreeRestore(
          item=optimizer_state,
          restore_args=opt_restore_args,
      )

    restored = self._checkpoint_manager.restore(
        step,
        args=checkpoint.args.Composite(**cp_restore_args),
    )

    nnx.update(target_model, restored.model_params)
    if optimizer is not None:
      nnx.update(optimizer, restored.optimizer_state)

    metadata = self._checkpoint_manager.metadata(step)
    if metadata and hasattr(metadata, "custom_metadata") and metadata.custom_metadata is not None:
      custom_metadata = metadata.custom_metadata
    else:
      custom_metadata = {}

    return step, dict(custom_metadata)

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
