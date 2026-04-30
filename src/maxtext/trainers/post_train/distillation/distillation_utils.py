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

import abc
import pickle
from typing import Any, Callable, Iterator, List, Literal, Optional, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from array_record.python import array_record_module
from orbax import checkpoint

from maxtext.utils import max_logging
# Reuse MaxText's native checkpointing logic
from maxtext.common.checkpointing import GrainCheckpointHandler, GrainCheckpointSave, GrainCheckpointRestore
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager
from tunix.sft import peft_trainer


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
  #: moe load balance loss
  moe_lb_loss: jax.Array | None = None


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

# Clamp CE before exp() so a divergence spike doesn't poison PPL averages
# with inf. 20 nats is well above plausible CE (Llama random-init ~11.76)
# and far below fp32 exp overflow (~88).
_PPL_CE_CAP = 20.0


def compute_schedule(
    step: jax.Array,
    max_steps: int,
    start_value: float,
    end_value: float | None,
    schedule_type: str,
) -> jax.Array:
  """Computes a scheduled value based on training progress.

  Args:
    step: Current training step as a JAX array.
    max_steps: Total number of training steps.
    start_value: Value at the beginning of training.
    end_value: Value at the end of training. If None, returns start_value.
    schedule_type: One of "constant", "linear", or "cosine".

  Returns:
    The scheduled value as a JAX scalar.
  """
  if end_value is None or schedule_type == "constant":
    return jnp.array(start_value, dtype=jnp.float32)

  progress = jnp.clip(step.astype(jnp.float32) / max_steps, 0.0, 1.0)

  if schedule_type == "linear":
    return start_value + (end_value - start_value) * progress
  elif schedule_type == "cosine":
    return end_value + (start_value - end_value) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
  else:
    raise ValueError(f"Unsupported schedule_type: {schedule_type!r}. Must be 'constant', 'linear', or 'cosine'.")


def weighted_mean(sum_count_pairs: Sequence[tuple[Any, Any]] | np.ndarray) -> float:
  """Aggregates `(sum, count)` pairs into a single token-weighted mean.

  Used as the aggregation function for metrics emitted by `compute_loss` and
  `compute_eval_loss`. Robust to per-host imbalance and to varying valid-token
  counts across logging steps:
    final_value = sum(sums) / sum(counts)

  Accepts either a list of (sum, count) tuples or an ndarray of shape (N, 2).
  Tunix's metrics writer can pass either form, so we normalize here.

  Returns 0.0 for an empty input or when total count is non-positive.
  """
  arr = np.asarray(sum_count_pairs, dtype=np.float32)
  if arr.size == 0:
    return 0.0
  # Normalize shape. Single pair -> (1, 2); list of pairs -> (N, 2).
  if arr.ndim == 1:
    arr = arr.reshape(1, -1)
  if arr.ndim != 2 or arr.shape[1] != 2:
    return 0.0
  total = float(arr[:, 1].sum())
  if total <= 0.0:
    return 0.0
  return float(arr[:, 0].sum() / total)


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
      step: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Computes the distillation loss.

    Args:
        student_output: The forward pass output of the student model.
        teacher_output: The forward pass output of the frozen teacher model.
        labels: The masked one-hot encoded ground truth labels.
        step: Current training step for dynamic scheduling. If None, uses fixed values.

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
      feature_loss_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array] | None = None,
      feature_loss_type: Literal["cosine", "l2"] = "cosine",
      cosine_distance_axis: int | tuple[int, ...] = -1,
      vocab_size: int = 0,
      alpha_end: float | None = None,
      alpha_schedule: Literal["constant", "linear", "cosine"] = "constant",
      temperature_end: float | None = None,
      temperature_schedule: Literal["constant", "linear", "cosine"] = "constant",
      beta_end: float | None = None,
      beta_schedule: Literal["constant", "linear", "cosine"] = "constant",
      max_steps: int = 1,
  ):
    """Initializes the Combined distillation strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        temperature: Temperature for softening probabilities (> 0).
        alpha: Weight to balance distillation loss and task loss (0.0 to 1.0).
        beta_feature: Weight to balance feature loss (0.0 to 1.0). 0.0 disables feature loss.
        layer_indices: Layer indices to apply feature loss.
        feature_loss_type: The type of feature loss to use if `feature_loss_fn` is None.
          Can be "cosine" (default) or "l2".
        feature_loss_fn: A function (student_features, teacher_features, mask) -> scalar
          where features are [L, B, T, D] and mask is [B, T] (1.0 for valid tokens).
          Defaults to a masked cosine distance with epsilon-floored safe-norm.
        cosine_distance_axis: The axis to use for cosine distance computation if
          feature_loss_fn is not provided. Defaults to -1.
        alpha_end: Target alpha value at end of training. None keeps alpha fixed.
        alpha_schedule: Schedule type for alpha annealing.
        temperature_end: Target temperature at end of training. None keeps temperature fixed.
        temperature_schedule: Schedule type for temperature annealing.
        beta_end: Target beta_feature value at end of training. None keeps beta fixed.
        beta_schedule: Schedule type for beta annealing.
        max_steps: Total training steps, used for schedule computation.
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

    # Schedule parameters
    self.alpha_end = alpha_end
    self.alpha_schedule = alpha_schedule
    self.temperature_end = temperature_end
    self.temperature_schedule = temperature_schedule
    self.beta_end = beta_end
    self.beta_schedule = beta_schedule
    self.max_steps = max_steps

    # Validate schedule parameter ranges
    if alpha_end is not None and not 0.0 <= alpha_end <= 1.0:
      raise ValueError(f"alpha_end must be in [0, 1], got {alpha_end}")
    if temperature_end is not None and temperature_end <= 0.0:
      raise ValueError(f"temperature_end must be > 0, got {temperature_end}")
    if beta_end is not None and beta_end < 0.0:
      raise ValueError(f"beta_end must be >= 0, got {beta_end}")
    if beta_feature == 0.0 and beta_end is not None and beta_end > 0.0:
      raise ValueError(
          f"distill_beta=0.0 but distill_beta_end={beta_end}. Feature extraction is disabled when "
          "distill_beta starts at 0.0 (the model does not sow intermediate activations). "
          "Set distill_beta to a small positive value (e.g., 1e-6) to enable feature extraction."
      )
    for param_name, schedule, end_value in [
        ("alpha", alpha_schedule, alpha_end),
        ("temperature", temperature_schedule, temperature_end),
        ("beta", beta_schedule, beta_end),
    ]:
      if schedule != "constant" and end_value is None:
        raise ValueError(
            f"{param_name}_schedule is '{schedule}' but {param_name}_end is None. "
            f"Set {param_name}_end to a target value or use schedule='constant'."
        )

    # Mask keeps zero-norm pad activations out of the cosine denominator to avoid 0/0 NaN.
    self.feature_loss_fn = feature_loss_fn
    if feature_loss_fn is None:
      if feature_loss_type == "cosine":

        def _masked_cosine(student_features, teacher_features, mask):
          # epsilon>0 floors the safe-norm so an all-zero row can't divide by zero.
          cd = optax.cosine_distance(
              student_features, teacher_features, axis=cosine_distance_axis, epsilon=1e-6
          )  # [L, B, T]
          mask_b = mask.astype(cd.dtype)
          num_valid_terms = jnp.maximum(jnp.sum(mask_b), 1.0) * cd.shape[0]
          return jnp.sum(cd * mask_b[None, :, :]) / num_valid_terms

        self.feature_loss_fn = _masked_cosine
      elif feature_loss_type == "l2":

        def _masked_l2(student_features, teacher_features, mask):
          sq = jnp.mean(jnp.square(student_features - teacher_features), axis=-1)  # [L, B, T]
          mask_b = mask.astype(sq.dtype)
          num_valid_terms = jnp.maximum(jnp.sum(mask_b), 1.0) * sq.shape[0]
          return jnp.sum(sq * mask_b[None, :, :]) / num_valid_terms

        self.feature_loss_fn = _masked_l2
      else:
        raise ValueError(f"Unsupported feature_loss_type: {feature_loss_type!r}")

  def _get_scheduled_weights(self, step: jax.Array | None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolves the current alpha, temperature, and beta values from schedules.

    Args:
      step: Current training step. If None, returns the fixed initial values.

    Returns:
      A tuple of (alpha, temperature, beta_feature) as JAX scalars.
    """
    if step is None:
      return (
          jnp.array(self.alpha, dtype=jnp.float32),
          jnp.array(self.temperature, dtype=jnp.float32),
          jnp.array(self.beta_feature, dtype=jnp.float32),
      )
    alpha = compute_schedule(step, self.max_steps, self.alpha, self.alpha_end, self.alpha_schedule)
    temperature = compute_schedule(
        step, self.max_steps, self.temperature, self.temperature_end, self.temperature_schedule
    )
    beta_feature = compute_schedule(step, self.max_steps, self.beta_feature, self.beta_end, self.beta_schedule)
    return alpha, temperature, beta_feature

  def compute_loss(
      self,
      student_output: DistillationForwardOutput,
      teacher_output: DistillationForwardOutput,
      labels: jax.Array,
      step: jax.Array | None = None,
  ) -> tuple[jax.Array, dict[str, tuple[jax.Array, jax.Array]]]:
    """Computes Loss and Auxiliary Metrics.

    Metrics are emitted as (sum, count) pairs so that they can be aggregated
    across hosts and across logging windows in a token-weighted (unbiased) way:
    final_value = sum(sums) / sum(counts).
    """
    # Resolve scheduled weights for this step
    alpha, temperature, beta_feature = self._get_scheduled_weights(step)

    s_logits = student_output.logits.astype(jnp.float32)
    t_logits = teacher_output.logits.astype(jnp.float32)

    s_features = student_output.out_projection_activations
    t_features = teacher_output.out_projection_activations

    if (s_features is None or t_features is None) and self.beta_feature > 0.0:
      raise ValueError(
          "Features extracted from student or teacher model are None, but distill_beta > 0.0. "
          "Ensure the model architecture supports feature extraction (e.g., 'out_projection_activations' is sowed)."
      )

    # Per-token validity mask, derived from the one-hot labels so we don't need
    # a separate mask input. A padded (fully-zero) row yields `any != 0 == False`.
    mask = jnp.any(labels != 0, axis=-1).astype(jnp.float32)  # [B, T]
    valid_count = jnp.sum(mask)
    safe_count = jnp.maximum(valid_count, 1.0)

    # --- Soft loss: KL on temperature-softened distributions ---
    log_s_T = jax.nn.log_softmax(s_logits / temperature, axis=-1)
    t_p_T = jax.nn.softmax(t_logits / temperature, axis=-1)
    # KL(teacher || student) per position. optax.kl_divergence(log_pred, target) = KL(target || pred).
    kl_softened_per_pos = optax.kl_divergence(log_s_T, t_p_T)  # [B, T]
    kl_softened_sum = jnp.sum(kl_softened_per_pos * mask)
    # Scale by T^2 (Hinton). Apply once at the loss; logged metric is the scaled sum too.
    soft_loss_sum_scaled = kl_softened_sum * (temperature**2)
    soft_loss_mean = soft_loss_sum_scaled / safe_count

    # --- Hard loss: student CE against ground-truth ---
    ce_student_per_pos = optax.softmax_cross_entropy(logits=s_logits, labels=labels)
    ce_student_sum = jnp.sum(ce_student_per_pos * mask)
    hard_loss_mean = ce_student_sum / safe_count

    # --- Teacher CE (verification metric) ---
    ce_teacher_per_pos = optax.softmax_cross_entropy(logits=t_logits, labels=labels)
    ce_teacher_sum = jnp.sum(ce_teacher_per_pos * mask)

    # --- Always-T=1 KL for cross-run / cross-anneal comparability ---
    log_s_1 = jax.nn.log_softmax(s_logits, axis=-1)
    t_p_1 = jax.nn.softmax(t_logits, axis=-1)
    kl_t1_per_pos = optax.kl_divergence(log_s_1, t_p_1)
    kl_t1_sum = jnp.sum(kl_t1_per_pos * mask)

    base_logit_loss = (alpha * soft_loss_mean) + ((1.0 - alpha) * hard_loss_mean)

    feature_loss = jnp.array(0.0, dtype=jnp.float32)
    if self.beta_feature > 0.0:
      if self.layer_indices is not None:
        s_features_sliced = jnp.take(s_features, self.layer_indices, axis=0)
        t_features_sliced = jnp.take(t_features, self.layer_indices, axis=0)
      else:
        s_features_sliced = s_features
        t_features_sliced = t_features

      s_features_sliced = s_features_sliced.astype(jnp.float32)
      t_features_sliced = t_features_sliced.astype(jnp.float32)

      feature_loss = beta_feature * self.feature_loss_fn(s_features_sliced, t_features_sliced, mask)

    total_loss = base_logit_loss + feature_loss

    moe_lb_loss = jnp.array(0.0)
    if student_output.moe_lb_loss is not None:
      # The moe_lb_loss collected from the model is already scaled by load_balance_loss_weight
      # within the MoE layer itself (see load_balance_loss in moe.py).
      moe_lb_loss = student_output.moe_lb_loss
      total_loss += moe_lb_loss

    teacher_moe_lb_loss = jnp.array(0.0)
    if teacher_output.moe_lb_loss is not None:
      teacher_moe_lb_loss = teacher_output.moe_lb_loss

    # Per-step next-token perplexity. Note: this is mean(exp(per-step CE)), not
    # exp(window-CE-mean) — close to true perplexity in steady state. For the exact
    # perplexity over a logging window compute exp(distill/hard_loss) on the TB side.
    teacher_loss_mean = ce_teacher_sum / safe_count
    student_perplexity_step = jnp.exp(jnp.minimum(hard_loss_mean, _PPL_CE_CAP))
    teacher_perplexity_step = jnp.exp(jnp.minimum(teacher_loss_mean, _PPL_CE_CAP))

    one = jnp.array(1.0, dtype=jnp.float32)
    metrics: dict[str, tuple[jax.Array, jax.Array]] = {
        # Token-weighted: emit (sum, valid_count) so multi-host averaging is unbiased.
        "distill/soft_loss": (soft_loss_sum_scaled, valid_count),
        "distill/hard_loss": (ce_student_sum, valid_count),
        "distill/teacher_loss": (ce_teacher_sum, valid_count),
        # Next-token prediction perplexity (per-step approximation of exp(hard_loss)).
        # The headline `_train_perplexity` Tunix prints is exp(total_loss) which for
        # distillation is exp(α·soft + (1-α)·hard + β·feature) and NOT next-token PPL.
        "distill/student_perplexity": (student_perplexity_step, one),
        "distill/teacher_perplexity": (teacher_perplexity_step, one),
        # KL at the current (scheduled) temperature T, without the T^2 scaling
        # that soft_loss applies. Pair with kl_div_T1 to compare T vs T=1.
        "distill/kl_div_at_T": (kl_softened_sum, valid_count),
        # KL at T=1: comparable across runs / annealing schedules.
        "distill/kl_div_T1": (kl_t1_sum, valid_count),
        # Per-step quantities: (value, 1.0) so the aggregator yields a simple mean over steps.
        "distill/out_proj_feature_loss": (feature_loss, one),
        "distill/moe_lb_loss": (moe_lb_loss, one),
        "distill/teacher_moe_lb_loss": (teacher_moe_lb_loss, one),
        "distill/total_loss": (total_loss, one),
        "distill/temperature": (temperature, one),
        "distill/alpha": (alpha, one),
        "distill/beta_feature": (beta_feature, one),
    }
    return total_loss, metrics

  def compute_eval_loss(
      self,
      student_output: DistillationForwardOutput,
      labels: jax.Array,
  ) -> tuple[jax.Array, dict[str, tuple[jax.Array, jax.Array]]]:
    """Computes Eval Loss. Returns (loss, metrics) with (sum, count) metric pairs."""
    s_logits = student_output.logits.astype(jnp.float32)

    mask = jnp.any(labels != 0, axis=-1).astype(jnp.float32)
    valid_count = jnp.sum(mask)
    safe_count = jnp.maximum(valid_count, 1.0)

    ce_per_pos = optax.softmax_cross_entropy(logits=s_logits, labels=labels)
    ce_sum = jnp.sum(ce_per_pos * mask)
    task_loss = ce_sum / safe_count

    metrics = {
        "eval/hard_loss": (ce_sum, valid_count),
        "eval/student_perplexity": (jnp.exp(jnp.minimum(task_loss, _PPL_CE_CAP)), jnp.array(1.0, dtype=jnp.float32)),
    }
    return task_loss, metrics

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
      root_directory: str | None,
      student_config: Any | None,
      options: checkpoint.CheckpointManagerOptions | None = None,
  ):
    super().__init__(root_directory=root_directory, options=options)
    self.student_config = student_config
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

  def save(
      self,
      step,
      model,
      optimizer=None,
      save_only_lora_params=False,
      force=False,
      custom_metadata=None,
  ):
    """Saves the checkpoint including the input pipeline state (if available)."""
    if self._checkpoint_manager is None:
      return False
    if not force and not self._checkpoint_manager.should_save(step):
      return False

    # Standard Tunix Logic for Model/Optimizer.
    # Accept either a ModelBundle (common path) or a plain nnx module.
    target_model = getattr(model, "student_model", model)
    if save_only_lora_params:
      params = nnx.state(target_model, nnx.LoRAParam)
    else:
      params = nnx.state(target_model)

    # Define standard SaveArgs once to reuse
    default_save_args = checkpoint.SaveArgs()
    cp_save_args = {
        "model_params": checkpoint.args.PyTreeSave(
            item=params, save_args=jax.tree.map(lambda _: default_save_args, params)
        ),
    }
    # Exclude optimizer state if the flag is set OR if learn_to_init_mode is active.
    exclude_opt = self.student_config.learn_to_init_mode

    if optimizer is not None and not exclude_opt:
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
    """Restores model + optimizer by delegating to upstream Tunix.

    Unwraps `ModelBundle` if present (we only restore `student_model`).

    Returns:
      (restored step, custom_metadata dict). Step is 0 if no checkpoint exists.
    """
    if self._checkpoint_manager is None:
      return 0, {}

    target_model = getattr(model, "student_model", model)

    step, _ = super().maybe_restore(
        model=target_model,
        optimizer=optimizer,
        restore_only_lora_params=restore_only_lora_params,
    )
    if step == 0:
      return 0, {}

    max_logging.log(f"Restored from checkpoint step {step}.")

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
