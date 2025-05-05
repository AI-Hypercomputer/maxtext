# Copyright 2025 Google LLC
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

"""Implements Distillation Strategies."""

import abc
from typing import Any, Callable, Protocol, TypeAlias, TypeVar

from flax import nnx
import jax
import jax.numpy as jnp
import optax

ArrayLike: TypeAlias = jax.typing.ArrayLike
R = TypeVar("R")


class ModelForwardCallable(Protocol[R]):

  def __call__(self, model: nnx.Module, *args: Any, **kwargs: Any) -> R:
    ...


class BaseDistillationStrategy(abc.ABC):
  """Abstract Base Class for all distillation strategies.

  Defines the common interface for computing the distillation loss. Concrete
  strategies must implement the `compute_loss` method.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[Any],
      teacher_forward_fn: ModelForwardCallable[Any],
      labels_fn: Callable[..., ArrayLike],
  ):
    """Initializes the BaseDistillationStrategy.

    Args:
      student_forward_fn: Function to compute student model outputs.
      teacher_forward_fn: Function to compute teacher model outputs.
      labels_fn: Function to compute labels from model inputs.
    """
    self._student_forward_fn = student_forward_fn
    self._teacher_forward_fn = teacher_forward_fn
    self._labels_fn = labels_fn

  @abc.abstractmethod
  def compute_loss(
      self, student_output: Any, teacher_output: Any, labels: ArrayLike
  ) -> ArrayLike:
    """Computes the distillation loss based on model outputs and labels."""

  @abc.abstractmethod
  def compute_eval_loss(
      self, student_output: Any, labels: ArrayLike
  ) -> ArrayLike:
    """Computes the distillation loss based on model outputs and labels."""

  def get_teacher_outputs(
      self,
      teacher_model: nnx.Module,
      inputs: dict[str, ArrayLike],
  ) -> Any:
    """Computes the teacher model outputs."""
    return self._teacher_forward_fn(teacher_model, **inputs)

  def get_train_loss(
      self,
      student_model: nnx.Module,
      teacher_output: Any,
      inputs: dict[str, ArrayLike],
  ) -> ArrayLike:
    """Computes the distillation loss."""
    student_output = self._student_forward_fn(student_model, **inputs)
    labels = self._labels_fn(**inputs)
    return self.compute_loss(
        student_output=student_output,
        teacher_output=teacher_output,
        labels=labels,
    )

  def get_eval_loss(
      self,
      student_model: nnx.Module,
      inputs: dict[str, ArrayLike],
  ) -> ArrayLike:
    """Computes the task loss based on student model forward pass and labels."""
    student_output = self._student_forward_fn(student_model, **inputs)
    labels = self._labels_fn(**inputs)
    return self.compute_eval_loss(
        student_output=student_output,
        labels=labels,
    )


class LogitDistillation(BaseDistillationStrategy):
  """Implements Logit Distillation.

  This strategy minimizes the KL divergence between the student and teacher
  logits. It combines this attention loss with a standard task loss (softmax
  cross-entropy) on the student's final logits.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[ArrayLike],
      teacher_forward_fn: ModelForwardCallable[ArrayLike],
      labels_fn: Callable[..., ArrayLike],
      temperature: float = 2.0,
      alpha: float = 0.5,
  ):
    """Initializes the LogitDistillation strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        labels_fn: Function to compute labels from model inputs.
        temperature: Temperature for softening probabilities (> 0).
        alpha: Weight to balance distillation loss and task loss (0.0 to 1.0).
    """
    super().__init__(student_forward_fn, teacher_forward_fn, labels_fn)
    if temperature <= 0:
      raise ValueError(
          f"Temperature must be a positive number, got {temperature}"
      )
    if not (0.0 <= alpha <= 1.0):
      raise ValueError(
          f"Alpha must be a float between 0.0 and 1.0, got {alpha}"
      )

    self.temperature = float(temperature)
    self.alpha = alpha

  def compute_eval_loss(
      self,
      student_output: ArrayLike,
      labels: ArrayLike,
  ) -> ArrayLike:
    """Computes the task loss.

    Args:
        student_output: The logits from the student model.
        labels: The ground truth labels for the examples.

    Returns:
        A JAX array representing the task loss, averaged over the batch.
    """
    # Calculate Task Loss (Cross-Entropy on original student logits)
    ce_loss = optax.softmax_cross_entropy(logits=student_output, labels=labels)
    task_loss = jnp.mean(ce_loss)

    return task_loss

  def compute_loss(
      self,
      student_output: ArrayLike,
      teacher_output: ArrayLike,
      labels: ArrayLike,
  ) -> ArrayLike:
    """Computes the loss for logit distillation.

    Args:
        student_output: The logits from the student model.
        teacher_output: The logits from the teacher model.
        labels: The ground truth labels for the examples.

    Returns:
        A JAX array representing the combined distillation and task loss,
        averaged over the batch.
    """
    # Calculate Distillation Loss (KL Divergence on softened targets)
    log_student_probs_temp = jax.nn.log_softmax(
        student_output / self.temperature, axis=-1
    )
    teacher_probs_temp = jax.nn.softmax(
        teacher_output / self.temperature, axis=-1
    )
    kl_loss = optax.kl_divergence(log_student_probs_temp, teacher_probs_temp)
    # Applying temperature scaling to KL loss as per
    # (https://arxiv.org/pdf/1503.02531)
    scaled_kl_loss = kl_loss * (self.temperature**2)
    distillation_loss = jnp.mean(scaled_kl_loss)

    # Calculate Task Loss (Cross-Entropy on original student logits)
    ce_loss = optax.softmax_cross_entropy(logits=student_output, labels=labels)
    task_loss = jnp.mean(ce_loss)

    # Combine the losses
    combined_loss = (self.alpha * distillation_loss) + (
        (1.0 - self.alpha) * task_loss
    )

    return combined_loss


class AttentionTransfer(BaseDistillationStrategy):
  """Implements Attention Transfer distillation.

  This strategy minimizes the difference (typically MSE) between student and
  teacher attention maps from attention layers. It combines this attention loss
  with a standard task loss (softmax cross-entropy) on the student's final
  logits.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[dict[str, ArrayLike]],
      teacher_forward_fn: ModelForwardCallable[ArrayLike],
      labels_fn: Callable[..., ArrayLike],
      alpha: float = 0.5,
      attention_loss_fn: (
          Callable[[ArrayLike, ArrayLike], ArrayLike] | None
      ) = None,
  ):
    """Initializes the AttentionTransfer strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        labels_fn: Function to compute labels from model inputs.
        alpha: Weight to balance attention loss and task loss (0.0 to 1.0).
        attention_loss_fn: A function that takes two jax. Arrays (student_map,
          teacher_map) and returns a scalar loss. Defaults to Mean Squared
          Error.
    """
    super().__init__(student_forward_fn, teacher_forward_fn, labels_fn)
    if not (0.0 <= alpha <= 1.0):
      raise ValueError(
          f"Alpha must be a float between 0.0 and 1.0, got {alpha}"
      )

    self.alpha = alpha
    self.attention_loss_fn = attention_loss_fn or (
        lambda s, t: jnp.mean(jnp.square(s - t))
    )

  def compute_eval_loss(
      self,
      student_output: dict[str, ArrayLike],
      labels: ArrayLike,
  ) -> ArrayLike:
    """Computes the task loss.

    Args:
        student_output: Dictionary from student model, must contain 'logits'.
        labels: The ground truth labels for the examples.

    Returns:
        A JAX array representing the task loss, averaged over the batch.
    """
    # Calculate Task Loss (Cross-Entropy on original student logits)
    ce_loss = optax.softmax_cross_entropy(
        logits=student_output["logits"], labels=labels
    )
    task_loss = jnp.mean(ce_loss)

    return task_loss

  def compute_loss(
      self,
      student_output: dict[str, ArrayLike],
      teacher_output: ArrayLike,
      labels: ArrayLike,
  ) -> ArrayLike:
    """Computes the combined attention transfer and task loss.

    Args:
        student_output: Dictionary from student model, must contain 'logits' and
          'attentions'.
        teacher_output: List of teacher attention arrays.
        labels: The ground truth labels (e.g., one-hot encoded).

    Returns:
        A JAX array representing the combined distillation and task loss,
        averaged over the batch.
    """
    if "logits" not in student_output or "attentions" not in student_output:
      raise TypeError(
          "student_outputs must be a dict containing 'logits' and 'attentions'"
      )

    student_logits = student_output["logits"]
    student_attentions = student_output["attentions"]

    attention_loss = jnp.nan_to_num(
        self.attention_loss_fn(student_attentions, teacher_output)
    )

    # Calculate Task Loss (Cross-Entropy)
    ce_loss_per_example = optax.softmax_cross_entropy(
        logits=student_logits, labels=labels
    )
    task_loss = jnp.mean(ce_loss_per_example)

    # Combine the losses
    combined_loss = (self.alpha * attention_loss) + (
        (1.0 - self.alpha) * task_loss
    )

    return combined_loss
