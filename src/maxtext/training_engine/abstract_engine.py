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

"""Trainer abstractions.

Defines the core Trainer interface, data payload interfaces, and on-device
metrics structures (WeightedMetric, MetricsBuffer) used by the training loop.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
from typing import Any

import flax.struct
import jax
from jax.typing import ArrayLike  # pylint: disable=g-importing-member


@flax.struct.dataclass
class WeightedMetric:
  """A metric that requires weighted reduction.

  Attributes:
    unreduced_sum: Sum of the metric values across tokens/examples.
    denominator: Weight or count of valid tokens/examples.
    eps: Optional epsilon added to denominator for numerical stability.
    min_denom: Optional minimum bound for the denominator.
  """

  unreduced_sum: jax.Array
  denominator: jax.Array
  eps: float | None = flax.struct.field(default=None, pytree_node=False)
  min_denom: float | None = flax.struct.field(default=None, pytree_node=False)

  def compute_scale(self) -> jax.Array:
    """Safely computes the scale factor (1 / denominator) with bounds.

    Returns:
      Safe scaling factor array preventing division-by-zero NaNs.
    """
    denom = self.denominator
    if self.min_denom is not None:
      denom = jax.numpy.maximum(denom, self.min_denom)
    if self.eps is not None:
      denom = denom + self.eps
    safe_denom = jax.numpy.where(denom == 0, 1.0, denom)
    scale = 1.0 / safe_denom
    return jax.numpy.where(denom == 0, 0.0, scale)

  def compute(self) -> jax.Array:
    """Safely computes total / count with numerical stability bounds.

    Returns:
      Reduced metric array equal to unreduced_sum * compute_scale().
    """
    return self.unreduced_sum * self.compute_scale()


@flax.struct.dataclass
class MetricsBuffer:
  """A buffer for storing and aggregating unreduced metrics on-device.

  Attributes:
    id: Identifier for the buffer (e.g., training iteration or step index).
    weighted_metrics: Dictionary of WeightedMetric objects on accelerator HBM.
    scalar_metrics: Dictionary of scalar JAX arrays on accelerator HBM.
    aggregation_fns: Host-side reduction/aggregation callbacks (untraced).
    mode: Execution mode string ("train" or "eval").
  """

  id: Any
  weighted_metrics: dict[str, WeightedMetric] = flax.struct.field(default_factory=dict)
  scalar_metrics: dict[str, jax.Array] = flax.struct.field(default_factory=dict)
  aggregation_fns: dict[str, Callable[[jax.Array], Any]] = flax.struct.field(default_factory=dict, pytree_node=False)
  mode: str = flax.struct.field(default="train", pytree_node=False)


@dataclasses.dataclass(kw_only=True)
class TrainerPayload(abc.ABC):
  """Base class for packed micro-batches ready for gradient descent.

  The base carries only what generic machinery must read to stay
  algorithm-agnostic. Algorithm-specific tensors live on subclasses and are
  reached by the trainer's gen_model_input_fn, not by the generic loop. Users
  subclass this to carry their own fields.

  Attributes:
    token_ids: [B, T] token IDs. By default, structured as left-padded prompt
      tokens concatenated with right-padded completion tokens.
    token_mask: [B, T] token mask to differentiate padding tokens from valid
      tokens.
    segment_ids: Optional [B, T] packing segment ids.
  """

  token_ids: ArrayLike
  token_mask: ArrayLike
  segment_ids: ArrayLike | None = None


@dataclasses.dataclass
class TrainingConfig:
  """Configuration for the abstract trainer.

  Defines standard hyperparameters and operational settings for the ML training
  loop.
  """

  eval_every_n_steps: int = 0
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None
  checkpoint_root_directory: str | None = None
  metrics_prefix: str = ""
  max_inflight_computations: int = 2


class AbstractTrainingEngine(abc.ABC):
  """Core trainer interface executing model updates and Multi-Tier Checkpointing.

  The Trainer owns the model weights in accelerator HBM and executes forward/
  backward passes, weight updates, evaluation steps, and checkpoint saving/
  restoring.
  """

  @abc.abstractmethod
  def __init__(self, training_config: TrainingConfig) -> None:
    """Initializes the Trainer based on the training configuration.

    Args:
      training_config: Training hyperparameters and runtime configuration.
    """

  @abc.abstractmethod
  def with_loss_fn(self, customized_fn: Callable[..., Any]) -> None:
    """Updates the trainer's loss function.

    Args:
      customized_fn: Custom loss function callable.
    """

  @abc.abstractmethod
  def with_gen_model_input_fn(self, gen_model_input_fn: Callable[[Any], dict[str, Any]]) -> "AbstractTrainingEngine":
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs.

    This adapter enables the trainer to accept arbitrary payloads (SFT, RL,
    etc.) by transforming them into kwargs for the loss function via
    `gen_model_input_fn(payload)`.
    Args:
      gen_model_input_fn: Maps a payload to a dict of loss-fn keyword arguments.

    Returns:
      self, for chaining.
    """

  @abc.abstractmethod
  def compile(self, dummy_data: TrainerPayload) -> None:
    """Triggers JAX compilation. `with_loss_fn` must be called first.

    Args:
      dummy_data: Payload with representative shapes used for JAX tracing.
    """

  @abc.abstractmethod
  def fwd_bwd(self, payload: TrainerPayload) -> None:
    """Executes forward and backward passes.

    Metrics are cached to overlap train steps.

    Args:
      payload: Packed micro-batch payload for training.
    """

  @abc.abstractmethod
  def update(self) -> None:
    """Executes a model weight update step using accumulated gradients."""

  @abc.abstractmethod
  def eval_step(self, payload: TrainerPayload, **kwargs: Any) -> None:
    """Executes one evaluation step on the given payload.

    Args:
      payload: Packed micro-batch payload for evaluation.
      **kwargs: Additional evaluation keyword arguments.
    """

  @abc.abstractmethod
  def save_checkpoint(self, metadata: Any, **kwargs: Any) -> None:
    """Forces the trainer to serialize its state (model + optimizer).

    Args:
      metadata: Checkpoint identifier or UUID metadata pytree.
      **kwargs: Additional checkpointing keyword arguments.
    """

  @abc.abstractmethod
  def restore_checkpoint(self, **kwargs: Any) -> Any:
    """Restores state from latest checkpoint and returns the metadata pytree.

    The returned metadata (e.g., global_step) matches what was stored in
    save_checkpoint.

    Args:
      **kwargs: Additional restoration keyword arguments.

    Returns:
      The metadata PyTree stored with the checkpoint.
    """

  @abc.abstractmethod
  def get_metrics(self, clear_cache: bool = True) -> MetricsBuffer:
    """Returns cached metrics and optionally clears the metrics cache.

    Args:
      clear_cache: Whether to reset cached metrics after retrieval.

    Returns:
      The accumulated on-device MetricsBuffer.
    """

  @abc.abstractmethod
  def prepare_weight_sync(self, **kwargs: Any) -> Any:
    """Stages weights for transfer and returns metadata/coordinates.

    Args:
      **kwargs: Weight staging configuration parameters.

    Returns:
      Synchronization endpoints or file coordinates for weight transfer.
    """
