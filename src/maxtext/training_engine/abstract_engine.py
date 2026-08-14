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

Defines the core Trainer interface, republishes the data types shared with
Tunix (WeightedMetric, LossOutput, TrainerPayload), and defines the on-device
MetricsBuffer used by the training loop.

Note: this module imports Tunix, so `training_engine` requires MaxText's
post-training dependency bundle (`google-tunix`, declared in
`tpu-post-train-requirements.txt`) rather than the base install. That is
acceptable because `training_engine` is post-training tier by intent and
nothing under `src/` imports it.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
from typing import Any

import flax.struct
import jax

# The shared trainer data types below are deliberately Tunix's, re-exported
# rather than redefined. Do not "fix" this by declaring local copies.
#
# `training_engine` is driven by Tunix's TrainerWorker, so a Tunix loss function
# (e.g. `tunix.rl.algo_core.grpo_loss_fn`) returns a `tunix.sft.utils.LossOutput`.
# MaxText previously declared its own field-identical `LossOutput`/`WeightedMetric`,
# which made every `isinstance` check in `maxtext_engine.diff_wrapper` miss on a
# Tunix loss and fail with "Unsupported return type from loss function". Sharing
# the class object makes those checks pass by construction, and removes a
# duplicate that had already drifted (the two `compute_scale` implementations
# applied `eps` and `min_denom` in opposite orders).
#
# `TrainerPayload` is adopted for a related reason: what actually reaches
# `fwd_bwd` at runtime is Tunix's `RLTrainerPayload`, which was never a subclass
# of MaxText's identically-named class, so the annotation described a type that
# never appeared.
#
# WeightedMetric must come from `tunix.sft.utils`, NOT from
# `tunix.experimental.metrics.metrics` -- the latter declares a same-named class
# whose `compute()` and `compute_scale()` raise NotImplementedError.
#
# The redundant-alias form (`X as X`) marks these as intentional re-exports so
# linters do not flag them as unused imports.
from tunix.experimental.common.datatypes import TrainerPayload as TrainerPayload
from tunix.sft.utils import LossOutput as LossOutput
from tunix.sft.utils import WeightedMetric as WeightedMetric


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

  @property
  @abc.abstractmethod
  def train_step(self) -> int:
    """Returns the current training step integer."""

  @abc.abstractmethod
  def close(self) -> None:
    """Cleans up engine resources and blocks until async saves complete."""
