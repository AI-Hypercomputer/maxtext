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

"""MaxText concrete implementation of AbstractTrainer for RL post-training.

Adapts MaxText's single-step compilation and execution primitives to implement
the MaxRL AbstractTrainer interface without running an outer loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train as maxtext_train
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import checkpointing
from maxtext.training_engine import metrics
from maxtext.utils import gradient_accumulation
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import train_utils


class MaxTextTrainingEngine(abstract_engine.AbstractTrainingEngine):
  """Concrete trainer wrapping MaxText single-step SPMD execution for NNX models."""

  def __init__(
      self,
      training_config: pyconfig.HyperParameters,
      mesh: jax.sharding.Mesh | None = None,
  ) -> None:
    """Initializes the MaxText trainer state and sharded model.

    Args:
      training_config: MaxText HyperParameters configuration instance.
      mesh: Optional SPMD device mesh.

    Raises:
      TypeError: If training_config is not a pyconfig.HyperParameters instance.
      ValueError: If training_config.model_name is not specified or empty.
    """
    if not isinstance(training_config, pyconfig.HyperParameters):
      raise TypeError(
          "MaxTextTrainingEngine requires a pyconfig.HyperParameters instance," f" got {type(training_config).__name__}"
      )
    self._config = training_config
    self._mesh = mesh
    self._init_rng = jax.random.PRNGKey(getattr(training_config, "init_weights_seed", 0))
    self._loss_fn: Callable[..., Any] | None = None
    self._gen_model_input_fn: Callable[[Any], dict[str, Any]] | None = None
    self._compiled = False
    if not getattr(training_config, "model_name", None):
      raise ValueError("training_config.model_name must be specified")
    self._model = model_creation_utils.from_pretrained(
        config=self._config,
        mesh=self._mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=self._init_rng,
    )
    self._state: Any = None
    self._accumulated_grads: Any = None
    self._micro_step_count = 0
    self._cached_losses: list[jax.Array] = []
    self._learning_rate_schedule, self._optimizer = train_utils.create_training_optimizer(self._config, self._model)
    self._train_step: int = 0

    self._checkpoint_manager = checkpointing.CheckpointManager(
        checkpoint_dir=getattr(self._config, "checkpoint_directory", ""),
        config=self._config,
    )
    self._metrics_logger = metrics.MetricsLogger(self._config)

  @property
  def model(self) -> Any:
    """Returns the NNX model instance."""
    return self._model

  @model.setter
  def model(self, new_model: Any) -> None:
    """Sets the NNX model instance."""
    self._model = new_model

  @property
  def optimizer(self) -> Any:
    """Returns the NNX optimizer instance."""
    return self._optimizer

  @optimizer.setter
  def optimizer(self, new_optimizer: Any) -> None:
    """Sets the NNX optimizer instance."""
    self._optimizer = new_optimizer

  @property
  def train_step(self) -> int:
    """Returns the current step integer."""
    return self._train_step

  @train_step.setter
  def train_step(self, step: int) -> None:
    """Sets the current step integer."""
    self._train_step = step

  def with_loss_fn(self, customized_fn: Callable[..., Any]) -> None:
    """Overrides the default autoregressive loss function with a custom RL loss.

    Args:
      customized_fn: Custom loss callable matching the MaxText loss signature.
    """
    self._loss_fn = customized_fn
    self._compiled = False

  def with_gen_model_input_fn(self, gen_model_input_fn: Callable[[Any], dict[str, Any]]) -> "MaxTextTrainingEngine":
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs."""
    self._gen_model_input_fn = gen_model_input_fn
    return self

  def compile(self, dummy_data: abstract_engine.TrainerPayload) -> None:
    """Triggers SPMD JIT compilation of fwd_bwd, update, and eval steps.

    Args:
      dummy_data: Sample TrainerPayload providing representative tensor shapes.
    """
    self._compiled = True

  def fwd_bwd(self, payload: abstract_engine.TrainerPayload) -> None:
    """Executes a micro-batch forward-backward pass and accumulates gradients.

    Args:
      payload: Packed micro-batch training input.
    """
    if self._gen_model_input_fn is not None:
      batch = self._gen_model_input_fn(payload)
    else:
      batch = payload
    loss_callable = self._loss_fn if self._loss_fn is not None else maxtext_train.loss_fn

    model = getattr(self._state, "model", None) if self._state is not None else self._model
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxRL requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")
    # TODO(mazumdera): This function call should be pre-compiled.
    loss, _, micro_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        loss_callable,
        self._config,
        model,
        None,
        None,
        batch,
        None,
    )

    if isinstance(loss, abstract_engine.WeightedMetric):
      self.record_metrics("loss", loss)

    self._cached_losses.append(loss)
    if self._accumulated_grads is None:
      self._accumulated_grads = micro_grads
    else:
      self._accumulated_grads = jax.tree.map(jnp.add, self._accumulated_grads, micro_grads)
    self._micro_step_count += 1

  def update(self) -> None:
    """Applies accumulated gradients to update NNX model weights in HBM.

    Reuses NNX optimizer step from train.py (lines 511-535).
    """
    if self._accumulated_grads is None:
      return
    # TODO(mazumdera): The logic below should be pre-compiled.
    if self._state is not None:
      # TODO(mazumdera): Figure out how exactly we should normalize the losses
      # (if at all). Given that inputs are varying in size, it not correct to
      # simply divide by the number of micro-steps.
      grads = jax.tree.map(lambda g: g / max(self._micro_step_count, 1), self._accumulated_grads)
      if getattr(self._config, "gradient_clipping_threshold", 0.0) > 0:
        grads = maxtext_utils.apply_gradient_clipping(grads, None, self._config.gradient_clipping_threshold)
      if hasattr(self._state, "apply_gradients"):
        if getattr(self._config, "skip_step_on_spikes", False):
          grad_norm = max_utils.l2norm_pytree(grads)
          mean_loss = jnp.mean(jnp.array(self._cached_losses)) if self._cached_losses else jnp.array(0.0)
          self._state.apply_gradients(grads, loss=mean_loss, grad_norm=grad_norm)
        else:
          self._state.apply_gradients(grads)
    self._cached_losses.clear()
    self._accumulated_grads = None
    self._micro_step_count = 0
    if hasattr(self, "_learning_rate_schedule") and self._learning_rate_schedule is not None:
      try:
        lr = self._learning_rate_schedule(self.train_step)
        self.record_metrics("lr", lr)
      except Exception:  # pylint: disable=broad-except
        pass
    self._train_step += 1

  def eval_step(self, payload: abstract_engine.TrainerPayload, **kwargs: Any) -> None:
    """Executes an evaluation step on the given payload.

    Args:
      payload: Packed micro-batch evaluation input.
      **kwargs: Additional keyword arguments for evaluation.
    """

  def save_checkpoint(self, metadata: Any, **kwargs: Any) -> None:
    """Forces asynchronous Orbax checkpoint serialization.

    Args:
      metadata: Checkpoint metadata payload from Orchestrator.
      **kwargs: Additional checkpoint saving options.
    """
    step = kwargs.get("step", self.train_step)
    # TODO(mazumdera): Also save self._accumulated_grads and _micro_step_count.
    ckpt_saved = self._checkpoint_manager.save_checkpoint(
        step=step,
        model=self.model,
        optimizer=self.optimizer,
        custom_metadata=metadata,
    )
    if ckpt_saved:
      logging.info("Checkpoint saved at step %d.", step)

  def restore_checkpoint(self, **kwargs: Any) -> Any:
    """Restores the latest Multi-Tier Checkpoint and returns its metadata.

    Args:
      **kwargs: Additional checkpoint restoration options.

    Returns:
      The metadata PyTree of the restored checkpoint.
    """
    step = kwargs.get("step", None)
    restored_step, restored_metadata = self._checkpoint_manager.restore_checkpoint(
        model=self.model,
        optimizer=self.optimizer,
        step=step,
    )
    if restored_step:
      logging.info("Checkpoint restored from step %d.", restored_step)
      self.train_step = restored_step
    return restored_metadata

  def record_metrics(
      self,
      name: str,
      metric: abstract_engine.WeightedMetric | jax.Array | float | int,
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
  ) -> None:
    """Records a metric into the buffer, appending to JAX arrays.

    Args:
      name: The name of the metric.
      metric: The metric to record.
      aggregation_fn: The aggregation function to apply to the metric.
    """
    self._metrics_logger.buffer_metrics(
        train_step=self.train_step,
        name=name,
        metric=metric,
        aggregation_fn=aggregation_fn,
    )

  def get_metrics(self, clear_cache: bool = True) -> abstract_engine.MetricsBuffer:
    """Returns accumulated step metrics as an on-device MetricsBuffer.

    Args:
      clear_cache: Whether to reset cached metrics after retrieval.

    Returns:
      On-device MetricsBuffer containing WeightedMetric and scalar arrays.
    """
    return self._metrics_logger.get_metrics(clear_cache=clear_cache)

  def prepare_weight_sync(self, **kwargs: Any) -> Any:
    """Stages weights for transfer and returns access coordinates.

    Args:
      **kwargs: Weight staging parameters.

    Returns:
      Synchronization endpoints or coordinates for rollout actors.
    """
    return {}

  def close(self) -> None:
    """Closes the trainer and its associated resources."""
    self._checkpoint_manager.close()
