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
import dataclasses
from typing import Any

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train as maxtext_train
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import checkpointing
from maxtext.training_engine import inflight_throttler
from maxtext.training_engine import metrics as metrics_module
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
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
    self._init_rng = jax.random.PRNGKey(training_config.init_weights_seed)
    self._loss_fn: Callable[..., Any] | None = None
    self._gen_model_input_fn: Callable[[Any], dict[str, Any]] | None = None
    self._compiled = False
    if not training_config.model_name:
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
    self._cached_losses: list[abstract_engine.WeightedMetric | jax.Array] = []
    self._learning_rate_schedule, self._optimizer = train_utils.create_training_optimizer(self._config, self._model)
    self._train_step: int = 0

    self._checkpoint_manager = checkpointing.CheckpointManager(
        checkpoint_dir=self._config.checkpoint_dir,
        config=self._config,
    )
    self._metrics_recorder = metrics_module.MetricsRecorder()
    self._throttler = inflight_throttler.InflightThrottler(config=self._config)

  @property
  def model(self) -> Any:
    """Returns the NNX model instance."""
    return self._model

  @model.setter
  def model(self, new_model: Any) -> None:
    """Sets the NNX model instance."""
    self._model = new_model
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_update = None
    self._model_graphdef = None

  @property
  def optimizer(self) -> Any:
    """Returns the NNX optimizer instance."""
    return self._optimizer

  @optimizer.setter
  def optimizer(self, new_optimizer: Any) -> None:
    """Sets the NNX optimizer instance."""
    self._optimizer = new_optimizer
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_update = None
    self._state_graphdef = None

  @property
  def train_step(self) -> int:
    """Returns the current step integer."""
    return self._train_step

  @train_step.setter
  def train_step(self, step: int) -> None:
    """Sets the current step integer."""
    self._train_step = step

  @property
  def state(self) -> Any:
    """Returns the current train state, initializing it if necessary."""
    if self._state is None and self._model is not None and self._optimizer is not None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    return self._state

  @state.setter
  def state(self, new_state: Any) -> None:
    """Sets the current train state."""
    self._state = new_state
    self._compiled = False
    self._compiled_fwd_bwd = None
    self._compiled_update = None
    self._state_graphdef = None

  @property
  def micro_step_count(self) -> int:
    """Returns the current micro-batch count in gradient accumulation."""
    return self._micro_step_count

  @property
  def has_accumulated_grads(self) -> bool:
    """Returns True if accumulated gradients are present."""
    return self._accumulated_grads is not None

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

  def _fwd_bwd_kernel(self, params, rest, batch):
    """Executes a single forward and backward pass to compute gradients."""
    loss_callable = self._loss_fn if self._loss_fn is not None else maxtext_train.loss_fn

    def diff_wrapper(p, r, b):
      mdl = nnx.merge(self._model_graphdef, p, r, copy=True)
      out = loss_callable(mdl, self._config, b, None, None, is_train=True)
      _, _, new_r = nnx.split(mdl, nnx.Param, ...)

      if isinstance(out, abstract_engine.LossOutput):
        return out.primary_loss.unreduced_sum, (out, new_r)
      elif isinstance(out, abstract_engine.WeightedMetric):
        loss_out = abstract_engine.LossOutput(
            primary_loss=out,
            aux_metrics={},
        )
        return out.unreduced_sum, (loss_out, new_r)
      elif isinstance(out, (tuple, list)) and len(out) == 2:
        loss_val, aux = out
        if isinstance(loss_val, abstract_engine.WeightedMetric):
          primary_loss = loss_val
        elif isinstance(aux, dict) and "xent_sum" in aux and "total_weights" in aux:
          primary_loss = abstract_engine.WeightedMetric(
              unreduced_sum=aux["xent_sum"],
              denominator=aux["total_weights"],
          )
        else:
          raise TypeError(
              f"Cannot construct WeightedMetric from 2-tuple loss return with elements "
              f"of type ({type(loss_val).__name__}, {type(aux).__name__}). Expected first element to be a "
              "WeightedMetric, or second element to be a dict containing 'xent_sum' and 'total_weights'."
          )

        loss_out = abstract_engine.LossOutput(
            primary_loss=primary_loss,
            aux_metrics=aux if isinstance(aux, dict) else {},
        )
        return primary_loss.unreduced_sum, (loss_out, new_r)
      else:
        raise TypeError(
            f"Unsupported return type from loss function: {type(out)}. "
            "Expected abstract_engine.LossOutput, abstract_engine.WeightedMetric, "
            "or a 2-element tuple/list: (loss, aux_metrics)."
        )

    grad_func = jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)
    (loss_val, (loss_out, new_rest)), micro_grads = grad_func(params, rest, batch)
    if isinstance(loss_out, abstract_engine.LossOutput):
      scale = loss_out.primary_loss.compute_scale()
      micro_grads = jax.tree.map(lambda g: g * scale, micro_grads)

    micro_grads = jax.tree.map(
        lambda x: (x.astype(self._config.grad_dtype) if hasattr(x, "dtype") and x.dtype == jnp.float32 else x),
        micro_grads,
    )

    if isinstance(loss_out, abstract_engine.LossOutput):
      return loss_out.primary_loss, loss_out.aux_metrics, new_rest, micro_grads
    else:
      return loss_val, {}, new_rest, micro_grads

  def _update_kernel(self, state_pure, accumulated_grads, micro_step_count, mean_loss):
    """Applies accumulated gradients to update the NNX model state."""
    grad_norm = None
    is_skipped_val = None
    if state_pure is not None:
      if micro_step_count <= 1:
        grads = accumulated_grads
      else:
        grads = jax.tree.map(
            lambda g: g / micro_step_count,
            accumulated_grads,
        )
      if self._config.gradient_clipping_threshold > 0:
        grads = maxtext_utils.apply_gradient_clipping(grads, None, self._config.gradient_clipping_threshold)
      local_state = nnx.merge(self._state_graphdef, state_pure, copy=True)
      if hasattr(local_state, "apply_gradients"):
        if self._config.skip_step_on_spikes:
          grad_norm = max_utils.l2norm_pytree(grads)
          local_state.apply_gradients(grads, loss=mean_loss, grad_norm=grad_norm)
          opt_obj = getattr(local_state, "optimizer", self._optimizer)
          if opt_obj is not None:
            opt_state = nnx.to_pure_dict(nnx.state(opt_obj)).get("opt_state", {})
            is_skipped = opt_state.get("is_skipped") if isinstance(opt_state, dict) else None
            if is_skipped is not None:
              is_skipped_val = is_skipped.astype(jnp.float32)
        else:
          local_state.apply_gradients(grads)
      _, new_state_pure = nnx.split(local_state)
      return new_state_pure, grad_norm, is_skipped_val
    return state_pure, grad_norm, is_skipped_val

  def compile(self, dummy_data: abstract_engine.TrainerPayload) -> None:
    """Triggers SPMD JIT compilation of fwd_bwd and update steps.

    Args:
      dummy_data: Sample TrainerPayload providing representative tensor shapes.
    """
    if self._compiled:
      return

    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)

    self._state_graphdef, state_pure = nnx.split(self._state)
    self._model_graphdef, params_pure, rest_pure = nnx.split(self._model, nnx.Param, ...)

    if self._mesh is not None:
      data_sharding = sharding.get_input_data_sharding(self._config, self._mesh)
      state_mesh_shardings = jax.tree.map(
          lambda x: getattr(
              x,
              "sharding",
              jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec()),
          ),
          state_pure,
      )
      params_shardings = jax.tree.map(
          lambda x: getattr(
              x,
              "sharding",
              jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec()),
          ),
          params_pure,
      )
      rest_shardings = jax.tree.map(
          lambda x: getattr(
              x,
              "sharding",
              jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec()),
          ),
          rest_pure,
      )
      fwd_bwd_in_shardings = (params_shardings, rest_shardings, data_sharding)
      fwd_bwd_out_shardings = (None, None, rest_shardings, params_shardings)
      update_in_shardings = (state_mesh_shardings, params_shardings, None)
      update_out_shardings = (state_mesh_shardings, None, None)
    else:
      fwd_bwd_in_shardings = None
      fwd_bwd_out_shardings = None
      update_in_shardings = None
      update_out_shardings = None

    # 1. JIT Compile Micro FWD/BWD Pass
    self._compiled_fwd_bwd = jax.jit(
        self._fwd_bwd_kernel,
        in_shardings=fwd_bwd_in_shardings,
        out_shardings=fwd_bwd_out_shardings,
    )

    # 2. JIT Compile Optimizer Update Pass
    self._compiled_update = jax.jit(
        self._update_kernel,
        in_shardings=update_in_shardings,
        out_shardings=update_out_shardings,
        static_argnums=(2,),
    )
    self._compiled = True

  def fwd_bwd(self, payload: abstract_engine.TrainerPayload) -> None:
    """Executes a micro-batch forward-backward pass and accumulates gradients.

    Args:
      payload: Packed micro-batch training input.
    """
    if self._gen_model_input_fn is not None:
      batch = self._gen_model_input_fn(payload)
    elif dataclasses.is_dataclass(payload):
      batch = {k: getattr(payload, k) for k in payload.__dataclass_fields__ if getattr(payload, k) is not None}
    else:
      batch = payload

    model = getattr(self._state, "model", None) if self._state is not None else self._model
    if not isinstance(model, nnx.Module):
      raise TypeError("MaxTextTrainingEngine requires an NNX model (flax.nnx.Module), got" f" {type(model).__name__}")

    # Wait for previous computations to finish before dispatching the next one to TPU.
    self._throttler.wait_for_next()

    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    model = getattr(self._state, "model", self._model)
    self._model_graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    if self._compiled and hasattr(self, "_compiled_fwd_bwd"):
      loss, aux, new_rest, micro_grads = self._compiled_fwd_bwd(params, rest, batch)
    else:
      loss, aux, new_rest, micro_grads = self._fwd_bwd_kernel(params, rest, batch)
    nnx.update(model, new_rest)

    # Don't add metrics to the throttler queue because metrics are logged after
    # the update step.
    self._throttler.add_computation(computation=loss, metrics=None)

    if isinstance(loss, abstract_engine.WeightedMetric):
      self.record_metrics("loss", loss)

    # Record auxiliary metrics.
    if isinstance(aux, dict):
      for key, value in aux.items():
        if value is not None:
          self.record_metrics(key, value)

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

    if self._learning_rate_schedule is not None:
      lr = self._learning_rate_schedule(self.train_step)
      self.record_metrics("learning_rate", lr)

    # Wait for previous computations to finish before dispatching the update step to TPU.
    self._throttler.wait_for_next()

    # TODO(mazumdera): The logic below should be pre-compiled.
    if self._state is None:
      self._state = train_state_nnx.TrainStateNNX(self._model, self._optimizer)
    self._state_graphdef, state_pure = nnx.split(self._state)

    if self._cached_losses:
      loss_values = [l.compute() if isinstance(l, abstract_engine.WeightedMetric) else l for l in self._cached_losses]
      mean_loss = jnp.mean(jnp.stack(loss_values)) if len(loss_values) > 1 else loss_values[0]
    else:
      mean_loss = jnp.array(0.0)
    if self._compiled and hasattr(self, "_compiled_update"):
      new_state_pure, grad_norm, is_skipped = self._compiled_update(
          state_pure, self._accumulated_grads, self._micro_step_count, mean_loss
      )
    else:
      new_state_pure, grad_norm, is_skipped = self._update_kernel(
          state_pure, self._accumulated_grads, self._micro_step_count, mean_loss
      )
    nnx.update(self._state, new_state_pure)

    if grad_norm is not None:
      self.record_metrics("gradient_norm", grad_norm)
    if is_skipped is not None:
      self.record_metrics("step_skipped", is_skipped)

    # Add the state to the throttler queue so jax.block_until_ready() waits
    # for the optimizer update to complete before logging the metrics.
    self._throttler.add_computation(
        self._state if self._state is not None else self._model,
        self._metrics_recorder.get_step_metrics(self.train_step),
    )

    self._cached_losses.clear()
    self._accumulated_grads = None
    self._micro_step_count = 0
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
    # Drain all inflight computations and log pending metrics before checkpointing.
    self._throttler.wait_for_all()

    step = kwargs.get("step", self.train_step)
    force_ckpt_save = kwargs.get("force", False)

    custom_metadata = {}
    if self._micro_step_count > 0:
      logging.info(
          "Saving intra-step checkpoint at step %d (micro_step_count=%d).",
          step,
          self._micro_step_count,
      )
      force_ckpt_save = True
      custom_metadata["micro_step_count"] = self._micro_step_count
    else:
      logging.info("Saving checkpoint at step %d.", step)

    if metadata:
      # Metadata from Orchestrator
      custom_metadata["additional_metadata"] = metadata

    ckpt_saved = self._checkpoint_manager.save_checkpoint(
        step=step,
        checkpoint_state=checkpointing.CheckpointState(
            model=self.model,
            optimizer=self.optimizer,
            accumulated_metrics=self.get_metrics(clear_cache=False),
            accumulated_grads=self._accumulated_grads,
        ),
        custom_metadata=custom_metadata,
        force_ckpt_save=force_ckpt_save,
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
    checkpoint_state = checkpointing.CheckpointState(
        model=self.model,
        optimizer=self.optimizer,
        accumulated_grads=self._accumulated_grads,
    )

    restored_step, restored_checkpoint_state, restored_metadata = self._checkpoint_manager.restore_checkpoint(
        checkpoint_state=checkpoint_state,
        step=step,
    )
    if restored_step is None:
      return None

    logging.info("Checkpoint restored from step %d.", restored_step)
    self.train_step = restored_step

    if restored_checkpoint_state.accumulated_metrics:
      buffers = []
      for b in restored_checkpoint_state.accumulated_metrics:
        if isinstance(b, dict):
          wms = {}
          for k, wm in b.get("weighted_metrics", {}).items():
            if isinstance(wm, dict):
              wms[k] = abstract_engine.WeightedMetric(**wm)
            else:
              wms[k] = wm
          buffers.append(
              abstract_engine.MetricsBuffer(
                  id=b.get("id", 0),
                  mode=b.get("mode", "train"),
                  weighted_metrics=wms,
                  scalar_metrics=b.get("scalar_metrics", {}),
                  aggregation_fns=b.get("aggregation_fns", {}),
              )
          )
        else:
          buffers.append(b)
      # pylint: disable-next=protected-access
      self._metrics_recorder._metrics_buffer = buffers

    restored_additional_metadata = None
    if restored_metadata:
      self._micro_step_count = restored_metadata.get("micro_step_count", 0)
      restored_additional_metadata = restored_metadata.get("additional_metadata", None)

    # Restore intra-step state if it exists.
    if restored_checkpoint_state.accumulated_grads:
      self._accumulated_grads = restored_checkpoint_state.accumulated_grads

      if self._micro_step_count > 0 and self._metrics_recorder._metrics_buffer:  # pylint: disable=protected-access
        active_buf = self._metrics_recorder.get_step_metrics(restored_step)
        if active_buf and "loss" in active_buf.weighted_metrics:
          wm = active_buf.weighted_metrics["loss"]
          if wm.unreduced_sum.ndim > 0:
            self._cached_losses = [
                abstract_engine.WeightedMetric(
                    unreduced_sum=wm.unreduced_sum[i],
                    denominator=wm.denominator[i],
                    eps=wm.eps,
                    min_denom=wm.min_denom,
                )
                for i in range(wm.unreduced_sum.shape[0])
            ]
          else:
            self._cached_losses = [wm]

    return restored_additional_metadata

  def record_metrics(
      self,
      name: str,
      metric: abstract_engine.WeightedMetric | jax.Array | float | int | dict[str, Any],
      aggregation_fn: Callable[[jax.Array], Any] | None = None,
  ) -> None:
    """Records a metric into the buffer, appending to JAX arrays.

    Args:
      name: The name of the metric.
      metric: The metric to record.
      aggregation_fn: The aggregation function to apply to the metric.
    """
    if metric is None:
      return
    if isinstance(metric, dict):
      for sub_k, sub_v in metric.items():
        if sub_v is not None:
          self.record_metrics(
              f"{name}/{sub_k}" if name else sub_k,
              sub_v,
              aggregation_fn=aggregation_fn,
          )
    else:
      self._metrics_recorder.buffer_metrics(
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
    return self._metrics_recorder.get_metrics(clear_cache=clear_cache)

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
    self._throttler.cleanup()
    self._metrics_recorder.cleanup()
    self._checkpoint_manager.close()
