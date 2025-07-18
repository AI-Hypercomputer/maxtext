#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Training and data loading hooks for SFT"""

from flax import nnx
from jax.typing import ArrayLike  # pylint: disable=g-importing-member

import jax

from MaxText.metric_logger import MetricLogger
from MaxText import exceptions
from MaxText import maxtext_utils
from MaxText import max_logging
from MaxText.data_loader import DataLoader
from tunix.sft.hooks import DataHooks, TrainingContext, TrainingHooks
from typing_extensions import override

class SFTTrainingHooks(TrainingHooks):
  """Training hooks for SFT."""

  def __init__(self, config, mesh, learning_rate_schedule):
    self.config = config
    self.mesh = mesh
    self.learning_rate_schedule = learning_rate_schedule
    self.metric_logger = MetricLogger(self.config, self.learning_rate_schedule)

  @override
  def on_train_start(self, train_ctx: TrainingContext):
    """Called at the beginning of training."""
    state = nnx.state(train_ctx.model)
    params = state.filter(nnx.Param)

    # TODO: Surbhi
    # Set `using_pipeline_parallelism=True` until https://buganizer.corp.google.com/issues/431026929#comment5 is fixed
    if not self.config.using_pipeline_parallelism:
      maxtext_utils.assert_params_sufficiently_sharded(params, self.mesh, self.config.sharding_tolerance)

    self.metric_logger.write_setup_info_to_tensorboard(params)

  @override
  def on_train_end(self):
    """Called at the end of training."""
    if self.metric_logger:
      self.metric_logger.flush_metrics_and_cleanup()

  @override
  def on_train_step_end(self, step: int, train_loss: ArrayLike):
    """Called at the end of training step."""
    metrics_to_write = {
      "scalar": {}
    }
    metrics_to_write["scalar"]["train/loss"] = train_loss
    metrics_to_write["scalar"].update({"perf/per_device_tflops": self.metric_logger.metadata["per_device_tflops"]})
    metrics_to_write["scalar"].update({"perf/per_device_tokens": self.metric_logger.metadata["per_device_tokens"]})
    metrics_to_write["scalar"].update({"learning/current_learning_rate": self.learning_rate_schedule(step-1)})

    if self.config.enable_tensorboard:
      self.metric_logger.write_metrics_to_tensorboard(metrics_to_write, step, is_training=True)

    if self.config.metrics_file:
      self.metric_logger.write_metrics_locally(metrics_to_write, step)

    if self.config.gcs_metrics and jax.process_index() == 0:
      self.metric_logger.write_metrics_for_gcs(metrics_to_write, step, is_training=True)

  @override
  def on_eval_step_start(self, eval_step: int):
    if eval_step >= self.config.eval_steps:
      return -1

  @override
  def on_eval_step_end(self, step: int, eval_loss: ArrayLike):
    """Called at the end of evaluation step."""
    metrics_to_write = {
      "scalar": {}
    }
    metrics_to_write["scalar"]["eval/loss"] = eval_loss
    
    if self.config.metrics_file:
      self.metric_logger.write_metrics_locally(metrics_to_write, step)

    if self.config.gcs_metrics and jax.process_index() == 0:
      self.metric_logger.write_metrics_for_gcs(metrics_to_write, step, is_training=False)

    if metrics_to_write["scalar"]["eval/loss"] <= self.config.target_eval_loss:
      raise exceptions.StopTraining(f"Target loss {self.config.target_eval_loss=} is achieved.")

class SFTDataHooks(DataHooks):
  """Data hooks for SFT."""

  def __init__(self, config, mesh, train_data_iterator):
    self.train_data_loader = DataLoader(config, mesh, train_data_iterator, goodput_recorder=None)

  @override
  def load_next_train_batch(self, step: int):
    train_batch = None
    try:
      train_batch = self.train_data_loader.load_next_batch()
    except Exception as e:
      max_logging.log(f"Exception in load_next_train_batch: {str(e)}")
    return train_batch
