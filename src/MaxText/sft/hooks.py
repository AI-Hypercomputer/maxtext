# Copyright 2023–2025 Google LLC
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


"""Training and data loading hooks for SFT"""

from flax import nnx
from collections import defaultdict
import jax
import jax.numpy as jnp

from maxtext.src.maxtext.metric_logger import MetricLogger
from maxtext.src.maxtext.utils import gcs_utils
from maxtext.src.maxtext import exceptions
from maxtext.src.maxtext import maxtext_utils
from maxtext.src.maxtext import max_logging
from maxtext.src.maxtext import max_utils
from maxtext.src.maxtext.data_loader import DataLoader
from maxtext.src.maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from tunix.sft import peft_trainer
from tunix.sft.hooks import DataHooks, TrainingHooks
from typing_extensions import override
from maxtext.src.maxtext.utils.goodput_utils import (
    GoodputEvent,
    record_goodput,
)


class SFTTrainingHooks(TrainingHooks):
  """Training hooks for SFT."""

  def __init__(self, config, mesh, learning_rate_schedule, goodput_recorder):
    self.config = config
    self.mesh = mesh
    self.metric_logger = MetricLogger(self.config, learning_rate_schedule)
    self.goodput_recorder = goodput_recorder
    self.metadata = {}
    self.train_metadata = defaultdict(float)
    self.eval_metadata = defaultdict(float)

  @override
  def on_train_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of training."""
    state = nnx.state(train_ctx.model)
    params = state.filter(nnx.Param)

    if not self.config.using_pipeline_parallelism:
      maxtext_utils.assert_params_sufficiently_sharded(params, self.mesh, self.config.sharding_tolerance)

    self.metric_logger.write_setup_info_to_tensorboard(params)

    if self.config.dump_hlo:
      jax.block_until_ready(state)  # Ensure compilation has finished
      gcs_utils.upload_dump(
          self.config.dump_hlo_local_dir,
          self.config.dump_hlo_gcs_dir,
          module_name=self.config.dump_hlo_module_name,
          delete_local_after=self.config.dump_hlo_delete_local_after,
          all_host_upload=self.config.dump_hlo_upload_all,
      )

    self.metadata["first_train_step"] = train_ctx.train_steps

  @override
  def on_train_end(self, train_ctx: peft_trainer.PeftTrainer):  # pylint: disable=unused-argument
    """Called at the end of training."""
    assert (
        "first_train_step" in self.metadata
    ), "SFTTrainingHooks.on_train_start() must be called before SFTTrainingHooks.on_train_end()"

    if self.metric_logger:
      self.metric_logger.flush_metrics_and_cleanup()

  @override
  def on_train_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of a training step."""
    if self.config.enable_goodput_recording:
      record_goodput(self.goodput_recorder, f"record_{GoodputEvent.STEP.value}_start_time", train_ctx.train_steps)

    # Calculate the number of non-padded tokens in the batch
    total_weights = jnp.sum(train_ctx.data_hooks.train_batch["targets_segmentation"] != 0)

    self.train_metadata[train_ctx.train_steps] = {
      "total_weights": total_weights,
    }

  @override
  def on_train_step_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      train_step: int,
      train_loss: float,
      step_time: float,
  ):
    """Called at the end of training step."""
    assert train_step in self.train_metadata, (
        "SFTTrainingHooks.on_train_step_start() must be called before"
        " SFTTrainingHooks.on_train_step_end()"
    )

    if self.metadata["first_train_step"] == train_step:
      max_utils.print_mem_stats("After params initialized")

    metrics = {
        "scalar": {
            "learning/loss": train_loss,
            "learning/total_weights": self.train_metadata[train_step][
                "total_weights"
            ],
        }
    }
    self.metric_logger.record_train_metrics(metrics, train_step, step_time)
    self.metric_logger.write_metrics(metrics, train_step)
    del self.train_metadata[train_step]

  @override
  def on_eval_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of an evaluation step."""
    self.eval_metadata["eval_step_count"] += 1.0
    # Calculate the number of non-padded tokens in the batch
    self.eval_metadata["total_weights"] += jnp.sum(train_ctx.data_hooks.eval_batch["targets_segmentation"] != 0)

  @override
  def on_eval_step_end(self, train_ctx: peft_trainer.PeftTrainer, eval_loss: float):
    """Called at the end of evaluation step."""
    assert (
        self.eval_metadata["eval_step_count"] != 0
    ), "SFTTrainingHooks.on_eval_step_start() must be called before SFTTrainingHooks.on_eval_step_end()"

    avg_loss = eval_loss / self.eval_metadata["eval_step_count"]
    metrics = {
        "scalar": {
            "eval/total_loss": eval_loss,
            "eval/avg_loss": avg_loss,
            "eval/total_weights": self.eval_metadata["total_weights"],
        }
    }
    self.metric_logger.write_metrics(metrics, train_ctx.train_steps, is_training=False)
    self.eval_metadata.clear()

    if avg_loss <= self.config.target_eval_loss:
      raise exceptions.StopTraining(f"Target loss {self.config.target_eval_loss=} is achieved.")


class SFTDataHooks(DataHooks):
  """Data hooks for SFT."""

  def __init__(self, config, mesh, goodput_recorder):
    self.config = config
    self.train_data_iterator, self.eval_data_iterator = create_data_iterator(config, mesh)
    self.train_data_loader = DataLoader(config, mesh, self.train_data_iterator, goodput_recorder=goodput_recorder)
    self.train_batch = None
    self.eval_batch = None

  @override
  def load_next_train_batch(self, train_ctx: peft_trainer.PeftTrainer):  # pylint: disable=unused-argument
    """Loads the next batch of data for training."""
    try:
      self.train_batch = self.train_data_loader.load_next_batch()
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Exception in load_next_train_batch: {str(e)}")
      self.train_batch = None
    return self.train_batch

  @override
  def load_next_eval_batch(self, train_ctx: peft_trainer.PeftTrainer):
    """Loads the next batch of data for evaluation."""
    try:
      # Run evaluation only for `config.eval_steps` steps.
      if self.config.eval_steps > 0 and train_ctx.training_hooks.eval_metadata["eval_step_count"] >= self.config.eval_steps:
        self.eval_batch = None
      else:
        self.eval_batch = next(self.eval_data_iterator)
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Exception in load_next_eval_batch: {str(e)}")
      self.eval_batch = None
    return self.eval_batch
