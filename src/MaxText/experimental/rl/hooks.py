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


"""Training and data loading hooks for GRPO/RL training"""

from collections import defaultdict
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from MaxText import exceptions
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText.data_loader import DataLoader
from MaxText.experimental.rl import grpo_input_pipeline
from MaxText.metric_logger import MetricLogger
from MaxText.utils import gcs_utils
from MaxText.utils.goodput_utils import GoodputEvent, record_goodput


class GRPOTrainingHooks:
  """Training hooks for GRPO/RL training.

  This class provides hooks for monitoring and logging during GRPO training,
  similar to SFTTrainingHooks but adapted for reinforcement learning.
  """

  def __init__(self, config, mesh, learning_rate_schedule, goodput_recorder):
    """Initialize GRPO training hooks.

    Args:
      config: Training configuration object.
      mesh: JAX mesh for distributed training.
      learning_rate_schedule: Learning rate schedule function.
      goodput_recorder: Goodput recorder for performance tracking.
    """
    self.config = config
    self.mesh = mesh
    self.metric_logger = MetricLogger(self.config, learning_rate_schedule)
    self.goodput_recorder = goodput_recorder
    self.metadata = {}
    self.train_metadata = defaultdict(dict)
    self.eval_metadata = defaultdict(float)
    self.generation_metadata = defaultdict(list)

  def on_train_start(self, state, step: int):
    """Called at the beginning of training.

    Args:
      state: Training state containing model parameters.
      step: Current training step.
    """
    params = state.params

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

    self.metadata["first_train_step"] = step
    max_logging.log(f"GRPO training started at step {step}")

  def on_train_end(self, step: int):
    """Called at the end of training.

    Args:
      step: Final training step.
    """
    assert (
        "first_train_step" in self.metadata
    ), "GRPOTrainingHooks.on_train_start() must be called before GRPOTrainingHooks.on_train_end()"

    if self.metric_logger:
      self.metric_logger.flush_metrics_and_cleanup()

    max_logging.log(f"GRPO training completed at step {step}")

  def on_train_step_start(self, step: int, data: Optional[Dict[str, Any]] = None):
    """Called at the beginning of a training step.

    Args:
      step: Current training step.
      data: Optional batch data for the step.
    """
    if self.config.enable_goodput_recording:
      record_goodput(self.goodput_recorder, f"record_{GoodputEvent.STEP.value}_start_time", step)

    # Store metadata for this step
    self.train_metadata[step] = {
        "start_time": jax.process_time(),
    }

    # Calculate batch statistics if data is provided
    if data is not None:
      if "targets_segmentation" in data:
        total_weights = jnp.sum(data["targets_segmentation"] != 0)
        self.train_metadata[step]["total_weights"] = total_weights
      if "completions" in data:
        completion_length = data["completions"].shape[-1]
        self.train_metadata[step]["completion_length"] = completion_length

  def on_train_step_end(self, step: int, metrics: Dict[str, Any], step_time: float):
    """Called at the end of training step.

    Args:
      step: Current training step.
      metrics: Dictionary of metrics from the training step.
      step_time: Time taken for the step in seconds.
    """
    assert step in self.train_metadata, (
        "GRPOTrainingHooks.on_train_step_start() must be called before" " GRPOTrainingHooks.on_train_step_end()"
    )

    if self.config.enable_goodput_recording:
      record_goodput(self.goodput_recorder, f"record_{GoodputEvent.STEP.value}_end_time", step)

    if self.metadata["first_train_step"] == step:
      max_utils.print_mem_stats("After params initialized")

    # Add stored metadata to metrics
    if "total_weights" in self.train_metadata[step]:
      if "scalar" not in metrics:
        metrics["scalar"] = {}
      metrics["scalar"]["learning/total_weights"] = self.train_metadata[step]["total_weights"]

    # Record and write metrics
    self.metric_logger.record_train_metrics(metrics, step, step_time)
    self.metric_logger.write_metrics(metrics, step)

    # Clean up metadata for this step
    del self.train_metadata[step]

  def on_generation_start(self, step: int):
    """Called at the beginning of completion generation.

    Args:
      step: Current training step.
    """
    if self.config.enable_goodput_recording:
      record_goodput(self.goodput_recorder, f"record_{GoodputEvent.DATA_LOADING.value}_start_time", step)

    max_logging.log(f"Starting completion generation for step {step}")

  def on_generation_end(self, step: int, num_completions: int, generation_time: float):
    """Called at the end of completion generation.

    Args:
      step: Current training step.
      num_completions: Number of completions generated.
      generation_time: Time taken for generation in seconds.
    """
    if self.config.enable_goodput_recording:
      record_goodput(self.goodput_recorder, f"record_{GoodputEvent.DATA_LOADING.value}_end_time", step)

    self.generation_metadata["num_completions"].append(num_completions)
    self.generation_metadata["generation_time"].append(generation_time)

    max_logging.log(f"Generated {num_completions} completions in {generation_time:.2f}s for step {step}")

  def on_eval_start(self, step: int):
    """Called at the beginning of evaluation.

    Args:
      step: Current training step.
    """
    self.eval_metadata["eval_step_count"] = 0
    self.eval_metadata["total_loss"] = 0.0
    self.eval_metadata["total_weights"] = 0.0
    max_logging.log(f"Starting evaluation at step {step}")

  def on_eval_step(self, eval_metrics: Dict[str, Any]):
    """Called for each evaluation step.

    Args:
      eval_metrics: Dictionary of evaluation metrics.
    """
    self.eval_metadata["eval_step_count"] += 1

    # Accumulate metrics
    if "scalar" in eval_metrics:
      if "eval/loss" in eval_metrics["scalar"]:
        self.eval_metadata["total_loss"] += eval_metrics["scalar"]["eval/loss"]
      if "learning/total_weights" in eval_metrics["scalar"]:
        self.eval_metadata["total_weights"] += eval_metrics["scalar"]["learning/total_weights"]

  def on_eval_end(self, step: int):
    """Called at the end of evaluation.

    Args:
      step: Current training step.
    """
    assert (
        self.eval_metadata["eval_step_count"] > 0
    ), "GRPOTrainingHooks.on_eval_step() must be called before GRPOTrainingHooks.on_eval_end()"

    eval_step_count = self.eval_metadata["eval_step_count"]
    avg_loss = self.eval_metadata["total_loss"] / eval_step_count

    metrics = {
        "scalar": {
            "eval/total_loss": self.eval_metadata["total_loss"],
            "eval/avg_loss": avg_loss,
            "eval/total_weights": self.eval_metadata["total_weights"],
            "eval/step_count": eval_step_count,
        }
    }

    self.metric_logger.write_metrics(metrics, step, is_training=False)
    self.eval_metadata.clear()

    max_logging.log(f"Evaluation completed at step {step}: avg_loss={avg_loss:.4f}")

    if avg_loss <= self.config.target_eval_loss:
      raise exceptions.StopTraining(f"Target loss {self.config.target_eval_loss=} is achieved.")

  def on_checkpoint_save(self, step: int, checkpoint_path: str):
    """Called when a checkpoint is saved.

    Args:
      step: Current training step.
      checkpoint_path: Path where checkpoint was saved.
    """
    max_logging.log(f"Checkpoint saved at step {step}: {checkpoint_path}")

  def log_reward_statistics(self, step: int, rewards: jnp.ndarray):
    """Log reward statistics for the current step.

    Args:
      step: Current training step.
      rewards: Array of rewards for the batch.
    """
    reward_stats = {
        "scalar": {
            "rewards/mean": jnp.mean(rewards),
            "rewards/std": jnp.std(rewards),
            "rewards/min": jnp.min(rewards),
            "rewards/max": jnp.max(rewards),
        }
    }
    self.metric_logger.write_metrics(reward_stats, step)

  def log_advantage_statistics(self, step: int, advantages: jnp.ndarray):
    """Log advantage statistics for the current step.

    Args:
      step: Current training step.
      advantages: Array of advantages for the batch.
    """
    advantage_stats = {
        "scalar": {
            "advantages/mean": jnp.mean(advantages),
            "advantages/std": jnp.std(advantages),
            "advantages/min": jnp.min(advantages),
            "advantages/max": jnp.max(advantages),
        }
    }
    self.metric_logger.write_metrics(advantage_stats, step)

  def get_generation_summary(self) -> Dict[str, float]:
    """Get summary statistics for completion generation.

    Returns:
      Dictionary containing generation statistics.
    """
    if not self.generation_metadata["num_completions"]:
      return {}

    return {
        "total_completions": sum(self.generation_metadata["num_completions"]),
        "avg_completions_per_step": jnp.mean(jnp.array(self.generation_metadata["num_completions"])),
        "avg_generation_time": jnp.mean(jnp.array(self.generation_metadata["generation_time"])),
        "total_generation_time": sum(self.generation_metadata["generation_time"]),
    }


class GRPODataHooks:
  """Data hooks for GRPO/RL training.

  This class provides hooks for data loading during GRPO training,
  handling both prompt loading and completion generation using GRPO's
  multi-host data pipeline infrastructure.
  """

  def __init__(self, config, mesh, goodput_recorder=None):
    """Initialize GRPO data hooks with multi-host data pipeline.

    Args:
      config: Training configuration object.
      mesh: JAX mesh for distributed training.
      goodput_recorder: Optional goodput recorder for performance tracking.
    """
    self.config = config
    self.mesh = mesh
    self.goodput_recorder = goodput_recorder
    self.train_batch = None
    self.eval_batch = None

    # Create multi-host data iterator using GRPO's input pipeline
    self.train_data_iterator = grpo_input_pipeline.create_data_iterator(config, mesh)
    self.eval_data_iterator = None  # GRPO doesn't support eval yet

    # Wrap train iterator with DataLoader for goodput tracking
    self.train_data_loader = DataLoader(config, mesh, self.train_data_iterator, goodput_recorder=goodput_recorder)

    max_logging.log("GRPO data hooks initialized with multi-host data pipeline")

  def load_next_train_batch(self, step: int) -> Optional[Dict[str, Any]]:
    """Loads the next batch of training data using multi-host data pipeline.

    This method uses MaxText's DataLoader which handles:
    - Multi-host data loading and sharding
    - Goodput tracking for data loading time
    - Proper error handling and logging

    Args:
      step: Current training step.

    Returns:
      Dictionary containing the batch data, or None if loading failed.
    """
    try:
      self.train_batch = self.train_data_loader.load_next_batch()
      return self.train_batch
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Exception in load_next_train_batch at step {step}: {str(e)}")
      self.train_batch = None
      return None

  def load_next_eval_batch(self, step: int, eval_step_count: int) -> Optional[Dict[str, Any]]:
    """Loads the next batch of evaluation data.

    Note: GRPO input pipeline does not currently support evaluation data.

    Args:
      step: Current training step.
      eval_step_count: Current evaluation step count.

    Returns:
      None, as GRPO doesn't support eval data yet.
    """
    max_logging.log(f"GRPO eval data not supported yet at step {step}")
