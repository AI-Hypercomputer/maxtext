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

"""Distillation Trainer for MaxText + Tunix.

This script implements the "Post-Pruning Recovery" distillation process: recovering model quality
via soft distillation from a Teacher model. It leverages the Tunix Distillation library
for the training loop and loss calculation, while using MaxText for efficient
TPU model execution and data loading.

Architecture Overview:
----------------------
1. **Dual Model Loading**: Uniquely, this script initializes two distinct MaxText models:
   - Student: The model being trained (can be pruned/smaller).
   - Teacher: The frozen reference model (usually larger or same size).

2. **Configuration Isolation**: To support different architectures (e.g., a pruned Student
   vs. a full Teacher), we use `pyconfig` to generate two separate configuration objects
   derived from the same base YAML but applied with different overrides.

3. **Tunix Integration**: We wrap the MaxText models in `TunixMaxTextAdapter` to expose
   a standard interface (call signature) that the Tunix `DistillationTrainer` expects.
"""

import os
from typing import Any, Iterator, Sequence, Dict

from absl import app
import flax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint

# MaxText Imports
from MaxText import max_logging
from MaxText import maxtext_utils
from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText import train_utils
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from MaxText.rl import utils_rl

# Tunix Imports
from tunix.distillation import distillation_trainer
from tunix.distillation.strategies import logit
from tunix.sft import metrics_logger
from tunix.sft import profiler


# -----------------------------------------------------------------------------
# Custom Data Structures & Trainer
# -----------------------------------------------------------------------------


@flax.struct.dataclass(frozen=True)
class MaxTextTrainingInput(distillation_trainer.TrainingInput):
  """Extended TrainingInput dataclass to carry MaxText-specific fields.

  Attributes:
    positions: Position indices for the tokens (for RoPE).
    decoder_segment_ids: Segment IDs for packed sequences (0=padding, 1+=examples).
    targets: Ground truth target tokens (used for loss calculation and logging).
  """

  positions: Any = None
  decoder_segment_ids: Any = None
  targets: Any = None


@jax.jit
def _compute_debug_metrics(
    student_logits: jax.Array, teacher_logits: jax.Array, targets: jax.Array, temperature: float, alpha: float
) -> Dict[str, jax.Array]:
  """JIT-compiled metric calculation to reduce dispatch overhead during debug.

  Calculates validation metrics (Hard Loss vs Truth, Soft Loss vs Teacher) efficiently
  on device to avoid stalling the training pipeline with Python-level math.

  Args:
    student_logits: Logits output by the student model [Batch, SeqLen, Vocab].
    teacher_logits: Logits output by the teacher model [Batch, SeqLen, Vocab].
    targets: Ground truth target token indices [Batch, SeqLen].
    temperature: The temperature used for soft target scaling.
    alpha: The weighting factor for the soft loss component (0.0 to 1.0).

  Returns:
    A dictionary containing scalar arrays for 'hard_loss', 'soft_loss', and 'total_proxy'.
  """
  # Masks
  pad_id = 0
  vocab_size = teacher_logits.shape[-1]
  mask = (targets != pad_id).astype(jnp.float32)
  num_valid = mask.sum() + 1e-6
  one_hot_targets = jax.nn.one_hot(targets, vocab_size)

  # Cast to float32 for stability
  s_logits = student_logits.astype(jnp.float32)
  t_logits = teacher_logits.astype(jnp.float32)

  # Hard Loss (Student vs Truth)
  hard_loss_per_token = optax.softmax_cross_entropy(logits=s_logits, labels=one_hot_targets)
  hard_loss = (hard_loss_per_token * mask).sum() / num_valid

  # Soft Loss (Student vs Teacher)
  # Scale logits by temperature T
  soft_targets = jax.nn.softmax(t_logits / temperature)
  soft_loss_per_token = optax.softmax_cross_entropy(logits=s_logits / temperature, labels=soft_targets)
  soft_loss = (soft_loss_per_token * mask).sum() / num_valid
  # Rescale gradient magnitude (Hinton et al.)
  soft_loss_scaled = soft_loss * (temperature**2)

  # Total Proxy
  total_proxy = (alpha * soft_loss_scaled) + ((1 - alpha) * hard_loss)

  return {"hard_loss": hard_loss, "soft_loss": soft_loss_scaled, "total_proxy": total_proxy}


def _log_config_details(config: pyconfig.HyperParameters, label: str) -> None:
  """Logs detailed architecture configuration for verification.

  Args:
    config: The HyperParameters object to inspect.
    label: A string label (e.g., 'Student', 'Teacher') for the log output.
  """
  kv_heads = getattr(config, "num_kv_heads", config.num_query_heads)
  max_logging.log(f"--- {label} Configuration ---")
  max_logging.log(f"  Model Name:      {config.model_name}")
  max_logging.log(
      f"  Dimensions:      {config.num_decoder_layers} Layers, " f"{config.emb_dim} Emb Dim, {config.head_dim} Head Dim"
  )
  max_logging.log(f"  Attention Heads: {config.num_query_heads} Query, {kv_heads} KV")
  max_logging.log(f"  Vocab Size:      {config.vocab_size}")
  max_logging.log(f"  Checkpoint:      {config.load_parameters_path}")


class MaxTextDistillationTrainer(distillation_trainer.DistillationTrainer):
  """Custom Trainer to preserve MaxText fields and log Teacher metrics.

  This class overrides `_prepare_inputs` to:
  1. Calculate Teacher Loss/Perplexity periodically for monitoring.
  2. Ensure MaxText-specific fields (positions, segment_ids) are passed to the model.
  """

  def __init__(self, log_period: int, *args, debug_mode: bool = False, **kwargs):
    """Initializes the trainer.

    Args:
      log_period: How often (in steps) to compute and log Teacher metrics.
      debug_mode: If True, runs expensive teacher loss calculation every log_period.
      *args: Positional arguments for the base Tunix Trainer.
      **kwargs: Keyword arguments for the base Tunix Trainer.
    """
    super().__init__(*args, **kwargs)
    self._log_step_counter = 0
    self.log_period = log_period
    self.debug_mode = debug_mode

  def _prepare_inputs(self, input_data: MaxTextTrainingInput) -> MaxTextTrainingInput:
    """Prepares inputs for the student model and runs the teacher model.

    This function generates the "Soft Targets" (logits) from the Teacher model
    that the Student will learn to mimic.

    Args:
      input_data: The batch of data from the iterator.

    Returns:
      A new MaxTextTrainingInput containing the Teacher's outputs (logits).
    """
    # 1. Generate inputs dictionary for the Teacher model
    inputs = self.gen_model_input_fn(input_data)["inputs"]

    if self._mode == metrics_logger.Mode.EVAL:
      teacher_output = None
    else:
      # 2. Run Teacher to get soft targets (logits)
      teacher_output = self.strategy.get_teacher_outputs(self.teacher_model, inputs)

      # 3. Monitor Performance (Debug Mode)
      self._log_step_counter += 1
      if self.debug_mode and self._log_step_counter % self.log_period == 0:
        try:
          # Run Student inference (filtering out targets/extra keys)
          # We manually filter to ensure safety against strict call signatures
          student_inputs = {
              k: v for k, v in inputs.items() if k in ["input_tokens", "positions", "attention_mask", "cache"]
          }

          # Forward pass (inference only)
          student_output, _ = self.model(**student_inputs)

          # Calculate metrics
          metrics = _compute_debug_metrics(
              student_logits=student_output,
              teacher_logits=teacher_output,
              targets=input_data.targets,
              temperature=self.strategy.temperature,
              alpha=self.strategy.alpha,
          )

          # Force synchronization to print (expensive!)
          max_logging.log(f"--- Step {self._log_step_counter} Debug Metrics ---")
          max_logging.log(f"Hard Loss (vs Truth):   {metrics['hard_loss']:.4f}")
          max_logging.log(f"Soft Loss (vs Teacher): {metrics['soft_loss']:.4f}")
          max_logging.log(f"Total Loss (Proxy):     {metrics['total_proxy']:.4f}")

        except Exception as e:  # pylint: disable=broad-exception-caught
          max_logging.log(f"Warning: Failed to compute debug metrics: {e}")

    # 4. Return extended object so fields are available for Student training step
    # pylint: disable=unexpected-keyword-arg
    return MaxTextTrainingInput(
        input_tokens=input_data.input_tokens,
        input_mask=input_data.input_mask,
        teacher_output=teacher_output,
        positions=input_data.positions,
        decoder_segment_ids=input_data.decoder_segment_ids,
        targets=input_data.targets,
    )


# -----------------------------------------------------------------------------
# Data Loading Adapter
# -----------------------------------------------------------------------------


class MaxTextToTunixIterator:
  """Adapts the raw dictionary output of MaxText's data loader to Tunix objects.

  MaxText's `train_utils.create_data_iterator` yields a dictionary.
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
      input_mask = jnp.ones_like(batch["inputs"], dtype=jnp.bool_)
      seg_ids = None

    # pylint: disable=unexpected-keyword-arg
    return MaxTextTrainingInput(
        input_tokens=batch["inputs"],
        input_mask=input_mask,
        teacher_output=None,
        positions=batch["inputs_position"],
        decoder_segment_ids=seg_ids,
        targets=batch["targets"],
    )


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def get_maxtext_model(config: pyconfig.HyperParameters, mesh: jax.sharding.Mesh) -> nnx.Module:
  """Loads a MaxText model and wraps it in a Tunix adapter.

  Args:
    config: The configuration object for this specific model (Student or Teacher).
    mesh: The global device mesh for sharding weights.

  Returns:
    A TunixMaxTextAdapter instance wrapping the loaded MaxText model.
  """
  max_logging.log(f"Initializing model: {config.model_name}...")
  model, _ = model_creation_utils.create_nnx_model(config, mesh=mesh)

  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model, use_no_op_mappings=True)
  return tunix_model


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------


def train_distill(student_config: pyconfig.HyperParameters, teacher_config: pyconfig.HyperParameters) -> None:
  """Main distillation training loop.

  Orchestrates the loading of both student and teacher models, configures the
  distillation strategy, and executes the training loop via the Tunix Trainer.

  Args:
    student_config: Configuration object for the Student model (learnable).
    teacher_config: Configuration object for the Teacher model (frozen).
  """
  # Validate vocab size match between Student and Teacher
  if student_config.vocab_size != teacher_config.vocab_size:
    raise ValueError(
        f"Vocab size mismatch! Student: {student_config.vocab_size}, Teacher: {teacher_config.vocab_size}. "
        "Distillation requires matching vocabularies."
    )

  # 1. Setup Mesh
  devices = jax.devices()
  devices_array = maxtext_utils.create_device_mesh(student_config, devices)
  mesh = jax.sharding.Mesh(devices_array, student_config.mesh_axes)

  # 2. Load Models
  max_logging.log(f"Loading Student from {student_config.load_parameters_path}...")
  _log_config_details(student_config, "Student")
  student_model = get_maxtext_model(student_config, mesh)

  max_logging.log(f"Loading Teacher from {teacher_config.load_parameters_path}...")
  _log_config_details(teacher_config, "Teacher")
  teacher_model = get_maxtext_model(teacher_config, mesh)

  # 3. Define Distillation Strategy
  def labels_fn(targets, **kwargs):
    """Converts integer targets to masked one-hot vectors for hard label loss."""
    del kwargs  # Unused
    one_hot = jax.nn.one_hot(targets, student_config.vocab_size)
    mask = (targets != 0).astype(one_hot.dtype)[..., None]
    return one_hot * mask

  def student_forward_fn(model, input_tokens, positions, attention_mask, cache, **kwargs):
    """Forward pass wrapper for the Student model."""
    del kwargs  # Unused
    # Tunix adapter ensures __call__ signature matches this
    outputs = model(input_tokens=input_tokens, positions=positions, cache=cache, attention_mask=attention_mask)
    return outputs[0]  # Return logits only

  # Teacher forward fn is identical for MaxText
  teacher_forward_fn = student_forward_fn

  strategy = logit.LogitStrategy(
      student_forward_fn=student_forward_fn,
      teacher_forward_fn=teacher_forward_fn,
      labels_fn=labels_fn,
      temperature=student_config.distill_temperature,
      alpha=student_config.distill_alpha,
  )

  # 4. Optimizer & Config
  optimizer = utils_rl.get_optimizer(student_config, student_config.steps)

  checkpointing_options = checkpoint.CheckpointManagerOptions(
      save_interval_steps=student_config.checkpoint_period, max_to_keep=student_config.max_num_checkpoints_to_keep
  )

  profiler_options = None
  if student_config.profiler == "xplane":
    profiler_options = profiler.ProfilerOptions(
        log_dir=student_config.tensorboard_dir,
        skip_first_n_steps=student_config.skip_first_n_steps_for_profiler,
        profiler_steps=student_config.profiler_steps,
        set_profile_options=False,
    )

  metrics_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=student_config.tensorboard_dir, flush_every_n_steps=student_config.log_period
  )

  train_config = distillation_trainer.TrainingConfig(
      max_steps=student_config.steps,
      eval_every_n_steps=student_config.eval_interval,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
      checkpoint_root_directory=student_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
  )

  # 5. Initialize Trainer
  trainer = MaxTextDistillationTrainer(
      log_period=student_config.log_period,
      debug_mode=False,  # Set to True to enable periodic full-graph metric calculation
      student_model=student_model,
      teacher_model=teacher_model,
      strategy=strategy,
      optimizer=optimizer,
      training_config=train_config,
  )
  trainer.is_managed_externally = True

  # 6. Configure Input Mapping
  # Maps the attributes of MaxTextTrainingInput to the kwargs expected by the models
  trainer = trainer.with_gen_model_input_fn(
      lambda batch: {
          "input_tokens": batch.input_tokens,
          "positions": batch.positions,
          "attention_mask": batch.input_mask,
          "targets": batch.targets,  # Passed to strategy (labels_fn), removed for model
          "cache": None,
      }
  )

  # 7. Data Iterators
  # We use MaxText's native create_data_iterator which creates both train and eval iterators
  # based on the config parameters (dataset_type, eval_interval, etc.)
  max_logging.log("Initializing Data Iterators via MaxText pipeline...")
  raw_train_iter, raw_eval_iter = train_utils.create_data_iterator(student_config, mesh)

  train_iter = MaxTextToTunixIterator(raw_train_iter)

  eval_iter = None
  if raw_eval_iter is not None:
    max_logging.log("Evaluation iterator successfully initialized.")
    eval_iter = MaxTextToTunixIterator(raw_eval_iter)
  elif student_config.eval_interval > 0:
    max_logging.log("Warning: eval_interval > 0 but create_data_iterator returned None for eval_iter.")

  # 8. Train
  max_logging.log("Starting Distillation Training...")
  with mesh, nn_partitioning.axis_rules(student_config.logical_axis_rules):
    # Pass both iterators to the trainer
    trainer.train(train_iter, eval_iter)

  # 9. Final Save (Conditional)
  if student_config.save_checkpoint_on_completion:
    should_save = student_config.steps % student_config.checkpoint_period

    if should_save:
      max_logging.log(f"Saving final checkpoint to {student_config.checkpoint_dir}...")
      try:
        saved = trainer.checkpoint_manager.save(
            trainer.train_steps, trainer.model, save_only_lora_params=getattr(trainer, "_lora_enabled", False), force=True
        )
        if saved:
          # Ensure underlying orbax manager finishes writing
          # pylint: disable=protected-access
          if trainer.checkpoint_manager._checkpoint_manager is not None:
            trainer.checkpoint_manager._checkpoint_manager.wait_until_finished()
          # pylint: enable=protected-access
          max_logging.log("Final checkpoint saved.")

      except Exception as e:  # pylint: disable=broad-exception-caught
        max_logging.log(f"Warning: Failed to save final checkpoint: {e}")

    else:
      max_logging.log("Waiting for automatic periodic checkpoint to finish...")
      trainer.checkpoint_manager.wait_until_finished()

  trainer.close()
  max_logging.log("Distillation Complete.")


def main(argv: Sequence[str]) -> None:
  """Entry point for the script.

  Parses configuration, isolates Student and Teacher overrides, and triggers the
  training loop.

  Args:
    argv: List of command-line arguments. Expects [script_name, config_file, ...].
  """
  # 1. Parse Global Config to extract Overrides
  global_config = pyconfig.initialize(argv)

  # 2. Initialize STUDENT Config
  # Order of precedence: YAML < CLI < kwargs (student_overrides).
  student_overrides = global_config.student_overrides
  student_config = pyconfig.initialize(argv, **student_overrides)

  # 3. Initialize TEACHER Config
  # We isolate the Teacher from Student CLI arguments (like pruning params).
  teacher_overrides = global_config.teacher_overrides

  # Ensure load_parameters_path is set (check overrides, then env var)
  if not teacher_overrides.get("load_parameters_path"):
    ckpt_path = os.environ.get("TEACHER_CHECKPOINT_PATH")
    if ckpt_path:
      teacher_overrides["load_parameters_path"] = ckpt_path
    else:
      max_logging.log("Warning: No load_parameters_path found for Teacher.")

  # Construct sanitized argv: [script_name, config_file]
  # This ensures flags like `num_query_heads=16` passed in CLI don't affect the Teacher.
  teacher_argv = [argv[0], argv[1]]
  teacher_config = pyconfig.initialize(teacher_argv, **teacher_overrides)

  # 4. Run Training
  train_distill(student_config, teacher_config)


if __name__ == "__main__":
  app.run(main)
