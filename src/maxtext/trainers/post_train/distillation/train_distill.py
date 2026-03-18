# Copyright 2023-2026 Google LLC
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

from typing import Sequence, Callable
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint

# MaxText Imports
from maxtext.configs import pyconfig
from maxtext.input_pipeline import tokenizer
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.optimizers import optimizers
from maxtext.trainers.post_train.distillation import distillation_utils
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils

# Tunix Imports
from tunix.sft import peft_trainer
from tunix.sft import metrics_logger
from tunix.sft import profiler


# -----------------------------------------------------------------------------
# Distillation Optimizer with cosine decay and warmup
# -----------------------------------------------------------------------------


def get_distillation_optimizer(config, max_train_steps):
  """Creates a custom optimizer for distillation that enables Learning Rate logging.

  This function constructs an optax optimizer using standard MaxText settings but
  wraps it with `optax.inject_hyperparams`. This wrapper is strictly required
  by the Tunix `PeftTrainer` to log the learning rate to TensorBoard; without it,
  the trainer cannot find the LR in the optimizer state.

  Args:
    config: The HyperParameters object containing optimizer settings (e.g.,
        `learning_rate`, `adam_b1`, `opt_type`, `gradient_clipping_threshold`).
    max_train_steps: The total number of training steps, used to calculate
        the warmup and cosine decay schedule.

  Returns:
    An optax optimizer that:
    1. Uses the optimizer type specified in `config.opt_type` (AdamW, SGD, etc.).
    2. Follows the MaxText cosine decay schedule.
    3. Applies gradient clipping if configured.
    4. Exposes the learning rate as a hyperparameter in the state for logging.
  """
  # Check for unsupported Muon optimizer
  if config.opt_type == "muon":
    raise ValueError("Muon optimizer is not currently supported in distillation mode.")

  # 1. Define Schedule
  schedule = optax.schedules.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.learning_rate,
      warmup_steps=int(config.warmup_steps_fraction * max_train_steps),
      decay_steps=max_train_steps,
      end_value=config.learning_rate_final_fraction * config.learning_rate,
  )

  # 2. Define Factory (Required for inject_hyperparams)
  def optimizer_factory(learning_rate):
    # Reuse MaxText's standard logic to create the base optimizer.
    # We pass 'learning_rate' (which is the injected schedule) directly.
    opt = optimizers.get_optimizer(config, learning_rate, model=None)

    # Apply Gradient Clipping
    if config.gradient_clipping_threshold > 0:
      opt = optax.chain(
          optax.clip_by_global_norm(max_norm=config.gradient_clipping_threshold),
          opt,
      )
    return opt

  # 3. Create Injectable Optimizer
  # This wraps the factory so 'learning_rate' sits at the top level of the state
  optimizer = optax.inject_hyperparams(optimizer_factory)(learning_rate=schedule)

  return optimizer


def create_forward_fn(config: pyconfig.HyperParameters) -> Callable[..., distillation_utils.DistillationForwardOutput]:
  """Creates a forward function closure that binds the specific model configuration.

  Args:
    config: The HyperParameters object for the specific model being wrapped.

  Returns:
    A callable `model_forward_fn` that matches the signature expected by the
    Tunix `LogitStrategy` and handles the MaxText-specific forward call.
  """

  def model_forward_fn(
      model, input_tokens, positions, attention_mask, decoder_segment_ids=None, cache=None, **kwargs
  ) -> distillation_utils.DistillationForwardOutput:
    """Forward pass wrapper adapted for raw MaxText models."""
    del attention_mask  # Unused
    del cache  # Unused
    logits = model(
        decoder_input_tokens=input_tokens,
        decoder_positions=positions,
        decoder_segment_ids=decoder_segment_ids,
        enable_dropout=config.enable_dropout,
        decoder_target_tokens=kwargs.get("decoder_target_tokens", None),
        decoder_target_mask=kwargs.get("decoder_target_mask", None),
    )
    out_projection_activations = None
    if config.distill_beta > 0.0:
      out_projection_activations = maxtext_utils.get_intermediate_value(model, "out_projection_activations", clear=True)

    retval = distillation_utils.DistillationForwardOutput(
        logits=logits, out_projection_activations=out_projection_activations
    )
    return retval

  return model_forward_fn


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


class ModelBundle(nnx.Module):
  """Wrapper for teacher and student modules."""

  def __init__(self, teacher_model: nnx.Module, student_model: nnx.Module):
    self.teacher_model = teacher_model
    self.student_model = student_model

  def __call__(self, *args, **kwargs):
    raise NotImplementedError("Use `call_student` or `call_teacher` explicitly.")

  def call_student(self, *args, **kwargs):
    return self.student_model(*args, **kwargs)

  def call_teacher(self, *args, **kwargs):
    return jax.lax.stop_gradient(self.teacher_model(*args, **kwargs))


class MaxTextDistillationTrainer(peft_trainer.PeftTrainer):
  """Custom Trainer to preserve MaxText fields and log Teacher metrics.

  This class overrides `_prepare_inputs` to ensure MaxText-specific fields
  (positions, segment_ids) are passed to the model.
  """

  def __init__(self, model, strategy, optimizer, training_config, **kwargs):
    super().__init__(model=model, optimizer=optimizer, training_config=training_config, **kwargs)

    self.strategy = strategy

    # override optimizer to only use student_model.
    if training_config.gradient_accumulation_steps is not None and training_config.gradient_accumulation_steps > 1:
      optimizer = optax.MultiSteps(optimizer, training_config.gradient_accumulation_steps)
    wrt = nnx.LoRAParam if self._lora_enabled else nnx.Param
    self.optimizer = nnx.Optimizer(model.student_model, optimizer, wrt=wrt)

  def _train_step(self, model, optimizer, inputs):
    """Overrides the main JIT block to natively handle ModelBundle module."""

    batch = self.gen_model_input_fn(inputs)

    def loss_wrapper(student, teacher, batch):
      if "teacher_output" in batch:
        teacher_output = batch["teacher_output"]
      else:
        teacher_output = self.strategy.teacher_forward_fn(
            model=teacher,
            input_tokens=batch["input_tokens"],
            positions=batch["positions"],
            attention_mask=batch.get("attention_mask"),
            decoder_segment_ids=batch.get("decoder_segment_ids"),
            decoder_target_tokens=batch.get("targets", None),
            decoder_target_mask=batch.get("targets_segmentation", None),
            cache=None,
        )

      teacher_output = jax.tree.map(jax.lax.stop_gradient, teacher_output)

      student_output = self.strategy.student_forward_fn(
          model=student,
          input_tokens=batch["input_tokens"],
          positions=batch["positions"],
          attention_mask=batch.get("attention_mask"),
          decoder_segment_ids=batch.get("decoder_segment_ids"),
          decoder_target_tokens=batch.get("targets", None),
          decoder_target_mask=batch.get("targets_segmentation", None),
          cache=None,
      )
      # we should apply a mask for labels to disable segment-separator tokens
      labels = self.strategy.labels_fn(batch["targets"], targets_segmentation=batch.get("targets_segmentation", None))
      return self.strategy.compute_loss(student_output, teacher_output, labels)

    # Because student is the 0th argument, argnums=0 guarantees
    # we only compute gradients for the student.
    grad_fn = nnx.value_and_grad(
        loss_wrapper,
        argnums=0,
        has_aux=True,
    )

    out, grads = grad_fn(model.student_model, model.teacher_model, batch)

    optimizer.update(model.student_model, grads)

    return out[0], out[1]  # loss, aux

  def _eval_step(self, model, inputs):
    """Evaluation only needs the student."""
    inputs = self.gen_model_input_fn(inputs)

    student_output = self.strategy.student_forward_fn(
        model=model.student_model,
        input_tokens=inputs["input_tokens"],
        positions=inputs["positions"],
        attention_mask=inputs.get("attention_mask"),
        decoder_segment_ids=inputs.get("decoder_segment_ids"),
        cache=None,
    )
    labels = self.strategy.labels_fn(inputs["targets"])
    return self.strategy.compute_eval_loss(student_output, labels)

  def _prepare_inputs(
      self, input_data: distillation_utils.MaxTextTrainingInput
  ) -> distillation_utils.MaxTextTrainingInput:
    """Prepares inputs for the student model and runs the teacher model.

    This function generates the "Soft Targets" (logits) from the Teacher model
    that the Student will learn to mimic.

    Args:
      input_data: The batch of data from the iterator.

    Returns:
      A new MaxTextTrainingInput containing the Teacher's outputs (logits).
    """

    # 3. Return extended object so fields are available for Student training step
    # pylint: disable=unexpected-keyword-arg
    return distillation_utils.MaxTextTrainingInput(
        input_tokens=input_data.input_tokens,
        input_mask=input_data.input_mask,
        positions=input_data.positions,
        decoder_segment_ids=input_data.decoder_segment_ids,
        targets=input_data.targets,
        targets_position=input_data.targets_position,
        targets_segmentation=input_data.targets_segmentation,
    )

  def _post_process_train_step(self, aux: dict[str, jax.Array]) -> None:
    """Extracts auxiliary metrics from the strategy and buffers them for logging."""
    if self._buffered_train_metrics is None:
      return

    # 'aux' contains the dictionary we returned from compute_loss:
    # {"distill/soft_loss": ..., "distill/hard_loss": ...}
    for name, value in aux.items():
      # We accumulate these values. PeftTrainer handles the averaging.
      # The structure expected is: dict[metric_name, (list_of_values, aggregation_fn)]
      if name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[name] = ([], np.mean)

      self._buffered_train_metrics.additional_metrics[name][0].append(value)


def _setup_and_restore_input_pipeline(trainer, raw_train_iter, config, train_config):
  """Configures the trainer to save/restore Grain iterator state.

  This function unconditionally replaces the default CheckpointManager with
  MaxTextCheckpointManager. This ensures consistent API availability (like
  wait_until_finished) and enables Grain checkpointing if the iterator supports it.

  Args:
    trainer: The active DistillationTrainer instance.
    raw_train_iter: The input pipeline iterator.
    config: The MaxText HyperParameters.
    train_config: The Tunix TrainingConfig.

  Returns:
    The iterator to use for training (restored or original).
  """
  is_grain_dataset = config.dataset_type == "grain"
  has_save_method = hasattr(raw_train_iter, "save")
  enable_checkpointing = raw_train_iter is not None and (is_grain_dataset or has_save_method)

  iterator_to_manage = raw_train_iter if enable_checkpointing else None

  if enable_checkpointing:
    max_logging.log("Input Pipeline Checkpointing: ENABLED")
    max_logging.log(f"Details: dataset_type='{config.dataset_type}', has_save={has_save_method}")
  else:
    max_logging.log("Input Pipeline Checkpointing: DISABLED")
    if raw_train_iter is None:
      max_logging.log("Reason: train_iter is None")
    else:
      max_logging.log(
          f"Reason: Iterator '{type(raw_train_iter).__name__}' is not recognized as Grain "
          f"(dataset_type='{config.dataset_type}', has_save={has_save_method})"
      )

  # 1. Create the specialized manager (always)
  maxtext_manager = distillation_utils.MaxTextCheckpointManager(
      raw_iterator=iterator_to_manage,
      root_directory=train_config.checkpoint_root_directory,
      options=train_config.checkpointing_options,
  )

  # 2. Swap managers (ensure clean resource release)
  if trainer.checkpoint_manager:
    trainer.checkpoint_manager.close()
  trainer.checkpoint_manager = maxtext_manager

  # 3. Restore input state (if applicable)
  if enable_checkpointing:
    restored_iter = trainer.checkpoint_manager.restore_iterator()
    if restored_iter is not None:
      max_logging.log("Restored input pipeline state to match model step.")
      return restored_iter

  return raw_train_iter


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def get_maxtext_model(config: pyconfig.HyperParameters, mesh: jax.sharding.Mesh) -> nnx.Module:
  """Loads a MaxText model.

  Args:
    config: The configuration object for this specific model (Student or Teacher).
    mesh: The global device mesh for sharding weights.

  Returns:
    The loaded MaxText model.
  """
  max_logging.log(f"Initializing model: {config.model_name}...")
  model, _ = model_creation_utils.create_nnx_model(config, mesh=mesh)
  return model


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

  # 2. Load Models & Tokenizer Info
  tok = tokenizer.build_tokenizer(
      tokenizer_path=student_config.tokenizer_path,
      tokenizer_type=student_config.tokenizer_type,
      add_bos=student_config.add_bos,
      add_eos=student_config.add_eos,
      hf_access_token=student_config.hf_access_token,
  )
  pad_id = tok.pad_id if tok.pad_id is not None else 0

  max_logging.log(f"Loading Student from {student_config.load_parameters_path}...")
  _log_config_details(student_config, "Student")
  student_model = get_maxtext_model(student_config, mesh)

  max_logging.log(f"Loading Teacher from {teacher_config.load_parameters_path}...")
  _log_config_details(teacher_config, "Teacher")
  teacher_model = get_maxtext_model(teacher_config, mesh)

  # 3. Define Distillation Strategy
  def labels_fn(targets, targets_segmentation=None, **kwargs):
    """Converts integer targets to masked one-hot vectors for hard label loss."""
    del kwargs  # Unused
    one_hot = jax.nn.one_hot(targets, student_config.vocab_size)
    mask = jnp.not_equal(targets, pad_id).astype(one_hot.dtype)[..., None]
    if targets_segmentation is not None:
      mask = mask * (targets_segmentation != 0)[..., None]
    return one_hot * mask

  # Both Student and Teacher use the same forward logic via the adapter
  student_forward_fn = create_forward_fn(student_config)
  teacher_forward_fn = create_forward_fn(teacher_config)

  # Use Monitored strategy from Utils
  strategy = distillation_utils.CombinedDistillationStrategy(
      student_forward_fn=student_forward_fn,
      teacher_forward_fn=teacher_forward_fn,
      labels_fn=labels_fn,
      temperature=student_config.distill_temperature,
      alpha=student_config.distill_alpha,
      beta_feature=student_config.distill_beta,
      layer_indices=student_config.distill_layer_indices,
      sft_mode=student_config.use_sft,
  )

  # 4. Optimizer & Config
  optimizer = get_distillation_optimizer(student_config, student_config.steps)

  checkpointing_options = checkpoint.CheckpointManagerOptions(
      save_interval_steps=student_config.checkpoint_period,
      max_to_keep=student_config.max_num_checkpoints_to_keep,
      enable_async_checkpointing=student_config.async_checkpointing,
      create=True,
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

  train_config = peft_trainer.TrainingConfig(
      max_steps=student_config.steps,
      eval_every_n_steps=student_config.eval_interval,
      metrics_logging_options=metrics_logging_options,
      profiler_options=profiler_options,
      checkpoint_root_directory=student_config.checkpoint_dir,
      checkpointing_options=checkpointing_options,
      gradient_accumulation_steps=student_config.gradient_accumulation_steps,
  )

  # 5. Data Iterators (Init BEFORE Trainer)
  # We use MaxText's native create_data_iterator which creates both train and eval iterators
  max_logging.log("Initializing Data Iterators via MaxText pipeline...")
  raw_train_iter, raw_eval_iter = input_pipeline_interface.create_data_iterator(student_config, mesh)

  teacher_model.eval()
  student_model.train()

  model_bundle = ModelBundle(teacher_model, student_model)

  # 6. Initialize Trainer
  trainer = MaxTextDistillationTrainer(
      model=model_bundle,
      strategy=strategy,
      optimizer=optimizer,
      training_config=train_config,
  )
  trainer.is_managed_externally = True
  trainer._has_aux = True  # pylint: disable=protected-access

  # 7. Input Pipeline Checkpointing
  # Replace the default CheckpointManager with a Grain-aware one, which enables iterator checkpointing for grain datasets.
  raw_train_iter = _setup_and_restore_input_pipeline(trainer, raw_train_iter, student_config, train_config)

  # 8. Configure Input Mapping
  trainer = trainer.with_gen_model_input_fn(
      lambda batch: {
          "input_tokens": batch.input_tokens,
          "positions": batch.positions,
          "attention_mask": batch.input_mask,
          "decoder_segment_ids": batch.decoder_segment_ids,
          "targets": batch.targets,  # Passed to strategy (labels_fn)
          "targets_position": batch.targets_position,  # Passed to strategy (labels_fn)
          "targets_segmentation": batch.targets_segmentation,  # Passed to strategy (labels_fn)
          "cache": None,
      }
  )

  # 9. Create Iterator Wrappers (Use Utils)
  train_iter = distillation_utils.MaxTextToTunixIterator(raw_train_iter)

  eval_iter = None
  if raw_eval_iter is not None:
    max_logging.log("Evaluation iterator successfully initialized.")
    eval_iter = distillation_utils.MaxTextToTunixIterator(raw_eval_iter)
  elif student_config.eval_interval > 0:
    max_logging.log("Warning: eval_interval > 0 but create_data_iterator returned None for eval_iter.")

  # 10. Train
  max_logging.log("Starting Distillation Training...")
  with mesh, nn_partitioning.axis_rules(student_config.logical_axis_rules):
    # Pass both iterators to the trainer
    trainer.train(train_iter, eval_iter)

  # 11. Final Save (Conditional)
  if student_config.save_checkpoint_on_completion:
    should_save = student_config.steps % student_config.checkpoint_period

    if should_save:
      max_logging.log(f"Saving final checkpoint to {student_config.checkpoint_dir}...")
      try:
        saved = trainer.checkpoint_manager.save(
            trainer.train_steps,
            trainer.model.student_model,
            save_only_lora_params=getattr(trainer, "_lora_enabled", False),
            force=True,
        )
        if saved:
          # Ensure underlying orbax manager finishes writing
          trainer.checkpoint_manager.wait_until_finished()
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

  # Ensure load_parameters_path is set in overrides
  if not teacher_overrides.get("load_parameters_path"):
    raise ValueError(
        "Teacher model path is missing! You must provide 'teacher_overrides.load_parameters_path' "
        "in your config or arguments."
    )

  # Construct sanitized argv: [script_name, config_file]
  # This ensures flags like `num_query_heads=16` passed in CLI don't affect the Teacher.
  teacher_argv = [argv[0], argv[1]]
  teacher_config = pyconfig.initialize(teacher_argv, **teacher_overrides)

  # 4. Run Training
  train_distill(student_config, teacher_config)


if __name__ == "__main__":
  app.run(main)
