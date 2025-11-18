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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

from typing import Any, Sequence
import datetime
import functools
import os

from absl import app

import numpy as np

import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from MaxText import checkpointing
from MaxText import exceptions
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText import profiler
from MaxText import pyconfig
from MaxText import sharding
from MaxText.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from MaxText.common_types import ShardMode
from MaxText.data_loader import create_dataloader
from MaxText.globals import EPS
from MaxText.metric_logger import MetricLogger
from MaxText.utils import gcs_utils
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

from MaxText.gradient_accumulation import gradient_accumulation_loss_and_grad
from MaxText.vocabulary_tiling import vocab_tiling_linen_loss
from MaxText.dpo_utils import _merge_dpo_state, _split_dpo_state, dpo_loss_fn
from MaxText.train_utils import validate_train_config
from MaxText.metric_logger import record_activation_metrics
# pylint: disable=too-many-positional-arguments


def get_first_step(state):
  return int(state.step)


# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------


def _get_mutable_collections(config, is_train):
  """Returns the list of mutable collections for the model apply."""
  mutable_collections = ["intermediates"]
  if config.mtp_num_layers > 0 and is_train:
    mutable_collections.append("mtp_losses")
  if config.mtp_eval_target_module > 0 and not is_train:
    mutable_collections.append("mtp_acceptance")
  return mutable_collections


def _calculate_cross_entropy_loss(logits, data, config):
  """Calculates the standard cross-entropy loss, masking paddings."""
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  return jnp.sum(xent)


def _model_forward_pass(model, params, data, config, dropout_rngs, mutable_collections, is_train):
  """Runs the forward pass for either a Linen or NNX model."""
  if isinstance(model, nn.Module):
    # Flax Linen model
    logits, intermediate_outputs = model.apply(
        params,
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal else None,
        enable_dropout=config.enable_dropout if is_train else False,
        rngs=dropout_rngs,
        mutable=mutable_collections,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
  else:
    # Flax NNX model
    logits = model(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        encoder_image_masks=data["image_masks"] if config.use_multimodal else None,
        enable_dropout=config.enable_dropout if is_train else False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    # NNX models don't return intermediates dict by default
    intermediate_outputs = {}
  return logits, intermediate_outputs


def _calculate_primary_loss(logits, intermediate_outputs, data, config, model, params, is_train):
  """Calculates the primary loss, handling vocab tiling."""
  if config.num_vocab_tiling > 1:
    hidden_state_key = ("intermediates", "decoder", "hidden_states")
    hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
    # Assumes vocab_tiling_linen_loss is defined elsewhere
    total_loss = vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)
  else:
    total_loss = _calculate_cross_entropy_loss(logits, data, config)
  return total_loss


def _calculate_moe_loss(intermediate_outputs, config):
  """Calculates the MoE load balancing loss."""
  if config.num_experts <= 1:
    return 0.0

  nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
  total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
  return jnp.mean(jnp.array(total_moe_lb_loss))


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module or NNX GraphDef
    config: Config of parameters
    data: Batch of data to apply to the model. Assumed to be correctly sliced.
    dropout_rng: A key to use to generate rng for dropout
    params: Model params (for Linen)
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # Setup
  mutable_collections = _get_mutable_collections(config, is_train)
  rng1, aqt_rng = jax.random.split(dropout_rng)
  dropout_rngs = {"dropout": rng1, "params": aqt_rng}

  # Run model forward pass
  logits, intermediate_outputs = _model_forward_pass(
      model, params, data, config, dropout_rngs, mutable_collections, is_train
  )

  # Calculate primary loss
  total_loss = _calculate_primary_loss(logits, intermediate_outputs, data, config, model, params, is_train)

  total_weights = jnp.sum(data["targets_segmentation"] != 0)

  if config.gradient_accumulation_steps > 1:
    loss = total_loss
  else:
    loss = total_loss / (total_weights + EPS)

  # Calculate auxiliary losses
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    # Assumes calculate_mtp_loss is defined elsewhere
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)

  moe_lb_loss = _calculate_moe_loss(intermediate_outputs, config)

  # Combine loss
  loss += mtp_loss + moe_lb_loss

  # Assemble auxiliary outputs
  intermediate_outputs["logits"] = logits
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,  # Primary loss
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "mtp_loss": mtp_loss,
  }
  return loss, aux


# -----------------------------------------------------------------------------
# Train Step Function
# -----------------------------------------------------------------------------


def _prepare_loss_functions_and_args(state, state_mesh_shardings, config):
  """Sets up DPO arguments and selects the appropriate loss function."""
  reference_params, reference_params_sharding, extra_dpo_args = [], [], []
  selected_loss_fn = loss_fn

  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    extra_dpo_args = [reference_params]
    selected_loss_fn = dpo_loss_fn

  return (
      state,
      state_mesh_shardings,
      reference_params,
      reference_params_sharding,
      extra_dpo_args,
      selected_loss_fn,
  )


def _compute_gradients_and_loss(
    model,
    params,
    params_shardings,
    data,
    dropout_rng,
    config,
    selected_loss_fn,
    extra_dpo_args,
    reference_params,
    reference_params_sharding,
):
  """Computes gradients and loss, handling accumulation and offloading logic."""
  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        selected_loss_fn,
        config,
        model,
        params,
        params_shardings,
        data,
        dropout_rng,
        extra_dpo_args,
    )
  else:
    # Handle Optimizer Memory Offloading for DPO
    if config.optimizer_memory_host_offload and config.use_dpo:
      reference_params = jax.device_put(
          reference_params,
          max_utils.with_memory_kind(reference_params_sharding, "device"),
      )
      extra_dpo_args = [reference_params]

    # Handle Sharding Constraints
    if config.shard_optimizer_over_data:
      params = jax.tree.map(jax.lax.with_sharding_constraint, params, params_shardings)

    grad_func = jax.value_and_grad(selected_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(
        model,
        config,
        data,
        dropout_rng,
        params,
        *extra_dpo_args,
        is_train=True,
    )

  # Cast grads if necessary
  raw_grads = jax.tree_util.tree_map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )
  return loss, aux, raw_grads


def _apply_updates(state, raw_grads, state_mesh_shardings, config):
  """Applies gradient clipping, handles memory moves, and updates the state."""
  # Gradient Clipping
  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads

  # Optimizer State Offloading
  if config.optimizer_memory_host_offload:
    state = state.replace(
        opt_state=jax.device_put(
            state.opt_state,
            jax.tree_util.tree_map(
                lambda x: x.with_memory_kind(kind="device"),
                state_mesh_shardings.opt_state,
            ),
        )
    )

  # Parameter Offloading (Move to device before update)
  if config.parameter_memory_host_offload:
    max_logging.log("\nMoving all parameters to device before optimizer update")

    def move(path, value):
      max_logging.log(f"train.py: Moving f{path} to device")
      return value.with_memory_kind(kind="device")

    state = state.replace(
        params=jax.device_put(
            state.params,
            jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params),
        )
    )

  # Apply Gradients
  new_state = state.apply_gradients(grads=grads)
  return new_state, grads


def _compute_metrics(loss, aux, grads, raw_grads, new_state, config):
  """Assembles the metrics dictionary."""
  intermediate_outputs = aux["intermediate_outputs"]

  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": aux["moe_lb_loss"],
      "learning/mtp_loss": aux["mtp_loss"],
      "learning/total_weights": aux["total_weights"],
  }

  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)

  if config.use_dpo:
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]

  metrics = {"scalar": scalar_metrics, "scalars": {}}

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  return metrics


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng):
  """
  Refactored train_step with modular helper functions.
  """
  (
      state,
      state_mesh_shardings,
      reference_params,
      reference_params_sharding,
      extra_dpo_args,
      selected_loss_fn,
  ) = _prepare_loss_functions_and_args(state, state_mesh_shardings, config)

  loss, aux, raw_grads = _compute_gradients_and_loss(
      model,
      state.params,
      params_shardings,
      data,
      dropout_rng,
      config,
      selected_loss_fn,
      extra_dpo_args,
      reference_params,
      reference_params_sharding,
  )

  new_state, grads = _apply_updates(state, raw_grads, state_mesh_shardings, config)

  metrics = _compute_metrics(loss, aux, grads, raw_grads, new_state, config)

  if config.use_dpo:
    new_state = _merge_dpo_state(new_state, reference_params)

  return new_state, metrics


# -----------------------------------------------------------------------------
# Eval Step Function
# -----------------------------------------------------------------------------


def eval_step(model, config, state, data, dropout_rng):
  """eval_step no backprop and new state compared with train_step."""

  reference_params, extra_dpo_args, _loss_fn = [], [], loss_fn
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  eval_loss_fn = functools.partial(_loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(state.params, *extra_dpo_args)

  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)

  total_loss = aux["total_loss"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  mtp_loss = aux["mtp_loss"]
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/total_loss": total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
          "evaluation/mtp_loss": mtp_loss,
          "evaluation/mtp_acceptance_rate_percent": mtp_acceptance_rate,
      },
  }
  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]

  return metrics


# -----------------------------------------------------------------------------
# Train Loop Function
# -----------------------------------------------------------------------------


def _compile_and_analyze_memory(config, p_train_step, state, state_mesh_shardings, mesh, init_rng):
  """Compiles the train step and prints memory analysis."""
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data:
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
    compiled_stats = compiled.memory_analysis()
    max_utils.print_compiled_memory_stats(compiled_stats)


def _run_evaluation_cycle(
    config,
    p_eval_step,
    state,
    eval_data_iterator,
    metric_logger,
    step,
    nextrng,
    mesh,
    prof,
):
  """Runs the evaluation loop and checks for early stopping."""
  if config.eval_interval > 0 and step > 0 and (step + 1) % config.eval_interval == 0:
    assert eval_data_iterator
    eval_data_iterator.reset()
    metric_logger.reset_eval_metrics()

    eval_step_count = 0
    for eval_batch in eval_data_iterator:
      if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
        break
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        eval_metrics = p_eval_step(state, eval_batch, nextrng)
      metric_logger.record_eval_metrics(step, metrics=eval_metrics)
      max_logging.log(f"Completed eval step {eval_step_count}")
      eval_step_count += 1

    metric_logger.record_eval_metrics(step, eval_step_count=eval_step_count)

    if metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"] <= config.target_eval_loss:
      prof.deactivate()
      raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} is achieved.")


def _handle_checkpoint_and_dump(config, state, checkpoint_manager, data_iterator, step):
  """Handles checkpoint saving and HLO dumping."""
  state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
  checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator, step)

  if config.dump_hlo and step == (config.dump_step if config.dump_step >= 0 else 0):
    jax.block_until_ready(state)
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )


def train_loop(config, recorder, state=None):
  """Main Training loop."""
  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config,
      model,
      mesh,
      state,
      state_mesh_shardings,
      train_step,
      eval_step,
      eval_data_iterator,
      params_shardings,
  )

  _compile_and_analyze_memory(config, p_train_step, state, state_mesh_shardings, mesh, init_rng)

  start_step = get_first_step(state)
  prof = profiler.Profiler(config, offset_step=start_step)
  data_loader = create_dataloader(config, mesh, data_iterator, recorder)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)
  metric_logger.write_setup_info_to_tensorboard(state.params)

  try:
    last_step_completion = datetime.datetime.now()

    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch()
        example_batch = sharding.maybe_shard_with_name(
            example_batch,
            sharding.get_input_data_sharding(config, mesh),
            shard_mode=config.shard_mode,
        )
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)

        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.shard_optimizer_over_data:
              state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
            state, metrics = p_train_step(state, example_batch, nextrng)

      step_time_delta = datetime.datetime.now() - last_step_completion
      last_step_completion = datetime.datetime.now()

      _handle_checkpoint_and_dump(config, state, checkpoint_manager, data_iterator, step)

      _run_evaluation_cycle(
          config,
          p_eval_step,
          state,
          eval_data_iterator,
          metric_logger,
          step,
          nextrng,
          mesh,
          prof,
      )

      prof.maybe_deactivate_profiler(step, state)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)

    if config.save_checkpoint_on_completion:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator)
    if checkpoint_manager is not None:
      checkpoint_manager.wait_until_finished()

  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
  finally:
    metric_logger.flush_metrics_and_cleanup()

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any, Any]:
  """Initialization of hyperparameters and utilities"""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  # TODO: mazumdera@ : ensure missing mandatory fields in base.yml are filled in in argv,
  # or fill in here
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  validate_train_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  # update explicit sharding-supported config
  if config.shard_mode == ShardMode.EXPLICIT:
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Create the Goodput recorder
  recorder = create_goodput_recorder(config)

  # Stack traces configurations
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  return config, recorder, diagnostic_config


def run(config, recorder, diagnostic_config):
  """Run the job given hyperparameters and utilities"""
  with (
      diagnostic.diagnose(diagnostic_config),
      maybe_record_goodput(recorder, GoodputEvent.JOB),
      max_utils.maybe_get_transformer_engine_context(config),
      maybe_monitor_goodput(config),
  ):
    train_loop(config, recorder)


def main(argv: Sequence[str]) -> None:
  config, recorder, diagnostic_config = initialize(argv)
  run(config, recorder, diagnostic_config)


if __name__ == "__main__":
  app.run(main)
