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
from MaxText.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from MaxText.data_loader import DataLoader
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

from MaxText.maxtext_utils import compute_loss_nnx, compute_loss_linen
from MaxText.dpo_utils import _merge_dpo_state, _split_dpo_state, dpo_loss_fn
from MaxText.train_utils import validate_train_config
from MaxText.metric_logger import record_activation_metrics
# pylint: disable=too-many-positional-arguments


def get_first_step(state):
  return int(state.step)


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]
  mutable_collections = ["intermediates"]
  if config.mtp_num_layers > 0 and is_train:
    # The single model.apply call now triggers the entire chain if MTP is enabled:
    # Decoder runs -> returns hidden_state -> MTPBlock uses it -> MTPBlock sows losses -> we reap them here.
    mutable_collections.append("mtp_losses")

  # During evaluation, if the acceptance rate test is enabled, we must
  # make its specific collection mutable so the MTPBlock can sow into it.
  if config.mtp_eval_target_module > 0 and not is_train:
    mutable_collections.append("mtp_acceptance")

  if isinstance(model, nn.Module):
    # inputs, targets, segments, positions = apply_args
    rng1, aqt_rng = jax.random.split(dropout_rng)

    # Flax Linen model
    logits, intermediate_outputs = model.apply(
        params,
        data["inputs"],
        data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        enable_dropout=config.enable_dropout if is_train else False,
        rngs={"dropout": rng1, "params": aqt_rng},
        mutable=mutable_collections,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )

    total_loss = compute_loss_linen(intermediate_outputs, logits, data, config, model, params, is_train)
  else:
    # Flax NNX model
    logits = model(
        decoder_input_tokens=data["inputs"],
        decoder_positions=data["inputs_position"],
        decoder_segment_ids=data["inputs_segmentation"],
        encoder_images=data["images"] if config.use_multimodal else None,
        enable_dropout=config.enable_dropout if is_train else False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )
    intermediate_outputs = {}
    total_loss = compute_loss_nnx(intermediate_outputs, logits, data, config, model, params, is_train)

  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  # If gradient accumulation is enabled, we don't need to divide total_loss
  # by total_weights and then multiply the computed gradient by total_weights,
  # since it's equivalent to computing the gradient from total_loss.
  # This simplification reduces the number of operations and makes it easier
  # for XLA to move all-reduce out of the gradient accumulation loop when use
  # Zero1+GA to reduce communication overhead.
  # EPS was used to avoid division by zero, but it's not needed when gradient
  # accumulation is enabled since there's no division.
  if config.gradient_accumulation_steps > 1:
    loss = total_loss
  else:
    loss = total_loss / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get moe load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "mtp_loss": mtp_loss,
  }
  return loss, aux


def train_step(model, config, state_mesh_shardings, state, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  reference_params, reference_params_sharding, extra_dpo_args, _loss_fn = [], [], [], loss_fn
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  if config.gradient_accumulation_steps > 1:

    def accumulate_gradient(acc_grad_and_loss, data):
      grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
      (_, aux), cur_batch_gradient = grad_func(
          model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True
      )
      acc_grad_and_loss["loss"] += aux["total_loss"]
      acc_grad_and_loss["moe_lb_loss"] += aux["moe_lb_loss"]
      acc_grad_and_loss["mtp_loss"] += aux["mtp_loss"]
      acc_grad_and_loss["grad"] = jax.tree_util.tree_map(
          lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"]
      )
      acc_grad_and_loss["total_weights"] += aux["total_weights"]
      return acc_grad_and_loss, aux

    def reshape_to_microbatch_accumulations(batch_arr):
      """Reshape global batch to microbatches, assuming batch axis is leading."""
      microbatches = config.gradient_accumulation_steps
      microbatch_shape = (microbatches, batch_arr.shape[0] // microbatches) + batch_arr.shape[1:]
      return jnp.reshape(batch_arr, microbatch_shape)

    data = jax.tree_util.tree_map(reshape_to_microbatch_accumulations, data)
    init_grad = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init_grad_and_loss = {"loss": 0.0, "grad": init_grad, "total_weights": 0, "moe_lb_loss": 0.0, "mtp_loss": 0.0}

    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
    )
    loss = (
        grad_and_loss["loss"] / grad_and_loss["total_weights"]
        + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
        + grad_and_loss["mtp_loss"] / config.gradient_accumulation_steps
    )
    raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], grad_and_loss["grad"])
    aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)  # pytype: disable=module-attr
  else:
    if config.optimizer_memory_host_offload:
      if config.use_dpo:
        reference_params = jax.device_put(
            reference_params, max_utils.with_memory_kind(reference_params_sharding, "device")
        )
        extra_dpo_args = [reference_params]
    grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True)
  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  mtp_loss = aux["mtp_loss"]

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  if config.optimizer_memory_host_offload:
    state = state.replace(
        opt_state=jax.device_put(
            state.opt_state,
            jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="device"), state_mesh_shardings.opt_state),
        )
    )
  # Move all parameters to device before optimizer update
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
  new_state = state.apply_gradients(grads=grads)

  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": moe_lb_loss,
      "learning/mtp_loss": mtp_loss,
      "learning/total_weights": total_weights,
  }
  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)
  if config.use_dpo:
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  if config.use_dpo:
    new_state = _merge_dpo_state(new_state, reference_params)

  return new_state, metrics


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

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config, model, mesh, state, state_mesh_shardings, train_step, eval_step, eval_data_iterator
  )

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
    compiled_stats = compiled.memory_analysis()
    max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  data_loader = DataLoader(config, mesh, data_iterator, recorder)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params)

  try:
    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch()
        # pylint: disable=not-callable
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            state, metrics = p_train_step(state, example_batch, nextrng)

      step_time_delta = datetime.datetime.now() - last_step_completion
      last_step_completion = datetime.datetime.now()

      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator, step)

      if config.dump_hlo and step == (config.dump_step if config.dump_step >= 0 else start_step):
        jax.block_until_ready(state)  # Ensure compilation has finished.
        gcs_utils.upload_dump(
            config.dump_hlo_local_dir,
            config.dump_hlo_gcs_dir,
            module_name=config.dump_hlo_module_name,
            delete_local_after=config.dump_hlo_delete_local_after,
            all_host_upload=config.dump_hlo_upload_all,
        )

      if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
        assert eval_data_iterator
        # Explicitly reset the eval iterator and counters before starting the eval loop
        eval_data_iterator.reset()
        metric_logger.reset_eval_metrics()

        eval_step_count = 0
        # pylint: disable=not-callable
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

      prof.maybe_deactivate_profiler(step, state)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)

    if config.save_checkpoint_on_completion:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator)
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
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Goodput configurations
  maybe_monitor_goodput(config)
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
  with diagnostic.diagnose(diagnostic_config):
    with maybe_record_goodput(recorder, GoodputEvent.JOB):
      train_loop(config, recorder)


def main(argv: Sequence[str]) -> None:
  config, recorder, diagnostic_config = initialize(argv)
  run(config, recorder, diagnostic_config)


if __name__ == "__main__":
  app.run(main)
