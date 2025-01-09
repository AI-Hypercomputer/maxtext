"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
import sys
import functools
import time

from typing import Sequence, Optional
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import optimizers
import profiler
import pyconfig
import pathwaysutils  # pylint: disable=unused-import
import tensorflow as tf

from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import models

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import quantizations

from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring

# pylint: disable=too-many-positional-arguments

Transformer = models.Transformer
EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3


def validate_train_config(config):
  """Validates the configuration is set correctly for train.py"""

  assert config.run_name, "Erroring out, need a real run_name"
  if not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."

  if config.quantization == "fp8":
    # pylint: disable=line-too-long
    assert (
        config.gradient_accumulation_steps == 1
    ), "fp8 can't be used with gradient_accumulation_steps right now. Please use other quantization or set gradient_accumulation_steps to 1"


def get_first_step(state):
  with jax.spmd_mode("allow_all"):
    return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons"""

  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return next(train_iter)


def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr, per_device_tokens):
  """Records scalar metrics to be written to tensorboard"""
  metrics["scalar"].update({"perf/step_time_seconds": step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tflops": per_device_tflops})
  metrics["scalar"].update({"perf/per_device_tflops_per_sec": per_device_tflops / step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tokens": per_device_tokens})
  metrics["scalar"].update({"perf/per_device_tokens_per_sec": per_device_tokens / step_time_delta.total_seconds()})
  metrics["scalar"].update({"learning/current_learning_rate": lr})


_buffered_step = None
_buffered_metrics = None


def write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config, is_training=True):
  """Entry point for all metrics writing in Train's Main.
  TODO: would be better as a Class in the future (that initialized all state!)

  To avoid introducing an unnecessary dependency, we "double buffer" -- we hold
  onto the last metrics and step and only publish when we receive a new metrics and step.
  The logic is that this ensures that Jax is able to queues train_steps and we
  don't block when turning "lazy" Jax arrays into real Python numbers.
  """
  metrics_to_write, steps_to_write = None, None
  if is_training:
    global _buffered_step, _buffered_metrics
    if _buffered_metrics is not None:
      if _buffered_step is None:
        raise ValueError(f"When writing metrics, {_buffered_step=} was none")
      metrics_to_write = _buffered_metrics
      steps_to_write = _buffered_step
  else:
    metrics_to_write = metrics
    steps_to_write = step

  if metrics_to_write:
    write_metrics_to_tensorboard(writer, metrics_to_write, steps_to_write, config, is_training)

    if config.metrics_file:
      max_utils.write_metrics_locally(metrics_to_write, steps_to_write, config, local_metrics_file, is_training)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(
          metrics_to_write, steps_to_write, config, running_gcs_metrics, is_training
      )

  if is_training:
    _buffered_step = step
    _buffered_metrics = metrics


def write_metrics_to_tensorboard(writer, metrics, step, config, is_training=True):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode("allow_all"):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars", []):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    if is_training:
      full_log = step % config.log_period == 0

      max_logging.log(
          f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
          f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
          f"Tokens/s/device: {metrics['scalar']['perf/per_device_tokens_per_sec']:.3f}, "
          f"total_weights: {metrics['scalar']['learning/total_weights']}, "
          f"loss: {metrics['scalar']['learning/loss']:.3f}"
      )

      if full_log and jax.process_index() == 0:
        max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")
        writer.flush()


def clear_buffered_metrics():
  global _buffered_step
  global _buffered_metrics
  _buffered_step = None
  _buffered_metrics = None


def save_checkpoint(
    checkpoint_manager,
    step,
    state,
    dataset_type="c4",
    data_iterator=None,
    config: Optional[pyconfig.config] = None,
) -> bool:
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (step % config.checkpoint_period == 0) or (
        config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  if config:
    chunk_byte_size = config.checkpoint_storage_target_data_file_size_bytes
  save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=chunk_byte_size), state)

  if isinstance(checkpoint_manager, emergency_checkpoint_manager.CheckpointManager):
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.PyTreeSave(item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size),
    )

  if dataset_type == "grain":
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            ),
            iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
        ),
    )
  else:
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            )
        ),
    )


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """Adds the activation metrics to the metrics dict"""

  if config.scan_layers:
    metrics_dict = intermediate_outputs["intermediates"]["decoder"]["decoder"]

    for layer_num in range(config.num_decoder_layers):
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = metrics_dict["activation_fraction_zero"][0][
          layer_num
      ]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs["intermediates"]["decoder"][f"layers_{layer_num}"]
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = layer["activation_fraction_zero"][0]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = layer["activation_mean"][0]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = layer["activation_stdev"][0]


def _split_dpo_state(state):
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params


def _merge_dpo_state(state, reference_params):
  return state.replace(params=dict(state.params, reference_params=reference_params))


def dpo_loss_fn(model, config, data, dropout_rng, params, reference_params, is_train=True):
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
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]

  # for DPO we don't support packed sequence (they shouldn't be present in the first place)
  data["chosen_segmentation"] = (data["chosen_segmentation"] == 1).astype(jnp.int32)
  data["rejected_segmentation"] = (data["rejected_segmentation"] == 1).astype(jnp.int32)
  data["chosen_position"] = data["chosen_position"] * (data["chosen_segmentation"] == 1)
  data["rejected_position"] = data["rejected_position"] * (data["rejected_segmentation"] == 1)

  # concatenated model and reference model forward pass
  inputs = jnp.concatenate([data["chosen"], data["rejected"]], 0)
  inputs_position = jnp.concatenate([data["chosen_position"], data["rejected_position"]], 0)
  inputs_segmentation = jnp.concatenate([data["chosen_segmentation"], data["rejected_segmentation"]], 0)

  logits, intermediate_outputs = model.apply(
      params,
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  ref_logits = model.apply(
      {"params": reference_params},
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
  )
  ref_logits = jax.lax.stop_gradient(ref_logits)

  # extract token ids, segmentation and logits for chosen and rejected sequences
  chosen_ids = data["chosen"][..., 1:]
  rejected_ids = data["rejected"][..., 1:]
  chosen_segmentation = data["chosen_segmentation"][..., 1:]
  rejected_segmentation = data["rejected_segmentation"][..., 1:]
  n_logits = logits.shape[-3] // 2  # [B, S, E] - [batch, sequence, embedding/vocab]
  chosen_logits, rejected_logits = logits[:n_logits, :, :], logits[n_logits:, :, :]  # [B, S, E], [B, S, E]
  chosen_ref_logits, rejected_ref_logits = ref_logits[:n_logits, :, :], ref_logits[n_logits:, :, :]  # [B, S, E], [B, S, E]

  # common subsequence and padding mask
  common_prefix_mask = jnp.cumsum(chosen_ids != rejected_ids, axis=-1) == 0  # [B, S]
  valid_seq_mask = (chosen_segmentation != 0) & (rejected_segmentation != 0) & ~common_prefix_mask  # [B, S]

  # compute logratios from the sequence-reduced observed token log-probability
  chosen_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_logps = jnp.sum(chosen_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(chosen_ref_logits[..., :-1, :], axis=-1), chosen_ids[..., None], axis=-1
  )[..., 0]
  chosen_ref_logps = jnp.sum(chosen_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  chosen_logratios = chosen_logps - chosen_ref_logps  # [B]

  rejected_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_logps = jnp.sum(rejected_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_ref_logps_seq = jnp.take_along_axis(  # [B, S]
      jax.nn.log_softmax(rejected_ref_logits[..., :-1, :], axis=-1), rejected_ids[..., None], axis=-1
  )[..., 0]
  rejected_ref_logps = jnp.sum(rejected_ref_logps_seq * valid_seq_mask, axis=-1)  # [B]
  rejected_logratios = rejected_logps - rejected_ref_logps  # [B]

  # DPO loss from chosen and rejected logratios
  LABEL_SMOOTHING, BETA = config.dpo_label_smoothing, config.dpo_beta
  logratios_delta = BETA * (chosen_logratios - rejected_logratios)  # [B]
  losses = (  # [B]
      -jax.nn.log_sigmoid(BETA * logratios_delta) * (1 - LABEL_SMOOTHING)
      - jax.nn.log_sigmoid(-BETA * logratios_delta) * LABEL_SMOOTHING
  )
  total_loss, total_weights = jnp.mean(losses), losses.shape[0]
  loss = total_loss

  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss
  reward_accuracy = jnp.mean(chosen_logratios > rejected_logratios)
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "reward_accuracy": reward_accuracy,
  }
  return loss, aux


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
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]

  logits, intermediate_outputs = model.apply(
      params,
      data["inputs"],
      data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, 0.0)
  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  # Mask out paddings at the end of each example.
  xent = xent * (data["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + EPS)
  # get moe load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
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
      acc_grad_and_loss["grad"] = jax.tree_util.tree_map(
          lambda x, y: x * aux["total_weights"] + y, cur_batch_gradient, acc_grad_and_loss["grad"]
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
    init_grad_and_loss = {"loss": 0.0, "grad": init_grad, "total_weights": 0, "moe_lb_loss": 0.0}

    grad_and_loss, aux = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, data, length=config.gradient_accumulation_steps
    )
    loss = (
        grad_and_loss["loss"] / grad_and_loss["total_weights"]
        + grad_and_loss["moe_lb_loss"] / config.gradient_accumulation_steps
    )
    raw_grads = jax.tree_util.tree_map(lambda arr: arr / grad_and_loss["total_weights"], grad_and_loss["grad"])
    aux = jax.tree_map(lambda x: jnp.sum(x, axis=0), aux)
  else:
    if config.optimizer_memory_host_offload:
      cast_params = jax.device_put(state.params, max_utils.with_memory_kind(state_mesh_shardings.params, "device"))
      cast_params = max_utils.cast_to_bf16(cast_params)
      state = state.replace(params=cast_params)
      if config.use_dpo:
        reference_params = jax.device_put(reference_params, max_utils.with_memory_kind(reference_params_sharding, "device"))
        reference_params = max_utils.cast_to_bf16(reference_params)
        extra_dpo_args = [reference_params]
    grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, *extra_dpo_args, is_train=True)
  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]

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
  new_state = state.apply_gradients(grads=grads)

  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": moe_lb_loss,
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
  total_loss = aux["total_loss"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/total_loss": total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
      },
  }
  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]

  return metrics


def create_goodput_recorder(config):
  if config.enable_goodput_recording:
    logger_name = f"goodput_{config.run_name}"
    recorder = goodput.GoodputRecorder(config.run_name, logger_name, jax.process_index() == 0)
    return recorder
  return None


def record_goodput(
    recorder,
    config,
    record_func,
    *args,
):
  """Record data for Goodput and Badput computation."""
  if recorder and config.enable_goodput_recording:
    record_func(*args)


def check_example_batch(config, example_batch):
  if config.max_checkify:
    jittable_f = checkify.checkify(lambda x: checkify.check(jnp.any(x > -1), "Batch contains bad synthetic data!"))
    # Check if inputs in batch contains bad synthetic data.
    # pylint: disable=not-callable
    err, _ = jax.jit(jittable_f)(example_batch["inputs"][: config.global_batch_size_to_train_on, :])
    err.throw()


def setup_mesh_and_model(config):
  """Set up the mesh and the model for training

  Args:
    config

  Returns:
    init_rng: RNG key
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    tx:
  """

  init_rng = random.PRNGKey(config.init_weights_seed)
  writer = max_utils.initialize_summary_writer(config)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant=quant)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_emergency_checkpoint:
    if config.use_replicator_service:
      checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
          config.local_checkpoint_directory,
          config.local_checkpoint_period,
          mesh,
      )
    else:
      abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
      checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
          config.local_checkpoint_directory,
          config.checkpoint_dir,
          mesh,
          abstract_state,
          config.local_checkpoint_period,
          config.checkpoint_period,
          logger,
      )
  else:
    # TODO(b/368121306): Remove this once zarr3 support is plumbed on the backend
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller:
      use_ocdbt, use_zarr3 = False, False
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
    )

  return init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx


def setup_train_loop(config):
  """Set up prerequisites for the training loop -
      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args:
    config

  Returns:
    init_rng:
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    state: the initialized train state
  """
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, recorder.record_tpu_init_start_time if recorder else None)
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
  record_goodput(recorder, config, recorder.record_tpu_init_end_time if recorder else None)
  record_goodput(recorder, config, recorder.record_training_preparation_start_time if recorder else None)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)

  state, _, state_mesh_shardings, data_iterator = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )

  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

  if config.use_dpo:
    abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    max_logging.log(f"Restoring reference parameters for DPO from '{os.path.join(str(config.checkpoint_dir), str(0))}'")
    try:
      step0_restored, _ = checkpointing.load_state_if_possible(
          checkpoint_manager,
          data_iterator,
          load_parameters_from_path="",
          load_full_state_from_path="",
          abstract_unboxed_pre_state=abstract_state,
          enable_single_replica_ckpt_restoring=False,
          dataset_type=config.dataset_type,
          step=0,
      )
    except FileNotFoundError:
      step0_restored = None
    if step0_restored is not None:
      reference_params = step0_restored["items"].params["params"]
      state = _merge_dpo_state(state, reference_params)
    else:
      max_logging.log(
          f"Could not restore reference parameters for DPO from '{os.path.join(str(config.checkpoint_dir), str(0))}'"
      )

  record_goodput(recorder, config, recorder.record_training_preparation_end_time if recorder else None)
  return (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  )


def train_loop(config, state=None):
  """Main Training loop.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  # Create a GoodputRecorder to log information
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, recorder.record_job_start_time if recorder else None)

  (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  ) = setup_train_loop(config)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  # pylint: disable=line-too-long
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

  if eval_data_iterator:
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, config)

  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
  per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)

  # Write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  max_utils.add_config_to_summary_writer(config, writer)

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    print("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    # TODO: p_eval_step is not yet supported in load_compiled
    p_eval_step = None
    print("Loaded compiled function!", flush=True)
  else:
    p_train_step = jax.jit(
        functional_train,
        in_shardings=in_shard_train,
        out_shardings=out_shard_train,
        static_argnums=static_argnums_train,
        donate_argnums=donate_argnums_train,
    )

    if eval_data_iterator:
      p_eval_step = jax.jit(
          functional_eval,
          in_shardings=in_shard_eval,
          out_shardings=out_shard_eval,
          static_argnums=static_argnums_eval,
          donate_argnums=donate_argnums_eval,
      )
    else:
      p_eval_step = None

  local_metrics_file = open(config.metrics_file, "a", encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  first_profiling_step = prof.start_initial_profile_step
  if config.profiler != "" and first_profiling_step >= config.steps:
    raise ValueError("Profiling requested but initial profiling step set past training final step")
  last_profiling_step = prof.finished_initial_profile_step

  example_batch = None
  last_step_completion = datetime.datetime.now()

  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step or prof.should_activate_periodic_profile(step):
      optional_postfix = f"step_{step}" if config.profile_periodically_period > 0 else ""
      prof.activate(blocking_object=state, optional_postfix=optional_postfix)

    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      record_goodput(recorder, config, recorder.record_data_loading_start_time if recorder else None)
      example_batch = load_next_batch(data_iterator, example_batch, config)
      record_goodput(recorder, config, recorder.record_data_loading_end_time if recorder else None)
      check_example_batch(config, example_batch=example_batch)
      # pylint: disable=not-callable
      nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
      record_goodput(recorder, config, recorder.record_step_start_time if recorder else None, step)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state, metrics = p_train_step(state, example_batch, nextrng)

    new_time = datetime.datetime.now()
    record_scalar_metrics(
        metrics, new_time - last_step_completion, per_device_tflops, learning_rate_schedule(step), per_device_tokens
    )
    last_step_completion = new_time

    if checkpoint_manager is not None:
      state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
      if save_checkpoint(checkpoint_manager, int(step), state_to_save, config.dataset_type, data_iterator, config):
        checkpointing.print_save_message(step, config.async_checkpointing)

      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()

    write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config)

    if config.dump_hlo and step == start_step:
      jax.block_until_ready(state)  # Ensure compilation has finished.
      max_utils.upload_dump(
          config.dump_hlo_local_dir,
          config.dump_hlo_gcs_dir,
          module_name=config.dump_hlo_module_name,
          delete_local_after=config.dump_hlo_delete_local_after,
          all_host_upload=config.dump_hlo_upload_all,
      )

    if config.eval_interval > 0 and step > start_step and (step + 1) % config.eval_interval == 0:
      assert eval_data_iterator
      cumulative_eval_metrics = {
          "scalar": {
              "eval/total_loss": 0.0,
              "eval/total_weights": 0.0,
              "eval/avg_loss": 0.0,
              "eval/moe_lb_loss": 0.0,
          }
      }
      eval_dpo_reward_accuracy = 0.0
      eval_step_count = 0
      # pylint: disable=not-callable
      for eval_batch in eval_data_iterator:
        if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
          break
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          eval_metrics = p_eval_step(state, eval_batch, nextrng)
        cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
        cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
        cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
        eval_dpo_reward_accuracy += float(eval_metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0))  # for dpo only
        max_logging.log(f"Completed eval step {eval_step_count}")
        eval_step_count += 1
      eval_loss = (
          cumulative_eval_metrics["scalar"]["eval/total_loss"]
          / (cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS)
          + cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
      cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      if config.use_dpo:
        cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = eval_dpo_reward_accuracy / eval_step_count
      write_metrics(
          writer, local_metrics_file, running_gcs_metrics, cumulative_eval_metrics, step, config, is_training=False
      )
      max_logging.log(
          f"average loss after {step=}: {eval_step_count=}, {eval_loss=},"
          f" total_weights={cumulative_eval_metrics['scalar']['eval/total_weights']}"
      )
      if eval_loss <= config.target_eval_loss:
        max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
        prof.deactivate()
        break

    if step == last_profiling_step or prof.should_deactivate_periodic_profile(step):
      prof.deactivate(blocking_object=state)

    if step == start_step:
      max_utils.print_mem_stats("After params initialized")

  if checkpoint_manager is not None:
    checkpoint_manager.wait_until_finished()
  write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, config.steps - 1, config)  # final step metrics
  max_utils.close_summary_writer(writer)
  record_goodput(recorder, config, recorder.record_job_end_time if recorder else None)
  clear_buffered_metrics()
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    # pytype: disable=attribute-error
    compiled = p_train_step.lower(state, example_batch, nextrng).compile()
    compiled_stats = compiled.memory_analysis()
    if compiled_stats is not None:
      max_logging.log(
          f"Output size: {compiled_stats.output_size_in_bytes}, "
          f"temp size: {compiled_stats.temp_size_in_bytes}, "
          f"argument size: {compiled_stats.argument_size_in_bytes}, "
          f"host temp size: {compiled_stats.host_temp_size_in_bytes}, in bytes."
      )
  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  config = pyconfig.config
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  if config.monitor_goodput and jax.process_index() == 0:
    logger_name = f"goodput_{config.run_name}"
    goodput_monitor = monitoring.GoodputMonitor(
        job_name=config.run_name,
        logger_name=logger_name,
        tensorboard_dir=config.tensorboard_dir,
        upload_interval=config.goodput_upload_interval_seconds,
        monitoring_enabled=True,
        pathway_enabled=config.enable_pathways_goodput,
        include_badput_breakdown=True,
    )
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Goodput upload to Tensorboard in the background!")
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config)


if __name__ == "__main__":
  app.run(main)
