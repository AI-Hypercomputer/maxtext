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

from typing import Any, Dict, Sequence
import contextlib
import datetime
import functools
import os

from absl import app

import numpy as np

import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from flax import linen as nn, nnx
from flax.linen import partitioning as nn_partitioning

from MaxText import pyconfig
from MaxText import sharding
from MaxText.layers import train_state_nnx
from MaxText.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from MaxText.common_types import ShardMode
from MaxText.globals import EPS
# Placeholder: internal

from MaxText.gradient_accumulation import gradient_accumulation_loss_and_grad
from MaxText.vocabulary_tiling import vocab_tiling_linen_loss
# pylint: disable=too-many-positional-arguments

from maxtext.common import checkpointing, profiler
from maxtext.common.goodput import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from maxtext.common.gcloud_stub import cloud_diagnostics as _cloud_diag, is_decoupled
from maxtext.common.gcloud_stub import vertex_tensorboard_modules
from maxtext.common.metric_logger import MetricLogger, record_activation_metrics
from maxtext.trainers.post_train.dpo.dpo_utils import _merge_dpo_state, _split_dpo_state, dpo_loss_fn
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils, maxtext_utils_nnx
from maxtext.utils import train_utils

_diag_modules = _cloud_diag()
diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration = _diag_modules
VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()


def get_first_step(state):
  if isinstance(state, nnx.State):
    return int(state.optimizer.step.get_value())
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
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        rngs={"dropout": rng1, "params": aqt_rng},
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
        encoder_image_masks=data["image_masks"] if config.use_multimodal and "image_masks" in data else None,
        enable_dropout=config.enable_dropout if is_train else False,
        decoder_target_tokens=data["targets"],
        decoder_target_mask=data["targets_segmentation"],
    )

    # Capture NNX intermediates (MoE losses, hidden states, etc.)
    intermediate_outputs = nnx.state(model, nnx.Intermediate).to_pure_dict()

  # Vocab Tiling Logic
  if config.num_vocab_tiling > 1:
    hidden_state_key = ("intermediates", "decoder", "hidden_states")
    hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]

    if isinstance(model, nn.Module):
      total_loss = vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)
    else:
      raise NotImplementedError("Vocab tiling for NNX modules has not been implemented.")
  else:
    # 5. Standard Cross Entropy Loss
    one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)

    logical_axes = ("activation_embed_and_logits_batch", "activation_length")
    if isinstance(model, nn.Module):
      xent = sharding.maybe_shard_with_logical(
          xent, logical_axes, model.mesh, config.shard_mode, debug_sharding=config.debug_sharding
      )
    else:
      xent = nn.with_logical_constraint(xent, logical_axes)

    xent = xent * (data["targets_segmentation"] != 0)
    total_loss = jnp.sum(xent)

  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  # If gradient accumulation is enabled, we don't need to divide total_loss
  # by total_weights and then multiply the computed gradient by total_weights,
  # since it's equivalent to computing the gradient from total_loss.
  # This simplification reduces the number of operations and makes it easier
  # for XLA to move all-reduce out of the gradient accumulation loop when use
  # Zero1+GA to reduce communication overhead.
  # EPS was used to avoid division by zero, but it's not needed when gradient
  # accumulation is enabled since there's no division.
  if config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation:
    loss = total_loss
  else:
    # When using Tunix gradient accumulation, we revert to standard normalization.
    # Unlike the manual accumulation path above, Tunix (via optax.MultiSteps) expects
    # a normalized loss for each step. It handles the accumulation state
    # updates and scaling internally.
    loss = total_loss / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get MoE load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss

  # get MoE routed bias term updates
  moe_bias_updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    nested_key = ("intermediates", "decoder", "moe_layers", "moe_bias_updates")
    moe_bias_updates = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, None)

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "moe_bias_updates": moe_bias_updates,
      "mtp_loss": mtp_loss,
  }
  return loss, aux


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng):
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
  reference_params, reference_params_sharding, extra_dpo_args, _loss_fn = (
      [],
      [],
      [],
      loss_fn,
  )
  if config.use_dpo:
    state, reference_params = _split_dpo_state(state)
    state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    extra_dpo_args = [reference_params]
    _loss_fn = dpo_loss_fn

  params = state.params

  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        _loss_fn,
        config,
        model,
        params,
        params_shardings,
        data,
        dropout_rng,
        extra_dpo_args,
    )
  else:
    if config.optimizer_memory_host_offload:
      if config.use_dpo:
        reference_params = jax.device_put(
            reference_params,
            max_utils.with_memory_kind(reference_params_sharding, "device"),
        )
        extra_dpo_args = [reference_params]
    if config.shard_optimizer_over_data:
      params = jax.tree.map(
          functools.partial(sharding.maybe_shard_with_name, shard_mode=config.shard_mode),
          params,
          params_shardings,
      )
    grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, params, *extra_dpo_args, is_train=True)

  raw_grads = jax.tree_util.tree_map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )
  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  moe_bias_updates = aux["moe_bias_updates"]
  mtp_loss = aux["mtp_loss"]

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
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

  # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
  if config.routed_bias and config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
    target_path = ("params", "decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")
    # Flax 'sow' returns a tuple, so we take the first element [0].
    # Updates the shape to be aligned with state.
    moe_bias_updates = jnp.array(moe_bias_updates[0]).transpose()
    new_state = maxtext_utils.update_state_param(new_state, target_path, moe_bias_updates)

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


def train_step_nnx(
    graphdef: nnx.GraphDef,
    config: pyconfig.HyperParameters,
    state_mesh_shardings: nnx.State,  # sharding of the TrainStateNNX module
    params_shardings: nnx.State,  # sharding of the model module
    pure_state: nnx.State,  # primary state including model, optimizer, and rngs
    data: Any,
) -> tuple[nnx.Module, Dict[str, Any]]:
  """
  NNX version of the training step.

  Args:
    graphdef: The graphdef of the state.
    config: Hyperparameters.
    state_mesh_shardings: PyTree of PartitionSpecs for the whole TrainStateNNX container.
    params_shardings: PyTree of PartitionSpecs for the preserved primary model parameters, used for gradient accumulation.
    data: Training data batch.
    pure_state: The TrainStateNNX pure state holding the primary model, optimizer, and rngs.

  Returns:
    new_state_nnx: Updated TrainStateNNX container.
    metrics: Training metrics.
  """
  # DPO Logic: Reference params handling
  # reference_params = None
  extra_dpo_args = []
  _loss_fn = loss_fn

  # Re-construct the TrainStateNNX module
  state = nnx.merge(graphdef, pure_state)

  if config.use_dpo:
    raise NotImplementedError("DPO for NNX modules has not been implemented.")

  # 1. Gradient Computation
  if config.gradient_accumulation_steps > 1:
    # (Simplified for context, assumes GA helper is updated)
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        _loss_fn, config, state.model, None, params_shardings, data, None, extra_dpo_args
    )
  else:
    # Handle Memory Host Offload for Reference Params
    if config.optimizer_memory_host_offload and config.use_dpo:
      raise NotImplementedError("DPO for NNX modules has not been implemented.")

    graphdef, curr_params, rest = nnx.split(state.model, nnx.Param, ...)

    # Handle Sharding Optimizer Over Data
    if config.shard_optimizer_over_data:
      curr_params = jax.tree.map(
          functools.partial(sharding.maybe_shard_with_name, shard_mode=config.shard_mode),
          curr_params,
          params_shardings,
      )
      nnx.update(state.model, curr_params)

    # Define a local "pure" function for JAX to differentiate
    def diff_wrapper(param, rest, config, data):
      # Re-create the model object inside the trace using the static graphdef
      local_model = nnx.merge(graphdef, param, rest)
      loss, aux = _loss_fn(local_model, config, data, None, None, is_train=True)
      _, _, new_rest = nnx.split(local_model, nnx.Param, ...)

      return loss, (aux, new_rest)

    grad_func = jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)
    # grad_func = nnx.value_and_grad(diff_wrapper, argnums=0, has_aux=True)
    (loss, (aux, new_rest)), raw_grads = grad_func(curr_params, rest, config, data)
    nnx.update(state.model, new_rest)
  # 2. Process Gradients (Standardize precision)
  raw_grads = jax.tree.map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )

  # Gradient Clipping
  grads = raw_grads
  if config.gradient_clipping_threshold > 0:
    # Gradient clipping doesn't need state. So we pass OptState as None.
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, config.gradient_clipping_threshold)

  # 3. Memory Host Offload
  if config.optimizer_memory_host_offload:
    state.optimizer = jax.device_put(
        state.optimizer,
        jax.tree_util.tree_map_with_path(
            maxtext_utils_nnx.move_memory_to_device,
            state_mesh_shardings.optimizer,
            is_leaf=lambda x: isinstance(x, NamedSharding),
        ),
    )

  if config.parameter_memory_host_offload:
    _, state_params, _ = nnx.split(state_mesh_shardings, nnx.Param, ...)
    state_params = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_device,
        state_params,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )
    nnx.update(state_mesh_shardings, state_params)
    state.model = jax.device_put(state.model, state_mesh_shardings.model)

  # 4. Apply Gradients. See TrainStateNNX.apply_gradients().
  state.apply_gradients(grads)

  # 5. MoE Bias Updates
  if config.routed_bias and config.routed_bias_update_rate > 0.0 and aux["moe_bias_updates"] is not None:
    updates = jnp.array(aux["moe_bias_updates"][0]).transpose()
    target_bias = state.model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias
    target_bias.value = target_bias.value + updates

  # 6. Metrics Collection
  scalar_metrics = {
      "learning/loss": loss,
      "learning/moe_lb_loss": aux["moe_lb_loss"],
      "learning/mtp_loss": aux["mtp_loss"],
      "learning/total_weights": aux["total_weights"],
  }

  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    _, model_params, _ = nnx.split(state.model, nnx.Param, ...)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(model_params)

  if config.use_dpo:
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]

  metrics = {"scalar": scalar_metrics, "scalars": {}}
  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, aux["intermediate_outputs"], config)

  # 7. Return Updated State
  return nnx.state(state), metrics


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


def eval_step_nnx(model, config, data):
  """eval_step no backprop and new state compared with train_step."""

  # DPO Logic: Reference params handling
  # reference_params = None
  extra_dpo_args = []

  loss, aux = loss_fn(model, config, data, None, None, *extra_dpo_args, is_train=False)

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
      data_loader,
      rampup_manager,
      eval_data_iterator,
      train_state,  # Linen: train_state.TrainState; NNX: TrainStateNNX module
  ) = train_utils.setup_train_loop(config, recorder)

  if config.pure_nnx:
    # Split the TrainStateNNX instance into graphdef and state
    static_model, state = nnx.split(train_state)
  else:
    state = train_state
    static_model = model

  if config.use_dpo:
    if config.pure_nnx:
      raise NotImplementedError("DPO is not supprted yet by NNX models.")
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
  _train_step = train_step
  _eval_step = eval_step
  if config.pure_nnx:
    _train_step = train_step_nnx
    _eval_step = eval_step_nnx

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config,
      static_model,
      mesh,
      state,
      state_mesh_shardings,
      _train_step,
      _eval_step,
      eval_data_iterator,
      params_shardings,
  )

  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data:
      # TODO: check the pure_nnx compatibility of this function call
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    maxtext_utils.maybe_dump_jaxpr(config, p_train_step, (state, shaped_batch, init_rng))
    if config.compiled_trainstep_file == "":  # compile only when there is no pre-compiled file loaded
      if config.pure_nnx:
        compiled = p_train_step.lower(state, shaped_batch).compile()
      else:
        compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  if config.pure_nnx:
    _, model_params, _ = nnx.split(state.model, nnx.Param, ...)
  else:
    model_params = state.params
  metric_logger.write_setup_info_to_tensorboard(model_params)

  try:
    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        if not config.pure_nnx:
          # pylint: disable=not-callable
          nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.shard_optimizer_over_data:
              state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
            if config.pure_nnx:
              state, metrics = p_train_step(state, example_batch)
            else:
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
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.pure_nnx:
              eval_metrics = p_eval_step(state, eval_batch)
            else:
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
    if checkpoint_manager is not None:
      # in case the last checkpoint_period checkpoint is still in progress
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
  train_utils.validate_train_config(config)
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
  """Run the job given hyperparameters and utilities.

  In decoupled mode (DECOUPLE_GCLOUD=TRUE) cloud diagnostics may be stubbed; if so, skip wrapping.
  """
  # Use nullcontext when diagnostics are stubbed or in decoupled mode
  diagnostics_context = (
      contextlib.nullcontext()
      if is_decoupled() or getattr(diagnostic, "__class__", None).__name__ == "_StubDiag"
      else diagnostic.diagnose(diagnostic_config)
  )

  if is_decoupled() or getattr(diagnostic, "__class__", None).__name__ == "_StubDiag":
    max_logging.log("[DECOUPLED NO-OP] skipping cloud diagnostics wrapper.")

  with (
      diagnostics_context,
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
