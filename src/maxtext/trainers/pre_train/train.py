# Copyright 2023–2026 Google LLC
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
import contextlib
import datetime
import functools
import os

from absl import app

import numpy as np
import optax

import pathwaysutils  # pylint: disable=unused-import

import tensorflow as tf

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.utils.globals import EPS
from maxtext.utils import elastic_utils
# Placeholder: internal

# pylint: disable=too-many-positional-arguments
from maxtext.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from maxtext.common import checkpointing, profiler
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.common.gcloud_stub import cloud_diagnostics as _cloud_diag, is_decoupled
from maxtext.common.gcloud_stub import vertex_tensorboard_modules
from maxtext.common.metric_logger import MetricLogger, record_activation_metrics
from maxtext.trainers.post_train.dpo.dpo_utils import _merge_dpo_state, _split_dpo_state, dpo_loss_fn
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import qk_clip_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
from maxtext.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from maxtext.utils.vocabulary_tiling import vocab_tiling_linen_loss

_diag_modules = _cloud_diag()
diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration = _diag_modules
VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()


def get_first_step(model, state):
  if isinstance(model, nn.Module):
    return int(state.step)
  return int(state.optimizer.step.get_value())


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
    aux: a dictionary including intermediate_outputs, xent_sum, and total_weights
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

    if (config.use_indexer and not config.indexer_sparse_training) and is_train:
      # In Dense Warm-up stage, we skip main model loss calculation for efficiency.
      # The main model parameters are frozen and only the indexer is trained via KL divergence.
      xent_sum = 0.0
      total_z_loss = 0.0
    elif config.num_vocab_tiling > 1:
      hidden_state_key = ("intermediates", "decoder", "hidden_states")
      hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
      xent_sum, total_z_loss = vocab_tiling_linen_loss(hidden_states, data, config, model, params, is_train)
    else:
      one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
      xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)

      xent = sharding.maybe_shard_with_logical(
          xent,
          ("activation_embed_and_logits_batch", "activation_length"),
          model.mesh,
          config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
      z_loss = sharding.maybe_shard_with_logical(
          z_loss,
          ("activation_embed_and_logits_batch", "activation_length"),
          model.mesh,
          config.shard_mode,
          debug_sharding=config.debug_sharding,
      )

      # Mask out paddings at the end of each example.
      xent = xent * (data["targets_segmentation"] != 0)
      z_loss = z_loss * (data["targets_segmentation"] != 0)

      xent_sum = jnp.sum(xent)
      total_z_loss = jnp.sum(z_loss)
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
    intermediate_outputs = {}

    if (config.use_indexer and not config.indexer_sparse_training) and is_train:
      # In Dense Warm-up stage, we skip main model loss calculation for efficiency.
      # The main model parameters are frozen and only the indexer is trained via KL divergence.
      xent_sum = 0.0
      total_z_loss = 0.0
    else:
      one_hot_targets = jax.nn.one_hot(data["targets"], config.vocab_size)
      xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)

      xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
      z_loss = nn.with_logical_constraint(z_loss, ("activation_embed_and_logits_batch", "activation_length"))

      # Mask out paddings at the end of each example.
      xent = xent * (data["targets_segmentation"] != 0)
      z_loss = z_loss * (data["targets_segmentation"] != 0)

      xent_sum = jnp.sum(xent)
      total_z_loss = jnp.sum(z_loss)

  total_weights = jnp.sum(data["targets_segmentation"] != 0)
  # If gradient accumulation is enabled, we don't need to divide xent_sum
  # by total_weights and then multiply the computed gradient by total_weights,
  # since it's equivalent to computing the gradient from xent_sum.
  # This simplification reduces the number of operations and makes it easier
  # for XLA to move all-reduce out of the gradient accumulation loop when use
  # Zero1+GA to reduce communication overhead.
  # EPS was used to avoid division by zero, but it's not needed when gradient
  # accumulation is enabled since there's no division.
  if config.gradient_accumulation_steps > 1 and not config.use_tunix_gradient_accumulation:
    loss = xent_sum
  else:
    # When using Tunix gradient accumulation, we revert to standard normalization.
    # Unlike the manual accumulation path above, Tunix (via optax.MultiSteps) expects
    # a normalized loss for each step. It handles the accumulation state
    # updates and scaling internally.
    loss = xent_sum / (total_weights + EPS)

  # We keep z-loss normalized by total_weights.
  total_z_loss = total_z_loss / (total_weights + EPS)

  # Calculate and Add MTP Loss
  mtp_loss = 0.0
  if config.mtp_num_layers > 0 and is_train:
    mtp_loss = calculate_mtp_loss(intermediate_outputs, config)
    loss += mtp_loss

  # get indexer loss
  indexer_loss = 0.0
  if config.use_indexer and config.indexer_loss_scaling_factor > 0.0:
    indexer_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "self_attention", "indexer_loss")
    if indexer_losses:
      indexer_loss = jnp.mean(jnp.concatenate(indexer_losses))
      loss += indexer_loss
    else:
      max_logging.debug("No indexer loss found.")

  # get MoE load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    moe_lb_losses = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_lb_loss")
    if moe_lb_losses:
      moe_lb_loss = jnp.mean(jnp.concatenate(moe_lb_losses))
      loss += moe_lb_loss
    else:
      max_logging.debug("\nNo MoE load balance loss found. Defaulting to 0.0.")

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
      "xent_sum": xent_sum,
      "z_loss": total_z_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "indexer_loss": indexer_loss,
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
  if config.parameter_memory_host_offload:
    raw_grads = jax.device_put(
        raw_grads,
        max_utils.with_memory_kind(params_shardings, "device"),
    )
  intermediate_outputs = aux["intermediate_outputs"]
  xent_sum = aux["xent_sum"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux.get("indexer_loss", 0.0)
  z_loss = aux.get("z_loss", 0.0)
  moe_bias_updates = aux.get("moe_bias_updates")
  mtp_loss = aux.get("mtp_loss", 0.0)

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, state, config.gradient_clipping_threshold)
  else:
    grads = raw_grads

  # fp8 fix: sanitize NaN OWG (overwrite-with-gradient) stats before apply_gradients.
  # Under FSDP, the fp8 output gradient amax can be NaN at step 0, which propagates into
  # amax_history and corrupts future steps. Replace NaN OWG entries with the current state
  # values (skip the amax update for that step) instead of letting NaN flow through.
  # Also restore OWG values after apply_gradients to bypass optimizer corruption
  # (Adam should not update fp8 scale/amax_history).
  fp8_stats = dict(grads).get(maxtext_utils.OVERWRITE_WITH_GRADIENT, None)
  if fp8_stats is not None:
    if maxtext_utils.OVERWRITE_WITH_GRADIENT in state.params:
      current_fp8 = state.params[maxtext_utils.OVERWRITE_WITH_GRADIENT]
      fp8_stats = jax.tree_util.tree_map(
          lambda new, cur: jnp.where(jnp.isnan(new), cur, new),
          fp8_stats,
          current_fp8,
      )
    else:
      fp8_stats = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0), fp8_stats)
    grads = dict(grads)
    grads[maxtext_utils.OVERWRITE_WITH_GRADIENT] = fp8_stats
  # Zero out any remaining NaN in float gradients to prevent param corruption
  grads = jax.tree_util.tree_map(
      lambda x: jnp.nan_to_num(x, nan=0.0) if jnp.issubdtype(x.dtype, jnp.floating) else x,
      grads,
  )

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

  if getattr(config, "skip_step_on_spikes", False):
    grad_norm = max_utils.l2norm_pytree(grads)
    # TrainState.apply_gradients doesn't pass **kwargs to tx.update, so we unpack it manually.
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, loss=loss, grad_norm=grad_norm)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
  else:
    new_state = state.apply_gradients(grads=grads)

  # fp8 fix: restore sanitized OWG values, bypassing any optimizer update to fp8 stats.
  if fp8_stats is not None:
    new_params = dict(new_state.params)
    new_params[maxtext_utils.OVERWRITE_WITH_GRADIENT] = fp8_stats
    new_state = new_state.replace(params=new_params)

  # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
  if config.routed_bias and config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
    target_path = ("params", "decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")
    # Flax 'sow' returns a tuple, so we take the first element [0].
    # Updates the shape to be aligned with state.
    moe_bias_updates = jnp.array(moe_bias_updates[0]).transpose()
    new_state = maxtext_utils.update_state_param(new_state, target_path, moe_bias_updates)

  lm_loss = xent_sum / (total_weights + EPS)
  scalar_metrics = {
      "learning/loss": loss,
      "learning/lm_loss": lm_loss,
      "learning/perplexity": jnp.exp(lm_loss),
      "learning/z_loss": z_loss,
      "learning/moe_lb_loss": moe_lb_loss,
      "learning/indexer_loss": indexer_loss,
      "learning/mtp_loss": mtp_loss,
      "learning/total_weights": total_weights,
  }
  if config.use_qk_clip:
    # Apply QK-Clip
    new_state = qk_clip_utils.apply_qk_clip(new_state, intermediate_outputs, config)

    # Report max_logits metric
    global_max_logit = qk_clip_utils.calculate_max_logit_metric(intermediate_outputs)
    if global_max_logit is not None:
      scalar_metrics["learning/max_logits"] = global_max_logit

  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)
  if config.use_dpo:
    scalar_metrics["learning/dpo_loss"] = aux["dpo_loss"]
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

  xent_sum = aux["xent_sum"]
  z_loss = aux.get("z_loss", 0.0)
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux.get("indexer_loss", 0.0)
  mtp_loss = aux.get("mtp_loss", 0.0)
  # For DPO, report the unnormalized sum of per-sample preference losses so that
  # MetricLogger (which divides eval/total_loss by eval/total_weights) recovers
  # the correct mean DPO loss. xent_sum is always 0 for DPO and must not be used.
  eval_total_loss = aux["dpo_loss"] * total_weights if config.use_dpo else xent_sum
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/z_loss": z_loss,
          "evaluation/total_loss": eval_total_loss,
          "evaluation/total_weights": total_weights,
          "evaluation/moe_lb_loss": moe_lb_loss,
          "evaluation/indexer_loss": indexer_loss,
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
      state,
  ) = train_utils.setup_train_loop(config, recorder)

  if config.use_dpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_dpo_state(state, reference_params)
    state_mesh_shardings = _merge_dpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  with jax.set_mesh(mesh), mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
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
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    if config.shard_optimizer_over_data:
      state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
    maxtext_utils.maybe_dump_jaxpr(config, p_train_step, (state, shaped_batch, init_rng))
    if config.compiled_trainstep_file == "":  # compile only when there is no pre-compiled file loaded
      compiled = p_train_step.lower(state, shaped_batch, init_rng).compile()
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats)

  start_step = get_first_step(model, state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params)

  _job_completed_gracefully = False
  try:
    last_step_completion = datetime.datetime.now()
    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        # pylint: disable=not-callable
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            if config.shard_optimizer_over_data:
              state = sharding.maybe_shard_with_name(state, state_mesh_shardings, config.shard_mode)
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
    _job_completed_gracefully = True
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
    _job_completed_gracefully = True
  finally:
    if _job_completed_gracefully:
      record_goodput(recorder, RECORD_JOB_END_TIME)
    metric_logger.flush_metrics_and_cleanup()

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any, Any]:
  """Initialization of hyperparameters and utilities"""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
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
  jax.config.update("jax_remove_size_one_mesh_axis_from_type", config.remove_size_one_mesh_axis_from_type)
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
      max_utils.maybe_get_transformer_engine_context(config),
  ):
    train_loop(config, recorder)


def get_train_func(config, recorder, diagnostic_config, argv):
  """Returns the train function, wrapping in elastic_retry if elastic training is enabled."""
  if config.elastic_enabled:
    max_logging.log("Elastic utils: Elastic training enabled.")

    def elastic_train_wrapper(argv: Sequence[str]) -> None:
      """Wrapper for elastic training initializes variables and runs the train loop."""
      elastic_config, elastic_recorder, elastic_diagnostic_config = initialize(argv)
      run(
          elastic_config,
          elastic_recorder,
          elastic_diagnostic_config,
      )

    train_func = elastic_utils.elastic_retry(config)(functools.partial(elastic_train_wrapper, argv=argv))
  else:
    # Use the already initialized variables
    def train_func():
      run(config, recorder, diagnostic_config)

  return train_func


def main(argv: Sequence[str]) -> None:
  config, recorder, diagnostic_config = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, diagnostic_config, argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  app.run(main)
