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

from typing import Any, Sequence, TypedDict
import datetime
import functools
import gc
import os
import sys
import time
import logging

from absl import app

import numpy as np
import optax

import pathwaysutils  # pylint: disable=unused-import
import threading
from pathwaysutils.elastic import manager as pathways_manager
from pathwaysutils.elastic import elastic
from pathwaysutils.debug import watchdog
from orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter import Snapshotter

_logger = logging.getLogger(__name__)
logging.getLogger("pathwaysutils.debug.watchdog").setLevel(logging.DEBUG)

try:
  import tensorflow as tf

  _TF_AVAILABLE = True
except ImportError:
  _TF_AVAILABLE = False

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from flax import nnx, traverse_util
from flax.core.spmd import logical_axis_rules
from flax.nnx import variablelib

from maxtext.input_pipeline import input_pipeline_interface
from maxtext.common.data_loader import create_dataloader
from maxtext.common.common_types import ReorderStrategy

from maxtext.configs import pyconfig
from maxtext.configs.types import TeCommGemmOverlapPolicy
from maxtext.diffusion.block_diffusion import target_alignment as block_diffusion_target_alignment
from maxtext.utils.globals import EPS
from maxtext.utils import elastic_utils

# Placeholder: internal

# pylint: disable=too-many-positional-arguments
from maxtext.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss, mtp_acceptance, mtp_losses
from maxtext.layers.attention_mla import indexer_losses
from maxtext.common import checkpointing, profiler, train_state_nnx
from maxtext.common.goodput import (
    GoodputEvent,
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.common.gcloud_stub import vertex_tensorboard_modules
from maxtext.common import metric_logger
from maxtext.common.metric_logger import record_activation_metrics
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import qk_clip_utils
from maxtext.utils import sharding
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import train_utils
from maxtext.utils import mllog_utils
from maxtext.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from maxtext.utils.vocabulary_tiling import vocab_tiling_nnx_loss


class EncoderKwargs(TypedDict, total=False):
  """Multimodal encoder arguments forwarded to the model."""

  encoder_images: Any
  encoder_image_masks: Any
  encoder_videos: Any
  encoder_video_masks: Any
  encoder_video_grid_thw: Any


VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()


def get_first_step(model, state):
  del model  # NNX-only; kept for call-site signature parity
  if hasattr(state, "inner_state"):  # DiLoCoTrainState (NNX DiLoCo)
    step_val = state.step.get_value() if hasattr(state.step, "get_value") else state.step
    return int(step_val)
  return int(state.optimizer.step.get_value())


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: An NNX model.
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: Unused for NNX (kept for signature parity).
    params: Unused for NNX; params are part of the model.
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, xent_sum, and total_weights
  """
  del dropout_rng, params, sparsity_state  # unused for NNX (kept for signature parity)
  is_block_diffusion = getattr(config, "training_objective", "causal_lm") == "block_diffusion"
  if getattr(config, "attention_type", "global") == "block_diffusion" and not is_block_diffusion:
    raise ValueError(
        "Block-diffusion attention requires target-aligned block-diffusion losses; "
        "causal next-token labels would leak within a bidirectional block."
    )
  if is_block_diffusion:
    required_masks = {"corruption_mask", "targets_loss_mask"}
    missing_masks = required_masks - data.keys()
    if missing_masks:
      raise ValueError(f"Block-diffusion loss requires explicit batch masks; missing {sorted(missing_masks)}")
    target_shape = data["targets"].shape
    for mask_name in required_masks:
      if data[mask_name].shape != target_shape:
        raise ValueError(f"{mask_name} must match targets shape; got {data[mask_name].shape} and {target_shape}")

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_train_on, :]
  else:
    for k, v in data.items():
      data[k] = v[: config.micro_batch_size_to_eval_on, :]
  # Only forward the kwarg when router replay is actually in use, so models
  # and adapters whose __call__ predates the feature keep working.
  forced_routing_kwargs = (
      {"forced_routed_experts": data["forced_routed_experts"]} if "forced_routed_experts" in data else {}
  )

  if is_block_diffusion:
    targets_loss_mask = (data["targets_loss_mask"] != 0) & (data["targets_segmentation"] != 0)
    target_positions = data.get("targets_position", data["inputs_position"])
  else:
    targets_loss_mask = None
    target_positions = None
  # A multimodal model may receive a text-only batch while retaining its vision
  # parameters in the model and checkpoints. Only pass image inputs when present.
  encoder_images = data.get("images") if config.use_multimodal else None
  encoder_image_masks = data.get("image_masks") if config.use_multimodal else None
  is_video = "video_grid_thw" in data
  encoder_kwargs: EncoderKwargs
  if is_video:
    encoder_kwargs = {
        "encoder_videos": encoder_images,
        "encoder_video_masks": encoder_image_masks,
        "encoder_video_grid_thw": data.get("video_grid_thw"),
    }
  else:
    encoder_kwargs = {
        "encoder_images": encoder_images,
        "encoder_image_masks": encoder_image_masks,
    }
  # Flax NNX model: forward pass, then pop Intermediates sown during it.
  logits = model(
      decoder_input_tokens=data["inputs"],
      decoder_positions=data["inputs_position"],
      decoder_segment_ids=data["inputs_segmentation"],
      **encoder_kwargs,
      enable_dropout=config.enable_dropout if is_train else False,
      decoder_target_tokens=data["targets"],
      decoder_target_mask=data["targets_segmentation"],
      **forced_routing_kwargs,
  )
  # mtp_losses and mtp_acceptance subclass nnx.Intermediate, and nnx type filters match
  # subclasses. Pop them before the generic Intermediate pop below, which would otherwise
  # take them too and leave the MTP loss silently reading as 0.
  mtp_losses_state, mtp_acceptance_state = None, None
  if config.mtp_num_layers > 0:
    mtp_losses_state = nnx.pop(model, mtp_losses)
    mtp_acceptance_state = nnx.pop(model, mtp_acceptance)

  indexer_losses_state = None
  if config.use_indexer:
    # Pop dedicated indexer_losses to harvest auxiliary KL loss and prevent model state PyTree mismatches.
    indexer_losses_state = nnx.pop(model, indexer_losses)

  intermediates = nnx.pop(model, nnx.Intermediate)
  intermediate_outputs = intermediates.to_pure_dict()

  # Store them under the collection name so calculate_mtp_loss and
  # calculate_mtp_acceptance_rate find them at the path they expect.
  if mtp_losses_state is not None and mtp_acceptance_state is not None:
    intermediate_outputs["mtp_losses"] = mtp_losses_state.to_pure_dict()
    intermediate_outputs["mtp_acceptance"] = mtp_acceptance_state.to_pure_dict()

  if indexer_losses_state is not None:
    intermediate_outputs["indexer_losses"] = indexer_losses_state.to_pure_dict()

  if (config.use_indexer and not config.indexer_sparse_training) and is_train:
    # In Dense Warm-up stage, we skip main model loss calculation for efficiency.
    # The main model parameters are frozen and only the indexer is trained via KL divergence.
    xent_sum = 0.0
    total_z_loss = 0.0
  elif config.num_vocab_tiling > 1:
    hidden_state_key = ("decoder", "hidden_states")
    hidden_states = maxtext_utils.get_nested_value(intermediate_outputs, hidden_state_key)[0]
    xent_sum, total_z_loss = vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train)
  else:
    if is_block_diffusion:
      logits = block_diffusion_target_alignment.align_logits_to_targets(
          logits,
          config.block_diffusion_logit_alignment,
          target_positions,
          data["targets_segmentation"] != 0,
      )
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

    if is_block_diffusion:
      xent = xent * targets_loss_mask
      z_loss = z_loss * targets_loss_mask
    else:
      xent = xent * (data["targets_segmentation"] != 0)
      z_loss = z_loss * (data["targets_segmentation"] != 0)

    xent_sum = jnp.sum(xent)
    total_z_loss = jnp.sum(z_loss)

  if is_block_diffusion:
    assert targets_loss_mask is not None
    total_weights = jnp.sum(targets_loss_mask)
  else:
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

  # Calculate and add auxiliary Indexer loss
  indexer_loss = 0.0
  if config.use_indexer and config.indexer_loss_scaling_factor > 0.0:
    # Recursively collect per-layer indexer losses across all scanned transformer layers.
    indexer_losses_list = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "indexer_loss")
    if indexer_losses_list:
      indexer_loss = jnp.mean(jnp.concatenate([jnp.atleast_1d(x) for x in indexer_losses_list]))
      loss += indexer_loss  # Injects loss into scalar objective to drive backward gradients for indexer weights.
    else:
      max_logging.debug("No Indexer loss found. Defaulting to 0.0.")

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
  mtp_moe_bias_updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    # NNX intermediates are model-rooted (no "intermediates" prefix),
    # so match by suffix instead. Unlike collect_intermediates_by_suffix
    # we must not ravel: the decoder update is a 2-D matrix that's
    # transposed and MTP update is 1-D matrix.
    for path, val in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
      keys = tuple(k.key for k in path if hasattr(k, "key"))
      if not keys or keys[-1] != "moe_bias_updates":
        continue
      if "decoder" in keys:
        moe_bias_updates = (val,)
      elif "mtp_block" in keys:
        if mtp_moe_bias_updates is None:
          mtp_moe_bias_updates = []
        mtp_moe_bias_updates.append(val)

  # Add the model's primary output to the intermediates dict so it can be used
  # by the acceptance rate calculation in eval_step.
  if not is_train and config.mtp_eval_target_module > 0:
    intermediate_outputs["logits"] = logits

  aux = {
      "intermediate_outputs": intermediate_outputs,
      "xent_sum": xent_sum,
      "z_loss": total_z_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "indexer_loss": indexer_loss,
      "moe_bias_updates": moe_bias_updates,
      "mtp_moe_bias_updates": mtp_moe_bias_updates,
      "mtp_loss": mtp_loss,
      "batch_stats": (intermediate_outputs.get("batch_stats", None) if hasattr(intermediate_outputs, "get") else None),
  }
  te_moe_block = getattr(config, "te_moe_block", False)
  if te_moe_block:
    overflow_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_capacity_overflow")
    total_recv_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_total_recv_tokens")
    capacity_values = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "te_moe_recv_capacity_per_rank")
    if not overflow_values or not total_recv_values or not capacity_values:
      raise ValueError("te_moe_block=True did not produce TE MoE receive-capacity intermediates.")

    aux.update(
        {
            "te_moe_capacity_overflow": jnp.any(jnp.concatenate(overflow_values)),
            "te_moe_max_total_recv_tokens": jnp.max(jnp.concatenate(total_recv_values)),
            "te_moe_recv_capacity_per_rank": jnp.min(jnp.concatenate(capacity_values)),
        }
    )
  has_moe_overflow = jnp.bool_(False)
  if config.retry_when_tokens_dropped:
    moe_overflow_flags = maxtext_utils.collect_intermediates_by_suffix(intermediate_outputs, "moe_has_overflow")
    if moe_overflow_flags:
      has_moe_overflow = jnp.any(jnp.stack([jnp.any(x) for x in moe_overflow_flags]))
  aux["has_moe_overflow"] = has_moe_overflow
  return loss, aux


def _find_gate_bias(module: nnx.Module | None) -> nnx.Variable | None:
  """Finds the router gate bias parameter in a module graph."""
  if module is None:
    return None
  for _, node in nnx.iter_graph(module):
    if type(node).__name__ == "GateLogit" and hasattr(node, "bias") and node.bias is not None:
      return node.bias
  return None


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None):
  """Training step for the NNX model.

  Args:
    model: An nnx.GraphDef of the TrainStateNNX.
    config: Hyperparameters.
    state_mesh_shardings: PyTree of PartitionSpecs for the train state.
    params_shardings: PyTree of PartitionSpecs for model parameters, used for gradient accumulation.
    state: NNX pure State.
    data: Training data batch.
    dropout_rng: Unused for NNX (kept for jit signature parity).

  Returns:
    new_state: Updated NNX pure State.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
  """
  del dropout_rng  # unused for NNX (kept for jit signature parity)
  # pylint: disable=too-many-nested-blocks
  # --- Per-path initialization ---
  state = nnx.merge(model, state)  # reconstruct TrainStateNNX
  loss_model, loss_params, loss_rng = state.model, None, None

  # --- Gradient computation ---
  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        loss_fn,
        config,
        loss_model,
        loss_params,
        params_shardings,
        data,
        loss_rng,
    )
  else:
    owg_type = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
    custom_param_filter = nnx.Any(owg_type)
    train_param_type = (
        getattr(nnx, "LoRAParam", nnx.Param)
        if getattr(getattr(config, "lora", None), "enable_lora", False)
        else nnx.Param
    )
    nnx.pop(state.model, nnx.Intermediate)
    model_graphdef, curr_params, custom_params, rest = nnx.split(state.model, train_param_type, custom_param_filter, ...)
    if config.parameter_memory_host_offload:
      # Params are kept on host (pinned_host) in in_shardings. Move only Param
      # variables to device before the forward/backward pass so that all dot_general
      # operands share the same memory space (XLA on GPU requires this).
      # Using params_shardings (Param-only) avoids Shardy rank mismatches that
      # occur when applying PartitionSpec() (rank-0 in SDY) to rank-1 RNG key tensors.
      device_param_shardings = jax.tree_util.tree_map_with_path(
          maxtext_utils_nnx.move_memory_to_device,
          params_shardings,
          is_leaf=lambda x: isinstance(x, NamedSharding),
      )
      curr_params = jax.device_put(curr_params, device_param_shardings)
      nnx.update(state.model, curr_params)  # ensure state.model has device params for optimizer update
    if config.shard_optimizer_over_data:
      param_sharding_lookup = {}
      for p, s in jax.tree_util.tree_leaves_with_path(
          params_shardings, is_leaf=lambda x: isinstance(x, (nnx.Variable, NamedSharding, jax.sharding.Sharding))
      ):
        param_sharding_lookup[p] = s.get_value() if isinstance(s, nnx.Variable) else s

      def _maybe_shard_param(path, var):
        if path in param_sharding_lookup:
          return sharding.maybe_shard_with_name(var, param_sharding_lookup[path], shard_mode=config.shard_mode)
        return var

      curr_params = jax.tree_util.tree_map_with_path(
          _maybe_shard_param,
          curr_params,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      nnx.update(state.model, curr_params)

    def diff_wrapper(curr_params, custom_params, rest, config, data):
      local_model = nnx.merge(model_graphdef, curr_params, custom_params, rest, copy=True)
      loss, aux = loss_fn(local_model, config, data, None, None, is_train=True)
      # Exclude parameters, custom-gradient state, and intermediates. Custom-
      # gradient state is updated separately, so stale values must not overwrite
      # `custom_grads`.
      non_param_rest = nnx.state(
          local_model,
          nnx.Not(nnx.Any(nnx.Param, custom_param_filter, nnx.Intermediate)),
      )
      return loss, (aux, non_param_rest)

    grad_func = jax.value_and_grad(diff_wrapper, argnums=(0, 1), has_aux=True)
    (loss, (aux, non_param_rest)), (raw_grads, custom_grads) = grad_func(curr_params, custom_params, rest, config, data)
    nnx.update(state.model, nnx.State.merge(custom_grads, non_param_rest))

  raw_grads = jax.tree_util.tree_map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )
  if config.parameter_memory_host_offload:
    raw_grads = jax.device_put(
        raw_grads,
        max_utils.with_memory_kind(params_shardings, "device"),
    )

  # Extract aux fields into locals
  intermediate_outputs = aux["intermediate_outputs"]
  xent_sum = aux["xent_sum"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  has_moe_overflow = aux.get("has_moe_overflow")
  indexer_loss = aux.get("indexer_loss", 0.0)
  z_loss = aux.get("z_loss", 0.0)
  moe_bias_updates = aux.get("moe_bias_updates")
  mtp_moe_bias_updates = aux.get("mtp_moe_bias_updates")
  mtp_loss = aux.get("mtp_loss", 0.0)
  new_opt_state = None
  bias_metrics = {}

  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  if config.optimizer_memory_host_offload:
    # state.optimizer is an NNX Optimizer module; state_mesh_shardings.optimizer
    # is an NNX State. Use nnx.state() to get a compatible State for device_put.
    device_opt_shardings = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_device,
        state_mesh_shardings.optimizer,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )
    opt_state = nnx.state(state.optimizer)
    new_opt_state = jax.device_put(opt_state, device_opt_shardings)
    nnx.update(state.optimizer, new_opt_state)
  if config.skip_step_on_spikes:
    # The skip-step optimizer is a GradientTransformationExtraArgs that reads
    # loss/grad_norm to decide whether to zero the update on a spike. nnx
    # Optimizer.update forwards these kwargs to tx.update.
    grad_norm = max_utils.l2norm_pytree(grads)
    state.apply_gradients(grads, loss=loss, grad_norm=grad_norm)
  else:
    state.apply_gradients(grads)
  new_state = state

  # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
  # pylint: disable=too-many-nested-blocks
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    if getattr(config, "model_name", "").startswith("deepseek4"):
      max_logging.log("DeepSeek V4: Applying auxiliary-loss-free routing bias via pure NNX MoEBiasVar.")
      flat_intermediates = traverse_util.flatten_dict(aux.get("intermediate_outputs", {}))
      for path, update in flat_intermediates.items():
        if path[-1] != "moe_bias_updates":
          continue
        target = new_state.model
        prefix = path[1:-1] if path[0] == "intermediates" else path[:-1]
        for key in prefix:
          if hasattr(target, key):
            target = getattr(target, key)
          elif isinstance(target, dict) and key in target:
            target = target[key]
          else:
            target = None
            break
        if target is None:
          continue
        for _, node in nnx.iter_graph(target):
          if type(node).__name__ == "GateLogit" and hasattr(node, "bias") and node.bias is not None:
            update_val = update[0] if isinstance(update, (tuple, list)) else update
            name_prefix = "-".join(map(str, prefix))
            if getattr(config, "log_moe_bias_norms", False):
              bias_metrics[f"learning/moe_bias_before_norm_{name_prefix}"] = jnp.linalg.norm(node.bias.value)
            node.bias.value = node.bias.value + jnp.array(update_val)
            if getattr(config, "log_moe_bias_norms", False):
              bias_metrics[f"learning/moe_bias_update_norm_{name_prefix}"] = jnp.linalg.norm(jnp.array(update_val))
    else:
      # 1. Update main decoder scanned MoE layers.
      # The update from the scan is (num_moe_layers, num_experts) and must be transposed.
      decoder_layer = getattr(new_state.model.decoder, "moe_layers", new_state.model.decoder)
      decoder_bias = _find_gate_bias(decoder_layer)
      if decoder_bias is not None and moe_bias_updates is not None:
        decoder_bias.value = decoder_bias.value + jnp.array(moe_bias_updates[0])

      # 2. Update auxiliary MTP MoE layers (if enabled).
      # Unlike the main decoder, each MTP layer is an individual un-scanned layer
      # with a 1D bias of shape (num_experts,).
      if mtp_moe_bias_updates is not None and hasattr(new_state.model, "mtp_block"):
        for i, update in enumerate(mtp_moe_bias_updates):
          mtp_layer = getattr(new_state.model.mtp_block, f"mtp_layer_{i + 1}", None)
          mtp_bias = _find_gate_bias(mtp_layer)
          if mtp_bias is not None:
            mtp_bias.value = mtp_bias.value + jnp.array(update)

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
  if getattr(config, "te_moe_block", False):
    scalar_metrics.update(
        {
            "learning/te_moe_capacity_overflow": aux["te_moe_capacity_overflow"].astype(jnp.int32),
            "learning/te_moe_max_total_recv_tokens": aux["te_moe_max_total_recv_tokens"],
            "learning/te_moe_recv_capacity_per_rank": aux["te_moe_recv_capacity_per_rank"],
        }
    )
  scalar_metrics.update(bias_metrics)
  if config.use_qk_clip:
    new_state = qk_clip_utils.apply_qk_clip_nnx(new_state, intermediate_outputs, config)

    global_max_logit = qk_clip_utils.calculate_max_logit_metric(intermediate_outputs)
    if global_max_logit is not None:
      scalar_metrics["learning/max_logits"] = global_max_logit

  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    model_params = nnx.state(new_state.model, nnx.Param)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(model_params)

  # Surface skip-step rejections as a TB metric. The skip-step optimizer stores
  # is_skipped in its opt_state; read it back off the optimizer just updated in place.
  if config.skip_step_on_spikes:
    opt_state = nnx.to_pure_dict(nnx.state(new_state.optimizer)).get("opt_state", {})
    is_skipped = opt_state.get("is_skipped") if isinstance(opt_state, dict) else None
    if is_skipped is not None:
      scalar_metrics["optim/step_skipped"] = is_skipped.astype(jnp.float32)
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }
  if config.retry_when_tokens_dropped:
    metrics["has_moe_overflow"] = (  # pyrefly: ignore[bad-assignment]
        has_moe_overflow if has_moe_overflow is not None else jnp.bool_(False)  # pyrefly: ignore[bad-assignment]
    )
  if getattr(config, "record_internal_nn_metrics", False):
    record_activation_metrics(metrics, intermediate_outputs, config)

  # Drop Intermediates (e.g. sowed max_logits for QK-Clip) and the MTP sown
  # vars (mtp_losses/mtp_acceptance) before returning. They're absent from
  # state_mesh_shardings and would cause a leaf-count / structure mismatch.
  return nnx.state(new_state, nnx.Not(nnx.Intermediate)), metrics


def eval_step(model, config, state, data, dropout_rng=None):
  """eval_step no backprop and new state compared with train_step."""
  del dropout_rng  # unused for NNX (kept for jit signature parity)
  state = nnx.merge(model, state)  # reconstruct TrainStateNNX
  loss, aux = loss_fn(state.model, config, data, None, None, is_train=False)

  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)

  xent_sum = aux["xent_sum"]
  z_loss = aux.get("z_loss", 0.0)
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux.get("indexer_loss", 0.0)
  mtp_loss = aux.get("mtp_loss", 0.0)
  eval_total_loss = xent_sum
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
  if config.retry_when_tokens_dropped:
    metrics["has_moe_overflow"] = aux.get("has_moe_overflow", False)

  return metrics


def recreate_dataloaders(config, mesh, recorder, rampup_manager):
  """Recreates data and eval iterators and dataloader for the new mesh."""
  new_data_iter, new_eval_iter = input_pipeline_interface.create_data_iterator(config, mesh)
  context_parallel_size = mesh.shape.get(config.context_sharding, 1)
  with jax.set_mesh(mesh):
    if context_parallel_size > 1 and config.context_parallel_load_balance:
      reorder_strategy = (
          ReorderStrategy.STRIPED
          if config.packing and config.context_parallel_strategy.lower() == "ring"
          else ReorderStrategy.DUAL_CHUNK_SWAP
      )
      reorder_strategy = (
          config.context_parallel_reorder_strategy
          if config.context_parallel_reorder_strategy != ReorderStrategy.AUTO
          else reorder_strategy
      )
      reorder_fn = maxtext_utils.get_reorder_callable(
          context_parallel_size, config.shard_mode, reorder_strategy, config.hardware
      )
      new_data_iter = map(reorder_fn, new_data_iter)
      if new_eval_iter:
        new_eval_iter = map(reorder_fn, new_eval_iter)

  new_data_loader = create_dataloader(config, mesh, new_data_iter, recorder, rampup_manager)
  return new_data_loader, new_data_iter, new_eval_iter


def save_snapshot(snapshot_mgr, state, step):
  """Saves a host memory snapshot of the current model and optimizer state."""
  model_state = nnx.state(state.model)
  opt_state = nnx.state(state.optimizer)
  state_dict = {
      "model": nnx.to_pure_dict(model_state),
      "optimizer": nnx.to_pure_dict(opt_state),
  }
  state_dict = train_utils.replicate_single_device_sharded_arrays(state_dict)
  _logger.info("Saving in-memory snapshot at step %d...", step)
  snapshot_mgr.save(step, state_dict)


def training_loop_iteration(
    jax_device_state: dict[str, Any],
    python_vars: dict[str, Any],
    immutable_data: dict[str, Any],
):
  """Executes a single iteration of the training loop."""
  # Unpack jax_device_state
  state = jax_device_state["state"]
  init_rng = jax_device_state["init_rng"]
  mesh = jax_device_state["mesh"]
  p_train_step = jax_device_state["p_train_step"]
  p_train_step_dropless = jax_device_state.get("p_train_step_dropless", None)
  p_eval_step = jax_device_state["p_eval_step"]
  p_eval_step_dropless = jax_device_state.get("p_eval_step_dropless", None)

  # Unpack python_vars
  step = python_vars["step"]
  last_step_completion = python_vars["last_step_completion"]
  data_loader = python_vars["data_loader"]
  rampup_manager = python_vars["rampup_manager"]
  recorder = python_vars["recorder"]
  checkpoint_manager = python_vars["checkpoint_manager"]
  snapshot_mgr = python_vars["snapshot"]
  data_iterator = python_vars["data_iterator"]
  eval_data_iterator = python_vars["eval_data_iterator"]
  metric_logger_instance = python_vars["metric_logger_instance"]
  prof = python_vars["prof"]

  # Unpack immutable_data
  config = immutable_data["config"]  # for helpers
  logical_axis_rules_for_train = immutable_data["logical_axis_rules_for_train"]
  logical_axis_rules_for_eval = immutable_data["logical_axis_rules_for_eval"]
  eval_interval = immutable_data["eval_interval"]
  eval_steps = immutable_data["eval_steps"]
  start_step = immutable_data["start_step"]
  eval_start_step = immutable_data["eval_start_step"]

  # HLO dump config
  dump_hlo = immutable_data["dump_hlo"]
  dump_step = immutable_data["dump_step"]
  dump_hlo_local_dir = immutable_data["dump_hlo_local_dir"]
  dump_hlo_gcs_dir = immutable_data["dump_hlo_gcs_dir"]
  dump_hlo_module_name = immutable_data["dump_hlo_module_name"]
  dump_hlo_delete_local_after = immutable_data["dump_hlo_delete_local_after"]
  dump_hlo_upload_all = immutable_data["dump_hlo_upload_all"]

  prof.maybe_activate_profiler(step, state)

  with jax.profiler.StepTraceAnnotation("train", step_num=step):
    example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
    # DiLoCo's inner step takes the rng like the inner NNX step.
    if config.enable_diloco:
      # pylint: disable=not-callable
      step_rng_args = (jax.jit(jax.random.fold_in)(init_rng, step),)
    else:
      step_rng_args = ()
    with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
      with jax.set_mesh(mesh), logical_axis_rules(logical_axis_rules_for_train):
        if config.retry_when_tokens_dropped and p_train_step_dropless is not None:
          candidate_state, metrics = p_train_step(state, example_batch, *step_rng_args)
          if bool(metrics.get("has_moe_overflow")):
            max_logging.log(
                f"Step {step}: MoE ragged buffer overflow detected! "
                f"Discarding candidate state and replaying step with dropless buffer..."
            )
            # Explicitly deallocate device buffers held by candidate_state before replaying
            # with p_train_step_dropless to avoid pinning two ~13.4 GB model states in HBM.
            jax.tree_util.tree_map(
                lambda x: x.delete() if hasattr(x, "delete") else None,
                candidate_state,
            )
            del candidate_state
            gc.collect()
            state, metrics = p_train_step_dropless(state, example_batch, *step_rng_args)
          else:
            state = candidate_state
        else:
          state, metrics = p_train_step(state, example_batch, *step_rng_args)

  step_time_delta = datetime.datetime.now() - last_step_completion
  last_step_completion = datetime.datetime.now()

  completed_step = step + 1

  checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step)

  if dump_hlo and step == (dump_step if dump_step >= 0 else start_step):
    jax.block_until_ready(state)  # Ensure compilation has finished.
    gcs_utils.upload_dump(
        dump_hlo_local_dir,
        dump_hlo_gcs_dir,
        module_name=dump_hlo_module_name,
        delete_local_after=dump_hlo_delete_local_after,
        all_host_upload=dump_hlo_upload_all,
    )

  ran_eval = (
      eval_interval > 0
      and step >= start_step
      and step >= eval_start_step
      and (step - eval_start_step) % eval_interval == 0
  )
  if ran_eval:
    assert eval_data_iterator
    # Explicitly reset the eval iterator and counters before starting the eval loop
    if hasattr(eval_data_iterator, "reset"):
      eval_data_iterator.reset()
    metric_logger_instance.reset_eval_metrics()
    mllog_utils.eval_start(config, completed_step, start_step=start_step)
    max_logging.log(f"Starting eval after train step {step}")

    eval_step_count = 0
    last_eval_step_completion = datetime.datetime.now()
    metric_logger_instance.mark_eval_loop_start()
    # pylint: disable=not-callable
    for eval_batch in eval_data_iterator:
      # Shard input eval data
      eval_batch = jax.device_put(
          eval_batch, sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
      )
      with jax.set_mesh(mesh), logical_axis_rules(logical_axis_rules_for_eval):
        eval_metrics = p_eval_step(state, eval_batch, *step_rng_args)
        if (
            config.retry_when_tokens_dropped
            and p_eval_step_dropless is not None
            and bool(eval_metrics.get("has_moe_overflow"))
        ):
          max_logging.log(
              f"Eval step {eval_step_count}: MoE ragged buffer overflow detected! "
              f"Replaying eval step with dropless buffer..."
          )
          eval_metrics = p_eval_step_dropless(state, eval_batch, *step_rng_args)
      eval_step_time_delta = datetime.datetime.now() - last_eval_step_completion
      last_eval_step_completion = datetime.datetime.now()
      metric_logger_instance.buffer_and_write_metrics(
          eval_metrics, eval_step_count, step_time_delta=eval_step_time_delta, is_training=False
      )
      eval_step_count += 1
      # Stop before fetching another batch: the extra fetch would reshard a batch that
      # is never used, and the data loading hosts may already be out of data while the
      # placeholder iterators of the other hosts keep going.
      if 0 < eval_steps <= eval_step_count:
        break

  prof.maybe_deactivate_profiler(step, state)

  if step == start_step:
    max_utils.print_mem_stats("After params initialized")

  metric_logger_instance.buffer_and_write_metrics(metrics, step, step_time_delta)
  if not ran_eval:
    mllog_utils.step_end(config, completed_step)

  # Async Host Backup (Elastic Mode only)
  # Skip in-memory snapshot if persistent checkpointing is actively saving in the background
  # to prevent PCIe DMA, host RAM, and colocated sidecar thread contention.
  is_checkpoint_saving = (
      checkpoint_manager is not None
      and hasattr(checkpoint_manager, "is_saving_in_progress")
      and checkpoint_manager.is_saving_in_progress()
  )
  if snapshot_mgr is not None and step % config.elastic_snapshot_interval == 0:
    if not is_checkpoint_saving:
      save_snapshot(snapshot_mgr, state, step)
    else:
      _logger.info("Skipping in-memory snapshot at step %d because persistent checkpoint save is in progress.", step)

  # Pack mutated state back to dicts
  jax_device_state["state"] = state
  python_vars["last_step_completion"] = last_step_completion
  return metrics


def recover(
    jax_device_state: dict[str, Any],
    python_vars: dict[str, Any],
    immutable_data: dict[str, Any],
    active_state: Any = None,
):
  """Rebuilds MaxText JAX device state and restores state from host snapshot or active state."""
  config = immutable_data["config"]
  recorder = python_vars["recorder"]
  elastic_manager = python_vars["elastic_manager"]
  snapshot_mgr = python_vars["snapshot"]
  rampup_manager = python_vars["rampup_manager"]
  metric_logger_instance = python_vars["metric_logger_instance"]

  # Clear poisoned state to allow garbage collection
  jax_device_state["state"] = None
  jax_device_state["p_train_step"] = None
  jax_device_state["p_eval_step"] = None
  jax_device_state["model"] = None

  if metric_logger_instance is not None:
    metric_logger_instance.buffered_metrics.clear()

  # Delete old iterators and loaders to release colocated python resources
  for key in ["data_iterator", "eval_data_iterator", "data_loader"]:
    if key in python_vars:
      _logger.info("Deleting old %s to release colocated python resources...", key)
      del python_vars[key]

  while True:
    try:
      # 1. Find currently active slices (wait if none are active)
      min_slices = config.elastic_min_slice_count if config.elastic_min_slice_count > 0 else config.num_slices

      all_active_slices = elastic.wait_for_slices(
          slice_count=min_slices,
          poll_interval=1,
          slice_to_devices=elastic_manager.slice_to_devices,
          timeout=config.elastic_timeout_seconds,
      )
      elastic_manager.active_slice_indices = all_active_slices
      jax.config.update("jax_default_device", elastic_manager.default_device)
      # Slice topology for this recovery attempt is now known: close out the
      # "wait" badput window and open "reinit", logging the post-wait slice counts.
      elastic_utils.record_elastic_wait_end_and_reinit_start(recorder)
      _logger.info(
          "Active slices after recovery: %s",
          elastic_manager.active_slice_indices,
      )
      _logger.info(
          "Active devices after recovery: %d",
          len(elastic_utils.live_devices(config)),
      )

      # Dynamically mutate the config to match the new slice topology
      elastic_utils.mutate_config_for_topology(config, elastic_manager)
      # Immediately cancel any in-flight background checkpoint saving operations
      existing_checkpoint_manager = python_vars["checkpoint_manager"]
      if existing_checkpoint_manager is not None:
        checkpointing.cancel_checkpoint_manager(existing_checkpoint_manager)

      # Reset snapshotter to abandon in-flight host saves while preserving the latest snapshot
      if snapshot_mgr is not None:
        new_snapshot_mgr = Snapshotter(replica_axis_index=snapshot_mgr.replica_axis_index)
        with snapshot_mgr._lock:
          new_snapshot_mgr._latest_snapshot = snapshot_mgr._latest_snapshot
        python_vars["snapshot"] = new_snapshot_mgr
        snapshot_mgr = new_snapshot_mgr

      # 2. Re-run setup_train_loop to rebuild Mesh, Model, Optimizers
      (
          init_rng,
          checkpoint_manager,
          state_mesh_shardings,
          model,
          mesh,
          learning_rate_schedule,
          _,
          _,
          rampup_manager,
          eval_data_iterator,
          state,  # Newly initialized scratch state
      ) = train_utils.setup_train_loop(
          config,
          recorder,
          devices=elastic_utils.live_devices(config),
          restore_checkpoint=False,
          checkpoint_manager=existing_checkpoint_manager,
      )
      init_rng = jax.device_put(
          init_rng,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
      )

      params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
          config, state_mesh_shardings
      )

      # 3. Re-compile train and eval steps for the NEW mesh
      jit_model, _ = nnx.split(state)

      with (
          jax.set_mesh(mesh),
          logical_axis_rules(config.logical_axis_rules),
      ):
        p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
            config,
            jit_model,
            mesh,
            state,
            state_mesh_shardings,
            train_step,
            eval_step,
            eval_data_iterator,
            params_shardings,
        )

      # 4. Restore TrainState from active state (device-to-device) or host snapshot
      restored_state: Any = None
      restored_step = -1
      if active_state is not None:
        _logger.info("[*] Resharding active state directly (device-to-device)...")
        for_sharding_dict = {
            "model": nnx.to_pure_dict(nnx.state(state.model)),
            "optimizer": nnx.to_pure_dict(nnx.state(state.optimizer)),
        }
        sharding_dict = jax.tree.map(lambda x: x.sharding, for_sharding_dict)
        active_dict = {
            "model": nnx.to_pure_dict(nnx.state(active_state.model)),
            "optimizer": nnx.to_pure_dict(nnx.state(active_state.optimizer)),
        }
        restored_dict = jax.device_put(active_dict, sharding_dict)
        nnx.update(state.model, restored_dict["model"])
        nnx.update(state.optimizer, restored_dict["optimizer"])
        restored_state = state
        restored_step = int(state.optimizer.step.value)
        _logger.info(
            "Resharding complete. Retrying. Slices used: %s",
            elastic_manager.active_slice_indices,
        )
      else:
        def _safe_replace(state_obj, pure_dict):
          if not isinstance(pure_dict, dict):
            return
          current_flat = dict(nnx.statelib.to_flat_state(state_obj))
          flat_pure = traverse_util.flatten_dict(pure_dict)
          filtered_pure = {}
          for kp, v in flat_pure.items():
            if kp in current_flat:
              filtered_pure[kp] = v
            else:
              int_kp = tuple(int(x) if str(x).isdigit() else x for x in kp)
              if int_kp in current_flat:
                filtered_pure[int_kp] = v
          if filtered_pure:
            nnx.replace_by_pure_dict(state_obj, traverse_util.unflatten_dict(filtered_pure))

        snapshot_loaded = False
        if snapshot_mgr is not None and snapshot_mgr.latest is not None:
          try:
            restored_step = snapshot_mgr.latest.step
            _logger.info("Attempting to restore from in-memory snapshot at step %d...", restored_step)
            abstract_dict = {
                "model": nnx.to_pure_dict(nnx.state(state.model)),
                "optimizer": nnx.to_pure_dict(nnx.state(state.optimizer)),
            }
            replicated_abstract_dict = train_utils.replicate_single_device_sharded_arrays(abstract_dict)
            restored_dict = snapshot_mgr.load(replicated_abstract_dict)
            restored_dict = train_utils.restore_original_shardings(restored_dict, abstract_dict)

            merged = jax.tree.map(
                lambda ckpt, init: init if isinstance(ckpt, jax.ShapeDtypeStruct) else ckpt,
                restored_dict,
                abstract_dict,
                is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
            )

            m_state = nnx.state(state.model)
            _safe_replace(m_state, merged["model"])
            nnx.update(state.model, m_state)
            opt_state = nnx.state(state.optimizer)
            _safe_replace(opt_state, merged["optimizer"])
            nnx.update(state.optimizer, opt_state)
            restored_state = state

            snapshot_loaded = True
            _logger.info("Successfully restored in-memory snapshot at step %d!", restored_step)
          except (RuntimeError, jax.errors.JaxRuntimeError) as e:
            _logger.warning("In-memory snapshot recovery failed (%s). Falling back to persistent checkpoint.", e)

        if not snapshot_loaded:
          if existing_checkpoint_manager is None:
            raise RuntimeError("No snapshots or persistent checkpoints available to restore from. Cannot recover.")
          _logger.info("Restoring from persistent checkpoint...")
          restored, _ = checkpointing.load_state_if_possible(
              existing_checkpoint_manager,
              None,
              config.load_parameters_path,
              config.load_full_state_path,
              config.checkpoint_storage_concurrent_gb,
              state,
              config.enable_single_replica_ckpt_restoring,
              config.dataset_type,
              use_ocdbt=config.checkpoint_storage_use_ocdbt,
              use_zarr3=config.checkpoint_storage_use_zarr3,
              enable_orbax_v1=config.enable_orbax_v1,
              checkpoint_conversion_fn=config.checkpoint_conversion_fn,
              source_checkpoint_layout=config.source_checkpoint_layout,
              expansion_factor_real_data=config.expansion_factor_real_data,
              maxtext_config=config,
          )
          overlay = restored["items"] if hasattr(restored, "__getitem__") and "items" in restored else restored
          if isinstance(overlay, train_state_nnx.TrainStateNNX):
            overlay_model = nnx.to_pure_dict(nnx.state(overlay.model))
            overlay_opt = nnx.to_pure_dict(nnx.state(overlay.optimizer)) if overlay.optimizer is not None else None
          elif isinstance(overlay, nnx.State):
            overlay_dict = overlay.to_pure_dict()
            overlay_model = overlay_dict.get("model", overlay_dict)
            overlay_opt = overlay_dict.get("optimizer", None)
          elif isinstance(overlay, dict):
            overlay_model = overlay.get("model", overlay)
            overlay_opt = overlay.get("optimizer", None)
          else:
            overlay_model = overlay
            overlay_opt = None

          m_state = nnx.state(state.model)
          _safe_replace(m_state, overlay_model)
          nnx.update(state.model, m_state)
          if overlay_opt is not None and state.optimizer is not None:
            opt_state = nnx.state(state.optimizer)
            _safe_replace(opt_state, overlay_opt)
            nnx.update(state.optimizer, opt_state)
          restored_state = state
          restored_step = int(state.optimizer.step.value)

        if metric_logger_instance is not None:
          metric_logger_instance.learning_rate_schedule = learning_rate_schedule

      if restored_state is None:
        raise RuntimeError("Recovery completed without restoring a train state.")

      # Update jax_device_state with the newly built JAX objects
      if isinstance(restored_state, train_state_nnx.TrainStateNNX):
        _, restored_state = nnx.split(restored_state)
      jax_device_state["state"] = restored_state
      jax_device_state["init_rng"] = init_rng
      jax_device_state["model"] = model
      jax_device_state["mesh"] = mesh
      jax_device_state["state_mesh_shardings"] = state_mesh_shardings
      jax_device_state["p_train_step"] = p_train_step
      jax_device_state["p_eval_step"] = p_eval_step

      new_data_loader, new_data_iter, new_eval_iter = recreate_dataloaders(config, mesh, recorder, rampup_manager)

      python_vars["step"] = restored_step
      python_vars["data_loader"] = new_data_loader
      python_vars["data_iterator"] = new_data_iter
      python_vars["eval_data_iterator"] = new_eval_iter
      python_vars["checkpoint_manager"] = checkpoint_manager
      python_vars["rampup_manager"] = rampup_manager
      python_vars["last_step_completion"] = datetime.datetime.now()

      _logger.info("Recovery complete! Resuming safely at step %d...", restored_step)
      # State is fully restored on the new mesh: close out the "reinit" badput
      # window and log the fully-recovered slice counts.
      elastic_utils.record_elastic_reinit_end()
      break

    except (jax.errors.JaxRuntimeError, pathways_manager.ScaleUpSignalError, RuntimeError) as e:
      is_no_replicas_err = isinstance(e, RuntimeError) and "No active replicas found" in str(e)
      if (
          isinstance(e, pathways_manager.ScaleUpSignalError)
          or is_no_replicas_err
          or elastic.is_error_due_to_slice_down(e)
      ):
        _logger.warning("Slice state change or error caught during recovery: %s. Retrying recovery.", e)
      else:
        raise


def train_loop(config, recorder, state=None):
  """Main Training loop."""
  # pathwaysutils' elastic Manager; elastic_utils keeps the global untyped.
  elastic_manager: Any = None
  snapshot_mgr = None
  devices = None
  stop_event = None
  monitor_thread = None

  if config.elastic_enabled:
    elastic_utils.ensure_elastic_manager_initialized(config)
    elastic_manager = elastic_utils.elastic_manager
    if elastic_manager is None:
      raise RuntimeError("elastic_enabled is set but no elastic manager could be initialized.")
    _logger.info("[*] Active slices at startup: %s", elastic_manager.active_slice_indices)
    # Seed a slice-count record now that elastic_manager exists, so cumulative
    # slice-efficiency queries always have a record at/near job start to seed
    # from instead of treating the pre-first-event stretch as zero efficiency.
    elastic_utils.record_slice_state(recorder)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=elastic_manager._monitor_new_slices,  # pylint: disable=protected-access
        args=(stop_event, config.elastic_new_slice_check_period),
        daemon=True,
    )
    monitor_thread.start()
    elastic_utils.mutate_config_for_topology(config, elastic_manager)
    devices = elastic_utils.live_devices(config)
  else:
    _logger.info("[*] Standard Non-Elastic Training.")

  # Kills the workload if initialization takes longer than 20 minutes
  with watchdog.watchdog(name="initialization", timeout=20 * 60, repeat=False):
    while True:
      try:
        if config.elastic_enabled and elastic_manager:
          elastic_utils.mutate_config_for_topology(config, elastic_manager)
          devices = elastic_utils.live_devices(config)

        setup_results = {}
        init_complete_event = threading.Event()

        def run_setup():
          try:
            results = train_utils.setup_train_loop(config, recorder, devices=devices)
            setup_results["results"] = results
          except Exception as e:
            setup_results["exception"] = e
          finally:
            init_complete_event.set()

        setup_thread = threading.Thread(target=run_setup, daemon=True)
        setup_thread.start()

        while True:
          init_done = init_complete_event.wait(timeout=1)

          if elastic_manager and elastic_utils.elastic_enabled(config):
            new_slice = bool(elastic_manager.available_inactive_slices)

            if new_slice and not init_done:
              max_logging.log("New slice detected during initialization. Triggering retry.")
              raise pathways_manager.ScaleUpSignalError("Scale up during initialization")

            if init_done and new_slice:
              raise pathways_manager.ScaleUpSignalError("Both events set during initialization")

          if init_done:
            break

        if "exception" in setup_results:
          raise setup_results["exception"]

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
        ) = setup_results["results"]

        init_rng = jax.device_put(init_rng, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
        break  # Initialization succeeded!
      except (jax.errors.JaxRuntimeError, pathways_manager.ScaleUpSignalError) as e:
        is_scale_up = isinstance(e, pathways_manager.ScaleUpSignalError)
        is_slice_down = isinstance(e, jax.errors.JaxRuntimeError) and elastic.is_error_due_to_slice_down(e)
        if elastic_utils.elastic_snapshot(config) and (is_scale_up or is_slice_down):
          _logger.warning(
              "Elastic event or slice failure caught during initialization: %s. Refreshing slice topology and retrying setup.",
              e,
          )
          if elastic_manager:
            time.sleep(5)
            elastic_manager.active_slice_indices = elastic.get_active_slice_indices(elastic_manager.slice_to_devices)
        else:
          _logger.warning(
              "Elastic signal or JAX error caught during initialization: %s. Re-raising or bubbling to elastic_retry.",
              e,
          )
          raise

  # Throttling is applied only if configured (dcn_bandwidth_limit is set).
  # The default flag value is empty, meaning no throttling is applied by default.
  train_utils.maybe_apply_dcn_throttling(config)

  start_step = get_first_step(model, state)  # this is the start_step for training
  train_utils.validate_completed_steps(start_step, config.steps)

  jit_model_dropless = None

  if config.enable_diloco:
    # state is the DiLoCoTrainState; `model` is already the TrainStateNNX graphdef the inner step needs.
    jit_model = model
  else:
    jit_model, state = nnx.split(state)
    if config.retry_when_tokens_dropped:
      reconstructed = nnx.merge(jit_model, state)
      for _, module in nnx.iter_graph(reconstructed):
        if type(module).__name__ == "RoutedMoE":
          module.force_dropless = True
          module.num_moe_token_chunks = getattr(config, "retry_num_moe_token_chunks", 2)
          module.moe_chunk_barrier = True
        # The decoder (NNXDecoder); matching the class name "Decoder" matched nothing.
        elif hasattr(module, "get_remat_policy"):
          module.remat_policy_override = "full"
      jit_model_dropless, _ = nnx.split(reconstructed)
      del reconstructed, _
      gc.collect()

  if config.enable_diloco:
    # DiLoCoTrainState.params already holds the param shardings the inner step needs;
    # the Zero-1 opt overlay doesn't apply through the diloco wrapper.
    params_shardings = state_mesh_shardings.params
  else:
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config,
      jit_model,
      mesh,
      state,
      state_mesh_shardings,
      train_step,
      eval_step,
      eval_data_iterator,
      params_shardings,
  )

  p_train_step_dropless = None
  p_eval_step_dropless = None
  if jit_model_dropless is not None:
    p_train_step_dropless, p_eval_step_dropless = train_utils.jit_train_and_eval_step(
        config,
        jit_model_dropless,
        mesh,
        state,
        state_mesh_shardings,
        train_step,
        eval_step=eval_step,
        eval_data_iterator=eval_data_iterator,
        params_shardings=params_shardings,
    )

  # Do not enter the legacy `mesh` context manager here: the training loop calls
  # p_train_step without it, and the mismatch in jit's tracing-cache key would
  # cause train_step to be traced and compiled a second time on the first step.
  with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
    data_sharding = sharding.get_input_data_sharding(config, mesh)
    shaped_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding)
    if config.shard_optimizer_over_data:
      # NNX: reshard state so params match the data-sharded in_shardings (Zero-1 layout)
      state = jax.device_put(state, state_mesh_shardings)
    if config.enable_diloco:
      # The DiLoCo train step takes (state, batch, rng), like the inner NNX step.
      lower_args = (state, shaped_batch, init_rng)
    else:
      lower_args = (state, shaped_batch)
    maxtext_utils.maybe_dump_jaxpr(config, p_train_step, lower_args)
    if config.compiled_trainstep_file == "" and not jax.config.jax_enable_pgle:
      # Compile only when there is no pre-compiled file loaded. With AutoPGLE, an
      # ahead-of-time compiled executable can never be reused by the dispatch path
      # (the active PGLE profiler is part of JAX's executable cache key), so this
      # compile would only add a third full compilation on top of the profiling
      # compile and the FDO recompile; skip it and its memory stats.
      compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
      compiled = p_train_step.lower(*lower_args).compile(compiler_options=compiler_options)
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats, prefix="train")

  # Ahead-of-time compile the evaluation step alongside the training step to
  # warm up the XLA executable cache and avoid JIT compilation pause on the
  # first eval step.
  if p_eval_step is not None and config.compiled_trainstep_file == "" and not jax.config.jax_enable_pgle:
    with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules_for_eval):
      compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
      data_sharding_eval = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
      shaped_eval_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding_eval, is_eval=True)
      if config.enable_diloco:
        eval_lower_args = (state, shaped_eval_batch, init_rng)
      else:
        eval_lower_args = (state, shaped_eval_batch)
      compiled_eval = p_eval_step.lower(*eval_lower_args).compile(compiler_options=compiler_options)
      compiled_eval_stats = compiled_eval.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_eval_stats, prefix="eval")
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger_instance = metric_logger.MetricLogger(
      config=config, learning_rate_schedule=learning_rate_schedule, start_step=start_step
  )

  # Write train config params, num model params, and XLA flags to tensorboard
  if config.enable_diloco:
    setup_params = state.params  # DiLoCoTrainState.params: the outer (global) params
  else:
    _, setup_params, _ = nnx.split(state.model, nnx.Param, ...)
  metric_logger_instance.write_setup_info_to_tensorboard(setup_params)

  elastic_utils.record_elastic_reinit_end()

  # Initialize host snapshot manager only in elastic snapshot mode
  if elastic_utils.elastic_snapshot(config):
    replica_axis_idx = config.mesh_axes.index("data")
    snapshot_mgr = Snapshotter(replica_axis_index=replica_axis_idx)
    save_snapshot(snapshot_mgr, state, start_step)

  # Initialize dictionaries for refactored iteration
  jax_device_state = {
      "state": state,
      "init_rng": init_rng,
      "mesh": mesh,
      "state_mesh_shardings": state_mesh_shardings,
      "p_train_step": p_train_step,
      "p_train_step_dropless": p_train_step_dropless,
      "p_eval_step": p_eval_step,
      "p_eval_step_dropless": p_eval_step_dropless,
      "model": model,
  }

  python_vars = {
      "step": start_step,
      "last_step_completion": datetime.datetime.now(),
      "data_loader": data_loader,
      "rampup_manager": rampup_manager,
      "recorder": recorder,
      "checkpoint_manager": checkpoint_manager,
      "data_iterator": data_iterator,
      "eval_data_iterator": eval_data_iterator,
      "metric_logger_instance": metric_logger_instance,
      "prof": prof,
      "elastic_manager": elastic_manager,
      "snapshot": snapshot_mgr,
  }

  immutable_data = {
      "config": config,
      "logical_axis_rules_for_train": config.logical_axis_rules,
      "logical_axis_rules_for_eval": config.logical_axis_rules_for_eval,
      "shard_optimizer_over_data": config.shard_optimizer_over_data,
      "shard_mode": config.shard_mode,
      "steps": config.steps,
      "eval_interval": config.eval_interval,
      "eval_steps": config.eval_steps,
      "eval_start_step": config.eval_start_step,
      "save_checkpoint_on_completion": config.save_checkpoint_on_completion,
      "start_step": start_step,
      "dump_hlo": config.dump_hlo,
      "dump_step": config.dump_step,
      "dump_hlo_local_dir": config.dump_hlo_local_dir,
      "dump_hlo_gcs_dir": config.dump_hlo_gcs_dir,
      "dump_hlo_module_name": config.dump_hlo_module_name,
      "dump_hlo_delete_local_after": config.dump_hlo_delete_local_after,
      "dump_hlo_upload_all": config.dump_hlo_upload_all,
  }

  _job_completed_gracefully = False
  te_moe_overflow_window = []
  try:
    python_vars["last_step_completion"] = datetime.datetime.now()
    needs_recovery = False

    mllog_utils.init_print(config)
    mllog_utils.init_stop()
    mllog_utils.run_start()
    mllog_utils.block_start(config, start_step)

    # Using while loop to allow for potential dynamic 'steps' adjustment in future
    while python_vars["step"] < immutable_data["steps"]:
      step = python_vars["step"]
      # Stays None if this iteration ends in an elastic recovery instead of a
      # completed train step.
      metrics = None
      # Print the stacktrace every 60s and also exit the workload if longer than 600s
      with (
          watchdog.watchdog("step-stack-status", timeout=60),
          watchdog.watchdog("step-timebomb", timeout=15 * 60, repeat=False),
      ):
        is_scale_up = False
        try:
          # Scale-up check at the end of the step (only if elastic snapshot)
          if elastic_utils.elastic_snapshot(config) and elastic_manager.available_inactive_slices:
            elastic_utils.record_elastic_event_start(recorder, config)
            recover(
                jax_device_state,
                python_vars,
                immutable_data,
                active_state=jax_device_state["state"],
            )
            # Start snapshot save immediately on the new mesh
            save_snapshot(snapshot_mgr, jax_device_state["state"], python_vars["step"])

          metrics = training_loop_iteration(jax_device_state, python_vars, immutable_data)
          python_vars["step"] += 1

        except (jax.errors.JaxRuntimeError, pathways_manager.ScaleUpSignalError) as e:
          if elastic_utils.elastic_snapshot(config) and (
              isinstance(e, pathways_manager.ScaleUpSignalError) or elastic.is_error_due_to_slice_down(e)
          ):
            _logger.error("[!] Elastic event detected around step %d", python_vars["step"])
            elastic_utils.record_elastic_event_start(recorder, config)
            needs_recovery = True
            is_scale_up = isinstance(e, pathways_manager.ScaleUpSignalError)
          else:
            # Checkpoint mode or non-elastic error: bubble to elastic_retry
            elastic_utils.maybe_bubble_elastic_exception(config, e)
            # Non-elastic or unrelated JAX error: log and re-raise
            _logger.exception(
                "[!] JAX Runtime Error detected around step %d. Re-raising.",
                python_vars["step"],
            )
            raise

        if needs_recovery:
          needs_recovery = False

          if is_scale_up:
            _logger.info("[*] Scale up signal caught: resharding active state directly (device-to-device)...")
            recover(
                jax_device_state,
                python_vars,
                immutable_data,
                active_state=jax_device_state["state"],
            )
          else:
            # Slice Failure Recovery
            recover(jax_device_state, python_vars, immutable_data)

          # Save snapshot across the newly recovered mesh layout
          save_snapshot(snapshot_mgr, jax_device_state["state"], python_vars["step"])
          continue

      # A recovered step produces no metrics, so it contributes nothing to the
      # overflow window.
      if metrics is not None and getattr(config, "te_moe_block", False):
        te_moe_overflow_window.append(
            (
                int(step),
                metrics["scalar"]["learning/te_moe_capacity_overflow"],
                metrics["scalar"]["learning/te_moe_max_total_recv_tokens"],
                metrics["scalar"]["learning/te_moe_recv_capacity_per_rank"],
            )
        )
        check_overflow = (
            len(te_moe_overflow_window) == config.te_ep_overflow_check_every_n_steps or step == config.steps - 1
        )
        if check_overflow:
          checked_window = jax.device_get(tuple(te_moe_overflow_window))
          overflowing_steps = [entry for entry in checked_window if bool(np.asarray(entry[1]))]
          te_moe_overflow_window.clear()
        else:
          overflowing_steps = []

        if overflowing_steps:
          overflow_step = overflowing_steps[0][0]
          observed = max(int(np.asarray(entry[2])) for entry in overflowing_steps)
          capacity = min(int(np.asarray(entry[3])) for entry in overflowing_steps)
          prof.deactivate()
          message = (
              "TE MoE receive capacity overflow at training step "
              f"{overflow_step} (detected at the {config.te_ep_overflow_check_every_n_steps}-step "
              f"check ending at step {step}): "
              f"observed padded receive demand {observed} exceeds "
              f"recv_capacity_per_rank {capacity} "
              f"(ragged_buffer_factor={config.ragged_buffer_factor}). "
              "The optimizer update was skipped; increase ragged_buffer_factor, "
              "or set it to -1 to reserve worst-case dropless capacity."
          )
          max_logging.error(message)
          raise RuntimeError(message)

    # Unpack state for post-loop actions
    state = jax_device_state["state"]

    if immutable_data["save_checkpoint_on_completion"]:
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator)

    if checkpoint_manager is not None:
      # in case the last checkpoint_period checkpoint is still in progress
      checkpoint_manager.wait_until_finished()
    _job_completed_gracefully = True
  except exceptions.StopTraining as e:
    prof.deactivate()
    max_logging.log(f"Training stopped: {str(e)}")
    _job_completed_gracefully = True
  finally:
    # Flush before run_stop: the last buffered step still owes a tracked_stats event, and the
    # reference log ends at run_stop. Not enforced by the 6.0 compliance checker.
    metric_logger_instance.flush_metrics_and_cleanup()
    # Terminate monitoring thread (Elastic Mode only)
    if stop_event is not None:
      stop_event.set()
    if monitor_thread is not None:
      monitor_thread.join()
    if _job_completed_gracefully:
      record_goodput(recorder, RECORD_JOB_END_TIME)
      samples_count = (python_vars["step"] - immutable_data["start_step"]) * config.global_batch_size_to_train_on
      # Only reached when eval never hit target_eval_loss; a converged run already
      # logged RUN_STOP with status "success" from eval_stop.
      mllog_utils.run_stop(status="aborted", current_epoch_num=samples_count, step=python_vars["step"])
    train_utils.maybe_cleanup_dcn_throttling(config)

  return state


def initialize(argv: Sequence[str]) -> tuple[pyconfig.HyperParameters, Any]:
  """Initialization of hyperparameters and utilities"""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  if _TF_AVAILABLE:
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

  if config.te_comm_gemm_overlap != TeCommGemmOverlapPolicy.DISABLED:
    max_utils.bootstrap_transformer_engine_cgemm(config)

  # Create the Goodput recorder
  mllog_utils.init_start(config)
  recorder = create_goodput_recorder(config)

  return config, recorder


def run(config, recorder):
  """Run the job given hyperparameters and utilities."""
  with (max_utils.maybe_get_transformer_engine_context(config),):
    train_loop(config, recorder)


def get_train_func(config, recorder, argv):
  """Returns the train function, wrapping in elastic_retry if backup_kind is checkpoint."""
  if config.elastic_enabled:
    max_logging.log(f"Elastic utils: Elastic training enabled with {config.elastic_backup_kind} backup.")

  if config.elastic_enabled and config.elastic_backup_kind == "checkpoint":

    def on_elastic_event():
      elastic_utils.record_elastic_event_start(recorder, config)

    def on_slices_ready():
      elastic_utils.record_elastic_wait_end_and_reinit_start(recorder)

    def elastic_train_wrapper(argv: Sequence[str]) -> None:
      """Wrapper for elastic training initializes variables and runs the train loop."""
      elastic_config, elastic_recorder = initialize(argv)
      run(
          elastic_config,
          elastic_recorder,
      )

    train_func = elastic_utils.elastic_retry(
        config,
        callback_fn=on_elastic_event,
        pre_callback_fn=on_slices_ready,
    )(functools.partial(elastic_train_wrapper, argv=argv))
  else:
    # Use the already initialized variables
    def train_func():
      run(config, recorder)

  return train_func


def main(argv: Sequence[str]) -> None:
  config, recorder = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  app.run(main)
