# Copyright 2026 Google LLC
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

"""NNX-native pre-training loop for MaxText.

This module implements a pre-training loop that uses the Flax NNX API throughout,
in contrast to train.py which wraps NNX models inside Linen's TrainState.


  Architecture

  ┌─────────────────────────────────┬──────────────────────────────────────────────────────────────────────────┐
  │function                         │                               What it does                               │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ loss_fn                         │   Forward-pass + cross-entropy; for both train and eval;                 │
  │                                 │   called directly on an nnx.Module                                       │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ train_step                      │ Functional step — merges (graphdef, opt_state) → runs nnx.value_and_grad │
  │                                 │  → updates optimizer → returns new nnx.State                             │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ eval_step                       │ Same merge pattern, forward-only, no grads                               │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ _create_and_shard_optimizer     │ Wraps model + optax tx in nnx.Optimizer, derives partition specs via     │
  │                                 │ nnx.get_partition_spec, shards state with jax.jit(out_shardings=…)       │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ _build_jit_steps                │ Partially applies static (graphdef, config) then wraps with              │
  │                                 │ jax.jit(in_shardings, out_shardings, donate_argnums=(0,1))               │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ _maybe_restore_checkpoint /     │ Orbax round-trip using the NNX {"value": array} wire format              │
  │ _maybe_save_checkpoint          │                                                                          │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ train_loop                      │ Full loop: model → optimizer → data → checkpoint → JIT compile → step →  │
  │                                 │ eval → log                                                               │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ main / initialize / run         │ Entry-point boilerplate matching train.py conventions                    │
  └─────────────────────────────────┴──────────────────────────────────────────────────────────────────────────┘

  Key differences from train.py

  - No Linen TrainState — state lives in nnx.Optimizer (model params + optax state + step counter).
  - Gradient computation uses nnx.value_and_grad, which is NNX-graph-aware. It differentiates only through
    nnx.Param variables; non-differentiable NNX variables (RNGs, cache, …) are untouched.
  - Gradient clipping uses optax.clip_by_global_norm directly, avoiding the Linen-TrainState coupling in
  apply_gradient_clipping.
  - JIT boundary: graphdef is a Python-static closure; only opt_state (a plain pytree of arrays) crosses the JIT
  boundary with donate_argnums=(0,1)
  - The JIT boundary uses split/merge so that graphdef is static and state is
    donated as a pytree, preserving full sharding control via jax.jit shardings.
  - Checkpointing saves/restores the raw nnx.State pytree via Orbax.

Entry point:
  python -m maxtext.trainers.pre_train.nnx_train <config_file> [overrides…]
"""

import contextlib
import datetime
import functools
import os
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxtext.common import checkpointing, profiler
from maxtext.common.common_types import ShardMode
from maxtext.common.data_loader import create_dataloader
from maxtext.common.gcloud_stub import cloud_diagnostics as _cloud_diag
from maxtext.common.gcloud_stub import is_decoupled, vertex_tensorboard_modules
from maxtext.common.goodput import (
    RECORD_JOB_END_TIME,
    RECORD_JOB_START_TIME,
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
    record_goodput,
)
from maxtext.common.metric_logger import MetricLogger, record_activation_metrics
from maxtext.configs import pyconfig
from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from maxtext.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss
from maxtext.optimizers import optimizers
from maxtext.utils import exceptions, max_logging, max_utils, maxtext_utils, model_creation_utils, sharding
from maxtext.utils.globals import EPS
from maxtext.utils.gradient_accumulation import nnx_gradient_accumulation_loss_and_grad
from maxtext.utils.rampup_batch import create_rampup_manager

_diag_modules = _cloud_diag()
diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration = _diag_modules
VertexTensorboardManager, _vertex_tb_is_stub = vertex_tensorboard_modules()


# ---------------------------------------------------------------------------
# Loss computation for both train and eval
# ---------------------------------------------------------------------------


def loss_fn(model: nnx.Module, config, data: dict[str, jax.Array], dropout_rng: jax.Array, is_train=True):
  """Compute cross-entropy loss for one batch using an NNX model.

  Args:
    model: The NNX Transformer (or compatible) model. Called in-place; no
      explicit params argument is needed because the NNX module carries state.
    config: MaxText Config object.
    data: Batch dict with keys "inputs", "inputs_position", "inputs_segmentation",
      "targets", "targets_segmentation".
    dropout_rng: PRNG key used to seed dropout layers.
    is_train: True for train_step and False for eval_step.

  Returns:
    (loss, aux) where loss is a scalar and aux is a dict of auxiliary metrics.
  """
  # rng1, aqt_rng = jax.random.split(dropout_rng)

  # Trim to micro-batch size (handles per_device_batch_size < 1 cases)
  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    batch = {k: v[: config.micro_batch_size_to_train_on, :] for k, v in data.items()}
  else:
    batch = {k: v[: config.micro_batch_size_to_eval_on, :] for k, v in data.items()}

  # Flax NNX model
  logits = model(
      decoder_input_tokens=batch["inputs"],
      decoder_positions=batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      encoder_images=batch["images"] if config.use_multimodal else None,
      encoder_image_masks=batch["image_masks"] if config.use_multimodal and "image_masks" in batch else None,
      enable_dropout=config.enable_dropout if is_train else False,
      decoder_target_tokens=batch["targets"],
      decoder_target_mask=batch["targets_segmentation"],
  )
  intermediate_outputs = {}
  one_hot_targets = jax.nn.one_hot(batch["targets"], config.vocab_size)
  xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=config.z_loss_multiplier)

  xent = nn.with_logical_constraint(xent, ("activation_embed_and_logits_batch", "activation_length"))
  z_loss = nn.with_logical_constraint(z_loss, ("activation_embed_and_logits_batch", "activation_length"))

  # Mask out paddings at the end of each example.
  xent = xent * (batch["targets_segmentation"] != 0)
  z_loss = z_loss * (batch["targets_segmentation"] != 0)

  total_loss = jnp.sum(xent)
  total_z_loss = jnp.sum(z_loss)

  total_weights = jnp.sum(batch["targets_segmentation"] != 0)
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
    indexer_losses = []
    # Extract 'indexer_loss' from model intermediates.
    # We check for paths ending in ('self_attention', 'indexer_loss').
    # This handles varying paths caused by different layer names.
    for path, val in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
      path_keys = tuple(k.key for k in path if hasattr(k, "key"))
      if path_keys[-2:] == ("self_attention", "indexer_loss"):
        indexer_losses.append(jnp.ravel(val))

    if indexer_losses:
      indexer_loss = jnp.mean(jnp.concatenate(indexer_losses))
      loss += indexer_loss
    else:
      max_logging.debug("No indexer loss found.")

  # get MoE load balance loss
  moe_lb_loss = 0.0
  if config.num_experts > 1:
    # Note: the key is affected by the model implementation
    possible_keys = [
        ("intermediates", "decoder", "layers", "moe_lb_loss"),
        ("intermediates", "decoder", "moe_layers", "moe_lb_loss"),
    ]

    total_moe_lb_loss = 0.0
    found_loss = False
    for nested_key in possible_keys:
      total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
      if total_moe_lb_loss != 0.0:
        found_loss = True
        break

    if not found_loss:
      max_logging.debug("\nNo MoE load balance loss found. Defaulting to 0.0.")

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
      "z_loss": total_z_loss,
      "total_weights": total_weights,
      "moe_lb_loss": moe_lb_loss,
      "indexer_loss": indexer_loss,
      "moe_bias_updates": moe_bias_updates,
      "mtp_loss": mtp_loss,
  }
  return loss, aux


# ---------------------------------------------------------------------------
# Train / eval steps  (purely functional, JIT-able)
# ---------------------------------------------------------------------------


def train_step(
    model_graphdef: nnx.graph.NodeDef,
    opt_graphdef: nnx.graph.NodeDef,
    model_state: nnx.State,
    opt_state: nnx.State,
    data: dict[str, jax.Array],
    dropout_rng: jax.Array,
    config,
):
  """One training step: forward + backward + optimizer update.

  Args:
    model_graphdef: Static NNX graph definition for the model (JIT closure).
    opt_graphdef: Static NNX graph definition for the optimizer (JIT closure).
    model_state: Mutable model parameter pytree (donated).
    opt_state: Mutable optimizer state pytree (donated).
    data: Batch of token IDs and metadata.
    dropout_rng: PRNG key for dropout.
    config: MaxText Config.

  Returns:
    (new_model_state, new_opt_state): Updated pytrees.
    metrics: Dict of scalar training metrics.
  """
  model: nnx.Module = nnx.merge(model_graphdef, model_state)
  optimizer: nnx.Optimizer = nnx.merge(opt_graphdef, opt_state)
  if config.use_dpo:
    # Need impl on NNX
    pass
    # state, reference_params = _split_dpo_state(state)
    # state_mesh_shardings, reference_params_sharding = _split_dpo_state(state_mesh_shardings)
    # extra_dpo_args = [reference_params]
    # loss_fn = dpo_loss_fn

  # Compute loss and gradients w.r.t. model parameters.
  # nnx.value_and_grad differentiates only through nnx.Param variables,
  # keeping non-differentiable state (RNGs, cache, etc.) frozen.
  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = nnx_gradient_accumulation_loss_and_grad(loss_fn, model, config, data, dropout_rng)
  else:
    if config.optimizer_memory_host_offload:
      # Need impl on NNX
      pass
      # if config.use_dpo:
      #  reference_params = jax.device_put(
      #      reference_params,
      #      max_utils.with_memory_kind(reference_params_sharding, "device"),
      #  )
      #  extra_dpo_args = [reference_params]
    if config.shard_optimizer_over_data:
      # Need impl on NNX
      pass
      # params = jax.tree.map(
      #    functools.partial(sharding.maybe_shard_with_name, shard_mode=config.shard_mode),
      #    params,
      #    params_shardings,
      # )
    grad_fn = nnx.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, aux), raw_grads = grad_fn(model, config, data, dropout_rng, is_train=True)

  # Cast gradients to configured dtype before clipping / accumulation
  raw_grads = jax.tree.map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )
  intermediate_outputs = aux["intermediate_outputs"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux["indexer_loss"]
  z_loss = aux["z_loss"]
  moe_bias_updates = aux["moe_bias_updates"]
  mtp_loss = aux["mtp_loss"]

  # Gradient clipping (implemented directly to avoid Linen TrainState dependency)
  if config.gradient_clipping_threshold > 0:
    clip_tx = optax.clip_by_global_norm(config.gradient_clipping_threshold)
    grads, _ = clip_tx.update(raw_grads, clip_tx.init(raw_grads), None)
  else:
    grads = raw_grads
  if config.optimizer_memory_host_offload:
    # Need impl on NNX
    pass
    # state = state.replace(
    #    opt_state=jax.device_put(
    #        state.opt_state,
    #        jax.tree_util.tree_map(
    #            lambda x: x.with_memory_kind(kind="device"),
    #            state_mesh_shardings.opt_state,
    #        ),
    #    )
    # )
  # Move all parameters to device before optimizer update
  if config.parameter_memory_host_offload:
    max_logging.log("\nMoving all parameters to device before optimizer update")
    # Need impl on NNX
    # def move(path, value):
    #  max_logging.log(f"train.py: Moving f{path} to device")
    #  return value.with_memory_kind(kind="device")

    # state = state.replace(
    #    params=jax.device_put(
    #        state.params,
    #        jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params),
    #    )
    # )

  # NNX 0.11+: update takes (model, grads) explicitly.
  optimizer.update(model, grads)

  new_model_state = nnx.state(model)
  new_opt_state = nnx.state(optimizer)

  # Apply updates for Auxiliary-Loss-Free load balancing for DeepSeek family
  if config.routed_bias and config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
    # Need impl on NNX
    pass
    # target_path = ("params", "decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias")
    # Flax 'sow' returns a tuple, so we take the first element [0].
    # Updates the shape to be aligned with state.
    # moe_bias_updates = jnp.array(moe_bias_updates[0]).transpose()
    # new_state = maxtext_utils.update_state_param(new_state, target_path, moe_bias_updates)

  scalar_metrics = {
      "learning/loss": loss,
      "learning/z_loss": z_loss,
      "learning/moe_lb_loss": moe_lb_loss,
      "learning/indexer_loss": indexer_loss,
      "learning/mtp_loss": mtp_loss,
      "learning/total_weights": total_weights,
  }
  if config.use_qk_clip:
    # Apply QK-Clip
    # Need impl on NNX
    pass
    # new_state = qk_clip_utils.apply_qk_clip(new_state, intermediate_outputs, config)

    # Report max_logits metric
    # global_max_logit = qk_clip_utils.calculate_max_logit_metric(intermediate_outputs)
    # if global_max_logit is not None:
    #  scalar_metrics["learning/max_logits"] = global_max_logit

  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(nnx.state(model, nnx.Param))
  if config.use_dpo:
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  if config.use_dpo:
    # Need impl on NNX
    pass
    # new_state = _merge_dpo_state(new_state, reference_params)
  return (new_model_state, new_opt_state), metrics


def eval_step(
    model_graphdef: nnx.graph.NodeDef,
    model_state: nnx.State,
    data: dict[str, jax.Array],
    dropout_rng: jax.Array,
    config,
):
  """One evaluation step: forward only, no gradient computation.

  Args:
    model_graphdef: Static NNX graph definition for the model.
    model_state: Current model parameter pytree (read-only).
    data: Batch of token IDs and metadata.
    dropout_rng: PRNG key (dropout disabled for eval, but kept for API symmetry).
    config: MaxText Config.

  Returns:
    metrics: Dict of scalar evaluation metrics.
  """
  model: nnx.Module = nnx.merge(model_graphdef, model_state)
  loss, aux = loss_fn(model, config, data, dropout_rng, is_train=False)

  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)

  total_loss = aux["total_loss"]
  z_loss = aux["z_loss"]
  total_weights = aux["total_weights"]
  moe_lb_loss = aux["moe_lb_loss"]
  indexer_loss = aux["indexer_loss"]
  mtp_loss = aux["mtp_loss"]
  metrics = {
      "scalar": {
          "evaluation/loss": loss,
          "evaluation/z_loss": z_loss,
          "evaluation/total_loss": total_loss,
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


# ---------------------------------------------------------------------------
# Training-loop setup
# ---------------------------------------------------------------------------


def _create_and_shard_optimizer(model: nnx.Module, config, mesh: Mesh):
  """Creates an nnx.Optimizer and returns sharded model + optimizer states.

  In NNX 0.11+, the optimizer does not hold a model reference, so model and
  optimizer are kept as independent objects with separate graphdefs, state
  pytrees, and sharding specs throughout the training loop.

  Args:
    model: Sharded NNX model (already placed on devices).
    config: MaxText Config.
    mesh: JAX device mesh.

  Returns:
    model_graphdef: Static NNX graph definition for the model.
    opt_graphdef: Static NNX graph definition for the optimizer.
    model_state: Sharded model parameter pytree (donated to JIT steps).
    opt_state: Sharded optimizer state pytree (donated to JIT steps).
    model_shardings: Partition specs for model_state.
    opt_shardings: Partition specs for opt_state.
    learning_rate_schedule: Learning-rate schedule function.
  """
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule, model)
  # NNX 0.11+: wrt is mandatory; optimizer does not store a model reference.
  optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

  # Derive separate partition specs for model and optimizer.
  model_graphdef, abstract_model_state = nnx.split(nnx.eval_shape(lambda: model))
  opt_graphdef, abstract_opt_state = nnx.split(nnx.eval_shape(lambda: optimizer))

  with nn.logical_axis_rules(config.logical_axis_rules):
    model_shardings = nn.logical_to_mesh_sharding(
        nnx.get_partition_spec(abstract_model_state), mesh, config.logical_axis_rules
    )
    opt_shardings = nn.logical_to_mesh_sharding(
        nnx.get_partition_spec(abstract_opt_state), mesh, config.logical_axis_rules
    )

  _, model_state = nnx.split(model)
  _, opt_state = nnx.split(optimizer)

  @functools.partial(jax.jit, out_shardings=(model_shardings, opt_shardings))
  def shard_states(mshard, oshard):
    return mshard, oshard

  with mesh:
    model_state, opt_state = shard_states(model_state, opt_state)

  return model_graphdef, opt_graphdef, model_state, opt_state, model_shardings, opt_shardings, learning_rate_schedule


def _get_first_step(opt_state: nnx.State) -> int:
  """Extracts the current step counter from the optimizer state."""
  # nnx.Optimizer stores step as an nnx.Variable; its value is a scalar.
  step_leaves = [v for k, v in opt_state.flat_state().items() if "step" in str(k)]
  if step_leaves:
    return int(step_leaves[0])
  return 0


def _build_jit_steps(
    config,
    model_graphdef: nnx.graph.NodeDef,
    opt_graphdef: nnx.graph.NodeDef,
    mesh: Mesh,
    model_shardings: Any,
    opt_shardings: Any,
    eval_data_iterator,
):
  """JIT-compiles the train and eval step functions with sharding annotations.

  Returns:
    p_train_step: JIT-compiled train step.
    p_eval_step: JIT-compiled eval step (None if no eval data).
  """
  data_sharding = sharding.get_input_data_sharding(config, mesh)

  # Partial application captures static graphdefs and config outside JIT.
  _train_fn = functools.partial(train_step, model_graphdef, opt_graphdef, config=config)
  _train_fn.__name__ = "nnx_train_step"

  p_train_step = jax.jit(
      _train_fn,
      in_shardings=(model_shardings, opt_shardings, data_sharding, None),
      out_shardings=((model_shardings, opt_shardings), None),
      donate_argnums=(0, 1),  # donate both model_state and opt_state buffers
  )

  p_eval_step = None
  if eval_data_iterator is not None:
    # Eval only needs the model; optimizer state is not required.
    _eval_fn = functools.partial(eval_step, model_graphdef, config=config)
    _eval_fn.__name__ = "nnx_eval_step"
    p_eval_step = jax.jit(
        _eval_fn,
        in_shardings=(model_shardings, data_sharding, None),
        out_shardings=None,
        donate_argnums=(),
    )

  return p_train_step, p_eval_step


def _wrap_state(state: nnx.State):
  """Wraps each leaf in {"value": ...} to match the NNX checkpoint format."""
  return jax.tree.map(lambda v: {"value": v}, state, is_leaf=lambda n: isinstance(n, nnx.Variable))


def _unwrap_state(raw):
  """Unwraps {"value": ...} leaves back to plain arrays."""
  return jax.tree.map(lambda v: v["value"], raw, is_leaf=lambda x: isinstance(x, dict) and "value" in x)


def _maybe_restore_checkpoint(checkpoint_manager, model_state: nnx.State, opt_state: nnx.State, config, data_iterator):
  """Restores model and optimizer states from an Orbax checkpoint if one exists.

  Checkpoint layout: {"model": <model_state>, "optimizer": <opt_state>},
  with every leaf wrapped as {"value": <array>}.

  Returns:
    (model_state, opt_state, data_iterator, start_step)
  """
  if checkpoint_manager is None:
    return model_state, opt_state, data_iterator, 0

  try:
    import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

    latest = checkpoint_manager.latest_step()
    if latest is None:
      max_logging.log("No existing checkpoint found; starting from scratch.")
      return model_state, opt_state, data_iterator, 0

    max_logging.log(f"Restoring NNX checkpoint from step {latest}.")
    ckptr = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(
            restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
            save_concurrent_gb=config.checkpoint_storage_concurrent_gb,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
        )
    )

    target = {"model": _wrap_state(model_state), "optimizer": _wrap_state(opt_state)}
    restore_args = ocp.checkpoint_utils.construct_restore_args(target)
    checkpoint_dir = checkpoint_manager.directory / str(latest)
    restored_raw = ckptr.restore(checkpoint_dir, item=target, restore_args=restore_args)

    restored_model_state = _unwrap_state(restored_raw["model"])
    restored_opt_state = _unwrap_state(restored_raw["optimizer"])
    return restored_model_state, restored_opt_state, data_iterator, int(latest)

  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Checkpoint restore failed ({e}); starting from scratch.")
    return model_state, opt_state, data_iterator, 0


def _maybe_save_checkpoint(
    checkpoint_manager, model_state: nnx.State, opt_state: nnx.State, config, data_iterator, step: int
):
  """Saves model and optimizer states to an Orbax checkpoint."""
  if checkpoint_manager is None:
    return
  state_to_save = {"model": _wrap_state(model_state), "optimizer": _wrap_state(opt_state)}
  checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator, step)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_loop(config, recorder, state=None):
  """NNX pre-training loop.

  Args:
    config: MaxText Config.
    recorder: Goodput recorder (may be None).
    state: Unused; present for API symmetry with train.py.

  Returns:
    Final optimizer state pytree.
  """
  # ---- Model ----------------------------------------------------------------
  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    with nn.logical_axis_rules(config.logical_axis_rules):
      model, mesh = model_creation_utils.create_nnx_model(config)

  # ---- Optimizer + sharding -------------------------------------------------
  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    model_graphdef, opt_graphdef, model_state, opt_state, model_shardings, opt_shardings, learning_rate_schedule = (
        _create_and_shard_optimizer(model, config, mesh)
    )

    # ---- Data ---------------------------------------------------------------
    with jax.set_mesh(mesh):
      data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    rampup_manager = create_rampup_manager(config, checkpoint_manager=None)
    data_loader = create_dataloader(config, mesh, data_iterator, recorder, rampup_manager)

    # ---- Checkpointing -------------------------------------------------------
    logger = checkpointing.setup_checkpoint_logger(config)
    checkpoint_dir = config.checkpoint_dir if config.enable_checkpointing else ""
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        config.checkpoint_storage_use_ocdbt,
        config.checkpoint_storage_use_zarr3,
        config.enable_continuous_checkpointing,
        config.max_num_checkpoints_to_keep,
        config.checkpoint_storage_concurrent_gb,
        config.enable_single_controller,
        config.colocated_python_checkpointing,
        config.enable_single_replica_ckpt_restoring,
    )

    model_state, opt_state, data_iterator, start_step = _maybe_restore_checkpoint(
        checkpoint_manager, model_state, opt_state, config, data_iterator
    )

  # ---- JIT-compile steps ----------------------------------------------------
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    p_train_step, p_eval_step = _build_jit_steps(
        config, model_graphdef, opt_graphdef, mesh, model_shardings, opt_shardings, eval_data_iterator
    )

  # Trigger AOT compilation and print memory stats
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    shaped_batch = maxtext_utils.get_shaped_batch(config)
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    example_rng = jax.jit(jax.random.fold_in)(init_rng, 0)
    # Need imple below func on NNX
    # maxtext_utils.maybe_dump_jaxpr(config, p_train_step, (model_state, opt_state, shaped_batch, example_rng))
    if config.compiled_trainstep_file == "":  # compile only when there is no pre-compiled file loaded
      compiled = p_train_step.lower(model_state, opt_state, shaped_batch, example_rng).compile()
      compiled_stats = compiled.memory_analysis()
      max_utils.print_compiled_memory_stats(compiled_stats)

  # ---- Profiler / logger ----------------------------------------------------
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(model_state)

  # ---- Main loop ------------------------------------------------------------
  _job_completed_gracefully = False
  try:
    last_step_completion = datetime.datetime.now()
    max_logging.info(f"Entering train loop from start_step={start_step}")

    for step in np.arange(start_step, config.steps):
      prof.maybe_activate_profiler(step, opt_state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        example_batch = data_loader.load_next_batch(rampup_manager=rampup_manager)
        nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
        with maybe_record_goodput(recorder, GoodputEvent.STEP, step):
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            (model_state, opt_state), metrics = p_train_step(model_state, opt_state, example_batch, nextrng)

      step_time_delta = datetime.datetime.now() - last_step_completion
      last_step_completion = datetime.datetime.now()

      _maybe_save_checkpoint(checkpoint_manager, model_state, opt_state, config, data_iterator, step)

      # ---- Optional eval -------------------------------------------------------
      if (
          p_eval_step is not None
          and config.eval_interval > 0
          and step > start_step
          and (step + 1) % config.eval_interval == 0
      ):
        assert eval_data_iterator
        # Explicitly reset the eval iterator and counters before starting the eval loop
        eval_data_iterator.reset()
        metric_logger.reset_eval_metrics()

        eval_step_count = 0
        for eval_batch in eval_data_iterator:
          if config.eval_steps > 0 and eval_step_count >= config.eval_steps:
            break
          with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
            eval_metrics = p_eval_step(model_state, eval_batch, nextrng)
          metric_logger.record_eval_metrics(step, metrics=eval_metrics)
          max_logging.log(f"Completed eval step {eval_step_count}")
          eval_step_count += 1

        metric_logger.record_eval_metrics(step, eval_step_count=eval_step_count)
        if metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"] <= config.target_eval_loss:
          prof.deactivate()
          raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} achieved.")

      prof.maybe_deactivate_profiler(step, opt_state)

      if step == start_step:
        max_utils.print_mem_stats("After first step")

      metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)

    # Final checkpoint on loop completion
    if config.save_checkpoint_on_completion:
      _maybe_save_checkpoint(
          checkpoint_manager, model_state, opt_state, config, data_iterator, step=int(config.steps - 1)
      )
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

  return opt_state


# ---------------------------------------------------------------------------
# Entry-point helpers
# ---------------------------------------------------------------------------


def initialize(argv: Sequence[str]):
  """Initialise hyperparameters and utility objects."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  tf.config.set_visible_devices([], "GPU")

  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  if not config.enable_nnx:
    max_logging.log("WARNING: nnx_train.py requires enable_nnx=True. Forcing it on.")

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
  """Run the NNX training job.

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


def main(argv: Sequence[str]) -> None:
  config, recorder, diagnostic_config = initialize(argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  with maybe_monitor_goodput(config):
    run(config, recorder, diagnostic_config)


if __name__ == "__main__":
  app.run(main)
