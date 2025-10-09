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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, attribute-error
"""
This script implements Group Relative Policy Optimization (GRPO) training
using JAX. It optimizes a language model with reinforcement learning by
updating policy gradients based on reward functions.

The training process involves a producer-consumer pattern:
  - A "producer" thread continuously generates text completions from prompts
    using an offline inference engine. These completions, along with their
    log-probabilities, are stored in a shared buffer.
  - The main "consumer" thread fetches these generated samples from the buffer
    and uses them to perform GRPO training steps.

This decoupling allows the training process (consumer) to proceed without
being blocked by the potentially slower generation process (producer).
The script sets up separate configurations for training and inference,
handles model parameter resharding between the two, and manages the
entire training loop, including checkpointing and metric logging.
"""


import datetime
import time
import os
import functools
import threading

from typing import Sequence, Callable, Iterator

from absl import app

import tensorflow as tf

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

from flax.linen import partitioning as nn_partitioning
from flax import struct
from flax.nnx import TrainState

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

import transformers

from ml_goodput_measurement.src.goodput import GoodputRecorder

import MaxText as mt
from MaxText import checkpointing
from MaxText import exceptions
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText import profiler
from MaxText import pyconfig
from MaxText.checkpointing import CheckpointManager
from MaxText.utils import gcs_utils
from MaxText.inference import offline_engine
from MaxText.data_loader import DataLoader
from MaxText.experimental.rl import grpo_input_pipeline
from MaxText.experimental.rl import grpo_utils
from MaxText.experimental.rl.hooks import GRPOTrainingHooks, GRPODataHooks
from MaxText.globals import EPS
from MaxText.metric_logger import MetricLogger
from MaxText.train import get_first_step
from MaxText.train_utils import validate_train_config
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    create_goodput_recorder,
    maybe_monitor_goodput,
    maybe_record_goodput,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager


# pylint: disable=too-many-positional-arguments

# -----------------------------------------------------------------------------
# GRPO
# -----------------------------------------------------------------------------


def _split_grpo_state(state):
  """Splits the reference parameters from the main training state.

  This is a utility function to separate the reference model's parameters,
  which are kept frozen, from the policy model's parameters that are actively
  being trained.

  Args:
    state: The combined training state, expected to contain a 'reference_params'
      key within its `params` attribute.

  Returns:
    A tuple containing:
      - new_state: The training state with 'reference_params' removed.
      - reference_params: The extracted reference parameters.
  """
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params


def _merge_grpo_state(state, reference_params):
  """Merges the reference parameters back into the training state.

  This is the inverse operation of `_split_grpo_state`, used to reconstruct
  the full state object after a training step.

  Args:
    state: The training state, without 'reference_params'.
    reference_params: The frozen reference parameters to be added back.

  Returns:
    A new state object with the 'reference_params' re-integrated.
  """
  return state.replace(params=dict(state.params, reference_params=reference_params))


@struct.dataclass
class LossAux:
  """A dataclass to hold auxiliary outputs from the GRPO loss function.

  This structure is used to pass various metrics and intermediate values
  from the loss function for logging and analysis.

  Attributes:
    total_loss: The final computed loss value for the step.
    avg_reward: The average reward obtained across the batch.
    avg_reward_std: The standard deviation of rewards within the batch.
    avg_advantage: The average normalized advantage across the batch.
    avg_kl: The average KL divergence between the policy and reference models.
    completion_length: The average token length of the generated completions.
    moe_lb_loss: The load balancing loss for Mixture-of-Experts layers.
    total_weights: The total number of tokens used to normalize the loss.
  """

  total_loss: float
  avg_reward: float
  avg_reward_std: float
  avg_advantage: float
  avg_kl: float
  completion_length: float
  moe_lb_loss: float
  total_weights: float


def grpo_loss_fn(model, config, data, dropout_rng, params, reference_params, is_train=True):
  """
  GRPO loss function for training.

  This function performs the following steps:

    1. Compute the per-token log-probabilities for the full sequence (prompt + completion) both with
         the current model (policy) and the reference model.
    2. Compute a scalar reward for each generated completion via reward_fn.
    3. Group the rewards (each prompt yields “G = num_generations” completions), compute the mean and std,
       and then compute a normalized advantage.
    4. Compute a per-token loss that is given by
        - [min(exp(policy_logp - old_logp), clip(exp(policy_logp - old_logp), 1-e, 1+e)) * advantage - beta * kl
        Where:
        - `policy_logp`: The log probability of the current policy's output.
        - `old_logp`: The log probability of the behavior policy's output
          (i.e., the policy used to generate the samples).
        - For on-policy training, `old_logp` is a stop-gradient of the current `policy_logp`,
          ensuring that only the advantage term contributes to the gradients. This is because
          the samples are generated using the current policy.
        - For off-policy training, `old_logp` is obtained directly from the
          `data["completions_logprobs"]`, which stores the log probabilities
          from the behavior policy that generated the samples.
        - `advantage`: The advantage, representing how much better a given
          action is compared to the average action.
        - `beta`: A hyperparameter that controls the strength of the KL divergence penalty.
        - `kl_divergence`: The Kullback-Leibler divergence between the current
          policy and the behavior policy.
    5. Compute a per-token KL divergence:
         kl = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1.
    6. Restrict the loss calculations to the generated completion tokens.
    7. Finally the loss is the average (over examples) of the mean per-token loss - where only tokens before the
       first eos (according to tokenizer.eos_id) are taken into account.

  Args:
    model: A nn.Module.
    config: The training configuration (contains hyper-parameters and reward and tokenizer objects).
    data: A batch dict with key "prompt" containing prompts as token-ids of shape [B, L_prompt].
    dropout_rng: a PRNGKey.
    params: The current model parameters.
    reference_params: The reference model parameters.
    is_train: Boolean indicating training mode.

  Returns:
    loss: A scalar loss.
    aux: A dictionary with auxiliary metrics.
  """

  # completions shape: [B x G, max_target_length - max_prefill_length]
  # this includes the completion tokens + padding (upto max_target_length - max_prefill_length))
  # data["ar_completions"] contains tokens only upto the eos, no tokens thereafter other than pad_tokens
  prompt_with_completions = data[f"{config.train_data_columns}_completions"]

  # --- (1) Compute per-token log probabilities.
  prompt_completions_position = data[f"{config.train_data_columns}_completions_position"]
  prompt_completions_segmentation = data[f"{config.train_data_columns}_completions_segmentation"]
  completions_segmentation = data["ar_completions_segmentation"]

  # compute_log_probs returns logits.
  # We compute the log-probabilities for the entire generated sequence, then shift as usual.
  rng1, rng_fwd = random.split(dropout_rng)
  token_logps_policy, intermediate_outputs = grpo_utils.compute_log_probs(
      model,
      params,
      prompt_with_completions,
      prompt_completions_position,
      prompt_completions_segmentation,
      completions_segmentation,
      config,
      is_train=is_train,
      rngs={"dropout": rng1, "params": rng_fwd},
  )  # [BxG,S-1,E]

  completion_target_segmentation = data["ar_completions_segmentation"][..., 1:]  # [BxG,S-1]
  # Because of the shifting, token_logps have shape [BxG, S-1]. So, we create a mask for the valid tokens
  # Create a mask to clear out the last token position in the ar_completions
  # and to make sure loss is computed on non-padding tokens
  valid_seq_mask = completion_target_segmentation != 0  # [BxG, S-1]

  # --- (2) Compute a scalar reward for each generated completion via reward_fn.
  rewards = grpo_utils.dummy_reward_len(valid_seq_mask)
  rewards = jnp.array(rewards)  # shape [BxG]

  # --- (3) Group rewards and compute normalized advantage.
  G = config.num_generations
  rewards_grouped = rewards.reshape(-1, G)  # shape [B, G]
  group_mean = jnp.mean(rewards_grouped, axis=1)  # shape [B]
  group_std = jnp.std(rewards_grouped, axis=1)  # shape [B]
  repeated_group_mean = jnp.repeat(group_mean, G)  # shape [BxG]
  repeated_group_std = jnp.repeat(group_std, G)  # shape [BxG]
  advantages = (rewards - repeated_group_mean) / (repeated_group_std + EPS)  # shape [BxG]

  # --- (4) Compute per-token loss.
  # Make sure to expand advantage along the token dimension.
  advantages_exp = advantages[:, None]  # shape [BxG, 1]

  # We calculate the policy difference with old_per_token_logps for off-policy training,
  # else, for on-policy training old_per_token_logps = stop_gradient(token_logps_policy)
  if data["completions_logprobs"] is None:  # off-policy
    old_per_token_logps = jax.lax.stop_gradient(token_logps_policy)
  else:  # on-policy
    old_per_token_logps = data["completions_logprobs"]

  policy_diff = token_logps_policy - old_per_token_logps
  coef_1 = jnp.exp(policy_diff)
  coef_2 = jnp.clip(coef_1, 1 - config.grpo_epsilon, 1 + config.grpo_epsilon)
  loss_tokens = -jnp.minimum(
      coef_1 * advantages_exp,
      coef_2 * advantages_exp,
  )

  # --- (5) Compute per-token KL divergence for each token in the generated completion, if beta != 0.
  if config.grpo_beta != 0.0:
    token_logps_ref, _ = grpo_utils.compute_log_probs(
        model,
        {"params": reference_params},
        prompt_with_completions,
        prompt_completions_position,
        prompt_completions_segmentation,
        completions_segmentation,
        config,
        is_train=False,
        rngs={"dropout": rng1, "params": rng_fwd},
    )  # [BxG,S-1,E]

    token_diff_logps_ref_policy = token_logps_ref - token_logps_policy

    per_token_kl = jnp.exp(token_diff_logps_ref_policy) - (token_diff_logps_ref_policy) - 1
    # loss is computed on non-padding tokens
    per_token_kl = per_token_kl * valid_seq_mask
    loss_tokens += config.grpo_beta * per_token_kl

  # --- (6) Restrict the loss calculations to the generated completion tokens.
  # Average over tokens per generated completion.
  loss_per_example = jnp.sum(loss_tokens * valid_seq_mask, axis=1) / jnp.clip(jnp.sum(valid_seq_mask, axis=1), min=1)

  # --- (7) Finally the loss is the average (over examples) of the mean per-token loss
  loss = jnp.mean(loss_per_example)
  total_weights = jnp.sum(valid_seq_mask)

  moe_lb_loss = 0.0
  if config.num_experts > 1:
    nested_key = ("intermediates", "decoder", "layers", "moe_lb_loss")
    total_moe_lb_loss = maxtext_utils.get_nested_value(intermediate_outputs, nested_key, 0.0)
    moe_lb_loss = jnp.mean(jnp.array(total_moe_lb_loss))
    loss += moe_lb_loss

  # Compute auxiliary metrics.
  if config.grpo_beta != 0.0:
    avg_kl = jnp.mean((per_token_kl * valid_seq_mask) / jnp.clip(jnp.sum(valid_seq_mask, axis=1, keepdims=True), min=1))
  else:
    avg_kl = None
  avg_reward = jnp.mean(rewards)
  avg_advantage = jnp.mean(advantages)
  avg_completion_length = jnp.mean(jnp.sum(data["ar_completions_segmentation"] != 0, axis=1))
  aux = LossAux(
      total_loss=loss,
      avg_reward=avg_reward,
      avg_reward_std=jnp.mean(repeated_group_std),
      avg_advantage=avg_advantage,
      avg_kl=avg_kl,
      completion_length=avg_completion_length,
      moe_lb_loss=moe_lb_loss,
      total_weights=total_weights,
  )

  return loss, aux


# -----------------------------------------------------------------------------
# Trainer and top level training functions
# -----------------------------------------------------------------------------


def train_step(model, config, state_mesh_shardings, state, data, dropout_rng):
  """Performs a single training step of the GRPO algorithm.

  This function computes the GRPO loss, calculates gradients, and updates the
  model's parameters. It handles gradient accumulation and clipping as configured.
  The reference model's parameters are held constant during the update.

  Args:
    model: The transformer model to be trained.
    config: The training configuration object.
    state_mesh_shardings: Pytree of sharding specifications for the training state.
    state: The current training state, including parameters and optimizer state.
    data: A batch of training data, including prompts and generated completions.
    dropout_rng: JAX PRNG key for dropout.

  Returns:
    A tuple containing:
      - new_state: The updated training state after applying gradients.
      - metrics: A dictionary of metrics for logging, including loss, reward,
        and gradient norms.
  """
  state, reference_params = _split_grpo_state(state)
  state_mesh_shardings, reference_params_sharding = _split_grpo_state(state_mesh_shardings)
  extra_grpo_args = [reference_params]
  _loss_fn = grpo_loss_fn

  if config.gradient_accumulation_steps > 1:

    def accumulate_gradient(acc_grad_and_loss, data):
      grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
      (_, aux), cur_batch_gradient = grad_func(
          model, config, data, dropout_rng, state.params, *extra_grpo_args, is_train=True
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
    aux = jax.tree.map(lambda x: jnp.sum(x, axis=0), aux)
  else:
    if config.optimizer_memory_host_offload:
      cast_params = jax.device_put(state.params, max_utils.with_memory_kind(state_mesh_shardings.params, "device"))
      cast_params = max_utils.cast_to_bf16(cast_params)
      state = state.replace(params=cast_params)
      if config.use_grpo:
        reference_params = jax.device_put(
            reference_params, max_utils.with_memory_kind(reference_params_sharding, "device")
        )
        reference_params = max_utils.cast_to_bf16(reference_params)
        extra_grpo_args = [reference_params]
    grad_func = jax.value_and_grad(_loss_fn, argnums=4, has_aux=True)
    (loss, aux), raw_grads = grad_func(model, config, data, dropout_rng, state.params, *extra_grpo_args, is_train=True)

  total_weights = aux.total_weights
  moe_lb_loss = aux.moe_lb_loss

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
      "learning/avg_reward": aux.avg_reward,
      "learning/avg_reward_std": aux.avg_reward_std,
      "learning/avg_advantage": aux.avg_advantage,
      "learning/avg_kl": aux.avg_kl,
      "learning/completion_length": aux.completion_length,
      "learning/moe_lb_loss": moe_lb_loss,
      "learning/total_weights": total_weights,
  }
  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)
    scalar_metrics["learning/avg_reward"] = aux.avg_reward
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }

  new_state = _merge_grpo_state(new_state, reference_params)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  """Performs a single evaluation step.

  This function computes the loss and other metrics on an evaluation dataset
  without performing backpropagation or updating model parameters.

  Args:
    model: The transformer model.
    config: The training configuration object.
    state: The current training state.
    data: A batch of evaluation data.
    dropout_rng: JAX PRNG key for dropout.

  Returns:
    A dictionary of evaluation metrics.
  """

  reference_params, extra_grpo_args, _loss_fn = [], [], grpo_loss_fn
  state, reference_params = _split_grpo_state(state)
  extra_grpo_args = [reference_params]
  _loss_fn = grpo_loss_fn

  eval_loss_fn = functools.partial(_loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(state.params, *extra_grpo_args)
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
    metrics["scalar"]["evaluation/grpo_reward_accuracy"] = aux["reward_accuracy"]

  return metrics


def setup_train_loop(
    config,
    config_inference,
    recorder: GoodputRecorder,
) -> tuple[
    jax.Array,
    CheckpointManager,
    TrainState,
    TrainState,
    mt.Transformer,
    mt.Transformer,
    mt.Mesh,
    mt.Mesh,
    Callable[[int], float],
    Iterator,
    Iterator,
    TrainState,
]:
  """Initializes objects needed for the training loop.

  This function sets up the training and inference meshes, models, optimizers,
  learning rate schedules, and checkpoint managers. It also initializes the
  training state.

  Args:
    config: The main training configuration object.
    config_inference: The configuration object for the inference engine.
    recorder: A GoodputRecorder for performance tracking.

  Returns:
    A tuple containing:
      - init_rng: The initial JAX PRNG key.
      - checkpoint_manager: The Orbax checkpoint manager.
      - state_mesh_shardings: Sharding specifications for the training state.
      - inference_state_mesh_shardings: Sharding specs for the inference state.
      - model: The training model instance.
      - inference_model: The inference model instance.
      - mesh: The device mesh for training.
      - inference_mesh: The device mesh for inference.
      - learning_rate_schedule: The learning rate schedule function.
      - data_iterator: The iterator for the input prompt dataset.
      - eval_data_iterator: The iterator for the evaluation dataset (or None).
      - state: The initialized training state.
  """
  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    max_logging.log("Training mesh used for the workload")
    num_inference_devices = config.inference_devices_per_replica * config.inference_replicas
    training_devices = jax.devices()[num_inference_devices:]
    model = mt.from_config(config, devices=training_devices)
    mesh = model.mesh
    max_logging.log("Inference mesh used for the workload")
    inference_devices = jax.devices()[:num_inference_devices]
    inference_model = mt.from_config(config_inference, devices=inference_devices)
    inference_mesh = inference_model.mesh
    init_rng, checkpoint_manager, learning_rate_schedule, tx = train_utils.create_training_tools(config, model, mesh)

  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    data_iterator = grpo_input_pipeline.create_data_iterator(config_inference, inference_mesh)
    state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
        model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
    )

  # create inference_state_mesh_shardings from inference_mesh
  inference_state_mesh_shardings = maxtext_utils.get_abstract_state(
      inference_model, tx, config_inference, init_rng, inference_mesh, is_training=False
  )[2]
  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

  return (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      inference_state_mesh_shardings,
      model,
      inference_model,
      mesh,
      inference_mesh,
      learning_rate_schedule,
      data_iterator,
      iter(()),  # GRPO does not support eval_dataset
      state,
  )


def generate_completions(
    worker_data_loader,
    worker_inference_engine,
    worker_tokenizer_model,
    worker_config_inference,
    worker_config_train,
    worker_data_buffer,
    worker_data_buffer_lock,
    worker_input_data_shardings,
    engine_lock,
):
  """Loads a batch of prompts and generates completions using the inference engine.

  This function is designed to be run by the generation worker thread. It loads
  a batch of prompts from the data loader, uses the offline inference engine to
  generate multiple completions for each prompt, processes the results, and adds
  them to a shared data buffer for the training loop to consume.

  Args:
    worker_data_loader: The DataLoader instance for fetching prompt data.
    worker_inference_engine: The offline inference engine for generation.
    worker_tokenizer_model: The tokenizer model.
    worker_config_inference: The configuration for the inference process.
    worker_config_train: The main training configuration.
    worker_data_buffer: A list acting as a shared buffer to store generated data.
    worker_data_buffer_lock: A lock to ensure thread-safe access to the buffer.
    worker_input_data_shardings: Sharding specifications for the data.
    engine_lock: A lock to ensure thread-safe use of the inference engine.
  """
  with engine_lock:
    thread_example_batch = worker_data_loader.load_next_batch()
    # Trim data for inference processing
    thread_example_batch_trimmed = jax.tree_util.tree_map(
        lambda arr: arr[
            : int(
                worker_config_inference.per_device_batch_size
                * worker_config_train.inference_replicas
                * worker_config_train.inference_devices_per_replica
            )
        ],
        thread_example_batch,
    )
    processed_batch = grpo_utils.generate_offline_completions(
        worker_config_inference, worker_tokenizer_model, worker_inference_engine, thread_example_batch_trimmed
    )
    processed_batch = jax.device_put(processed_batch, worker_input_data_shardings)
  with worker_data_buffer_lock:
    if not worker_data_buffer:
      worker_data_buffer.append(processed_batch)
    else:
      worker_data_buffer[0] = jax.tree_util.tree_map(
          lambda a, b: np.concatenate([a, b], axis=0), worker_data_buffer[0], processed_batch
      )


def train_loop(config, config_inference, recorder, state=None):
  """The main GRPO training loop.

  This function orchestrates the entire training process. It initializes the
  necessary components, starts a background thread for continuous data generation,
  and then enters a loop to perform training steps.

  The loop consists of:
  1. Fetching pre-generated prompt-completion pairs from a shared buffer.
  2. Executing a training step with the fetched batch.
  3. Periodically resharding the updated policy parameters to the inference engine.
  4. Logging metrics and saving checkpoints.
  5. Handling evaluation if configured.

  The loop continues for the configured number of steps and manages the lifecycle
  of the generation worker thread.

  Args:
    config: The main training configuration object.
    config_inference: The configuration for the inference engine.
    recorder: A GoodputRecorder for performance tracking.
    state: An optional pre-existing training state to resume from.

  Returns:
    The final training state after the loop completes.
  """

  (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      inference_state_mesh_shardings,
      model,
      _,  # inference_model
      mesh,
      inference_mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  ) = setup_train_loop(config, config_inference, recorder)
  tokenizer_model = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      add_bos_token=config.add_bos,
      add_eos_token=config.add_eos,
      model_max_length=config.max_target_length,
      legacy=False,
      token=config.hf_access_token,
  )

  if "reference_params" not in state.params:
    reference_params = jax.tree.map(jnp.copy, state.params["params"])
    state = _merge_grpo_state(state, reference_params)
  state_mesh_shardings = _merge_grpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])

  p_train_step, p_eval_step = train_utils.jit_train_and_eval_step(
      config, model, mesh, state, state_mesh_shardings, train_step, eval_step, eval_data_iterator
  )

  data_sharding = maxtext_utils.get_input_data_sharding(config, mesh)

  inference_engine = offline_engine.OfflineEngine(
      config=config_inference,
      mesh=inference_mesh,
  )
  data_buffer = []
  data_buffer_lock = threading.Lock()

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  metric_logger = MetricLogger(config=config, learning_rate_schedule=learning_rate_schedule)

  # Initialize GRPO training hooks
  training_hooks = GRPOTrainingHooks(
      config=config, mesh=mesh, learning_rate_schedule=learning_rate_schedule, goodput_recorder=recorder
  )

  # Initialize GRPO data hooks with multi-host data pipeline
  # This replaces the old DataLoader with improved multi-host data loading
  data_hooks = GRPODataHooks(config=config, mesh=mesh, goodput_recorder=recorder)

  # Use the data_hooks' train_data_loader for loading prompts
  data_loader = data_hooks.train_data_loader

  # Write train config params, num model params, and XLA flags to tensorboard
  metric_logger.write_setup_info_to_tensorboard(state.params["params"])

  # Call on_train_start hook
  training_hooks.on_train_start(state, start_step)

  def generation_worker_fn(
      worker_inference_engine,
      worker_tokenizer_model,
      worker_config_inference,
      worker_config_train,
      worker_data_buffer,
      worker_data_buffer_lock,
      worker_input_data_shardings,
      engine_lock,
      stop_event,
  ):
    """The target function for the data generation worker thread.

    This function runs in a loop, continuously calling `generate_completions`
    to populate the shared data buffer. It stops when the `stop_event` is set
    by the main thread.

    Args:
      worker_inference_engine: The offline inference engine.
      worker_tokenizer_model: The tokenizer model.
      worker_config_inference: The inference configuration.
      worker_config_train: The training configuration.
      worker_data_buffer: The shared list used as a data buffer.
      worker_data_buffer_lock: A lock for thread-safe buffer access.
      worker_input_data_shardings: Sharding specs for the generated data.
      engine_lock: A lock for thread-safe inference engine access.
      stop_event: A threading.Event to signal when the worker should stop.
    """
    while not stop_event.is_set():
      try:
        with jax.profiler.StepTraceAnnotation("inference"):
          generate_completions(
              data_loader,
              worker_inference_engine,
              worker_tokenizer_model,
              worker_config_inference,
              worker_config_train,
              worker_data_buffer,
              worker_data_buffer_lock,
              worker_input_data_shardings,
              engine_lock,
          )
      except StopIteration:
        max_logging.log("Data iterator exhausted in generation worker. Stopping.")
        break
      except Exception as e:  # pylint: disable=broad-except
        max_logging.log(f"Error in generation worker: {e}")
        break
    max_logging.log("Generation worker thread finished.")

  stop_event = threading.Event()
  inference_engine_lock = threading.Lock()

  max_logging.log("Inference Rollout")
  # Track initial generation
  training_hooks.on_generation_start(start_step)
  gen_start_time = time.time()
  generate_completions(
      data_loader,
      inference_engine,
      tokenizer_model,
      config_inference,
      config,
      data_buffer,
      data_buffer_lock,
      data_sharding,
      inference_engine_lock,
  )
  gen_time = time.time() - gen_start_time
  with data_buffer_lock:
    num_completions = sum(batch[config.train_data_columns].shape[0] for batch in data_buffer)
  training_hooks.on_generation_end(start_step, num_completions, gen_time)

  required_batch_size = int(config.per_device_batch_size * config.num_generations * mesh.size)
  generation_thread = threading.Thread(
      target=generation_worker_fn,
      args=(
          inference_engine,  # Shared inference engine
          tokenizer_model,
          config_inference,
          config,  # Main config for load_next_batch
          data_buffer,
          data_buffer_lock,
          data_sharding,  # Sharding for the data put into the buffer
          inference_engine_lock,
          stop_event,
      ),
      daemon=True,  # So it exits when the main thread exits
  )
  generation_thread.start()

  try:
    last_step_completion = datetime.datetime.now()
    step = start_step  # Initialize step variable
    for step in np.arange(start_step, config.steps):
      # Call on_train_step_start hook
      training_hooks.on_train_step_start(step)

      prof.maybe_activate_profiler(step, state)

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        while True:
          with data_buffer_lock:
            if not data_buffer and not generation_thread.is_alive():
              max_logging.log("Generation worker is not alive and data buffer is empty. Exiting.")
              break
            if data_buffer:
              example_batch = data_buffer[0]
              if example_batch[config.train_data_columns].shape[0] >= required_batch_size:
                example_batch = jax.tree_util.tree_map(lambda arr: arr[:required_batch_size], data_buffer[0])
                data_buffer[0] = jax.tree_util.tree_map(lambda arr: arr[required_batch_size:], data_buffer[0])
                break
              else:
                time.sleep(0.1)
                continue
            else:
              time.sleep(0.1)
              continue
        train_rng, rng = random.split(init_rng)
        example_batch = jax.device_put(example_batch, data_sharding)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          state, metrics = p_train_step(state, example_batch, train_rng)
      with jax.profiler.StepTraceAnnotation("transfer data", step_num=step):
        if step != 0 and step % config.inference_rollouts == 0:
          grpo_utils.pathways_reshard(
              config_inference,
              inference_engine,
              {"params": state.params["params"]},
              {"params": state_mesh_shardings.params["params"]},
              mesh,
              inference_state_mesh_shardings,
          )

      step_time_delta = datetime.datetime.now() - last_step_completion
      last_step_completion = datetime.datetime.now()

      state_to_save = _split_grpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator, step)
      # Note: maybe_save_checkpoint doesn't return a value, so we check if it's a checkpoint step
      if step % config.checkpoint_period == 0:
        training_hooks.on_checkpoint_save(step, config.checkpoint_dir)

      if config.dump_hlo and step == start_step:
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
        # Call on_eval_start hook
        training_hooks.on_eval_start(step)
        eval_step_count = 0
        # pylint: disable=not-callable
        for eval_batch in eval_data_iterator:
          if 0 < config.eval_steps <= eval_step_count:
            break
          with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            eval_metrics = p_eval_step(state, eval_batch, rng)
          # Call on_eval_step hook
          training_hooks.on_eval_step(eval_metrics)
          metric_logger.record_eval_metrics(step, metrics=eval_metrics)
          max_logging.log(f"Completed eval step {eval_step_count}")
          eval_step_count += 1
        metric_logger.record_eval_metrics(step, eval_step_count=eval_step_count)
        # Call on_eval_end hook
        training_hooks.on_eval_end(step)
        if metric_logger.cumulative_eval_metrics["scalar"]["eval/avg_loss"] <= config.target_eval_loss:
          prof.deactivate()
          raise exceptions.StopTraining(f"Target loss {config.target_eval_loss=} is achieved.")

      prof.maybe_deactivate_profiler(step, state)

      if step == start_step:
        max_utils.print_mem_stats("After params initialized")

      metric_logger.buffer_and_write_train_metrics(metrics, step, step_time_delta)

      # Call on_train_step_end hook
      training_hooks.on_train_step_end(step, metrics, step_time_delta.total_seconds())

      state_to_save = _split_grpo_state(state)[0]
      checkpointing.maybe_save_checkpoint(checkpoint_manager, state_to_save, config, data_iterator)
  except exceptions.StopTraining as e:
    max_logging.log(f"Training stopped: {str(e)}")
  finally:
    # Call on_train_end hook
    training_hooks.on_train_end(step)
    metric_logger.flush_metrics_and_cleanup()
    max_logging.log("Training loop finished or exited. Signaling generation worker to stop.")
    stop_event.set()
    # Wait for the generation thread to finish
    generation_thread.join(timeout=60.0)  # Increased timeout
    if generation_thread.is_alive():
      max_logging.log("Warning: Generation worker did not stop in time after loop completion.")

  return state


def main(argv: Sequence[str]) -> None:
  """Main entry point for the GRPO training script.

  This function parses command-line arguments, initializes configurations for
  training and inference, sets up system environment variables, and launches
  the `train_loop`.
  """
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
  configs_argv = max_utils.parse_custom_args(argv)
  config = pyconfig.initialize(configs_argv[0])
  if not config.use_grpo:
    raise ValueError("Please set the value of use_grpo to True")
  if config.inference_rollouts < 1 or config.inference_rollouts > config.steps:
    raise ValueError(
        f"Please set the value of inference_rollouts to be less than {config.steps} or greater than 1. "
        f"Current value: {config.inference_rollouts}"
    )
  if config.decode_sampling_strategy == "greedy" or config.decode_sampling_temperature == 0.0:
    raise ValueError(
        "Please set decode_sampling_strategy as 'weighted' and decode_sampling_temperature as a positive number"
    )
  if config.inference_devices_per_replica * config.inference_replicas >= jax.device_count():
    raise ValueError(
        f"Invalid value chosen for {config.inference_devices_per_replica=} and {config.inference_replicas=} "
        f"with {jax.device_count()} devices"
    )
  config_inference = pyconfig.initialize(configs_argv[1])
  if config.per_device_batch_size < 1.0 or config_inference.per_device_batch_size < 1.0:
    raise ValueError("GRPO does not support setting per_device_batch_size < 1.0")
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
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

  with diagnostic.diagnose(diagnostic_config):
    with maybe_record_goodput(recorder, GoodputEvent.JOB):
      train_loop(config, config_inference, recorder)


if __name__ == "__main__":
  app.run(main)
