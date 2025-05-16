"""
Copyright 2025 Google LLC

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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, attribute-error
"""
This script implements Group Relative Policy Optimization (GRPO) training
using JAX. It optimizes a language model with reinforcement learning by
updating policy gradients based on reward functions
"""


import datetime
import os
import sys
import functools
import queue
from typing import Sequence
from collections.abc import Callable

from absl import app

import tensorflow as tf

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

from flax.linen import partitioning as nn_partitioning
from flax import struct

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from ml_goodput_measurement import monitoring

import transformers

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxengine
from MaxText import maxtext_utils
from MaxText import profiler
from MaxText import pyconfig
from MaxText.common_types import Array
from MaxText.experimental.rl import grpo_input_pipeline
from MaxText.gcp_workload_monitor import GCPWorkloadMonitor
from MaxText.layers import models
from MaxText.metric_logger import MetricLogger
from MaxText.train import (
    validate_train_config,
    get_first_step,
    load_next_batch,
    record_scalar_metrics,
    save_checkpoint,
    record_goodput,
    create_goodput_recorder,
    check_example_batch,
    setup_mesh_and_model,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager

# pylint: disable=too-many-positional-arguments

Transformer = models.Transformer
EPS = 1e-8


# -----------------------------------------------------------------------------
# GRPO
# -----------------------------------------------------------------------------


def _split_grpo_state(state):
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params


def _merge_grpo_state(state, reference_params):
  return state.replace(params=dict(state.params, reference_params=reference_params))


@struct.dataclass
class LossAux:
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
    2. Compute a per-token KL divergence:
         kl = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1.
    3. Compute a scalar reward for each generated completion via reward_fn.
    4. Group the rewards (each prompt yields “G = num_generations” completions), compute the mean and std,
       and then compute a normalized advantage.
    5. Compute a per-token loss that is given by
         - [exp(policy_logp - stop_gradient(policy_logp)) * advantage - beta * kl]
       (the jax.lax.stop_gradient ensures that only the advantage contributes to gradients).
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
  token_logps_policy, intermediate_outputs = compute_log_probs(
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

  token_logps_ref, _ = compute_log_probs(
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

  completion_target_segmentation = data["ar_completions_segmentation"][..., 1:]  # [BxG,S-1]
  # Because of the shifting, token_logps have shape [BxG, S-1]. So, we create a mask for the valid tokens
  # Create a mask to clear out the last token position in the ar_completions
  # and to make sure loss is computed on non-padding tokens
  valid_seq_mask = completion_target_segmentation != 0  # [BxG, S-1]

  # --- (2) Compute per-token KL divergence for each token in the generated completion.
  token_diff_logps_ref_policy = token_logps_ref - token_logps_policy

  per_token_kl = jnp.exp(token_diff_logps_ref_policy) - (token_diff_logps_ref_policy) - 1
  # loss is computed on non-padding tokens
  per_token_kl = per_token_kl * valid_seq_mask

  # --- (3) Compute a scalar reward for each generated completion via reward_fn.
  rewards = dummy_reward_len(valid_seq_mask)
  rewards = jnp.array(rewards)  # shape [BxG]

  # --- (4) Group rewards and compute normalized advantage.
  G = config.num_generations
  rewards_grouped = rewards.reshape(-1, G)  # shape [B, G]
  group_mean = jnp.mean(rewards_grouped, axis=1)  # shape [B]
  group_std = jnp.std(rewards_grouped, axis=1)  # shape [B]
  repeated_group_mean = jnp.repeat(group_mean, G)  # shape [BxG]
  repeated_group_std = jnp.repeat(group_std, G)  # shape [BxG]
  advantages = (rewards - repeated_group_mean) / (repeated_group_std + EPS)  # shape [BxG]

  # --- (5) Compute per-token loss.
  # We follow the TRL GRPO loss:
  #   loss_token = - [ exp(policy_logp - stop_gradient(policy_logp)) * advantage - beta * kl ]
  # Make sure to expand advantage along the token dimension.
  advantages_exp = advantages[:, None]  # shape [BxG, 1]

  policy_diff = token_logps_policy - jax.lax.stop_gradient(token_logps_policy)
  loss_tokens = -(jnp.exp(policy_diff) * advantages_exp - config.grpo_beta * per_token_kl)

  # --- (6) Restrict the loss calculations to the generated completion tokens.
  # Average over tokens per generated completion.
  loss_per_example = jnp.sum(loss_tokens * valid_seq_mask, axis=1) / (jnp.sum(valid_seq_mask, axis=1) + EPS)

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
  avg_kl = jnp.mean((per_token_kl * valid_seq_mask) / (jnp.sum(valid_seq_mask, axis=1, keepdims=True) + EPS))
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


# --- GRPO Helpers ---


def prefill(engine, params, prompts, true_length, num_generations, decode_state, rng):
  """
  Args:
    engine: The generation engine instance responsible for managing decoding and inference.
    params: Model parameters used for generating logits during inference.
    decode_state: Current decoding state, which maintains token positions, masks, and cached states.
    data: Input batch containing prompt tokens, segementation and position.
    num_generations: Number of completions to generate per prompt (G in many RLHF-style pipelines).
    rng: JAX PRNG key for controlling stochastic behavior (e.g., sampling, dropout).

  Returns:
    decode_state: Updated decode state after prefill, including new cached key/value pairs.
    prefill_slots: A structure containing token positions and model states needed for next-stage decoding.
  """
  """
  JIT-compatible version of prefill using a single lax.scan over the repeated batch.
  """
  # Repeat each prompt `num_generations` times
  repeated_prompts = jnp.repeat(prompts, num_generations, axis=0)
  repeated_true_length = jnp.repeat(true_length, num_generations, axis=0)

  def _scan_prefill_step(carry, inputs):
    decode_state, rng, slot = carry
    tokens, true_len = inputs
    rng, rng_prefill = jax.random.split(rng)
    prefill_result, _ = engine.prefill(params=params, padded_tokens=tokens, true_length=true_len, rng=rng_prefill)
    decode_state = engine.insert(prefill_result, decode_state, slot)
    return (decode_state, rng, slot + 1), None

  (decode_state, _, _), _ = jax.lax.scan(
      _scan_prefill_step, init=(decode_state, rng, 0), xs=(repeated_prompts, repeated_true_length)
  )

  return decode_state


def generate(engine, params, num_decode_steps, decode_state, rng):
  """
  Args:
    engine: The generation engine instance used to run autoregressive decoding.
    params: Model parameters used to compute logits during token generation.
    decode_state: The current decode state containing cached attention key/value pairs and positions.
    num_decode_steps: Number of decoding steps to perform (i.e., target length - prefill length).
    rng: JAX PRNG key used for sampling, top-k/top-p filtering, or any stochastic decoding behavior.

  Returns:
    completions: Generated sequences (e.g., token IDs) of shape [num_prompts * num_generations, num_decode_steps].
  """

  def _scan_generate_step(carry, _):
    rng, decode_state = carry
    rng, rng_generate = jax.random.split(rng)
    decode_state, result_tokens = engine.generate(params, decode_state, rng=rng_generate)
    return (rng, decode_state), result_tokens.data[:, 0]

  (_, all_tokens) = jax.lax.scan(_scan_generate_step, init=(rng, decode_state), xs=None, length=num_decode_steps)
  return jnp.transpose(all_tokens, (1, 0))


def concatenate_prompt_with_completions(config, tokenizer_model, prompts, true_length, completions):
  """
  Args:
    config: Configuration object containing generation settings such as max sequence length or EOS token ID.
    tokenizer_model: Tokenizer used to decode or manipulate tokens (e.g., identifying special tokens like EOS).
    data: Input batch containing prompt tokens, segementation and position.
    completions: Generated token sequences to be appended to the corresponding prompts.

  Returns:
    prompt_completions: Concatenated sequences of prompt + generated completions for each sample.
    eos_positions: Indices indicating the position of the first EOS token in each concatenated sequence.
  """

  def _concat_and_find_eos(prompt, true_len, completion):
    total_len = prompt.shape[0] + completion.shape[0]
    prompt_mask = jnp.arange(prompt.shape[0]) < true_len
    trimmed_prompt = jnp.where(prompt_mask, prompt, 0)

    # Initialize with padded prompt
    full_seq = jnp.zeros((total_len,), dtype=prompt.dtype)
    full_seq = full_seq.at[: prompt.shape[0]].set(trimmed_prompt)

    # Dynamically insert completion at true_len position
    full_seq = jax.lax.dynamic_update_slice(full_seq, completion, (true_len,))

    # Find EOS index
    eos_mask = full_seq == tokenizer_model.eos_token_id
    eos_indices = jnp.where(eos_mask, jnp.arange(total_len), total_len)
    eos_index = jnp.min(eos_indices)

    return full_seq, eos_index

  batched_concat_and_eos = jax.vmap(_concat_and_find_eos, in_axes=(0, 0, 0))
  prompts = jnp.repeat(prompts, config.num_generations, axis=0)
  true_length = jnp.repeat(true_length, config.num_generations, axis=0)
  prompt_completions, eos_positions = batched_concat_and_eos(prompts, true_length, completions)
  return prompt_completions, eos_positions


def generate_completions(config, tokenizer_model, engine, data, params, rng):
  """
  Autoregressively generates completions for a batch of prompts.

  Args:
    prompts: Array of shape [B, S] containing token ids.
    config: Configuration containing:
         - num_generations: number of completions to generate per prompt.
         - max_completion_length: maximum number of tokens to generate.
         - temperature: sampling temperature.
    rng: JAX PRNGKeys.
    tokenizer_model: Tokenizer for generate

  Returns:
    A jnp.array of shape [B x num_generations, S] where S = length_of_prompt + max_completion_length.
  """
  # decimate proportion of data when per_device_batch_size<1
  for k, v in data.items():
    assert v.ndim in (1, 2), f"Invalid {v.shape=} found for key={k}"
    if v.ndim == 2:
      data[k] = v[: config.micro_batch_size_to_train_on, :]
    else:
      data[k] = v[: config.micro_batch_size_to_train_on]

  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode)

  prompts, true_length = data[f"{config.train_data_columns}"], data[f"{config.train_data_columns}_true_length"]
  rng, rng_prefill = jax.random.split(rng)
  decode_state = jax.jit(
      functools.partial(prefill, engine, params, prompts, true_length, config.num_generations), donate_argnums=(0,)
  )(decode_state, rng_prefill)

  rng, rng_generate = jax.random.split(rng)
  completions = jax.jit(
      functools.partial(generate, engine, params, config.max_target_length - config.max_prefill_predict_length)
  )(decode_state, rng_generate)

  data[f"{config.train_data_columns}_completions"], eos_positions = concatenate_prompt_with_completions(
      config, tokenizer_model, prompts, true_length, completions
  )

  data[f"{config.train_data_columns}_completions_segmentation"] = (
      jnp.arange(data[f"{config.train_data_columns}_completions"].shape[1])[None, :] < eos_positions[:, None]
  ).astype(jnp.int32)
  data[f"{config.train_data_columns}_completions_position"] = jnp.where(
      data[f"{config.train_data_columns}_completions_segmentation"],
      jnp.arange(data[f"{config.train_data_columns}_completions"].shape[1]),
      0,
  )
  true_length = jnp.repeat(true_length, config.num_generations, axis=0)
  completion_mask = data[f"{config.train_data_columns}_completions_position"] >= true_length[:, None] - 1
  data["ar_completions_segmentation"] = data[
      f"{config.train_data_columns}_completions_segmentation"
  ] * completion_mask.astype(jnp.int32)
  return data


def dummy_reward_len(valid_seq_mask):
  # adding a 1 because valid_seq_mask is actually one less than the number of valid tokens
  reward = -abs(20 - (1 + jnp.sum(valid_seq_mask, axis=-1)))  # [BxG]
  return reward


def jaccard_reward_fn(tokens1, tokens2, vocab_size):
  """
  A simple Jaccard similarity for now
  # TODO: Include more reward functions
  """

  # Convert token id arrays to one-hot representations.
  # The result has shape (..., seq_length, vocab_size).
  tokens1_onehot = jax.nn.one_hot(tokens1, vocab_size, dtype=bool)
  tokens2_onehot = jax.nn.one_hot(tokens2, vocab_size, dtype=bool)

  # Reduce along the sequence dimension (axis=-2) to obtain a boolean presence vector
  # for each token id in the vocabulary. This effectively converts each row to a set.
  a_set = jnp.any(tokens1_onehot, axis=-2)
  b_set = jnp.any(tokens2_onehot, axis=-2)

  # Compute the intersection and union along the vocabulary dimension (axis=-1).
  intersection = jnp.sum(jnp.logical_and(a_set, b_set), axis=-1)
  union = jnp.sum(jnp.logical_or(a_set, b_set), axis=-1)

  # Avoid division by zero: if union is 0 (e.g. both rows are empty), return 1.0.
  return jnp.where(union == 0, 1.0, intersection / union)


def compute_log_probs(
    model, params, inputs, inputs_position, inputs_segmentation, completion_segmentation, config, is_train=False, rngs=None
):
  """
  Given a sequence of tokens (shape [B, L]), this helper calls model.apply (with dropout enabled
  if is_train) to obtain logits and then computes per-token log-probabilities.

  Note: We assume that tokens have been already appropriately padded.
  """
  if not is_train:
    params = jax.lax.stop_gradient(params)
  logits, intermediate_outputs = model.apply(
      params,
      inputs,
      inputs_position,
      decoder_segment_ids=inputs_segmentation,
      enable_dropout=(config.enable_dropout if is_train else False),
      rngs=rngs,
      mutable="intermediates",
  )  # [B, S, E] - [batch, sequence, embedding/vocab]
  logits = logits / config.decode_sampling_temperature
  # Remove last time step since there is no target for the final position.
  targets = inputs[:, 1:]
  # Shift left using dynamic slice (skip first column)
  shifted_completion_segmentation = jax.lax.dynamic_slice(
      completion_segmentation, (0, 1), (completion_segmentation.shape[0], completion_segmentation.shape[1] - 1)
  )
  # Pad with 0 at the end to maintain the original shape
  shifted_completion_segmentation = jnp.pad(
      shifted_completion_segmentation, ((0, 0), (0, 1)), mode="constant", constant_values=0
  )

  mask = shifted_completion_segmentation[..., None]
  mask = jnp.broadcast_to(mask, logits.shape)

  masked_logits = jnp.where(mask, logits, -jnp.inf)
  log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
  log_probs = jnp.where(mask, log_probs, -0.0)
  log_probs = log_probs[:, :-1, :]
  # Gather the log probabilities corresponding to each target token.
  token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
  token_log_probs = token_log_probs * shifted_completion_segmentation[:, :-1]

  return token_log_probs, intermediate_outputs


# -----------------------------------------------------------------------------
# Trainer and top level training functions
# -----------------------------------------------------------------------------


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
        reference_params = jax.device_put(reference_params, max_utils.with_memory_kind(reference_params_sharding, "device"))
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
  """eval_step no backprop and new state compared with train_step."""

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
  data_iterator, eval_data_iterator = grpo_input_pipeline.create_data_iterator(config, mesh)
  state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )

  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

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


def train_loop(config, config_inference, state=None):
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

  # pylint: disable=line-too-long
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

  # Initializing maxengine and everything related from decode.py
  # TODO: Creating an engine here but might have two model compilation, need to initialize engine while passing model object
  engine = maxengine.MaxEngine(config_inference)
  init_rng, rng_load_params = jax.random.split(init_rng)
  # TODO: loading parameters from GCS here, need to pass in the same params to engine which already loaded
  _ = engine.load_params(rng_load_params)

  if eval_data_iterator:
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(eval_step, mesh, state_mesh_shardings, model, config)

  # TODO: fix tflops calculations for grpo setting
  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)
  per_device_tokens = maxtext_utils.calculate_tokens_training_per_device(config)

  # Write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  maxtext_utils.add_config_to_summary_writer(config, writer)

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

  data_sharding = in_shard_train[1]
  param_sharding = state_mesh_shardings.params
  p_generate_completions: Callable[[dict, dict, Array], Array] = jax.jit(
      functools.partial(generate_completions, config, tokenizer_model, engine),
      in_shardings=(data_sharding, param_sharding, None),
      out_shardings=data_sharding,
      donate_argnums=(0,),
  )

  running_gcs_metrics = [] if config.gcs_metrics else None

  start_step = get_first_step(state)  # this is the start_step for training
  prof = profiler.Profiler(config, offset_step=start_step)
  first_profiling_step = prof.start_initial_profile_step
  if config.profiler != "" and first_profiling_step >= config.steps:
    raise ValueError("Profiling requested but initial profiling step set past training final step")
  last_profiling_step = prof.finished_initial_profile_step

  example_batch = None
  last_step_completion = datetime.datetime.now()

  performance_metric_queue = None
  if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
    gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
    if config.report_heartbeat_metric_for_gcp_monitoring:
      gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
    if config.report_performance_metric_for_gcp_monitoring:
      performance_metric_queue = queue.Queue()
      gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)

  metric_logger = MetricLogger(writer, config)
  input_data_shardings = maxtext_utils.get_input_data_sharding(config, mesh)
  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step or prof.should_activate_periodic_profile(step):
      optional_postfix = f"step_{step}" if config.profile_periodically_period > 0 else ""
      prof.activate(blocking_object=state, optional_postfix=optional_postfix)

    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      record_goodput(recorder, config, recorder.record_data_loading_start_time if recorder else None)
      example_batch = load_next_batch(data_iterator, example_batch, config)
      example_batch = jax.lax.with_sharding_constraint(example_batch, input_data_shardings)
      record_goodput(recorder, config, recorder.record_data_loading_end_time if recorder else None)
      check_example_batch(config, example_batch=example_batch)
      # pylint: disable=not-callable
      rng = jax.jit(jax.random.fold_in)(init_rng, step)
      record_goodput(recorder, config, recorder.record_step_start_time if recorder else None, step)
      rng, rng_gen = random.split(rng)
      example_batch = p_generate_completions(example_batch, state.params, rng_gen)

      # TODO: ensure this partitioning is correct
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state, metrics = p_train_step(state, example_batch, rng)

    step_time_delta = datetime.datetime.now() - last_step_completion
    last_step_completion = datetime.datetime.now()
    record_scalar_metrics(metrics, step_time_delta, per_device_tflops, learning_rate_schedule(step), per_device_tokens)
    if performance_metric_queue:
      performance_metric_queue.put(step_time_delta.total_seconds())

    if checkpoint_manager is not None:
      state_to_save = state if not config.use_dpo else _split_grpo_state(state)[0]
      if save_checkpoint(checkpoint_manager, int(step), state_to_save, config.dataset_type, data_iterator, config):
        checkpointing.print_save_message(step, config.async_checkpointing)

      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()

    metric_logger.write_metrics(running_gcs_metrics, metrics, step)

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
          eval_metrics = p_eval_step(state, eval_batch, rng)
        cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
        cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
        cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(eval_metrics["scalar"]["evaluation/moe_lb_loss"])
        eval_dpo_reward_accuracy += float(eval_metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0))  # for dpo only
        max_logging.log(f"Completed eval step {eval_step_count}")
        eval_step_count += 1
      eval_loss = cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
          cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
      )
      cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
          cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
      if config.use_dpo:
        cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = eval_dpo_reward_accuracy / eval_step_count
      metric_logger.write_metrics(running_gcs_metrics, cumulative_eval_metrics, step, is_training=False)
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
  metric_logger.write_metrics(running_gcs_metrics, metrics, config.steps - 1)  # final step metrics
  max_utils.close_summary_writer(writer)
  record_goodput(recorder, config, recorder.record_job_end_time if recorder else None)
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    compiled = p_train_step.lower(state, example_batch, rng).compile()
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
  config = pyconfig.initialize(argv)
  if not config.use_grpo:
    raise ValueError("Please set the value of use_grpo to True")
  if config.decode_sampling_strategy == "greedy" or config.decode_sampling_temperature == 0.0:
    raise ValueError(
        "Please set decode_sampling_strategy as 'weighted' and decode_sampling_temperature as a positive number"
    )
  config_inference = pyconfig.initialize(
      list(argv)
      + ["ici_tensor_parallelism=4", "per_device_batch_size=" + str(config.per_device_batch_size * config.num_generations)]
  )
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
    train_loop(config, config_inference)


if __name__ == "__main__":
  app.run(main)
