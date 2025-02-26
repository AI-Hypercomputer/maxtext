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
from collections import defaultdict
import time
import queue

from typing import Sequence, Optional
from absl import app
import tensorflow as tf
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import grain.python as grain
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager


current_dir = os.path.dirname(os.path.abspath(__file__))
maxtext_parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(maxtext_parent_dir)

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import optimizers
import profiler
import pyconfig
import pathwaysutils  # pylint: disable=unused-import

from vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

# from input_pipeline.input_pipeline_interface import create_data_iterator
from grpo_input_pipeline import create_data_iterator
from layers import models

from gcp_workload_monitor import GCPWorkloadMonitor

from train import (validate_train_config,
                   get_first_step, load_next_batch,
                   record_scalar_metrics, save_checkpoint,
                   record_goodput, create_goodput_recorder,
                   check_example_batch, )

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import quantizations

from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring

import maxengine
import transformers

# pylint: disable=too-many-positional-arguments

Transformer = models.Transformer
EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3





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
          #TODO: Do we need this?
          # f"total_weights: {metrics['scalar']['learning/total_weights']}, "
          f"loss: {metrics['scalar']['learning/loss']:.3f}, "
          f"learning/avg_reward: {metrics['scalar']['learning/avg_reward']:.3f}, "
          f"learning/avg_reward_std: {metrics['scalar']['learning/avg_reward_std']:.3f}, "
          f"learning/avg_advantage: {metrics['scalar']['learning/avg_advantage']:.3f}, "
          f"learning/avg_kl: {metrics['scalar']['learning/avg_kl']:.3f}, "
      )

      if full_log and jax.process_index() == 0:
        max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")
        writer.flush()


def clear_buffered_metrics():
  global _buffered_step
  global _buffered_metrics
  _buffered_step = None
  _buffered_metrics = None



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


# -----------------------------------------------------------------------------
# GRPO
# -----------------------------------------------------------------------------


def _split_grpo_state(state):
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params


def _merge_grpo_state(state, reference_params):
  return state.replace(params=dict(state.params, reference_params=reference_params))


@jax.jit
def compute_comp_logps_policy(token_logps_policy: jnp.ndarray, L_prompt: int) -> jnp.ndarray:
    """
    Computes comp_logps_policy using jax.lax.dynamic_slice.

    Args:
        token_logps_policy: A JAX array of shape [batch_size, sequence_length].
        L_prompt: The dynamic starting index for the slice (an integer).

    Returns:
        The sliced array.
    """
    if not isinstance(token_logps_policy, jnp.ndarray):
        raise TypeError("token_logps_policy must be a jnp.ndarray")
    if token_logps_policy.ndim != 2:
        raise ValueError("token_logps_policy must be a 2D array (batch_size, sequence_length)")
    # if not isinstance(L_prompt, int): # L_prompt MUST be a Python int for dynamic_slice
    #     raise TypeError("L_prompt must be a Python integer")

    batch_size, sequence_length = token_logps_policy.shape

    # Calculate the start index along the sequence dimension.
    start_index = jnp.maximum(0, L_prompt - 1)  # Ensure start_index >= 0

    # Calculate the slice size along the sequence dimension.
    slice_size = jnp.maximum(0, sequence_length - start_index) # Ensure non-negative size

    # Create a 1D array for start_indices.
    start_indices = (0, start_index)

    # Create a 1D array for slice_sizes using dynamic batch size.
    slice_sizes = (batch_size, slice_size)
    slice_sizes = jnp.array(slice_sizes, dtype=jnp.int32)  # Convert to int32

    # Add explicit shape checks *before* dynamic_slice
    if not isinstance(start_indices, tuple) or len(start_indices) != 2:
      raise ValueError(f"start_indices must be a tuple of length 2, got {start_indices}")
    
    if not isinstance(slice_sizes, jnp.ndarray) or slice_sizes.ndim != 1 or slice_sizes.shape[0] != 2 or slice_sizes.dtype != jnp.int32:
        raise ValueError(f"slice_sizes must be a 1D jnp.ndarray of shape (2,) and dtype int32, got {slice_sizes} with shape={slice_sizes.shape} and dtype={slice_sizes.dtype}")

    comp_logps_policy = jax.lax.dynamic_slice(token_logps_policy, start_indices, slice_sizes)

    return comp_logps_policy




def grpo_loss_fn(model, config, data, dropout_rng, params, reference_params, is_train=True):
  """
  GRPO loss function for training.
 
  This function performs the following steps:
 
    1. From the batch (data["prompt+completion"]) extract BxG prompts.
    -----2. For each prompt, generate `config.num_generations` completions using autoregressive sampling.
    3. Compute the per-token log-probabilities for the full sequence (prompt + completion) both with the
       current model (policy) and the reference model.
    4. Restrict the log-probabilities to the generated completion tokens.
    5. Compute a per-token KL divergence:
         kl = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1.
    6. Decode the (repeated) prompts and generated completions back to text (using tokenizer.batch_decode)
       and compute a scalar reward for each generated completion via reward_fn.
    7. Group the rewards (each prompt yields “G = num_generations” completions), compute the mean and std,
       and then compute a normalized advantage.
    8. Compute a per-token loss that is given by
         - [exp(policy_logp - stop_gradient(policy_logp)) * advantage - beta * kl]
       (the jax.lax.stop_gradient ensures that only the advantage contributes to gradients).
    9. Finally the loss is the average (over examples) of the mean per-token loss - where only tokens before the
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

  """
  # Split the dropout key.
  rng1, rng_gen = random.split(dropout_rng)
  
  # --- (1) Prepare prompts and tokenizer
  # Calling it prompts, because currently it only has left padded prompts such that all the prompts are aligned
  # at the right index, and then there is padding to the right till max_target_length
  # prompts = data["prompt"]
  if is_train:
    # restrict batch size when per_device_batch_size<1
    prompts = prompts[: config.micro_batch_size_to_train_on, :]
  """

  # # tokenizer_model = tokenizer.build_tokenizer(config.tokenizer_path, config.add_bos, config.add_eos)
  tokenizer_model = transformers.AutoTokenizer.from_pretrained(
    config.tokenizer_path,
    add_bos_token=config.add_bos,
    add_eos_token=config.add_eos,
    model_max_length=config.max_target_length,
    legacy=False,
    token=config.hf_access_token,
  )
  
  # completions shape: [B x G, max_target_length - max_prefill_length]
  # this includes the completion tokens + padding (upto max_target_length - max_prefill_length))
  # data["ar_completions"] contains tokens only upto the eos, no tokens thereafter other than pad_tokens
  ar_completions = data["ar_completions"] 
  L_total = ar_completions.shape[1] # max_target_length - max_prefill_length

  # --- (3) Compute per-token log probabilities.
  ar_completions_position = data["ar_completions_position"] 
  ar_completions_segmentation = data["ar_completions_segmentation"]

  # compute_log_probs returns logits.
  # We compute the log-probabilities for the entire generated sequence, then shift as usual.
  # 'completions' contains only the AR generated tokens, so token_logps_ref and token_logps_policy only contain the logprob for the ar tokens and not the prompts
  rng1, rng_fwd = random.split(dropout_rng)
  token_logps_policy = compute_log_probs(model, 
                                        params, 
                                        ar_completions, 
                                        ar_completions_position, 
                                        ar_completions_segmentation,
                                        config,
                                        is_train=is_train, rngs={"dropout": rng1, "params": rng_fwd}) # [BxG,S-1,E]
  # jax.debug.print("token_logps_policy={token_logps_policy}",token_logps_policy=token_logps_policy)
  # token_logps_policy_has_negative = jnp.any(token_logps_policy<0)
  # jax.debug.print("token_logps_policy_has_negative={token_logps_policy_has_negative}",token_logps_policy_has_negative=token_logps_policy_has_negative)
  token_logps_ref = compute_log_probs(model, 
                                      {"params": reference_params}, 
                                        ar_completions, 
                                        ar_completions_position, 
                                        ar_completions_segmentation,
                                        config,
                                        is_train=is_train, rngs={"dropout": rng1, "params": rng_fwd}) # [BxG,S-1,E]
  # jax.debug.print("token_logps_ref={token_logps_ref}",token_logps_ref=token_logps_ref)
  # token_logps_ref_has_negative = jnp.any(token_logps_ref<0)
  # jax.debug.print("token_logps_ref_has_negative={token_logps_ref_has_negative}",token_logps_ref_has_negative=token_logps_ref_has_negative)

  completion_target_segmentation = data["ar_completions_segmentation"][..., 1:] # [BxG,S-1]
  # Because of the shifting, token_logps have shape [BxG, S-1]. So, we create a mask for the valid tokens
  # Create a mask to clear out the last token position in the ar_completions 
  # and to make sure loss is computed on non-padding tokens
  valid_seq_mask = (completion_target_segmentation != 0)  # [BxG, S-1]
  valid_seq_mask_has_non_zero = jnp.sum(valid_seq_mask,axis=1) #[BxG]
  jax.debug.print("valid_seq_mask_has_non_zero={valid_seq_mask_has_non_zero}",valid_seq_mask_has_non_zero=valid_seq_mask_has_non_zero)
      
  # --- (4) Compute per-token KL divergence for each token in the generated completion.
  token_diff_logps_ref_policy = token_logps_ref - token_logps_policy
  token_diff_logps_ref_policy_mean = jnp.mean(token_diff_logps_ref_policy,axis=1)
  token_diff_logps_ref_policy_max = jnp.max(token_diff_logps_ref_policy,axis=1)
  token_diff_logps_ref_policy_min = jnp.min(token_diff_logps_ref_policy,axis=1)
  jax.debug.print("token_diff_logps_ref_policy_mean={token_diff_logps_ref_policy_mean}",token_diff_logps_ref_policy_mean=token_diff_logps_ref_policy_mean)
  jax.debug.print("token_diff_logps_ref_policy_max={token_diff_logps_ref_policy_max}",token_diff_logps_ref_policy_max=token_diff_logps_ref_policy_max)
  jax.debug.print("token_diff_logps_ref_policy_min={token_diff_logps_ref_policy_min}",token_diff_logps_ref_policy_min=token_diff_logps_ref_policy_min)


  per_token_kl = jnp.exp(token_diff_logps_ref_policy) - (token_diff_logps_ref_policy) - 1
  # loss is computed on non-padding tokens
  per_token_kl = per_token_kl * valid_seq_mask
  # jax.debug.print("per_token_kl={per_token_kl}",per_token_kl=per_token_kl)
  # per_token_kl_has_negative = jnp.any(per_token_kl<0)
  # jax.debug.print("per_token_kl_has_negative={per_token_kl_has_negative}",per_token_kl_has_negative=per_token_kl_has_negative)
  
  
  # --- (6) Decode prompts and completions (as text) to compute rewards.
  # Golden completions for each generation provided to us in data["completion"].
  golden_completions = jnp.repeat(data["completion"], config.num_generations, axis=0)  # shape: [BxG, ?]

  # Compute rewards (a scalar per generated completion); assume reward_fn is pure python.
  # jax.debug.print("ar_completions.shape={tokens1}",tokens1=ar_completions.shape)
  # jax.debug.print("golden_completions.shape={tokens2}",tokens2=golden_completions.shape)  
  rewards = dummy_reward_len(valid_seq_mask)
  # rewards = jaccard_reward_fn(ar_completions, golden_completions, config.vocab_size)
  rewards = jnp.array(rewards)  # shape [BxG]

  # --- (7) Group rewards and compute normalized advantage.
  G = config.num_generations
  rewards_grouped = rewards.reshape(-1, G)  # shape [B, G]
  group_mean = jnp.mean(rewards_grouped, axis=1)  # shape [B]
  group_std = jnp.std(rewards_grouped, axis=1)    # shape [B]
  repeated_group_mean = jnp.repeat(group_mean, G)   # shape [BxG]
  repeated_group_std = jnp.repeat(group_std, G)     # shape [BxG]
  advantages = (rewards - repeated_group_mean) / (repeated_group_std + 1e-4)  # shape [BxG]
 
  # --- (8) Compute per-token loss.
  # We follow the TRL GRPO loss:
  #   loss_token = - [ exp(policy_logp - stop_gradient(policy_logp)) * advantage - beta * kl ]
  # Make sure to expand advantage along the token dimension.
  advantages_exp = advantages[:, None]  # shape [BxG, 1]
  # jax.debug.print("advantages_exp={advantages_exp}",advantages_exp=advantages_exp)


  policy_diff = token_logps_policy - jax.lax.stop_gradient(token_logps_policy)
  # jax.debug.print("policy_diff={policy_diff}",policy_diff=policy_diff)
  loss_tokens = -( jnp.exp(policy_diff) * advantages_exp - config.grpo_beta * per_token_kl )
  # jax.debug.print("loss_tokens={loss_tokens}",loss_tokens=loss_tokens)
  # loss_tokens_has_negative = jnp.any(loss_tokens<0)
  # jax.debug.print("loss_tokens_has_negative={loss_tokens_has_negative}",loss_tokens_has_negative=loss_tokens_has_negative)
 
  # Average over tokens per generated completion.
  loss_per_example = jnp.sum(loss_tokens*valid_seq_mask, axis=1) / (jnp.sum(valid_seq_mask, axis=1) + EPS) 

  # jax.debug.print("loss_per_example={loss_per_example}",loss_per_example=loss_per_example)
  loss = jnp.mean(loss_per_example)

  # --- (9) Compute auxiliary metrics.
  avg_kl = jnp.mean((per_token_kl * valid_seq_mask) / (jnp.sum(valid_seq_mask, axis=1, keepdims=True) + EPS))
  avg_reward = jnp.mean(rewards)
  avg_advantage = jnp.mean(advantages)
  avg_completion_length = jnp.mean(jnp.sum(data["ar_completions_segmentation"]!=0, axis=1))
 
  aux = {
      "total_loss": loss,
      "avg_reward": avg_reward,
      "avg_reward_std": jnp.mean(repeated_group_std),
      "avg_advantage": avg_advantage,
      "avg_kl": avg_kl,
      "completion_length": avg_completion_length,
  }

  return loss, aux


# --- GRPO Helpers ---

def generate_completions(params, data, config, rng, tokenizer_model, engine, true_length):
  """
  Autoregressively generates completions for a batch of prompts.
  We assume the prompts are all left padded, so all of them have the same length=true_length

  Args:
    prompts: Array of shape [B, S] containing token ids.
    config: Configuration containing:
         - num_generations: number of completions to generate per prompt.
         - max_completion_length: maximum number of tokens to generate.
         - temperature: sampling temperature.
    rng: JAX PRNGKeys.
    tokenizer_model: Tokenizer for generate
    true_length: Length of the prompt out of the max_target_length
 
  Returns:
    A jnp.array of shape [B x num_generations, S] where S = length_of_prompt + max_completion_length.
  """
  prompts = data['prompt']
  rng, rng_load_params = jax.random.split(rng)
  G = config.num_generations
  # Repeat each prompt G times.
  # prompts = jnp.repeat(prompts, repeats=G, axis=0)  # shape [BxG, L_prompt]

  #TODO: Improve the token generation by using batch inference
  rng, rng_init_decode = jax.random.split(rng_load_params)
  decode_state = engine.init_decode_state(rng_init_decode)
  slot = 0
  for i in range(prompts.shape[0]):
    tokens = prompts[i]
    current_token_true_length = true_length[i]

    # Split RNG before calling prefill
    rng, rng_prefill = jax.random.split(rng)
    # generate the KV cache by prefilling the prompt tokens
    # Generate G completions for a prompt with different rng
    for _ in range(G):
      prefill_result, _ = engine.prefill(params=params, padded_tokens=tokens, true_length=current_token_true_length[0], rng=rng_prefill)
      decode_state = engine.insert(prefill_result, decode_state, slot=slot)
      slot += 1
  steps = config.max_target_length - config.max_prefill_predict_length
  completions = defaultdict(list)
  for _ in range(steps):
    rng, rng_generate = jax.random.split(rng)
    decode_state, result_tokens = engine.generate(params, decode_state, rng=rng_generate)
    for i in range(slot):
      completions[i].append(result_tokens.get_result_at_slot(i).tokens.item())
  completions = jnp.array(list(completions.values()))
  eos_positions = jnp.argmax(completions == tokenizer_model.eos_token_id, axis=1, keepdims=True)
  eos_not_found = jnp.all(eos_positions == 0, axis=1, keepdims=True)
  eos_positions = jnp.where(eos_not_found, steps, eos_positions)
  row_indices = jnp.arange(completions.shape[1])
  mask = row_indices <= eos_positions
  data['ar_completions'] = completions * mask
  data['ar_completions_segmentation'] = mask.astype(jnp.int32)
  data['ar_completions_position'] = jnp.where(mask, row_indices + 1, 0)
  return data

def prompt_completions(config, engine, tokenizer_model, data, params, rng):
  """ Complete input prompts
  """
  for k, v in data.items():
    if v.ndim == 2:
      data[k] = v[: config.micro_batch_size_to_train_on, :]
    else:
      data[k] = v[:config.micro_batch_size_to_train_on]

  rng, rng_gen = random.split(rng)
  # engine.load_params(params)
  L_prompt = data['prompt_true_length']
  data = generate_completions(
                              params=params, #TODO: this needs to be \theta_old, but for now we are using \theta_old = \theta
                              data=data,
                              config=config,
                              rng=rng_gen,
                              tokenizer_model=tokenizer_model,
                              engine=engine,
                              true_length=L_prompt,
                              )
  return data


def dummy_reward_len(valid_seq_mask):
  # adding a 1 because valid_seq_mask is actually one less than the number of valid tokens
  reward = -abs(20-(1+jnp.sum(valid_seq_mask,axis=-1))) # [BxG]
  jax.debug.print("reward={reward}",reward=reward)
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

def compute_log_probs(model, 
                      params, 
                      completions, 
                      completions_position, 
                      completions_segmentation,
                      config, is_train=False, rngs=None):
  """
  Given a sequence of tokens (shape [B, L]), this helper calls model.apply (with dropout enabled
  if is_train) to obtain logits and then computes per-token log-probabilities.
 
  Note: We assume that tokens have been already appropriately padded.
  """
  #TODO: Ensure attention mask takes into account the left paading
  
  logits = model.apply(
      params,
      completions,
      completions_position,
      decoder_segment_ids = completions_segmentation,
      enable_dropout=(config.enable_dropout if is_train else False),
      rngs=rngs,
  ) # [B, S, E] - [batch, sequence, embedding/vocab]
  if not is_train:
    logits = jax.lax.stop_gradient(logits)
  # Remove last time step since there is no target for the final position.
  logits = logits[:, :-1, :]
  targets = completions[:, 1:]
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  # Gather the log probabilities corresponding to each target token.
  token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
  return token_log_probs





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
  extra_dpo_args = [reference_params]
  _loss_fn = grpo_loss_fn

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
  # intermediate_outputs = aux["intermediate_outputs"]
  # total_weights = aux["total_weights"]
  # moe_lb_loss = aux["moe_lb_loss"]

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
      "learning/avg_reward": aux['avg_reward'],
      "learning/avg_reward_std": aux['avg_reward_std'],
      "learning/avg_advantage": aux['avg_advantage'],
      "learning/avg_kl": aux['avg_kl'],
      # TODO: Do we need these?
      # "learning/moe_lb_loss": moe_lb_loss,
      # "learning/total_weights": total_weights,
  }
  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = max_utils.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = max_utils.l2norm_pytree(raw_grads)
    scalar_metrics["learning/param_norm"] = max_utils.l2norm_pytree(new_state.params)
  if config.use_grpo:
    scalar_metrics["learning/avg_reward"] = aux["avg_reward"]
  metrics = {
      "scalar": scalar_metrics,
      "scalars": {},
  }

  # TODO: how to record intermediate_outputs
  # if config.record_internal_nn_metrics:
  #   record_activation_metrics(metrics, intermediate_outputs, config)

  if config.use_dpo or config.use_grpo:
    new_state = _merge_grpo_state(new_state, reference_params)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  """eval_step no backprop and new state compared with train_step."""

  reference_params, extra_dpo_args, _loss_fn = [], [], grpo_loss_fn
  # TODO: Add support for eval dataset in GRPO 

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
  # data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  state, _, state_mesh_shardings, data_iterator = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )

  if not config.using_pipeline_parallelism:
    # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
    maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

  if config.use_dpo or config.use_grpo:
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
      state = _merge_grpo_state(state, reference_params)
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

  if config.use_grpo:
    if "reference_params" not in state.params:
      reference_params = jax.tree.map(jnp.copy, state.params["params"])
      state = _merge_grpo_state(state, reference_params)
    state_mesh_shardings = _merge_grpo_state(state_mesh_shardings, state_mesh_shardings.params["params"])
  else:
    raise TypeError("Non grpo code calling grpo_trainer")

  # pylint: disable=line-too-long
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)

  # Initializing maxengine and everything related from decode.py
  # Creating an engine here but might have two model compilation, need to initialize engine while passing model object
  engine = maxengine.MaxEngine(config_inference)
  init_rng, rng_load_params = jax.random.split(init_rng)
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

  performance_metric_queue = None
  if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
    gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
    if config.report_heartbeat_metric_for_gcp_monitoring:
      gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
    if config.report_performance_metric_for_gcp_monitoring:
      performance_metric_queue = queue.Queue()
      gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)

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
      # do AR decoding here
      assert config.use_grpo, "Non grpo setting calling grpo_trainer"
      # engine_params = engine.load_params(engine_rng)      
      example_batch = prompt_completions(config_inference, engine, tokenizer_model, example_batch, state.params, init_rng)
      # jax.debug.print("golden completion {x}",x=example_batch['completion'][0])
      jax.debug.print("ar_completion[0] {x}",x=tokenizer_model.decode(example_batch['ar_completions'][0]))
      # jax.debug.print("ar_completion_segmentation {x}",x=example_batch['ar_completions_segmentation'][0])
      # jax.debug.print("ar_completion_position {x}",x=example_batch['ar_completions_position'][0])
      # TODO: ensure this partitioning is correct
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state, metrics = p_train_step(state, example_batch, nextrng)

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
      eval_loss = cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
          cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
      )
      cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
          cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
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
  config = pyconfig.initialize(argv)
  config_inference = pyconfig.initialize(argv + ['ici_tensor_parallelism=4', 'per_device_batch_size='+str(config.per_device_batch_size * config.num_generations)])
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
  # TODO: ensure that we have same configs as decode.py
  # TODO: we probably don't need everything in pyconfig.py to be present in pyconfig_inference.py
  # TODO: modify argv with sharding (e.g.,no fsdp) and attention_type (ideally prefill flash attention and AR with dot_product)
  
  # TODO: ensure we can run decode with full_state_path 
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config, config_inference)


if __name__ == "__main__":
  app.run(main)
