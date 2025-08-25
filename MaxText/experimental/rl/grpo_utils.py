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

"""Utility functions for GRPO (Generative Rejection-based Policy Optimization)."""

import jax
import jax.numpy as jnp
import numpy as np
from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import DecoderBlockType
from MaxText.inference.offline_engine import InputData


def compute_log_probs(
    model,
    params,
    inputs,
    inputs_position,
    inputs_segmentation,
    completion_segmentation,
    config,
    is_train=False,
    rngs=None,
):
  """Computes per-token log-probabilities for a sequence of tokens.

  This helper calls model.apply (with dropout enabled if is_train) to obtain
  logits and then computes per-token log-probabilities.

  Note: We assume that tokens have been already appropriately padded.

  Args:
    model: The transformer model.
    params: Model parameters.
    inputs: A [B, L] array of input token IDs.
    inputs_position: A [B, L] array of token positions.
    inputs_segmentation: A [B, L] array of segment IDs.
    completion_segmentation: A [B, L] array that masks the completion part of
      the sequence.
    config: The configuration object.
    is_train: Whether to run in training mode (e.g., with dropout).
    rngs: JAX PRNG keys for dropout.

  Returns:
    A tuple containing:
      - token_log_probs: A [B, L-1] array of log-probabilities for each token
        in the completion.
      - intermediate_outputs: A dictionary of intermediate activations from the
        model.
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


def generate_offline_completions(config, tokenizer_model, inference_engine, data):
  """Generates completions for a batch of prompts using an offline engine.

  Args:
    config: The configuration object.
    tokenizer_model: The tokenizer model.
    inference_engine: The offline inference engine for generation.
    data: A dictionary containing the input prompts and their true lengths.

  Returns:
    The input `data` dictionary updated with the generated completions,
    segmentations, positions, and log-probabilities.
  """
  data[config.train_data_columns] = jnp.repeat(data[config.train_data_columns], config.num_generations, axis=0)
  
  data[f"{config.train_data_columns}_true_length"] = jnp.repeat(data[f"{config.train_data_columns}_true_length"], config.num_generations, axis=0)
  input_data = []
  for i, d in enumerate(data[config.train_data_columns]):
    input_data.append(
        InputData(
            id=i,
            tokens=np.array(d),
            true_length=np.array(data[f"{config.train_data_columns}_true_length"][i])[0],
        )
    )

  results = inference_engine.batch_inference(input_data)

  prompt_completions_segmentation = []
  completion_segmentation = []
  prompt_completions = []
  prompt_completions_logprobs = []
  for i, r in enumerate(results):
    indices = np.arange(r.token_ids.shape[0])
    completion_mask = (indices >= np.array(data[f"{config.train_data_columns}_true_length"][i])[0]).astype(jnp.int32)
    completion_segmentation.append(completion_mask)
    prompt_completions.append(r.token_ids)
    prompt_completions_segmentation.append(np.full((r.token_ids.shape[0],), 1))
    prompt_completions_logprobs.append(r.logprobs)

  prompt_completions = pad_or_trim(prompt_completions, config.max_target_length, 0)  # assume 0 for pad_token_id
  completion_segmentation = pad_or_trim(completion_segmentation, config.max_target_length, 0)
  prompt_completions_segmentation = pad_or_trim(prompt_completions_segmentation, config.max_target_length, 0)
  prompt_completions_logprobs = pad_or_trim(prompt_completions_logprobs, config.max_target_length, -np.inf)

  data[f"{config.train_data_columns}_completions"] = prompt_completions
  data[f"{config.train_data_columns}_completions_segmentation"] = prompt_completions_segmentation
  data[f"{config.train_data_columns}_completions_position"] = np.where(
      data[f"{config.train_data_columns}_completions_segmentation"],
      np.arange(data[f"{config.train_data_columns}_completions"].shape[1]),
      0,
  )
  data["ar_completions_segmentation"] = completion_segmentation
  # off-policy
  if config.inference_rollouts > 1:
    data["completions_logprobs"] = prompt_completions_logprobs
  else:
    data["completions_logprobs"] = None
  return data


def pathways_reshard(config, inference_engine, params, source_shardings, source_mesh, destination_shardings):
  """Reshards model parameters from training to inference sharding.

  This function handles the resharding of parameters between different device
  meshes and sharding specifications, which is often necessary when moving from
  a training setup to an inference setup. It also handles unscanning of
  parameters if `config.scan_layers` is False.

  Args:
    config: The configuration object.
    inference_engine: The inference engine whose parameters will be updated.
    params: The model parameters to be resharded.
    source_shardings: The sharding specification of the source parameters.
    source_mesh: The source device mesh.
    destination_shardings: The sharding specification for the destination.
  """
  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    layer_groups = [
        ("dense_layers", config.first_num_dense_layers),
        ("moe_layers", config.base_num_decoder_layers - config.first_num_dense_layers),
    ]
  else:
    layer_groups = [("layers", config.base_num_decoder_layers)]
  if not config.scan_layers:
    max_utils.unscan_train_state_params(
        params, source_shardings, source_mesh, scan_axis=config.param_scan_axis, layer_groups=layer_groups
    )

  inference_engine.update_params(
      params, jax.tree_util.tree_map(lambda x: x.spec, destination_shardings.params), is_pw_reshard=True
  )

  if not config.scan_layers:
    max_utils.rescan_train_state_params(
        params, source_shardings, scan_axis=config.param_scan_axis, layer_groups=layer_groups
    )


def dummy_reward_len(valid_seq_mask):
  """Calculates a dummy reward based on the length of the valid sequence.

  Args:
    valid_seq_mask: A [BxG] mask indicating the valid (non-padded) tokens in
      the sequence.

  Returns:
    A [BxG] array of rewards, where the reward is the negative absolute
    difference between the sequence length and 20.
  """
  # adding a 1 because valid_seq_mask is actually one less than the number of valid tokens
  reward = -abs(20 - (1 + jnp.sum(valid_seq_mask, axis=-1)))  # [BxG]
  return reward


def concatenate_prompt_with_completions(config, tokenizer_model, data, completions):
  """Concatenates prompts with their generated completions.

  This function takes a batch of prompts and a corresponding batch of
  completions, concatenates them, and then truncates each sequence at the
  first end-of-sequence (EOS) token. It updates the data dictionary with the
  new combined sequences and their corresponding segmentation and position
  arrays.

  Args:
    config: Configuration object containing generation settings such as max
      sequence length or EOS token ID.
    tokenizer_model: Tokenizer used to decode or manipulate tokens (e.g.,
      identifying special tokens like EOS).
    data: Input batch containing prompt tokens, segmentation and position.
    completions: Generated token sequences to be appended to the corresponding
      prompts.

  Returns:
    The `data` dictionary updated with the concatenated sequences and new
    segmentation and position information.
  """

  def _concat_and_find_eos(prompt, true_len, completion):
    total_len = prompt.shape[0] + completion.shape[0]
    prompt_mask = jnp.arange(prompt.shape[0]) < true_len[0]
    trimmed_prompt = jnp.where(prompt_mask, prompt, 0)

    # Initialize with padded prompt
    full_seq = jnp.zeros((total_len,), dtype=prompt.dtype)
    full_seq = full_seq.at[: prompt.shape[0]].set(trimmed_prompt)

    # Dynamically insert completion at true_len position
    full_seq = jax.lax.dynamic_update_slice(full_seq, completion, (true_len[0],))

    # Find EOS index
    eos_mask = full_seq == tokenizer_model.eos_token_id
    eos_indices = jnp.where(eos_mask, jnp.arange(total_len), total_len)
    eos_index = jnp.min(eos_indices)

    return full_seq, eos_index

  batched_concat_and_eos = jax.vmap(_concat_and_find_eos, in_axes=(0, 0, 0))
  prompts = data[config.train_data_columns]
  true_length = data[f"{config.train_data_columns}_true_length"]
  prompt_completions, eos_positions = batched_concat_and_eos(prompts, true_length, completions)
  data[f"{config.train_data_columns}_completions"] = prompt_completions
  data[f"{config.train_data_columns}_completions_segmentation"] = (
      jnp.arange(data[f"{config.train_data_columns}_completions"].shape[1])[None, :] < eos_positions[:, None]
  ).astype(jnp.int32)
  data[f"{config.train_data_columns}_completions_position"] = jnp.where(
      data[f"{config.train_data_columns}_completions_segmentation"],
      jnp.arange(data[f"{config.train_data_columns}_completions"].shape[1]),
      0,
  )
  completion_mask = data[f"{config.train_data_columns}_completions_position"] >= true_length - 1
  data["ar_completions_segmentation"] = data[
      f"{config.train_data_columns}_completions_segmentation"
  ] * completion_mask.astype(jnp.int32)
  return data


def pad_or_trim(arr, max_target_length, pad_token):
  """Pads or trims a list of sequences to a maximum target length.

  Args:
    arr: A list of 1D numpy arrays (sequences).
    max_target_length: The desired length for all sequences.
    pad_token: The token ID to use for padding.

  Returns:
    A 2D numpy array of shape `(len(arr), max_target_length)`.
  """
  padded = np.array(
      [
          np.pad(seq[:max_target_length], (0, max(0, max_target_length - len(seq))), constant_values=pad_token)
          for seq in arr
      ]
  )
  return padded


def filter_and_split(config, example_batch, num_groups, global_batch_size_per_group):
  """Splits an example_batch into a list of smaller batches.

  Samples are taken from the beginning of the input batch, and extras are
  dropped if there are not enough to form the requested number of groups.

  Args:
    config: The configuration object.
    example_batch: A dictionary where keys are feature names (e.g., 'inputs')
      and values are arrays. The first dimension of each array is the batch
      size.
    num_groups: The number of smaller batches to create.
    global_batch_size_per_group: The size of each smaller batch.

  Returns:
    A list of dictionaries. Each dictionary has the same keys as
    `example_batch` but with values being arrays sliced for that group.
    Returns an empty list if not enough samples to form the required groups.
  """
  if not example_batch:  # Handles None or empty dict
    return []

  if num_groups <= 0 or global_batch_size_per_group <= 0:
    max_logging.log(
        f"Warning: config_inference.inference_replicas ({num_groups}) or config_inference.per_device_batch_size "
        f"({global_batch_size_per_group}) is not positive. Cannot split batch."
    )
    return []

  total_samples_needed = num_groups * global_batch_size_per_group
  total_samples_available = example_batch[config.train_data_columns].shape[0]
  if total_samples_available < total_samples_needed:
    max_logging.log(
        f"Warning: Not enough samples ({total_samples_available}) in batch to create {num_groups} groups of size"
        f" {global_batch_size_per_group} (needed {total_samples_needed}). Dropping batch."
    )
    return []

  # Slice the required number of samples
  sliced_batch = jax.tree_util.tree_map(lambda arr: arr[:total_samples_needed], example_batch)

  list_of_output_batches = []
  for i in range(num_groups):
    current_group_dict = {}
    start_index = i * global_batch_size_per_group
    end_index = start_index + global_batch_size_per_group
    for key, sliced_array in sliced_batch.items():
      # Slice each group
      current_group_dict[key] = sliced_array[start_index:end_index]
    list_of_output_batches.append(current_group_dict)

  return list_of_output_batches
