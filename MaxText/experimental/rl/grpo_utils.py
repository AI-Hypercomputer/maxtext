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

import jax
import jax.numpy as jnp
import numpy as np


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



def concatenate_prompt_with_completions(config, tokenizer_model, data, completions):
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
  data["ar_completions_segmentation"] = data[f"{config.train_data_columns}_completions_segmentation"] * completion_mask.astype(jnp.int32)
  return data

def pad_or_trim(arr, max_target_length, pad_token):
  padded = np.array([
    np.pad(seq[:max_target_length], (0, max(0, max_target_length - len(seq))), constant_values=pad_token)
    for seq in arr
  ])
  return padded