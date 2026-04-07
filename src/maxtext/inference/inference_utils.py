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

"""Common utilities for MaxText inference, including sampling and log probability calculations."""

import jax
import jax.numpy as jnp

NEG_INF = -1.0e7  # Masking purpose


# pylint: disable=bare-except, consider-using-generator, too-many-positional-arguments
def str2bool(v: str) -> bool:
  """Convert a string of truth to True or False.

  Args:
    - v (str):
      - True values are 'y', 'yes', 't', 'true', and '1';
      - False values are 'n', 'no', 'f', 'false', and '0'.

  Returns:
    bool: True or False

  Raises:
    ValueError if v is anything else.
  """
  v = v.lower()
  true_values = ["y", "yes", "t", "true", "1"]
  false_values = ["n", "no", "f", "false", "0"]
  if v in true_values:
    return True
  elif v in false_values:
    return False
  else:
    raise ValueError(f"Invalid value '{v}'!")


@jax.jit
def prompt_logprobs_from_packed_prefill(
    logits: jax.Array,  # [B, S, V] predicts token t+1 at position t
    input_tokens: jax.Array,  # [B, S]
    decoder_positions: jax.Array,  # [B, S] position within its own prompt
    decoder_segment_ids: jax.Array,  # [B, S] which prompt each token belongs to
    true_lengths: jax.Array,  # [num_prompts] true lengths per prompt
) -> jax.Array:
  """
  Returns [B, S] where out[b, t] = log P(token[t] | tokens[:t] of its prompt).
  - First token of each segment = NaN (no prediction).
  - Tokens at or beyond the true length of their segment = NaN.
  """
  B, _, _ = logits.shape  # B, S, V

  # Compute next-token logprobs
  logps = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)  # [B, S-1, V]
  targets = input_tokens[:, 1:]  # [B, S-1]
  scored = jnp.take_along_axis(logps, targets[..., None], axis=-1)[..., 0]  # [B, S-1]

  # Shift so index matches token position (pad NaN at t=0)
  pad = jnp.full((B, 1), jnp.nan, dtype=logits.dtype)  # [B, 1]
  shifted = jnp.concatenate([pad, scored], axis=1)  # [B, S]

  # Get per-token true length by segment
  tl_tokens = jnp.take(true_lengths, decoder_segment_ids, mode="clip")  # [B, S]

  # Valid if not the first token in its segment and before true length
  valid = (decoder_positions > 0) & (decoder_positions < tl_tokens)  # [B, S]

  return jnp.where(valid, shifted, jnp.nan)


@jax.jit
def prompt_logprobs_from_prefill(
    logits: jax.Array,  # [B, S, V]  predicts token t+1 at position t
    input_tokens: jax.Array,  # [B, S]
    true_length,  # int or jax.Array with shape [] or [B]
) -> jax.Array:
  """
  Returns [B, S] where out[:, t] = log P(token[t] | tokens[:t]).
  - Position 0 is NaN (To match OpenAI format).
  - Positions >= true_length are masked to NaN.
  """
  B, S = input_tokens.shape

  # Next-token logprobs for steps 0..S-2
  logps = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)  # [B, S-1, V]
  targets = input_tokens[:, 1:]  # [B, S-1]
  scored = jnp.take_along_axis(logps, targets[..., None], -1)[..., 0]  # [B, S-1]

  # Align to token positions (pad NaN at t=0)
  pad = jnp.full((B, 1), jnp.nan, dtype=logps.dtype)  # [B, 1]
  out = jnp.concatenate([pad, scored], axis=1)  # [B, S]

  # Mask padding (and keep t>0)
  tl = jnp.asarray(true_length)
  tl = jnp.broadcast_to(tl, (B,)) if tl.ndim == 0 else tl  # [B]
  pos = jnp.arange(S)[None, :]  # [1, S]
  valid = (pos < tl[:, None]) & (pos > 0)  # [B, S]
  out = jnp.where(valid, out, jnp.nan)

  return out


@jax.jit
def log_prob_of_chosen_token(logits, chosen_index):
  """
  logits: unnormalized logits, shape [batch, seq, vocab]
  chosen_index: index of the chosen token, shape [batch, seq]
  """
  logps = jax.nn.log_softmax(logits, axis=-1)  # [batch, seq, vocab]
  chosen_prob = jnp.take_along_axis(logps, chosen_index[..., None], axis=-1)  # [batch, seq, 1]
  return chosen_prob[..., 0]  # [batch, seq]


def sampling(logits, rng, algorithm, topk=0, nucleus_topp=0, temperature=1.0):
  """
  logits: unnormalized logits to sample, shaped [YOUR_LEADING_DIMS, Vocab], before logit
  rng: rng key to use
  algorithm: string representing supported algorithms
  topk: restricting to topk logits before sampling
  nucleus_topp: restricting to p probability mass before sampling
  temperature: temperature parameter for scaling probability
  """
  if algorithm == "greedy":
    return jnp.argmax(logits, axis=-1)
  elif algorithm == "weighted":
    return jax.random.categorical(rng, logits / temperature)
  elif algorithm == "nucleus":
    return sample_nucleus_topp_logits(logits, nucleus_topp, temperature, rng)
  elif algorithm == "topk":
    return sample_topk_logits(logits, topk, temperature, rng)
  elif algorithm == "composite":
    return sample_topk_topp_weighted(logits, topk, nucleus_topp, temperature, rng)
  elif algorithm == "diverse_beam_search":
    # This expects a special call signature with cumulative_logprobs
    raise ValueError("diverse_beam_search must be called via sampling_dbs")
  else:
    raise ValueError(f"Sampling {algorithm=} not supported!")


def sampling_dbs(
    logits, cumulative_logprobs, num_beams, num_groups, diversity_penalty, topk=None
):
  """Router for Diverse Beam Search."""
  # logits shape: (total_batch_size, 1, vocab_size)
  # cumulative_logprobs shape: (total_batch_size, 1)
  # total_batch_size = (user_batch * num_beams)
  # Returns: (chosen_tokens, chosen_scores, chosen_parents)
  return sample_diverse_beam_search_step(
      logits, cumulative_logprobs, num_beams, num_groups, diversity_penalty, topk
  )


def sample_nucleus_topp_logits(logits, nucleus_topp, temperature, rng):
  """Restrict sampling to the top logits with cumulative probability >= nucleus_topp.

  The nucleus sampling method is proposed in the paper `The Curious Case of
  Neural Text Degeneration (https://arxiv.org/pdf/1904.09751.pdf)`

  """
  if nucleus_topp < 0:
    raise ValueError("Can't apply nucleus with parameter {nucleus_topp=} less zero")
  logits_sorted = jnp.sort(logits, axis=-1)[..., ::-1]  # sort descending
  sorted_cum_probs = jnp.cumsum(jax.nn.softmax(logits_sorted, axis=-1), axis=-1)  # get cumsum probs
  cutoff_index = jnp.sum(sorted_cum_probs < nucleus_topp, axis=-1, keepdims=True)  # find cutoff index
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit, jnp.full_like(logits, NEG_INF), logits)
  return jax.random.categorical(rng, logits / temperature)


def sample_topk_logits(logits, topk, temperature, rng):
  """Restricting sampling to the best k logits."""
  if topk <= 0:
    raise ValueError("Can't apply algorithm topk with parameter {topk=} less than or equal to zero")
  topk_logits, topk_idxs = jax.lax.top_k(logits, topk)
  topk_token = jnp.expand_dims(jax.random.categorical(rng, topk_logits / temperature).astype(jnp.int32), axis=-1)
  sampled_tokens = jnp.squeeze(jnp.take_along_axis(topk_idxs, topk_token, axis=-1), axis=-1).astype(jnp.int32)
  return sampled_tokens


def sample_topk_topp_weighted(logits, topk, nucleus_topp, temperature, rng):
  """Applies top-k, top-p, and temperature sampling to logits.

  This function combines three common sampling techniques to control the
  randomness and diversity of the generated text. The operations are applied
  sequentially:

  1.  **Top-k filtering**: The vocabulary is restricted to the `topk` most
      likely tokens.
  2.  **Top-p (nucleus) filtering**: From the `topk` tokens, the smallest
      set of tokens whose cumulative probability exceeds `nucleus_topp` is
      selected.
  3.  **Temperature scaling**: The logits of the filtered tokens are scaled
      by the `temperature`. Higher temperatures result in a flatter
      distribution (more randomness), while lower temperatures make the
      distribution sharper (less randomness).
  4.  **Sampling**: A token is sampled from the final probability
      distribution using composite sampling.

  Args:
    logits: The unnormalized log probabilities of the vocabulary tokens,
      with shape `[batch, sequence, vocab_size]`.
    topk: The number of most likely tokens to consider. Must be positive.
    nucleus_topp: The cumulative probability threshold for nucleus sampling.
      Must be in the range (0, 1].
    temperature: The temperature for scaling the logits.
    rng: The JAX random number generator key.

  Returns:
    The sampled token indices, with shape `[batch, sequence]`.
  """
  if topk <= 0:
    raise ValueError(f"topk must be positive, got {topk=}")
  if not 0.0 < nucleus_topp <= 1.0:
    raise ValueError(f"nucleus_topp must be in (0, 1], got {nucleus_topp=}")

  # 1. Top-K filtering
  topk_logits, topk_idxs = jax.lax.top_k(logits, topk)

  # 2. Top-P filtering on the top-k results
  sorted_cum_probs = jnp.cumsum(jax.nn.softmax(topk_logits, axis=-1), axis=-1)

  # Find the number of elements to keep. We keep all elements until the cumulative
  # probability exceeds nucleus_topp. This is equivalent to finding the index of
  # the first element that is >= nucleus_topp and keeping all elements up to that index.
  # `jnp.sum(sorted_cum_probs < nucleus_topp)` gives the index of the last element
  # strictly within the nucleus. We need to include the next element that crosses the threshold.
  cutoff_index = jnp.sum(sorted_cum_probs < nucleus_topp, axis=-1, keepdims=True)

  # Create a mask that is True for indices we want to keep.
  # We keep all indices up to and including the cutoff_index.
  indices = jnp.arange(topk_logits.shape[-1])
  mask = indices <= cutoff_index

  # Apply the mask to filter the logits.
  filtered_topk_logits = jnp.where(mask, topk_logits, jnp.full_like(topk_logits, NEG_INF))

  # 3. Apply temperature
  scaled_logits = filtered_topk_logits / jnp.maximum(temperature, 1e-6)  # add epsilon for stability

  # 4. Sample
  sampled_topk_index = jax.random.categorical(rng, scaled_logits).astype(jnp.int32)

  # Map the index back to the original vocabulary
  sampled_token = jnp.squeeze(
      jnp.take_along_axis(topk_idxs, jnp.expand_dims(sampled_topk_index, axis=-1), axis=-1), axis=-1
  ).astype(jnp.int32)

  return sampled_token

def sample_diverse_beam_search_step(
    logits, cumulative_logprobs, num_beams, num_groups, diversity_penalty, pool_size=None
):
  """Implementation of Diverse Beam Search using an optional candidate pool.

  Args:
    logits: The log probabilities of the vocabulary tokens with shape
      `[total_batch_size, 1, vocab_size]`.
    cumulative_logprobs: The cumulative log probabilities of the beams with
      shape `[total_batch_size, 1]`.
    num_beams: The number of beams to use.
    num_groups: The number of groups to use per beam.
    diversity_penalty: The diversity penalty to use.
    pool_size: The size of the candidate pool. Default to num_beams.

  Returns:
    chosen_tokens: The sampled token indices, one per batch, with shape
      `[batch, 1]`. Here batch is the total batch size, i.e.
      user_batch_size * num_beams.
    chosen_scores: The log probabilities of the chosen tokens, with shape
      `[batch, 1]`.
    chosen_parents: The parent indices of the chosen tokens, with shape
      `[batch, 1]`.
  """
  if num_beams % num_groups != 0:
    raise ValueError(
        f"num_beams ({num_beams}) must be divisible by num_groups ({num_groups})"
    )
  if pool_size is None:
    pool_size = num_beams

  total_batch_size = logits.shape[0]
  # Safety for JAX tracing: ensure user_batch_size is at least 1 during trace.
  user_batch_size = max(1, total_batch_size // num_beams)
  vocab_size = logits.shape[-1]
  beams_per_group = num_beams // num_groups

  # 1. Convert to log probabilities and add parent scores
  # logits shape: (total_batch_size, 1, vocab_size)
  # logprobs shape: (total_batch_size, vocab_size)
  logprobs = jax.nn.log_softmax(jnp.squeeze(logits, axis=1), axis=-1)
  # logprobs shape: (user_batch_size, num_beams, vocab_size)
  logprobs = logprobs.reshape((user_batch_size, num_beams, vocab_size))
  # path_scores shape: (user_batch_size, num_beams, vocab_size)
  path_scores = logprobs + cumulative_logprobs.reshape((user_batch_size, num_beams, 1))

  # 2. Extract top candidates for EACH beam to create a small search pool
  # pool_scores shape: (user_batch, num_beams, pool_size)
  pool_scores, pool_token_ids = jax.lax.top_k(path_scores, pool_size)

  all_chosen_tokens = []
  all_chosen_scores = []
  all_chosen_parents = []
  diversity_mask = jnp.zeros((user_batch_size, vocab_size))

  # 3. Process groups to apply diversity penalties
  group_pool_scores = pool_scores.reshape((user_batch_size, num_groups, beams_per_group, pool_size))
  group_pool_token_ids = pool_token_ids.reshape((user_batch_size, num_groups, beams_per_group, pool_size))

  for g in range(num_groups):
    # penalized_scores shape: (user_batch, beams_per_group, pool_size)
    current_token_ids = group_pool_token_ids[:, g, :, :]
    
    # Apply penalty based on global token IDs
    # penalties: (user_batch, beams_per_group, pool_size)
    penalties = jnp.take_along_axis(
        # diversity_mask shape: (user_batch, vocab_size) converted to
        # (user_batch, 1, vocab_size)
        diversity_mask[:, None, :], 
        # current_token_ids shape: (user_batch, beams_per_group, pool_size)
        current_token_ids, 
        axis=2
    ).reshape((user_batch_size, beams_per_group, pool_size))
    
    # penalized_scores shape: (user_batch, 1, beams_per_group, pool_size)
    penalized_scores = group_pool_scores[:, g, :, :] - (penalties * diversity_penalty)
    
    # Pick top winners for this group from the pool
    # flat_scores shape: (user_batch, beams_per_group * pool_size)
    flat_scores = penalized_scores.reshape((user_batch_size, -1))
    # top_scores shape: (user_batch, beams_per_group) for this group.
    # top_pool_indices shape: (user_batch, beams_per_group) for this group
    top_scores, top_pool_indices = jax.lax.top_k(flat_scores, beams_per_group)

    # Get the parent index in the current group. Since each group has pool_size
    # candidates, the top_pool_indices range from 0 to
    # beams_per_group * pool_size - 1.
    # So top_pool_indices // pool_size will get the parent index in the current
    # group.
    # e.g. If beams_per_group=2, pool_size=4, then top_pool_indices can be
    # 0, 1, 2, 3. Then parent_in_group_idx will be 0, 0, 1, 1.
    # parent_in_group_idx shape: (user_batch, beams_per_group)
    parent_in_group_idx = top_pool_indices // pool_size
    
    # Map back to global token IDs
    # actual_token_ids shape: (user_batch, beams_per_group)
    actual_token_ids = jnp.take_along_axis(
        current_token_ids.reshape((user_batch_size, -1)), 
        top_pool_indices, 
        axis=1
    )
    
    # Calculate global parent index in the KV cache
    # global_parent_idx shape: (user_batch, beams_per_group)
    # Note that the broadcast rule applies at the last operation. i.e. 
    # (user_batch, 1) + (user_batch, beams_per_group)
    global_parent_idx = (
        # jnp.arange(user_batch_size) shape: (user_batch, 1)
        jnp.arange(user_batch_size)[:, None] * num_beams
        # g * beams_per_group is scalar, so the shape is still (user_batch, 1)
        + g * beams_per_group
        # parent_in_group_idx shape: (user_batch, beams_per_group), broadcast
        # rule applies here to make the final shape:
        # (user_batch, beams_per_group)
        + parent_in_group_idx
    )

    all_chosen_tokens.append(actual_token_ids)
    all_chosen_scores.append(top_scores)
    all_chosen_parents.append(global_parent_idx)

    # Update diversity mask efficiently without materializing a full one-hot matrix
    diversity_mask = diversity_mask.at[jnp.arange(user_batch_size)[:, None], actual_token_ids].add(1.0)

  # 4. Final assembly into (total_batch_size, 1)
  return (
      jnp.concatenate(all_chosen_tokens, axis=1).reshape((total_batch_size, 1)),
      jnp.concatenate(all_chosen_scores, axis=1).reshape((total_batch_size, 1)),
      jnp.concatenate(all_chosen_parents, axis=1).reshape((total_batch_size, 1))
  )
