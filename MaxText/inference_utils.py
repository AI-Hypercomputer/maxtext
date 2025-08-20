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

import jax
import jax.numpy as jnp

NEG_INF = -1.0e7  # Masking purpose


# pylint: disable=bare-except, consider-using-generator, too-many-positional-arguments
""" Common Maxtext inference utilities. These seem like they should be a library.

    Inspired by an Google-internal implementation, Global Vision Transformer.
"""


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
    logits: jax.Array,               # [B, S, V] predicts token t+1 at position t
    input_tokens: jax.Array,         # [B, S]
    decoder_positions: jax.Array,    # [B, S] position within its own prompt
    decoder_segment_ids: jax.Array,  # [B, S] which prompt each token belongs to
    true_lengths: jax.Array          # [num_prompts] true lengths per prompt
) -> jax.Array:
  """
  Returns [B, S] where out[b, t] = log P(token[t] | tokens[:t] of its prompt).
  - First token of each segment = NaN (no prediction).
  - Tokens at or beyond the true length of their segment = NaN.
  """
  B, _, _ = logits.shape  # B, S, V

  # Compute next-token logprobs
  logps   = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)               # [B, S-1, V]
  targets = input_tokens[:, 1:]                                          # [B, S-1]
  scored  = jnp.take_along_axis(logps, targets[..., None], axis=-1)[..., 0]  # [B, S-1]

  # Shift so index matches token position (pad NaN at t=0)
  pad = jnp.full((B, 1), jnp.nan, dtype=logits.dtype)                     # [B, 1]
  shifted = jnp.concatenate([pad, scored], axis=1)                        # [B, S]

  # Get per-token true length by segment
  tl_tokens = jnp.take(true_lengths, decoder_segment_ids, mode="clip")    # [B, S]

  # Valid if not the first token in its segment and before true length
  valid = (decoder_positions > 0) & (decoder_positions < tl_tokens)      # [B, S]

  return jnp.where(valid, shifted, jnp.nan)


@jax.jit
def prompt_logprobs_from_prefill(
    logits: jax.Array,       # [B, S, V]  predicts token t+1 at position t
    input_tokens: jax.Array, # [B, S]
    true_length              # int or jax.Array with shape [] or [B]
) -> jax.Array:
  """
  Returns [B, S] where out[:, t] = log P(token[t] | tokens[:t]).
  - Position 0 is NaN (To match OpenAI format).
  - Positions >= true_length are masked to NaN.
  """
  B, S = input_tokens.shape

  # Next-token logprobs for steps 0..S-2
  logps   = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)        # [B, S-1, V]
  targets = input_tokens[:, 1:]                                    # [B, S-1]
  scored  = jnp.take_along_axis(logps, targets[..., None], -1)[..., 0]  # [B, S-1]

  # Align to token positions (pad NaN at t=0)
  pad = jnp.full((B, 1), jnp.nan, dtype=logps.dtype)               # [B, 1]
  out = jnp.concatenate([pad, scored], axis=1)                      # [B, S]

  # Mask padding (and keep t>0)
  tl = jnp.asarray(true_length)
  tl = jnp.broadcast_to(tl, (B,)) if tl.ndim == 0 else tl          # [B]
  pos = jnp.arange(S)[None, :]                                     # [1, S]
  valid = (pos < tl[:, None]) & (pos > 0)                          # [B, S]
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
  elif algorithm == "stochastic":
    return sample_topk_topp_weighted(logits, topk, nucleus_topp, temperature, rng)
  else:
    raise ValueError(f"Sampling {algorithm=} not supported!")


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
  """Combines top-k, top-p, and temperature sampling.

  Sequence of operations:
  1. Filter to top-k logits.
  2. Filter the remaining logits by top-p (nucleus).
  3. Apply temperature scaling.
  4. Sample from the final distribution.
  """
  if topk <= 0:
    raise ValueError(f"topk must be positive, got {topk=}")
  if not (0.0 < nucleus_topp <= 1.0):
    raise ValueError(f"nucleus_topp must be in (0, 1], got {nucleus_topp=}")

  # 1. Top-K filtering
  topk_logits, topk_idxs = jax.lax.top_k(logits, topk)

  # 2. Top-P filtering on the top-k results
  # Sort the top-k logits and get their cumulative probabilities
  topk_logits_sorted = jnp.sort(topk_logits, axis=-1)[..., ::-1]
  sorted_cum_probs = jnp.cumsum(jax.nn.softmax(topk_logits_sorted, axis=-1), axis=-1)

  # Find the cutoff index for nucleus sampling
  cutoff_index = jnp.sum(sorted_cum_probs < nucleus_topp, axis=-1, keepdims=True)
  cutoff_logit = jnp.take_along_axis(topk_logits_sorted, cutoff_index, axis=-1)

  # Mask logits that are below the cutoff
  filtered_topk_logits = jnp.where(
      topk_logits < cutoff_logit, jnp.full_like(topk_logits, NEG_INF), topk_logits
  )

  # 3. Apply temperature
  scaled_logits = filtered_topk_logits / temperature

  # 4. Sample
  sampled_topk_index = jax.random.categorical(rng, scaled_logits).astype(jnp.int32)

  # Map the index back to the original vocabulary
  sampled_token = jnp.squeeze(
      jnp.take_along_axis(topk_idxs, jnp.expand_dims(sampled_topk_index, axis=-1), axis=-1),
      axis=-1
  ).astype(jnp.int32)

  return sampled_token
