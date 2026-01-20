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

import abc
from typing import List, Optional

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
  """Computes log probabilities for packed prefill.

  Returns [B, S] where out[b, t] = log P(token[t] | tokens[:t] of its prompt).
  - First token of each segment = NaN (no prediction).
  - Tokens at or beyond the true length of their segment = NaN.

  Args:
    logits: [B, S, V] predicts token t+1 at position t.
    input_tokens: [B, S] input token IDs.
    decoder_positions: [B, S] position within its own prompt.
    decoder_segment_ids: [B, S] which prompt each token belongs to.
    true_lengths: [num_prompts] true lengths per prompt.

  Returns:
    jax.Array: The log probabilities [B, S].
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
  """Computes log probabilities for prefill.

  Args:
    logits: [B, S, V] predicts token t+1 at position t.
    input_tokens: [B, S] input token IDs.
    true_length: int or jax.Array with shape [] or [B].

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
def log_prob_of_chosen_token(logits: jax.Array, chosen_index: jax.Array) -> jax.Array:
  """Calculates the log probability of chosen tokens.

  Args:
    logits: unnormalized logits, shape [batch, seq, vocab].
    chosen_index: index of the chosen token, shape [batch, seq].

  Returns:
    jax.Array: The log probability of the chosen tokens [batch, seq].
  """
  logps = jax.nn.log_softmax(logits, axis=-1)  # [batch, seq, vocab]
  chosen_prob = jnp.take_along_axis(logps, chosen_index[..., None], axis=-1)  # [batch, seq, 1]
  return chosen_prob[..., 0]  # [batch, seq]


# --- Chain of Responsibility for Logits Processing ---


class LogitsProcessor(abc.ABC):
  """Abstract base class for all LogitsProcessors that modify the distribution."""

  @abc.abstractmethod
  def __call__(self, logits: jax.Array) -> jax.Array:
    """Process logits and return new logits.

    Args:
      logits: The input logits tensor [batch_size, vocab_size].

    Returns:
      jax.Array: The processed logits.
    """
    raise NotImplementedError


class TemperatureLogitsWarper(LogitsProcessor):
  """Logits processor that divides logits by a temperature value."""

  def __init__(self, temperature: float):
    """Initializes the TemperatureLogitsWarper.

    Args:
      temperature: The value to divide the logits by. Must be > 0.
    """
    self.temperature = temperature

  def __call__(self, logits: jax.Array) -> jax.Array:
    # Ensure temperature is not zero to avoid division by zero
    temp = jnp.maximum(self.temperature, 1e-7)
    return logits / temp


class TopKLogitsWarper(LogitsProcessor):
  """Logits processor that performs top-k filtering."""

  def __init__(self, top_k: int, filter_value: float = NEG_INF):
    """Initializes the TopKLogitsWarper.

    Args:
      top_k: The number of highest probability vocabulary tokens to keep.
      filter_value: The value to set for filtered tokens (default: -1.0e7).
    """
    self.top_k = top_k
    self.filter_value = filter_value

  def __call__(self, logits: jax.Array) -> jax.Array:
    # Safety check for top_k > 0
    k = jnp.maximum(self.top_k, 1)
    # Get the value of the k-th largest logit
    top_k_values = jax.lax.top_k(logits, k)[0][..., -1, None]
    # Mask logits smaller than the k-th value
    return jnp.where(logits < top_k_values, self.filter_value, logits)


class TopPLogitsWarper(LogitsProcessor):
  """Logits processor that performs nucleus (top-p) filtering.

  It keeps the top tokens with cumulative probability >= top_p.
  """

  def __init__(self, top_p: float, filter_value: float = NEG_INF, min_tokens_to_keep: int = 1):
    """Initializes the TopPLogitsWarper.

    Args:
      top_p: The cumulative probability threshold (0 < top_p <= 1).
      filter_value: The value to set for filtered tokens.
      min_tokens_to_keep: Minimum number of tokens to keep regardless of p.
    """
    self.top_p = top_p
    self.filter_value = filter_value
    self.min_tokens_to_keep = min_tokens_to_keep

  def __call__(self, logits: jax.Array) -> jax.Array:
    # Sort logits in descending order
    sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # Find the cutoff index where cumulative probability exceeds top_p
    # sum(cumulative_probs < top_p) gives the count of items to *keep* strictly < p.
    # We want to include the first item that crosses the threshold, so we don't -1 here.
    # We enforce keeping at least 'min_tokens_to_keep'.
    cutoff_index = jnp.sum(cumulative_probs < self.top_p, axis=-1, keepdims=True)
    cutoff_index = jnp.maximum(cutoff_index, self.min_tokens_to_keep - 1)

    # Get the logit value corresponding to the cutoff index
    cutoff_logit = jnp.take_along_axis(sorted_logits, cutoff_index, axis=-1)

    # Mask logits smaller than the cutoff logit
    return jnp.where(logits < cutoff_logit, self.filter_value, logits)


class MinPLogitsWarper(LogitsProcessor):
  """Logits processor that performs Min-P filtering.

  It masks tokens with probability < min_p * max_probability.
  This is implemented using logs as: logit < max_logit + log(min_p).
  """

  def __init__(self, min_p: float, filter_value: float = NEG_INF):
    """Initializes the MinPLogitsWarper.

    Args:
      min_p: The scaling factor for the minimum probability threshold (0 <= min_p <= 1).
      filter_value: The value to set for filtered tokens.
    """
    self.min_p = min_p
    self.filter_value = filter_value

  def __call__(self, logits: jax.Array) -> jax.Array:
    # Calculate max logit (which corresponds to max probability)
    top_logit = jnp.max(logits, axis=-1, keepdims=True)
    # Threshold in log space: log(p_threshold) = log(p_max * min_p) = log(p_max) + log(min_p)
    # Since log(p_max) = max_logit - log(Z), and log(p_item) = logit - log(Z),
    # The comparison log(p_item) < log(p_threshold) simplifies to logit < max_logit + log(min_p).
    logit_threshold = top_logit + jnp.log(self.min_p)
    return jnp.where(logits < logit_threshold, self.filter_value, logits)


class LogitsProcessorList(LogitsProcessor):
  """A container to apply a list of LogitsProcessors sequentially."""

  def __init__(self, processors: Optional[List[LogitsProcessor]] = None):
    """Initializes the LogitsProcessorList.

    Args:
      processors: A list of LogitsProcessor instances.
    """
    self.processors = processors if processors is not None else []

  def append(self, processor: LogitsProcessor):
    """Appends a processor to the list."""
    self.processors.append(processor)

  def __call__(self, logits: jax.Array) -> jax.Array:
    """Applies all processors in order."""
    for processor in self.processors:
      logits = processor(logits)
    return logits


class Sampler:
  """Base sampler class to perform the final random selection."""

  def __call__(self, logits: jax.Array, rng: jax.Array) -> jax.Array:
    """Samples from the logits using categorical distribution.

    Args:
      logits: The processed logits tensor.
      rng: The JAX random key.

    Returns:
      jax.Array: The sampled token ID (int32).
    """
    return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)


def sampling(
    logits: jax.Array,
    rng: jax.Array,
    algorithm: str,
    topk: int = 0,
    nucleus_topp: float = 0.0,
    temperature: float = 1.0,
    min_p: float = 0.0,
) -> jax.Array:
  """Performs sampling on logits using the specified algorithm and parameters.

  This function constructs a `LogitsProcessorList` based on the provided arguments
  to warp the distribution, and then samples a token.

  Args:
    logits: The unnormalized log probabilities [batch, ..., vocab].
    rng: The JAX random number generator key.
    algorithm: The sampling strategy ('greedy', 'topk', 'nucleus', 'min_p', 'composite', 'weighted').
    topk: The K value for top-k sampling (used if algorithm is 'topk' or 'composite').
    nucleus_topp: The P value for nucleus sampling (used if algorithm is 'nucleus' or 'composite').
    temperature: The temperature value (used for all stochastic sampling).
    min_p: The P value for min-p sampling (used if algorithm is 'min_p' or 'composite').

  Returns:
    jax.Array: The sampled token IDs.

  Raises:
    ValueError: If an unsupported algorithm is provided.
  """
  if algorithm == "greedy":
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)

  # Construct the processor chain
  processors = LogitsProcessorList()

  # 1. Temperature (Apply first to affect probability distribution for P/Min-P logic)
  if temperature != 1.0 and temperature > 0:
    processors.append(TemperatureLogitsWarper(temperature))

  # 2. Min P
  if algorithm == "min_p" or (algorithm == "composite" and min_p > 0.0):
    processors.append(MinPLogitsWarper(min_p))

  # 3. Top K (Prune hard tail)
  if algorithm == "topk" or (algorithm == "composite" and topk > 0):
    processors.append(TopKLogitsWarper(topk))

  # 4. Top P (Prune stochastic tail)
  if algorithm == "nucleus" or (algorithm == "composite" and 0.0 < nucleus_topp < 1.0):
    processors.append(TopPLogitsWarper(nucleus_topp))

  # Apply processors
  logits = processors(logits)

  # Sample
  sampler = Sampler()
  return sampler(logits, rng)
