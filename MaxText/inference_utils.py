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

import jax
import jax.numpy as jnp

NEG_INF = -1.0e7  # Masking purpose


# pylint: disable=bare-except, consider-using-generator
""" Common Maxtext inference utilities. These seem like they should be a library.

    Inspired by an Google-internal implementation, Global Vision Transformer.
"""

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
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted, axis=-1), axis=-1)  # get cumsum probs
  cutoff_index = jnp.sum(
      sorted_cum_probs < nucleus_topp, axis=-1, keepdims=True)  # find cutoff index
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit,
                     jnp.full_like(logits, NEG_INF), logits)
  return jax.random.categorical(rng, logits / temperature)

def sample_topk_logits(logits, topk, temperature, rng):
  """ Restricting sampling to the best k logits. """
  if topk <= 0:
    raise ValueError("Can't apply algorithm topk with parameter {topk=} less than or equal to zero")
  topk_logits, topk_idxs = jax.lax.top_k(logits, topk)
  topk_token = jnp.expand_dims(
      jax.random.categorical(rng, topk_logits/temperature).astype(jnp.int32),
      axis=-1)
  sampled_tokens = jnp.squeeze(
      jnp.take_along_axis(topk_idxs, topk_token, axis=-1),
      axis=-1).astype(jnp.int32)
  return sampled_tokens


