# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions for GRPO Trainer."""

import jax
from jax import numpy as jnp


def compute_advantages(rewards: jax.Array, num_generations: int) -> jax.Array:
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=1)
  std_grouped_rewards = rewards.reshape(-1, num_generations).std(axis=1, ddof=1)

  mean_grouped_rewards = mean_grouped_rewards.repeat(num_generations)
  std_grouped_rewards = std_grouped_rewards.repeat(num_generations)
  return (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)


def compute_kl_divergence(
    per_token_logps: jax.Array, ref_per_token_logps: jax.Array
) -> jax.Array:
  """Compute per token KL divergence between trained and reference policy.

  Args:
    per_token_logps: Per token log probabilities from the trained policy.
    ref_per_token_logps: Per token log probabilities from the reference policy.

  Returns:
    KL divergence.
  """
  per_token_kl = (
      jnp.exp(ref_per_token_logps - per_token_logps)
      - (ref_per_token_logps - per_token_logps)
      - 1
  )
  return per_token_kl
