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


"""Training and data loading hooks for DPO"""

from typing import override

import jax
import jax.numpy as jnp

from maxtext.trainers.post_train.hooks import BaseTrainingHooks, BaseDataHooks


class DPOTrainingHooks(BaseTrainingHooks):
  """Training hooks for DPO."""

  @override
  def get_metrics_to_pull(self, is_eval: bool = False) -> list[tuple[str, str]]:
    metrics = super().get_metrics_to_pull(is_eval)
    if self.config.use_dpo:
      if not is_eval:
        metrics.extend([
            ("learning/reward_accuracy", "rewards/accuracy"),
            ("learning/reward_margin", "rewards/margin"),
        ])
      else:
        metrics.extend([
            ("evaluation/dpo_reward_accuracy", "rewards/accuracy"),
            ("evaluation/dpo_reward_margin", "rewards/margin"),
        ])
    else:
      # ORPO specific metrics
      if not is_eval:
        metrics.extend([
            ("learning/lm_loss", "sft_loss"),
            ("learning/dpo_loss", "or_loss"),
        ])
      else:
        metrics.extend([
            ("eval/avg_sft_loss", "sft_loss"),
            ("eval/avg_or_loss", "or_loss"),
        ])
    return metrics

  @override
  def get_total_weights(self, batch) -> jax.Array:
    # For DPO, we sum both chosen and rejected masks
    return jnp.sum(batch["chosen_mask"] != 0) + jnp.sum(batch["rejected_mask"] != 0)


class DPODataHooks(BaseDataHooks):
  """Data hooks for DPO."""
