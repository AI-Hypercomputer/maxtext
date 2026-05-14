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


"""Training and data loading hooks for SFT"""

from typing import override

import jax
import jax.numpy as jnp

from maxtext.trainers.post_train.hooks import BaseTrainingHooks, BaseDataHooks


class SFTTrainingHooks(BaseTrainingHooks):
  """Training hooks for SFT."""

  @override
  def get_metrics_to_pull(self, is_eval: bool = False) -> list[tuple[str, str]]:
    metrics = super().get_metrics_to_pull(is_eval)
    if not is_eval:
      metrics.append(("learning/lm_loss", "sft_loss"))
    else:
      metrics.append(("eval/avg_sft_loss", "sft_loss"))
    return metrics

  @override
  def get_total_weights(self, batch) -> jax.Array:
    """Calculate the number of non-padded tokens in the batch."""
    return jnp.sum(batch["targets_segmentation"] != 0)


class SFTDataHooks(BaseDataHooks):
  """Data hooks for SFT."""
