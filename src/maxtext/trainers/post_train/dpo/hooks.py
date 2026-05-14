# Copyright 2023–2026 Google LLC
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
  def get_total_weights(self, batch) -> jax.Array:
    # For DPO, we sum both chosen and rejected masks
    return jnp.sum(batch["chosen_mask"] != 0) + jnp.sum(batch["rejected_mask"] != 0)


class DPODataHooks(BaseDataHooks):
  """Data hooks for DPO."""
