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

""" The NNX Unified TrainState. """

from typing import Any

from flax import nnx


class TrainStateNNX(nnx.Module):
  """
  A unified container for NNX models and optimizers.
  This replaces Linen's TrainState for checkpointing.

  Linen TrainState pytree:
    {“params”: {...}, “opt_state”: {}...}
  TrainStateNNX state pytree:
    {“model”: {...}, “optimizer”: {“opt_state”: {...}}
  """

  def __init__(self, model: nnx.Module, optimizer: nnx.Optimizer | None):
    self.model = model
    self.optimizer = optimizer

  def apply_gradients(self, grads: Any):
    """
    Mimics the Linen apply_gradients function.
    Updates the optimizer state, applies updates to parameters,
    and increments the step counter.
    """
    if self.optimizer is None:
      raise RuntimeError(
          "Cannot call apply_gradients on a TrainStateNNX initialized without an optimizer. "
          "This usually happens when the state was created for inference only."
      )
    self.optimizer.update(self.model, grads)
