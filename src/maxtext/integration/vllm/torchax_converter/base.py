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

"""Shared base classes for MaxText to vLLM converters."""

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

GREEN = "\033[92m"
RESET = "\033[0m"


@contextmanager
def timer(name):
  start = time.perf_counter()
  yield
  end = time.perf_counter()
  print(f"{name} took {end - start:.4f} seconds")


class BaseMaxTextToVLLMConverter(ABC):
  """Shared converter contract for MaxText to vLLM weight conversion."""

  def __init__(self, config, mesh):
    self.config = config
    self.mesh = mesh
    self.num_layers = config.base_num_decoder_layers
    self.vllm_tp = self.config.rollout_tensor_parallelism
    self.vllm_state = {}

  def convert(self, model_state: dict):
    """Convert a MaxText model state into vLLM weight tensors."""
    logging.info("\n%sStarting Conversion...%s", GREEN, RESET)
    self.vllm_state = {}

    with timer("Convert Global Weights"):
      self._convert_global(model_state)

    with timer("Convert Attention Weights"):
      self._convert_attn(model_state)

    with timer("Convert MoE Weights"):
      self._convert_moe(model_state)

    return self.vllm_state

  @abstractmethod
  def _convert_global(self, params):
    """Convert non-layered weights."""

  @abstractmethod
  def _convert_attn(self, params):
    """Convert attention weights."""

  @abstractmethod
  def _convert_moe(self, params):
    """Convert MLP/MoE weights."""
