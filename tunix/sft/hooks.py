# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hooks for training and data loading."""

import abc
from typing import Any

PeftTrainer = Any


class TrainingHooks(abc.ABC):
  """Hooks to be used for training."""

  @abc.abstractmethod
  def on_train_start(self, train_ctx: "PeftTrainer.PeftTrainer"):
    """Called at the beginning of training."""
    pass

  @abc.abstractmethod
  def on_train_end(self, train_ctx: "PeftTrainer.PeftTrainer"):
    """Called at the end of training."""
    pass

  @abc.abstractmethod
  def on_train_step_start(self, train_ctx: "PeftTrainer.PeftTrainer"):
    """Called at the beginning of a training step."""
    pass

  @abc.abstractmethod
  def on_train_step_end(
      self, train_ctx: "PeftTrainer.PeftTrainer", train_loss: float
  ):
    """Called at the end of a training step."""
    pass

  @abc.abstractmethod
  def on_eval_step_start(self, train_ctx: "PeftTrainer.PeftTrainer"):
    """Called at the beginning of an evaluation step."""
    pass

  @abc.abstractmethod
  def on_eval_step_end(
      self, train_ctx: "PeftTrainer.PeftTrainer", eval_loss: float
  ):
    """Called at the end of an evaluation step."""
    pass


class DataHooks(abc.ABC):
  """Hooks to wire in external data loader and processing logic."""

  @abc.abstractmethod
  def load_next_train_batch(self, train_ctx: "PeftTrainer.PeftTrainer") -> Any:
    """Loads the next batch of data for training."""
    raise NotImplementedError()

  @abc.abstractmethod
  def load_next_eval_batch(self, train_ctx: "PeftTrainer.PeftTrainer") -> Any:
    """Loads the next batch of data for evaluation."""
    raise NotImplementedError()
