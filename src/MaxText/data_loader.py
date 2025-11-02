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

# pytype: disable=unsupported-operands
"""Module to load data for training."""

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from MaxText import exceptions
from MaxText import max_logging
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    maybe_record_goodput,
)


class DataLoader:
  """
  Loads preprocessed data for training.
  """

  def __init__(self, config, mesh, data_iterator, goodput_recorder):
    self.config = config
    self.goodput_recorder = goodput_recorder
    if isinstance(data_iterator, list):
      self.data_iterator_list = data_iterator
      self.data_iterator_index = 0
      self.data_iterator = self.data_iterator_list[self.data_iterator_index]
    else:
      self.data_iterator = data_iterator
    self.last_batch = None

  def update_data_iterator(self):
    """Update to the next data iterator in the list, if applicable."""
    if hasattr(self, "data_iterator_list"):
      self.data_iterator_index = (self.data_iterator_index + 1) % len(self.data_iterator_list)
      self.data_iterator = self.data_iterator_list[self.data_iterator_index]

  def load_next_batch(self):
    """Loads the next batch. Can keep reusing the same batch for performance reasons."""
    with maybe_record_goodput(self.goodput_recorder, GoodputEvent.DATA_LOADING):
      try:
        if self.config.reuse_example_batch and self.last_batch:
          example_batch = self.last_batch
        else:
          example_batch = next(self.data_iterator)
          self.update_data_iterator()
        self.last_batch = example_batch
        self.check_example_batch()
      except Exception as e:  # pylint: disable=broad-except
        if isinstance(e, StopIteration):
          raise exceptions.StopTraining(f"You may have run out of training data. Received {type(e)} exception: ({e})")
        else:
          raise exceptions.StopTraining(f"`load_next_batch()` failed with {type(e)} exception: ({e}).")
    return self.last_batch

  def check_example_batch(self):
    if self.config.max_checkify:
      jittable_f = checkify.checkify(lambda x: checkify.check(jnp.any(x > -1), "Batch contains bad synthetic data!"))
      # Check if inputs in batch contains bad synthetic data.
      # pylint: disable=not-callable
      err, _ = jax.jit(jittable_f)(self.last_batch["inputs"][: self.config.global_batch_size_to_train_on, :])
      err.throw()


class RampUpDataLoader(DataLoader):
  """
  A DataLoader that implements batch size ramp-up.

  It dynamically increases the 'global_batch_size_current' in the config
  object based on the training step. The rest of the training pipeline
  (including the parent's `check_example_batch` and the training step itself)
  is assumed to read this config value to determine the logical batch size.
  """

  def __init__(self, config, mesh, data_iterator, goodput_recorder):
    # Call parent constructor
    super().__init__(config, mesh, data_iterator, goodput_recorder)

    # Get ramp-up parameters from config, with safe defaults
    self.global_batch_size_end = config.global_batch_size_to_load
    self.global_batch_size_start = config.global_batch_size_to_load_start
    self.increment = config.global_batch_size_to_load_increment
    self.samples_per_increment = config.rampup_samples_per_increment_to_load

    # Check if ramp-up is active
    self.rampup_active = self.global_batch_size_start < self.global_batch_size_end

    # State for tracking ramp-up
    self.accum_samples = 0
    self.global_batch_size_current = self.global_batch_size_start
    self.batch_buffer = None
    self.buffer_start = 0

  def load_next_batch(self):
    """
    Updates the batch size based on the schedule and then loads the next
    batch using the parent method.
    """
    # If ramp-up is not active, just behave like the parent
    if not self.rampup_active:
      return super().load_next_batch()

    # If in rampup phase, we use batch buffer to save data
    # Check if it's time to increment the batch size
    is_time_to_increment = self.accum_samples >= self.samples_per_increment

    if is_time_to_increment:
      # Update current batch size and refresh accumulate samples
      max_logging.log(
          f"Global batch size increments from {self.global_batch_size_current}"
          f" to {self.global_batch_size_current + self.increment}"
      )
      self.global_batch_size_current += self.increment
      self.accum_samples = 0
      self.rampup_active = self.global_batch_size_current < self.global_batch_size_end

    self.accum_samples += self.global_batch_size_current
    slice_start, slice_end = self.buffer_start, self.buffer_start + self.global_batch_size_current

    # Load new batch if batch_buffer is None or slice overpast the buffer end
    if self.batch_buffer is None:
      self.batch_buffer = super().load_next_batch()
      slice_start, slice_end = 0, self.global_batch_size_current

    if slice_end > self.global_batch_size_end:
      old_buffer, self.batch_buffer = self.batch_buffer, super().load_next_batch()

      # self.global_batch_size_end is batch_buffer size
      def _slice_and_concat(old_data, new_data):
        sliced_old_data = jax.lax.dynamic_slice_in_dim(
            old_data,
            slice_start,
            self.global_batch_size_end - slice_start,
            axis=0,
        )
        sliced_new_data = jax.lax.dynamic_slice_in_dim(
            new_data,
            0,
            slice_end - self.global_batch_size_end,
            axis=0,
        )
        return jax.lax.concatenate((sliced_old_data, sliced_new_data), dimension=0)

      self.buffer_start = slice_end - self.global_batch_size_end
      return jax.tree.map(_slice_and_concat, old_buffer, self.batch_buffer)
    else:

      def _slice(data):
        return jax.lax.dynamic_slice_in_dim(
            data,
            slice_start,
            self.global_batch_size_current,
            axis=0,
        )

      self.buffer_start = slice_end
      return jax.tree.map(_slice, self.batch_buffer)


def create_dataloader(config, mesh, data_iterator, goodput_recorder):
  """
  Create the dataloader
  """
  if config.enable_rampup_batch_size:
    return RampUpDataLoader(config, mesh, data_iterator, goodput_recorder)
  else:
    return DataLoader(config, mesh, data_iterator, goodput_recorder)
