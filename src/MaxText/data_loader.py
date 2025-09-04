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
from MaxText import maxtext_utils
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
    self.data_iterator = data_iterator
    self.last_batch = None
    self.input_data_shardings = maxtext_utils.get_input_data_sharding(config, mesh)

  def load_next_batch(self):
    """Loads the next batch. Can keep reusing the same batch for performance reasons."""
    with maybe_record_goodput(self.goodput_recorder, GoodputEvent.DATA_LOADING):
      try:
        if self.config.reuse_example_batch and self.last_batch:
          example_batch = self.last_batch
        else:
          example_batch = next(self.data_iterator)
        # Reshard data from loaded sharding to performant activation sharding
        self.last_batch = jax.lax.with_sharding_constraint(example_batch, self.input_data_shardings)
        self.check_example_batch()
      except Exception as e:  # pylint: disable=broad-except
        if "StopIteration" in str(e):
          raise exceptions.StopTraining("You may have run out of training data.")
        else:
          raise exceptions.StopTraining(f"`load_next_batch()` failed ({e}).")
    return self.last_batch

  def check_example_batch(self):
    if self.config.max_checkify:
      jittable_f = checkify.checkify(lambda x: checkify.check(jnp.any(x > -1), "Batch contains bad synthetic data!"))
      # Check if inputs in batch contains bad synthetic data.
      # pylint: disable=not-callable
      err, _ = jax.jit(jittable_f)(self.last_batch["inputs"][: self.config.global_batch_size_to_train_on, :])
      err.throw()
