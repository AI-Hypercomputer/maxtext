# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for elastic training."""

import logging
from typing import Any, Sequence

import jax
from elastic import utils


PyTree = Any

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

#  pylint: disable=logging-fstring-interpolation


class ElasticUtilsSimulator(utils.ElasticUtils):
  """Utility class for elastic training.

  This class will simulate slices going down and coming back up.
  """
  simulated_good_slice_indices: set[int]

  def __init__(
      self,
      devices: Sequence[jax.Device],
      total_slice_count: int,
      save_period: int | None = None,
      reshard_check_period: int | None = None,
      max_failure_count: int | None = None,
      max_reshard_retry_count: int | None = None,
  ):
    self.simulated_good_slice_indices = set(d.slice_index for d in devices)

    super().__init__(
        devices,
        total_slice_count,
        save_period,
        reshard_check_period,
        max_failure_count,
        max_reshard_retry_count,
    )

  def update_good_slice_indices(self, good_slice_indices: set[int]):
    """Start step handler."""
    self.simulated_good_slice_indices = good_slice_indices
    logger.info(f"Updated: {self.simulated_good_slice_indices=}")

  @utils.timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    good_slice_indices = self.simulated_good_slice_indices

    logger.info(f"{good_slice_indices=}")

    return good_slice_indices

