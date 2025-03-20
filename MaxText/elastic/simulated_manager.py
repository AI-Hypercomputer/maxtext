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
from typing import Sequence

import jax
from elastic.debug import timing
from elastic import manager


logger = logging.getLogger(__name__)


class SimulatedManager(manager.Manager):
  """Utility class for elastic training.

  This class will simulate slices going down and coming back up.
  """

  simulated_good_slice_indices: set[int]

  def __init__(
      self,
      devices: Sequence[jax.Device],
      snapshot_period: int = 1,
      snapshot_buffer_size: int = 1,
      reshard_check_period: int = 1,
      max_failure_count: int | None = None,
      max_reshard_retry_count: int | None = None,
  ) -> None:
    self.simulated_good_slice_indices = set(d.slice_index for d in devices)

    super().__init__(
        devices,
        snapshot_period,
        reshard_check_period,
        max_failure_count,
        max_reshard_retry_count,
    )

  def update_good_slice_indices(self, good_slice_indices: set[int]) -> None:
    """Start step handler."""
    self.simulated_good_slice_indices = good_slice_indices
    logger.debug(
        "Updated: simumlated_good_slice_indices=%s",
        self.simulated_good_slice_indices,
    )

  @timing.timeit
  def get_slice_availability(self) -> set[int]:
    """Returns the set of good and bad slices."""
    good_slice_indices = self.simulated_good_slice_indices

    logger.debug("good_slice_indices=%s", good_slice_indices)

    return good_slice_indices
