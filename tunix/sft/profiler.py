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

"""Profiler class for Tunix trainers."""

import dataclasses
from absl import logging
import jax


@dataclasses.dataclass(frozen=True)
class ProfilerOptions:
  # Directory to write the profile to.
  log_dir: str
  # Number of steps to skip before profiling.
  skip_first_n_steps: int
  # Number of steps to profile.
  profiler_steps: int


class Profiler:
  """Activate/deactivate a profiler based on the ProfilerOptions."""

  def __init__(
      self,
      initial_step: int,
      max_step: int | None,
      profiler_options: ProfilerOptions | None,
  ):
    if jax.process_index() != 0 or profiler_options is None:
      self._do_not_profile = True
      return
    self._do_not_profile = False
    self._output_path = profiler_options.log_dir
    self._first_profile_step = (
        initial_step + profiler_options.skip_first_n_steps
    )
    self._last_profile_step = self._set_last_profile_step(
        profiler_options.profiler_steps, max_step
    )

  def maybe_activate(self, step: int):
    """Start the profiler."""
    if self._do_not_profile or step != self._first_profile_step:
      return
    logging.info("Starting JAX profiler at step %d.", step)
    jax.profiler.start_trace(self._output_path)

  def maybe_deactivate(self, step: int):
    """End the profiler."""
    if self._do_not_profile or step != self._last_profile_step:
      return
    logging.info("Stopping JAX profiler at step %d.", step)
    jax.profiler.stop_trace()

  def _set_last_profile_step(self, profiler_steps, max_step):
    calculated_last_step = self._first_profile_step + profiler_steps
    if max_step is None:
      return calculated_last_step
    return min(calculated_last_step, max_step)
