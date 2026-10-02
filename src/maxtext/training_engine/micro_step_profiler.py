# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Micro step driven profiling for the MaxText training engine."""

from __future__ import annotations

import jax
from typing import Any

from absl import logging
from maxtext.configs import pyconfig


class MicroStepProfiler:
  """Opens and closes one profile around a range of engine train steps."""

  def __init__(self, config: pyconfig.HyperParameters) -> None:
    """Initializes the profiler."""
    self.do_not_profile = False
    if jax.process_index() != 0 or config.profiler_steps == 0:
      self.do_not_profile = True
      return

    self.output_path = config.tensorboard_dir
    self.first_profile_step = config.skip_first_n_steps_for_profiler
    self.last_profile_step = self.first_profile_step + config.profiler_steps - 1
    if self.first_profile_step > self.last_profile_step:
      self.do_not_profile = True
      logging.warning(
          "Skipping profiling: the first profile step %d exceeds the last profile step %d.",
          self.first_profile_step,
          self.last_profile_step,
      )
      return
    self.profiling_options = jax.profiler.ProfileOptions()
    self.profile_active: bool = False
    # profiler_period is the distance between successive profiling window,
    # profiler_steps=3, profiler_period=4 gives [1,3], [5,7], [9,11].
    self.profiler_period = config.profile_periodically_period
    if 0 < self.profiler_period < config.profiler_steps:
      self.do_not_profile = True
      logging.warning(
          "profile_periodically_period=%d is shorter than "
          "profiler_steps=%d. The period is the gap between window "
          "starts, so it must be at least the window length or the windows overlap.",
          self.profiler_period,
          config.profiler_steps,
      )

  def maybe_activate(
      self, total_micro_steps: int, current_micro_step: int, train_step: int, blocking_object: Any = None
  ) -> None:
    """Starts a profile.

    Args:
      total_micro_steps: Total micro step this run has ever folded in across optimizer steps.
      current_micro_step: The micro step of the current train step about to run.
      train_step: The train step.
      blocking_object: Train state or another pytree to wait on before the profile
        starts.
    """
    if self.do_not_profile or self.profile_active or not self.should_activate_profile(total_micro_steps):
      return

    self.profile_active = True
    logging.info(
        "Started profiling at micro step %d and train step %d.",
        current_micro_step,
        train_step,
    )

    if blocking_object is not None:
      jax.block_until_ready(blocking_object)

    jax.profiler.start_trace(self.output_path, profiler_options=self.profiling_options)

  def maybe_deactivate(
      self, total_micro_steps: int, current_micro_step: int, train_step: int, blocking_object: Any = None
  ) -> None:
    """Stops the open profile.

    Args:
      total_micro_steps: Total micro step this run has ever folded in across optimizer steps.
      train_step: The train step.
      blocking_object: Train state or another pytree to wait on before the profile
        stops.
    """
    if self.do_not_profile or not self.profile_active or not self.should_deactivate_profile(total_micro_steps):
      return

    if blocking_object is not None:
      jax.block_until_ready(blocking_object)

    jax.profiler.stop_trace()
    logging.info(
        "Stopped profiling at micro step %d and train step %d.",
        current_micro_step,
        train_step,
    )
    self.profile_active = False

  def should_activate_profile(self, step):
    if self.profiler_period > 0:
      return (step - self.first_profile_step) % self.profiler_period == 0
    return step == self.first_profile_step

  def should_deactivate_profile(self, step):
    if self.profiler_period > 0:
      return (step - self.last_profile_step) % self.profiler_period == 0
    return step == self.last_profile_step

  def close(self, blocking_object: Any = None) -> None:
    """Stops a profile left open when the engine shuts down mid-window."""
    if self.do_not_profile or not self.profile_active:
      return

    logging.warning("Profiling was still active when engine close was called, stopping it now.")

    if blocking_object is not None:
      jax.block_until_ready(blocking_object)

    jax.profiler.stop_trace()
    self.profile_active = False
