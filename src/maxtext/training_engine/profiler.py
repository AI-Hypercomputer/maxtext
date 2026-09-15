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

"""Step-driven profiling for the MaxText training engine."""

from __future__ import annotations

import jax
from typing import Any

from absl import logging
from maxtext.configs import pyconfig


class Profiler:
  """Opens and closes one profile around a range of engine train steps."""

  def __init__(self, config: pyconfig.HyperParameters, offset_step: int = 0) -> None:
    """Initializes the profiler.

    Args:
      config: The training configuration.
      offset_step: The train step the run starts from.
        On a resumed run this is the restored step, not zero, so a resume
        profiles the steps it actually runs rather than steps already behind it.

    Raises:
      ValueError: If profiling is requested but the window starts past `config.steps`.
    """
    self.do_not_profile = False
    if jax.process_index() != 0 or config.profiler_steps == 0:
      self.do_not_profile = True
      return

    self.output_path = config.tensorboard_dir
    self.first_profile_step = config.skip_first_n_steps_for_profiler + offset_step
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

  def maybe_activate(self, step: int, blocking_object: Any = None) -> None:
    """Starts a profile if `step` opens a profiling window and none is open yet.

    Args:
      step: The train step about to run.
      blocking_object: Train state or another pytree to wait on before the profile
        starts.
    """
    if self.do_not_profile or step != self.first_profile_step or self.profile_active:
      return

    self.profile_active = True
    logging.info("Started profiling at train step %d.", step)

    if blocking_object is not None:
      jax.block_until_ready(blocking_object)

    jax.profiler.start_trace(self.output_path, profiler_options=self.profiling_options)

  def maybe_deactivate(self, step: int, blocking_object: Any = None) -> None:
    """Stops the open profile if `step` closes its window.

    Args:
      step: The train step that just finished.
      blocking_object: Train state or another pytree to wait on before the profile
        stops.
    """
    if self.do_not_profile or not self.profile_active or step != self.last_profile_step:
      return

    if blocking_object is not None:
      jax.block_until_ready(blocking_object)

    jax.profiler.stop_trace()
    logging.info("Stopped profiling at train step %d.", step)
    self.profile_active = False

  def close(self, blocking_object: Any = None) -> None:
    """Stops a profile left open when the engine shuts down mid-window."""
    if self.do_not_profile or not self.profile_active:
      return

    logging.warning("Profiling was still active when engine close was called, stopping it now.")

    if blocking_object is not None:
      jax.block_until_ready(blocking_object)

    jax.profiler.stop_trace()
    self.profile_active = False
