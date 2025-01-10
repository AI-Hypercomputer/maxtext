"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Dispatch to the chosen profiler"""
from ctypes import cdll
import os
import subprocess
import shutil

import max_logging

import jax


class Profiler:
  """Activate/deactivate a profiler based on the 'profiler' config"""

  def __init__(self, config, offset_step=0):
    self.libcudart = None
    self.mode = config.profiler
    if self.mode != "":
      self.base_output_dir = config.tensorboard_dir
    self.output_path = ""
    self.upload_all_profiler_results = config.upload_all_profiler_results
    self.profile_cleanly = config.profile_cleanly
    self.profile_period = config.profile_periodically_period
    self.start_initial_profile_step = self._set_first_profiler_step(config.skip_first_n_steps_for_profiler, offset_step)
    self.finished_initial_profile_step = self._set_last_profiler_step(config.profiler_steps, config.steps)

  def activate(self, blocking_object=None, optional_postfix=""):
    """Start the profiler.
    nsys profiler becomes no-op when libcudart.so is not available on the system"""
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)
    if not (self.upload_all_profiler_results or jax.process_index() == 0):
      return
    if self.mode != "":
      self.output_path = os.path.join(self.base_output_dir, optional_postfix)
    if self.mode == "nsys":
      try:
        self.libcudart = cdll.LoadLibrary("libcudart.so")
      except Exception as e:  # pylint: disable=broad-except
        max_logging.log(f"WARNING: Failed to load library for nsys: {e}\n" "profiler has no effect")
        return
      self.libcudart.cudaProfilerStart()
    elif self.mode == "xplane":
      jax.profiler.start_trace(self.output_path)

  def deactivate(self, blocking_object=None):
    """End the profiler.
    The result is uploaded to the output bucket"""
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)
    if not (self.upload_all_profiler_results or jax.process_index() == 0):
      return
    if self.mode == "nsys":
      if self.libcudart is not None:
        self.libcudart.cudaProfilerStop()
      else:
        max_logging.log("WARNING: library for nsys was not loaded \n" "profiler has no effect")
        return
      # Popen() instead of run() for non-blocking behavior
      if shutil.which("gsutil") is not None:
        subprocess.Popen(["gsutil", "cp", "*nsys-rep", self.output_path])  # pylint: disable=consider-using-with
      else:
        max_logging.log("WARNING: gsutil is not installed or not found in the system's PATH. Skipping upload...")
    elif self.mode == "xplane":
      jax.profiler.stop_trace()

  def _set_first_profiler_step(self, skip_steps, start_step):
    return start_step + skip_steps

  def _set_last_profiler_step(self, profiler_steps, last_job_step):
    return min(self.start_initial_profile_step + profiler_steps - 1, last_job_step - 1)

  def should_activate_periodic_profile(self, step):
    return self.profile_period > 0 and (step - self.start_initial_profile_step) % self.profile_period == 0

  def should_deactivate_periodic_profile(self, step):
    return self.profile_period > 0 and (step - self.finished_initial_profile_step) % self.profile_period == 0
