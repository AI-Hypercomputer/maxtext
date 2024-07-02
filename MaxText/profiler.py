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

import max_logging

import jax

class Profiler:
  """Activate/deactivate a profiler based on the 'profiler' config"""

  def __init__(self, config, optional_postfix=""):
    self.libcudart = None
    self.mode = config.profiler
    self.upload_all_profiler_results = config.upload_all_profiler_results
    if self.mode != "":
      self.output_path = os.path.join(config.tensorboard_dir, optional_postfix)

  def activate(self):
    """Start the profiler.
    nsys profiler becomes no-op when libcudart.so is not available on the system"""
    if not (self.upload_all_profiler_results or jax.process_index() == 0):
      return
    if self.mode == "nsys":
      try:
        self.libcudart = cdll.LoadLibrary('libcudart.so')
      except Exception as e: # pylint: disable=broad-except
        max_logging.log(f"WARNING: Failed to load library for nsys: {e}\n"
                        "profiler has no effect")
        return
      self.libcudart.cudaProfilerStart()
    elif self.mode == "xplane":
      jax.profiler.start_trace(self.output_path)

  def deactivate(self):
    """End the profiler.
    The result is uploaded to the output bucket"""
    if not (self.upload_all_profiler_results or jax.process_index() == 0):
      return
    if self.mode == "nsys":
      if self.libcudart is not None:
        self.libcudart.cudaProfilerStop()
      else:
        max_logging.log("WARNING: library for nsys was not loaded \n"
                        "profiler has no effect")
        return
      # Popen() instead of run() for non-blocking behavior
      subprocess.Popen(["gsutil", "cp", "*nsys-rep", self.output_path]) # pylint: disable=consider-using-with
    elif self.mode == "xplane":
      jax.profiler.stop_trace()
