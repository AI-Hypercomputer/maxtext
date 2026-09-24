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

"""Dispatch to the chosen profiler."""

from ctypes import cdll
import os
import subprocess
import shutil

import time
import traceback

import jax

from maxtext.common.gcloud_stub import mldiagnostics_modules

mldiag, _ = mldiagnostics_modules()

from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
from maxtext.utils import max_logging


class Profiler:
  """Activate/deactivate a profiler based on the 'profiler' config."""

  def __init__(self, config, offset_step=0):
    self.libcudart = None
    self.config = config
    self.mode = config.profiler
    if self.mode != "":
      self.base_output_dir = config.tensorboard_dir
    self.output_path = ""
    self.upload_all_profiler_results = config.upload_all_profiler_results
    self.profile_cleanly = config.profile_cleanly
    self.profile_period = config.profile_periodically_period
    self.start_initial_profile_step = self._set_first_profiler_step(config.skip_first_n_steps_for_profiler, offset_step)
    self.finished_initial_profile_step = self._set_last_profiler_step(config.profiler_steps, config.steps)
    if config.profiler != "" and self.start_initial_profile_step >= config.steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    self.prof = None  # managed mldiagnostics xprof collector.
    self.is_active = False
    self.managed_mldiagnostics = config.managed_mldiagnostics
    if config.managed_mldiagnostics:
      ManagedMLDiagnostics(config)  # Initialize the MLRun instance.

    self.profiling_options = jax.profiler.ProfileOptions()
    self.profiling_options.host_tracer_level = 3
    self.profiling_options.enable_hlo_proto = True
    advanced_config = {}

    if self.mode == "xplane" and not self.managed_mldiagnostics and (
        config.profile_power_events or config.xprof_tpu_power_trace_level > 0
    ):
      advanced_config.update(
          {
              "tpu_power_trace_level": config.xprof_tpu_power_trace_level,
              "e2e_enable_fw_throttle_event": config.xprof_e2e_enable_fw_throttle_event,
              "e2e_enable_fw_power_level_event": config.xprof_e2e_enable_fw_power_level_event,
              "e2e_enable_fw_thermal_event": config.xprof_e2e_enable_fw_thermal_event,
          }
      )

    if self.mode == "xplane" and config.enable_tpu_profiling_options:
      advanced_config.update(
          {
              "tpu_num_chips_to_profile_per_task": config.tpu_num_chips_to_profile_per_task,
              "tpu_num_sparse_core_tiles_to_trace": config.tpu_num_sparse_core_tiles_to_trace,
              "tpu_num_sparse_cores_to_trace": config.tpu_num_sparse_cores_to_trace,
              "tpu_enable_kernel_profiling": True,
              "tpu_perf_counters": True,
              "tpu_trace_mode": "TRACE_ALL",
              "max_trace_buffers": 65536,
          }
      )

    if advanced_config:
      self.profiling_options.advanced_configuration = advanced_config

  def maybe_activate_profiler(self, step, state):
    """Conditionally activates the profiler based on the current step.
    This method checks if the current training step matches the step designated
    for starting an initial profile, or if it meets the criteria for
    activating a new periodic profile.
    """
    if self.mode != "" and (step == self.start_initial_profile_step or self.should_activate_periodic_profile(step)):
      optional_postfix = f"step_{step}" if self.profile_period > 0 else ""
      self.activate(blocking_object=state, optional_postfix=optional_postfix)

  def _pathways_max_num_hosts(self) -> int:
    """Upper bound on the number of Pathways worker hosts to profile.

    `pathwaysutils.profiling.start_trace` defaults `max_num_hosts=1`, which
    silently yields a single worker's TPU plane and drops the rest. There is no
    public "number of hosts" accessor under Pathways (`jax.process_count()`
    reports 1 because it is single-controller), so derive it from the device
    topology and log what we saw, then clamp to a safe upper bound.

    A host can never own fewer than one device, so `len(jax.devices())` is
    always a valid upper bound; `max_num_hosts` is documented as a *limit*, so
    overshooting is safe while undershooting silently loses planes.
    """
    override = getattr(self.config, "profiler_max_num_hosts", 0) or 0
    try:
      devs = jax.devices()
      n_dev = len(devs)

      # Try to group devices into hosts. Different jax/Pathways versions expose
      # this differently, so probe several attributes and log which worked.
      # Take the attribute that yields the MOST groups: on v6e Pathways,
      # `host_id` is constant within a slice (v6e-nscc-p1 logged
      # "host_groups=2 (via host_id)" for 2 slices x 2 hosts), which made us
      # profile only 2 of 4 worker hosts. Undershooting silently drops planes;
      # max_num_hosts is documented as a limit (overshoot not verified on HW).
      # The only hardware-validated path is the explicit override
      # profiler_max_num_hosts=4 (v6e-pwns-prof1: all 4 hosts x 4 chips captured).
      groups = set()
      attr_used = None
      for cand in ("task_index", "logical_task", "host_id", "process_index"):
        if all(hasattr(d, cand) for d in devs):
          cand_groups = {(getattr(d, "slice_index", 0), getattr(d, cand)) for d in devs}
          if len(cand_groups) > len(groups):
            groups = cand_groups
            attr_used = cand
      derived = len(groups) if groups else 0

      max_logging.log(
          f"Profiler: device topology n_devices={n_dev} "
          f"slices={len({getattr(d, 'slice_index', 0) for d in devs})} "
          f"host_groups={derived} (via {attr_used}) "
          f"process_count={jax.process_count()}"
      )

      if override > 0:
        max_logging.log(f"Profiler: max_num_hosts overridden by config -> {override}")
        return int(override)

      # Prefer the derived host count when it looks sane, otherwise fall back to
      # the device count as a safe over-estimate.
      if 1 < derived <= n_dev:
        return int(derived)
      return int(max(1, n_dev))
    except Exception as e:  # pylint: disable=broad-except
      max_logging.error(f"Profiler: could not derive host count ({e}); using 8")
      return int(override) if override > 0 else 8

  def activate(self, blocking_object=None, optional_postfix=""):
    """Start the profiler.
    nsys profiler becomes no-op when libcudart.so is not available on the system."""
    if self.is_active:
      return
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)

    if self.managed_mldiagnostics and self.mode == "xplane":
      # Handle the special profiling logic for managed_mldiagnostics
      if self.prof is None:
        # Starts xprof collector.
        # Only profiling on the first device, if not upload_all_profiler_results. None is for all devices.
        self.prof = mldiag.xprof(process_index_list=None if self.upload_all_profiler_results else [0])
      self.prof.start()
      self.is_active = True
      return

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
      if self.output_path.startswith("gs://"):
        # ------------------------------------------------------------------
        # Pathways profiling requires TWO things that are easy to get wrong and
        # that both fail SILENTLY, producing empty/partial TPU traces:
        #
        #  1. `pathwaysutils.profiling.monkey_patch_jax()` must have run, else
        #     `jax.profiler.start_trace` is the stock PJRT one, which only
        #     traces the CLIENT process. The TPU worker planes then never
        #     appear at all.
        #
        #  2. `max_num_hosts` DEFAULTS TO 1 in pathwaysutils
        #     (profiling.py: `max_num_hosts: int = 1`, forwarded as
        #     `"maxNumHosts"` in the profile request). With N worker hosts you
        #     get 1 host's trace and N-1 missing ones. On this cluster a
        #     v6e-8 slice is 2 hosts (Pathways reports logical_task=0 for
        #     devices 0-3 and logical_task=1 for 4-7), so 2 slices = 4 hosts,
        #     and the previous `max(2, dcn_diloco_parallelism, process_count())`
        #     computed 2 -- exactly half.
        #
        # We therefore compute an upper bound on the host count from the device
        # topology, and we LOG the outcome + any exception instead of silently
        # degrading to a client-only trace.
        # ------------------------------------------------------------------
        max_hosts = self._pathways_max_num_hosts()
        try:
          import pathwaysutils.profiling as pwp  # pylint: disable=import-outside-toplevel

          pwp.monkey_patch_jax()
          max_logging.log(
              f"Profiler: Pathways start_trace path={self.output_path} "
              f"max_num_hosts={max_hosts}"
          )
          jax.profiler.start_trace(
              self.output_path,
              profiler_options=self.profiling_options,
              max_num_hosts=max_hosts,
          )
        except Exception as e:  # pylint: disable=broad-except
          # Do NOT swallow this. A fallback here means client-only traces and
          # empty TPU planes, which previously looked like a mysterious
          # "profiling synchronicity" problem.
          max_logging.error(
              f"Profiler: Pathways start_trace FAILED ({type(e).__name__}: {e}). "
              "Falling back to stock jax.profiler.start_trace -- TPU worker "
              "planes will be MISSING from this trace."
          )
          max_logging.error(traceback.format_exc())
          jax.profiler.start_trace(self.output_path, profiler_options=self.profiling_options)
      else:
        jax.profiler.start_trace(self.output_path, profiler_options=self.profiling_options)
    self.is_active = True

  def maybe_deactivate_profiler(self, step, state):
    """Conditionally deactivates the profiler based on the current step.
    This method checks if the current training step matches the step designated
    for finishing the initial profile, or if it meets the criteria for
    deactivating a periodic profile.
    """
    if self.mode != "" and (step == self.finished_initial_profile_step or self.should_deactivate_periodic_profile(step)):
      self.deactivate(blocking_object=state)

  def deactivate(self, blocking_object=None):
    """End the profiler.
    The result is uploaded to the output bucket."""
    if not self.is_active:
      return
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)

    if self.managed_mldiagnostics and self.mode == "xplane":
      # Handle the special profileing logic for managed_mldiagnostics
      if self.prof is not None:
        self.prof.stop()
      self.is_active = False
      return

    if not (self.upload_all_profiler_results or jax.process_index() == 0):
      return
    if self.mode == "nsys":
      if self.libcudart is not None:
        self.libcudart.cudaProfilerStop()
      else:
        max_logging.log("WARNING: library for nsys was not loaded \n" "profiler has no effect")
        return
      # Popen() instead of run() for non-blocking behavior
      if shutil.which("gcloud") is not None:
        subprocess.Popen(["gcloud", "storage", "cp", "*nsys-rep", self.output_path])  # pylint: disable=consider-using-with
      else:
        max_logging.log("WARNING: gcloud is not installed or not found in the system's PATH. Skipping upload...")
    elif self.mode == "xplane":
      jax.profiler.stop_trace()
    self.is_active = False

  def _set_first_profiler_step(self, skip_steps, start_step):
    return start_step + skip_steps

  def _set_last_profiler_step(self, profiler_steps, last_job_step):
    return min(self.start_initial_profile_step + profiler_steps - 1, last_job_step - 1)

  def should_activate_periodic_profile(self, step):
    return self.profile_period > 0 and (step - self.start_initial_profile_step) % self.profile_period == 0

  def should_deactivate_periodic_profile(self, step):
    return self.profile_period > 0 and (step - self.finished_initial_profile_step) % self.profile_period == 0

  def post_process(self):
    pass
