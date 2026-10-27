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

import contextlib
import contextvars
from ctypes import cdll
import functools
import os
import shutil
import subprocess
from typing import Any
import weakref

from flax import nnx
import jax
from jax.experimental import multihost_utils

from maxtext.common.gcloud_stub import mldiagnostics_modules

mldiag, _ = mldiagnostics_modules()

from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
from maxtext.utils import max_logging


class Profiler:
  """Activate/deactivate a profiler based on the 'profiler' config."""

  def __init__(self, config, offset_step=0):
    self.libcudart = None
    self.mode = config.profiler
    if self.mode != "":
      self.base_output_dir = config.tensorboard_dir
    self.output_path = ""
    self.upload_all_profiler_results = config.upload_all_profiler_results
    self.profile_cleanly = config.profile_cleanly
    self.profiler_sync_delay_ms = config.profiler_sync_delay_ms
    self.xplane_trace_active = False
    self.profile_period = config.profile_periodically_period
    self.start_initial_profile_step = self._set_first_profiler_step(config.skip_first_n_steps_for_profiler, offset_step)
    self.finished_initial_profile_step = self._set_last_profiler_step(config.profiler_steps, config.steps)
    if config.profiler != "" and self.start_initial_profile_step >= config.steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    self.prof = None  # managed mldiagnostics xprof collector.
    self.managed_mldiagnostics = config.managed_mldiagnostics
    if config.managed_mldiagnostics:
      ManagedMLDiagnostics(config)  # Initialize the MLRun instance.

    self.profiling_options = jax.profiler.ProfileOptions()
    advanced_config = {}

    if self.mode == "xplane" and not self.managed_mldiagnostics and config.profile_power_events:
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
          }
      )

    if self.mode == "xplane" and config.enable_continuous_profiling:
      advanced_config["enable_continuous_profiling"] = True

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

  def activate(self, blocking_object=None, optional_postfix=""):
    """Start the profiler.
    nsys profiler becomes no-op when libcudart.so is not available on the system."""
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)
    self._start(optional_postfix)
    if self.mode != "" and self.profiler_sync_delay_ms >= 0:
      self._sync_devices()

  def _sync_devices(self):
    """Synchronizes the devices of all processes once the profiler has started.

    Starting the profiler takes a different amount of time in each process;
    without a barrier, the first collective of the profiled step would absorb
    that skew. The device-side delay before the barrier lets each host enqueue
    the profiled step before the devices start it, as in steady state. Called in
    every process, including those that do not profile, and does not wait for
    the devices.
    """
    if not hasattr(multihost_utils, "wait_and_sync_devices_async"):
      max_logging.warning("This version of JAX cannot synchronize devices after starting the profiler")
      return
    multihost_utils.wait_and_sync_devices_async(delay_ms=self.profiler_sync_delay_ms)

  def _start(self, optional_postfix):
    """Starts the profiler, in the processes that profile."""
    if self.managed_mldiagnostics and self.mode == "xplane":
      # Handle the special profiling logic for managed_mldiagnostics
      if self.prof is None:
        # Starts xprof collector.
        # Only profiling on the first device, if not upload_all_profiler_results. None is for all devices.
        self.prof = mldiag.xprof(process_index_list=None if self.upload_all_profiler_results else [0])
      self.prof.start()
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
      jax.profiler.start_trace(self.output_path, profiler_options=self.profiling_options)
      self.xplane_trace_active = True

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
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)

    if self.managed_mldiagnostics and self.mode == "xplane":
      # Handle the special profileing logic for managed_mldiagnostics
      if self.prof is not None:
        self.prof.stop()
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
    elif self.mode == "xplane" and self.xplane_trace_active:
      # train_loop also calls this when training stops early, possibly after the trace has ended.
      jax.profiler.stop_trace()
      self.xplane_trace_active = False

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


_NNX_NAMED_SCOPE_INSTALLED = False
_NNX_NAMED_SCOPE_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar("nnx_named_scope_enabled", default=False)
# Keyed by id() rather than the module itself: dataclass modules (e.g. attentions.L2Norm) set
# __hash__ = None, so a WeakKeyDictionary cannot hold them. Entries are evicted via weakref.finalize.
_MODULE_SCOPE_NAMES: dict[int, str] = {}
_WRAPPED_CALL_FUNCS: set[Any] = set()
_ACTIVE_SCOPED_MODULES: contextvars.ContextVar[tuple[int, ...]] = contextvars.ContextVar(
    "active_scoped_modules", default=()
)


def _make_scoped_call(cls, orig_call):
  """Wraps an nnx.Module.__call__ implementation in a jax.named_scope."""

  @functools.wraps(orig_call)
  def _wrapped_call(self, *args, **kwargs):
    active = _ACTIVE_SCOPED_MODULES.get()
    if not _NNX_NAMED_SCOPE_ENABLED.get() or (active and active[-1] == id(self)):
      return orig_call(self, *args, **kwargs)
    scope = _MODULE_SCOPE_NAMES.get(id(self)) or cls.__name__
    token = _ACTIVE_SCOPED_MODULES.set(active + (id(self),))
    try:
      with jax.named_scope(scope):
        return orig_call(self, *args, **kwargs)
    finally:
      _ACTIVE_SCOPED_MODULES.reset(token)

  _WRAPPED_CALL_FUNCS.add(_wrapped_call)
  return _wrapped_call


def _wrap_nnx_cls(cls):
  """Recursively wraps __call__ on cls and its existing nnx.Module subclasses."""
  call_fn = cls.__dict__.get("__call__")
  if call_fn is not None and call_fn not in _WRAPPED_CALL_FUNCS:
    setattr(cls, "__call__", _make_scoped_call(cls, call_fn))
  for sub in cls.__subclasses__():
    _wrap_nnx_cls(sub)


@contextlib.contextmanager
def nnx_named_scope_context(enabled: bool = True):
  """Wraps NNX module calls in hierarchical jax.named_scope blocks for XProf traces."""
  global _NNX_NAMED_SCOPE_INSTALLED  # pylint: disable=global-statement
  if enabled and not _NNX_NAMED_SCOPE_INSTALLED:
    orig_getattribute = nnx.Module.__getattribute__

    def _scoped_getattribute(self, name):
      val = orig_getattribute(self, name)
      if _NNX_NAMED_SCOPE_ENABLED.get() and not name.startswith("_") and isinstance(val, nnx.Module):
        key = id(val)
        if key not in _MODULE_SCOPE_NAMES:
          try:
            weakref.finalize(val, _MODULE_SCOPE_NAMES.pop, key, None)
          except TypeError:  # Not weak-referenceable; skip rather than risk a stale id entry.
            return val
        _MODULE_SCOPE_NAMES[key] = name
      return val

    nnx.Module.__getattribute__ = _scoped_getattribute  # type: ignore[method-assign]

    orig_init_subclass = nnx.Module.__init_subclass__.__func__

    def _patched_init_subclass(cls, **kwargs):
      orig_init_subclass(cls, **kwargs)
      _wrap_nnx_cls(cls)

    nnx.Module.__init_subclass__ = classmethod(_patched_init_subclass)  # type: ignore[method-assign,assignment]
    _wrap_nnx_cls(nnx.Module)
    _NNX_NAMED_SCOPE_INSTALLED = True
  elif enabled:
    _wrap_nnx_cls(nnx.Module)

  token = _NNX_NAMED_SCOPE_ENABLED.set(enabled)
  try:
    yield
  finally:
    _NNX_NAMED_SCOPE_ENABLED.reset(token)
