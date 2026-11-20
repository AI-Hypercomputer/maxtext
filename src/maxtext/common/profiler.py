# Copyright 2023–2026 Google LLC
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

"""Dispatch to the chosen profiler and orchestrate automatic pipeline stage XProf traces.

When pipeline parallelism is enabled with ``profiler='xplane'`` and ``enable_pipeline_stage_profiling=True``,
``Profiler`` selects one representative process per pipeline stage, writes ``pp_trace_metadata/v0`` sidecars,
synchronizes trace flush across all processes via SPMD barriers, and augments Process 0's ``.xplane.pb``
in-place using ``maxtext.utils.pipeline_trace_lens``. See
``docs/guides/optimization/pipeline_parallelism_profiling.md`` for contributor and schedule-registration docs.
"""

import contextlib
import contextvars
from ctypes import cdll
import functools
import json
import os
import shutil
import socket
import subprocess
import time
from typing import Any
import weakref

from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import numpy as np

from maxtext.common.gcloud_stub import mldiagnostics_modules

mldiag, _ = mldiagnostics_modules()

from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
from maxtext.utils import max_logging
from maxtext.utils import pipeline_trace_lens

PIPELINE_TRACE_METADATA_SCHEMA = "pp_trace_metadata/v0"
PIPELINE_TRACE_METADATA_FILENAME = "pipeline_trace_metadata.json"
PIPELINE_STAGE_AXIS = "stage"


def _mesh_stage_axis(mesh, stage_axis=PIPELINE_STAGE_AXIS):
  """Returns the index of `stage_axis` in `mesh.axis_names`, or None if the mesh has no such axis."""
  if mesh is None:
    return None
  axis_names = list(getattr(mesh, "axis_names", ()) or ())
  if stage_axis not in axis_names:
    return None
  return axis_names.index(stage_axis)


def get_pipeline_stage_profile_processes(mesh, stage_axis=PIPELINE_STAGE_AXIS):
  """Returns the sorted process indices that profile one pipeline stage each, or None.

  For every slice of `mesh.devices` along `stage_axis`, the process that owns the first device of that slice is
  selected, so every stage is covered by exactly one profiled process. Returns None when the mesh has no stage axis
  or only a single stage (no pipeline parallelism), in which case the default profiling behavior applies.
  """
  st = _mesh_stage_axis(mesh, stage_axis)
  if st is None:
    return None
  devices = np.asarray(mesh.devices)
  num_stages = devices.shape[st]
  if num_stages <= 1:
    return None
  per_stage = np.moveaxis(devices, st, 0)
  return sorted({int(per_stage[s].flat[0].process_index) for s in range(num_stages)})


def num_local_chips(devices):
  """Returns the number of distinct chips among `devices` (devices sharing chip coordinates are one chip)."""
  devices = list(devices)
  coords = [getattr(d, "coords", None) for d in devices]
  if devices and all(c is not None for c in coords):
    return len({tuple(int(x) for x in c) for c in coords})
  return len(devices)


def num_local_pipeline_stages(mesh, devices, stage_axis=PIPELINE_STAGE_AXIS):
  """Returns the number of distinct pipeline stages among `devices` in `mesh`."""
  st = _mesh_stage_axis(mesh, stage_axis)
  if st is None:
    return 0
  mesh_devices = np.asarray(mesh.devices)
  if not mesh_devices.size:
    return 0
  local_ids = {int(d.id) for d in devices}
  per_stage = np.moveaxis(mesh_devices, st, 0)
  return sum(
      1 for s in range(per_stage.shape[0]) if any(d is not None and int(d.id) in local_ids for d in per_stage[s].flat)
  )


def resolve_tpu_num_chips_to_profile(
    config,
    local_chip_count,
    pipeline_stage_profiling=False,
    local_stage_count=None,
    *,
    mesh=None,
    local_devices=None,
):
  """Returns `tpu_num_chips_to_profile_per_task`, auto-raising when a process owns multiple stages."""
  requested = int(config.tpu_num_chips_to_profile_per_task)
  local_chip_count = int(local_chip_count)
  if local_stage_count is None:
    local_stage_count = (
        num_local_pipeline_stages(mesh, local_devices) if (mesh is not None and local_devices is not None) else 0
    )
  else:
    local_stage_count = int(local_stage_count)
  if pipeline_stage_profiling and local_stage_count > 1 and requested < local_chip_count:
    max_logging.log(
        f"Raising tpu_num_chips_to_profile_per_task from {requested} to {local_chip_count} "
        f"because this process owns {local_stage_count} pipeline stages across local chips."
    )
    return local_chip_count
  if config.upload_all_profiler_results and requested < local_chip_count:
    max_logging.log(
        f"enable_tpu_profiling_options=True with upload_all_profiler_results=True and"
        f" tpu_num_chips_to_profile_per_task={requested} traces {requested} of {local_chip_count} local chips;"
        " raise tpu_num_chips_to_profile_per_task if all local chips are needed."
    )
  return requested


def get_profile_output_path(root_output_path, process_index, stage_trace_staging):
  """Returns the directory passed to `jax.profiler.start_trace`.

  Process 0 always writes directly to `root_output_path` so XProf logdir discovery finds its single primary
  `.xplane.pb` cleanly. When `stage_trace_staging` is True, non-zero stage-representative processes write to
  `<root_output_path>/_pp_stage_traces/process_<idx:05d>` so processes never collide and raw stage traces do not
  clutter the root profile directory.
  """
  if not stage_trace_staging or int(process_index) == 0:
    return root_output_path
  return str(epath.Path(root_output_path) / pipeline_trace_lens.STAGE_TRACES_SUBDIR / f"process_{int(process_index):05d}")


def _broadcast_string_from_process0(value: str, max_bytes: int = 4096) -> str:
  """Broadcasts a UTF-8 string from Process 0 to all processes using a collective over a fixed-size uint8 buffer."""
  encoded = value.encode("utf-8")
  if len(encoded) > max_bytes - 4:
    raise ValueError(f"Broadcast path exceeds {max_bytes - 4} UTF-8 bytes: {len(encoded)}")
  buf = np.zeros(max_bytes, dtype=np.uint8)
  is_source = jax.process_index() == 0
  if is_source:
    length = len(encoded)
    buf[:4] = np.frombuffer(length.to_bytes(4, "little"), dtype=np.uint8)
    buf[4 : 4 + length] = np.frombuffer(encoded, dtype=np.uint8)
  out = np.asarray(multihost_utils.broadcast_one_to_all(buf, is_source=is_source), dtype=np.uint8)
  length = int.from_bytes(bytes(out[:4]), "little")
  return bytes(out[4 : 4 + length]).decode("utf-8")


def _enum_value(x):
  return str(getattr(x, "value", x))


def build_pipeline_trace_metadata(
    config,
    mesh,
    devices,
    *,
    process_index=None,
    process_count=None,
    hostname=None,
    extra_profiler_fields=None,
):
  """Builds the `pp_trace_metadata/v0` record that maps this process's devices to mesh coordinates and stages.

  Args:
    config: MaxText config (only pipeline and profiler fields are read).
    mesh: The training mesh (anything with `axis_names` and a `devices` ndarray), or None.
    devices: This process's devices in `jax.local_devices()` order; entry k corresponds to trace plane
      `/device:<platform>:k`.
    process_index: Overrides `jax.process_index()`.
    process_count: Overrides `jax.process_count()`.
    hostname: Overrides `socket.gethostname()`.
    extra_profiler_fields: Optional dict merged into the `profiler` section.

  Returns:
    A JSON-serializable dict.
  """
  stage_axis = _mesh_stage_axis(mesh)
  mesh_axes, mesh_shape, mesh_device_ids = None, None, None
  index_of_id = {}
  if mesh is not None:
    mesh_devices = np.asarray(mesh.devices)
    ids = np.vectorize(lambda d: int(d.id), otypes=[np.int64])(mesh_devices) if mesh_devices.size else mesh_devices
    mesh_axes = [str(a) for a in mesh.axis_names]
    mesh_shape = [int(x) for x in mesh_devices.shape]
    mesh_device_ids = np.asarray(ids).tolist()
    for idx in np.ndindex(*mesh_devices.shape):
      index_of_id[int(ids[idx])] = [int(i) for i in idx]

  local_devices = []
  for d in devices:
    mesh_index = index_of_id.get(int(d.id))
    coords = getattr(d, "coords", None)
    local_devices.append(
        {
            "id": int(d.id),
            "process_index": int(getattr(d, "process_index", 0)),
            "coords": [int(c) for c in coords] if coords is not None else [],
            "core_on_chip": int(getattr(d, "core_on_chip", 0) or 0),
            "device_kind": str(getattr(d, "device_kind", "")),
            "mesh_index": mesh_index,
            "stage": mesh_index[stage_axis] if (mesh_index is not None and stage_axis is not None) else None,
        }
    )

  if stage_axis is not None:
    num_stages = mesh_shape[stage_axis]
  else:
    num_stages = int(getattr(config, "ici_pipeline_parallelism", 1)) * int(getattr(config, "dcn_pipeline_parallelism", 1))

  num_repeats = int(getattr(config, "num_pipeline_repeats", 1) or 1)
  raw_schedule = _enum_value(getattr(config, "pipeline_schedule", "") or "").strip()
  if not raw_schedule:
    raw_schedule = "circular" if num_repeats > 1 else "gpipe"

  profiler_fields = {
      "profiler": _enum_value(getattr(config, "profiler", "")),
      "skip_first_n_steps_for_profiler": int(getattr(config, "skip_first_n_steps_for_profiler", 0)),
      "profiler_steps": int(getattr(config, "profiler_steps", 0)),
      "upload_all_profiler_results": bool(getattr(config, "upload_all_profiler_results", False)),
      "enable_pipeline_stage_profiling": bool(getattr(config, "enable_pipeline_stage_profiling", True)),
  }
  if extra_profiler_fields:
    profiler_fields.update(extra_profiler_fields)

  return {
      "schema": PIPELINE_TRACE_METADATA_SCHEMA,
      "hostname": hostname if hostname is not None else socket.gethostname(),
      "process_index": int(jax.process_index() if process_index is None else process_index),
      "process_count": int(jax.process_count() if process_count is None else process_count),
      "mesh_axes": mesh_axes,
      "mesh_shape": mesh_shape,
      "mesh_device_ids": mesh_device_ids,
      "stage_axis": stage_axis,
      "local_devices": local_devices,
      "pipeline": {
          "schedule": raw_schedule,
          "num_stages": int(num_stages),
          "num_microbatches": int(getattr(config, "num_pipeline_microbatches", 0) or 0),
          "num_repeats": num_repeats,
          "layers_per_stage": int(getattr(config, "num_layers_per_pipeline_stage", 1) or 1),
          "forwarding_delay": 2 if getattr(config, "pipeline_delay_activation_forwarding", False) else 1,
          "scan_pipeline_iterations": bool(getattr(config, "scan_pipeline_iterations", False)),
          "fsdp_ag_once": bool(getattr(config, "pipeline_fsdp_ag_once", False)),
      },
      "profiler": profiler_fields,
      "jax_version": jax.__version__,
      "time_unix": time.time(),
  }


def write_pipeline_trace_metadata(output_dir, metadata, filename=PIPELINE_TRACE_METADATA_FILENAME):
  """Writes `metadata` as JSON to `<output_dir>/<filename>` (local or cloud storage path) and returns the path."""
  out_dir = epath.Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_file = out_dir / filename
  out_file.write_text(json.dumps(metadata, indent=1))
  return str(out_file)


class Profiler:
  """Activate/deactivate a profiler based on the 'profiler' config."""

  def __init__(self, config, offset_step=0, mesh=None):
    self.libcudart = None
    self.config = config
    self.mesh = mesh
    self.mode = config.profiler
    if self.mode != "":
      self.base_output_dir = config.tensorboard_dir
    self.output_path = ""
    self.upload_all_profiler_results = config.upload_all_profiler_results
    self.enable_pipeline_stage_profiling = bool(getattr(config, "enable_pipeline_stage_profiling", True))
    # None keeps legacy selection: Process 0 only, or every process with upload_all_profiler_results.
    self.profile_process_indices = None
    self.pipeline_stage_profiling_active = False
    self._pp_session_active = False
    self._pp_session_root_path = ""
    self._pp_existing_xplanes: set[str] = set()
    using_pp = getattr(config, "using_pipeline_parallelism", None)
    if self.mode == "xplane" and self.enable_pipeline_stage_profiling and (using_pp is None or bool(using_pp)):
      stage_procs = get_pipeline_stage_profile_processes(mesh)
      if stage_procs is not None:
        self.profile_process_indices = stage_procs
        self.pipeline_stage_profiling_active = True
        if not self.upload_all_profiler_results:
          max_logging.log(f"Profiling one representative process per pipeline stage: processes {stage_procs}")
    self.profile_cleanly = config.profile_cleanly
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
      local_devices = jax.local_devices()
      advanced_config.update(
          {
              "tpu_num_chips_to_profile_per_task": resolve_tpu_num_chips_to_profile(
                  config,
                  num_local_chips(local_devices),
                  pipeline_stage_profiling=self.pipeline_stage_profiling_active,
                  mesh=self.mesh,
                  local_devices=local_devices,
              ),
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

  def _should_profile(self):
    """Returns whether this process captures a trace."""
    if self.upload_all_profiler_results:
      return True
    if self.profile_process_indices is not None:
      return jax.process_index() in self.profile_process_indices
    return jax.process_index() == 0

  def _managed_process_index_list(self):
    """Process selection for the managed xprof collector. None means every process."""
    if self.upload_all_profiler_results:
      return None
    if self.profile_process_indices is not None:
      return list(self.profile_process_indices)
    return [0]

  def activate(self, blocking_object=None, optional_postfix=""):
    """Start the profiler.
    nsys profiler becomes no-op when libcudart.so is not available on the system."""
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)

    if self.managed_mldiagnostics and self.mode == "xplane":
      # Handle the special profiling logic for managed_mldiagnostics
      if self.prof is None:
        # Starts xprof collector on the selected processes (process 0 by default, None for all processes).
        self.prof = mldiag.xprof(process_index_list=self._managed_process_index_list())
      self.prof.start()
      return

    root_output_path = os.path.join(self.base_output_dir, optional_postfix) if self.mode != "" else ""
    if self.pipeline_stage_profiling_active:
      # Unconditional across ALL processes so SPMD collectives never deadlock when some processes do not trace.
      if jax.process_count() > 1:
        root_output_path = _broadcast_string_from_process0(root_output_path)
      self._pp_session_root_path = root_output_path
      self._pp_session_active = True
      if jax.process_index() == 0:
        self._pp_existing_xplanes = set(
            pipeline_trace_lens.list_profile_xplanes(root_output_path, exclude_stage_subdir=True)
        )

    if not self._should_profile():
      return
    if self.mode != "":
      stage_trace_staging = self.pipeline_stage_profiling_active and not self.upload_all_profiler_results
      self.output_path = get_profile_output_path(root_output_path, jax.process_index(), stage_trace_staging)
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
      if self.pipeline_stage_profiling_active:
        self._write_pipeline_trace_metadata()

  def _write_pipeline_trace_metadata(self):
    """Writes the pipeline trace metadata sidecar next to this process's trace. Never raises."""
    try:
      process_index = jax.process_index()
      stage_trace_staging = self.pipeline_stage_profiling_active and not self.upload_all_profiler_results
      metadata = build_pipeline_trace_metadata(
          self.config,
          self.mesh,
          jax.local_devices(),
          process_index=process_index,
          extra_profiler_fields={
              "profile_process_indices": self.profile_process_indices,
              "output_path": self.output_path,
          },
      )
      # Process 0 (and flat upload-all layouts) share the root dir, so include process index in the filename;
      # staged per-process stage directories use a single canonical filename.
      filename = (
          PIPELINE_TRACE_METADATA_FILENAME
          if (stage_trace_staging and process_index != 0)
          else f"pipeline_trace_metadata_process_{process_index:05d}.json"
      )
      path = write_pipeline_trace_metadata(self.output_path, metadata, filename=filename)
      max_logging.log(f"Wrote pipeline trace metadata to {path}")
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"WARNING: Failed to write pipeline trace metadata: {e}")

  def _augment_host0_pipeline_trace(self):
    """Merges decoded cross-stage pipeline timeline into Process 0's `.xplane.pb` in-place. Never raises."""
    try:
      host0_xplane = pipeline_trace_lens.find_host0_xplane(
          self._pp_session_root_path,
          existing_paths=self._pp_existing_xplanes,
      )
      if host0_xplane is None:
        max_logging.log(
            f"WARNING: Pipeline trace augmentation skipped: no .xplane.pb found under {self._pp_session_root_path}"
        )
        return
      stats = pipeline_trace_lens.augment_host0_xplane_in_place(
          host0_xplane,
          search_dir=self._pp_session_root_path,
      )
      fallback_note = f" fallback=True ({stats['fallback_reason']})" if stats.get("fallback_mode") else ""
      max_logging.log(
          f"Augmented Process 0 trace in-place ({stats['host0_xplane_path']}): "
          f"schedule={stats['schedule']} S={stats['num_stages']} M={stats['num_microbatches']} "
          f"stages_measured={stats['stages_measured']} planes_added={stats['planes_added']} "
          f"events_added={stats['events_added']} align_p95={stats['align_p95_us']}us "
          f"elapsed={stats['elapsed_s']}s{fallback_note}"
      )
    except Exception as e:  # pylint: disable=broad-except
      max_logging.log(f"WARNING: Failed to augment Process 0 trace with pipeline timeline: {e}")

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

    pp_session_active = self._pp_session_active
    self._pp_session_active = False

    if self._should_profile():
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

    if pp_session_active and self.mode == "xplane":
      # Unconditional across ALL processes so non-profiled processes do not deadlock waiting on the barrier.
      if jax.process_count() > 1:
        multihost_utils.sync_global_devices("maxtext_pp_profile_flush")
      if jax.process_index() == 0:
        self._augment_host0_pipeline_trace()

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
