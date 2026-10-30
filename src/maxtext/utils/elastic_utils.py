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

"""Utility functions for Elastic Training."""

from collections import Counter
import dataclasses
import functools
from types import SimpleNamespace
from typing import Any, Callable

import jax
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
import pathwaysutils
from pathwaysutils.elastic import elastic
from pathwaysutils.elastic import manager

elastic_manager: manager.Manager | None = None
pending_reinit_recorder = None
pending_elastic_event_type = None


def record_slice_state(recorder, active_slices_override: int | None = None) -> None:
  """Records live slice counts and logs them to the GoodputRecorder."""
  if (
      recorder is None
      or not hasattr(recorder, "record_elastic_slice_counts")
      or not pathwaysutils.is_pathways_backend_used()
      or elastic_manager is None
  ):
    return

  available_slices = len(elastic.get_active_slice_indices())
  active_slices = (
      active_slices_override if active_slices_override is not None else len(elastic_manager.active_slice_indices)
  )
  total_slices = len(elastic.get_slice_to_devices(jax.devices()))

  recorder.record_elastic_slice_counts(
      available_slices=available_slices,
      active_slices=active_slices,
      total_slices=total_slices,
  )


def record_elastic_event_start(recorder, scale_up: bool) -> None:
  """Records the start of an elastic event.

  Args:
    recorder: Goodput recorder.
    scale_up: True if the attempt was interrupted by `maybe_elastic_scale_up` because new slices are available, False
      if it failed because a slice went down.
  """
  global pending_elastic_event_type
  if scale_up:
    event_type = "elastic_scale_up"
  else:
    event_type = "elastic_slice_down"
  pending_elastic_event_type = event_type
  if recorder and hasattr(recorder, "record_elastic_wait_start_time"):
    recorder.record_elastic_wait_start_time(event_type=event_type)
    record_slice_state(recorder, active_slices_override=0)


def record_elastic_wait_end_and_reinit_start(recorder) -> None:
  """Records end of elastic slice event and start of reinitialization event."""
  global pending_reinit_recorder, pending_elastic_event_type
  if pending_elastic_event_type is None:
    return
  event_type = pending_elastic_event_type
  pending_elastic_event_type = None
  if recorder and hasattr(recorder, "record_elastic_wait_end_time"):
    recorder.record_elastic_wait_end_time(event_type=event_type)
    recorder.record_elastic_reinit_start_time()
    record_slice_state(recorder)
  pending_reinit_recorder = recorder


def record_elastic_reinit_end() -> None:
  """Records end of elastic reinitialization event."""
  global pending_reinit_recorder
  if pending_reinit_recorder is not None and hasattr(pending_reinit_recorder, "record_elastic_reinit_end_time"):
    pending_reinit_recorder.record_elastic_reinit_end_time()
    record_slice_state(pending_reinit_recorder)
  pending_reinit_recorder = None


def elastic_enabled(config) -> bool:
  """Returns whether elastic mode is enabled."""
  return pathwaysutils.is_pathways_backend_used() and config.elastic_enabled


def elastic_snapshot(config) -> bool:
  """Returns whether elastic snapshot mode is enabled."""
  return elastic_enabled(config) and config.elastic_backup_kind == "snapshot"


def maybe_bubble_elastic_exception(config, e: Exception) -> None:
  """Checks JAX/ScaleUp elastic errors and re-raises them if elasticity is enabled.

  Args:
    config: Maxtext configuration object.
    e: The exception currently being evaluated.
  """
  if elastic_enabled(config) and isinstance(e, (jax.errors.JaxRuntimeError, manager.ScaleUpSignalError)):
    raise e


def should_use_elastic(config) -> bool:
  """Returns whether elastic training should be used."""
  return config is not None and elastic_enabled(config)


def clean_up_incomplete_checkpoints(checkpoint_dir: str):
  """Cleans up incomplete checkpoints after an elastic event."""
  max_logging.log("Elastic utils: Checking for incomplete checkpoint after an elastic event...")
  checkpoint_dir = gcs_utils.add_trailing_slash(checkpoint_dir)

  # 1. List the "directories" (steps)
  checkpoints = gcs_utils.gcs_list_directories(checkpoint_dir)

  # 2. Filter for directories that are numbers
  checkpoints = [cp for cp in checkpoints if cp.isdigit()]

  if not checkpoints:
    max_logging.log("Found no existing checkpoints. Continuing")
    return

  # Sort naturally (numerical sort) and get the last one
  checkpoints.sort(key=int)
  latest_checkpoint_name = checkpoints[-1]
  latest_checkpoint_path = f"{checkpoint_dir}{latest_checkpoint_name}/"

  max_logging.log(f"Checking latest checkpoint: {latest_checkpoint_path}")

  # 3. Check for commit_success file
  success_markers = gcs_utils.gcs_glob_pattern(f"{latest_checkpoint_path}commit_success*")

  if not success_markers:
    max_logging.log(f"No commit_success file found. Deleting {latest_checkpoint_path}...")
    # TODO: Use Orbax 'Cancel Ongoing Checkpointing' API when available to
    # prevent deleting a checkpoint that is currently being written.
    gcs_utils.gcs_delete_directory(latest_checkpoint_path)
  else:
    max_logging.log(f"Found commit_success file. Keeping {latest_checkpoint_path}.")


def ensure_elastic_manager_initialized(config):
  """Initializes elastic manager if it's not initialized and pathways is used."""
  global elastic_manager
  if should_use_elastic(config) and elastic_manager is None:
    elastic_manager = manager.Manager()


def is_pause_resume(config) -> bool:
  """Returns whether every attempt waits for all the slices in the jobset, i.e. pause/resume rather than resize.

  In pause/resume every attempt comes back on the same slices, so the compiled steps still fit and the setup of the
  previous attempt can be reused. `elastic_min_slice_count` is -1 (wait for every slice) or equals the number of
  slices in the jobset. `total_slice_count` counts every slice in the jobset, live or not; `config.num_slices` can't
  be used here because it only counts the live slices.

  Args:
    config: Config object.
  """
  ensure_elastic_manager_initialized(config)
  assert elastic_manager is not None
  if config.elastic_min_slice_count == -1:
    return True
  return config.elastic_min_slice_count == elastic_manager.total_slice_count


def get_local_batch_size(config) -> int:
  """Returns the local batch size based on the config."""
  return config.per_device_batch_size * get_devices_per_host(config)


def live_devices(config=None):
  """Returns the list of live devices."""
  # If pathways is not used or elastic_manager is not initialized, return all devices
  if should_use_elastic(config):
    ensure_elastic_manager_initialized(config)
    assert elastic_manager is not None
    # Filter devices that are in active slices
    return [
        d for d in jax.devices() if d is not None and getattr(d, "slice_index", 0) in elastic_manager.active_slice_indices
    ]
  return jax.devices()


def live_slice_indices(config) -> set[int]:
  """Returns the set of live slice indices."""
  return {getattr(d, "slice_index", 0) for d in live_devices(config) if d is not None}


def get_devices_per_host(config):
  """Dynamically calculates the number of chips per physical worker VM."""
  devices = Counter(d.task_id for d in live_devices(config))

  max_logging.log(f"elastic_utils: Device counts per task: {devices}")
  if not devices:
    raise ValueError("elastic_utils: get_devices_per_host: No devices found.")

  devices_per_host = next(iter(devices.values()))
  if devices_per_host == 0:
    raise ValueError("elastic_utils: get_devices_per_host: Devices per host is 0.")
  max_logging.log(f"elastic_utils: Devices per host: {devices_per_host}")

  return devices_per_host


def chain_callbacks(*funcs):
  """Helper function to chain callbacks."""

  def wrapper():
    for func in funcs:
      func()

  return wrapper


class RetryCache:
  """Setup objects that a retry on the same slices can reuse, e.g. the config, mesh and compiled steps.

  `elastic_retry` turns the cache on for the duration of one elastic run and empties it whenever the active slices
  change, so an entry is only ever returned to an attempt on the same slices as the attempt that stored it.

  In practice only pause/resume (see `is_pause_resume`) gets cache hits: every attempt runs on the same slices, so
  the config, mesh and compiled steps of the previous attempt are reused. In replica-resize mode every elastic event
  changes the active slices, so the cache is emptied before the next attempt and everything is rebuilt. Every
  elastic run still gets a cache, so both modes share one code path.

  Don't cache the train state or other large arrays: they would keep the previous attempt's memory alive. The one
  deliberate exception is `snapshotter`, which is not a cache entry: it is set once for the run, outlives every
  attempt and every slice change, and is reset by the elastic event callback rather than by this class.
  """

  def __init__(self):
    # Whether `get` and `put` do anything. Only `elastic_retry` changes it: it is set to True when the decorated
    # function starts and back to False in its `finally` block, once the run is over. While it is False, `get`
    # always misses and `put` is a no-op, so a `train_loop` that is not running under `elastic_retry` never caches.
    self._enabled = False
    self._slices: frozenset[int] | None = None
    self._values: dict[str, Any] = {}
    # Host-memory snapshots of the train state for `elastic_backup_kind: snapshot`. Not wired up yet: it stays None
    # until snapshot save and restore land, and `reset` and `clear_if_slices_changed` never touch it.
    self.snapshotter: Any = None

  @property
  def enabled(self) -> bool:
    return self._enabled

  def reset(self, enabled: bool) -> None:
    """Empties the cache and turns it on or off."""
    self._enabled = enabled
    self._slices = None
    self._values.clear()

  def clear_if_slices_changed(self, active_slices: frozenset[int]) -> None:
    """Empties the cache if this attempt runs on different slices than the previous one.

    Args:
      active_slices: The slices the attempt about to start runs on.
    """
    if self._slices != active_slices:
      self._values.clear()
      self._slices = active_slices

  def contains(self, name: str) -> bool:
    """Returns whether an earlier attempt on the same slices cached an object under `name`.

    Use this rather than `get(name) is None` to test for a hit: a cached value may itself be None.
    """
    return self._enabled and name in self._values

  def get(self, name: str) -> Any:
    """Returns the object cached under `name` by an earlier attempt on the same slices, or None.

    None is also returned when the cache is off or the entry is missing; use `contains` to tell these apart.
    """
    if not self._enabled:
      return None
    return self._values.get(name)

  def put(self, name: str, value: Any) -> None:
    """Caches `value` under `name`, so that a retry on the same slices can reuse it. A no-op while the cache is off.

    Args:
      name: The name to cache `value` under.
      value: The object to reuse, for example a mesh or a jitted step.
    """
    if self._enabled:
      self._values[name] = value


def get_or_build(retry_cache: RetryCache | None, name: str, build_fn: Callable[[], Any]) -> Any:
  """Returns the object cached under `name`, or builds it with `build_fn` and caches it for the next attempt.

  Args:
    retry_cache: The cache of the current elastic run, or None when training is not elastic. With None, `build_fn`
      is simply called.
    name: The name the object is cached under.
    build_fn: Builds the object when there is no cached one.
  """
  if retry_cache is None:
    return build_fn()
  if retry_cache.contains(name):
    return retry_cache.get(name)
  value = build_fn()
  retry_cache.put(name, value)
  return value


@dataclasses.dataclass
class _AttemptOutcome:
  """Why the last attempt of an elastic run ended, for the elastic event callback.

  pathwaysutils resets `available_inactive_slices` before it runs the elastic event callback, so the callback can't
  tell a scale-up from a slice-down by looking at the manager. The attempt itself records it here on the way out.
  """

  scale_up: bool = False


def elastic_retry(config, callback_fn=None, pre_callback_fn=None, retry_cache=None):
  """Decorator for elastic retry.

  If an elastic event occurs, the decorator will retry the decorated function
  up to `config.elastic_max_retries` times.
  Before each retry, it cleans up partial checkpoints by calling
  `clean_up_incomplete_checkpoints`. If `callback_fn` is provided, it is
  called after `clean_up_incomplete_checkpoints`.

  The decorator covers both ways of coming back from an elastic event:

  *   Pause/resume: `elastic_min_slice_count` equals the number of slices in the jobset, or is -1, which means the
      same. Every attempt waits for all the slices, so it always comes back on the same slices and the compiled steps
      cached in `retry_cache` are not compiled again.
  *   Replica resize: `elastic_min_slice_count` is smaller than the number of slices in the jobset. An attempt can
      come back on fewer slices (slice-down) or on more (scale-up). The retry cache is emptied whenever the slices
      change, so the next attempt rebuilds the config, mesh and compiled steps for the slices it has.

  The device memory of a failed attempt is freed by JAX once the attempt has unwound and nothing references its
  arrays any more; nothing is deleted explicitly.

  Args:
    config: Config object.
    callback_fn: Optional callback called after `clean_up_incomplete_checkpoints` on an elastic event, as
      `callback_fn(scale_up=...)`: True if the attempt was interrupted by `maybe_elastic_scale_up` because new slices
      are available, False if a slice went down.
    pre_callback_fn: Optional callback function to be called before each attempt, once the slices are ready.
    retry_cache: Optional `RetryCache` shared with the decorated function. It is turned on for the duration of the
      elastic run and emptied whenever the active slices change. The caller owns it rather than `config` because
      the config built for an attempt is itself one of the cached objects. Without it nothing is reused.

  Returns:
    A decorator for elastic retry.
  """
  if not elastic_enabled(config):
    msg = (
        "Elastic training requires the Pathways backend, and elastic_enabled"
        " must be set to True: current config.elastic_enabled:"
        f" {config.elastic_enabled}, pathways backend used:"
        f" {pathwaysutils.is_pathways_backend_used()}"
    )
    raise ValueError(msg)

  max_logging.log("Elastic Retry Enabled")

  ensure_elastic_manager_initialized(config)
  assert elastic_manager is not None

  def cleanup_iterators_and_checkpoints():
    # pylint: disable=import-outside-toplevel, broad-exception-caught
    try:
      from maxtext.input_pipeline.multihost_dataloading import cleanup_all_iterators

      cleanup_all_iterators()
    except Exception as e:
      max_logging.log(f"Failed to cleanup iterators during elastic scale up: {e}")
    clean_up_incomplete_checkpoints(config.checkpoint_dir)

  outcome = _AttemptOutcome()

  def effective_callback():
    scale_up = outcome.scale_up
    # `scale_up` only decides the goodput label of this event (`elastic_scale_up` vs `elastic_slice_down`). Only
    # `attempt` sets it, and only to True when the attempt raised `ScaleUpSignalError`; a slice-down leaves it
    # untouched. Clear it here so it describes just the event being reported: otherwise a slice-down that follows a
    # scale-up in the same run would be reported to goodput as a second scale-up.
    outcome.scale_up = False
    cleanup_iterators_and_checkpoints()
    if callback_fn is not None:
      callback_fn(scale_up=scale_up)

  def effective_pre_callback():
    if retry_cache is not None:
      # Objects built for other slices are stale after a resize.
      retry_cache.clear_if_slices_changed(frozenset(elastic_manager.active_slice_indices))
    if pre_callback_fn is not None:
      pre_callback_fn()

  if config.elastic_min_slice_count == -1:
    minimum_slice_count = None  # Wait for every slice in the jobset.
  else:
    minimum_slice_count = config.elastic_min_slice_count
  if minimum_slice_count is not None and minimum_slice_count > elastic_manager.total_slice_count:
    raise ValueError(
        f"elastic_min_slice_count ({minimum_slice_count}) is larger than the number of slices in the jobset"
        f" ({elastic_manager.total_slice_count})."
    )
  if not is_pause_resume(config) and config.dataset_type == "grain" and not config.grain_use_elastic_iterator:
    # The regular grain iterator checkpoints one state file per host, which can't be restored onto a different
    # number of hosts. Only `ElasticIterator`, whose state is a single global position, survives a resize.
    raise ValueError(
        "Replica resize (elastic_min_slice_count smaller than the number of slices in the jobset) restores the data"
        " iterator onto a different number of hosts, which only grain's ElasticIterator supports. Set"
        " grain_use_elastic_iterator=True."
    )

  pathways_retry = elastic_manager.elastic_retry(
      max_retries=config.elastic_max_retries,
      timeout=config.elastic_timeout_seconds,
      minimum_slice_count=minimum_slice_count,
      pre_callback=effective_pre_callback,
      on_elastic_event_callback=effective_callback,
  )

  def decorator(func):
    @functools.wraps(func)
    def attempt(*args, **kwargs):
      try:
        return func(*args, **kwargs)
      except manager.ScaleUpSignalError:
        # pathwaysutils runs the elastic event callback next; let it know this was a scale-up.
        outcome.scale_up = True
        raise

    retried_func = pathways_retry(attempt)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      if retry_cache is not None:
        # The cache lives for one elastic run, so a later run in the same process starts empty.
        retry_cache.reset(enabled=True)
      try:
        return retried_func(*args, **kwargs)
      finally:
        if retry_cache is not None:
          retry_cache.reset(enabled=False)
        # A scale-up that ended the run (retries exhausted) must not label the next run's first event.
        outcome.scale_up = False

    return wrapper

  return decorator


def is_scale_up_event(config) -> bool:
  """Returns whether a scale up event is detected."""
  if elastic_enabled(config):
    ensure_elastic_manager_initialized(config)
    assert elastic_manager is not None
    return bool(elastic_manager.available_inactive_slices)

  return False


def maybe_elastic_scale_up(config, checkpoint_manager):
  """Waits for a checkpoint to finish before interrupting for scale up."""
  if not should_use_elastic(config):
    max_logging.log("maybe_elastic_scale_up: Elastic training is not enabled.")
    return
  if is_scale_up_event(config):
    max_logging.log(
        "Started a checkpoint and a new slice is available. Waiting for current"
        " checkpoint to finish before interrupting."
    )
    if checkpoint_manager is not None:
      # The v1 Checkpointer exposes `.wait()`, the v0 emergency/replicator
      # managers expose `.wait_until_finished()`; this module cannot import
      # `checkpointing`'s dispatcher (checkpointing imports elastic_utils).
      if hasattr(checkpoint_manager, "wait"):
        checkpoint_manager.wait()
      else:
        checkpoint_manager.wait_until_finished()
    max_logging.log("Checkpoint save completed. Interrupting")
    # pylint: disable=import-outside-toplevel, broad-exception-caught
    try:
      from maxtext.input_pipeline.multihost_dataloading import cleanup_all_iterators

      cleanup_all_iterators()
    except Exception as e:
      max_logging.log(f"Error in maybe_elastic_scale_up cleanup: {e}")
    raise manager.ScaleUpSignalError()


def single_controller_mtc_init_kwargs(raw_keys):
  """Returns topology kwargs for single-controller MTC initialization."""
  kwargs = {
      "data_parallelism": raw_keys["mtc_data_parallelism"],
      "num_slices": raw_keys["num_slices"],
  }
  if not raw_keys.get("elastic_enabled", False):
    return kwargs

  config = SimpleNamespace(**raw_keys)
  if not should_use_elastic(config):
    return kwargs

  active_devices = tuple(live_devices(config))
  active_slice_indices = {getattr(device, "slice_index", 0) for device in active_devices if device is not None}
  if not active_devices or not active_slice_indices:
    raise ValueError("Elastic single-controller MTC initialization found no active devices.")

  kwargs["devices"] = active_devices
  kwargs["num_slices"] = len(active_slice_indices)
  if not kwargs["data_parallelism"]:
    kwargs["data_parallelism"] = kwargs["num_slices"]
  max_logging.log(
      "Using active elastic devices for single-controller MTC initialization: "
      f"active_num_slices={kwargs['num_slices']}, "
      f"active_device_count={len(active_devices)}, "
      f"configured_num_slices={raw_keys['num_slices']}."
  )
  return kwargs
