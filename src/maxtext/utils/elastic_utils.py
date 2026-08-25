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

import functools
import math
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from maxtext.common import train_state_nnx
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
import pathwaysutils
from pathwaysutils.elastic import elastic
from pathwaysutils.elastic import manager

elastic_manager: manager.Manager | None = None
pending_reinit_recorder = None
pending_elastic_event_type = None


def maybe_snapshot_state(
    elastic_mgr: Any,
    step: int,
    state: Any,
    force: bool = False,
    block: bool = False,
) -> None:
  """Takes an elasticity snapshot of TrainStateNNX or Linen TrainState."""
  if isinstance(state, train_state_nnx.TrainStateNNX):
    model_state = nnx.state(state.model)
    opt_state = nnx.state(state.optimizer)
    snapshot_jax_arrays = {
        "model": nnx.to_pure_dict(model_state),
        "optimizer": nnx.to_pure_dict(opt_state),
    }
  else:
    linen_dict = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
    }
    snapshot_jax_arrays = train_state_nnx.from_linen_checkpoint_dict(linen_dict)

  elastic_mgr.maybe_snapshot(
      step=step,
      snapshot_jax_arrays=snapshot_jax_arrays,
      force=force,
      block=block,
  )


def restore_resharded_state(elastic_mgr: Any, mesh: Any, state: Any):
  """Restores state from an elasticity snapshot on a new mesh."""
  step, snapshot_jax_arrays, _ = elastic_mgr.get_resharded_snapshot(mesh)

  if isinstance(state, train_state_nnx.TrainStateNNX):
    if "model" in snapshot_jax_arrays:
      nnx.update(state.model, snapshot_jax_arrays["model"])
    if "optimizer" in snapshot_jax_arrays:
      nnx.update(state.optimizer, snapshot_jax_arrays["optimizer"])
      state.optimizer.step.value = jnp.asarray(step, dtype=jnp.uint32)
  else:
    linen_dict = train_state_nnx.to_linen_checkpoint_dict(snapshot_jax_arrays)
    state = state.replace(**linen_dict)
    state = state.replace(step=state.step.at[None].set(step))

  return step, state



def record_slice_state(recorder, active_slices_override: int | None = None) -> None:
  """Records live slice counts and logs them to the GoodputRecorder."""
  if (
      recorder is None
      or not hasattr(recorder, "record_elastic_slice_counts")
      or not pathwaysutils.is_pathways_backend_used()
      or elastic_manager is None
  ):
    return

  try:
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
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"Goodput: record_slice_state failed to record slice counts: {e}")


def record_elastic_event_start(recorder, config) -> None:
  """Records start of an elastic scale up event."""
  global pending_elastic_event_type
  event_type = "elastic_scale_up" if is_scale_up_event(config) else "elastic_slice_down"
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


def should_use_elastic(config) -> bool:
  """Returns whether elastic training should be used."""
  return config is not None and elastic_enabled(config)


def clean_up_checkpoints(checkpoint_dir: str):
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
  """Initializes elastic manager and waits for slices if not initialized and pathways is used."""
  global elastic_manager
  if should_use_elastic(config) and elastic_manager is None:
    min_slices = config.elastic_min_slice_count
    if min_slices <= 0:
      min_slices = config.num_slices
    all_devices = jax.devices()
    slice_to_devices = elastic.get_slice_to_devices(all_devices)
    if min_slices <= 0:
      min_slices = len(slice_to_devices)
    timeout = config.elastic_timeout_seconds
    max_logging.log(f"[*] Waiting for {min_slices} slices to be active before initializing config...")
    all_active_slices = elastic.wait_for_slices(
        slice_count=min_slices,
        slice_to_devices=slice_to_devices,
        timeout=timeout,
    )

    max_logging.log("[*] Pathways Elastic Training enabled. Initializing Pathways Manager...")
    elastic_manager = manager.Manager()
    if all_active_slices:
      elastic_manager.active_slice_indices = all_active_slices
    if elastic_manager.active_slice_indices:
      jax.config.update("jax_default_device", elastic_manager.default_device)

def mutate_config_for_topology(config, el_manager):
  """Dynamically mutate the config to match the degraded slice topology."""
  new_slice_count = el_manager.active_slice_count
  max_logging.log(
      f"[*] Dynamically mutating config.num_slices and "
      f"config.dcn_data_parallelism to: {new_slice_count}"
  )
  object.__setattr__(config, "num_slices", new_slice_count)
  object.__setattr__(config, "dcn_data_parallelism", new_slice_count)

  # Update DCN data parallel axis in dcn_parallelism list
  if hasattr(config, "mesh_axes") and hasattr(config, "dcn_parallelism"):
    if "data" in config.mesh_axes:
      data_axis_idx = config.mesh_axes.index("data")
      config.dcn_parallelism[data_axis_idx] = new_slice_count

  # Recalculate num_target_devices and batch sizes for the new topology
  new_num_devices = len([
      d for d in jax.devices() if getattr(d, "slice_index", 0) in el_manager.active_slice_indices
  ])
  recalculate_batch_sizes(config, new_num_devices)

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
    active_devices = [
        d for d in jax.devices() if d.slice_index in elastic_manager.active_slice_indices
    ]
    return sorted(active_devices, key=lambda d: (d.slice_index, d.process_index))

  return jax.devices()


def live_slice_indices(config) -> set[int]:
  """Returns the set of live slice indices."""
  return {d.slice_index for d in live_devices(config)}


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


def wait_for_devices_placed(config, timeout: float = 60.0, poll_interval: float = 1.0) -> list[jax.Device]:
  """Actively polls until surviving TPU devices are placed and ready for tensor transfers.

  If another slice drops mid-poll, it dynamically updates active slices and checks min_slice_count.
  """
  start_time = time.time()
  ensure_elastic_manager_initialized(config)
  assert elastic_manager is not None
  min_slices = config.elastic_min_slice_count if config.elastic_min_slice_count > 0 else 1

  while time.time() - start_time < timeout:
    try:
      # Reuse live_devices helper safely inside try...except
      active_devices = live_devices(config)

      if len(elastic_manager.active_slice_indices) < min_slices or not active_devices:
        max_logging.log(f"Active slices ({len(elastic_manager.active_slice_indices)}) < min ({min_slices}). Waiting for slices...")
        time.sleep(poll_interval)
        continue

      test_val = np.zeros(len(active_devices), dtype=np.float32)
      sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(np.array(active_devices), ("d",)),
          jax.sharding.PartitionSpec("d"),
      )
      arr = jax.device_put(test_val, sharding)
      jax.block_until_ready(arr)
      arr.delete()
      max_logging.log(f"Confirmed {len(active_devices)} devices on slices {elastic_manager.active_slice_indices} are placed and ready.")
      return active_devices
    except Exception as e:
      max_logging.log(f"Waiting for Pathways device placement to stabilize ({e}). Retrying poll...")
      try:
        elastic_manager.active_slice_indices = elastic.get_active_slice_indices(elastic_manager.slice_to_devices)
      except Exception:
        pass
      time.sleep(poll_interval)

  return live_devices(config)


def elastic_retry(config, callback_fn=None, pre_callback_fn=None):
  """Decorator for elastic retry.

  If an elastic event occurs, the decorator will retry the decorated function
  up to `config.elastic_max_retries` times.
  Before each retry, it cleans up partial checkpoints by calling
  `clean_up_checkpoints`. If `callback_fn` is provided, it is
  called after `clean_up_checkpoints`.

  Args:
    config: Config object.
    callback_fn: Optional callback function to be called after
      `clean_up_checkpoints` on an elastic event.

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

  cleanup_partial = functools.partial(clean_up_checkpoints, config.checkpoint_dir)

  if callback_fn is None:
    effective_callback = cleanup_partial
  else:
    effective_callback = chain_callbacks(cleanup_partial, callback_fn)

  def effective_pre_callback():
    wait_for_devices_placed(config)
    if pre_callback_fn is not None:
      pre_callback_fn()

  return elastic_manager.elastic_retry(
      max_retries=config.elastic_max_retries,
      timeout=config.elastic_timeout_seconds,
      minimum_slice_count=None if config.elastic_min_slice_count == -1 else config.elastic_min_slice_count,
      pre_callback=effective_pre_callback,
      on_elastic_event_callback=effective_callback,
  )


def is_scale_up_event(config) -> bool:
  """Returns whether a scale up event is detected."""
  if elastic_enabled(config):
    ensure_elastic_manager_initialized(config)
    assert elastic_manager is not None
    return bool(elastic_manager.available_inactive_slices)

  return False


def maybe_elastic_scale_up(config, checkpoint_manager):
  """Waits for a checkpoint to finish before interrupting for scale up."""
  if is_scale_up_event(config):
    max_logging.log(
        "Started a checkpoint and a new slice is available. Waiting for current"
        " checkpoint to finish before interrupting."
    )
    if checkpoint_manager is not None:
      checkpoint_manager.wait_until_finished()
    max_logging.log("Checkpoint save completed. Interrupting")
    raise manager.ScaleUpSignalError()


def single_controller_mtc_init_kwargs(raw_keys):
  """Returns topology kwargs for single-controller MTC initialization."""
  kwargs = {
      "data_parallelism": raw_keys["mtc_data_parallelism"],
      "num_slices": raw_keys["num_slices"],
  }
  if not raw_keys.get("elastic_enabled", False):
    return kwargs

  if "elastic_min_slice_count" not in raw_keys:
    raw_keys["elastic_min_slice_count"] = raw_keys.get("num_slices", 0)
  if "elastic_timeout_seconds" not in raw_keys:
    raw_keys["elastic_timeout_seconds"] = 600
  config = SimpleNamespace(**raw_keys)
  if not should_use_elastic(config):
    return kwargs

  active_devices = tuple(live_devices(config))
  active_slice_indices = live_slice_indices(config)
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


def recalculate_batch_sizes(config, new_num_devices: int):
  """Recalculates config.num_target_devices and all dependent batch sizes for new_num_devices."""
  if new_num_devices <= 0:
    return
  object.__setattr__(config, "num_target_devices", new_num_devices)

  def calc_gbs(
      per_device_batch_size, expansion_factor, num_devices, grad_accum_steps
  ):
    if per_device_batch_size < 1.0:
      mbs_load = int(num_devices * (expansion_factor if expansion_factor > 0 else 1))
    else:
      mbs_load = int(
          num_devices
          * per_device_batch_size
          * (expansion_factor if expansion_factor > 0 else 1)
      )
    mbs_train = int(num_devices * per_device_batch_size)
    gbs_load = int(mbs_load * grad_accum_steps)
    gbs_train = int(mbs_train * grad_accum_steps)
    return gbs_load, gbs_train, mbs_train

  # Update train batch sizes
  gbs_load, gbs_train, mbs_train = calc_gbs(
      config.per_device_batch_size,
      config.expansion_factor_real_data,
      new_num_devices,
      config.gradient_accumulation_steps,
  )
  object.__setattr__(config, "global_batch_size_to_load", gbs_load)
  object.__setattr__(config, "global_batch_size_to_train_on", gbs_train)
  object.__setattr__(config, "micro_batch_size_to_train_on", mbs_train)

  # Update eval batch sizes
  gbs_load_eval, gbs_eval, mbs_eval = calc_gbs(
      config.eval_per_device_batch_size,
      config.expansion_factor_real_data,
      new_num_devices,
      1,
  )
  object.__setattr__(
      config, "global_batch_size_to_load_eval", gbs_load_eval
  )
  object.__setattr__(config, "global_batch_size_to_eval_on", gbs_eval)
  object.__setattr__(config, "micro_batch_size_to_eval_on", mbs_eval)

  if config.enable_rampup_batch_size:
    gbs_load_start = calc_gbs(
        config.per_device_batch_size_start,
        config.expansion_factor_real_data,
        new_num_devices,
        config.gradient_accumulation_steps,
    )[0]
    gbs_load_inc, _, _ = calc_gbs(
        config.per_device_batch_size_increment,
        config.expansion_factor_real_data,
        new_num_devices,
        config.gradient_accumulation_steps,
    )
    object.__setattr__(
        config, "global_batch_size_to_load_start", gbs_load_start
    )
    object.__setattr__(
        config, "global_batch_size_to_load_increment", gbs_load_inc
    )

    diff_batch_size = gbs_load - gbs_load_start
    if gbs_load_inc > 0:
      num_increments = diff_batch_size // gbs_load_inc
      if num_increments > 0:
        rampup_samples_per_increment = (
            config.global_rampup_samples / num_increments
        )
        object.__setattr__(
            config,
            "rampup_samples_per_increment_to_load",
            rampup_samples_per_increment,
        )

        total_rampup_steps = 0
        current_batch_size = gbs_load_start
        for _ in range(int(num_increments)):
          steps_for_this_stage = (
              math.ceil(rampup_samples_per_increment / current_batch_size)
              if current_batch_size > 0
              else 0
          )
          total_rampup_steps += steps_for_this_stage
          current_batch_size += gbs_load_inc
        object.__setattr__(config, "rampup_end_step", total_rampup_steps)
      else:
        object.__setattr__(config, "rampup_end_step", 0)
    else:
      object.__setattr__(config, "rampup_end_step", 0)

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


