"""
Copyright 2026 Google LLC
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

"""Utils for MLPerf submission compliance."""

import os
import jax
from maxtext.utils import max_logging

try:
  from mlperf_logging import mllog

  mllogger = mllog.get_mllogger()
except ImportError:
  mllog = None
  mllogger = None

_destination_path = None
_local_staging_file = None
_is_configured = False
_run_stopped = False
_enabled = False


def is_mllog_enabled():
  return _enabled


def setup_mllog(config):
  """Configures mllogger to output to a file in base_output_directory (or custom mllog_file)."""
  global _destination_path, _local_staging_file, _is_configured, _enabled
  if _is_configured or mllog is None:
    return

  _enabled = bool(getattr(config, "enable_mllog", False))
  if not _enabled:
    _is_configured = True
    return

  if jax.process_index() != 0:
    _is_configured = True
    return

  target_path = getattr(config, "mllog_file", "") or ""
  if not target_path and getattr(config, "base_output_directory", ""):
    run_name = getattr(config, "run_name", "")
    target_path = (
        os.path.join(config.base_output_directory, run_name, "mllog.log")
        if run_name
        else os.path.join(config.base_output_directory, "mllog.log")
    )

  if not target_path:
    _is_configured = True
    return

  _destination_path = target_path
  if target_path.startswith("gs://"):
    run_name = getattr(config, "run_name", "") or "maxtext"
    _local_staging_file = f"/tmp/mllog_{run_name}.log"
    try:
      with open(_local_staging_file, "w", encoding="utf8"):
        pass
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    mllog.config(filename=_local_staging_file)
    max_logging.log(f"Configured mllog to staging file {_local_staging_file} (destination: {_destination_path})")
  else:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    mllog.config(filename=target_path)
    _local_staging_file = None
    max_logging.log(f"Configured mllog to file {_destination_path}")

  _is_configured = True


_last_sync_time = 0.0
_min_sync_interval = 5.0


def flush_and_sync(force=False):
  """Flushes mllog handlers and uploads the local staging file to GCS if needed."""
  global _last_sync_time
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return

  if hasattr(mllogger, "logger") and mllogger.logger:
    for handler in mllogger.logger.handlers:
      try:
        handler.flush()
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  if _destination_path and _local_staging_file and _destination_path.startswith("gs://"):
    import time  # pylint: disable=import-outside-toplevel

    now = time.time()
    if force or (now - _last_sync_time >= _min_sync_interval):
      _upload_file_to_gcs(_destination_path, _local_staging_file)
      _last_sync_time = now


def _upload_file_to_gcs(dest_gcs: str, src_local: str):
  """Uploads a local file to GCS using google-cloud-storage client or etils.epath fallback."""
  if not os.path.exists(src_local):
    return
  try:
    from google.cloud import storage  # pylint: disable=import-outside-toplevel

    path_parts = dest_gcs.replace("gs://", "").split("/")
    bucket_name = path_parts.pop(0)
    blob_name = "/".join(path_parts)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(src_local)
  except Exception:  # pylint: disable=broad-exception-caught
    try:
      from etils import epath  # pylint: disable=import-outside-toplevel

      epath.Path(dest_gcs).write_bytes(epath.Path(src_local).read_bytes())
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Warning: Failed to sync mllog to GCS destination {dest_gcs}: {e}")


def init_start(config=None):
  if config is not None:
    setup_mllog(config)
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  mllogger.event(mllog.constants.CACHE_CLEAR)
  mllogger.start(mllog.constants.INIT_START)
  flush_and_sync()


def init_stop():
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  mllogger.end(mllog.constants.INIT_STOP)
  flush_and_sync()


def run_start():
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  mllogger.start(mllog.constants.RUN_START)
  flush_and_sync()


def block_start(config, step=0):
  """Logs BLOCK_START for an MLPerf evaluation block."""
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  eval_frequency_samples = config.eval_interval * config.global_batch_size_to_train_on
  mllogger.start(
      mllog.constants.BLOCK_START,
      metadata={
          "samples_count": eval_frequency_samples,
          "step": step,
      },
  )
  flush_and_sync()


def init_print(config, start_step):
  """The initial mllog for mlperf submission compliance check."""
  setup_mllog(config)
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  # General
  mllogger.event(mllog.constants.SUBMISSION_ORG, "Google")
  mllogger.event(mllog.constants.SUBMISSION_PLATFORM, "TPU-Ironwood")
  mllogger.event(mllog.constants.SUBMISSION_STATUS, mllog.constants.CLOUD)
  mllogger.event(mllog.constants.SUBMISSION_DIVISION, mllog.constants.CLOSED)

  # Model specific
  mllogger.event(mllog.constants.SUBMISSION_BENCHMARK, mllog.constants.DEEPSEEKV3_671B)
  mllogger.event(mllog.constants.SEED, config.data_shuffle_seed)
  mllogger.event(mllog.constants.MAX_STEPS, config.steps)
  mllogger.event(mllog.constants.GLOBAL_BATCH_SIZE, config.global_batch_size_to_train_on)
  mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS, config.gradient_accumulation_steps)
  mllogger.event(mllog.constants.MAX_SEQUENCE_LENGTH, config.max_target_length)
  mllogger.event(mllog.constants.EVAL_SAMPLES, config.global_batch_size_to_eval_on * config.eval_steps)
  mllogger.event(mllog.constants.TRAIN_SAMPLES, 1574207408)
  mllogger.event(mllog.constants.INIT_CHECKPOINT_STEP, 0)
  mllogger.event(mllog.constants.OPT_NAME, mllog.constants.ADAMW)
  mllogger.event(mllog.constants.OPT_BASE_LR, config.learning_rate)
  mllogger.event(mllog.constants.OPT_ADAMW_BETA_1, config.adam_b1)
  mllogger.event(mllog.constants.OPT_ADAMW_BETA_2, config.adam_b2)
  mllogger.event(mllog.constants.OPT_ADAMW_EPSILON, config.adam_eps)
  mllogger.event(mllog.constants.OPT_ADAMW_WEIGHT_DECAY, config.adam_weight_decay)
  mllogger.event(mllog.constants.OPT_GRADIENT_CLIP_NORM, config.gradient_clipping_threshold)
  mllogger.event(mllog.constants.MOE_AUX_LOSS_COEFF, config.load_balance_loss_weight)
  mllogger.event(mllog.constants.OPT_END_LR, config.learning_rate * config.learning_rate_final_fraction)
  mllogger.event(
      mllog.constants.OPT_LR_WARMUP_STEPS, int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
  )
  mllogger.event(
      mllog.constants.OPT_LR_DECAY_STEPS,
      int(config.learning_rate_schedule_steps * (1 - config.warmup_steps_fraction) + 1),
  )
  mllogger.event(mllog.constants.OPT_LR_DECAY_SCHEDULE, "cosine with linear warmup")
  mllogger.event("target_accuracy", config.target_eval_loss)
  flush_and_sync()


def run_stop(status="success", current_epoch_num=None):
  """Logs RUN_STOP for MLPerf completion."""
  global _run_stopped
  if not _enabled or _run_stopped:
    return
  if mllogger is not None and jax.process_index() == 0:
    metadata = {"status": status}
    if current_epoch_num is not None:
      metadata["samples_count"] = current_epoch_num
    mllogger.end(mllog.constants.RUN_STOP, metadata=metadata)
    flush_and_sync(force=True)
  _run_stopped = True


def eval_start(config, step, start_step=0):
  """Logs BLOCK_STOP and EVAL_START before the evaluation loop."""
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  eval_frequency_samples = config.eval_interval * config.global_batch_size_to_train_on
  samples_count = (step - start_step) * config.global_batch_size_to_train_on
  mllogger.end(
      mllog.constants.BLOCK_STOP,
      metadata={
          "samples_count": eval_frequency_samples,
          "step": step,
      },
  )
  mllogger.start(
      mllog.constants.EVAL_START,
      metadata={
          "samples_count": samples_count,
          "step": step,
      },
  )
  flush_and_sync()


def eval_stop(config, step, eval_loss, start_step=0):
  """Logs EVAL_ACCURACY, EVAL_STOP, and starts a new BLOCK_START or triggers RUN_STOP."""
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  samples_count = (step - start_step) * config.global_batch_size_to_train_on
  eval_frequency_samples = config.eval_interval * config.global_batch_size_to_train_on
  is_early_stop = bool(config.target_eval_loss and eval_loss <= config.target_eval_loss)

  mllogger.event(
      mllog.constants.EVAL_ACCURACY,
      float(eval_loss),
      metadata={"samples_count": samples_count},
  )
  mllogger.end(
      mllog.constants.EVAL_STOP,
      metadata={
          "samples_count": samples_count,
          "step": step,
      },
  )
  if is_early_stop:
    run_stop(status="success", current_epoch_num=samples_count)
  else:
    mllogger.start(
        mllog.constants.BLOCK_START,
        metadata={
            "samples_count": eval_frequency_samples,
            "step": step,
        },
    )
  flush_and_sync()


def check_eval(config, step, eval_loss, start_step=0):
  """Logs an MLPerf evaluation block completion, checks for early stopping, and starts a new block if continuing."""
  eval_stop(config, step, eval_loss, start_step)


def tracked_stats(config, step, step_time, loss, start_step=0):
  """Logs tracked_stats for MLPerf training compliance."""
  if not _enabled or mllogger is None or jax.process_index() != 0:
    return
  loss_val = loss.item() if hasattr(loss, "item") else float(loss)
  samples_count = (step - start_step) * config.global_batch_size_to_train_on
  value = {"reduced_train_loss": loss_val}
  if step_time is not None:
    value["train_step_time"] = step_time
  mllogger.event(
      key="tracked_stats",
      metadata={mllog.constants.SAMPLES_COUNT: samples_count},
      value=value,
  )
  flush_and_sync()
