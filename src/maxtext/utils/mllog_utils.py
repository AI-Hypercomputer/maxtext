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

import atexit
import os
import threading
import time

import jax

from maxtext.utils import max_logging

try:
  from mlperf_logging import mllog

  mllogger = mllog.get_mllogger()
except ImportError:
  mllog = None
  mllogger = None

# Minimum wall-clock seconds between two uploads of the staging log to GCS.
_MIN_SYNC_INTERVAL_SECONDS = 5.0

_enabled = False
_is_configured = False
_destination_path = None
_local_staging_file = None
_run_stopped = False
# Metadata of the currently open BLOCK_START, or None when no block is open.
_open_block = None

_last_sync_time = 0.0
_upload_thread = None
_gcs_client = None


def _should_log():
  """Returns True when this process is responsible for emitting mllog events."""
  return _enabled and mllogger is not None and jax.process_index() == 0


def setup_mllog(config):
  """Points mllogger at ``config.mllog_file``, staging locally for GCS destinations."""
  global _destination_path, _local_staging_file, _is_configured, _enabled
  if _is_configured or mllog is None:
    return
  _is_configured = True

  _enabled = bool(getattr(config, "enable_mllog", False))
  if not _enabled or jax.process_index() != 0:
    return

  _destination_path = getattr(config, "mllog_file", "") or ""
  if not _destination_path:
    max_logging.log("Warning: enable_mllog is set but mllog_file is empty. MLPerf logging is disabled.")
    _enabled = False
    return

  # mllog can only write to a local file, so GCS destinations are written to
  # local disk first and uploaded periodically by `sync_log`.
  is_gcs = _destination_path.startswith("gs://")
  if is_gcs:
    log_path = os.path.join("/tmp", f"mllog_{getattr(config, 'run_name', '') or 'maxtext'}.log")
  else:
    log_path = _destination_path

  try:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf8"):
      pass
    mllog.config(filename=log_path)
  except OSError as e:
    max_logging.log(f"Warning: cannot write mllog file {log_path}: {e}. MLPerf logging is disabled.")
    _enabled = False
    return

  if is_gcs:
    _local_staging_file = log_path
    # A job can die before train_loop's cleanup runs (e.g. a compilation
    # failure), so make sure the staged log still reaches GCS on any exit.
    atexit.register(sync_log, force=True)
  max_logging.log(f"Configured mllog at {log_path} (destination: {_destination_path})")


def _upload_file_to_gcs(dest_gcs: str, src_local: str):
  """Uploads a local file to GCS using a cached storage client, falling back to etils.epath."""
  global _gcs_client
  if not os.path.exists(src_local):
    return
  try:
    from google.cloud import storage  # pylint: disable=import-outside-toplevel

    if _gcs_client is None:
      _gcs_client = storage.Client()
    bucket_name, _, blob_name = dest_gcs.removeprefix("gs://").partition("/")
    _gcs_client.bucket(bucket_name).blob(blob_name).upload_from_filename(src_local)
  except Exception:  # pylint: disable=broad-exception-caught
    try:
      from etils import epath  # pylint: disable=import-outside-toplevel

      epath.Path(dest_gcs).write_bytes(epath.Path(src_local).read_bytes())
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Warning: Failed to sync mllog to GCS destination {dest_gcs}: {e}")


def sync_log(force=False):
  """Copies the staging log to its GCS destination, in the background when possible.

  mllog writes through a ``logging.FileHandler``, which flushes on every record,
  so the local log is always up to date and never needs an explicit flush. Only
  the GCS copy has to be refreshed, and that is a blocking network call: running
  it inline on process 0 would stall every other process at the next collective.

  Args:
    force: if True, ignore the throttling interval and upload synchronously. Used
      at the end of a run and from the `atexit` hook.
  """
  global _last_sync_time, _upload_thread
  if not _should_log() or not _local_staging_file:
    return

  if _upload_thread is not None and _upload_thread.is_alive():
    if not force:
      return  # An upload is already in flight; the next sync picks up the tail.
    _upload_thread.join()

  if force:
    # Upload inline: the caller wants the log durable before moving on, and this
    # also runs at interpreter shutdown where starting threads is best avoided.
    _last_sync_time = time.monotonic()
    _upload_file_to_gcs(_destination_path, _local_staging_file)
    return

  now = time.monotonic()
  if now - _last_sync_time < _MIN_SYNC_INTERVAL_SECONDS:
    return
  _last_sync_time = now
  _upload_thread = threading.Thread(
      target=_upload_file_to_gcs,
      args=(_destination_path, _local_staging_file),
      daemon=True,
  )
  _upload_thread.start()


def init_start(config):
  """Logs CACHE_CLEAR and INIT_START, the first events of a submission log."""
  setup_mllog(config)
  if not _should_log():
    return
  mllogger.event(mllog.constants.CACHE_CLEAR, value=True)
  mllogger.start(mllog.constants.INIT_START)


def _mllog_precision(val, fallback="bfloat16") -> str:
  """Normalizes a MaxText dtype or quantization setting to an MLPerf v6.1 precision string."""
  raw = str(val if val is not None else "").strip().lower()
  if "." in raw:
    raw = raw.rsplit(".", 1)[-1]
  if not raw or raw == "none":
    raw = str(fallback).strip().lower()
  aliases = {
      "float64": "fp64",
      "float32": "fp32",
      "float16": "fp16",
      "bf16": "bfloat16",
  }
  if raw in aliases:
    return aliases[raw]
  if raw.startswith("fp8"):
    return "fp8"
  if raw.startswith("int8") or raw == "intmp":
    return "int8"
  if raw.startswith("int4"):
    return "int4"
  return raw


def _axis_product(config, *names) -> int:
  """Returns the product of positive mesh axis sizes (>= 1) for MLPerf parallelism disclosure."""
  product = 1
  for name in names:
    val = int(getattr(config, name, 1) or 1)
    if val > 1:
      product *= val
  return product


def init_print(config):
  """Logs the static submission and hyperparameter events for compliance checking."""
  setup_mllog(config)
  if not _should_log():
    return
  # General
  mllogger.event(mllog.constants.SUBMISSION_ORG, "Google")
  mllogger.event(mllog.constants.SUBMISSION_PLATFORM, "TPU-Ironwood")
  mllogger.event(mllog.constants.SUBMISSION_STATUS, mllog.constants.CLOUD)
  mllogger.event(mllog.constants.SUBMISSION_DIVISION, mllog.constants.CLOSED)

  # Model specific
  warmup_steps = int(config.learning_rate_schedule_steps * config.warmup_steps_fraction)
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
  mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS, warmup_steps)
  mllogger.event(mllog.constants.OPT_LR_DECAY_STEPS, config.learning_rate_schedule_steps - warmup_steps)
  mllogger.event(mllog.constants.OPT_LR_DECAY_SCHEDULE, "cosine with linear warmup")
  mllogger.event("target_accuracy", config.target_eval_loss)

  # MLPerf v6.1 mandatory precision, parallelism, micro-batch size, and config filename disclosure.
  dtype_str = _mllog_precision(getattr(config, "dtype", "bfloat16"))
  linear_prec = _mllog_precision(getattr(config, "quantization", None), fallback=dtype_str)
  comm_prec = _mllog_precision(getattr(config, "grad_dtype", None), fallback=dtype_str)
  mllogger.event(mllog.constants.LOWEST_NUMERICAL_PRECISION_IN_LINEAR, linear_prec)
  mllogger.event(mllog.constants.LOWEST_NUMERICAL_PRECISION_IN_ATTN, dtype_str)
  mllogger.event(mllog.constants.LOWEST_NUMERICAL_PRECISION_IN_COMM, comm_prec)
  mllogger.event(
      mllog.constants.TENSOR_PARALLELISM,
      _axis_product(
          config,
          "ici_tensor_parallelism",
          "dcn_tensor_parallelism",
          "ici_tensor_sequence_parallelism",
          "dcn_tensor_sequence_parallelism",
      ),
  )
  mllogger.event(
      mllog.constants.PIPELINE_PARALLELISM,
      _axis_product(config, "ici_pipeline_parallelism", "dcn_pipeline_parallelism"),
  )
  mllogger.event(
      mllog.constants.CONTEXT_PARALLELISM,
      _axis_product(config, "ici_context_parallelism", "dcn_context_parallelism"),
  )
  mllogger.event(
      mllog.constants.EXPERT_PARALLELISM,
      _axis_product(config, "ici_expert_parallelism", "dcn_expert_parallelism"),
  )
  # TPU v7x exposes 2 JAX devices (TensorCores) per chip, while system descriptions count chips.
  mllogger.event(
      mllog.constants.MICRO_BATCH_SIZE,
      max(1, int(round(getattr(config, "per_device_batch_size", 1) * 2))),
  )
  mllogger.event(
      mllog.constants.CONFIG_FILENAME,
      getattr(config, "mllog_config_filename", "") or "config.yml",
  )


def init_stop():
  if not _should_log():
    return
  mllogger.end(mllog.constants.INIT_STOP)


def run_start():
  if not _should_log():
    return
  mllogger.start(mllog.constants.RUN_START)


def block_start(config, step=0):
  """Logs BLOCK_START, opening a new single-step train block."""
  global _open_block
  if not _should_log():
    return
  _open_block = {
      "samples_count": config.global_batch_size_to_train_on,
      "step": step,
  }
  mllogger.start(mllog.constants.BLOCK_START, metadata=dict(_open_block))


def _block_stop(step=None):
  """Logs BLOCK_STOP for the open block, if any. No-op when no block is open."""
  global _open_block
  if _open_block is None:
    return
  metadata = dict(_open_block)
  if step is not None:
    metadata["step"] = step
  _open_block = None
  mllogger.end(mllog.constants.BLOCK_STOP, metadata=metadata)


def step_end(config, step):
  """Closes the current training step's block and opens the next step's block when no eval ran."""
  if not _should_log():
    return
  _block_stop(step)
  if step < config.steps:
    block_start(config, step)


def eval_start(config, step, start_step=0):
  """Logs BLOCK_STOP and EVAL_START before the evaluation loop."""
  if not _should_log():
    return
  # BLOCK_STOP reuses the metadata recorded by the matching block_start.
  _block_stop(step)
  mllogger.start(
      mllog.constants.EVAL_START,
      metadata={
          "samples_count": (step - start_step) * config.global_batch_size_to_train_on,
          "step": step,
      },
  )


def eval_stop(config, step, eval_loss, start_step=0):
  """Logs EVAL_ACCURACY and EVAL_STOP, then opens the next block or stops the run."""
  if not _should_log():
    return
  samples_count = (step - start_step) * config.global_batch_size_to_train_on

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
  if config.target_eval_loss and eval_loss <= config.target_eval_loss:
    run_stop(status="success", current_epoch_num=samples_count)
  elif step >= config.steps:
    # Step budget exhausted on an eval step. Stop here rather than opening a block that
    # train_loop's run_stop would immediately close as a zero-length block.
    run_stop(status="aborted", current_epoch_num=samples_count, step=step)
  else:
    block_start(config, step)


def run_stop(status="success", current_epoch_num=None, step=None):
  """Logs BLOCK_STOP for any open block, then RUN_STOP. Only the first call takes effect."""
  global _run_stopped
  if _run_stopped or not _should_log():
    return
  _run_stopped = True

  _block_stop(step)
  metadata = {"status": status}
  if current_epoch_num is not None:
    metadata["samples_count"] = current_epoch_num
  mllogger.end(mllog.constants.RUN_STOP, metadata=metadata)
  sync_log(force=True)


def tracked_stats(config, step, step_time, loss, start_step=0):
  """Logs per-step tracked_stats and opportunistically syncs the log to GCS."""
  if not _should_log():
    return
  value = {"reduced_train_loss": loss.item() if hasattr(loss, "item") else float(loss)}
  if step_time is not None:
    value["train_step_time"] = step_time
  mllogger.event(
      key="tracked_stats",
      metadata={mllog.constants.SAMPLES_COUNT: (step - start_step) * config.global_batch_size_to_train_on},
      value=value,
  )
  sync_log()


def validation_time(step, seconds):
  """Logs the eval loop duration as a tracked_stats event.

  Mirrors the reference submission, which keys this variant by `step` rather
  than `samples_count`.
  """
  if not _should_log():
    return
  mllogger.event(key="tracked_stats", value={"validation_time": seconds}, metadata={"step": step})
