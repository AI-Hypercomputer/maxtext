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

"""
Utilities for monitoring and recording job's goodput.

This module provides methods to monitor and record goodput metrics
to various logging platforms, including cloud logging and TensorBoard.
"""

import contextlib
import jax
from enum import Enum
from maxtext.utils import max_logging
from maxtext.common.gcloud_stub import goodput_modules

goodput, monitoring, _GOODPUT_STUB = goodput_modules()


class GoodputEvent(Enum):
  JOB = "job"
  TPU_INIT = "tpu_init"
  TRAINING_PREPARATION = "training_preparation"
  DATA_LOADING = "data_loading"
  STEP = "step"


@contextlib.contextmanager
def maybe_monitor_goodput(config):
  """Monitor cumulative goodput if enabled on the lead host.

  When the goodput module is stubbed or monitoring is disabled, this
  becomes a lightweight no-op context manager.
  """
  if _GOODPUT_STUB:
    if config.monitor_goodput and jax.process_index() == 0:
      max_logging.log("[GOODPUT NO-OP] monitoring disabled (decoupled stub).")
    yield
    return
  if not config.monitor_goodput or jax.process_index() != 0:
    yield
    return
  goodput_monitor = None
  try:
    if config.report_performance_metric_for_gcp_monitoring:
      config.enable_gcp_step_deviation_metrics = False

    gcp_options = monitoring.GCPOptions(
        enable_gcp_goodput_metrics=config.enable_gcp_goodput_metrics,
        enable_gcp_step_deviation_metrics=config.enable_gcp_step_deviation_metrics,
    )
    goodput_monitor = monitoring.GoodputMonitor(
        job_name=config.run_name,
        logger_name=f"goodput_{config.run_name}",
        tensorboard_dir=config.tensorboard_dir,
        upload_interval=config.goodput_upload_interval_seconds,
        monitoring_enabled=True,
        pathway_enabled=config.enable_pathways_goodput,
        include_badput_breakdown=True,
        include_step_deviation=config.monitor_step_time_deviation,
        step_deviation_interval_seconds=config.step_deviation_interval_seconds,
        gcp_options=gcp_options,
    )
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Goodput upload to Tensorboard & GCM in the background!")
    yield
  finally:
    if goodput_monitor:
      goodput_monitor.stop_goodput_uploader()
      max_logging.log("Flushed final metrics and safe exited from Goodput monitoring.")


@contextlib.contextmanager
def maybe_record_goodput(recorder, event_name, *args):
  """Record goodput if `enable_goodput_recording=True`."""
  try:
    start_event_name = f"record_{event_name.value}_start_time"
    record_goodput(recorder, start_event_name, *args)
    yield
  except BaseException:  # pylint: disable=W0706
    raise
  else:
    end_event_name = f"record_{event_name.value}_end_time"
    record_goodput(recorder, end_event_name, *args)
  finally:
    pass


def record_goodput(recorder, event_name, *args):
  """Record goodput to cloud logging."""
  if recorder:
    record_func = getattr(recorder, event_name, None)
    if record_func:
      record_func(*args)


def create_goodput_recorder(config):
  """Create goodput recorder if `enable_goodput_recording=True`."""
  if _GOODPUT_STUB:
    if config.enable_goodput_recording and jax.process_index() == 0:
      max_logging.log("[GOODPUT NO-OP] recorder skipped (decoupled stub).")
    return None
  if config.enable_goodput_recording:
    logger_name = f"goodput_{config.run_name}"
    recorder = goodput.GoodputRecorder(config.run_name, logger_name, jax.process_index() == 0)
    return recorder
  return None
