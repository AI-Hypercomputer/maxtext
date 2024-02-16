"""Goodput package API implementations.

This file contains all the methods exposed through the cloud_tpu_goodput library
for users to log necessary information to compute Goodput, and to query the
computed Goodput.
"""

import datetime
import logging
import time
from typing import Any, Optional
import google.cloud.logging

_JOB_NAME = 'job_name'
_STEP_COUNT = 'step_count'
_STEP_START_TIME = 'step_start_time'
_JOB_RUN_START_TIME = 'job_run_start_time'
_JOB_RUN_END_TIME = 'job_run_end_time'

class _CloudLogger:
  """A helper class for reading and writing to Cloud Logging.

  Attributes:
    job_run_name: Identifier for a specific job run.
    logger: The Cloud Logging logger object.
  """

  def __init__(self, job_name: str, log_name: str):
    """Initializes the logger.

    Args:
      job_name: Name of the job started on a cloud VM.
      log_name: Name of the log being written.
    """
    self.job_run_name = job_name
    logging_client = google.cloud.logging.Client()
    self.logger = logging_client.logger(log_name)

  def _filter_entries_for_job(self, entries) -> list[Any]:
    """Filters entries for the current job run.

    Args:
      entries: Cloud Logging entries from user-specified logger.

    Returns:
      Filtered entries for the current job run.
    """
    filtered_entries = []
    for entry in entries:
      payload = entry.payload
      if 'job_name' not in payload:
        continue
      if payload['job_name'] != self.job_run_name:
        continue
      filtered_entries.append(payload)

    return filtered_entries

  def write_cloud_logging_entry(self, entry) -> None:
    """Writes an entry to the Cloud Logging logger at INFO level.

    Args:
      entry: JSON-serializable structured log dictionary.
    """
    if entry is None:
      return
    if entry[_JOB_NAME] == self.job_run_name:
      self.logger.log_struct(
          entry,
          severity='INFO',
      )

  def read_cloud_logging_entries(self):
    return self._filter_entries_for_job(
        self.logger.list_entries(order_by='timestamp asc')
    )


class GoodputRecorder:
  """The Goodput recorder class, responsible for recording Goodput metrics from the user application."""

  def __init__(
      self,
      job_run_name: str,
      logger_name: str,
      logging_enabled=False,
      logger: Optional[_CloudLogger] = None,
  ):
    """Initializes the Goodput recorder.

    Args:
      job_run_name: Identifier for a specific job run.
      logger_name: The name of the Cloud Logging logger object that the
        application wants logs to be written to and read from.
      logging_enabled: A boolean value to indicate whether the current process
        should send logs to Cloud Logging or not. The application should set
        this value to True if the Recorder is being called from TPU worker 0 and
        the application's configurations request Goodput logging.
    """
    self.job_run_name = job_run_name
    # If logging is disabled for this process, do not create a _cloud_logger
    # object and exit early if any record record_* API is called.
    if not logging_enabled:
      self._cloud_logger = None
      logging.info('Logging is disabled for this process.')
      return

    if logger is not None:
      self._cloud_logger = logger
    else:
      self._cloud_logger = _CloudLogger(job_run_name, logger_name)

  def record_step_start_time(
      self, step: int, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log an individual step's start time.

    Args:
      step: The count of the step that timing information is recorded for.
      start_time: Optional backfill start time of the training step.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now()
    start_timestamp = start_time.timestamp()
    step_start_time_dict = {
        _JOB_NAME: self.job_run_name,
        _STEP_COUNT: int(step),
        _STEP_START_TIME: float(start_timestamp),
    }
    self._cloud_logger.write_cloud_logging_entry(step_start_time_dict)

  def record_checkpoint_progress(self, step, checkpoint_start_time):
    """Main recorder function to log information on a successful checkpoint.

    This method is intended to log the progress for a checkpoint (last step
    included in the checkpoint) and when the checkpoint starts. This information
    will be retrieved in the future to determine whether training progress from
    a completed step contributes to Goodput or wasted progress Badput.

    Args:
      step: The step count of the last step included in the saved checkpoint.
      checkpoint_start_time: Timestamp at which the checkpoint containing
        progress upto "step" starts to save.
    """
    pass

  def record_job_start_time(
      self, start_time: Optional[datetime.datetime] = None
  ):
    """Main recorder function to log a job run's start time.

    Args:
      start_time: Optional backfill start time of the job run.
    """
    if self._cloud_logger is None:
      return
    if start_time is None:
      start_time = datetime.datetime.now()
    start_timestamp = start_time.timestamp()
    job_run_start_dict = {
        _JOB_NAME: self.job_run_name,
        _JOB_RUN_START_TIME: float(start_timestamp),
    }
    self._cloud_logger.write_cloud_logging_entry(job_run_start_dict)

  def record_job_end_time(self, end_time: Optional[datetime.datetime] = None):
    """Main recorder function to log a job run's end time.

    Args:
      end_time: Optional backfull end time of the job run.
    """
    if self._cloud_logger is None:
      return
    if end_time is None:
      end_time = datetime.datetime.now()
    end_timestamp = end_time.timestamp()
    job_run_end_dict = {
        _JOB_NAME: self.job_run_name,
        _JOB_RUN_END_TIME: float(end_timestamp),
    }
    self._cloud_logger.write_cloud_logging_entry(job_run_end_dict)

  def record_tpu_init_start_time(self, start_time):
    """Main recorder function to log the start time for TPU initialization.

    Note: TPU initialization may include the time spent in completing
    jax.devices() which is responsible for device scanning and Slice Builder
    initialization.

    Args:
      start_time: Start time of TPU initialization.
    """
    pass

  def record_tpu_init_end_time(self, end_time):
    """Main recorder function to log the end time for TPU initialization.

    Args:
      end_time: End time of TPU initialization.
    """
    pass

  def record_training_preparation_start_time(self, start_time):
    """Main recorder function to log the start time of training preparation before starting a training loop.

    Note: Training preparation may include the time spent in creation of
    checkpoint managers, checkpoint loading, running mesh and model optimizers
    etc.

    Args:
      start_time: Start time of training preparation.
    """
    pass

  def record_training_preparation_end_time(self, end_time):
    """Main recorder function to log the end time of training preparation before starting a training loop.

    Args:
      end_time: End time of training preparation.
    """
    pass

  def record_data_loading_start_time(self, start_time):
    """Main recorder function to log the start time of training's data loading.

    Args:
      start_time: Start time of data loading.
    """
    pass

  def record_data_loading_end_time(self, end_time):
    """Main recorder function to log the end time of training's data loading.

    Args:
      end_time: End time of data loading.
    """
    pass


class GoodputCalculator:
  """The Goodput calculator class, responsible for querying necessary information and computing Goodput metrics to return to the user application.

  Attributes:
    job_run_name: Identifier for a specific job run.
    logger_name: The name of the Cloud Logging logger object that the
      application wants logs to be written to and read from.

  """

  def __init__(
      self,
      job_run_name: str,
      logger_name: str,
      logger: Optional[_CloudLogger] = None,
  ):
    self.job_run_name = job_run_name
    if logger is not None:
      self._cloud_logger = logger
    else:
      self._cloud_logger = _CloudLogger(job_run_name, logger_name)

  def _get_total_productive_training_time(self) -> float:
    """Helper function to compute the total productive training time.

    Returns:
      The job's total productive training time.

    """
    entries = self._cloud_logger.read_cloud_logging_entries()
    # Build a deserialized dictionary from cloud logging entries to store step
    # start times. The dictionary maps from step count to start time and will be
    # used to each step's productive time by looking for its completion in the
    # next step's start.
    # Note in the instance where progress is lost due to a disruption and the
    # last successful checkpoint did not include all the steps, the last set of
    # records of the step information will be kept and the previous set will be
    # overwritten by design so as to correct for the the previously computed
    # additional time that was counted as productive but lost due to a
    # disruption.
    step_start_data = {}
    job_end_time = None
    for payload in entries:
      if _STEP_START_TIME in payload:
        # Overwrite start time information to record the latest for each step.
        step_start_data[payload[_STEP_COUNT]] = payload[_STEP_START_TIME]
      if _JOB_RUN_END_TIME in payload:
        # Locate the last instance of job's end time if the job has completed.
        job_end_time = payload[_JOB_RUN_END_TIME]

    # Structure to compute and store productive training time for each step.
    training_time_data = {}
    # Largest step count that has been recorded. For the last step, we check if
    # the step completed by looking for the corresponding run's end time, since
    # there are are no more steps to check for its completion or compute step
    # time.
    last_step = max(step_start_data.keys()) if step_start_data else 0

    for step_count, step_start_time in step_start_data.items():
      if step_count == last_step and job_end_time is not None:
        training_time_data[step_count] = job_end_time - step_start_time
      elif step_count + 1 in step_start_data:
        training_time_data[step_count] = (
            step_start_data[step_count + 1] - step_start_time
        )
    # Accumulate the productive time for all the steps.
    return sum(training_time_data.values())

  def _get_total_job_time(self) -> float:
    """Helper function to compute the total job runtime.

    Returns:
      The job's total runtime.
    """
    entries = self._cloud_logger.read_cloud_logging_entries()
    # De-serealize jon start and end times from cloud logging entries. These
    # will be used to compute total runtime of the job.
    total_job_start_time = None
    total_job_end_time = None
    for payload in entries:
      # Locate the earliest timestamp recorded for the job's start.
      if _JOB_RUN_START_TIME in payload and total_job_start_time is None:
        total_job_start_time = payload[_JOB_RUN_START_TIME]
      # Locate the latest timestamp recorded for the job's end.
      if _JOB_RUN_END_TIME in payload:
        total_job_end_time = payload[_JOB_RUN_END_TIME]

    total_job_time = 0.0
    # If the job has completed, use total_job_end_time to compute total job
    # time.
    if total_job_end_time is not None:
      total_job_time = total_job_end_time - total_job_start_time
    elif total_job_start_time is not None:
      # If the job did not complete, use current time to compute total job time.
      curr_time = datetime.datetime.now().timestamp()
      total_job_time = curr_time - total_job_start_time

    return total_job_time

  def get_job_goodput(self):
    """Method to get the cumulative Goodput of the job computed until now.

    If the application is interested in retrieving the overall Goodput of the
    job throughout its lifetime, this method provides the singular Goodput
    computation for the entire job run.

    Returns:
      Goodput percentage of the entire job run.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
    """
    productive_training_time = self._get_total_productive_training_time()
    total_job_time = self._get_total_job_time()
    # No calculations can be made if total job time is zero. This can happen if
    # logs for the job are not present, sent to an invalid location or contain
    # bad data. Raise a ValueError if this happens.
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Goodput cannot be calculated. Please fix the'
          ' logging entries.'
      )
    return (float(productive_training_time) / total_job_time) * 100

  def get_job_goodput_interval(self, interval_start, interval_end):
    """Method to get the Goodput of the job within an interval window.

    If the application is interested in retrieving the Goodput of the job within
    a specific window of time, this method provides the singular Goodput
    computation between the start and end of this window.

    Args:
      interval_start: The start time of the window for which Goodput is to be
        computed.
      interval_end: The end time of the window for which Goodput is to be
        computed.

    Returns:
      Goodput percentage of the job within specified time window.
    """
    pass
