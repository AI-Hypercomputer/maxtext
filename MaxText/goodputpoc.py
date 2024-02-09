"""Goodput package API implementations.

This file contains all the methods exposed through the cloud_tpu_goodput library
for users to log necessary information to compute Goodput, and to query the
computed Goodput.
"""

import datetime
import time
from typing import Optional

import google.cloud.logging


class CloudLogger:
  """A helper class for reading and writing to Cloud Logging.

  Attributes:
    job_run_name: Identifier for a specific job run.
    logger_name: The Cloud Logging logger name.
    logger: The Cloud Logging logger object.
  """

  job_run_name: str
  logger_name: str
  logger: google.cloud.logging.Logger

  def __init__(self, job_name: str, log_name: str):
    """Initializes the logger.

    Args:
      job_name: Name of the job started on a cloud VM.
      log_name: Name of the log being written.
    """
    self.job_run_name = job_name
    self.logger_name = log_name
    logging_client = google.cloud.logging.Client()
    self.logger = logging_client.logger(log_name)

  def write_cloud_logging_entry(self, entry) -> None:
    """Writes an entry to the Cloud Logging logger at INFO level.

    Args:
      entry: JSON-serializable structured log dictionary.
    """
    self.logger.log_struct(
        entry,
        severity='INFO',
    )

  def read_cloud_logging_entries(self):
    return self.logger.list_entries()


class GoodputRecorder:
  """The Goodput recorder class, responsible for recording Goodput metrics from the user application.

  Attributes:
    job_run_name: Identifier for a specific job run.
    cloud_logger: The Cloud Logging logger object.
    user_logs_dir: The path to the user logs for the job run.
  """

  def __init__(
      self,
      job_run_name: str,
      user_logs_dir: str,
      logger: Optional[CloudLogger] = None,
  ):
    self.job_run_name = job_run_name
    self.user_logs_dir = user_logs_dir
    logger_name = f'goodput_metrics_{job_run_name}'
    if logger is not None:
      self.cloud_logger = logger
    else:
      self.cloud_logger = CloudLogger(job_run_name, logger_name)

  def record_step_start_time(self, step, start_time):
    """Main recorder function to log an individual step's start time.

    Args:
      step: The count of the step that timing information is recorded for.
      start_time: Start time of the training step.
    """
    log_unix_time = time.mktime(start_time.timetuple())
    step_start_time_dict = {
        'job_name': str(self.job_run_name),
        'step_count': int(step),
        'step_start_time': float(log_unix_time),
    }
    self.cloud_logger.write_cloud_logging_entry(step_start_time_dict)

  def record_step_end_time(self, step, end_time):
    """Main recorder function to log an individual step's end time.

    Note: a step's progress could be wasted if the last completed
    checkpoint was saved before this step's end_time. This consideration will be
    made before computing overall Goodput and Badput components.

    Args:
      step: The count of the step that timing information is recorded for.
      end_time: End time of the training step.
    """
    log_unix_time = time.mktime(end_time.timetuple())
    step_end_time_dict = {
        'job_name': str(self.job_run_name),
        'step_count': int(step),
        'step_end_time': float(log_unix_time),
    }
    self.cloud_logger.write_cloud_logging_entry(step_end_time_dict)

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

  def record_job_start_time(self, start_time):
    """Main recorder function to log a job run's start time.

    Args:
      start_time: Start time of the job run.
    """
    log_unix_time = time.mktime(start_time.timetuple())
    job_run_start_dict = {
        'job_name': str(self.job_run_name),
        'job_run_start_time': float(log_unix_time),
    }
    self.cloud_logger.write_cloud_logging_entry(job_run_start_dict)

  def record_job_end_time(self, end_time):
    """Main recorder function to log a job run's end time.

    Args:
      end_time: End time of the job run.
    """
    log_unix_time = time.mktime(end_time.timetuple())
    job_run_end_dict = {
        'job_name': str(self.job_run_name),
        'job_run_end_time': float(log_unix_time),
    }
    self.cloud_logger.write_cloud_logging_entry(job_run_end_dict)

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
    cloud_logger: The Cloud Logging logger object.
  """

  def __init__(
      self,
      job_run_name: str,
      logger: Optional[CloudLogger] = None,
  ):
    self.job_run_name = job_run_name
    logger_name = f'goodput_metrics_{job_run_name}'
    if logger is not None:
      self.cloud_logger = logger
    else:
      self.cloud_logger = CloudLogger(job_run_name, logger_name)

  def _get_total_productive_training_time(self) -> float:
    """Helper function to compute the total productive training time.

    Returns:
      The job's total productive training time.
    """
    entries = self.cloud_logger.read_cloud_logging_entries()
    # Build two dictionaries from cloud logging entries to store step start and
    # end times. These map from step count to start and end time and will be
    # used to compute total step time.
    step_start_data = {}
    step_end_data = {}
    for entry in entries:
      entry_payload = entry.payload
      if (
          'step_start_time' in entry_payload
          and entry_payload['step_count'] not in step_start_data
      ):
        step_start_data[entry_payload['step_count']] = entry_payload[
            'step_start_time'
        ]
      if (
          'step_end_time' in entry_payload
          and entry_payload['step_count'] not in step_end_data
      ):
        step_end_data[entry_payload['step_count']] = entry_payload[
            'step_end_time'
        ]
    training_time_data = {}
    for step_count in step_start_data:
      # If the training step started but did not complete, productive time is 0.
      if step_count not in step_end_data:
        training_time_data[step_count] = 0.0
      else:
        # Training step time is the delta between start and end time.
        training_time_data[step_count] = (
            step_end_data[step_count] - step_start_data[step_count]
        )
    total_step_time = sum(training_time_data.values())
    return total_step_time

  def _get_total_job_time(self) -> float:
    """Helper function to compute the total job runtime.

    Returns:
      The job's total runtime.

    Raises:
      ValueError if computed total job time is zero. In this case, Goodput
      cannot be computed.
    """
    entries = self.cloud_logger.read_cloud_logging_entries()
    # Build two dictionaries from cloud logging entries to store job's start and
    # end times. These will be used to compute total runtime of the job.
    job_start_data = {}
    job_end_data = {}
    for entry in entries:
      entry_payload = entry.payload
      if (
          'job_run_start_time' in entry_payload
          and entry_payload['job_name'] not in job_start_data
      ):
        job_start_data[entry_payload['job_name']] = entry_payload[
            'job_run_start_time'
        ]
      if (
          'job_run_end_time' in entry_payload
          and entry_payload['job_name'] not in job_end_data
      ):
        job_end_data[entry_payload['job_name']] = entry_payload[
            'job_run_end_time'
        ]
    total_job_time = 1.0
    for job_name in job_start_data:
      # Job completed, use job_end_data to compute total job time.
      if job_name in job_end_data:
        total_job_time = job_end_data[job_name] - job_start_data[job_name]
      else:
        # Job did not complete, use current time to compute total job time.
        curr_time = datetime.datetime.now()
        curr_unix_time = time.mktime(curr_time.timetuple())
        total_job_time = curr_unix_time - job_start_data[job_name]
    if total_job_time == 0.0:
      raise ValueError(
          'Total job time is zero, Goodput cannot be calculated. Please fix the'
          ' logging entries.'
      )
    return total_job_time

  def get_job_goodput(self):
    """Method to get the cumulative Goodput of the job computed until now.

    If the application is interested in retrieving the overall Goodput of the
    job throughout its lifetime, this method provides the singular Goodput
    computation for the entire job run.

    Returns:
      Goodput percentage of the entire job run.
    """
    productive_training_time = self._get_total_productive_training_time()
    total_job_time = self._get_total_job_time()
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
